from __future__ import annotations

import io
import json
from pathlib import Path
import threading
from typing import Any

from PIL import Image

from bottle_qc.domain.association import associate_detections
from bottle_qc.domain.config import InspectionConfig
from bottle_qc.domain.entities import EndReason, RunMode
from bottle_qc.domain.image_inspection import inspect_image
from bottle_qc.domain.inspection import InspectionEngine
from bottle_qc.pipeline.contracts import DetectionBatch, Frame
from bottle_qc.pipeline.runner import process_frame
from bottle_qc.pipeline.sources import ImageFileSource, VideoFileSource
from bottle_qc.settings import Settings
from bottle_qc.storage.assets import AssetStore
from bottle_qc.storage.evidence import EvidenceStore
from bottle_qc.storage.repository import Repository
from bottle_qc.vision.model_registry import ModelRegistry
from bottle_qc.vision.overlay import render_overlay
from bottle_qc.vision.simulation import SimulationDetectionProvider, SimulationFrameSource
from bottle_qc.vision.yolo_provider import open_yolo_provider


class RunWorker:
    """One backend-owned worker; file sources are consumed sequentially without frame dropping."""

    def __init__(
        self,
        settings: Settings,
        repository: Repository,
        evidence_store: EvidenceStore,
        model_registry: ModelRegistry,
    ):
        self.settings = settings
        self.repository = repository
        self.evidence_store = evidence_store
        self.model_registry = model_registry
        default_values, _, _ = settings.load_config("default")
        self.asset_store = AssetStore(settings.assets_dir, max_upload_bytes=int(default_values["max_upload_bytes"]))
        self._lock = threading.RLock()
        self._threads: dict[str, threading.Thread] = {}
        self._stops: dict[str, threading.Event] = {}
        self._progress: dict[str, dict[str, Any]] = {}
        self._previews: dict[str, bytes] = {}

    def start(self, run: dict[str, Any]) -> None:
        run_id = run["run_id"]
        with self._lock:
            if any(thread.is_alive() for thread in self._threads.values()):
                raise RuntimeError("The single inspection worker is already active")
            # Durable run history belongs to SQLite. Keep transient thread,
            # stop, progress, and preview state for only the latest run.
            self._threads.clear()
            self._stops.clear()
            self._progress.clear()
            self._previews.clear()
            stop = threading.Event()
            thread = threading.Thread(target=self._execute, args=(dict(run), stop), name=f"bottleqc-{run_id}", daemon=False)
            self._stops[run_id] = stop
            self._threads[run_id] = thread
            self._progress[run_id] = {"frames_read": 0, "pending": 0, "status": "RUNNING"}
            thread.start()

    def request_stop(self, run_id: str) -> bool:
        with self._lock:
            stop = self._stops.get(run_id)
            thread = self._threads.get(run_id)
            if stop is None or thread is None or not thread.is_alive():
                return False
            stop.set()
            return True

    def progress(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._progress.get(run_id, {"frames_read": None, "pending": 0, "status": "TERMINAL"}))

    def preview(self, run_id: str) -> bytes | None:
        with self._lock:
            return self._previews.get(run_id)

    def wait(self, run_id: str, timeout: float | None = None) -> bool:
        with self._lock:
            thread = self._threads.get(run_id)
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def shutdown(self) -> None:
        with self._lock:
            threads = list(self._threads.items())
            for run_id, _ in threads:
                self._stops[run_id].set()
        for _, thread in threads:
            thread.join()

    def _execute(self, run: dict[str, Any], stop: threading.Event) -> None:
        run_id = run["run_id"]
        source = None
        engine: InspectionEngine | None = None
        last_frame: Frame | None = None
        last_batch: DetectionBatch | None = None
        status, error = "FAILED", None
        evidence_errors: list[str] = []
        terminal_outcomes: dict[str, str] = {}
        try:
            config_values = json.loads(run["config_snapshot"])
            config = InspectionConfig.from_mapping(config_values)
            mode = RunMode(run["mode"])
            if run["source_type"] == "simulation":
                source = SimulationFrameSource()
                provider = SimulationDetectionProvider()
            else:
                asset = self.repository.get_asset(run["asset_id"])
                if asset is None:
                    raise ValueError("Registered source asset is unavailable")
                asset_path = self.asset_store.path_for(asset)
                if asset["kind"] == "image":
                    source = ImageFileSource(asset_path, source_item_id=asset["asset_id"])
                elif asset["kind"] == "video":
                    source = VideoFileSource(asset_path, source_item_id=asset["asset_id"])
                else:
                    raise ValueError("Registered source kind is unsupported")
                model_record = self.model_registry.get(run["model_id"])
                if not model_record or model_record.get("status") != "AVAILABLE":
                    raise ValueError("Registered real model capability became unavailable")
                provider = open_yolo_provider(
                    Path(model_record["manifest_path"]),
                    models_root=self.settings.models_dir,
                    tracker=asset["kind"] == "video",
                    minimum_confidence=config.minimum_confidence,
                    minimum_region_edge_variance=float(config_values["minimum_region_edge_variance"]),
                    tracker_iou_threshold=float(config_values["tracker_iou_threshold"]),
                    tracker_ambiguity_margin=float(config_values["tracker_ambiguity_margin"]),
                    maximum_missed_frames=int(config_values["maximum_missed_frames"]),
                    maximum_tracks=config.maximum_tracks,
                )

            if source.kind == "image":
                frame = source.read()
                if frame is None:
                    raise ValueError("Image source produced no frame")
                batch = provider.detect(frame)
                prepared_batch = _prepare_batch_for_inspection(batch, run_id=run_id, config=config, mode=mode)
                observations = associate_detections(
                    prepared_batch.detections,
                    run_id=run_id,
                    config=config,
                    observability=prepared_batch.observability,
                    uncertain_track_ids=prepared_batch.uncertain_track_ids,
                )
                events = inspect_image(
                    observations,
                    run_id=run_id,
                    source_item_id=run["asset_id"],
                    config=config,
                    mode=mode,
                    config_id=run["config_id"],
                    config_hash=run["config_hash"],
                    model_id=run["model_id"],
                    model_hash=run["model_hash"],
                    source_index=frame.index,
                    source_timestamp_ms=frame.source_timestamp_ms,
                )
                last_frame, last_batch = frame, prepared_batch
                evidence_errors.extend(self._persist(events, frame, last_batch, mode))
                terminal_outcomes.update((event.track_id, event.outcome.value) for event in events)
                self._publish_preview(run_id, frame, last_batch, mode, terminal_outcomes)
                self._set_progress(run_id, frames_read=1, pending=0)
                status = "COMPLETED"
            else:
                engine = InspectionEngine(
                    run_id=run_id,
                    config=config,
                    mode=mode,
                    config_id=run["config_id"],
                    config_hash=run["config_hash"],
                    model_id=run["model_id"],
                    model_hash=run["model_hash"],
                )
                frames_read = 0
                maximum_frames = int(config_values["maximum_frames_per_run"])
                while True:
                    if stop.is_set():
                        events = engine.finish(
                            reason=EndReason.STOP,
                            frame_index=last_frame.index if last_frame else 0,
                            timestamp=last_frame.timestamp if last_frame else 0.0,
                        )
                        evidence_errors.extend(self._persist(events, last_frame, last_batch, mode))
                        terminal_outcomes.update((event.track_id, event.outcome.value) for event in events)
                        status = "STOPPED"
                        break
                    frame = source.read()
                    if frame is None:
                        stop_requested = stop.is_set()
                        events = engine.finish(
                            reason=EndReason.STOP if stop_requested else EndReason.EOF,
                            frame_index=last_frame.index if last_frame else 0,
                            timestamp=last_frame.timestamp if last_frame else 0.0,
                        )
                        evidence_errors.extend(self._persist(events, last_frame, last_batch, mode))
                        terminal_outcomes.update((event.track_id, event.outcome.value) for event in events)
                        status = "STOPPED" if stop_requested else "COMPLETED"
                        break
                    if frames_read >= maximum_frames:
                        frame.image.close()
                        events = engine.finish(
                            reason=EndReason.ERROR,
                            frame_index=last_frame.index if last_frame else 0,
                            timestamp=last_frame.timestamp if last_frame else 0.0,
                        )
                        evidence_errors.extend(self._persist(events, last_frame, last_batch, mode))
                        terminal_outcomes.update((event.track_id, event.outcome.value) for event in events)
                        status = "FAILED"
                        error = (
                            f"maximum_frames_per_run reached after {frames_read} processed frames; "
                            f"source frame {frame.index} remains unprocessed and the run was failed explicitly"
                        )
                        break
                    batch = provider.detect(frame)
                    prepared_batch = _prepare_batch_for_inspection(batch, run_id=run_id, config=config, mode=mode)
                    engine.mark_identity_uncertain(prepared_batch.uncertain_track_ids)
                    events = process_frame(engine, config, frame, prepared_batch)
                    last_frame, last_batch = frame, prepared_batch
                    for event in events:
                        event.source_timestamp_ms = frame.source_timestamp_ms
                    evidence_errors.extend(self._persist(events, frame, last_batch, mode))
                    terminal_outcomes.update((event.track_id, event.outcome.value) for event in events)
                    frames_read += 1
                    self._publish_preview(run_id, frame, last_batch, mode, terminal_outcomes)
                    self._set_progress(run_id, frames_read=frames_read, pending=sum(value.value == "PENDING" for value in engine.statuses.values()))
            if evidence_errors:
                error = "; ".join(([error] if error else []) + evidence_errors)
            self.repository.set_run_status(run_id, status, error)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if engine is not None:
                try:
                    events = engine.finish(
                        reason=EndReason.ERROR,
                        frame_index=last_frame.index if last_frame else 0,
                        timestamp=last_frame.timestamp if last_frame else 0.0,
                    )
                    if last_frame is not None and last_batch is not None:
                        for event in events:
                            event.source_timestamp_ms = last_frame.source_timestamp_ms
                    self._persist(events, last_frame, last_batch, RunMode(run["mode"]))
                except Exception as finalize_exc:
                    error += f"; finalization failed: {type(finalize_exc).__name__}: {finalize_exc}"
            try:
                self.repository.set_run_status(run_id, "FAILED", error)
            except Exception:
                error += "; run status could not be persisted"
        finally:
            if source is not None:
                try:
                    source.close()
                except Exception:
                    pass
            try:
                stored = self.repository.get_run(run_id)
                final_status = stored["status"] if stored else status
            except Exception:
                final_status = status
            self._set_progress(run_id, status=final_status, error=error)

    def _persist(self, events, frame, batch, mode: RunMode) -> list[str]:
        errors = []
        for event in events:
            self.repository.insert_event(event)
            if frame is None or batch is None:
                error = "No source frame was available for evidence"
                self.repository.update_evidence(event.event_id, state="FAILED", error=error)
                errors.append(f"{event.event_id}: {error}")
                continue
            try:
                image = render_overlay(
                    frame.image,
                    batch.detections,
                    mode=mode.value,
                    outcomes={event.track_id: event.outcome.value},
                )
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                _, digest = self.evidence_store.write(event.run_id, event.event_id, buffer.getvalue())
                self.repository.update_evidence(event.event_id, state="AVAILABLE", sha256=digest)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                self.repository.update_evidence(event.event_id, state="FAILED", error=error)
                errors.append(f"{event.event_id}: evidence failed: {error}")
        return errors

    def _publish_preview(self, run_id: str, frame: Frame, batch: DetectionBatch, mode: RunMode, outcomes: dict[str, str]) -> None:
        image = render_overlay(frame.image, batch.detections, mode=mode.value, outcomes=outcomes)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=84)
        with self._lock:
            self._previews[run_id] = buffer.getvalue()

    def _set_progress(self, run_id: str, **values: Any) -> None:
        with self._lock:
            self._progress.setdefault(run_id, {}).update(values)


def _prepare_batch_for_inspection(
    batch: DetectionBatch,
    *,
    run_id: str,
    config: InspectionConfig,
    mode: RunMode,
) -> DetectionBatch:
    """REAL eligibility needs a quality-adequate, correctly associated positive detection."""
    if mode != RunMode.REAL:
        return batch
    observations = associate_detections(
        batch.detections,
        run_id=run_id,
        config=config,
        observability=batch.observability,
        uncertain_track_ids=batch.uncertain_track_ids,
    )
    region_quality = {
        item.track_id: (
            item.cap_present and item.cap_observable,
            item.label_present and item.label_observable,
        )
        for item in observations
    }
    return DetectionBatch(batch.detections, region_quality, batch.uncertain_track_ids)
