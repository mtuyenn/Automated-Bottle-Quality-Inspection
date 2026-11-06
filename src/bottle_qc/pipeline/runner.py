from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import uuid

from bottle_qc.domain.association import associate_detections
from bottle_qc.domain.config import InspectionConfig
from bottle_qc.domain.entities import EndReason, RunMode
from bottle_qc.domain.inspection import InspectionEngine
from bottle_qc.pipeline.contracts import DetectionBatch, Frame
from bottle_qc.settings import Settings
from bottle_qc.storage.evidence import EvidenceStore
from bottle_qc.storage.repository import Repository
from bottle_qc.vision.overlay import render_overlay
from bottle_qc.vision.simulation import SimulationDetectionProvider, SimulationFrameSource


def process_frame(engine: InspectionEngine, config: InspectionConfig, frame: Frame, batch: DetectionBatch):
    """Shared orchestration seam: detector output -> association -> accepted domain engine."""
    observations = associate_detections(
        batch.detections,
        run_id=engine.run_id,
        config=config,
        observability=batch.observability,
        uncertain_track_ids=batch.uncertain_track_ids,
    )
    return engine.process_frame(
        frame_index=frame.index,
        timestamp=frame.timestamp,
        frame_size=frame.image.size,
        observations=observations,
    )


def run_simulation_demo(output_dir: Path) -> dict:
    """Run the fixed four-product fixture through domain, SQLite and evidence storage."""
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    db_path = root / "bottleqc-demo.sqlite3"
    overlay_path = root / "overlay.png"
    report_path = root / "report.json"
    evidence_root = root / "evidence"
    if any(path.exists() for path in (db_path, overlay_path, report_path, evidence_root)):
        raise FileExistsError(f"Demo output already exists; choose a new directory: {root}")

    values, config_hash, _ = Settings().load_config("default")
    config = InspectionConfig.from_mapping(values)
    run_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "bottleqc:phase-2:simulation-demo:v1"))
    repository = Repository(db_path)
    evidence = EvidenceStore(evidence_root)
    source = SimulationFrameSource()
    provider = SimulationDetectionProvider()
    engine = InspectionEngine(
        run_id=run_id,
        config=config,
        config_hash=config_hash,
        config_id="default",
        mode=RunMode.SIMULATION,
    )
    frame: Frame | None = None
    last_frame: Frame | None = None
    last_batch: DetectionBatch | None = None
    events = []
    try:
        repository.create_run({
            "run_id": run_id,
            "source_type": "simulation",
            "scenario": "phase-2-four-bottle-demo",
            "kind": "video",
            "mode": RunMode.SIMULATION.value,
            "config_id": "default",
            "config_hash": config_hash,
            "config_snapshot": json.dumps(values, sort_keys=True),
            "idempotency_key": "phase-2-four-bottle-demo",
            "request_hash": hashlib.sha256(b"phase-2-four-bottle-demo-v1").hexdigest(),
        })
        while (frame := source.read()) is not None:
            last_frame = frame
            last_batch = provider.detect(frame)
            for event in process_frame(engine, config, frame, last_batch):
                repository.insert_event(event)
                event_image = render_overlay(
                    frame.image,
                    last_batch.detections,
                    mode=RunMode.SIMULATION.value,
                    outcomes={event.track_id: event.outcome.value},
                )
                buffer = io.BytesIO()
                event_image.save(buffer, format="PNG", optimize=False)
                _, digest = evidence.write(run_id, event.event_id, buffer.getvalue())
                repository.update_evidence(event.event_id, state="AVAILABLE", sha256=digest)
                events.append(event)
        if last_frame is None or last_batch is None:
            raise RuntimeError("Simulation fixture produced no frames")
        for event in engine.finish(reason=EndReason.EOF, frame_index=last_frame.index, timestamp=last_frame.timestamp):
            repository.insert_event(event)
            events.append(event)
        repository.set_run_status(run_id, "COMPLETED")
        summary = repository.summary(run_id)
        outcomes = {event.track_id: event.outcome.value for event in events}
        overlay = render_overlay(
            last_frame.image,
            last_batch.detections,
            mode=RunMode.SIMULATION.value,
            outcomes=outcomes,
        )
        overlay.save(overlay_path, format="PNG", optimize=False)
        evidence_items = []
        for event in events:
            stored = repository.get_event(run_id, event.event_id)
            path = stored["evidence_path"]
            evidence_items.append({
                "path": f"evidence/{path}",
                "sha256": stored["evidence_sha256"],
                "track_id": event.track_id,
                "outcome": event.outcome.value,
            })
        result = {
            "mode": RunMode.SIMULATION.value,
            "scope": "Synthetic fixture through detection association, InspectionEngine, SQLite and verified evidence storage; not model evaluation.",
            "source": {"kind": "simulation_video", "frames": last_frame.index + 1, "timestamps": "ordered source seconds"},
            "summary": {key: summary[key] for key in ("PASS", "FAIL", "UNINSPECTED", "total")},
            "events": [
                {
                    "event_id": event.event_id,
                    "track_id": event.track_id,
                    "outcome": event.outcome.value,
                    "reason_codes": list(event.reason_codes),
                    "source_index": event.source_index,
                    "source_timestamp_ms": event.source_timestamp_ms,
                    "mode": event.mode.value,
                }
                for event in sorted(events, key=lambda item: item.track_id)
            ],
            "artifacts": {
                "overlay": "overlay.png",
                "overlay_sha256": hashlib.sha256(overlay_path.read_bytes()).hexdigest(),
                "evidence": evidence_items,
                "database": db_path.name,
            },
        }
        report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result
    except Exception as exc:
        try:
            current = repository.get_run(run_id)
            if current and current["status"] == "RUNNING":
                repository.set_run_status(run_id, "FAILED", str(exc))
        except Exception:
            pass
        raise
    finally:
        source.close()
        repository.close()
