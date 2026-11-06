from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from bottle_qc.domain.entities import Box, Detection
from bottle_qc.pipeline.contracts import DetectionBatch, Frame
from bottle_qc.vision.model_manifest import ModelManifest, ModelUnavailableError, load_manifest


@dataclass(slots=True)
class _TrackerState:
    box: Box
    missed_frames: int = 0


class YoloProvider:
    """Whole-frame detection with one previous box and gap count per run-scoped ID.

    State is capped by ``maximum_tracks``. IoU comparisons flag uncertainty heuristically;
    they cannot establish identity through occlusion or prove robustness on real media.
    """

    def __init__(
        self,
        manifest: ModelManifest,
        *,
        tracker: bool,
        minimum_confidence: float,
        minimum_region_edge_variance: float,
        tracker_iou_threshold: float = 0.20,
        tracker_ambiguity_margin: float = 0.025,
        maximum_missed_frames: int = 2,
        maximum_tracks: int = 4096,
    ):
        _validate_tracker_controls(
            tracker_iou_threshold,
            tracker_ambiguity_margin,
            maximum_missed_frames,
            maximum_tracks,
        )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelUnavailableError("Install the optional real-model dependencies before selecting REAL mode") from exc
        self.manifest = manifest
        self.tracker = tracker
        self.minimum_confidence = minimum_confidence
        self.minimum_region_edge_variance = minimum_region_edge_variance
        self.tracker_iou_threshold = tracker_iou_threshold
        self.tracker_ambiguity_margin = tracker_ambiguity_margin
        self.maximum_missed_frames = maximum_missed_frames
        self.maximum_tracks = maximum_tracks
        self._tracker_states: dict[str, _TrackerState] = {}
        self._uncertain_track_ids: set[str] = set()
        try:
            self._model = YOLO(str(manifest.weights_path))
        except Exception as exc:
            raise ModelUnavailableError(f"YOLO could not load the registered weights: {exc}") from exc
        if _normalize_task(getattr(self._model, "task", None)) != "detect":
            raise ModelUnavailableError("Loaded weights are not a detection model")
        self._validate_names(getattr(self._model, "names", None))
        self.device = str(getattr(self._model, "device", "automatic"))

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path,
        *,
        models_root: Path,
        tracker: bool,
        minimum_confidence: float,
        minimum_region_edge_variance: float,
        tracker_iou_threshold: float = 0.20,
        tracker_ambiguity_margin: float = 0.025,
        maximum_missed_frames: int = 2,
        maximum_tracks: int = 4096,
        probe_image: Image.Image | None = None,
    ) -> YoloProvider:
        manifest = load_manifest(manifest_path, models_root=models_root)
        provider = cls(
            manifest,
            tracker=tracker,
            minimum_confidence=minimum_confidence,
            minimum_region_edge_variance=minimum_region_edge_variance,
            tracker_iou_threshold=tracker_iou_threshold,
            tracker_ambiguity_margin=tracker_ambiguity_margin,
            maximum_missed_frames=maximum_missed_frames,
            maximum_tracks=maximum_tracks,
        )
        provider.probe(probe_image)
        return provider

    def probe(self, image: Image.Image | None = None) -> dict[str, object]:
        sample = np.zeros((640, 640, 3), dtype=np.uint8) if image is None else _to_bgr(image)
        try:
            results = self._model.predict(sample, verbose=False)
        except Exception as exc:
            raise ModelUnavailableError(f"Registered weights failed a real inference probe: {exc}") from exc
        if not isinstance(results, (tuple, list)) or not results or not hasattr(results[0], "boxes"):
            raise ModelUnavailableError("Inference probe returned no detection result structure")
        self._validate_names(getattr(results[0], "names", getattr(self._model, "names", None)))
        return {
            "available": True,
            "task": "detect",
            "classes": [self.manifest.class_mapping[index] for index in sorted(self.manifest.class_mapping)],
            "weights_sha256": self.manifest.weights_sha256,
            "device": self.device,
            "probe": "inference_completed",
        }

    def detect(self, frame: Frame) -> DetectionBatch:
        image = _to_bgr(frame.image)
        try:
            if self.tracker:
                results = self._model.track(image, persist=True, tracker="bytetrack.yaml", verbose=False)
            else:
                results = self._model.predict(image, verbose=False)
        except Exception as exc:
            raise RuntimeError(f"REAL inference failed: {exc}") from exc
        if not isinstance(results, (tuple, list)) or not results:
            raise RuntimeError("REAL inference returned no frame result")
        result = results[0]
        self._validate_names(getattr(result, "names", getattr(self._model, "names", None)))
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            raise RuntimeError("REAL inference result has no boxes structure")
        xyxy = _as_list(getattr(boxes, "xyxy", []))
        classes = _as_list(getattr(boxes, "cls", []))
        confidences = _as_list(getattr(boxes, "conf", []))
        ids = _as_list(getattr(boxes, "id", None)) if getattr(boxes, "id", None) is not None else None
        detections = []
        observability = {}
        tracked_bottles: list[tuple[str, Box]] = []
        image_width, image_height = frame.image.size
        for index, (coordinates, class_value, confidence_value) in enumerate(zip(xyxy, classes, confidences)):
            class_id = int(class_value)
            class_name = self.manifest.class_mapping.get(class_id)
            if class_name is None:
                raise ModelUnavailableError(f"Inference returned undeclared class id {class_id}")
            confidence = float(confidence_value)
            x1, y1, x2, y2 = (float(value) for value in coordinates)
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(image_width), x2), min(float(image_height), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            track_id = None
            if class_name == "bottle" and self.tracker:
                if ids is None or index >= len(ids) or ids[index] is None:
                    raise RuntimeError(
                        "REAL video tracking returned a bottle detection without a tracker ID "
                        f"at source frame {frame.index}; verify tracker output and model configuration"
                    )
                try:
                    raw_id = ids[index]
                    if type(raw_id) not in (int, float):
                        raise ValueError("tracker ID is not numeric")
                    if type(raw_id) is float and not math.isfinite(raw_id):
                        raise ValueError("tracker ID is not finite")
                    numeric_id = int(raw_id)
                    if numeric_id != raw_id:
                        raise ValueError("tracker ID is not an integer")
                    track_id = str(numeric_id)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise RuntimeError(
                        "REAL video tracking returned a bottle detection without a valid tracker ID "
                        f"at source frame {frame.index}; verify tracker output and model configuration"
                    ) from exc
            elif class_name == "bottle" and not self.tracker:
                track_id = f"image-{index + 1}"
            if confidence < self.minimum_confidence:
                continue
            box = Box(x1, y1, x2, y2)
            detections.append(Detection(
                class_name,
                box,
                confidence,
                frame.index,
                frame.timestamp,
                track_id,
            ))
            if class_name == "bottle":
                observability[track_id] = self._region_quality_flags(frame.image, box)
                if self.tracker:
                    tracked_bottles.append((track_id, box))
        uncertain = self._update_tracker_uncertainty(tracked_bottles) if self.tracker else frozenset()
        return DetectionBatch(tuple(detections), observability, uncertain)

    def _validate_names(self, names) -> None:
        if isinstance(names, dict):
            actual = {int(key): str(value).casefold() for key, value in names.items()}
        elif isinstance(names, (tuple, list)):
            actual = {index: str(value).casefold() for index, value in enumerate(names)}
        else:
            raise ModelUnavailableError("Loaded model does not expose a class-name mapping")
        if any(actual.get(index) != name for index, name in self.manifest.class_mapping.items()):
            raise ModelUnavailableError("Loaded weights do not expose manifest classes bottle, cap, and label")

    def _region_quality_flags(self, image: Image.Image, bottle: Box) -> tuple[bool, bool]:
        width, height = bottle.x2 - bottle.x1, bottle.y2 - bottle.y1
        cap = _crop(image, bottle.x1 - width * 0.08, bottle.y1 - height * 0.08,
                    bottle.x2 + width * 0.08, bottle.y1 + height * 0.28)
        label = _crop(image, bottle.x1 - width * 0.08, bottle.y1 + height * 0.20,
                      bottle.x2 + width * 0.08, bottle.y1 + height * 0.86)
        return (
            _region_image_quality_is_adequate(cap, self.minimum_region_edge_variance),
            _region_image_quality_is_adequate(label, self.minimum_region_edge_variance),
        )

    def _update_tracker_uncertainty(self, current_bottles: list[tuple[str, Box]]) -> frozenset[str]:
        current = dict(current_bottles)
        if len(set(self._tracker_states).union(current)) > self.maximum_tracks:
            raise RuntimeError(
                f"Post-tracker identity state reached maximum_tracks={self.maximum_tracks}; "
                "end the run or increase its configured track bound"
            )

        previous = {
            track_id: _TrackerState(state.box, state.missed_frames)
            for track_id, state in self._tracker_states.items()
        }

        for track_id, state in previous.items():
            if track_id not in current:
                missed = state.missed_frames + 1
                self._tracker_states[track_id].missed_frames = missed
                if missed > self.maximum_missed_frames:
                    self._uncertain_track_ids.add(track_id)

        for track_id, box in current.items():
            old = previous.get(track_id)
            candidates = [
                (candidate_id, candidate.box.intersection_over_union(box))
                for candidate_id, candidate in previous.items()
                if candidate.missed_frames <= self.maximum_missed_frames
            ]
            candidates.sort(key=lambda item: item[1], reverse=True)
            plausible = [
                item for item in candidates
                if item[1] > 0 and item[1] >= self.tracker_iou_threshold
            ]
            if old is not None:
                own_iou = old.box.intersection_over_union(box)
                if own_iou == 0 or own_iou < self.tracker_iou_threshold:
                    self._uncertain_track_ids.add(track_id)
                    if plausible:
                        self._uncertain_track_ids.add(plausible[0][0])
                elif len(plausible) > 1:
                    best = plausible[0][1]
                    competitors = [
                        candidate_id
                        for candidate_id, score in plausible
                        if best - score <= self.tracker_ambiguity_margin
                    ]
                    if len(competitors) > 1:
                        self._uncertain_track_ids.add(track_id)
                        self._uncertain_track_ids.update(competitors)
                self._tracker_states[track_id] = _TrackerState(box)
            else:
                self._tracker_states[track_id] = _TrackerState(box)
                if plausible:
                    self._uncertain_track_ids.add(track_id)
                    self._uncertain_track_ids.add(plausible[0][0])

        current_items = list(current.items())
        for index, (first_id, first_box) in enumerate(current_items):
            for second_id, second_box in current_items[index + 1:]:
                overlap = first_box.intersection_over_union(second_box)
                if overlap > 0 and overlap >= self.tracker_iou_threshold:
                    self._uncertain_track_ids.update((first_id, second_id))

        return frozenset(self._uncertain_track_ids)


def open_yolo_provider(
    manifest_path: Path,
    *,
    models_root: Path,
    tracker: bool,
    minimum_confidence: float,
    minimum_region_edge_variance: float,
    tracker_iou_threshold: float = 0.20,
    tracker_ambiguity_margin: float = 0.025,
    maximum_missed_frames: int = 2,
    maximum_tracks: int = 4096,
    probe_image: Image.Image | None = None,
) -> YoloProvider:
    return YoloProvider.from_manifest(
        manifest_path,
        models_root=models_root,
        tracker=tracker,
        minimum_confidence=minimum_confidence,
        minimum_region_edge_variance=minimum_region_edge_variance,
        tracker_iou_threshold=tracker_iou_threshold,
        tracker_ambiguity_margin=tracker_ambiguity_margin,
        maximum_missed_frames=maximum_missed_frames,
        maximum_tracks=maximum_tracks,
        probe_image=probe_image,
    )


def _normalize_task(value: object) -> str | None:
    return "detect" if value in {"detect", "detection"} else None


def _to_bgr(image: Image.Image) -> np.ndarray:
    import cv2

    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _as_list(value) -> list:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _crop(image: Image.Image, x1: float, y1: float, x2: float, y2: float) -> Image.Image:
    left, top = max(0, int(math.floor(x1))), max(0, int(math.floor(y1)))
    right, bottom = min(image.width, int(math.ceil(x2))), min(image.height, int(math.ceil(y2)))
    return image.convert("RGB").crop((left, top, right, bottom))


def _region_image_quality_is_adequate(region: Image.Image, threshold: float) -> bool:
    """Laplacian variance is an image-quality heuristic, not an occlusion/clear-region test."""
    if region.width < 8 or region.height < 8:
        return False
    import cv2

    gray = cv2.cvtColor(np.asarray(region), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) >= threshold


def _validate_tracker_controls(iou_threshold, ambiguity_margin, maximum_missed_frames, maximum_tracks) -> None:
    for name, value in (("tracker_iou_threshold", iou_threshold), ("tracker_ambiguity_margin", ambiguity_margin)):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and between zero and one")
    if type(maximum_missed_frames) is not int or not 0 <= maximum_missed_frames <= 1000:
        raise ValueError("maximum_missed_frames must be an integer between zero and 1000")
    if type(maximum_tracks) is not int or not 1 <= maximum_tracks <= 100000:
        raise ValueError("maximum_tracks must be an integer between one and 100000")
