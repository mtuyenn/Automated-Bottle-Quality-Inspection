from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import math
from typing import Any


class Outcome(StrEnum):
    PENDING = "PENDING"
    PASS = "PASS"
    FAIL = "FAIL"
    UNINSPECTED = "UNINSPECTED"


class RunMode(StrEnum):
    SIMULATION = "SIMULATION"
    REAL = "REAL"


class EndReason(StrEnum):
    EOF = "VIDEO_ENDED"
    STOP = "RUN_STOPPED"
    ERROR = "RUN_FAILED"


class ObservationReason(StrEnum):
    AMBIGUOUS_COMPONENT_ASSOCIATION = "AMBIGUOUS_COMPONENT_ASSOCIATION"
    IDENTITY_UNCERTAIN = "IDENTITY_UNCERTAIN"
    CAP_NOT_OBSERVABLE = "CAP_NOT_OBSERVABLE"
    LABEL_NOT_OBSERVABLE = "LABEL_NOT_OBSERVABLE"
    CAP_OCCLUDED = "CAP_OCCLUDED"
    LABEL_OCCLUDED = "LABEL_OCCLUDED"
    LOW_BOTTLE_CONFIDENCE = "LOW_BOTTLE_CONFIDENCE"


def validate_timestamp(timestamp: float) -> None:
    if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("timestamp must be a finite, nonnegative number of source seconds")


def validate_frame_index(frame_index: int) -> None:
    if type(frame_index) is not int or frame_index < 0:
        raise ValueError("frame_index must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2)
        if not all(type(value) in (int, float) and math.isfinite(value) for value in values):
            raise ValueError("box coordinates must be finite numbers")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("box must have positive width and height")

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def intersection_over_union(self, other: Box) -> float:
        width = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        height = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        intersection = width * height
        union = self.area + other.area - intersection
        return intersection / union if union else 0.0


@dataclass(frozen=True, slots=True)
class Detection:
    class_name: str
    box: Box
    confidence: float
    frame_index: int
    timestamp: float
    track_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.class_name, str) or not self.class_name.strip():
            raise ValueError("class_name must not be empty")
        if not isinstance(self.box, Box):
            raise ValueError("box must be a Box")
        if type(self.confidence) not in (int, float) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")
        validate_frame_index(self.frame_index)
        validate_timestamp(self.timestamp)
        if self.track_id is not None and (not isinstance(self.track_id, str) or not self.track_id.strip()):
            raise ValueError("track_id must be a nonempty string when supplied")


@dataclass(frozen=True, slots=True)
class BottleObservation:
    run_id: str
    track_id: str
    bottle: Detection
    cap_present: bool
    label_present: bool
    cap_observable: bool
    label_observable: bool
    ambiguous: bool = False
    identity_uncertain: bool = False
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if not isinstance(self.track_id, str) or not self.track_id.strip():
            raise ValueError("track_id must not be empty")
        if self.bottle.class_name.casefold() != "bottle" or self.bottle.track_id != self.track_id:
            raise ValueError("observation must refer to its tracked bottle detection")
        for name in ("cap_present", "label_present", "cap_observable", "label_observable", "ambiguous", "identity_uncertain"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if not isinstance(self.reasons, tuple) or any(not isinstance(reason, str) or reason not in ObservationReason._value2member_map_ for reason in self.reasons):
            raise ValueError("reasons must be a tuple of supported observation reason codes")

    @property
    def timestamp(self) -> float:
        return self.bottle.timestamp

    @property
    def frame_index(self) -> int:
        return self.bottle.frame_index


@dataclass(slots=True)
class InspectionEvent:
    event_id: str
    run_id: str
    source_item_id: str
    track_id: str
    outcome: Outcome
    mode: RunMode
    reason_codes: tuple[str, ...]
    source_index: int | None
    source_timestamp_ms: float | None
    first_seen_at: float
    finalized_at: float
    valid_observation_count: int
    observation_duration: float
    cap_support: float | None
    label_support: float | None
    model_id: str | None = None
    model_hash: str | None = None
    config_id: str | None = None
    config_hash: str | None = None
    evidence_state: str = "PENDING"
    evidence_path: str | None = None
    evidence_sha256: str | None = None
    evidence_error: str | None = None
    created_at: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
