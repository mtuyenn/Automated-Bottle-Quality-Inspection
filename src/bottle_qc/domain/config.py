"""Validated domain policy. Coordinates are normalized; durations are source seconds."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
import hashlib
import json
import math
from typing import Any

from bottle_qc.domain.entities import Box


FRAME_ASPECT_RATIO_RELATIVE_TOLERANCE = 0.005
FRAME_ASPECT_RATIO_ABSOLUTE_TOLERANCE = 1e-6


def frame_aspect_ratio_matches(width: int, height: int, configured_ratio: float) -> bool:
    """Use the same 0.5% rounding tolerance at admission and engine validation."""
    measured_ratio = width / height
    tolerance = max(FRAME_ASPECT_RATIO_ABSOLUTE_TOLERANCE,
                    abs(configured_ratio) * FRAME_ASPECT_RATIO_RELATIVE_TOLERANCE)
    return abs(measured_ratio - configured_ratio) <= tolerance


class Direction(StrEnum):
    LEFT_TO_RIGHT = "LEFT_TO_RIGHT"
    RIGHT_TO_LEFT = "RIGHT_TO_LEFT"


@dataclass(frozen=True, slots=True)
class InspectionConfig:
    roi: Box = Box(0.1, 0.1, 0.9, 0.9)
    entry_line: float = 0.2
    exit_line: float = 0.8
    direction: Direction = Direction.LEFT_TO_RIGHT
    frame_aspect_ratio: float = 16 / 9
    line_hysteresis: float = 0.01
    minimum_confidence: float = 0.35
    minimum_observations: int = 3
    minimum_observation_duration: float = 0.2
    cap_support_threshold: float = 0.7
    label_support_threshold: float = 0.7
    track_timeout_seconds: float = 0.5
    maximum_tracks: int = 4096
    association_ambiguity_margin: float = 0.03
    minimum_component_overlap: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.direction, Direction):
            raise ValueError("direction must be a Direction")
        if not isinstance(self.roi, Box) or not (0 <= self.roi.x1 < self.roi.x2 <= 1 and 0 <= self.roi.y1 < self.roi.y2 <= 1):
            raise ValueError("roi must be a box within normalized image coordinates")
        for name in (
            "entry_line", "exit_line", "frame_aspect_ratio", "line_hysteresis", "minimum_confidence",
            "minimum_observation_duration", "cap_support_threshold", "label_support_threshold",
            "track_timeout_seconds", "association_ambiguity_margin", "minimum_component_overlap",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        for name in ("minimum_observations", "maximum_tracks"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.frame_aspect_ratio <= 0 or self.track_timeout_seconds <= 0 or self.line_hysteresis <= 0:
            raise ValueError("aspect ratio, timeout and hysteresis must be positive")
        if self.minimum_observation_duration < 0 or self.association_ambiguity_margin < 0:
            raise ValueError("duration and ambiguity margin must be nonnegative")
        if not 0 <= self.minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be between zero and one")
        for name in ("cap_support_threshold", "label_support_threshold", "minimum_component_overlap"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be greater than zero and at most one")
        left, right = sorted((self.entry_line, self.exit_line))
        if not self.roi.x1 < left - self.line_hysteresis < right + self.line_hysteresis < self.roi.x2:
            raise ValueError("entry/exit lines and hysteresis must fit inside roi")
        if self.entry_progress + 2 * self.line_hysteresis >= self.exit_progress:
            raise ValueError("entry must precede exit with nonoverlapping hysteresis bands")

    def progress(self, x: float) -> float:
        return x if self.direction == Direction.LEFT_TO_RIGHT else 1 - x

    @property
    def entry_progress(self) -> float:
        return self.progress(self.entry_line)

    @property
    def exit_progress(self) -> float:
        return self.progress(self.exit_line)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> InspectionConfig:
        policy = {item.name: values[item.name] for item in fields(cls)}
        policy["roi"] = Box(*policy["roi"])
        policy["direction"] = Direction(policy["direction"])
        return cls(**policy)
