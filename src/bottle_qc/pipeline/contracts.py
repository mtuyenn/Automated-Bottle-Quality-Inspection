from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from PIL import Image

from bottle_qc.domain.entities import Detection


@dataclass(frozen=True, slots=True)
class Frame:
    """One decoded source frame. Timestamps are source seconds."""

    index: int
    timestamp: float
    image: Image.Image
    source_item_id: str
    source_timestamp_ms: float | None = None


@dataclass(frozen=True, slots=True)
class DetectionBatch:
    """Frame detections plus region-quality flags and provider-scoped uncertain IDs.

    Region quality is not evidence that an undetected component region is clear.
    """

    detections: tuple[Detection, ...]
    observability: dict[str, tuple[bool, bool]]
    uncertain_track_ids: frozenset[str] = frozenset()


class FrameSource(Protocol):
    kind: str

    def read(self) -> Frame | None: ...

    def close(self) -> None: ...


class DetectionProvider(Protocol):
    def detect(self, frame: Frame) -> DetectionBatch: ...
