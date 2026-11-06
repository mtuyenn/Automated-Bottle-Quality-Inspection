from __future__ import annotations

import math
from pathlib import Path

import cv2
from PIL import Image

from bottle_qc.pipeline.contracts import Frame


class ImageFileSource:
    """One still image, with one source item and no temporal state."""

    kind = "image"

    def __init__(self, path: Path, *, source_item_id: str):
        self.path = Path(path)
        self.source_item_id = source_item_id
        try:
            with Image.open(self.path) as source:
                source.verify()
            with Image.open(self.path) as source:
                self._image = source.convert("RGB")
        except Exception as exc:
            raise ValueError(f"Image cannot be decoded: {exc}") from exc
        self._read = False

    def read(self) -> Frame | None:
        if self._read:
            return None
        self._read = True
        return Frame(0, 0.0, self._image.copy(), self.source_item_id, 0.0)

    def close(self) -> None:
        self._image.close()


class VideoFileSource:
    """Sequential file decoder. Every decoded inspection frame is returned in order."""

    kind = "video"

    def __init__(self, path: Path, *, source_item_id: str):
        self.path = Path(path)
        self.source_item_id = source_item_id
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            self._capture.release()
            raise ValueError("Video cannot be opened by the local decoder")
        self._index = 0
        fps = self._capture.get(cv2.CAP_PROP_FPS)
        self._fps = float(fps) if math.isfinite(fps) and fps > 0 else None
        count = self._capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self._frame_count = int(count) if math.isfinite(count) and count >= 0 else None
        self._last_timestamp_ms = -1.0
        self._closed = False

    def read(self) -> Frame | None:
        if self._closed:
            return None
        ok, bgr = self._capture.read()
        if not ok:
            if self._frame_count and self._index < self._frame_count:
                raise ValueError(f"Video decode stopped at frame {self._index} before declared frame count {self._frame_count}")
            return None
        raw_timestamp_ms = float(self._capture.get(cv2.CAP_PROP_POS_MSEC))
        if math.isfinite(raw_timestamp_ms) and raw_timestamp_ms >= 0 and (self._index == 0 or raw_timestamp_ms > self._last_timestamp_ms):
            source_timestamp_ms = raw_timestamp_ms
        elif self._fps is not None:
            source_timestamp_ms = self._index / self._fps * 1000.0
        else:
            source_timestamp_ms = None
        domain_timestamp = source_timestamp_ms / 1000.0 if source_timestamp_ms is not None else float(self._index)
        if source_timestamp_ms is not None:
            self._last_timestamp_ms = source_timestamp_ms
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        frame = Frame(
            self._index,
            domain_timestamp,
            image,
            f"{self.source_item_id}:frame-{self._index:08d}",
            source_timestamp_ms,
        )
        self._index += 1
        return frame

    @property
    def frames_read(self) -> int:
        return self._index

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._capture.release()
