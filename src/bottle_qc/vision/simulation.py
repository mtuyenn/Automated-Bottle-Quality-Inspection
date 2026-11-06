from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageDraw

from bottle_qc.domain.entities import Box, Detection
from bottle_qc.pipeline.contracts import DetectionBatch, Frame


FRAME_SIZE = (960, 540)
POSITIONS = (0.10, 0.30, 0.40, 0.50, 0.60, 0.70, 0.82)
STAGGER_FRAMES = 2


@dataclass(frozen=True, slots=True)
class SimulationCase:
    name: str
    cap_present: bool
    label_present: bool
    cap_observable: bool = True
    dropout_frame: int | None = None


DEMO_CASES = (
    SimulationCase("normal_with_one_dropout", True, True, dropout_frame=3),
    SimulationCase("missing_cap", False, True),
    SimulationCase("missing_label", True, False),
    SimulationCase("occluded_cap_region", False, True, cap_observable=False),
)
FRAME_COUNT = len(POSITIONS) + STAGGER_FRAMES * (len(DEMO_CASES) - 1)


class SimulationFrameSource:
    """Deterministic offline video fixture; never represents a camera or real media."""

    kind = "video"

    def __init__(self, *, width: int = FRAME_SIZE[0], height: int = FRAME_SIZE[1], fps: float = 10.0):
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("simulation dimensions and source FPS must be positive")
        self.width, self.height, self.fps = width, height, fps
        self._index = 0
        self._closed = False

    def read(self) -> Frame | None:
        if self._closed:
            return None
        if self._index >= FRAME_COUNT:
            return None
        index = self._index
        self._index += 1
        image = Image.new("RGB", (self.width, self.height), (246, 248, 250))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, self.width, 42), fill=(24, 38, 56))
        draw.text((16, 13), "SIMULATION  |  deterministic fixture video", fill=(255, 255, 255))
        for row, case in enumerate(DEMO_CASES):
            local_index = index - row * STAGGER_FRAMES
            if not 0 <= local_index < len(POSITIONS):
                continue
            center_x = int(POSITIONS[local_index] * self.width)
            center_y = int((0.17 + row * 0.22) * self.height)
            half_width = max(18, int(self.width * 0.034))
            half_height = max(24, int(self.height * 0.09))
            bottle = (center_x - half_width, center_y - half_height, center_x + half_width, center_y + half_height)
            draw.rounded_rectangle(bottle, radius=8, fill=(218, 226, 234), outline=(52, 73, 94), width=3)
            if case.cap_present and case.dropout_frame != local_index:
                draw.rectangle((center_x - 9, center_y - half_height - 6, center_x + 9, center_y - half_height + 5), fill=(36, 112, 76))
            if case.label_present:
                draw.rectangle((center_x - 14, center_y - 7, center_x + 14, center_y + 15), fill=(241, 194, 72), outline=(102, 83, 40))
            draw.text((center_x - 52, center_y + half_height + 6), case.name, fill=(35, 44, 55))
        return Frame(index, index / self.fps, image, f"frame-{index:06d}", index / self.fps * 1000)

    def close(self) -> None:
        self._closed = True


class SimulationDetectionProvider:
    """Fixture detections for the four named cases, not a learned detector."""

    def detect(self, frame: Frame) -> DetectionBatch:
        width, height = frame.image.size
        timestamp = frame.timestamp
        detections: list[Detection] = []
        observability: dict[str, tuple[bool, bool]] = {}
        for row, case in enumerate(DEMO_CASES):
            local_index = frame.index - row * STAGGER_FRAMES
            if not 0 <= local_index < len(POSITIONS):
                continue
            x = POSITIONS[local_index] * width
            track_id = str(row + 1)
            y = (0.17 + row * 0.22) * height
            half_width, half_height = width * 0.034, height * 0.09
            box = _clip_box(x - half_width, y - half_height, x + half_width, y + half_height, width, height)
            detections.append(Detection("bottle", box, 0.99, frame.index, timestamp, track_id))
            if case.cap_present and case.dropout_frame != local_index:
                detections.append(Detection(
                    "cap", _clip_box(x - width * 0.008, y - half_height - height * 0.005,
                                     x + width * 0.008, y - half_height + height * 0.035, width, height),
                    0.95, frame.index, timestamp,
                ))
            if case.label_present:
                detections.append(Detection(
                    "label", _clip_box(x - width * 0.012, y + height * 0.005,
                                       x + width * 0.012, y + height * 0.05, width, height),
                    0.95, frame.index, timestamp,
                ))
            observability[track_id] = (case.cap_observable, True)
        return DetectionBatch(tuple(detections), observability)


def _clip_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Box:
    return Box(max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2))
