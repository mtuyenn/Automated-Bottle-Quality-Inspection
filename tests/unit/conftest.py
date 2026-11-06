from dataclasses import replace

import pytest

from bottle_qc.domain.association import associate_detections
from bottle_qc.domain.config import Direction, InspectionConfig
from bottle_qc.domain.entities import Box, Detection, RunMode
from bottle_qc.domain.inspection import InspectionEngine


@pytest.fixture
def config():
    return InspectionConfig(roi=Box(0.05, 0.1, 0.95, 0.9), frame_aspect_ratio=1.0)


@pytest.fixture
def engine(config):
    return InspectionEngine(run_id="run-1", config=config, mode=RunMode.SIMULATION)


@pytest.fixture
def detections():
    def make(index, progress, *, track_id="1", cap=True, label=True, cap_confidence=0.95,
             bottle_confidence=0.99, timestamp=None, direction=Direction.LEFT_TO_RIGHT, size=1000, y=0.5,
             half_height=.3):
        timestamp = index / 10 if timestamp is None else timestamp
        x = progress if direction == Direction.LEFT_TO_RIGHT else 1 - progress
        cx, cy = x * size, y * size
        values = [Detection("bottle", Box(max(0, cx - .04 * size), max(0, cy - half_height * size), min(size, cx + .04 * size), min(size, cy + half_height * size)), bottle_confidence, index, timestamp, track_id)]
        if cap:
            values.append(Detection("cap", Box(cx - .015 * size, cy - .31 * size, cx + .015 * size, cy - .27 * size), cap_confidence, index, timestamp))
        if label:
            values.append(Detection("label", Box(cx - .025 * size, cy - .05 * size, cx + .025 * size, cy + .07 * size), .95, index, timestamp))
        return values
    return make


@pytest.fixture
def observe(detections):
    def make(engine, index, progress, *, clear=(True, True), ambiguous=False, identity_uncertain=False, **kwargs):
        values = detections(index, progress, direction=engine.config.direction, **kwargs)
        track_id = values[0].track_id
        observations = associate_detections(
            values, run_id=engine.run_id, config=engine.config,
            observability={} if clear is None else {track_id: clear},
            uncertain_track_ids=(track_id,) if identity_uncertain else (),
        )
        if ambiguous:
            observations = [replace(item, ambiguous=True, reasons=("AMBIGUOUS_COMPONENT_ASSOCIATION",)) for item in observations]
        return observations
    return make


@pytest.fixture
def step(observe):
    def feed(engine, index, progress=None, *, timestamp=None, size=1000, **kwargs):
        timestamp = index / 10 if timestamp is None else timestamp
        observations = [] if progress is None else observe(engine, index, progress, timestamp=timestamp, size=size, **kwargs)
        return engine.process_frame(frame_index=index, timestamp=timestamp, frame_size=(size, size), observations=observations)
    return feed
