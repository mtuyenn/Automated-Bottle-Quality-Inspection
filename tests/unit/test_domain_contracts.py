from dataclasses import replace

import pytest

from bottle_qc.domain.config import Direction, InspectionConfig
from bottle_qc.domain.entities import EndReason, ObservationReason, RunMode
from bottle_qc.domain.inspection import InspectionEngine
from bottle_qc.settings import ConfigurationError, Settings


@pytest.mark.parametrize("changes", [
    {"minimum_observations": 0}, {"minimum_observations": True}, {"minimum_observations": 2.5},
    {"maximum_tracks": 0}, {"cap_support_threshold": 0}, {"label_support_threshold": 1.1},
    {"track_timeout_seconds": 0}, {"minimum_observation_duration": -1},
    {"frame_aspect_ratio": float("nan")}, {"frame_aspect_ratio": float("inf")},
    {"minimum_confidence": True}, {"direction": "RIGHT_TO_LEFT"},
    {"direction": Direction.RIGHT_TO_LEFT}, {"entry_line": .85}, {"line_hysteresis": .4},
])
def test_invalid_policy_is_rejected(config, changes):
    with pytest.raises(ValueError):
        replace(config, **changes)


def test_default_toml_loads_typed_policy_and_its_hash_changes_with_threshold():
    data, digest, path = Settings().load_config("default")
    policy = InspectionConfig.from_mapping(data)
    assert path.is_file() and len(digest) == 64
    assert policy.minimum_observations == 3
    assert policy.fingerprint != replace(policy, cap_support_threshold=.9).fingerprint


def test_default_tracker_uncertainty_controls_are_bounded_in_run_config():
    values, _, _ = Settings().load_config("default")
    assert values["tracker_iou_threshold"] == .2
    assert values["tracker_ambiguity_margin"] == .025
    assert values["maximum_missed_frames"] == 2


def test_maximum_missed_frames_must_be_an_integer(tmp_path):
    original = Settings().load_config("default")[2].read_text(encoding="utf-8")
    (tmp_path / "bad.toml").write_text(
        original.replace("maximum_missed_frames = 2", "maximum_missed_frames = 1.5"),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="maximum_missed_frames"):
        Settings(configs_dir=tmp_path).load_config("bad")


@pytest.mark.parametrize(("setting", "original", "invalid"), [
    ("maximum_missed_frames", "maximum_missed_frames = 2", "maximum_missed_frames = 1001"),
    ("tracker_iou_threshold", "tracker_iou_threshold = 0.20", "tracker_iou_threshold = 1.01"),
    ("tracker_ambiguity_margin", "tracker_ambiguity_margin = 0.025", "tracker_ambiguity_margin = -0.01"),
])
def test_tracker_uncertainty_controls_reject_values_outside_their_bounds(tmp_path, setting, original, invalid):
    source = Settings().load_config("default")[2].read_text(encoding="utf-8")
    (tmp_path / "bad.toml").write_text(source.replace(original, invalid), encoding="utf-8")
    with pytest.raises(ConfigurationError, match=setting):
        Settings(configs_dir=tmp_path).load_config("bad")


def test_invalid_domain_policy_in_toml_is_reported(tmp_path):
    original = Settings().load_config("default")[2].read_text(encoding="utf-8")
    (tmp_path / "bad.toml").write_text(original.replace("cap_support_threshold = 0.7", "cap_support_threshold = 0.0"), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="inspection policy"):
        Settings(configs_dir=tmp_path).load_config("bad")


def test_same_frame_cannot_inflate_observation_count(config, step):
    engine = InspectionEngine(run_id="run-1", config=replace(config, minimum_observation_duration=0), mode=RunMode.SIMULATION)
    step(engine, 0, .1)
    step(engine, 1, .3)
    with pytest.raises(ValueError, match="increasing"):
        step(engine, 1, .3)
    step(engine, 2, .5)
    event, = step(engine, 3, .82)
    assert event.valid_observation_count == 2 and event.outcome.value == "UNINSPECTED"


@pytest.mark.parametrize("error", ["run", "frame", "timestamp", "duplicate", "aspect", "bounds"])
def test_invalid_frame_does_not_partly_mutate_engine(engine, observe, step, error):
    step(engine, 0, .1)
    observations = observe(engine, 1, .4)
    frame_size = (1000, 1000)
    if error == "run":
        observations[0] = replace(observations[0], run_id="other")
    elif error in {"frame", "timestamp"}:
        name = "frame_index" if error == "frame" else "timestamp"
        observations[0] = replace(observations[0], bottle=replace(observations[0].bottle, **{name: 99}))
    elif error == "duplicate":
        observations *= 2
    elif error == "aspect":
        frame_size = (1000, 900)
    else:
        from bottle_qc.domain.entities import Box
        observations[0] = replace(observations[0], bottle=replace(observations[0].bottle, box=Box(0, 0, 1001, 1000)))
    with pytest.raises(ValueError):
        engine.process_frame(frame_index=1, timestamp=.1, frame_size=frame_size, observations=observations)
    assert step(engine, 1, .4) == ()  # The rejected frame did not consume index 1.
    event, = engine.finish(reason=EndReason.EOF, frame_index=1, timestamp=.1)
    assert event.valid_observation_count == 1


def test_engine_accepts_admission_rounding_tolerance_and_rejects_beyond_it():
    config = InspectionConfig(frame_aspect_ratio=1.0)
    engine = InspectionEngine(run_id="ratio", config=config, mode=RunMode.SIMULATION)
    engine.process_frame(frame_index=0, timestamp=0, frame_size=(1005, 1000), observations=[])

    rejected = InspectionEngine(run_id="ratio-outside", config=config, mode=RunMode.SIMULATION)
    with pytest.raises(ValueError, match="aspect ratio"):
        rejected.process_frame(frame_index=0, timestamp=0, frame_size=(1006, 1000), observations=[])


def test_engine_accepts_854_by_480_for_configured_16_by_9():
    config = InspectionConfig()
    engine = InspectionEngine(run_id="rounded-16x9", config=config, mode=RunMode.SIMULATION)
    engine.process_frame(frame_index=0, timestamp=0, frame_size=(854, 480), observations=[])


def test_capacity_keeps_finalized_identity_to_prevent_recount(config, step):
    engine = InspectionEngine(run_id="run-1", config=replace(config, maximum_tracks=1), mode=RunMode.SIMULATION)
    for index, progress in enumerate((.1, .3, .4, .5, .6, .7)):
        step(engine, index, progress)
    event, = step(engine, 6, .82)
    assert event.outcome.value == "PASS"
    with pytest.raises(ValueError, match="maximum_tracks"):
        step(engine, 7, .1, track_id="2")
    assert step(engine, 7, .82) == ()
    assert engine.counters["PASS"] == 1


def test_frame_that_bypasses_roi_vertical_band_is_not_a_valid_exit(engine, step):
    for index, progress in enumerate((.1, .3, .4, .5, .6, .7)):
        step(engine, index, progress)
    assert step(engine, 6, .82, y=.95, half_height=.02) == ()
    assert step(engine, 7, .85) == ()
    assert step(engine, 8, .86) == ()
    event, = engine.finish(reason=EndReason.EOF, frame_index=8, timestamp=.8)
    assert event.outcome.value == "UNINSPECTED"


def test_support_threshold_is_configurable(config, step):
    engine = InspectionEngine(run_id="run-1", config=replace(config, cap_support_threshold=.9), mode=RunMode.SIMULATION)
    for index, progress in enumerate((.1, .3, .4, .5, .6, .7)):
        step(engine, index, progress, cap=index != 3)
    event, = step(engine, 6, .82)
    assert event.cap_support == .8
    assert event.outcome.value == "FAIL" and "CAP_MISSING" in event.reason_codes


def test_identity_reason_code_cannot_be_overridden_by_later_good_observations(engine, observe, step):
    step(engine, 0, .1)
    observation, = observe(engine, 1, .3)
    observation = replace(observation, reasons=(ObservationReason.IDENTITY_UNCERTAIN,))
    engine.process_frame(frame_index=1, timestamp=.1, frame_size=(1000, 1000), observations=[observation])
    for index, progress in enumerate((.4, .5, .6, .7), start=2):
        step(engine, index, progress)
    event, = step(engine, 6, .82)
    assert event.outcome.value == "UNINSPECTED" and "IDENTITY_UNCERTAIN" in event.reason_codes


def test_unknown_observation_reason_is_rejected_instead_of_growing_track_history(engine, observe):
    observation, = observe(engine, 0, .1)
    with pytest.raises(ValueError, match="reason codes"):
        replace(observation, reasons=("unbounded arbitrary diagnostic text",))
