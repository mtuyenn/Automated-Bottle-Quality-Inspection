from dataclasses import replace

import pytest

from bottle_qc.domain.association import associate_detections
from bottle_qc.domain.config import Direction
from bottle_qc.domain.entities import EndReason, Outcome, RunMode
from bottle_qc.domain.inspection import InspectionEngine


PATH = (.1, .3, .4, .5, .6, .7, .82)


@pytest.mark.parametrize("direction", list(Direction))
@pytest.mark.parametrize("cap,label,expected,reason", [
    (True, True, Outcome.PASS, "COMPONENTS_CONFIRMED"),
    (False, True, Outcome.FAIL, "CAP_MISSING"),
    (True, False, Outcome.FAIL, "LABEL_MISSING"),
    (False, False, Outcome.FAIL, "CAP_MISSING"),
])
def test_complete_transit_has_one_terminal_event(config, step, direction, cap, label, expected, reason):
    # Five clear opportunities inside the entry/exit interval; presence determines the outcome.
    if direction == Direction.RIGHT_TO_LEFT:
        config = replace(config, direction=direction, entry_line=.8, exit_line=.2)
    engine = InspectionEngine(run_id="complete", config=config, mode=RunMode.SIMULATION)
    for index, progress in enumerate(PATH[:-1]):
        assert step(engine, index, progress, cap=cap, label=label) == ()
        assert engine.statuses == {"1": Outcome.PENDING}
        assert sum(engine.counters.values()) == 0
    events = step(engine, 6, PATH[-1], cap=cap, label=label)
    assert len(events) == 1
    event = events[0]
    assert event.outcome == expected
    assert reason in event.reason_codes and "EXIT_LINE_CROSSED" in event.reason_codes
    assert event.valid_observation_count == 5
    assert event.cap_support == float(cap) and event.label_support == float(label)
    assert event.observation_duration == pytest.approx(.4)
    assert event.first_seen_at == 0 and event.finalized_at == pytest.approx(.6)
    assert event.source_index == 6 and event.source_timestamp_ms == pytest.approx(600)
    assert event.config_hash == config.fingerprint and event.mode == RunMode.SIMULATION
    assert engine.counters[expected.value] == 1
    assert engine.finish(reason=EndReason.EOF, frame_index=6, timestamp=.6) == ()

def test_one_cap_dropout_is_not_an_immediate_failure(engine, step):
    for index, progress in enumerate(PATH[:-1]):
        assert step(engine, index, progress, cap=index != 3) == ()
    event, = step(engine, 6, .82)
    assert event.outcome == Outcome.PASS
    assert event.valid_observation_count == 5 and event.cap_support == .8
    assert "CAP_MISSING" not in event.reason_codes


@pytest.mark.parametrize("clear", [(False, True), (True, False), (False, False), None])
def test_missing_or_unknown_visibility_cannot_become_failure(engine, step, clear):
    for index, progress in enumerate(PATH[:-1]):
        assert step(engine, index, progress, cap=False, clear=clear) == ()
    event, = step(engine, 6, .82, cap=False, clear=clear)
    assert event.outcome == Outcome.UNINSPECTED
    assert event.valid_observation_count == 0 and event.cap_support is None
    assert "INSUFFICIENT_OBSERVATIONS" in event.reason_codes
    assert {"CAP_NOT_OBSERVABLE", "LABEL_NOT_OBSERVABLE"}.intersection(event.reason_codes)


def test_enough_later_clear_frames_resolve_a_component_ambiguity(engine, step):
    for index, progress in enumerate(PATH[:-1]):
        assert step(engine, index, progress, ambiguous=index == 2) == ()
    event, = step(engine, 6, .82)
    assert event.outcome == Outcome.PASS and event.valid_observation_count == 4
    assert event.extra["rejected_observation_reasons"] == ["AMBIGUOUS_COMPONENT_ASSOCIATION"]


def test_unresolved_association_ambiguity_is_uninspected(engine, step):
    for index, progress in enumerate(PATH[:-1]):
        assert step(engine, index, progress, ambiguous=True) == ()
    event, = step(engine, 6, .82, ambiguous=True)
    assert event.outcome == Outcome.UNINSPECTED
    assert "AMBIGUOUS_COMPONENT_ASSOCIATION" in event.reason_codes


def test_identity_uncertainty_survives_later_clear_frames(engine, step):
    for index, progress in enumerate(PATH[:-1]):
        assert step(engine, index, progress, identity_uncertain=index == 2) == ()
    event, = step(engine, 6, .82)
    assert event.outcome == Outcome.UNINSPECTED
    assert "IDENTITY_UNCERTAIN" in event.reason_codes


def test_hysteresis_and_repeated_crossings_never_emit_twice(engine, step):
    for index, progress in enumerate(PATH[:-1] + (.795, .805, .799, .809)):
        assert step(engine, index, progress) == ()
    event, = step(engine, 10, .82)
    assert event.outcome == Outcome.PASS
    for index, progress in enumerate((.79, .82, .8, .9), start=11):
        assert step(engine, index, progress) == ()
    assert engine.counters == {"PASS": 1, "FAIL": 0, "UNINSPECTED": 0}


@pytest.mark.parametrize("reason", list(EndReason))
def test_finish_flushes_pending_tracks_once_even_with_enough_good_observations(engine, step, reason):
    for index, progress in enumerate(PATH[:4]):
        assert step(engine, index, progress) == ()
    event, = engine.finish(reason=reason, frame_index=3, timestamp=.3)
    assert event.outcome == Outcome.UNINSPECTED and reason.value in event.reason_codes
    assert event.valid_observation_count == 3
    assert engine.finish(reason=reason, frame_index=3, timestamp=.3) == ()
    assert sum(engine.counters.values()) == 1
    with pytest.raises(RuntimeError, match="ended"):
        step(engine, 4, .82)


def test_timeout_applies_on_empty_frames_and_reappearance_cannot_double_count(engine, step):
    step(engine, 0, .1, clear=(False, False))
    step(engine, 1, .3, clear=(False, False))
    assert step(engine, 2, timestamp=.6) == ()  # Exactly the configured .5 seconds is tolerated.
    event, = step(engine, 3, timestamp=.61)
    assert event.outcome == Outcome.UNINSPECTED and "TRACKING_TIMEOUT" in event.reason_codes
    assert "CAP_NOT_OBSERVABLE" in event.reason_codes
    assert step(engine, 4, .82, timestamp=.7) == ()
    assert engine.finish(reason=EndReason.STOP, frame_index=4, timestamp=.7) == ()


def test_track_id_is_scoped_to_run(config, step):
    first = InspectionEngine(run_id="first", config=config, mode=RunMode.SIMULATION)
    second = InspectionEngine(run_id="second", config=config, mode=RunMode.SIMULATION)
    for index, progress in enumerate(PATH[:-1]):
        step(first, index, progress, cap=False)
        step(second, index, progress)
    fail, = step(first, 6, .82, cap=False)
    passed, = step(second, 6, .82)
    assert fail.track_id == passed.track_id == "1"
    assert fail.outcome == Outcome.FAIL and passed.outcome == Outcome.PASS
    assert fail.event_id != passed.event_id
    assert first.counters["FAIL"] == second.counters["PASS"] == 1


def test_new_id_mid_region_is_not_stitched_into_a_good_product(engine, step):
    for index, progress in enumerate(PATH[:3]):
        step(engine, index, progress, track_id="old")
    for index, progress in enumerate((.5, .6, .7), start=3):
        assert step(engine, index, progress, track_id="new") == ()
    event, = step(engine, 6, .82, track_id="new")
    assert event.track_id == "new" and event.outcome == Outcome.UNINSPECTED
    assert "ENTRY_NOT_OBSERVED" in event.reason_codes
    old, = step(engine, 8, timestamp=.8)
    assert old.track_id == "old" and old.outcome == Outcome.UNINSPECTED
    assert "TRACKING_TIMEOUT" in old.reason_codes
    assert engine.counters == {"PASS": 0, "FAIL": 0, "UNINSPECTED": 2}


def test_many_frames_in_too_little_time_are_not_enough(engine, step):
    for index, progress in enumerate(PATH[:-1]):
        step(engine, index, progress, timestamp=index / 1000)
    event, = step(engine, 6, .82, timestamp=.006)
    assert event.valid_observation_count == 5
    assert event.outcome == Outcome.UNINSPECTED and "INSUFFICIENT_DURATION" in event.reason_codes


def test_too_few_clear_frames_are_uninspected(engine, step):
    step(engine, 0, .1)
    step(engine, 1, .4)
    event, = step(engine, 2, .82)
    assert event.valid_observation_count == 1
    assert event.outcome == Outcome.UNINSPECTED and "INSUFFICIENT_OBSERVATIONS" in event.reason_codes


def test_wrong_direction_cannot_pass(engine, step):
    events = []
    for index, progress in enumerate(reversed(PATH)):
        events.extend(step(engine, index, progress))
    assert events == []
    event, = engine.finish(reason=EndReason.EOF, frame_index=6, timestamp=.6)
    assert event.outcome == Outcome.UNINSPECTED


def test_roi_scales_with_resolution(engine, step):
    for index, progress in enumerate(PATH[:-1]):
        step(engine, index, progress, size=2000)
    event, = step(engine, 6, .82, size=2000)
    assert event.outcome == Outcome.PASS


def test_exact_minimum_valid_duration_can_pass(engine, step):
    for index, progress in enumerate((.1, .3, .5, .7)):
        step(engine, index, progress)
    event, = step(engine, 4, .82)
    assert event.outcome == Outcome.PASS
    assert event.valid_observation_count == 3
    assert event.observation_duration == pytest.approx(.2)


def test_identity_uncertain_on_exit_frame_still_blocks_pass(engine, step):
    for index, progress in enumerate(PATH[:-1]):
        step(engine, index, progress)
    event, = step(engine, 6, .82, identity_uncertain=True)
    assert event.outcome == Outcome.UNINSPECTED and "IDENTITY_UNCERTAIN" in event.reason_codes


def test_real_missing_component_is_uninspected_even_with_clear_region_assertion(config, detections):
    engine = InspectionEngine(run_id="real-missing", config=config, mode=RunMode.REAL)
    events = []
    for index, progress in enumerate(PATH):
        observations = associate_detections(
            detections(index, progress, cap=False),
            run_id=engine.run_id,
            config=config,
            observability={"1": (True, True)},
        )
        events.extend(engine.process_frame(
            frame_index=index,
            timestamp=index / 10,
            frame_size=(1000, 1000),
            observations=observations,
        ))

    event, = events
    assert event.outcome == Outcome.UNINSPECTED
    assert event.valid_observation_count == 0
    assert "CAP_MISSING" not in event.reason_codes
    assert "CAP_NOT_OBSERVABLE" in event.reason_codes


def test_low_confidence_bottle_cannot_supply_valid_observations(engine, step):
    for index, progress in enumerate(PATH[:-1]):
        step(engine, index, progress, bottle_confidence=.1)
    event, = step(engine, 6, .82)
    assert event.valid_observation_count == 0
    assert event.outcome == Outcome.UNINSPECTED and "LOW_BOTTLE_CONFIDENCE" in event.reason_codes


def test_tracks_that_never_enter_roi_are_ignored(engine, step):
    for index, progress in enumerate(PATH):
        assert step(engine, index, progress, y=.95, half_height=.02) == ()
    assert engine.statuses == {}
    assert engine.finish(reason=EndReason.EOF, frame_index=6, timestamp=.6) == ()
