from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from bottle_qc.domain.association import associate_detections
from bottle_qc.domain.entities import Box, Detection, Outcome
from bottle_qc.pipeline.contracts import Frame
from bottle_qc.vision.model_manifest import ModelManifest
from bottle_qc.vision.yolo_provider import YoloProvider


def test_component_with_clear_owner_is_used_for_only_one_bottle(config, detections):
    first = detections(0, .4, track_id="a", label=False)
    second = detections(0, .46, track_id="b", cap=False, label=False)
    observations = associate_detections(first + second, run_id="run-1", config=config)
    assert {item.track_id: item.cap_present for item in observations} == {"a": True, "b": False}
    assert all(not item.ambiguous for item in observations)


def test_component_ambiguous_between_bottles_is_uninspected_for_both(engine, detections):
    # The shared cap is equidistant from both bottles in every frame.
    events = []
    for index, progress in enumerate((.1, .3, .4, .5, .6, .7, .82)):
        first = detections(index, progress, track_id="a", cap=False, label=False)
        second = detections(index, progress + .04, track_id="b", cap=False, label=False)
        shared_cap = Detection("cap", Box((progress + .005) * 1000, 190, (progress + .035) * 1000, 230), .95, index, index / 10)
        observations = associate_detections(first + second + [shared_cap], run_id=engine.run_id, config=engine.config,
                                            observability={"a": (True, True), "b": (True, True)})
        assert all(item.ambiguous and not item.cap_present for item in observations)
        events.extend(engine.process_frame(frame_index=index, timestamp=index / 10, frame_size=(1000, 1000), observations=observations))
    assert len(events) == 2
    assert all(event.outcome == Outcome.UNINSPECTED for event in events)
    assert all("AMBIGUOUS_COMPONENT_ASSOCIATION" in event.reason_codes for event in events)
    assert engine.counters == {"PASS": 0, "FAIL": 0, "UNINSPECTED": 2}


def test_detection_order_does_not_change_track_identity(config, detections):
    values = detections(0, .3, track_id="left") + detections(0, .7, track_id="right")
    first = associate_detections(values, run_id="run-1", config=config)
    second = associate_detections(list(reversed(values)), run_id="run-1", config=config)
    assert {item.track_id: (item.cap_present, item.label_present) for item in first} == {
        item.track_id: (item.cap_present, item.label_present) for item in second
    } == {"left": (True, True), "right": (True, True)}


def test_oversized_component_with_only_center_in_zone_is_rejected(config, detections):
    values = detections(0, .5, cap=False)
    values.append(Detection("cap", Box(0, 0, 1000, 500), .99, 0, 0))
    observation, = associate_detections(values, run_id="run-1", config=config)
    assert not observation.cap_present


def test_low_confidence_cap_is_not_support(config, detections):
    values = detections(0, .5, cap_confidence=.1)
    observation, = associate_detections(values, run_id="run-1", config=config)
    assert not observation.cap_present and observation.label_present


def test_detection_does_not_imply_observability(config, detections):
    observation, = associate_detections(detections(0, .5), run_id="run-1", config=config)
    assert observation.cap_present and observation.label_present
    assert not observation.cap_observable and not observation.label_observable


@pytest.mark.parametrize("mutation", ["missing_identity", "duplicate_identity", "mixed_timestamp", "mixed_frame"])
def test_invalid_association_inputs_are_rejected(config, detections, mutation):
    values = detections(0, .5)
    if mutation == "missing_identity":
        values[0] = replace(values[0], track_id=None)
    elif mutation == "duplicate_identity":
        values.append(values[0])
    elif mutation == "mixed_timestamp":
        values[1] = replace(values[1], timestamp=.1)
    else:
        values[1] = replace(values[1], frame_index=1)
    with pytest.raises(ValueError):
        associate_detections(values, run_id="run-1", config=config)


def _provider(
    monkeypatch,
    batches,
    *,
    tracker_iou_threshold=0.2,
    tracker_ambiguity_margin=0.025,
    maximum_missed_frames=2,
    maximum_tracks=4096,
    patch_region_quality=True,
    minimum_region_edge_variance=0,
):
    class ScriptedModel:
        task = "detect"
        names = {0: "bottle", 1: "cap", 2: "label"}
        device = "controlled"

        def __init__(self, outputs):
            self.outputs = iter(outputs)

        def track(self, image, **kwargs):
            return [next(self.outputs)]

    model = ScriptedModel(batches)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda _: model))
    manifest = ModelManifest(
        model_id="controlled",
        manifest_path=Path("controlled.json"),
        weights_path=Path("controlled.pt"),
        weights_sha256="0" * 64,
        task="detect",
        class_mapping={0: "bottle", 1: "cap", 2: "label"},
        framework_version="controlled",
        raw={},
    )
    provider = YoloProvider(
        manifest,
        tracker=True,
        minimum_confidence=0.2,
        minimum_region_edge_variance=minimum_region_edge_variance,
        tracker_iou_threshold=tracker_iou_threshold,
        tracker_ambiguity_margin=tracker_ambiguity_margin,
        maximum_missed_frames=maximum_missed_frames,
        maximum_tracks=maximum_tracks,
    )
    if patch_region_quality:
        monkeypatch.setattr(provider, "_region_quality_flags", lambda image, box: (True, True))
    return provider


def _result(*bottles):
    return SimpleNamespace(
        names={0: "bottle", 1: "cap", 2: "label"},
        boxes=SimpleNamespace(
            xyxy=[box for _, box in bottles],
            cls=[0 for _ in bottles],
            conf=[0.99 for _ in bottles],
            id=[track_id for track_id, _ in bottles],
        ),
    )


def _frame(index):
    return Frame(index, index / 10, Image.new("RGB", (160, 120)), "controlled")


def _bottle(track_id, x1, y1=10, x2=None, y2=90):
    return track_id, (x1, y1, x1 + 40 if x2 is None else x2, y2)


def test_post_tracker_uncertainty_leaves_separated_stable_tracks_certain(monkeypatch):
    provider = _provider(monkeypatch, [
        _result(_bottle(1, 10), _bottle(2, 100)),
        _result(_bottle(1, 12), _bottle(2, 102)),
    ])
    assert provider.detect(_frame(0)).uncertain_track_ids == frozenset()
    assert provider.detect(_frame(1)).uncertain_track_ids == frozenset()


def test_post_tracker_overlap_marks_only_involved_ids_and_stays_sticky(monkeypatch):
    provider = _provider(monkeypatch, [
        _result(_bottle(1, 10), _bottle(2, 70), _bottle(3, 115, y1=95, y2=115)),
        _result(_bottle(1, 40), _bottle(2, 50), _bottle(3, 117, y1=95, y2=115)),
        _result(_bottle(1, 10), _bottle(2, 100), _bottle(3, 119, y1=95, y2=115)),
    ])
    assert provider.detect(_frame(0)).uncertain_track_ids == frozenset()
    overlap = provider.detect(_frame(1))
    assert overlap.uncertain_track_ids == frozenset({"1", "2"})
    separated = provider.detect(_frame(2))
    assert separated.uncertain_track_ids == frozenset({"1", "2"})


def test_crossing_correspondences_mark_only_the_two_crossing_ids(monkeypatch):
    provider = _provider(monkeypatch, [
        _result(_bottle(1, 10), _bottle(2, 100), _bottle(3, 55)),
        _result(_bottle(1, 100), _bottle(2, 10), _bottle(3, 57)),
    ])
    provider.detect(_frame(0))
    crossing = provider.detect(_frame(1))
    assert crossing.uncertain_track_ids == frozenset({"1", "2"})


def test_close_competing_correspondence_marks_both_previous_ids(monkeypatch):
    provider = _provider(monkeypatch, [
        _result(_bottle(1, 10), _bottle(2, 55)),
        _result(_bottle(1, 32, x2=72)),
    ])
    provider.detect(_frame(0))
    batch = provider.detect(_frame(1))
    assert batch.uncertain_track_ids == frozenset({"1", "2"})


def test_tracker_iou_threshold_and_ambiguity_margin_change_correspondence_decision(monkeypatch):
    strict = _provider(monkeypatch, [
        _result(_bottle(1, 10)),
        _result(_bottle(1, 32, x2=72)),
    ], tracker_iou_threshold=0.3)
    strict.detect(_frame(0))
    assert strict.detect(_frame(1)).uncertain_track_ids == frozenset({"1"})

    wide_margin = _provider(monkeypatch, [
        _result(_bottle(1, 10), _bottle(2, 55)),
        _result(_bottle(1, 33, x2=73)),
    ], tracker_ambiguity_margin=0.025)
    narrow_margin = _provider(monkeypatch, [
        _result(_bottle(1, 10), _bottle(2, 55)),
        _result(_bottle(1, 33, x2=73)),
    ], tracker_ambiguity_margin=0.01)
    wide_margin.detect(_frame(0))
    narrow_margin.detect(_frame(0))
    assert wide_margin.detect(_frame(1)).uncertain_track_ids == frozenset({"1", "2"})
    assert narrow_margin.detect(_frame(1)).uncertain_track_ids == frozenset()


def test_missed_frame_boundary_and_reappearance_keep_uncertainty_sticky(monkeypatch):
    tolerated = _provider(monkeypatch, [
        _result(_bottle(1, 10)), _result(), _result(), _result(_bottle(1, 10)),
    ])
    assert tolerated.detect(_frame(0)).uncertain_track_ids == frozenset()
    assert tolerated.detect(_frame(1)).uncertain_track_ids == frozenset()
    assert tolerated.detect(_frame(2)).uncertain_track_ids == frozenset()
    assert tolerated.detect(_frame(3)).uncertain_track_ids == frozenset()

    exceeded = _provider(monkeypatch, [
        _result(_bottle(1, 10)), _result(), _result(), _result(), _result(_bottle(1, 10)),
    ])
    exceeded.detect(_frame(0))
    exceeded.detect(_frame(1))
    exceeded.detect(_frame(2))
    third_missing = exceeded.detect(_frame(3))
    assert third_missing.uncertain_track_ids == frozenset({"1"})
    assert exceeded.detect(_frame(4)).uncertain_track_ids == frozenset({"1"})


def test_new_provider_instance_starts_with_certain_identity_state(monkeypatch):
    first = _provider(monkeypatch, [
        _result(_bottle(1, 10), _bottle(2, 55)),
        _result(_bottle(1, 32, x2=72)),
    ])
    first.detect(_frame(0))
    assert first.detect(_frame(1)).uncertain_track_ids == frozenset({"1", "2"})

    fresh = _provider(monkeypatch, [_result(_bottle(1, 32, x2=72))])
    assert fresh.detect(_frame(0)).uncertain_track_ids == frozenset()


def test_tracker_state_is_bounded_by_configured_track_limit(monkeypatch):
    provider = _provider(monkeypatch, [_result(_bottle(1, 10)), _result(_bottle(2, 100))], maximum_tracks=1)
    provider.detect(_frame(0))
    with pytest.raises(RuntimeError, match="maximum_tracks=1"):
        provider.detect(_frame(1))


def test_real_video_bottle_without_tracker_id_fails_actionably(monkeypatch):
    missing = _result(_bottle(None, 10))
    provider = _provider(monkeypatch, [missing])
    with pytest.raises(RuntimeError, match="without a tracker ID.*source frame 0.*verify"):
        provider.detect(_frame(0))


def test_laplacian_quality_is_measured_separately_for_cap_and_label_regions(monkeypatch):
    from PIL import ImageDraw

    provider = _provider(
        monkeypatch,
        [_result(_bottle(1, 40, y1=10, x2=120, y2=110))],
        patch_region_quality=False,
        minimum_region_edge_variance=12,
    )
    image = Image.new("RGB", (160, 120), (127, 127, 127))
    draw = ImageDraw.Draw(image)
    for x in range(45, 116, 8):
        color = (0, 0, 0) if x % 16 else (255, 255, 255)
        draw.rectangle((x, 8, x + 3, 25), fill=color)
    batch = provider.detect(Frame(0, 0.0, image, "quality"))

    assert batch.observability["1"] == (True, False)


def test_provider_uncertainty_reaches_real_domain_outcome(monkeypatch, config):
    from bottle_qc.domain.entities import EndReason, RunMode
    from bottle_qc.domain.inspection import InspectionEngine

    progress_path = (.1, .3, .4, .5, .6, .7, .82)
    results = []
    for progress in progress_path:
        center = progress * 160
        crossing = 0.6 * 160
        results.append(_result(
            _bottle(1, center - 6, y1=20, x2=center + 6, y2=100),
            _bottle(2, crossing - 6, y1=20, x2=crossing + 6, y2=100),
        ))
    provider = _provider(monkeypatch, results)
    run_config = replace(config, frame_aspect_ratio=160 / 120)
    engine = InspectionEngine(run_id="provider-run", config=run_config, mode=RunMode.REAL)
    events = []
    for index in range(len(results)):
        batch = provider.detect(_frame(index))
        observations = associate_detections(
            batch.detections,
            run_id=engine.run_id,
            config=run_config,
            observability=batch.observability,
            uncertain_track_ids=batch.uncertain_track_ids,
        )
        events.extend(engine.process_frame(
            frame_index=index,
            timestamp=index / 10,
            frame_size=(160, 120),
            observations=observations,
        ))
    events.extend(engine.finish(reason=EndReason.EOF, frame_index=6, timestamp=.6))

    event = next(item for item in events if item.track_id == "1")
    assert event.outcome == Outcome.UNINSPECTED
    assert "IDENTITY_UNCERTAIN" in event.reason_codes


def test_lost_provider_identity_marks_pending_domain_track_uncertain(monkeypatch, config):
    from bottle_qc.domain.entities import EndReason, RunMode
    from bottle_qc.domain.inspection import InspectionEngine

    provider = _provider(monkeypatch, [
        _result(_bottle(1, 10, y1=20, x2=22, y2=100)),
        _result(), _result(), _result(),
    ])
    run_config = replace(config, frame_aspect_ratio=160 / 120)
    engine = InspectionEngine(run_id="lost-provider-run", config=run_config, mode=RunMode.REAL)
    for index in range(4):
        batch = provider.detect(_frame(index))
        observations = associate_detections(
            batch.detections,
            run_id=engine.run_id,
            config=run_config,
            observability=batch.observability,
            uncertain_track_ids=batch.uncertain_track_ids,
        )
        engine.mark_identity_uncertain(batch.uncertain_track_ids)
        engine.process_frame(
            frame_index=index,
            timestamp=index / 10,
            frame_size=(160, 120),
            observations=observations,
        )

    event, = engine.finish(reason=EndReason.EOF, frame_index=3, timestamp=.3)
    assert event.outcome == Outcome.UNINSPECTED
    assert "IDENTITY_UNCERTAIN" in event.reason_codes
