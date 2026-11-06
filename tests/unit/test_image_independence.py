from bottle_qc.domain.association import associate_detections
from bottle_qc.domain.config import InspectionConfig
from bottle_qc.domain.entities import Box, Detection, RunMode
from bottle_qc.domain.image_inspection import inspect_image


def _observations(run_id, *, cap=True, label=True, cap_visible=True):
    detections = [
        Detection("bottle", Box(30, 10, 70, 90), 0.99, 0, 0.0, "1"),
        Detection("label", Box(42, 48, 58, 68), 0.95, 0, 0.0),
    ]
    if cap:
        detections.append(Detection("cap", Box(43, 13, 57, 22), 0.95, 0, 0.0))
    return associate_detections(
        detections,
        run_id=run_id,
        config=InspectionConfig(roi=Box(0.05, 0.05, 0.95, 0.95), frame_aspect_ratio=1.0),
        observability={"1": (cap_visible, True)},
    )


def test_still_images_have_independent_per_bottle_outcomes():
    config = InspectionConfig(roi=Box(0.05, 0.05, 0.95, 0.95), frame_aspect_ratio=1.0)
    complete = inspect_image(
        _observations("image-run-a"), run_id="image-run-a", source_item_id="asset-a",
        config=config, mode=RunMode.REAL, config_id="default", config_hash="config-a",
    )
    missing = inspect_image(
        _observations("image-run-b", cap=False), run_id="image-run-b", source_item_id="asset-b",
        config=config, mode=RunMode.REAL, config_id="default", config_hash="config-b",
    )
    obscured = inspect_image(
        _observations("image-run-c", cap=False, cap_visible=False), run_id="image-run-c",
        source_item_id="asset-c", config=config, mode=RunMode.REAL,
        config_id="default", config_hash="config-c",
    )

    assert [event.outcome.value for event in (complete[0], missing[0], obscured[0])] == ["PASS", "FAIL", "UNINSPECTED"]
    assert complete[0].run_id != missing[0].run_id
    assert complete[0].event_id != missing[0].event_id
    assert missing[0].source_item_id != obscured[0].source_item_id


def test_one_image_emits_a_separate_event_for_each_bottle():
    config = InspectionConfig(roi=Box(0.05, 0.05, 0.95, 0.95), frame_aspect_ratio=2.0)
    detections = []
    for track_id, center_x in (("left", 50), ("right", 150)):
        detections.extend((
            Detection("bottle", Box(center_x - 20, 10, center_x + 20, 90), 0.99, 0, 0.0, track_id),
            Detection("cap", Box(center_x - 7, 8, center_x + 7, 20), 0.95, 0, 0.0),
            Detection("label", Box(center_x - 9, 48, center_x + 9, 68), 0.95, 0, 0.0),
        ))
    observations = associate_detections(
        detections,
        run_id="one-image-run",
        config=config,
        observability={"left": (True, True), "right": (True, True)},
    )
    events = inspect_image(
        observations,
        run_id="one-image-run",
        source_item_id="one-image-asset",
        config=config,
        mode=RunMode.REAL,
        config_id="default",
        config_hash="config-hash",
    )

    assert len(events) == 2
    assert {event.track_id for event in events} == {"left", "right"}
    assert {event.outcome.value for event in events} == {"PASS"}
    assert len({event.event_id for event in events}) == 2
