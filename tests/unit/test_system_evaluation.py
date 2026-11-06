import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


MODEL_HASH = "a" * 64
CONFIG_HASH = "b" * 64
EVENT_FIELDS = (
    "event_id", "source_timestamp_ms", "source_index", "outcome", "track_id",
    "run_id", "video_id", "mode", "model_hash", "config_hash",
)


def _run(*arguments: str, run_id: str = "run-a") -> subprocess.CompletedProcess:
    project = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-B", "-m", "bottle_qc.evaluation.system", "--run-id", run_id, *arguments],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )


def _product(product_id, start_ms, end_ms, expected_outcome, *, run_id="run-a", video_id="line-a", reviewed_track_ids=None):
    row = {
        "run_id": run_id,
        "video_id": video_id,
        "product_id": product_id,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "expected_outcome": expected_outcome,
    }
    if reviewed_track_ids is not None:
        row["reviewed_track_ids"] = reviewed_track_ids
    return row


def _event(event_id, timestamp_ms, outcome="PASS", *, track_id="track-1", source_index=None, **changes):
    row = {
        "event_id": event_id,
        "source_timestamp_ms": timestamp_ms,
        "source_index": "" if source_index is None else source_index,
        "outcome": outcome,
        "track_id": track_id,
        "run_id": "run-a",
        "video_id": "line-a",
        "mode": "REAL",
        "model_hash": MODEL_HASH,
        "config_hash": CONFIG_HASH,
    }
    row.update(changes)
    return row


def _evaluate_case(tmp_path, truth_rows, event_rows, *, run_id="run-a", source_id="line-a", output=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    ground_truth = tmp_path / "ground-truth.jsonl"
    ground_truth.write_text("".join(json.dumps(row) + "\n" for row in truth_rows), encoding="utf-8")
    events = tmp_path / "events.csv"
    with events.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        writer.writerows(event_rows)
    arguments = [
        "--ground-truth", str(ground_truth),
        "--events", str(events),
        "--source-id", source_id,
    ]
    output_path = tmp_path / "evaluation.json"
    if output:
        arguments.extend(("--output", str(output_path)))
    result = _run(*arguments, run_id=run_id)
    return result, json.loads(result.stdout), ground_truth, events, output_path


def test_system_evaluation_reports_product_errors_misses_duplicates_unmatched_and_provenance(tmp_path):
    truth_rows = [
        _product("bad-1", 0, 100, "FAIL"),
        _product("good-1", 200, 300, "PASS"),
        _product("missed-1", 400, 500, "FAIL"),
        _product("unclear-1", 600, 700, "FAIL"),
    ]
    events = [
        _event("e1", 50, "PASS", track_id="1"),
        _event("e2", 250, "FAIL", track_id="2"),
        _event("e3", 650, "UNINSPECTED", track_id="3", source_index=1),
        _event("e4", 660, "FAIL", track_id="4", source_index=2),
        _event("e5", 900, "PASS", track_id="5"),
    ]

    result, report, ground_truth, _, output_path = _evaluate_case(tmp_path, truth_rows, events, output=True)

    assert result.returncode == 0
    assert report["status"] == "EVALUATED"
    assert report["counts"] == {
        "ground_truth_products": 4,
        "false_pass": 1,
        "false_fail": 1,
        "uninspected": 1,
        "missed_products": 1,
        "duplicate_events": 1,
        "unmatched_events": 1,
    }
    assert report["products"][3]["event_ids"] == ["e3", "e4"]
    assert report["provenance"]["model_hash"] == MODEL_HASH
    assert report["provenance"]["config_hash"] == CONFIG_HASH
    assert report["provenance"]["ground_truth_sha256"] == hashlib.sha256(ground_truth.read_bytes()).hexdigest()
    assert report["evidence_scope"] == "controlled evaluation-contract proof; not real-model validation"
    assert output_path.is_file()


def test_overlapping_good_bad_products_without_reviewed_mappings_are_ambiguous(tmp_path):
    truth_rows = [
        _product("bad", 0, 1000, "FAIL"),
        _product("good", 400, 800, "PASS"),
    ]

    result, report, *_ = _evaluate_case(tmp_path, truth_rows, [_event("overlap", 500, "PASS", track_id="good")])

    assert result.returncode == 1
    assert report["status"] == "INVALID"
    assert "ambiguous" in report["error"].lower()


def test_unique_interval_correspondence_is_limited_to_the_requested_run_and_source(tmp_path):
    truth_rows = [
        _product("current", 0, 100, "PASS"),
        _product("other-run", 0, 100, "FAIL", run_id="run-b"),
        _product("other-source", 0, 100, "FAIL", video_id="line-b"),
    ]

    result, report, *_ = _evaluate_case(tmp_path, truth_rows, [_event("current-event", 50)])

    assert result.returncode == 0
    assert report["status"] == "EVALUATED"
    assert report["counts"]["ground_truth_products"] == 1
    assert report["products"] == [{
        "run_id": "run-a",
        "video_id": "line-a",
        "product_id": "current",
        "expected_outcome": "PASS",
        "predicted_outcome": "PASS",
        "event_ids": ["current-event"],
        "duplicate_count": 0,
    }]


def test_explicit_track_mappings_match_both_products_in_good_bad_overlap(tmp_path):
    truth_rows = [
        _product("bad", 0, 1000, "FAIL", reviewed_track_ids=["track-bad"]),
        _product("good", 400, 800, "PASS", reviewed_track_ids=["track-good"]),
    ]
    events = [
        _event("bad-event", 500, "FAIL", track_id="track-bad"),
        _event("good-event", 500, "PASS", track_id="track-good"),
    ]

    result, report, *_ = _evaluate_case(tmp_path, truth_rows, events)

    assert result.returncode == 0
    assert [product["event_ids"] for product in report["products"]] == [["bad-event"], ["good-event"]]
    assert report["counts"] == {
        "ground_truth_products": 2,
        "false_pass": 0,
        "false_fail": 0,
        "uninspected": 0,
        "missed_products": 0,
        "duplicate_events": 0,
        "unmatched_events": 0,
    }


def test_conflicting_reviewed_track_mappings_are_rejected(tmp_path):
    truth_rows = [
        _product("first", 0, 100, "PASS", reviewed_track_ids=["track-1"]),
        _product("second", 200, 300, "FAIL", reviewed_track_ids=["track-1"]),
    ]

    result, report, *_ = _evaluate_case(tmp_path, truth_rows, [_event("mapped", 50, track_id="track-1")])

    assert result.returncode == 1
    assert report["status"] == "INVALID"
    assert "conflicting" in report["error"].lower()


def test_reviewed_mapping_contradicting_unique_interval_membership_is_rejected(tmp_path):
    truth_rows = [
        _product("timestamp-product", 0, 100, "PASS"),
        _product("mapped-product", 200, 300, "FAIL", reviewed_track_ids=["track-1"]),
    ]

    result, report, *_ = _evaluate_case(tmp_path, truth_rows, [_event("contradiction", 50, track_id="track-1")])

    assert result.returncode == 1
    assert report["status"] == "INVALID"
    assert "contradict" in report["error"].lower()


def test_reviewed_mapping_with_no_source_time_candidate_is_rejected(tmp_path):
    truth_rows = [_product("mapped-product", 0, 100, "PASS", reviewed_track_ids=["track-1"])]

    result, report, *_ = _evaluate_case(tmp_path, truth_rows, [_event("outside", 500, track_id="track-1")])

    assert result.returncode == 1
    assert report["status"] == "INVALID"
    assert "contradict" in report["error"].lower()


def test_singular_reviewed_track_mapping_shape_is_rejected(tmp_path):
    truth = _product("product", 0, 100, "PASS")
    truth["reviewed_track_id"] = "track-1"

    result, report, *_ = _evaluate_case(tmp_path, [truth], [_event("event", 50, track_id="track-1")])

    assert result.returncode == 1
    assert report["status"] == "INVALID"
    assert "reviewed_track_ids" in report["error"]


def test_reused_track_mapping_ids_are_isolated_by_run_and_source(tmp_path):
    truth_rows = [
        _product("run-a-line-a", 0, 1000, "PASS", reviewed_track_ids=["track-1"]),
        _product("run-a-line-a-other", 400, 800, "FAIL", reviewed_track_ids=["other-a"]),
        _product("run-b-line-a", 1000, 2000, "FAIL", run_id="run-b", reviewed_track_ids=["track-1"]),
        _product("run-b-line-a-other", 1400, 1800, "PASS", run_id="run-b", reviewed_track_ids=["other-b"]),
        _product("run-a-line-b", 2000, 3000, "PASS", video_id="line-b", reviewed_track_ids=["track-1"]),
        _product("run-a-line-b-other", 2400, 2800, "FAIL", video_id="line-b", reviewed_track_ids=["other-c"]),
    ]

    run_a_line_a, report_a, *_ = _evaluate_case(
        tmp_path / "run-a-line-a", truth_rows, [_event("event-a", 500, track_id="track-1")],
    )
    run_b_line_a, report_b, *_ = _evaluate_case(
        tmp_path / "run-b-line-a",
        truth_rows,
        [_event("event-b", 1500, track_id="track-1", run_id="run-b")],
        run_id="run-b",
    )
    run_a_line_b, report_c, *_ = _evaluate_case(
        tmp_path / "run-a-line-b",
        truth_rows,
        [_event("event-c", 2500, track_id="track-1", video_id="line-b")],
        source_id="line-b",
    )

    assert [run_a_line_a.returncode, run_b_line_a.returncode, run_a_line_b.returncode] == [0, 0, 0]
    assert [report_a["products"][0]["product_id"], report_b["products"][0]["product_id"], report_c["products"][0]["product_id"]] == [
        "run-a-line-a", "run-b-line-a", "run-a-line-b",
    ]
    assert [report_a["products"][0]["event_ids"], report_b["products"][0]["event_ids"], report_c["products"][0]["event_ids"]] == [
        ["event-a"], ["event-b"], ["event-c"],
    ]


def test_duplicate_track_ids_within_one_reviewed_list_are_rejected(tmp_path):
    truth_rows = [_product("product", 0, 100, "PASS", reviewed_track_ids=["track-1", "track-1"])]

    result, report, *_ = _evaluate_case(tmp_path, truth_rows, [_event("event", 50, track_id="track-1")])

    assert result.returncode == 1
    assert report["status"] == "INVALID"
    assert "duplicate" in report["error"].lower()


def test_overlapping_product_duplicates_count_with_multiple_reviewed_track_ids(tmp_path):
    truth_rows = [
        _product("bad", 0, 1000, "FAIL", reviewed_track_ids=["bad-track-1", "bad-track-2"]),
        _product("good", 400, 800, "PASS", reviewed_track_ids=["good-track"]),
    ]
    events = [
        _event("bad-primary", 700, "FAIL", track_id="bad-track-1", source_index=1),
        _event("bad-duplicate", 701, "FAIL", track_id="bad-track-2", source_index=2),
        _event("good-primary", 450, "PASS", track_id="good-track", source_index=0),
    ]

    result, report, *_ = _evaluate_case(tmp_path, truth_rows, events)

    assert result.returncode == 0
    assert [product["event_ids"] for product in report["products"]] == [
        ["bad-primary", "bad-duplicate"], ["good-primary"],
    ]
    assert report["counts"] == {
        "ground_truth_products": 2,
        "false_pass": 0,
        "false_fail": 0,
        "uninspected": 0,
        "missed_products": 0,
        "duplicate_events": 1,
        "unmatched_events": 0,
    }


def test_primary_event_uses_source_clock_then_source_index_not_outcome_or_csv_order(tmp_path):
    truth_rows = [_product("product", 0, 100, "FAIL")]
    events = [
        _event("m-later-clock", 20, "FAIL", track_id="later-clock", source_index=1),
        _event("a-same-clock-later-index", 10, "FAIL", track_id="later-index", source_index=100),
        _event("z-earliest-clock", 10, "PASS", track_id="earliest-clock", source_index=99),
    ]

    result, report, *_ = _evaluate_case(tmp_path, truth_rows, events)

    assert result.returncode == 0
    assert report["products"][0]["predicted_outcome"] == "PASS"
    assert report["products"][0]["event_ids"] == ["z-earliest-clock", "a-same-clock-later-index", "m-later-clock"]
    assert report["counts"]["duplicate_events"] == 2
    assert report["counts"]["false_pass"] == 1


def test_source_interval_boundaries_are_inclusive(tmp_path):
    truth_rows = [_product("boundary-product", 10, 20, "PASS")]
    events = [
        _event("end", 20, "PASS", track_id="at-end"),
        _event("start", 10, "FAIL", track_id="at-start"),
    ]

    result, report, *_ = _evaluate_case(tmp_path, truth_rows, events)

    assert result.returncode == 0
    assert report["products"][0]["event_ids"] == ["start", "end"]
    assert report["products"][0]["predicted_outcome"] == "FAIL"
    assert report["counts"]["duplicate_events"] == 1
    assert report["counts"]["unmatched_events"] == 0


def test_event_scope_conflicting_with_export_invocation_is_rejected(tmp_path):
    result, report, *_ = _evaluate_case(
        tmp_path,
        [_product("product", 0, 100, "PASS")],
        [_event("wrong-run", 50, run_id="run-b")],
    )

    assert result.returncode == 1
    assert report["status"] == "INVALID"
    assert "run" in report["error"].lower()


@pytest.mark.parametrize("mutation", [
    "missing_mode",
    "simulation_mode",
    "mixed_mode",
    "missing_model_hash",
    "malformed_model_hash",
    "mixed_model_hash",
    "missing_config_hash",
    "malformed_config_hash",
    "mixed_config_hash",
])
def test_incomplete_mixed_or_malformed_real_provenance_is_rejected(tmp_path, mutation):
    events = [_event("first", 20), _event("second", 30)]
    if mutation == "missing_mode":
        events[0]["mode"] = ""
    elif mutation == "simulation_mode":
        events[0]["mode"] = "SIMULATION"
    elif mutation == "mixed_mode":
        events[1]["mode"] = "SIMULATION"
    elif mutation == "missing_model_hash":
        events[0]["model_hash"] = ""
    elif mutation == "malformed_model_hash":
        events[0]["model_hash"] = "m"
    elif mutation == "mixed_model_hash":
        events[1]["model_hash"] = "c" * 64
    elif mutation == "missing_config_hash":
        events[0]["config_hash"] = ""
    elif mutation == "malformed_config_hash":
        events[0]["config_hash"] = "not-a-sha256"
    elif mutation == "mixed_config_hash":
        events[1]["config_hash"] = "d" * 64

    result, report, *_ = _evaluate_case(tmp_path, [_product("product", 0, 100, "PASS")], events)

    assert result.returncode == 1
    assert report["status"] == "INVALID"
    assert "provenance" in report["error"].lower() or "simulation" in report["error"].lower()
