"""Evaluate saved real run events against reviewed per-product ground truth."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


OUTCOMES = {"PASS", "FAIL", "UNINSPECTED"}
SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required_text(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} must be a nonempty string")
    return value.strip()


def _source_time(value: Any, description: str) -> float:
    try:
        timestamp = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{description} must be a finite source-clock time in milliseconds") from exc
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError(f"{description} must be a finite nonnegative source-clock time in milliseconds")
    return timestamp


def _source_index(value: Any, event_id: str) -> int | None:
    if value in (None, ""):
        return None
    try:
        index = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Event {event_id} has an invalid source_index") from exc
    if index < 0 or str(index) != str(value).strip():
        raise ValueError(f"Event {event_id} has an invalid source_index")
    return index


def evaluate(
    ground_truth_path: Path,
    events_path: Path,
    *,
    run_id: str,
    source_id: str,
) -> dict[str, Any]:
    ground_truth_path, events_path = ground_truth_path.resolve(), events_path.resolve()
    if not ground_truth_path.is_file() or not events_path.is_file():
        missing = [str(path) for path in (ground_truth_path, events_path) if not path.is_file()]
        return {
            "status": "WAITING_FOR_INPUT",
            "missing": missing,
            "next_step": "Provide reviewed product ground truth and a BottleQC run CSV export; no metrics were calculated.",
        }

    run_id = _required_text(run_id, "run_id")
    source_id = _required_text(source_id, "source_id")

    truth: list[dict[str, Any]] = []
    product_keys: set[tuple[str, str, str]] = set()
    reviewed_tracks: dict[tuple[str, str, str], tuple[str, str, str]] = {}
    with ground_truth_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Ground truth line {line_number} must be a JSON object")
            required = ("run_id", "video_id", "product_id", "start_ms", "end_ms", "expected_outcome")
            if any(key not in row for key in required):
                raise ValueError(f"Ground truth line {line_number} needs {', '.join(required)}")

            row_run = _required_text(row["run_id"], f"Ground truth line {line_number} run_id")
            row_source = _required_text(row["video_id"], f"Ground truth line {line_number} video_id")
            product_id = _required_text(row["product_id"], f"Ground truth line {line_number} product_id")
            expected = str(row["expected_outcome"]).strip().upper()
            start = _source_time(row["start_ms"], f"Ground truth line {line_number} start_ms")
            end = _source_time(row["end_ms"], f"Ground truth line {line_number} end_ms")
            if expected not in OUTCOMES or end < start:
                raise ValueError(f"Ground truth line {line_number} has an invalid interval or outcome")

            product_key = (row_run, row_source, product_id)
            if product_key in product_keys:
                raise ValueError("Ground-truth product_id must be unique within each run_id and video_id")
            product_keys.add(product_key)
            product = {
                "run_id": row_run,
                "video_id": row_source,
                "product_id": product_id,
                "start_ms": start,
                "end_ms": end,
                "expected_outcome": expected,
            }

            if "reviewed_track_id" in row:
                raise ValueError(f"Ground truth line {line_number} must use reviewed_track_ids, not reviewed_track_id")
            if "reviewed_track_ids" in row:
                raw_track_ids = row["reviewed_track_ids"]
                if not isinstance(raw_track_ids, list):
                    raise ValueError(f"Ground truth line {line_number} reviewed_track_ids must be a list")
                seen_track_ids: set[str] = set()
                for raw_track_id in raw_track_ids:
                    track_id = _required_text(raw_track_id, f"Ground truth line {line_number} reviewed_track_ids item")
                    if track_id in seen_track_ids:
                        raise ValueError(f"Ground truth line {line_number} has a duplicate reviewed track id: {track_id}")
                    seen_track_ids.add(track_id)
                    mapping_key = (row_run, row_source, track_id)
                    previous_product = reviewed_tracks.get(mapping_key)
                    if previous_product is not None and previous_product != product_key:
                        raise ValueError(f"Conflicting reviewed track mappings for {track_id} in the same run/source")
                    if previous_product is not None:
                        raise ValueError(f"Duplicate reviewed track mapping for {track_id} in the same run/source")
                    reviewed_tracks[mapping_key] = product_key
            truth.append(product)

    if not truth:
        return {
            "status": "WAITING_FOR_INPUT",
            "missing": ["per-product ground truth records"],
            "next_step": "Add reviewed ground-truth product records; no metrics were calculated.",
        }

    with events_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"event_id", "outcome", "source_timestamp_ms", "mode", "model_hash", "config_hash"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Events CSV must contain event_id, outcome, source_timestamp_ms, mode, model_hash, and config_hash columns")
        events = list(reader)

    if not events:
        raise ValueError("Events CSV has no event rows to establish REAL model/config provenance")

    event_ids: set[str] = set()
    model_hashes: set[str] = set()
    config_hashes: set[str] = set()
    for row in events:
        event_id = _required_text(row.get("event_id"), "Event event_id")
        if event_id in event_ids:
            raise ValueError(f"Event id {event_id} is duplicated")
        event_ids.add(event_id)

        mode = str(row.get("mode") or "").strip().upper()
        if mode != "REAL":
            raise ValueError(f"Event {event_id} has missing, mixed, or non-REAL provenance mode")

        model_hash = str(row.get("model_hash") or "").strip()
        if not SHA256_RE.fullmatch(model_hash):
            raise ValueError(f"Event {event_id} has missing or malformed model_hash provenance")
        model_hashes.add(model_hash.lower())

        config_hash = str(row.get("config_hash") or "").strip()
        if not SHA256_RE.fullmatch(config_hash):
            raise ValueError(f"Event {event_id} has missing or malformed config_hash provenance")
        config_hashes.add(config_hash.lower())

        row_run = str(row.get("run_id") or "").strip()
        if row_run and row_run != run_id:
            raise ValueError(f"Event {event_id} run_id conflicts with the requested run scope")
        row_source = str(row.get("video_id") or "").strip()
        if row_source and row_source != source_id:
            raise ValueError(f"Event {event_id} video_id conflicts with the requested source scope")

        outcome = str(row.get("outcome") or "").strip().upper()
        if outcome not in OUTCOMES:
            raise ValueError(f"Event {event_id} has an invalid outcome")
        row["event_id"] = event_id
        row["outcome"] = outcome
        row["timestamp"] = _source_time(row.get("source_timestamp_ms"), f"Event {event_id} source_timestamp_ms")
        row["source_index_value"] = _source_index(row.get("source_index"), event_id)
        row["track_id_value"] = str(row.get("track_id") or "").strip()

    if len(model_hashes) != 1:
        raise ValueError("Event rows have mixed model_hash provenance")
    if len(config_hashes) != 1:
        raise ValueError("Event rows have mixed config_hash provenance")

    scoped_truth = [
        product for product in truth
        if product["run_id"] == run_id and product["video_id"] == source_id
    ]
    if not scoped_truth:
        return {
            "status": "WAITING_FOR_INPUT",
            "missing": [f"ground-truth products for run_id={run_id}, video_id={source_id}"],
            "next_step": "Add reviewed product records for the requested run and source; no metrics were calculated.",
        }

    products_by_key = {
        (product["run_id"], product["video_id"], product["product_id"]): product
        for product in scoped_truth
    }
    assigned: dict[tuple[str, str, str], list[dict[str, Any]]] = {
        key: [] for key in products_by_key
    }
    unmatched = 0
    for event in events:
        timestamp = event["timestamp"]
        candidates = [
            product for product in scoped_truth
            if product["start_ms"] <= timestamp <= product["end_ms"]
        ]
        track_id = event["track_id_value"]
        mapped_key = reviewed_tracks.get((run_id, source_id, track_id)) if track_id else None

        if mapped_key is not None:
            candidate_keys = {
                (candidate["run_id"], candidate["video_id"], candidate["product_id"])
                for candidate in candidates
            }
            if not candidates or mapped_key not in candidate_keys:
                raise ValueError(f"Reviewed track mapping for event {event['event_id']} contradicts inclusive source-time interval membership")
            target_key = mapped_key
        elif len(candidates) == 1:
            candidate = candidates[0]
            target_key = (candidate["run_id"], candidate["video_id"], candidate["product_id"])
        elif len(candidates) > 1:
            raise ValueError(f"Event {event['event_id']} is ambiguous between overlapping ground-truth products; reviewed track mapping is required")
        else:
            unmatched += 1
            continue
        assigned[target_key].append(event)

    counts = {
        "ground_truth_products": len(scoped_truth),
        "false_pass": 0,
        "false_fail": 0,
        "uninspected": 0,
        "missed_products": 0,
        "duplicate_events": 0,
        "unmatched_events": unmatched,
    }
    product_results = []
    for product in scoped_truth:
        product_key = (product["run_id"], product["video_id"], product["product_id"])
        matched = sorted(
            assigned[product_key],
            key=lambda event: (
                event["timestamp"],
                event["source_index_value"] if event["source_index_value"] is not None else math.inf,
                event["event_id"],
            ),
        )
        duplicate_count = max(0, len(matched) - 1)
        counts["duplicate_events"] += duplicate_count
        expected = product["expected_outcome"]
        predicted = matched[0]["outcome"] if matched else None
        if not matched:
            counts["missed_products"] += 1
        elif predicted == "UNINSPECTED":
            counts["uninspected"] += 1
        elif expected == "FAIL" and predicted == "PASS":
            counts["false_pass"] += 1
        elif expected == "PASS" and predicted == "FAIL":
            counts["false_fail"] += 1
        product_results.append({
            "run_id": product["run_id"],
            "video_id": product["video_id"],
            "product_id": product["product_id"],
            "expected_outcome": expected,
            "predicted_outcome": predicted,
            "event_ids": [event["event_id"] for event in matched],
            "duplicate_count": duplicate_count,
        })

    return {
        "status": "EVALUATED",
        "evidence_scope": "controlled evaluation-contract proof; not real-model validation",
        "matching": (
            "Correspondence uses source_timestamp_ms in milliseconds from the exported source/media clock. "
            "Intervals are inclusive at both start_ms and end_ms and are considered only within the requested run_id/video_id. "
            "Exactly one interval member maps by time; overlapping membership requires a reviewed_track_ids mapping scoped to that run/source."
        ),
        "primary_event_order": "Ascending source_timestamp_ms, then source_index (missing indices last), then event_id; outcome is not considered.",
        "counts": counts,
        "rates": {
            "false_pass_per_ground_truth_fail": counts["false_pass"] / max(1, sum(product["expected_outcome"] == "FAIL" for product in scoped_truth)),
            "false_fail_per_ground_truth_pass": counts["false_fail"] / max(1, sum(product["expected_outcome"] == "PASS" for product in scoped_truth)),
            "uninspected_per_ground_truth_product": counts["uninspected"] / len(scoped_truth),
            "missed_per_ground_truth_product": counts["missed_products"] / len(scoped_truth),
        },
        "provenance": {
            "run_id": run_id,
            "source_id": source_id,
            "ground_truth_path": str(ground_truth_path),
            "ground_truth_sha256": _sha256(ground_truth_path),
            "events_path": str(events_path),
            "events_sha256": _sha256(events_path),
            "model_hash": next(iter(model_hashes)),
            "config_hash": next(iter(config_hashes)),
        },
        "products": product_results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True, help="reviewed per-product interval JSONL, scoped by run_id and video_id")
    parser.add_argument("--events", type=Path, required=True, help="REAL single-run CSV exported by the API")
    parser.add_argument("--run-id", required=True, help="run ID that scopes this single-run CSV export")
    parser.add_argument("--source-id", required=True, help="video_id/source identity for this single-run CSV export")
    parser.add_argument("--output", type=Path, help="new report path; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        report = evaluate(
            args.ground_truth,
            args.events,
            run_id=args.run_id,
            source_id=args.source_id,
        )
        status = report["status"]
        if args.output and status == "EVALUATED":
            if args.output.exists():
                raise FileExistsError(f"Refusing to overwrite evaluation report: {args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        code = 2 if status == "WAITING_FOR_INPUT" else 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report, code = {"status": "INVALID", "error": str(exc)}, 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
