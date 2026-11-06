import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image


def _run_data_tool(root: Path, *arguments: str) -> subprocess.CompletedProcess:
    project = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-B", "-m", "bottle_qc.training.dataset", *arguments],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )


def test_empty_inventory_has_actionable_waiting_state(tmp_path):
    (tmp_path / "raw").mkdir()
    result = _run_data_tool(
        tmp_path,
        "inventory",
        "--root", str(tmp_path / "raw"),
        "--annotations", str(tmp_path / "annotations"),
    )

    assert result.returncode == 2
    report = json.loads(result.stdout)
    assert report["status"] == "WAITING_FOR_INPUT"
    assert report["images"] == 0
    assert "authorized images" in report["next_step"].lower()


def test_validation_reports_image_pair_box_class_duplicate_and_split_errors(tmp_path):
    root = tmp_path / "raw"
    annotations = tmp_path / "annotations"
    root.mkdir()
    annotations.mkdir()
    Image.new("RGB", (20, 20), "white").save(root / "a.png")
    (root / "b.png").write_bytes((root / "a.png").read_bytes())
    (root / "corrupt.png").write_bytes(b"not an image")
    (annotations / "a.txt").write_text(
        "3 0.5 0.5 0.2 0.2\n"
        "0 nan 0.5 0.2 0.2\n"
        "0 0.5 0.5 inf 0.2\n"
        "x 0.5 0.5 0.2 0.2\n"
        "0 0.5 0.5 0.2\n",
        encoding="utf-8",
    )
    (annotations / "b.txt").write_text("0 0.95 0.5 0.2 0.2\n", encoding="utf-8")
    (annotations / "corrupt.txt").write_text("", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"image": "a.png", "capture_session": "shared", "split": "train"},
        {"image": "b.png", "capture_session": "shared", "split": "validation"},
        {"image": "corrupt.png", "capture_session": "third", "split": "test"},
        {"image": "missing.png", "capture_session": "fourth", "split": "test"},
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = _run_data_tool(
        tmp_path,
        "validate",
        "--root", str(root),
        "--annotations", str(annotations),
        "--manifest", str(manifest),
    )

    assert result.returncode == 1
    assert result.stdout.strip(), result.stderr
    report = json.loads(result.stdout)
    issue_codes = {item["code"] for item in report["issues"]}
    assert {
        "class_invalid", "box_out_of_bounds", "image_corrupt", "image_missing",
        "annotation_missing", "annotation_value", "annotation_columns", "duplicate_image", "split_leakage",
    } <= issue_codes


def test_validation_reports_unhashable_split_value_as_invalid_input(tmp_path):
    root = tmp_path / "raw"
    annotations = tmp_path / "annotations"
    root.mkdir()
    annotations.mkdir()
    Image.new("RGB", (20, 20), "white").save(root / "sample.png")
    (annotations / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({
        "image": "sample.png",
        "capture_session": "session-one",
        "split": [],
    }) + "\n", encoding="utf-8")

    result = _run_data_tool(
        tmp_path,
        "validate",
        "--root", str(root),
        "--annotations", str(annotations),
        "--manifest", str(manifest),
    )

    assert result.returncode == 1
    assert result.stdout.strip(), result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "INVALID"
    assert "split_invalid" in {item["code"] for item in report["issues"]}


def test_group_split_keeps_related_source_sessions_together(tmp_path):
    root = tmp_path / "raw"
    annotations = tmp_path / "annotations"
    root.mkdir()
    annotations.mkdir()
    rows = []
    for index in range(6):
        name = f"image-{index}.png"
        Image.new("RGB", (24, 24), (index * 30, 80, 120)).save(root / name)
        (annotations / f"image-{index}.txt").write_text("0 0.5 0.5 0.4 0.8\n", encoding="utf-8")
        group = index // 2
        rows.append({
            "image": name,
            "source_video": f"video-{group}",
            "capture_session": f"session-{group}",
        })
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    split_manifest = tmp_path / "split.jsonl"
    result = _run_data_tool(
        tmp_path,
        "split",
        "--root", str(root),
        "--annotations", str(annotations),
        "--manifest", str(manifest),
        "--output", str(split_manifest),
        "--seed", "17",
    )

    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert report["status"] == "SPLIT_CREATED"
    split_rows = [json.loads(line) for line in split_manifest.read_text(encoding="utf-8").splitlines()]
    assert {row["split"] for row in split_rows} == {"train", "validation", "test"}
    for field in ("source_video", "capture_session"):
        grouped = {}
        for row in split_rows:
            grouped.setdefault(row[field], set()).add(row["split"])
        assert all(len(splits) == 1 for splits in grouped.values())
    check = _run_data_tool(
        tmp_path,
        "validate",
        "--root", str(root),
        "--annotations", str(annotations),
        "--manifest", str(split_manifest),
    )
    assert check.returncode == 0
    assert json.loads(check.stdout)["status"] == "VALID"
