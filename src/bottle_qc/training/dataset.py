"""Inventory, validate and group-split reviewed BottleQC image annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
from typing import Any

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_NAMES = ("bottle", "cap", "label")
GROUP_FIELDS = ("source_video", "capture_session", "batch_id")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contained_path(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError("image path must be relative to the dataset root")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError("image path escapes the dataset root")
    return resolved


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"manifest line {number} is not valid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"manifest line {number} must be a JSON object")
            row["_line"] = number
            rows.append(row)
    return rows


def inventory(root: Path, annotations: Path) -> dict[str, Any]:
    root, annotations = root.resolve(), annotations.resolve()
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS)
    seen_hashes: dict[str, str] = {}
    items, duplicate_hashes, corrupt = [], [], []
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest = _sha256(path)
        duplicate_of = seen_hashes.get(digest)
        if duplicate_of:
            duplicate_hashes.append({"path": relative, "duplicate_of": duplicate_of, "sha256": digest})
        else:
            seen_hashes[digest] = relative
        try:
            with Image.open(path) as image:
                image_size = list(image.size)
                image.verify()
        except Exception as exc:
            corrupt.append({"path": relative, "error": str(exc)})
            image_size = None
        label_path = (annotations / Path(relative)).with_suffix(".txt")
        items.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": digest,
            "dimensions": image_size,
            "annotation": label_path.is_file(),
        })
    missing_labels = sum(not item["annotation"] for item in items)
    status = "WAITING_FOR_INPUT" if not files else ("ACTION_REQUIRED" if corrupt or duplicate_hashes or missing_labels else "READY_TO_VALIDATE")
    return {
        "status": status,
        "images": len(files),
        "corrupt_images": len(corrupt),
        "duplicate_images": len(duplicate_hashes),
        "missing_annotations": missing_labels,
        "items": items,
        "duplicates": duplicate_hashes,
        "corrupt": corrupt,
        "next_step": "Add reviewed, authorized images and source/session metadata before annotation or training." if not files else "Review the inventory, then run validate against manifest.jsonl.",
    }


def validate_dataset(root: Path, annotations: Path, manifest: Path) -> dict[str, Any]:
    root, annotations, manifest = root.resolve(), annotations.resolve(), manifest.resolve()
    if not manifest.is_file():
        return {
            "status": "WAITING_FOR_INPUT",
            "images": 0,
            "issues": [{"code": "manifest_missing", "path": str(manifest), "message": "Create a JSONL manifest with image and source/session fields after authorized data is available."}],
            "next_step": "Add authorized images, reviewed YOLO labels, and a manifest.jsonl; no dataset was created.",
        }
    try:
        rows = _read_manifest(manifest)
    except (OSError, ValueError) as exc:
        return {"status": "INVALID", "images": 0, "issues": [{"code": "manifest_invalid", "path": str(manifest), "message": str(exc)}]}
    if not rows:
        return {
            "status": "WAITING_FOR_INPUT",
            "images": 0,
            "issues": [{"code": "manifest_empty", "path": str(manifest), "message": "No image records are present."}],
            "next_step": "Add one manifest record per reviewed image; no dataset was created.",
        }
    issues: list[dict[str, str]] = []
    hashes: dict[str, str] = {}
    component_splits: dict[str, set[str]] = {}
    class_counts = {name: 0 for name in CLASS_NAMES}

    def issue(code: str, path: str, message: str) -> None:
        issues.append({"code": code, "path": path, "message": message})

    for row in rows:
        line = str(row["_line"])
        relative = row.get("image")
        if not isinstance(relative, str) or not relative:
            issue("image_path_missing", f"manifest:{line}", "Each record needs a relative image path.")
            continue
        try:
            image_path = _contained_path(root, relative)
        except (OSError, ValueError) as exc:
            issue("unsafe_image_path", f"manifest:{line}", str(exc))
            continue
        if not image_path.is_file():
            issue("image_missing", relative, "Image file does not exist under the dataset root.")
            label_path = _contained_path(annotations, Path(relative).with_suffix(".txt").as_posix())
            if not label_path.is_file():
                issue("annotation_missing", relative, f"Expected YOLO annotation file: {label_path}")
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as exc:
            issue("image_corrupt", relative, f"Image cannot be decoded: {exc}")
        digest = _sha256(image_path)
        if digest in hashes:
            issue("duplicate_image", relative, f"Same image bytes as {hashes[digest]}.")
        else:
            hashes[digest] = relative
        label_path = _contained_path(annotations, Path(relative).with_suffix(".txt").as_posix())
        if not label_path.is_file():
            issue("annotation_missing", relative, f"Expected YOLO annotation file: {label_path}")
        else:
            _validate_labels(label_path, relative, class_counts, issue)
        split_name = row.get("split")
        split_is_valid = isinstance(split_name, str) and split_name in {"train", "validation", "test"}
        if split_name is not None and not split_is_valid:
            issue("split_invalid", relative, "split must be train, validation, or test.")
        for field in GROUP_FIELDS:
            value = row.get(field)
            if isinstance(value, str) and value.strip():
                component_splits.setdefault(f"{field}:{value.strip()}", set())
                if split_is_valid:
                    component_splits[f"{field}:{value.strip()}"].add(split_name)
        if not any(isinstance(row.get(field), str) and row[field].strip() for field in GROUP_FIELDS):
            issue("source_group_missing", relative, "Provide source_video, capture_session, or batch_id to prevent split leakage.")
    for group, splits in sorted(component_splits.items()):
        if len(splits) > 1:
            issue("split_leakage", group, f"Related records occur in multiple splits: {', '.join(sorted(splits))}.")
    return {
        "status": "INVALID" if issues else "VALID",
        "images": len(rows),
        "class_counts": class_counts,
        "issues": issues,
        "next_step": "Fix every reported issue, review annotations, then run split." if issues else "Dataset annotations and source groups pass structural checks; this is not model-quality proof.",
    }


def _validate_labels(label_path: Path, image_relative: str, class_counts: dict[str, int], issue) -> None:
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        issue("annotation_unreadable", image_relative, str(exc))
        return
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            issue("annotation_columns", f"{image_relative}:{line_number}", "Expected class_id x_center y_center width height.")
            continue
        try:
            class_id = int(values[0])
            box = [float(value) for value in values[1:]]
        except ValueError:
            issue("annotation_value", f"{image_relative}:{line_number}", "Class and box values must be numeric.")
            continue
        if not 0 <= class_id < len(CLASS_NAMES):
            issue("class_invalid", f"{image_relative}:{line_number}", f"Class id must be one of 0, 1, 2 ({', '.join(CLASS_NAMES)}).")
            continue
        x, y, width, height = box
        if not all(math.isfinite(value) for value in box) or width <= 0 or height <= 0 or x - width / 2 < 0 or y - height / 2 < 0 or x + width / 2 > 1 or y + height / 2 > 1:
            issue("box_out_of_bounds", f"{image_relative}:{line_number}", "Normalized box must have positive size and fit fully inside the image.")
            continue
        class_counts[CLASS_NAMES[class_id]] += 1


def split_manifest(root: Path, annotations: Path, manifest: Path, output: Path, seed: int) -> dict[str, Any]:
    validation = validate_dataset(root, annotations, manifest)
    if validation["status"] != "VALID":
        return {"status": validation["status"], "validation": validation, "next_step": validation.get("next_step")}
    rows = _read_manifest(manifest)
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    owner: dict[str, int] = {}
    for index, row in enumerate(rows):
        for field in GROUP_FIELDS:
            value = row.get(field)
            if isinstance(value, str) and value.strip():
                key = f"{field}:{value.strip()}"
                if key in owner:
                    union(index, owner[key])
                else:
                    owner[key] = index
    groups: dict[int, list[int]] = {}
    for index in range(len(rows)):
        groups.setdefault(find(index), []).append(index)
    components = list(groups.values())
    if len(components) < 3:
        return {"status": "INVALID", "issues": [{"code": "too_few_groups", "message": f"Need at least 3 independent source/session groups for train/validation/test; found {len(components)}."}]}
    if output.exists():
        return {"status": "INVALID", "issues": [{"code": "output_exists", "message": f"Refusing to overwrite {output}. Choose a new output path."}]}
    rng = random.Random(seed)
    rng.shuffle(components)
    group_count = len(components)
    train_count = max(1, math.floor(group_count * 0.70))
    validation_count = max(1, math.floor(group_count * 0.15))
    test_count = group_count - train_count - validation_count
    while test_count < 1:
        if train_count > validation_count and train_count > 1:
            train_count -= 1
        elif validation_count > 1:
            validation_count -= 1
        else:
            return {"status": "INVALID", "issues": [{"code": "too_few_groups", "message": "Could not allocate at least one independent group to each split."}]}
        test_count = group_count - train_count - validation_count
    assignments = ["train"] * train_count + ["validation"] * validation_count + ["test"] * test_count
    split_by_index = {}
    for component, split_name in zip(components, assignments):
        split_by_index.update((index, split_name) for index in component)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(rows):
            row.pop("_line", None)
            row["split"] = split_by_index[index]
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "status": "SPLIT_CREATED",
        "manifest": str(output.resolve()),
        "seed": seed,
        "source_groups": len(components),
        "images": len(rows),
        "groups_by_split": {name: assignments.count(name) for name in ("train", "validation", "test")},
        "next_step": "Run validate with this split manifest and annotations before training.",
    }


def prepare_yolo(root: Path, annotations: Path, manifest: Path, output: Path) -> dict[str, Any]:
    validation = validate_dataset(root, annotations, manifest)
    if validation["status"] != "VALID":
        return {"status": validation["status"], "validation": validation, "next_step": validation.get("next_step")}
    rows = _read_manifest(manifest)
    if any(row.get("split") not in {"train", "validation", "test"} for row in rows):
        return {"status": "INVALID", "issues": [{"code": "split_missing", "message": "Run split before prepare."}]}
    if output.exists():
        return {"status": "INVALID", "issues": [{"code": "output_exists", "message": f"Refusing to overwrite {output}. Choose a new output path."}]}
    for split_name in ("train", "validation", "test"):
        (output / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split_name).mkdir(parents=True, exist_ok=True)
    root, annotations = root.resolve(), annotations.resolve()
    for row in rows:
        relative = Path(row["image"])
        image_source = _contained_path(root, str(relative))
        label_source = _contained_path(annotations, relative.with_suffix(".txt").as_posix())
        image_destination = output / "images" / row["split"] / relative
        label_destination = output / "labels" / row["split"] / relative.with_suffix(".txt")
        image_destination.parent.mkdir(parents=True, exist_ok=True)
        label_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_source, image_destination)
        shutil.copy2(label_source, label_destination)
    yaml_lines = [f"path: {output.resolve().as_posix()}"]
    yaml_lines.extend(("train: images/train", "val: images/validation", "test: images/test"))
    yaml_lines.append(f"names: {json.dumps(list(CLASS_NAMES))}")
    (output / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return {"status": "PREPARED", "images": len(rows), "data_yaml": str((output / "data.yaml").resolve()), "classes": list(CLASS_NAMES)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("inventory", "validate", "split", "prepare"):
        command = subparsers.add_parser(name)
        command.add_argument("--root", type=Path, default=Path("data/raw"))
        command.add_argument("--annotations", type=Path, default=Path("data/annotations/labels"))
        command.add_argument("--manifest", type=Path)
        if name == "split":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--seed", type=int, default=42)
        elif name == "prepare":
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = args.manifest or (args.root / "manifest.jsonl")
    try:
        if args.command == "inventory":
            report = inventory(args.root, args.annotations)
            status = report["status"]
        elif args.command == "validate":
            report = validate_dataset(args.root, args.annotations, manifest)
            status = report["status"]
        elif args.command == "split":
            report = split_manifest(args.root, args.annotations, manifest, args.output, args.seed)
            status = report["status"]
        else:
            report = prepare_yolo(args.root, args.annotations, manifest, args.output)
            status = report["status"]
    except (OSError, ValueError) as exc:
        report, status = {"status": "INVALID_INPUT", "error": str(exc)}, "INVALID_INPUT"
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if status == "WAITING_FOR_INPUT" else (0 if status in {"READY_TO_VALIDATE", "VALID", "SPLIT_CREATED", "PREPARED"} else 1)


if __name__ == "__main__":
    raise SystemExit(main())
