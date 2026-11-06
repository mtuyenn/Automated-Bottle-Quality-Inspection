"""Evaluate a validated local BottleQC detector on an explicitly named dataset split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


CLASSES = {0: "bottle", 1: "cap", 2: "label"}


def _read_dataset(path: Path, split: str) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, f"Dataset configuration is missing: {path}"
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"Dataset configuration cannot be read: {exc}"
    if not isinstance(data, dict):
        return None, "Dataset configuration must be a YAML object"
    names = data.get("names")
    if isinstance(names, list):
        names = {index: value for index, value in enumerate(names)}
    try:
        mapping = {int(key): str(value).casefold() for key, value in names.items()}
    except (AttributeError, TypeError, ValueError):
        return None, "Dataset config needs a bottle/cap/label class mapping"
    if mapping != CLASSES:
        return None, "Dataset classes must be exactly IDs 0=bottle, 1=cap, 2=label"
    key = {"train": "train", "val": "val", "test": "test"}[split]
    value = data.get(key)
    if not isinstance(value, str):
        return None, f"Dataset config does not define the requested {split} split"
    root = Path(data.get("path", path.parent)).resolve()
    split_dir = (root / value).resolve()
    if not split_dir.is_dir():
        return None, f"Requested {split} images are missing: {split_dir}"
    return data, None


def _numbers(value) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(value)]


def evaluate_detector(*, manifest_path: Path, models_root: Path, data_path: Path, split: str, device: str, imgsz: int, batch: int, workers: int) -> dict[str, Any]:
    from bottle_qc.settings import Settings
    from bottle_qc.vision.model_manifest import load_manifest, sha256_file
    data, issue = _read_dataset(data_path, split)
    if issue:
        return {"status": "WAITING_FOR_INPUT", "error": issue, "next_step": "Provide a structurally validated labelled dataset; no detector metrics were generated."}
    if not manifest_path.is_file():
        return {"status": "WAITING_FOR_INPUT", "error": f"Model manifest is missing: {manifest_path}", "next_step": "Register trained bottle/cap/label weights with provenance before evaluation."}
    manifest = load_manifest(manifest_path, models_root=models_root)
    from bottle_qc.vision.yolo_provider import open_yolo_provider

    values, _, _ = Settings.from_environment().load_config("default")
    open_yolo_provider(
        manifest_path,
        models_root=models_root,
        tracker=False,
        minimum_confidence=float(values["minimum_confidence"]),
        minimum_region_edge_variance=float(values["minimum_region_edge_variance"]),
    )
    try:
        from ultralytics import YOLO
    except ImportError:
        return {"status": "WAITING_FOR_INPUT", "error": "Optional real-model dependencies are not installed.", "next_step": "Install the real extra in the intended evaluation environment."}
    model = YOLO(str(manifest.weights_path))
    metrics = model.val(data=str(data_path.resolve()), split=split, device=device, imgsz=imgsz, batch=batch, workers=workers)
    box = getattr(metrics, "box", None)
    if box is None:
        raise ValueError("Validation returned no detection box metrics")
    class_ids = [int(value) for value in _numbers(getattr(box, "ap_class_index", []))]
    precision = _numbers(getattr(box, "p", []))
    recall = _numbers(getattr(box, "r", []))
    ap50 = _numbers(getattr(box, "ap50", []))
    ap = _numbers(getattr(box, "ap", []))
    class_metrics = {}
    for offset, class_id in enumerate(class_ids):
        class_name = CLASSES.get(class_id)
        if class_name is None:
            raise ValueError(f"Validation returned undeclared class id {class_id}")
        class_metrics[class_name] = {
            "precision": precision[offset] if offset < len(precision) else None,
            "recall": recall[offset] if offset < len(recall) else None,
            "map50": ap50[offset] if offset < len(ap50) else None,
            "map50_95": ap[offset] if offset < len(ap) else None,
        }
    for class_id, name in CLASSES.items():
        class_metrics.setdefault(name, {"precision": None, "recall": None, "map50": None, "map50_95": None, "status": "no_ground_truth_instances_in_split"})
    return {
        "status": "EVALUATED",
        "split": split,
        "device": device,
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "classes": class_metrics,
        "aggregate": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "map50": float(box.map50),
            "map50_95": float(box.map),
        },
        "provenance": {
            "manifest": str(manifest.manifest_path),
            "weights_sha256": manifest.weights_sha256,
            "dataset_config": str(data_path.resolve()),
            "dataset_config_sha256": sha256_file(data_path),
            "task": manifest.task,
            "framework_version": manifest.framework_version,
            "class_mapping": {str(key): value for key, value in sorted(manifest.class_mapping.items())},
        },
        "scope": "Detector metrics for the selected labelled split; not per-product system performance or pilot proof.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = evaluate_detector(
            manifest_path=args.manifest,
            models_root=args.models_root,
            data_path=args.data,
            split=args.split,
            device=args.device,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
        )
        status = report["status"]
        if args.output and status == "EVALUATED":
            if args.output.exists():
                raise FileExistsError(f"Refusing to overwrite evaluation report: {args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        code = 2 if status == "WAITING_FOR_INPUT" else 0
    except (OSError, ValueError, RuntimeError) as exc:
        report, code = {"status": "INVALID", "error": str(exc)}, 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
