"""Train only with explicit, locally available, validated inputs; Colab runs are deferred."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import tomllib
import uuid
from typing import Any

from PIL import Image

from bottle_qc.training.dataset import _validate_labels


CLASSES = {0: "bottle", 1: "cap", 2: "label"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_data(path: Path) -> tuple[dict[str, Any] | None, str | None]:
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
    if not isinstance(names, dict):
        return None, "INVALID: Dataset names must map class IDs to bottle, cap, label"
    try:
        normalized_names = {int(key): str(value).casefold() for key, value in names.items()}
    except (TypeError, ValueError):
        return None, "INVALID: Dataset class IDs must be numeric IDs 0, 1, and 2"
    if normalized_names != CLASSES:
        return None, "INVALID: Dataset classes must be exactly bottle, cap, label in class IDs 0, 1, 2"
    root_value = data.get("path", str(path.parent))
    if not isinstance(root_value, str) or not root_value.strip():
        return None, "INVALID: Dataset path must be a nonempty local directory"
    root = Path(root_value).resolve()
    split_paths: dict[str, set[Path]] = {}
    split_hashes: dict[str, set[str]] = {}
    for split in ("train", "val"):
        value = data.get(split)
        if not isinstance(value, str) or not value.strip():
            return None, f"Dataset must define {split} image paths"
        image_root = (root / value).resolve()
        if not image_root.is_relative_to(root):
            return None, f"INVALID: Dataset {split} image path escapes its dataset root"
        if not image_root.is_dir():
            return None, f"Dataset {split} images are missing: {image_root}"
        images = sorted(item for item in image_root.rglob("*") if item.is_file() and item.suffix.casefold() in IMAGE_EXTENSIONS)
        if not images:
            return None, f"Dataset {split} has no labelled images: {image_root}"
        image_directory = next((parent for parent in image_root.parents if parent.name.casefold() == "images"), None)
        if image_directory is None:
            return None, f"INVALID: Dataset {split} images must be paired under an images directory"
        labels_root = (image_directory.parent / "labels" / image_root.relative_to(image_directory)).resolve()
        if not labels_root.is_relative_to(root) or not labels_root.is_dir():
            return None, f"INVALID: Dataset {split} labels directory is missing or outside its dataset root"
        expected_labels = {item.relative_to(image_root).with_suffix(".txt") for item in images}
        actual_labels = {
            item.relative_to(labels_root)
            for item in labels_root.rglob("*")
            if item.is_file() and item.suffix.casefold() == ".txt"
        }
        if expected_labels != actual_labels:
            return None, f"INVALID: Dataset {split} image and label files do not form exact pairs"
        paths, hashes = split_paths.setdefault(split, set()), split_hashes.setdefault(split, set())
        for image_path in images:
            resolved_image = image_path.resolve()
            if not resolved_image.is_relative_to(image_root):
                return None, f"INVALID: Dataset {split} image escapes its split directory"
            label_path = (labels_root / image_path.relative_to(image_root)).with_suffix(".txt")
            if not label_path.is_file() or not label_path.resolve().is_relative_to(labels_root):
                return None, f"INVALID: Dataset label is missing or outside its split directory for {image_path}"
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:
                return None, f"Dataset image is invalid: {image_path}: {exc}"
            digest = _hash(image_path)
            for other_split, other_paths in split_paths.items():
                if other_split != split and (resolved_image in other_paths or digest in split_hashes[other_split]):
                    return None, "INVALID: Dataset train and val splits share an image"
            paths.add(resolved_image)
            hashes.add(digest)
            label_issues: list[str] = []
            _validate_labels(
                label_path,
                str(image_path.relative_to(image_root)),
                {name: 0 for name in CLASSES.values()},
                lambda code, location, message: label_issues.append(f"{code} at {location}: {message}"),
            )
            if label_issues:
                return None, f"INVALID: Dataset label is invalid: {label_issues[0]}"
    return data, None


def _write_model_manifest(*, model_id: str, output: Path, data_yaml: Path, run_id: str, config: dict[str, Any], metrics: dict[str, Any]) -> Path:
    weights = output / "weights" / "best.pt"
    if not weights.is_file():
        raise RuntimeError(f"Training completed without an exported best.pt: {weights}")
    names = config["names"]
    if isinstance(names, list):
        names = {index: value for index, value in enumerate(names)}
    mapping = {str(int(key)): str(value).casefold() for key, value in names.items()}
    if {int(key): value for key, value in mapping.items()} != CLASSES:
        raise RuntimeError("Trained model class mapping differs from the BottleQC contract")
    validation_path = output / "validation.json"
    validation_path.write_text(json.dumps({
        "split": "val",
        "weights_sha256": _hash(weights),
        "dataset_config_sha256": _hash(data_yaml),
        "metrics": metrics,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "model_id": model_id,
        "weights_path": "weights/best.pt",
        "weights_sha256": _hash(weights),
        "task": "detect",
        "class_mapping": mapping,
        "framework_version": _ultralytics_version(),
        "dataset_manifest_ref": f"{data_yaml.resolve()}#sha256={_hash(data_yaml)}",
        "training_run_ref": run_id,
        "validation_report_ref": str(validation_path.resolve()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_config": config,
        "validation_split": "val",
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _ultralytics_version() -> str:
    from importlib.metadata import version

    return version("ultralytics")


def resolve_training_options(config: dict[str, Any], args) -> tuple[dict[str, Any] | None, str | None]:
    if config.get("task", "detect") != "detect":
        return None, "BottleQC training configuration task must be detect"
    options = {
        "model": config.get("model", "yolo11n"),
        "model_id": config.get("model_id", "bottleqc-yolo11n"),
        "seed": config.get("seed", 42),
        "imgsz": config.get("imgsz", 640),
        "epochs": config.get("epochs", 100),
        "batch": config.get("batch", 16),
        "device": config.get("device", "cpu"),
        "workers": config.get("workers", 0),
        "resume": config.get("resume", False),
    }
    for name in ("model_id", "seed", "imgsz", "epochs", "batch", "device", "workers"):
        override = getattr(args, name, None)
        if override is not None:
            options[name] = override
    if getattr(args, "resume", None) is True:
        options["resume"] = True
    if not isinstance(options["model"], str) or not options["model"].strip():
        return None, "Training configuration model must be a nonempty name"
    if not isinstance(options["model_id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", options["model_id"]):
        return None, "model_id must use letters, digits, dot, underscore or dash."
    for name in ("seed", "imgsz", "epochs", "batch", "workers"):
        if type(options[name]) is not int:
            return None, f"Training option {name} must be an integer"
    if options["imgsz"] < 32 or options["epochs"] < 1 or options["batch"] < 1 or options["workers"] < 0:
        return None, "imgsz must be at least 32; epochs and batch must be positive; workers cannot be negative"
    if not isinstance(options["device"], str) or not options["device"].strip():
        return None, "Training option device must be a nonempty string"
    if type(options["resume"]) is not bool:
        return None, "Training option resume must be true or false"
    return options, None


def run_training(args) -> tuple[dict[str, Any], int]:
    config_path = Path(args.config)
    if not config_path.is_file():
        return {"status": "WAITING_FOR_INPUT", "error": f"Training configuration is missing: {config_path}", "next_step": "Provide the checked-in training config and a validated dataset."}, 2
    try:
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return {"status": "INVALID", "error": f"Training configuration cannot be read: {exc}"}, 2
    options, option_issue = resolve_training_options(config, args)
    if option_issue:
        return {"status": "INVALID", "error": option_issue}, 2
    data_yaml, weights, output = Path(args.data), Path(args.weights), Path(args.output).resolve()
    try:
        data, issue = _load_data(data_yaml)
    except (OSError, ValueError) as exc:
        return {"status": "INVALID", "error": f"Dataset validation failed: {type(exc).__name__}: {exc}"}, 2
    if issue:
        if issue.startswith("INVALID:"):
            return {"status": "INVALID", "error": issue.removeprefix("INVALID:").strip()}, 2
        return {"status": "WAITING_FOR_INPUT", "error": issue, "next_step": "Supply a validated dataset with grouped train/val splits and explicit local weights."}, 2
    if not weights.is_file():
        return {"status": "WAITING_FOR_INPUT", "error": f"Local initialization weights are missing: {weights}", "next_step": "Supply an authorized local training checkpoint; this command does not download or substitute weights."}, 2
    if options["resume"]:
        last_weights = output / "weights" / "last.pt"
        if not last_weights.is_file():
            return {"status": "INVALID", "error": f"Resume requested but checkpoint is missing: {last_weights}"}, 2
        weights = last_weights
    elif output.exists():
        return {"status": "INVALID", "error": f"Refusing to overwrite existing training output: {output}"}, 2
    try:
        from ultralytics import YOLO
    except ImportError:
        return {"status": "WAITING_FOR_INPUT", "error": "Install the optional real-model training dependencies first.", "next_step": "Install the real extra in the intended training environment; Colab execution remains deferred."}, 2
    run_id = str(uuid.uuid4())
    try:
        model = YOLO(str(weights))
        model.train(
            data=str(data_yaml.resolve()),
            project=str(output.parent),
            name=output.name,
            exist_ok=options["resume"],
            resume=options["resume"],
            seed=options["seed"],
            imgsz=options["imgsz"],
            epochs=options["epochs"],
            batch=options["batch"],
            device=options["device"],
            workers=options["workers"],
        )
        best_weights = output / "weights" / "best.pt"
        trained = YOLO(str(best_weights), task="detect")
        validation = trained.val(data=str(data_yaml.resolve()), split="val", device=options["device"], workers=options["workers"])
        metrics = {str(key): float(value) for key, value in getattr(validation, "results_dict", {}).items()}
        manifest_path = _write_model_manifest(
            model_id=options["model_id"],
            output=output,
            data_yaml=data_yaml,
            run_id=run_id,
            config={**config, **options, "names": getattr(trained, "names", {}),
                    "initial_weights_sha256": _hash(weights)},
            metrics=metrics,
        )
        return {
            "status": "TRAINED_AND_VALIDATED",
            "model_id": options["model_id"],
            "training_run_ref": run_id,
            "best_weights": str(best_weights),
            "weights_sha256": _hash(best_weights),
            "manifest": str(manifest_path),
            "validation_report": str(output / "validation.json"),
            "metrics": metrics,
        }, 0
    except Exception as exc:
        return {"status": "FAILED", "training_run_ref": run_id, "output": str(output), "error": f"{type(exc).__name__}: {exc}"}, 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="train a detector from explicit local data and weights")
    train.add_argument("--data", type=Path, required=True, help="validated YOLO data.yaml")
    train.add_argument("--weights", type=Path, required=True, help="existing local initialization checkpoint")
    train.add_argument("--output", type=Path, required=True, help="new run directory; use --resume only for an existing last.pt")
    train.add_argument("--model-id")
    train.add_argument("--config", type=Path, default=Path("configs/train.toml"))
    train.add_argument("--seed", type=int)
    train.add_argument("--imgsz", type=int)
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch", type=int)
    train.add_argument("--device", help="explicit Ultralytics device, e.g. cpu or 0")
    train.add_argument("--workers", type=int, help="Windows-safe default is 0")
    train.add_argument("--resume", action="store_true", default=None)
    args = parser.parse_args(argv)
    report, code = run_training(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
