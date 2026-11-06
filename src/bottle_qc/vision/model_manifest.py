from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
import os
from typing import Any


_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
REQUIRED_CLASSES = {0: "bottle", 1: "cap", 2: "label"}


class ModelUnavailableError(ValueError):
    """A model cannot be admitted for BottleQC inspection."""


@dataclass(frozen=True, slots=True)
class ModelManifest:
    model_id: str
    manifest_path: Path
    weights_path: Path
    weights_sha256: str
    task: str
    class_mapping: dict[int, str]
    framework_version: str
    raw: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path, *, models_root: Path | None = None) -> ModelManifest:
    path = Path(path).resolve()
    if not path.is_file():
        raise ModelUnavailableError(f"Model manifest is unavailable: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelUnavailableError(f"Model manifest cannot be read: {exc}") from exc
    if not isinstance(raw, dict):
        raise ModelUnavailableError("Model manifest must be a JSON object")
    model_id = raw.get("model_id")
    if not isinstance(model_id, str) or not _MODEL_ID.fullmatch(model_id):
        raise ModelUnavailableError("Model manifest has an invalid model_id")
    task = raw.get("task")
    if task not in {"detect", "detection"}:
        raise ModelUnavailableError("BottleQC inspection requires task='detect'")
    mapping = raw.get("class_mapping")
    try:
        class_mapping = {int(key): str(value).casefold() for key, value in mapping.items()}
    except (AttributeError, TypeError, ValueError) as exc:
        raise ModelUnavailableError("Model manifest class_mapping must map class IDs to names") from exc
    if class_mapping != REQUIRED_CLASSES:
        raise ModelUnavailableError("BottleQC inspection requires exactly class IDs 0=bottle, 1=cap, 2=label")
    digest = raw.get("weights_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise ModelUnavailableError("Model manifest needs a 64-character weights_sha256")
    relative_weights = raw.get("weights_path")
    if not isinstance(relative_weights, str) or not relative_weights:
        raise ModelUnavailableError("Model manifest needs weights_path")
    weights_path = Path(relative_weights)
    if weights_path.is_absolute():
        raise ModelUnavailableError("weights_path must be relative to its model manifest")
    weights_path = (path.parent / weights_path).resolve()
    containment_root = Path(models_root).resolve() if models_root else path.parent
    if not weights_path.is_relative_to(containment_root):
        raise ModelUnavailableError("weights_path escapes the configured models directory")
    if not weights_path.is_file():
        raise ModelUnavailableError(f"Model weights are unavailable: {weights_path}")
    actual_hash = sha256_file(weights_path)
    if actual_hash.casefold() != digest.casefold():
        raise ModelUnavailableError("Model weights SHA-256 does not match the manifest")
    framework_version = raw.get("framework_version")
    if not isinstance(framework_version, str) or not framework_version.strip():
        raise ModelUnavailableError("Model manifest needs framework_version provenance")
    for field in ("dataset_manifest_ref", "training_run_ref", "validation_report_ref", "created_at"):
        if not isinstance(raw.get(field), str) or not raw[field].strip():
            raise ModelUnavailableError(f"Model manifest needs {field} provenance")
    return ModelManifest(
        model_id=model_id,
        manifest_path=path,
        weights_path=weights_path,
        weights_sha256=actual_hash,
        task="detect",
        class_mapping=class_mapping,
        framework_version=framework_version,
        raw=raw,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate BottleQC model manifests and local inference capability.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="check manifest, weights hash, classes, task, and inference")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--models-root", type=Path, default=Path("models"))
    export = subparsers.add_parser("export", help="export a validated local model and write provenance")
    export.add_argument("--manifest", type=Path, required=True)
    export.add_argument("--models-root", type=Path, default=Path("models"))
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--format", choices=("onnx", "torchscript"), default="onnx")
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            report = _validate_and_probe(args.manifest, args.models_root)
        else:
            report = _export_model(args.manifest, args.models_root, args.output_dir, args.format)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (ModelUnavailableError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "UNAVAILABLE", "error": str(exc)}, indent=2, sort_keys=True))
        return 2


def _validate_and_probe(manifest_path: Path, models_root: Path) -> dict[str, Any]:
    from bottle_qc.settings import Settings
    from bottle_qc.vision.yolo_provider import open_yolo_provider

    manifest = load_manifest(manifest_path, models_root=models_root)
    values, _, _ = Settings.from_environment().load_config("default")
    provider = open_yolo_provider(
        manifest_path,
        models_root=models_root,
        tracker=False,
        minimum_confidence=float(values["minimum_confidence"]),
        minimum_region_edge_variance=float(values["minimum_region_edge_variance"]),
    )
    capability = provider.probe()
    return {
        "status": "AVAILABLE",
        "model_id": manifest.model_id,
        "weights_path": str(manifest.weights_path),
        "weights_sha256": manifest.weights_sha256,
        "task": manifest.task,
        "class_mapping": {str(key): value for key, value in sorted(manifest.class_mapping.items())},
        "framework_version": manifest.framework_version,
        "inference_capability": capability,
    }


def _export_model(manifest_path: Path, models_root: Path, output_dir: Path, format_name: str) -> dict[str, Any]:
    source = load_manifest(manifest_path, models_root=models_root)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise ModelUnavailableError(f"Refusing to overwrite existing export directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    source_copy = output_dir / source.weights_path.name
    shutil.copy2(source.weights_path, source_copy)
    if source_copy.suffix.casefold() == (".onnx" if format_name == "onnx" else ".torchscript"):
        raise ModelUnavailableError("Choose an export format different from the input weights format")
    try:
        os.environ["YOLO_AUTOINSTALL"] = "false"
        from ultralytics import YOLO

        model = YOLO(str(source_copy))
        exported = Path(model.export(format=format_name)).resolve()
    except Exception as exc:
        raise ModelUnavailableError(f"Local export failed without installing missing dependencies: {exc}") from exc
    if not exported.is_file() or not exported.is_relative_to(output_dir):
        raise ModelUnavailableError("Export did not produce a single model file inside the requested output directory")
    exported_manifest = {
        **source.raw,
        "model_id": f"{source.model_id}-{format_name}",
        "weights_path": exported.relative_to(output_dir).as_posix(),
        "weights_sha256": sha256_file(exported),
        "task": "detect",
        "class_mapping": {str(key): value for key, value in sorted(source.class_mapping.items())},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "exported_from": source.model_id,
        "export_format": format_name,
    }
    manifest_out = output_dir / "manifest.json"
    manifest_out.write_text(json.dumps(exported_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checked = load_manifest(manifest_out, models_root=output_dir)
    from bottle_qc.settings import Settings
    from bottle_qc.vision.yolo_provider import open_yolo_provider

    values, _, _ = Settings.from_environment().load_config("default")
    open_yolo_provider(
        manifest_out,
        models_root=output_dir,
        tracker=False,
        minimum_confidence=float(values["minimum_confidence"]),
        minimum_region_edge_variance=float(values["minimum_region_edge_variance"]),
    )
    return {
        "status": "EXPORTED_AND_PROBED",
        "model_id": checked.model_id,
        "manifest": str(manifest_out),
        "weights": str(exported),
        "weights_sha256": checked.weights_sha256,
        "format": format_name,
    }


if __name__ == "__main__":
    raise SystemExit(main())
