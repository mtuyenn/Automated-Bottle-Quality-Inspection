from __future__ import annotations

from pathlib import Path
from typing import Any

from bottle_qc.vision.model_manifest import ModelUnavailableError, load_manifest
from bottle_qc.vision.yolo_provider import open_yolo_provider


class ModelRegistry:
    """Read-only catalog. Every admitted manifest must pass a real local inference probe."""

    def __init__(self, models_dir: Path, *, minimum_confidence: float, minimum_region_edge_variance: float):
        self.models_dir = Path(models_dir).resolve()
        self.minimum_confidence = minimum_confidence
        self.minimum_region_edge_variance = minimum_region_edge_variance
        self._models: dict[str, dict[str, Any]] = {}

    def refresh(self) -> list[dict[str, Any]]:
        self._models.clear()
        if not self.models_dir.is_dir():
            return []
        for path in sorted(self.models_dir.rglob("manifest.json")):
            candidate_id = path.parent.name
            try:
                manifest = load_manifest(path, models_root=self.models_dir)
                candidate_id = manifest.model_id
                provider = open_yolo_provider(
                    path,
                    models_root=self.models_dir,
                    tracker=False,
                    minimum_confidence=self.minimum_confidence,
                    minimum_region_edge_variance=self.minimum_region_edge_variance,
                )
                capability = provider.probe()
                record = {
                    "model_id": candidate_id,
                    "status": "AVAILABLE",
                    "task": manifest.task,
                    "classes": [manifest.class_mapping[index] for index in sorted(manifest.class_mapping)],
                    "weights_sha256": manifest.weights_sha256,
                    "framework_version": manifest.framework_version,
                    "manifest_path": str(path),
                    "capability": capability,
                }
            except Exception as exc:
                record = {
                    "model_id": candidate_id,
                    "status": "UNAVAILABLE",
                    "manifest_path": str(path),
                    "reason": str(exc),
                }
            if candidate_id in self._models:
                record = {"model_id": candidate_id, "status": "UNAVAILABLE", "reason": "Duplicate model_id manifests exist"}
            self._models[candidate_id] = record
        return self.list()

    def list(self) -> list[dict[str, Any]]:
        return [dict(self._models[key]) for key in sorted(self._models)]

    def get(self, model_id: str) -> dict[str, Any] | None:
        item = self._models.get(model_id)
        return dict(item) if item else None

    def available_manifest(self, model_id: str) -> Path:
        model = self._models.get(model_id)
        if model is None:
            raise ModelUnavailableError(f"Model '{model_id}' is not registered")
        if model["status"] != "AVAILABLE":
            raise ModelUnavailableError(str(model.get("reason", "Model is unavailable")))
        return Path(model["manifest_path"])
