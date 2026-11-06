from __future__ import annotations

import json

import pytest

from bottle_qc.vision.model_manifest import ModelUnavailableError, load_manifest
from bottle_qc.vision.model_registry import ModelRegistry


def _manifest(**overrides):
    return {
        "model_id": "local-bottle-detector",
        "weights_path": "best.pt",
        "weights_sha256": "0" * 64,
        "task": "detect",
        "class_mapping": {"0": "bottle", "1": "cap", "2": "label"},
        "framework_version": "unverified",
        "dataset_manifest_ref": "not-a-training-record",
        "training_run_ref": "not-a-training-record",
        "validation_report_ref": "not-a-validation-report",
        "created_at": "not-a-training-record",
        **overrides,
    }


def test_manifest_rejects_wrong_task_and_class_mapping_before_weights(tmp_path):
    models = tmp_path / "models"
    model_dir = models / "candidate"
    model_dir.mkdir(parents=True)

    task_manifest = model_dir / "wrong-task.json"
    task_manifest.write_text(json.dumps(_manifest(task="segment")), encoding="utf-8")
    with pytest.raises(ModelUnavailableError, match="requires task='detect'"):
        load_manifest(task_manifest, models_root=models)

    class_manifest = model_dir / "wrong-classes.json"
    class_manifest.write_text(json.dumps(_manifest(class_mapping={"0": "bottle", "1": "cap"})), encoding="utf-8")
    with pytest.raises(ModelUnavailableError, match="exactly class IDs"):
        load_manifest(class_manifest, models_root=models)


def test_registry_keeps_manifest_without_local_weights_unavailable(tmp_path):
    models = tmp_path / "models"
    model_dir = models / "candidate"
    model_dir.mkdir(parents=True)
    (model_dir / "manifest.json").write_text(json.dumps(_manifest()), encoding="utf-8")

    registry = ModelRegistry(models, minimum_confidence=0.35, minimum_region_edge_variance=12)
    admitted = registry.refresh()

    assert len(admitted) == 1
    assert admitted[0]["status"] == "UNAVAILABLE"
    assert "weights are unavailable" in admitted[0]["reason"]
    assert registry.get("candidate")["status"] == "UNAVAILABLE"
