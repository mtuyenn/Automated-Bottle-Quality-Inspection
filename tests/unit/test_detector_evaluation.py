import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np

from bottle_qc.evaluation.detector import evaluate_detector


def test_detector_evaluation_maps_ultralytics_per_class_metrics_and_missing_classes(tmp_path, monkeypatch):
    models_root = tmp_path / "models"
    model_dir = models_root / "controlled"
    model_dir.mkdir(parents=True)
    weights_path = model_dir / "weights.pt"
    weights_path.write_bytes(b"controlled metric evaluation fixture")
    digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    manifest_path = model_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "model_id": "controlled",
        "weights_path": "weights.pt",
        "weights_sha256": digest,
        "task": "detect",
        "class_mapping": {"0": "bottle", "1": "cap", "2": "label"},
        "framework_version": "8.3.0",
        "dataset_manifest_ref": "controlled-dataset",
        "training_run_ref": "controlled-training",
        "validation_report_ref": "controlled-validation",
        "created_at": "controlled-fixture",
    }), encoding="utf-8")

    images = tmp_path / "dataset" / "images" / "val"
    images.mkdir(parents=True)
    data_path = tmp_path / "dataset.yaml"
    data_path.write_text(
        f"path: {tmp_path / 'dataset'}\nval: images/val\nnames: [bottle, cap, label]\n",
        encoding="utf-8",
    )

    metrics = SimpleNamespace(box=SimpleNamespace(
        ap_class_index=np.array([2, 0]),
        p=np.array([0.9, 0.8]),
        r=np.array([0.7, 0.6]),
        ap50=np.array([0.82, 0.72]),
        ap=np.array([0.64, 0.54]),
        mp=0.85,
        mr=0.65,
        map50=0.77,
        map=0.59,
    ))

    class ControlledYolo:
        def __init__(self, weights):
            assert weights == str(weights_path)

        def val(self, **kwargs):
            return metrics

    ultralytics = ModuleType("ultralytics")
    ultralytics.YOLO = ControlledYolo
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)

    provider_module = ModuleType("bottle_qc.vision.yolo_provider")
    provider_module.open_yolo_provider = lambda *args, **kwargs: object()
    monkeypatch.setitem(sys.modules, "bottle_qc.vision.yolo_provider", provider_module)

    report = evaluate_detector(
        manifest_path=manifest_path,
        models_root=models_root,
        data_path=data_path,
        split="val",
        device="cpu",
        imgsz=640,
        batch=2,
        workers=0,
    )

    assert report["status"] == "EVALUATED"
    assert report["classes"] == {
        "bottle": {"precision": 0.8, "recall": 0.6, "map50": 0.72, "map50_95": 0.54},
        "cap": {
            "precision": None,
            "recall": None,
            "map50": None,
            "map50_95": None,
            "status": "no_ground_truth_instances_in_split",
        },
        "label": {"precision": 0.9, "recall": 0.7, "map50": 0.82, "map50_95": 0.64},
    }
    assert report["aggregate"] == {
        "precision": 0.85,
        "recall": 0.65,
        "map50": 0.77,
        "map50_95": 0.59,
    }
