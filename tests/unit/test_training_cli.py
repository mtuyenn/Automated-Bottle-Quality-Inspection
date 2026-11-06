import builtins
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from bottle_qc.training.train import _load_data, resolve_training_options, run_training


def _run(module: str, *arguments: str) -> subprocess.CompletedProcess:
    project = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-B", "-m", module, *arguments],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_training_help_and_missing_inputs_are_bounded(tmp_path):
    top_level_help = _run("bottle_qc.training.train", "--help")
    assert top_level_help.returncode == 0 and "train" in top_level_help.stdout
    help_result = _run("bottle_qc.training.train", "train", "--help")
    assert help_result.returncode == 0
    assert "--data" in help_result.stdout and "--weights" in help_result.stdout

    result = _run(
        "bottle_qc.training.train",
        "train",
        "--data", str(tmp_path / "missing.yaml"),
        "--weights", str(tmp_path / "missing.pt"),
        "--output", str(tmp_path / "run"),
    )
    assert result.returncode == 2
    report = json.loads(result.stdout)
    assert report["status"] == "WAITING_FOR_INPUT"
    assert "validated dataset" in report["next_step"].lower()


def test_detector_evaluation_help_and_missing_input_are_bounded(tmp_path):
    help_result = _run("bottle_qc.evaluation.detector", "--help")
    assert help_result.returncode == 0
    assert "--split" in help_result.stdout and "--manifest" in help_result.stdout
    result = _run(
        "bottle_qc.evaluation.detector",
        "--manifest", str(tmp_path / "missing-manifest.json"),
        "--models-root", str(tmp_path / "models"),
        "--data", str(tmp_path / "missing-data.yaml"),
        "--split", "test",
    )
    assert result.returncode == 2
    report = json.loads(result.stdout)
    assert report["status"] == "WAITING_FOR_INPUT"
    assert "no detector metrics" in report["next_step"].lower()


def test_training_configuration_drives_defaults_and_cli_values_override_it():
    config = {
        "task": "detect",
        "model": "yolo11n",
        "model_id": "bottleqc-training-config",
        "seed": 19,
        "imgsz": 512,
        "epochs": 8,
        "batch": 4,
        "device": "cpu",
        "workers": 0,
        "resume": False,
    }
    defaults = SimpleNamespace(model_id=None, seed=None, imgsz=None, epochs=None, batch=None, device=None, workers=None, resume=None)
    configured, issue = resolve_training_options(config, defaults)
    assert issue is None
    assert configured["epochs"] == 8 and configured["imgsz"] == 512 and configured["model_id"] == "bottleqc-training-config"

    overrides = SimpleNamespace(model_id="explicit-model", seed=7, imgsz=640, epochs=3, batch=2, device="cpu", workers=0, resume=True)
    selected, issue = resolve_training_options(config, overrides)
    assert issue is None
    assert selected["model_id"] == "explicit-model" and selected["epochs"] == 3 and selected["seed"] == 7 and selected["resume"] is True


def test_training_invalid_class_mapping_returns_bounded_json_error(tmp_path):
    data_yaml = tmp_path / "invalid.yaml"
    data_yaml.write_text(
        "path: .\ntrain: missing-train\nval: missing-val\nnames:\n  bottle: bottle\n",
        encoding="utf-8",
    )
    result = _run(
        "bottle_qc.training.train",
        "train",
        "--data", str(data_yaml),
        "--weights", str(tmp_path / "missing.pt"),
        "--output", str(tmp_path / "run"),
    )
    assert result.returncode == 2
    report = json.loads(result.stdout)
    assert report["status"] == "INVALID"
    assert "class ids" in report["error"].lower()


def _training_dataset(tmp_path, *, train_label="", val_label="", same_image=False):
    for split in ("train", "val"):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
        color = "white" if split == "train" or same_image else "black"
        Image.new("RGB", (20, 20), color).save(tmp_path / "images" / split / "sample.png")
        label = train_label if split == "train" else val_label
        (tmp_path / "labels" / split / "sample.txt").write_text(label, encoding="utf-8")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        f"path: {tmp_path.as_posix()}\ntrain: images/train\nval: images/val\nnames: [bottle, cap, label]\n",
        encoding="utf-8",
    )
    return data_yaml


def _admit_training_data_without_trainer(tmp_path, data_yaml, monkeypatch):
    weights = tmp_path / "initial.pt"
    weights.write_bytes(b"checkpoint")
    config = tmp_path / "train.toml"
    config.write_text('task = "detect"\n', encoding="utf-8")
    imported = []
    real_import = builtins.__import__

    def block_trainer(name, *args, **kwargs):
        if name == "ultralytics":
            imported.append(name)
            raise ImportError("trainer import intercepted")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_trainer)
    report, code = run_training(SimpleNamespace(
        config=config,
        data=data_yaml,
        weights=weights,
        output=tmp_path / "output",
    ))
    return report, code, imported


def test_training_rejects_invalid_label_before_importing_trainer(tmp_path, monkeypatch):
    data_yaml = _training_dataset(tmp_path, train_label="99 0.5 0.5 0.2 0.2\n")
    report, code, imported = _admit_training_data_without_trainer(tmp_path, data_yaml, monkeypatch)

    assert report["status"] == "INVALID"
    assert code == 2
    assert imported == []


@pytest.mark.parametrize("label", [
    "0 -0.1 0.5 0.2 0.2\n",
    "0 1.1 0.5 0.2 0.2\n",
    "0 0.5 0.5 -0.2 0.2\n",
    "0 nan 0.5 0.2 0.2\n",
    "0 0.5 0.5 inf 0.2\n",
    "0 0.5 0.5 0.2\n",
])
def test_training_rejects_invalid_yolo_rows_before_importing_trainer(tmp_path, monkeypatch, label):
    data_yaml = _training_dataset(tmp_path, train_label=label)

    report, code, imported = _admit_training_data_without_trainer(tmp_path, data_yaml, monkeypatch)

    assert report["status"] == "INVALID"
    assert code == 2
    assert imported == []


@pytest.mark.parametrize("pairing", ["missing", "extra"])
def test_training_rejects_invalid_image_label_pairing_before_importing_trainer(tmp_path, monkeypatch, pairing):
    data_yaml = _training_dataset(tmp_path)
    labels = tmp_path / "labels" / "train"
    if pairing == "missing":
        (labels / "sample.txt").unlink()
    else:
        (labels / "unpaired.txt").write_text("", encoding="utf-8")

    report, code, imported = _admit_training_data_without_trainer(tmp_path, data_yaml, monkeypatch)

    assert report["status"] == "INVALID"
    assert code == 2
    assert imported == []


def test_training_rejects_train_validation_image_leakage_before_importing_trainer(tmp_path, monkeypatch):
    data_yaml = _training_dataset(
        tmp_path,
        train_label="0 0.5 0.5 0.2 0.2\n",
        val_label="0 0.5 0.5 0.2 0.2\n",
        same_image=True,
    )

    report, code, imported = _admit_training_data_without_trainer(tmp_path, data_yaml, monkeypatch)

    assert report["status"] == "INVALID"
    assert code == 2
    assert imported == []


def test_training_rejects_split_paths_outside_dataset_root_before_importing_trainer(tmp_path, monkeypatch):
    data_yaml = _training_dataset(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    (outside / "images" / "train").mkdir(parents=True)
    (outside / "labels" / "train").mkdir(parents=True)
    Image.new("RGB", (20, 20), "blue").save(outside / "images" / "train" / "sample.png")
    (outside / "labels" / "train" / "sample.txt").write_text("", encoding="utf-8")
    data_yaml.write_text(
        data_yaml.read_text(encoding="utf-8").replace("train: images/train", f"train: ../{outside.name}/images/train"),
        encoding="utf-8",
    )

    report, code, imported = _admit_training_data_without_trainer(tmp_path, data_yaml, monkeypatch)

    assert report["status"] == "INVALID"
    assert code == 2
    assert imported == []


def test_validated_train_and_validation_data_remains_admissible(tmp_path):
    data_yaml = _training_dataset(
        tmp_path,
        train_label="0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.1 0.1\n",
        val_label="2 0.5 0.5 0.3 0.3\n",
    )

    data, issue = _load_data(data_yaml)

    assert issue is None
    assert data["names"] == ["bottle", "cap", "label"]
