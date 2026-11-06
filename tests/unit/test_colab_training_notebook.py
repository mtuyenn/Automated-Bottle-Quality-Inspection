"""Exercise artifact restore/checkpoint persistence without torch, GPU or training."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
from types import SimpleNamespace
from types import ModuleType
import uuid
import zipfile

import pytest
import yaml


NOTEBOOK = Path(__file__).parents[2] / "notebooks/colab-training-preparation.ipynb"


def functions(notebook=NOTEBOOK):
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    env = dict(Path=Path, PurePosixPath=PurePosixPath, hashlib=hashlib, shutil=shutil,
               json=json, stat=stat, uuid=uuid, zipfile=zipfile, yaml=yaml,
               datetime=datetime, timezone=timezone)
    tags = {"training-runtime", "training-install", "training-data", "training-run",
            "training-evaluate", "training-video", "training-label-audit",
            "training-v2-repair", "training-speed"}
    for c in nb["cells"]:
        if c["cell_type"] != "code" or not tags.intersection(c.get("metadata", {}).get("tags", [])):
            continue
        tree = ast.parse("".join(c["source"]))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(NOTEBOOK), "exec"), env)
    return env


@pytest.mark.parametrize("problem", [None, "missing", "weights", "classes", "dataset", "manifest_hash"])
def test_new_runtime_video_bootstrap_opens_shared_run_without_dataset_or_training(tmp_path, problem):
    env = functions()
    env.update(RUN_NAME="test-run", CLASSES={0: "bottle", 1: "cap", 2: "label"},
               EXPECTED_DATASET_SHA="a" * 64)
    run = tmp_path / "shared-BottleQC/training-runs/test-run"
    weights = run / "weights/best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"selected trained model")
    digest = env["sha"](weights)
    binding = dict(dataset_sha256="a" * 64, classes={"0": "bottle", "1": "cap", "2": "label"},
                   torch="original-training-runtime")
    complete = dict(binding=binding, best_sha256=digest)
    manifest = dict(weights_sha256=digest, task="detect", class_mapping=dict(binding["classes"]))
    if problem == "classes":
        manifest["class_mapping"]["1"] = "label"
    if problem == "dataset":
        binding["dataset_sha256"] = "wrong"
    if problem == "manifest_hash":
        manifest["weights_sha256"] = "0" * 64
    env["write_json"](run / "TRAIN_COMPLETE.json", complete)
    env["write_json"](run / "manifest.json", manifest)
    if problem == "missing":
        (run / "manifest.json").unlink()
    if problem == "weights":
        weights.write_bytes(b"damaged model")
    before = {str(p.relative_to(run)): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    if problem:
        with pytest.raises(RuntimeError):
            env["select_video_run"](tmp_path / "shared-BottleQC", digest)
    else:
        selected, original_binding = env["select_video_run"](tmp_path / "shared-BottleQC", digest)
        assert selected == run
        assert original_binding == binding
    assert {str(p.relative_to(run)): p.read_bytes() for p in run.rglob("*") if p.is_file()} == before


def fixture_candidate(tmp_path, env, extra=None):
    src = tmp_path / "candidate"
    src.mkdir()
    records = []
    archive = src / "prepared-candidate.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for split in ("train", "val", "test"):
            image, label = f"images/{split}/a.jpg", f"labels/{split}/a.txt"
            image_bytes, label_bytes = b"fixture image " + split.encode(), b"0 .5 .5 .2 .2\n"
            z.writestr(image, image_bytes)
            z.writestr(label, label_bytes)
            records.append(dict(image=image, label=label, split=split,
                                image_sha256=hashlib.sha256(image_bytes).hexdigest(),
                                label_sha256=hashlib.sha256(label_bytes).hexdigest()))
        z.writestr("retained-checksums.json", json.dumps(records))
        z.writestr("data.yaml", yaml.safe_dump(dict(path=".", names=env["CLASSES"])))
        if extra:
            z.writestr(*extra)
    digest = env["sha"](archive)
    (src / "COMPLETE.json").write_text(json.dumps(dict(archive_sha256=digest)))
    (src / "residual-dhash-review.json").write_text(json.dumps(dict(
        archive_sha256=digest, status="ALL_WARNINGS_ALREADY_REVIEWED_DIFFERENT",
        requires_review=0, decisions_changed_since_9d=False)))
    return src, digest


def test_candidate_restore_portable_yaml_bytes_and_counts(tmp_path):
    env = functions()
    env["CLASSES"] = {0: "bottle", 1: "cap", 2: "label"}
    src, digest = fixture_candidate(tmp_path, env)
    before = {p.name: p.read_bytes() for p in src.iterdir()}
    data = env["restore_candidate"](src, tmp_path / "restored", digest,
                                    dict(train=1, val=1, test=1))
    assert yaml.safe_load(data.read_text())["path"] == str(data.parent)
    assert (data.parent / "labels/train/a.txt").read_bytes() == b"0 .5 .5 .2 .2\n"
    assert before == {p.name: p.read_bytes() for p in src.iterdir()}


@pytest.mark.parametrize("failure", ["checksum", "report", "path", "counts"])
def test_restore_rejects_invalid_artifacts(tmp_path, failure):
    env = functions()
    env["CLASSES"] = {0: "bottle", 1: "cap", 2: "label"}
    src, digest = fixture_candidate(tmp_path, env,
                                    ("../escape.txt", "bad") if failure == "path" else None)
    if failure == "checksum":
        (src / "prepared-candidate.zip").write_bytes(b"broken")
    if failure == "report":
        p = src / "residual-dhash-review.json"
        report = json.loads(p.read_text())
        report["archive_sha256"] = "0" * 64
        p.write_text(json.dumps(report))
    with pytest.raises(RuntimeError):
        env["restore_candidate"](src, tmp_path / "restored", digest,
                                  dict(train=2 if failure == "counts" else 1, val=1, test=1))
    assert not (tmp_path / "escape.txt").exists()


def test_interrupted_backup_preserves_previous_pointer_and_detects_corruption(tmp_path):
    env = functions()
    run, local = tmp_path / "drive", tmp_path / "local"
    local.mkdir()
    trainer = SimpleNamespace(epoch=0, last=local / "last.pt", best=local / "best.pt", save_dir=local)
    trainer.last.write_bytes(b"epoch1 optimizer")
    trainer.best.write_bytes(b"best1")
    env.update(RUN=run, BINDING={"dataset": "fixture"})
    env["backup_epoch"](trainer)
    prior = (run / "LATEST.json").read_bytes()
    assert env["read_snapshot"](run)["epoch"] == 1
    real_copy = env["copy_verified"]
    def fail_on_best(source, destination):
        if source.name == "best.pt":
            raise OSError("Drive disconnected")
        return real_copy(source, destination)
    env["copy_verified"] = fail_on_best
    trainer.epoch = 1
    trainer.last.write_bytes(b"epoch2 optimizer")
    with pytest.raises(OSError, match="disconnected"):
        env["backup_epoch"](trainer)
    assert (run / "LATEST.json").read_bytes() == prior
    snapshot = env["read_snapshot"](run)
    assert (run / snapshot["folder"] / "last.pt").read_bytes() == b"epoch1 optimizer"
    (run / snapshot["folder"] / "last.pt").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="lỗi"):
        env["read_snapshot"](run)


def test_resume_restores_last_and_best_and_does_not_download_initial_weights(tmp_path):
    env = functions()
    run, local = tmp_path / "drive", tmp_path / "local"
    local.mkdir()
    trainer = SimpleNamespace(epoch=0, last=local / "last.pt", best=local / "best.pt", save_dir=local)
    trainer.last.write_bytes(b"optimizer epoch1")
    trainer.best.write_bytes(b"selected best")
    env.update(RUN=run, LOCAL_RUN=tmp_path / "new-runtime/run", BINDING={"dataset": "fixture"},
               CONFIG={"epochs": 100, "workers": 2}, DATA_YAML=tmp_path / "new-data.yaml",
               require_gpu=lambda: None)
    env["backup_epoch"](trainer)
    calls = []
    class Model:
        def __init__(self, checkpoint, task):
            assert Path(checkpoint).read_bytes() == b"optimizer epoch1"
            assert task == "detect"
            self.trainer = SimpleNamespace(best=env["LOCAL_RUN"] / "weights/best.pt")
        def add_callback(self, event, callback):
            assert event == "on_model_save"
        def train(self, **kwargs):
            calls.append(kwargs)
    env["YOLO"] = Model
    env["train_or_resume"]()
    assert calls == [dict(resume=True, data=str(env["DATA_YAML"]), device=0, workers=2)]
    assert (run / "weights/best.pt").read_bytes() == b"selected best"
    env["train_or_resume"]()
    assert len(calls) == 1  # completed rerun skips training
    env["BINDING"] = {"dataset": "changed"}
    with pytest.raises(RuntimeError, match="binding"):
        env["train_or_resume"]()


def test_windows_guard_and_terminal_snapshot_skip_training(tmp_path, monkeypatch):
    env = functions()
    env["sys"] = sys
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="Colab"):
        env["require_gpu"]()
    local = tmp_path / "local"
    local.mkdir()
    trainer = SimpleNamespace(epoch=39, stop=True, last=local / "last.pt",
                              best=local / "best.pt", save_dir=local)
    trainer.last.write_bytes(b"last optimizer")
    trainer.best.write_bytes(b"best")
    (local / "results.csv").write_text("epoch,map\n40,0.5\n")
    env.update(RUN=tmp_path / "drive", LOCAL_RUN=tmp_path / "new-runtime",
               BINDING={"dataset": "fixture"}, CONFIG={"epochs": 100}, require_gpu=lambda: None)
    env["backup_epoch"](trainer)
    env["train_or_resume"]()  # no YOLO binding: would fail if it attempted training
    assert (env["LOCAL_RUN"] / "results.csv").read_text() == "epoch,map\n40,0.5\n"
    assert (env["RUN"] / "TRAIN_COMPLETE.json").exists()


@pytest.mark.parametrize("nonfinite", [False, True])
def test_entire_evaluation_cell_serializes_numpy_metrics_without_changing_weights(tmp_path, nonfinite):
    import numpy as np

    env = functions()
    run = tmp_path / "drive"
    work = tmp_path / "runtime"
    work.mkdir()
    classes = {0: "bottle", 1: "cap", 2: "label"}
    binding = {"dataset": "fixture"}
    env["write_json"](run / "TRAIN_COMPLETE.json", dict(
        binding=binding, best_sha256=hashlib.sha256(b"trained model").hexdigest()))
    weights = run / "weights/best.pt"
    weights.parent.mkdir()
    weights.write_bytes(b"trained model")
    old_report = b'{"previous":"valid report"}'
    (run / "validation.json").write_bytes(old_report)
    calls = []

    class Model:
        def __init__(self, path, task):
            assert Path(path).read_bytes() == weights.read_bytes()
            self.names, self.task = classes, task

        def val(self, **kwargs):
            calls.append(kwargs["split"])
            output = work / kwargs["split"]
            output.mkdir()
            (output / "confusion_matrix.png").write_bytes(b"plot fixture")
            return SimpleNamespace(
                results_dict={"metrics/mAP50(B)": np.float64(0.92)},
                summary=lambda: [dict(Class="bottle", Images=np.int64(300),
                                      Instances=np.int64(463), Recall=np.float32(0.94),
                                      extra={"flags": np.array([True, False]),
                                             "score": np.float32(np.nan if nonfinite else 0.75)})],
                speed={"inference": np.float64(5.2)}, save_dir=output)

    env.update(RUN=run, WORK=work, BINDING=binding, CLASSES=classes,
               CONFIG={"imgsz": 640, "batch": 16, "workers": 2},
               CANDIDATE_DIR=tmp_path / "candidate", RUN_NAME="test-run",
               EXPECTED_SHA="a" * 64, DATA_YAML=tmp_path / "data.yaml",
               require_gpu=lambda: None, YOLO=Model,
               ultralytics=SimpleNamespace(__version__="8.4.193"))
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = next(c for c in nb["cells"] if "training-evaluate" in c["metadata"].get("tags", []))
    code = compile("".join(cell["source"]), str(NOTEBOOK), "exec")
    if nonfinite:
        with pytest.raises(ValueError, match="JSON"):
            exec(code, env)
        assert (run / "validation.json").read_bytes() == old_report
        assert not (run / "manifest.json").exists()
    else:
        exec(code, env)
        assert calls == ["val", "test"]
        for filename in ("validation.json", "test.json"):
            report = json.loads((run / filename).read_text())
            assert report["per_class"][0]["Instances"] == 463
            assert report["per_class"][0]["extra"]["flags"] == [True, False]
        from bottle_qc.vision.model_manifest import load_manifest
        assert load_manifest(run / "manifest.json").class_mapping == classes
    assert weights.read_bytes() == b"trained model"
    assert (run / "TRAIN_COMPLETE.json").exists()


def test_video_download_checks_hash_and_size_without_clobbering_valid_input(tmp_path):
    import io

    env = functions()
    payload = b"small-video-fixture"
    env["urllib"] = SimpleNamespace(request=SimpleNamespace(urlopen=lambda *a, **k: io.BytesIO(payload)))
    target = tmp_path / "video.mp4"
    expected = hashlib.sha256(payload).hexdigest()
    env["download_demo_video"]("https://fixture/video", target, expected)
    assert target.read_bytes() == payload
    target.write_bytes(b"previous")
    with pytest.raises(RuntimeError, match="SHA"):
        env["download_demo_video"]("https://fixture/video", target, "0" * 64)
    assert target.read_bytes() == b"previous"
    with pytest.raises(RuntimeError, match="giới hạn"):
        env["download_demo_video"]("https://fixture/video", target, expected, limit=2)
    assert target.read_bytes() == b"previous"
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("device", [0, "cpu"])
@pytest.mark.parametrize("stride", [1, 2])
def test_video_demo_streams_real_decoder_and_releases_resources(tmp_path, monkeypatch, fail, device, stride):
    import csv
    import math
    import numpy as np
    cv2 = pytest.importorskip("cv2")
    source = tmp_path / "fixture.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (100, 80))
    assert writer.isOpened()
    for i in range(3):
        writer.write(np.full((80, 100, 3), i * 50, dtype=np.uint8))
    writer.release()
    env = functions()
    env.update(csv=csv, math=math, cv2=cv2, CLASSES={0: "bottle", 1: "cap", 2: "label"})
    capture_factory, writer_factory = cv2.VideoCapture, cv2.VideoWriter
    resources = []
    def capture(*args):
        item = capture_factory(*args)
        resources.append(item)
        return item
    def video_writer(*args):
        item = writer_factory(*args)
        resources.append(item)
        return item
    monkeypatch.setattr(cv2, "VideoCapture", capture)
    monkeypatch.setattr(cv2, "VideoWriter", video_writer)
    calls = []
    class Model:
        def predict(self, frame, **kwargs):
            calls.append(kwargs)
            if fail and len(calls) == 2:
                raise RuntimeError("inference failed")
            box = SimpleNamespace(cls=np.array(0), conf=np.array(0.8), xyxy=np.array([[1, 2, 30, 70]]))
            return [SimpleNamespace(boxes=[box], plot=lambda **k: frame.copy())]
    folder = tmp_path / "output"
    if fail:
        with pytest.raises(RuntimeError, match="inference failed"):
            env["predict_demo_frames"](Model(), source, folder, device=device, stride=stride)
    else:
        summary = env["predict_demo_frames"](Model(), source, folder, device=device, stride=stride)
        assert summary["frames"] == 3 and summary["duration"] == 0.3
        expected_indices = list(range(0, 3, stride))
        assert summary["predicted_frames"] == len(expected_indices)
        assert summary["output_fps"] == 10.0 / stride
        assert 0 <= summary["output_duration"] - summary["duration"] < stride / 10
        rendered = capture_factory(str(folder / "annotated-mp4v.mp4"))
        assert rendered.isOpened()
        assert int(rendered.get(cv2.CAP_PROP_FRAME_COUNT)) == len(expected_indices)
        assert rendered.get(cv2.CAP_PROP_FPS) == pytest.approx(10.0 / stride)
        rendered.release()
        with (folder / "detections.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert [r["frame"] for r in rows] == [str(i) for i in expected_indices]
        assert [float(r["time_seconds"]) for r in rows] == [i / 10 for i in expected_indices]
        assert {r["class_name"] for r in rows} == {"bottle"}
        assert all(call["device"] == device for call in calls)
        assert (folder / "frame-0000.jpg").exists()
    assert all(not item.isOpened() for item in resources)


@pytest.mark.parametrize("device,cuda,platform,error", [
    ("cpu", False, "linux", None),
    (0, True, "linux", None),
    (0, False, "linux", RuntimeError),
    ("other", True, "linux", ValueError),
    ("cpu", False, "win32", RuntimeError),
])
def test_video_runtime_explicit_cpu_gpu_and_workstation_guard(monkeypatch, device, cuda, platform, error):
    env = functions()
    env.update(sys=sys, Path=lambda value: SimpleNamespace(is_dir=lambda: True))
    monkeypatch.setattr(sys, "platform", platform)
    google = ModuleType("google")
    google.__path__ = []
    google.colab = ModuleType("google.colab")
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: cuda)
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", google.colab)
    monkeypatch.setitem(sys.modules, "torch", torch)
    if error:
        with pytest.raises(error):
            env["require_video_runtime"](device)
    else:
        assert env["require_video_runtime"](device) is torch


def saved_demo_fixture(tmp_path, env):
    run = tmp_path / "drive-run"
    folder = run / "video-tests/factory-line-20261103T173407Z-fixture"
    folder.mkdir(parents=True)
    content = {"factory-input.mp4": b"input video", "annotated.mp4": b"rendered H264",
               "detections.csv": b"frame,class\n0,bottle\n"}
    for name, data in content.items():
        (folder / name).write_bytes(data)
    env["VIDEO_SHA"] = env["sha"](folder / "factory-input.mp4")
    inference = dict(imgsz=640, conf=0.25, device="cpu", vid_stride=4, start_seconds=10.0)
    report = dict(weights_sha256="a" * 64, source_sha256=env["VIDEO_SHA"], inference=inference,
                  files={name: env["sha"](folder / name) for name in content})
    env["write_json"](folder / "video-demo.json", report)
    return run, folder, report


@pytest.mark.parametrize("problem", [None, "other_model", "other_device", "other_stride", "other_start", "pending",
                                     "missing", "corrupt", "unsafe_path"])
def test_saved_demo_reuse_requires_matching_report_and_intact_artifacts(tmp_path, problem):
    env = functions()
    run, folder, report = saved_demo_fixture(tmp_path, env)
    if problem == "other_model":
        report["weights_sha256"] = "b" * 64
    if problem == "other_device":
        report["inference"]["device"] = 0
    if problem == "other_stride":
        report["inference"]["vid_stride"] = 1
    if problem == "other_start":
        report["inference"]["start_seconds"] = 0.0
    if problem == "missing":
        (folder / "annotated.mp4").unlink()
    if problem == "corrupt":
        (folder / "annotated.mp4").write_bytes(b"damaged")
    if problem == "unsafe_path":
        report["files"]["../escape.mp4"] = "0" * 64
    env["write_json"](folder / "video-demo.json", report)
    if problem == "pending":
        (folder / "video-demo.json").unlink()
    kwargs = (run, "a" * 64, dict(imgsz=640, conf=0.25, device="cpu", vid_stride=4, start_seconds=10.0))
    if problem in {"missing", "corrupt", "unsafe_path"}:
        with pytest.raises(RuntimeError):
            env["find_saved_video_demo"](*kwargs)
    elif problem:
        assert env["find_saved_video_demo"](*kwargs) is None
    else:
        assert env["find_saved_video_demo"](*kwargs) == (folder, report)


def test_completed_video_is_shown_without_repeating_model_inference_or_writes(tmp_path):
    env = functions()
    run, folder, report = saved_demo_fixture(tmp_path, env)
    weights = run / "weights/best.pt"
    weights.parent.mkdir()
    weights.write_bytes(b"selected model")
    digest = env["sha"](weights)
    report["weights_sha256"] = digest
    env["write_json"](folder / "video-demo.json", report)
    binding = {"original": "training provenance"}
    env["write_json"](run / "TRAIN_COMPLETE.json", dict(binding=binding, best_sha256=digest))
    viewed = []
    env.update(RUN=run, BINDING=binding, VIDEO_DEVICE="cpu", VIDEO_STRIDE=4, VIDEO_START_SECONDS=10.0,
               require_video_runtime=lambda device: None, display_video_demo=viewed.append)
    # No YOLO, downloader, ffmpeg/subprocess or frame processor bound: a rerun would fail.
    before = {str(p.relative_to(run)): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    env["run_video_demo"]()
    assert viewed == [folder / "annotated.mp4"]
    assert before == {str(p.relative_to(run)): p.read_bytes() for p in run.rglob("*") if p.is_file()}


def test_video_html_payload_preserves_bytes_and_obeys_embedding_limit(tmp_path):
    import base64
    import re

    env = functions()
    path = tmp_path / "annotated.mp4"
    path.write_bytes(b"small H264 fixture")
    shown = []
    env.update(HTML=lambda html: html, display=shown.append)
    env["display_video_demo"](path)
    html = shown.pop()
    encoded = re.search(r"data:video/mp4;base64,([^\" ]+)", html).group(1)
    assert base64.b64decode(encoded) == path.read_bytes()
    assert "<video controls playsinline" in html
    env["display_video_demo"](path, limit=2)
    assert shown == []


def test_long_factory_clip_budget_and_sampling_keep_source_time(tmp_path):
    import csv
    import math
    import numpy as np
    cv2 = pytest.importorskip("cv2")
    source = tmp_path / "long-fixture.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
    assert writer.isOpened()
    for i in range(1002):
        writer.write(np.full((32, 32, 3), i % 255, dtype=np.uint8))
    writer.release()
    env = functions()
    env.update(csv=csv, math=math, cv2=cv2)
    calls = []
    class Model:
        def predict(self, frame, **kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(boxes=[], plot=lambda **k: frame)]
    with pytest.raises(RuntimeError, match="giới hạn frame"):
        env["predict_demo_frames"](Model(), source, tmp_path / "too-long", max_frames=1000)
    assert not calls
    summary = env["predict_demo_frames"](Model(), source, tmp_path / "factory",
                                          device="cpu", stride=4, max_frames=6000)
    assert summary["frames"] == 1002
    assert summary["predicted_frames"] == len(calls) == 251
    assert summary["duration"] == 100.2
    assert summary["output_fps"] == 2.5
    assert summary["output_duration"] == 100.4  # last sampled interval pads < stride/fps
    rendered = cv2.VideoCapture(str(tmp_path / "factory/annotated-mp4v.mp4"))
    assert int(rendered.get(cv2.CAP_PROP_FRAME_COUNT)) == 251
    assert rendered.get(cv2.CAP_PROP_FPS) == 2.5
    rendered.release()


@pytest.mark.parametrize("start_seconds", [1.0, 1.01, -1.0, float("nan"), 2.0])
def test_video_start_skips_intro_and_retains_source_timestamps(tmp_path, start_seconds):
    import csv
    import math
    import numpy as np
    cv2 = pytest.importorskip("cv2")
    source = tmp_path / "intro.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
    assert writer.isOpened()
    for i in range(20):
        writer.write(np.full((32, 32, 3), i * 10, dtype=np.uint8))
    writer.release()
    env = functions()
    env.update(csv=csv, math=math, cv2=cv2, CLASSES={0: "bottle", 1: "cap", 2: "label"})
    observed = []

    class Model:
        def predict(self, frame, **kwargs):
            observed.append(float(frame.mean()))
            box = SimpleNamespace(cls=np.array(0), conf=np.array(0.8), xyxy=np.array([[1, 2, 20, 30]]))
            return [SimpleNamespace(boxes=[box], plot=lambda **k: frame.copy())]

    folder = tmp_path / "result"
    if not math.isfinite(start_seconds) or start_seconds < 0 or start_seconds >= 2:
        with pytest.raises(ValueError):
            env["predict_demo_frames"](Model(), source, folder, start_seconds=start_seconds)
        assert not observed
        return
    summary = env["predict_demo_frames"](Model(), source, folder, device="cpu", stride=2,
                                          start_seconds=start_seconds)
    start_frame = math.ceil(start_seconds * 10)
    indices = list(range(start_frame, 20, 2))
    assert observed == pytest.approx([i * 10 for i in indices], abs=2)
    assert summary["frames"] == 20
    assert summary["start_frame"] == start_frame
    assert summary["start_seconds"] == start_frame / 10
    assert summary["clip_duration"] == (20 - start_frame) / 10
    assert summary["predicted_frames"] == len(indices)
    with (folder / "detections.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(r["frame"]) for r in rows] == indices
    assert [float(r["time_seconds"]) for r in rows] == [i / 10 for i in indices]
    assert (folder / ("frame-%04d.jpg" % start_frame)).is_file()
    assert not (folder / "frame-0000.jpg").exists()
    rendered = cv2.VideoCapture(str(folder / "annotated-mp4v.mp4"))
    assert int(rendered.get(cv2.CAP_PROP_FRAME_COUNT)) == len(indices)
    assert rendered.get(cv2.CAP_PROP_FPS) == 5
    ok, first = rendered.read()
    assert ok and first.mean() == pytest.approx(start_frame * 10, abs=6)
    rendered.release()


@pytest.mark.parametrize("problem", [None, "archive_hash", "image_hash", "invalid_label"])
def test_label_audit_renders_real_candidate_sample_without_modifying_source(tmp_path, problem):
    import io
    from PIL import Image

    archive = tmp_path / "candidate.zip"
    records = []
    with zipfile.ZipFile(archive, "w") as z:
        for split in ("train", "val", "test"):
            image_path, label_path = f"images/{split}/bottle.jpg", f"labels/{split}/bottle.txt"
            stream = io.BytesIO()
            Image.new("RGB", (100, 80), (50, 50, 50)).save(stream, format="JPEG")
            image = stream.getvalue()
            label = b"0 0.5 0.5 0.6 0.8\n1 0.5 0.2 0.2 0.1\n2 0.5 0.6 0.4 0.2\n"
            if problem == "invalid_label" and split == "train":
                label += b"5 0.5 0.5 0.1 0.1\n"
            z.writestr(image_path, image)
            z.writestr(label_path, label)
            records.append(dict(split=split, image=image_path, label=label_path,
                                image_sha256="0" * 64 if problem == "image_hash" else hashlib.sha256(image).hexdigest(),
                                label_sha256=hashlib.sha256(label).hexdigest()))
        z.writestr("retained-checksums.json", json.dumps(records))
    original = archive.read_bytes()
    env = functions()
    digest = "0" * 64 if problem == "archive_hash" else hashlib.sha256(original).hexdigest()
    output = tmp_path / "preview"
    args = (archive, output, digest, dict(train=1, val=1, test=1), dict(train=1, val=1, test=1))
    if problem in {"archive_hash", "image_hash"}:
        with pytest.raises(RuntimeError, match="checksum|khớp"):
            env["build_label_audit"](*args)
        assert not (output / "label-audit.json").exists()
    else:
        report = env["build_label_audit"](*args)
        assert report["review_status"] == "pending" and report["accuracy"] is None
        assert report["sampled_images"] == 3
        assert report["counts"] == dict(train=1, val=1, test=1)
        assert all(v == dict(bottle=1, cap=1, label=1) for v in report["label_instances"].values())
        assert len(report["label_issues"]) == (1 if problem == "invalid_label" else 0)
        for sample in report["samples"]:
            assert sample["review_status"] == "pending"
            assert sample["image"] in {r["image"] for r in records}
            with Image.open(output / sample["overlay"]) as preview:
                assert preview.size == (100, 80)
                # The known bottle boundary is at x=20; its interior stays the original grey.
                red, green, blue = preview.getpixel((20, 40))
                assert blue > red + 30
                assert preview.getpixel((5, 40)) == pytest.approx((50, 50, 50), abs=10)
        with Image.open(output / report["sheets"][0]) as sheet:
            assert sheet.size == (960, 1290)
        assert json.loads((output / "label-audit.json").read_text(encoding="utf-8"))["dataset_sha256"] == digest
    assert archive.read_bytes() == original


def test_v2_colab_stages_have_stable_ids_unexecuted_cells_and_historical_outputs():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    by_id = {cell.get("id"): cell for cell in notebook["cells"]}
    expected_code_ids = {
        "v2-core-contracts", "v2-full-candidate-audit", "v2-bounded-semantic-review",
        "v2-authorized-source-ingestion", "v2-reviewed-repair-attestation",
        "v2-grouped-candidate-build", "v2-yolo11n-train-resume", "v2-paired-evaluation",
        "v2-component-timing", "v2-not-run-regressions",
    }
    assert expected_code_ids <= set(by_id)
    for ident in expected_code_ids:
        cell = by_id[ident]
        assert cell["cell_type"] == "code"
        assert cell["execution_count"] is None
        assert cell["outputs"] == []

    historical_execution = [None, None, 1, None, 2, None, 3, None, 4, None,
                            6, None, 2, None, 3, None, 2, None, None]
    historical_output_counts = [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 3, 0, 25, 0, 29, 0, 0]
    for index, cell in enumerate(notebook["cells"][:19]):
        assert cell.get("execution_count") == historical_execution[index]
        assert len(cell.get("outputs", [])) == historical_output_counts[index]
    assert notebook["cells"][16]["id"] == "35b0f7bc"
    assert notebook["cells"][16]["execution_count"] == 2
    assert "CURRENT_LABELS" in "".join(notebook["cells"][16]["source"])
    assert "predict(" not in "".join(notebook["cells"][16]["source"])
    assert notebook["cells"][18]["id"] == "1049d3dd"
    assert notebook["cells"][18]["execution_count"] is None


def test_v2_notebook_displayed_contract_covers_policy_run_order_and_not_run_status():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                     if cell["cell_type"] in {"markdown", "code"})
    for stage in range(10, 20):
        assert f"## {stage} " in text
    for requirement in (
        "0=bottle", "1=cap", "2=label", "bị cắt", "partial", "hard negative",
        "rights", "quarantine", "holdout", "checkpoint", "resume", "AP50",
        "source_time_seconds", "NOT_RUN", "semantic-clean",
    ):
        assert requirement.lower() in text.lower()
