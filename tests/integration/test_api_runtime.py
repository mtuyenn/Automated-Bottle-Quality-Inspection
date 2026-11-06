import csv
import json
from io import BytesIO, StringIO
import sqlite3
from pathlib import Path
import threading
import uuid

from fastapi.testclient import TestClient
from PIL import Image

from bottle_qc.api.app import create_app
from bottle_qc.settings import Settings
from bottle_qc.storage.repository import Repository


PROJECT = Path(__file__).resolve().parents[2]


def _settings(tmp_path):
    return Settings(
        root_dir=tmp_path,
        data_dir=tmp_path / "runtime",
        models_dir=tmp_path / "models",
        configs_dir=PROJECT / "configs",
    )


def _simulation_request():
    return {
        "source_type": "simulation",
        "scenario": "phase-2-four-bottle-demo",
        "mode": "SIMULATION",
        "config_id": "default",
    }


def test_api_simulation_is_idempotent_persisted_and_export_consistent(tmp_path):
    with TestClient(create_app(_settings(tmp_path))) as client:
        assert client.get("/health").json() == {"status": "ok"}
        readiness = client.get("/readiness").json()
        assert readiness["status"] == "ready" and readiness["simulation_available"] is True
        assert readiness["models"] == []

        started = client.post("/runs", json=_simulation_request(), headers={"Idempotency-Key": "demo-one"})
        assert started.status_code == 202
        run_id = started.json()["run_id"]
        assert client.app.state.worker.wait(run_id, timeout=10)
        replay = client.post("/runs", json=_simulation_request(), headers={"Idempotency-Key": "demo-one"})
        assert replay.status_code == 200
        assert replay.json()["run_id"] == run_id and replay.json()["idempotent_replay"] is True

        run = client.get(f"/runs/{run_id}").json()
        assert run["status"] == "COMPLETED"
        assert run["counters"] == {"PASS": 1, "FAIL": 2, "UNINSPECTED": 1, "PENDING": 0, "total": 4}
        events_response = client.get(f"/runs/{run_id}/events")
        events = events_response.json()["items"]
        assert len(events) == 4
        assert all(event["mode"] == "SIMULATION" and event["evidence_state"] == "AVAILABLE" for event in events)
        assert {event["track_id"]: event["outcome"] for event in events} == {
            "1": "PASS", "2": "FAIL", "3": "FAIL", "4": "UNINSPECTED"
        }
        failures = {event["track_id"]: set(event["reasons"]) for event in events if event["outcome"] == "FAIL"}
        assert failures["2"] >= {"CAP_MISSING"}
        assert failures["3"] >= {"LABEL_MISSING"}
        assert all(event["mode"] == "SIMULATION" for event in events if event["outcome"] == "FAIL")
        evidence = client.get(f"/runs/{run_id}/events/{events[0]['event_id']}/evidence")
        assert evidence.status_code == 200 and evidence.content.startswith(b"\xff\xd8")
        report = client.get(f"/runs/{run_id}/report")
        rows = list(csv.DictReader(StringIO(report.text)))
        assert len(rows) == run["counters"]["total"]
        assert {row["outcome"] for row in rows} == {"PASS", "FAIL", "UNINSPECTED"}

        mismatch = client.post(
            "/runs",
            json={**_simulation_request(), "config_id": "unavailable"},
            headers={"Idempotency-Key": "demo-one"},
        )
        assert mismatch.status_code == 409


def test_large_report_streams_all_events_and_filtered_pages_match_database(tmp_path, monkeypatch):
    with TestClient(create_app(_settings(tmp_path))) as client:
        started = client.post("/runs", json=_simulation_request(), headers={"Idempotency-Key": "large-report"})
        run_id = started.json()["run_id"]
        assert client.app.state.worker.wait(run_id, timeout=10)
        run = client.app.state.repository.get_run(run_id)
        repository = client.app.state.repository

        def event_values():
            for index in range(4997):
                event_id = str(uuid.uuid4())
                outcome = ("PASS", "FAIL", "UNINSPECTED")[index % 3]
                yield (
                    event_id, run_id, f"bulk-source-{index}", f"bulk-track-{index}", outcome,
                    json.dumps(["CAP_MISSING"] if outcome == "FAIL" else []), 10_000 + index,
                    float(index * 40), "SIMULATION", None, None, run["config_id"], run["config_hash"],
                    float(index), float(index + 1), index % 5 + 1, 0.5, 0.75, 0.875,
                    json.dumps({"batch": "large-report", "sequence": index}),
                    f"{run_id}/{event_id}.jpg", f"2026-01-01T00:{index // 60:02d}:{index % 60:02d}+00:00",
                )

        with repository.db.transaction() as connection:
            connection.executemany(
                """INSERT INTO events(event_id,run_id,source_item_id,track_id,outcome,reasons_json,source_index,
                   source_timestamp_ms,mode,model_id,model_hash,config_id,config_hash,first_seen_at,finalized_at,
                   valid_observation_count,observation_duration,cap_support,label_support,extra_json,evidence_state,
                   evidence_relpath,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'PENDING',?,?)""",
                event_values(),
            )

        original_list_events = repository.list_events
        requested_page_sizes = []

        def observe_pages(selected_run_id, *, limit=5000, offset=0, outcome=None):
            requested_page_sizes.append(limit)
            if outcome is None:
                return original_list_events(selected_run_id, limit=limit, offset=offset)
            return original_list_events(selected_run_id, limit=limit, offset=offset, outcome=outcome)

        monkeypatch.setattr(repository, "list_events", observe_pages)
        report = client.get(f"/runs/{run_id}/report")
        assert report.status_code == 200
        rows = list(csv.DictReader(StringIO(report.text)))
        database_events = repository.db.fetchall(
            "SELECT * FROM events WHERE run_id=? ORDER BY COALESCE(source_index,-1),created_at,event_id",
            (run_id,),
        )
        assert len(rows) == len(database_events) == 5001
        assert max(requested_page_sizes) <= 1000
        assert len(requested_page_sizes) > 2

        for exported, stored in zip(rows, database_events):
            assert exported["event_id"] == stored["event_id"]
            assert exported["run_id"] == stored["run_id"]
            assert exported["source_item_id"] == stored["source_item_id"]
            assert exported["track_id"] == stored["track_id"]
            assert exported["outcome"] == stored["outcome"]
            assert exported["source_index"] == str(stored["source_index"])
            assert exported["source_timestamp_ms"] == str(stored["source_timestamp_ms"])
            assert exported["mode"] == stored["mode"]
            assert exported["model_id"] == (stored["model_id"] or "")
            assert exported["model_hash"] == (stored["model_hash"] or "")
            assert exported["config_id"] == stored["config_id"]
            assert exported["config_hash"] == stored["config_hash"]
            assert exported["first_seen_at"] == str(stored["first_seen_at"])
            assert exported["finalized_at"] == str(stored["finalized_at"])
            assert exported["valid_observation_count"] == str(stored["valid_observation_count"])
            assert exported["observation_duration"] == str(stored["observation_duration"])
            assert exported["cap_support"] == (str(stored["cap_support"]) if stored["cap_support"] is not None else "")
            assert exported["label_support"] == (str(stored["label_support"]) if stored["label_support"] is not None else "")
            assert exported["reasons"] == ";".join(json.loads(stored["reasons_json"]))
            assert json.loads(exported["extra"]) == json.loads(stored["extra_json"])
            assert exported["evidence_state"] == stored["evidence_state"]
            assert exported["evidence_path"] == (stored["evidence_relpath"] if stored["evidence_state"] == "AVAILABLE" else "")
            assert exported["evidence_sha256"] == (stored["evidence_sha256"] or "")
            assert exported["evidence_error"] == (stored["evidence_error"] or "")
            assert exported["created_at"] == stored["created_at"]

        final_run = client.get(f"/runs/{run_id}").json()
        counts = {outcome: sum(event["outcome"] == outcome for event in database_events) for outcome in ("PASS", "FAIL", "UNINSPECTED")}
        assert final_run["counters"] == {**counts, "PENDING": 0, "total": 5001}

        for offset in (0, 100):
            filtered = client.get(f"/runs/{run_id}/events", params={"outcome": "FAIL", "limit": 100, "offset": offset})
            expected = [event["event_id"] for event in database_events if event["outcome"] == "FAIL"][offset:offset + 100]
            assert [event["event_id"] for event in filtered.json()["items"]] == expected


def test_upload_identity_is_unique_and_real_run_never_falls_back(tmp_path):
    settings = _settings(tmp_path)
    payload = BytesIO()
    Image.new("RGB", (640, 360), "white").save(payload, format="PNG")
    with TestClient(create_app(settings)) as client:
        first = client.post("/assets", files={"file": ("../../sample.png", payload.getvalue(), "image/png")})
        second = client.post("/assets", files={"file": ("sample.png", payload.getvalue(), "image/png")})
        assert first.status_code == second.status_code == 201
        one, two = first.json(), second.json()
        assert one["kind"] == "image" and one["sha256"] == two["sha256"]
        assert one["asset_id"] != two["asset_id"]
        assert one["original_name"] == "sample.png"
        assert one["width"] == 640 and one["height"] == 360
        assert client.get("/models").json() == []
        blocked = client.post(
            "/runs",
            json={"source_type": "asset", "asset_id": one["asset_id"], "mode": "REAL", "model_id": "missing", "config_id": "default"},
            headers={"Idempotency-Key": "no-real-model"},
        )
        assert blocked.status_code == 422
        assert "never substituted" in blocked.json()["detail"].lower()
        assert client.get("/runs").json()["items"] == []


def test_real_image_controlled_provider_output_persists_events_evidence_and_csv(tmp_path, monkeypatch):
    from bottle_qc.domain.entities import Box, Detection
    from bottle_qc.pipeline.contracts import DetectionBatch
    from bottle_qc.pipeline import worker as worker_module

    class ControlledProvider:
        def detect(self, frame):
            detections = [
                Detection("bottle", Box(30, 30, 130, 330), 0.99, frame.index, frame.timestamp, "image-1"),
                Detection("cap", Box(55, 50, 105, 80), 0.99, frame.index, frame.timestamp),
                Detection("label", Box(45, 150, 115, 195), 0.99, frame.index, frame.timestamp),
                Detection("bottle", Box(230, 30, 330, 330), 0.99, frame.index, frame.timestamp, "image-2"),
                Detection("cap", Box(255, 50, 305, 80), 0.99, frame.index, frame.timestamp),
                Detection("bottle", Box(430, 30, 530, 330), 0.99, frame.index, frame.timestamp, "image-3"),
                Detection("bottle", Box(530, 30, 630, 330), 0.99, frame.index, frame.timestamp, "image-4"),
                Detection("cap", Box(555, 50, 605, 80), 0.99, frame.index, frame.timestamp),
            ]
            return DetectionBatch(
                tuple(detections),
                {
                    "image-1": (True, True), "image-2": (True, True),
                    "image-3": (True, True), "image-4": (True, True),
                },
            )

    settings = _settings(tmp_path)
    payload = BytesIO()
    Image.new("RGB", (640, 360), "white").save(payload, format="PNG")
    model = {
        "model_id": "controlled-output",
        "status": "AVAILABLE",
        "manifest_path": str(tmp_path / "controlled-manifest.json"),
        "weights_sha256": "a" * 64,
    }
    with TestClient(create_app(settings)) as client:
        monkeypatch.setattr(
            client.app.state.model_registry,
            "get",
            lambda model_id: dict(model) if model_id == model["model_id"] else None,
        )
        monkeypatch.setattr(worker_module, "open_yolo_provider", lambda *args, **kwargs: ControlledProvider())

        uploaded = client.post("/assets", files={"file": ("controlled.png", payload.getvalue(), "image/png")})
        assert uploaded.status_code == 201
        started = client.post(
            "/runs",
            json={
                "source_type": "asset",
                "asset_id": uploaded.json()["asset_id"],
                "mode": "REAL",
                "model_id": model["model_id"],
                "config_id": "default",
            },
            headers={"Idempotency-Key": "controlled-image-output"},
        )
        assert started.status_code == 202
        run_id = started.json()["run_id"]
        assert client.app.state.worker.wait(run_id, timeout=10)

        run = client.get(f"/runs/{run_id}").json()
        assert run["status"] == "COMPLETED"
        assert run["mode"] == "REAL"
        assert run["counters"] == {"PASS": 1, "FAIL": 0, "UNINSPECTED": 3, "PENDING": 0, "total": 4}
        events = client.get(f"/runs/{run_id}/events").json()["items"]
        assert {event["track_id"]: event["outcome"] for event in events} == {
            "image-1": "PASS",
            "image-2": "UNINSPECTED",
            "image-3": "UNINSPECTED",
            "image-4": "UNINSPECTED",
        }
        assert all(event["mode"] == "REAL" and event["evidence_state"] == "AVAILABLE" for event in events)
        assert all(
            client.get(f"/runs/{run_id}/events/{event['event_id']}/evidence").status_code == 200
            for event in events
        )
        preview = client.get(f"/runs/{run_id}/preview")
        assert preview.status_code == 200 and preview.content.startswith(b"\xff\xd8")
        rows = list(csv.DictReader(StringIO(client.get(f"/runs/{run_id}/report").text)))
        assert {row["track_id"]: (row["outcome"], row["mode"]) for row in rows} == {
            "image-1": ("PASS", "REAL"),
            "image-2": ("UNINSPECTED", "REAL"),
            "image-3": ("UNINSPECTED", "REAL"),
            "image-4": ("UNINSPECTED", "REAL"),
        }


def test_video_api_and_report_use_installed_cv2_decoder(tmp_path, monkeypatch):
    import cv2
    import numpy as np

    from bottle_qc.pipeline import sources, worker as worker_module
    from bottle_qc.pipeline.contracts import DetectionBatch

    assert sources.cv2 is cv2
    video_path = tmp_path / "runtime.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (640, 360))
    assert writer.isOpened()
    for index in range(3):
        writer.write(np.full((360, 640, 3), (30 + index * 20, 80, 120), dtype=np.uint8))
    writer.release()

    class EmptyProvider:
        def detect(self, frame):
            return DetectionBatch((), {})

    model = {
        "model_id": "video-decoder-integration",
        "status": "AVAILABLE",
        "manifest_path": str(tmp_path / "video-manifest.json"),
        "weights_sha256": "d" * 64,
    }
    with TestClient(create_app(_settings(tmp_path))) as client:
        monkeypatch.setattr(client.app.state.model_registry, "get", lambda _: dict(model))
        monkeypatch.setattr(worker_module, "open_yolo_provider", lambda *args, **kwargs: EmptyProvider())
        uploaded = client.post(
            "/assets",
            files={"file": ("runtime.avi", video_path.read_bytes(), "video/x-msvideo")},
        )
        assert uploaded.status_code == 201
        assert uploaded.json()["kind"] == "video"
        assert (uploaded.json()["width"], uploaded.json()["height"]) == (640, 360)

        started = client.post(
            "/runs",
            json={
                "source_type": "asset",
                "asset_id": uploaded.json()["asset_id"],
                "mode": "REAL",
                "model_id": model["model_id"],
                "config_id": "default",
            },
            headers={"Idempotency-Key": "actual-cv2-video"},
        )
        assert started.status_code == 202
        run_id = started.json()["run_id"]
        assert client.app.state.worker.wait(run_id, timeout=10)
        run = client.get(f"/runs/{run_id}").json()
        assert run["status"] == "COMPLETED"
        assert run["progress"]["frames_read"] == 3
        report = client.get(f"/runs/{run_id}/report")
        assert report.status_code == 200
        assert list(csv.DictReader(StringIO(report.text))) == []


def test_api_admission_uses_same_rounding_tolerance_as_engine(tmp_path, monkeypatch):
    from bottle_qc.pipeline.contracts import DetectionBatch
    from bottle_qc.pipeline import worker as worker_module

    class EmptyProvider:
        def detect(self, frame):
            return DetectionBatch((), {})

    settings = _settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        model = {
            "model_id": "controlled-geometry",
            "status": "AVAILABLE",
            "manifest_path": str(tmp_path / "controlled.json"),
            "weights_sha256": "c" * 64,
        }
        monkeypatch.setattr(client.app.state.model_registry, "get", lambda _: dict(model))
        monkeypatch.setattr(worker_module, "open_yolo_provider", lambda *args, **kwargs: EmptyProvider())

        def admit(size, key, geometry=None):
            payload = BytesIO()
            Image.new("RGB", size, "white").save(payload, format="PNG")
            uploaded = client.post("/assets", files={"file": (f"{key}.png", payload.getvalue(), "image/png")})
            assert uploaded.status_code == 201
            body = {
                "source_type": "asset",
                "asset_id": uploaded.json()["asset_id"],
                "mode": "REAL",
                "model_id": model["model_id"],
                "config_id": "default",
            }
            if geometry is not None:
                body["geometry"] = geometry
            return client.post("/runs", json=body, headers={"Idempotency-Key": key})

        rounded = admit((854, 480), "rounded-854x480")
        assert rounded.status_code == 202
        assert client.app.state.worker.wait(rounded.json()["run_id"], timeout=10)

        custom_geometry = {
            "roi": [0.05, 0.1, 0.95, 0.9],
            "entry_line": 0.2,
            "exit_line": 0.8,
            "direction": "LEFT_TO_RIGHT",
            "frame_aspect_ratio": 1.0,
        }
        boundary = admit((1005, 1000), "exact-ratio-boundary", custom_geometry)
        assert boundary.status_code == 202
        assert client.app.state.worker.wait(boundary.json()["run_id"], timeout=10)

        outside = admit((1006, 1000), "outside-ratio-boundary", custom_geometry)
        assert outside.status_code == 422


def test_video_tracker_failure_keeps_committed_events_and_evidence_on_last_processed_frame(tmp_path, monkeypatch):
    from bottle_qc.domain.entities import Box, Detection
    from bottle_qc.pipeline.contracts import DetectionBatch, Frame
    from bottle_qc.pipeline import worker as worker_module

    class ControlledSource:
        kind = "video"

        def __init__(self):
            self.frames = [
                Frame(index, index / 10, Image.new("RGB", (854, 480), (index * 20, 11, 17)), f"frame-{index}")
                for index in range(8)
            ]

        def read(self):
            return self.frames.pop(0) if self.frames else None

        def close(self):
            for frame in self.frames:
                frame.image.close()

    class FailingProvider:
        def detect(self, frame):
            if frame.index == 7:
                raise RuntimeError(
                    "REAL video tracking returned a bottle detection without a tracker ID "
                    "at source frame 7; verify tracker output and model configuration"
                )
            progress = (.1, .3, .4, .5, .6, .7, .82)[frame.index]
            center = progress * frame.image.width
            stationary = .4 * frame.image.width
            detections = [
                Detection("bottle", Box(center - 30, 140, center + 30, 340), .99, frame.index, frame.timestamp, "1"),
                Detection("cap", Box(center - 12, 138, center + 12, 160), .99, frame.index, frame.timestamp),
                Detection("label", Box(center - 20, 190, center + 20, 225), .99, frame.index, frame.timestamp),
            ]
            if frame.index < 6:
                detections.append(
                    Detection("bottle", Box(stationary - 22, 35, stationary + 22, 110), .99, frame.index, frame.timestamp, "2")
                )
            uncertain = frozenset({"2"}) if frame.index == 6 else frozenset()
            return DetectionBatch(tuple(detections), {"1": (True, True), "2": (True, True)}, uncertain)

    settings = _settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        repository = client.app.state.repository
        asset = repository.create_asset(
            sha256="a" * 64,
            kind="video",
            original_name="controlled.mp4",
            stored_name="controlled.mp4",
            size_bytes=1,
            width=854,
            height=480,
        )
        values, config_hash, _ = settings.load_config("default")
        run = repository.create_run({
            "source_type": "asset",
            "asset_id": asset["asset_id"],
            "scenario": None,
            "kind": "video",
            "mode": "REAL",
            "model_id": "controlled",
            "model_hash": "b" * 64,
            "config_id": "default",
            "config_hash": config_hash,
            "config_snapshot": json.dumps(values, sort_keys=True),
            "idempotency_key": "tracker-id-failure",
            "request_hash": "tracker-id-failure",
        })
        monkeypatch.setattr(worker_module, "VideoFileSource", lambda *args, **kwargs: ControlledSource())
        provider_options = {}

        def open_controlled_provider(*args, **kwargs):
            provider_options.update(kwargs)
            return FailingProvider()

        monkeypatch.setattr(worker_module, "open_yolo_provider", open_controlled_provider)
        monkeypatch.setattr(client.app.state.model_registry, "get", lambda _: {
            "status": "AVAILABLE",
            "manifest_path": str(tmp_path / "controlled.json"),
            "weights_sha256": "b" * 64,
        })
        monkeypatch.setattr(client.app.state.worker.asset_store, "path_for", lambda _: tmp_path / "controlled.mp4")

        client.app.state.worker.start(run)
        assert client.app.state.worker.wait(run["run_id"], timeout=10)
        failed = client.get(f"/runs/{run['run_id']}").json()
        events = client.get(f"/runs/{run['run_id']}/events").json()["items"]

        assert failed["status"] == "FAILED"
        assert "without a tracker ID" in failed["error"]
        assert provider_options["tracker"] is True
        assert provider_options["tracker_iou_threshold"] == .2
        assert provider_options["tracker_ambiguity_margin"] == .025
        assert provider_options["maximum_missed_frames"] == 2
        assert provider_options["maximum_tracks"] == 4096
        committed_event = next(event for event in events if event["track_id"] == "1")
        assert committed_event["outcome"] == "PASS"
        assert committed_event["source_index"] == 6
        assert committed_event["evidence_state"] == "AVAILABLE"
        assert client.get(
            f"/runs/{run['run_id']}/events/{committed_event['event_id']}/evidence"
        ).status_code == 200
        pending_event = next(event for event in events if event["track_id"] == "2")
        assert pending_event["source_index"] == 6
        assert "IDENTITY_UNCERTAIN" in pending_event["reasons"]
        assert pending_event["evidence_state"] == "AVAILABLE"
        evidence = client.get(f"/runs/{run['run_id']}/events/{pending_event['event_id']}/evidence")
        with Image.open(BytesIO(evidence.content)) as image:
            assert abs(image.convert("RGB").getpixel((500, 400))[0] - 120) <= 3


def test_upload_size_and_decoder_failures_leave_no_registered_asset(tmp_path):
    from bottle_qc.settings import Settings

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_text = (PROJECT / "configs" / "default.toml").read_text(encoding="utf-8")
    (config_dir / "default.toml").write_text(config_text.replace("max_upload_bytes = 268435456", "max_upload_bytes = 32"), encoding="utf-8")
    settings = Settings(
        root_dir=tmp_path,
        data_dir=tmp_path / "runtime",
        models_dir=tmp_path / "models",
        configs_dir=config_dir,
    )
    with TestClient(create_app(settings)) as client:
        oversized = client.post("/assets", files={"file": ("too-large.png", b"x" * 33, "image/png")})
        malformed = client.post("/assets", files={"file": ("../../invalid.png", b"not image or video", "image/png")})
        assert oversized.status_code == 413
        assert malformed.status_code == 415
        count = client.app.state.repository.db.fetchone("SELECT COUNT(*) AS count FROM assets")["count"]
        assert count == 0
        assert not list(settings.assets_dir.glob("*"))


def test_active_admission_and_restart_reconciliation(tmp_path):
    settings = _settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        config, config_hash, _ = settings.load_config("default")
        snapshot = __import__("json").dumps(config, sort_keys=True)
        active = client.app.state.repository.create_run({
            "source_type": "simulation", "scenario": "phase-2-four-bottle-demo", "kind": "video",
            "mode": "SIMULATION", "config_id": "default", "config_hash": config_hash,
            "config_snapshot": snapshot, "idempotency_key": "already-active", "request_hash": "active",
        })
        busy = client.post("/runs", json=_simulation_request(), headers={"Idempotency-Key": "another-run"})
        assert busy.status_code == 409
    with TestClient(create_app(settings)) as restarted:
        recovered = restarted.get(f"/runs/{active['run_id']}").json()
        assert recovered["status"] == "INTERRUPTED"
        assert restarted.post(f"/runs/{active['run_id']}/stop").json()["idempotent_stop"] is True


def test_failed_event_commit_creates_no_event_or_evidence(tmp_path):
    settings = _settings(tmp_path)
    repository = Repository(settings.database_path)
    repository.close()
    connection = sqlite3.connect(settings.database_path)
    connection.execute(
        "CREATE TRIGGER fail_event_insert BEFORE INSERT ON events BEGIN SELECT RAISE(ABORT, 'injected event commit failure'); END"
    )
    connection.commit()
    connection.close()
    with TestClient(create_app(settings)) as client:
        started = client.post("/runs", json=_simulation_request(), headers={"Idempotency-Key": "db-failure"})
        run_id = started.json()["run_id"]
        assert client.app.state.worker.wait(run_id, timeout=10)
        run = client.get(f"/runs/{run_id}").json()
        assert run["status"] == "FAILED"
        assert run["counters"]["total"] == 0
        assert client.get(f"/runs/{run_id}/events").json()["items"] == []
        assert not list(settings.evidence_dir.rglob("*.jpg"))


def test_evidence_state_failure_does_not_duplicate_or_hide_event_count(tmp_path):
    settings = _settings(tmp_path)
    repository = Repository(settings.database_path)
    repository.close()
    connection = sqlite3.connect(settings.database_path)
    connection.execute(
        "CREATE TRIGGER fail_evidence_available BEFORE UPDATE OF evidence_state ON events "
        "WHEN NEW.evidence_state='AVAILABLE' BEGIN SELECT RAISE(ABORT, 'injected evidence state failure'); END"
    )
    connection.commit()
    connection.close()
    with TestClient(create_app(settings)) as client:
        started = client.post("/runs", json=_simulation_request(), headers={"Idempotency-Key": "evidence-failure"})
        run_id = started.json()["run_id"]
        assert client.app.state.worker.wait(run_id, timeout=10)
        run = client.get(f"/runs/{run_id}").json()
        events = client.get(f"/runs/{run_id}/events").json()["items"]
        assert run["status"] == "COMPLETED"
        assert run["counters"]["total"] == len(events) == 4
        assert all(event["evidence_state"] == "FAILED" and event["evidence_path"] is None for event in events)
        assert all(client.get(f"/runs/{run_id}/events/{event['event_id']}/evidence").status_code == 410 for event in events)


def test_stop_during_source_read_finalizes_stopped_and_replays_idempotently(tmp_path, monkeypatch):
    from bottle_qc.pipeline import worker as worker_module

    entered_read = threading.Event()
    release_read = threading.Event()

    class BlockingSimulationSource:
        kind = "video"

        def read(self):
            entered_read.set()
            release_read.wait(timeout=5)
            return None

        def close(self):
            release_read.set()

    monkeypatch.setattr(worker_module, "SimulationFrameSource", BlockingSimulationSource)
    with TestClient(create_app(_settings(tmp_path))) as client:
        started = client.post("/runs", json=_simulation_request(), headers={"Idempotency-Key": "stop-in-read"})
        run_id = started.json()["run_id"]
        assert entered_read.wait(timeout=5)
        first_stop = client.post(f"/runs/{run_id}/stop")
        replay_stop = client.post(f"/runs/{run_id}/stop")
        assert first_stop.json()["stop_requested"] is True
        assert replay_stop.json()["stop_requested"] is True
        release_read.set()
        assert client.app.state.worker.wait(run_id, timeout=5)
        final = client.get(f"/runs/{run_id}").json()
        assert final["status"] == "STOPPED"
        assert final["counters"]["total"] == 0


def test_worker_keeps_only_one_run_of_transient_bookkeeping(tmp_path):
    with TestClient(create_app(_settings(tmp_path))) as client:
        worker = client.app.state.worker
        for index in range(4):
            response = client.post(
                "/runs",
                json=_simulation_request(),
                headers={"Idempotency-Key": f"bounded-bookkeeping-{index}"},
            )
            run_id = response.json()["run_id"]
            assert worker.wait(run_id, timeout=5)

        assert len(worker._threads) <= 1
        assert len(worker._stops) <= 1
        assert len(worker._progress) <= 1
        assert len(worker._previews) <= 1
        assert len(client.get("/runs").json()["items"]) == 4


def test_frame_limit_fails_visibly_and_finalizes_pending_products_as_uninspected(tmp_path):
    from bottle_qc.settings import Settings

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    source_config = (PROJECT / "configs" / "default.toml").read_text(encoding="utf-8")
    bounded_config = source_config.replace("maximum_frames_per_run = 100000", "maximum_frames_per_run = 3")
    (config_dir / "default.toml").write_text(bounded_config, encoding="utf-8")
    settings = Settings(
        root_dir=tmp_path,
        data_dir=tmp_path / "runtime",
        models_dir=tmp_path / "models",
        configs_dir=config_dir,
    )

    with TestClient(create_app(settings)) as client:
        started = client.post("/runs", json=_simulation_request(), headers={"Idempotency-Key": "frame-limit"})
        run_id = started.json()["run_id"]
        assert client.app.state.worker.wait(run_id, timeout=5)
        run = client.get(f"/runs/{run_id}").json()
        events = client.get(f"/runs/{run_id}/events").json()["items"]
        assert run["status"] == "FAILED"
        assert "maximum_frames_per_run" in run["error"]
        assert run["progress"]["frames_read"] == 3
        assert run["counters"]["UNINSPECTED"] == len(events) == 2
        assert {event["outcome"] for event in events} == {"UNINSPECTED"}
