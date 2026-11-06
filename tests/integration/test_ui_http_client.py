from __future__ import annotations

import csv
from io import BytesIO, StringIO
from pathlib import Path
import socket
import threading
import time

from PIL import Image
import uvicorn

from bottle_qc.api.app import create_app
from bottle_qc.settings import Settings
from bottle_qc.ui.api_client import APIClient


PROJECT = Path(__file__).resolve().parents[2]


def test_ui_http_client_uses_local_api_for_upload_run_results_and_export(tmp_path):
    settings = Settings(
        root_dir=tmp_path,
        data_dir=tmp_path / "runtime",
        models_dir=tmp_path / "models",
        configs_dir=PROJECT / "configs",
    )
    server = uvicorn.Server(uvicorn.Config(create_app(settings), log_level="error", lifespan="on"))
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started

        with APIClient(f"http://127.0.0.1:{port}", timeout=2) as client:
            assert client.health() == {"status": "ok"}
            assert client.readiness()["status"] == "ready"
            assert [item["config_id"] for item in client.configs()] == ["default"]
            assert client.models() == []

            payload = BytesIO()
            Image.new("RGB", (64, 36), "white").save(payload, format="PNG")
            uploaded = client.upload_asset("sample.png", payload.getvalue())
            assert uploaded["kind"] == "image" and uploaded["width"] == 64

            started = client.start_run(
                {
                    "source_type": "simulation",
                    "scenario": "phase-2-four-bottle-demo",
                    "mode": "SIMULATION",
                    "config_id": "default",
                },
                idempotency_key="ui-http-contract",
            )
            run_id = started["run_id"]
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                run = client.run(run_id)
                if run["status"] != "RUNNING":
                    break
                time.sleep(0.02)
            assert run["status"] == "COMPLETED"
            assert run["counters"] == {"PASS": 1, "FAIL": 2, "UNINSPECTED": 1, "PENDING": 0, "total": 4}
            assert client.events(run_id)["summary"] == run["counters"]
            assert client.preview(run_id).startswith(b"\xff\xd8")

            event = client.events(run_id)["items"][0]
            assert client.evidence(run_id, event["event_id"]).startswith(b"\xff\xd8")
            rows = list(csv.DictReader(StringIO(client.report(run_id).decode("utf-8"))))
            assert len(rows) == run["counters"]["total"]
            assert {row["outcome"] for row in rows} == {"PASS", "FAIL", "UNINSPECTED"}
            assert client.runs()["items"][0]["run_id"] == run_id
            assert client.stop_run(run_id)["idempotent_stop"] is True
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()
    assert not thread.is_alive()
