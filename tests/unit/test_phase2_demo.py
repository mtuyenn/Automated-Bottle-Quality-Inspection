import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def test_simulation_demo_persists_labeled_overlay_and_expected_events(tmp_path):
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    source_path = str(root / "src")
    env["PYTHONPATH"] = source_path + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-B", str(root / "scripts" / "demo_domain.py"), "--output-dir", str(tmp_path)],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["mode"] == "SIMULATION"
    assert result["summary"] == {"PASS": 1, "FAIL": 2, "UNINSPECTED": 1, "total": 4}
    assert len(result["events"]) == 4
    assert all(event["mode"] == "SIMULATION" for event in result["events"])
    assert {event["track_id"]: event["outcome"] for event in result["events"]} == {
        "1": "PASS", "2": "FAIL", "3": "FAIL", "4": "UNINSPECTED"
    }
    overlay = tmp_path / result["artifacts"]["overlay"]
    assert overlay.is_file()
    assert hashlib.sha256(overlay.read_bytes()).hexdigest() == result["artifacts"]["overlay_sha256"]
    evidence = result["artifacts"]["evidence"]
    assert len(evidence) == 4
    for item in evidence:
        path = tmp_path / item["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
