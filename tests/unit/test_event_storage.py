import sqlite3
import uuid

import pytest

from bottle_qc.domain.entities import RunMode
from bottle_qc.domain.inspection import InspectionEngine
from bottle_qc.storage.repository import Repository
from bottle_qc.storage.evidence import EvidenceStore


def test_temporal_event_metrics_survive_storage_and_duplicate_insert_is_rejected(tmp_path, config, step):
    repository = Repository(tmp_path / "events.sqlite3")
    try:
        run = repository.create_run({
            "source_type": "simulation", "scenario": "phase-1-domain-test", "kind": "video",
            "mode": "SIMULATION", "config_id": "default", "config_hash": config.fingerprint,
            "config_snapshot": "{}", "idempotency_key": "one-run", "request_hash": "test",
        })
        engine = InspectionEngine(run_id=run["run_id"], config=config, mode=RunMode.SIMULATION)
        for index, progress in enumerate((.1, .3, .4, .5, .6, .7)):
            step(engine, index, progress)
        event, = step(engine, 6, .82)
        stored = repository.insert_event(event)
        assert stored["track_id"] == "1" and stored["outcome"] == "PASS"
        assert stored["valid_observation_count"] == 5
        assert stored["cap_support"] == stored["label_support"] == 1
        assert stored["first_seen_at"] == 0 and stored["finalized_at"] == pytest.approx(.6)
        assert stored["observation_duration"] == pytest.approx(.4)
        assert stored["extra"]["cap_observation_count"] == 5
        assert stored["evidence_state"] == "PENDING" and stored["evidence_path"] is None
        with pytest.raises(sqlite3.IntegrityError):
            repository.insert_event(event)
        assert repository.summary(run["run_id"])["total"] == 1
    finally:
        repository.close()


def test_evidence_lookup_rejects_event_path_traversal(tmp_path):
    evidence = EvidenceStore(tmp_path / "evidence")
    run_id = str(uuid.uuid4())
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"not evidence")

    with pytest.raises(ValueError, match="Invalid event identity"):
        evidence.path_for(run_id, "../../outside")
    with pytest.raises(ValueError, match="Invalid event identity"):
        evidence.write(run_id, "../../outside", b"not an image")
    assert outside.read_bytes() == b"not evidence"
