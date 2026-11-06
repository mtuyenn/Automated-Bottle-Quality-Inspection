from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any
import uuid

from bottle_qc.domain.entities import InspectionEvent
from bottle_qc.storage.database import Database


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Repository:
    def __init__(self, database_path: Path):
        self.db = Database(database_path)

    def close(self) -> None:
        self.db.close()

    def create_asset(self, *, sha256: str, kind: str, original_name: str, stored_name: str, size_bytes: int, width: int, height: int) -> dict[str, Any]:
        asset_id = str(uuid.uuid4())
        created_at = utc_now()
        self.db.execute(
            "INSERT INTO assets VALUES(?,?,?,?,?,?,?,?,?)",
            (asset_id, sha256, kind, original_name, stored_name, size_bytes, width, height, created_at),
        )
        return self.get_asset(asset_id)

    def get_asset(self, asset_id: str) -> dict[str, Any] | None:
        row = self.db.fetchone("SELECT * FROM assets WHERE asset_id=?", (asset_id,))
        return dict(row) if row else None

    def find_idempotent_run(self, key: str) -> dict[str, Any] | None:
        row = self.db.fetchone("SELECT * FROM runs WHERE idempotency_key=?", (key,))
        return dict(row) if row else None

    def active_run(self) -> dict[str, Any] | None:
        row = self.db.fetchone("SELECT * FROM runs WHERE status='RUNNING' LIMIT 1")
        return dict(row) if row else None

    def create_run(self, values: dict[str, Any]) -> dict[str, Any]:
        run_id = values.get("run_id") or str(uuid.uuid4())
        created_at = utc_now()
        with self.db.transaction() as connection:
            connection.execute(
                """INSERT INTO runs(run_id,source_type,asset_id,scenario,kind,mode,model_id,model_hash,
                   config_id,config_hash,config_snapshot,status,idempotency_key,request_hash,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id, values["source_type"], values.get("asset_id"), values.get("scenario"), values["kind"],
                    values["mode"], values.get("model_id"), values.get("model_hash"), values["config_id"],
                    values["config_hash"], values["config_snapshot"], "RUNNING", values["idempotency_key"],
                    values["request_hash"], created_at,
                ),
            )
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self.db.fetchone("SELECT * FROM runs WHERE run_id=?", (run_id,))
        return dict(row) if row else None

    def set_run_status(self, run_id: str, status: str, error: str | None = None) -> None:
        finished = utc_now() if status != "RUNNING" else None
        self.db.execute("UPDATE runs SET status=?,finished_at=?,error=? WHERE run_id=?", (status, finished, error, run_id))

    def insert_event(self, event: InspectionEvent) -> dict[str, Any]:
        if event.outcome.value == "PENDING":
            raise ValueError("PENDING is not a terminal event outcome")
        relpath = f"{event.run_id}/{event.event_id}.jpg"
        self.db.execute(
            """INSERT INTO events(event_id,run_id,source_item_id,track_id,outcome,reasons_json,source_index,
               source_timestamp_ms,mode,model_id,model_hash,config_id,config_hash,first_seen_at,finalized_at,
               valid_observation_count,observation_duration,cap_support,label_support,extra_json,
               evidence_state,evidence_relpath,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'PENDING',?,?)""",
            (
                event.event_id, event.run_id, event.source_item_id, event.track_id, event.outcome.value,
                json.dumps(list(event.reason_codes), separators=(",", ":")), event.source_index,
                event.source_timestamp_ms, event.mode.value, event.model_id, event.model_hash, event.config_id,
                event.config_hash, event.first_seen_at, event.finalized_at, event.valid_observation_count,
                event.observation_duration, event.cap_support, event.label_support,
                json.dumps(event.extra, separators=(",", ":")), relpath, event.created_at or utc_now(),
            ),
        )
        return self.get_event(event.run_id, event.event_id)

    def get_event(self, run_id: str, event_id: str) -> dict[str, Any] | None:
        row = self.db.fetchone("SELECT * FROM events WHERE run_id=? AND event_id=?", (run_id, event_id))
        return self._event_dict(dict(row)) if row else None

    def update_evidence(self, event_id: str, *, state: str, sha256: str | None = None, error: str | None = None) -> None:
        self.db.execute(
            "UPDATE events SET evidence_state=?,evidence_sha256=?,evidence_error=? WHERE event_id=?",
            (state, sha256, error, event_id),
        )

    def list_events(self, run_id: str, *, limit: int = 5000, offset: int = 0, outcome: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM events WHERE run_id=?"
        values: tuple[Any, ...] = (run_id,)
        if outcome is not None:
            query += " AND outcome=?"
            values += (outcome,)
        query += " ORDER BY COALESCE(source_index,-1),created_at,event_id LIMIT ? OFFSET ?"
        values += (limit, offset)
        rows = self.db.fetchall(
            query,
            values,
        )
        return [self._event_dict(dict(row)) for row in rows]

    def list_runs(self, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        rows = self.db.fetchall("SELECT * FROM runs ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset))
        return [dict(row) for row in rows]

    def summary(self, run_id: str) -> dict[str, int]:
        counts = {"PASS": 0, "FAIL": 0, "UNINSPECTED": 0, "PENDING": 0, "total": 0}
        rows = self.db.fetchall("SELECT outcome,COUNT(*) AS n FROM events WHERE run_id=? GROUP BY outcome", (run_id,))
        for row in rows:
            counts[row["outcome"]] = row["n"]
            counts["total"] += row["n"]
        return counts

    def reconcile(self, evidence_root: Path) -> list[str]:
        with self.db.transaction() as connection:
            active = connection.execute("SELECT run_id FROM runs WHERE status='RUNNING'").fetchall()
            for row in active:
                connection.execute(
                    "UPDATE runs SET status='INTERRUPTED',finished_at=?,error=? WHERE run_id=?",
                    (utc_now(), "Process restarted while run was active", row["run_id"]),
                )
            pending = connection.execute("SELECT * FROM events WHERE evidence_state='PENDING'").fetchall()
        missing = []
        for row in pending:
            path = (evidence_root / row["evidence_relpath"]).resolve()
            if path.parent != (evidence_root / row["run_id"]).resolve() or not path.is_file():
                self.update_evidence(row["event_id"], state="FAILED", error="Evidence file missing during recovery")
                missing.append(row["event_id"])
                continue
            try:
                from PIL import Image

                with Image.open(path) as image:
                    image.verify()
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.update_evidence(row["event_id"], state="AVAILABLE", sha256=digest)
            except Exception as exc:
                self.update_evidence(row["event_id"], state="FAILED", error=f"Evidence recovery failed: {exc}")
                missing.append(row["event_id"])
        return missing

    @staticmethod
    def _event_dict(row: dict[str, Any]) -> dict[str, Any]:
        row["reasons"] = json.loads(row.pop("reasons_json"))
        row["extra"] = json.loads(row.pop("extra_json"))
        row["evidence_path"] = row.pop("evidence_relpath") if row["evidence_state"] == "AVAILABLE" else None
        return row
