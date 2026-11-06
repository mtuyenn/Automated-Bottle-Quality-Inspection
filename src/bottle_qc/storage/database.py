from __future__ import annotations

import sqlite3
from pathlib import Path
import threading


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS assets (
    asset_id TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('image','video')),
    original_name TEXT NOT NULL,
    stored_name TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL CHECK(source_type IN ('asset','simulation')),
    asset_id TEXT REFERENCES assets(asset_id),
    scenario TEXT,
    kind TEXT NOT NULL CHECK(kind IN ('image','video')),
    mode TEXT NOT NULL CHECK(mode IN ('SIMULATION','REAL')),
    model_id TEXT,
    model_hash TEXT,
    config_id TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    config_snapshot TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('RUNNING','COMPLETED','STOPPED','FAILED','INTERRUPTED')),
    idempotency_key TEXT NOT NULL UNIQUE,
    request_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    finished_at TEXT,
    error TEXT,
    CHECK((source_type = 'asset' AND asset_id IS NOT NULL AND scenario IS NULL) OR
          (source_type = 'simulation' AND asset_id IS NULL AND scenario IS NOT NULL))
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_active_run ON runs(status) WHERE status = 'RUNNING';
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    source_item_id TEXT NOT NULL,
    track_id TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK(outcome IN ('PASS','FAIL','UNINSPECTED')),
    reasons_json TEXT NOT NULL,
    source_index INTEGER,
    source_timestamp_ms REAL,
    mode TEXT NOT NULL CHECK(mode IN ('SIMULATION','REAL')),
    model_id TEXT,
    model_hash TEXT,
    config_id TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    first_seen_at REAL NOT NULL,
    finalized_at REAL NOT NULL,
    valid_observation_count INTEGER NOT NULL,
    observation_duration REAL NOT NULL,
    cap_support REAL,
    label_support REAL,
    extra_json TEXT NOT NULL,
    evidence_state TEXT NOT NULL CHECK(evidence_state IN ('PENDING','AVAILABLE','FAILED')),
    evidence_relpath TEXT,
    evidence_sha256 TEXT,
    evidence_error TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, source_item_id)
);
CREATE INDEX IF NOT EXISTS ix_events_run ON events(run_id, source_index, event_id);
"""


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None, timeout=15)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout = 15000")
        self._connection.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def execute(self, sql: str, values: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._connection.execute(sql, values)

    def fetchone(self, sql: str, values: tuple = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._connection.execute(sql, values).fetchone()

    def fetchall(self, sql: str, values: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._connection.execute(sql, values).fetchall()

    def transaction(self):
        return _Transaction(self)


class _Transaction:
    def __init__(self, database: Database):
        self.db = database

    def __enter__(self):
        self.db._lock.acquire()
        self.db._connection.execute("BEGIN IMMEDIATE")
        return self.db._connection

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.db._connection.execute("ROLLBACK" if exc_type else "COMMIT")
        finally:
            self.db._lock.release()
