"""Anonymous visitor tracking backed by transactional SQLite storage."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from data import GPUS, MODELS
from state import PlannerState


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _env_nonnegative_int(name: str, default: int = 0) -> int:
    try:
        return max(0, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _json_default(obj):
    if isinstance(obj, (frozenset, set)):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _json_dump(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def _serialize_state(state: Optional[PlannerState]) -> Optional[dict]:
    return asdict(state) if state is not None else None


def _gpu_summary(state: Optional[PlannerState]) -> str:
    if state is None or not state.gpus:
        return ""
    return ", ".join(f"{gp.count}x {GPUS[gp.gpu_type].name}" for gp in state.gpus)


def _model_summary(state: Optional[PlannerState]) -> str:
    if state is None or not state.models:
        return ""
    return ", ".join(
        f"{MODELS[a.model_key].name} ({a.gpu_count} GPU, {a.prec.upper()}, D {a.tp}/{a.pp}/{a.dp})"
        for a in state.models
    )


def _state_summary(state: Optional[PlannerState]) -> dict:
    if state is None:
        return {"mode": None, "gpu_summary": "", "model_summary": "", "gpu_pool_count": 0, "model_count": 0}
    return {
        "mode": state.mode,
        "gpu_summary": _gpu_summary(state),
        "model_summary": _model_summary(state),
        "gpu_pool_count": len(state.gpus),
        "model_count": len(state.models),
    }


class SnapshotStore:
    """SQLite snapshot store with one-time import of the legacy JSON log.

    Passing the historical ``planner_snapshots.json`` path remains supported: the
    database is created beside it as ``planner_snapshots.sqlite3`` and the JSON
    file is imported once. Full A/B state payloads are retained. Retention is
    unlimited unless explicitly configured through environment variables.
    """

    def __init__(self, path: Path, legacy_path: Optional[Path] = None):
        path = Path(path)
        if path.suffix.lower() == ".json":
            self.path = path.with_suffix(".sqlite3")
            self.legacy_path = path
        else:
            self.path = path
            self.legacy_path = Path(legacy_path) if legacy_path else path.with_name("planner_snapshots.json")
        self.retention_days = _env_nonnegative_int("PLANNER_SNAPSHOT_RETENTION_DAYS", 0)
        self.max_per_tab = _env_nonnegative_int("PLANNER_SNAPSHOT_MAX_PER_TAB", 0)
        self._lock = threading.RLock()
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA secure_delete = ON")
        for candidate in (self.path, Path(f"{self.path}-wal"), Path(f"{self.path}-shm")):
            try:
                candidate.chmod(0o600)
            except FileNotFoundError:
                pass
        return connection

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            with self._connect() as connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS snapshots (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        snapshot_id TEXT NOT NULL UNIQUE,
                        visitor_id TEXT NOT NULL,
                        tab_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        last_seen TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        path TEXT NOT NULL,
                        state_hash TEXT NOT NULL,
                        summary_json TEXT NOT NULL,
                        panel_a_json TEXT NOT NULL,
                        panel_b_json TEXT
                    );
                    CREATE INDEX IF NOT EXISTS snapshots_last_seen_idx
                        ON snapshots(last_seen DESC);
                    CREATE INDEX IF NOT EXISTS snapshots_visitor_tab_idx
                        ON snapshots(visitor_id, tab_id, sequence DESC);
                    """
                )
                self._migrate_legacy(connection)
            self._initialized = True

    def _migrate_legacy(self, connection: sqlite3.Connection) -> None:
        connection.execute("BEGIN IMMEDIATE")
        if self.legacy_path.exists():
            try:
                self.legacy_path.chmod(0o600)
            except OSError:
                pass
        marker = connection.execute(
            "SELECT value FROM metadata WHERE key = 'legacy_json_imported'"
        ).fetchone()
        if marker is not None or not self.legacy_path.exists():
            return

        try:
            payload = json.loads(self.legacy_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("visitors", {}), dict):
                raise ValueError("legacy snapshot root must contain a visitors object")
        except (json.JSONDecodeError, OSError, ValueError):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            quarantine = self.legacy_path.with_name(
                f"{self.legacy_path.stem}.corrupt-{stamp}{self.legacy_path.suffix}"
            )
            self.legacy_path.replace(quarantine)
            try:
                quarantine.chmod(0o600)
            except OSError:
                pass
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('legacy_json_imported', ?)",
                (f"quarantined:{quarantine.name}",),
            )
            return

        imported = 0
        for visitor_id, visitor in payload.get("visitors", {}).items():
            if not isinstance(visitor, dict):
                continue
            for tab_id, tab in visitor.get("tabs", {}).items():
                if not isinstance(tab, dict):
                    continue
                for snapshot in tab.get("snapshots", []):
                    if not isinstance(snapshot, dict):
                        continue
                    panel_a = snapshot.get("panel_a")
                    if not isinstance(panel_a, dict):
                        continue
                    panel_b = snapshot.get("panel_b")
                    summary = snapshot.get("summary") if isinstance(snapshot.get("summary"), dict) else {}
                    created_at = str(snapshot.get("created_at") or _utc_now())
                    last_seen = str(snapshot.get("last_seen") or created_at)
                    state_hash = str(snapshot.get("state_hash") or hashlib.sha256(
                        _json_dump({"panel_a": panel_a, "panel_b": panel_b}).encode("utf-8")
                    ).hexdigest())
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO snapshots(
                            snapshot_id, visitor_id, tab_id, created_at, last_seen,
                            reason, path, state_hash, summary_json, panel_a_json, panel_b_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(snapshot.get("snapshot_id") or uuid.uuid4()),
                            str(visitor_id), str(tab_id), created_at, last_seen,
                            str(snapshot.get("reason") or "legacy"), str(snapshot.get("path") or "/"),
                            state_hash, _json_dump(summary), _json_dump(panel_a),
                            _json_dump(panel_b) if panel_b is not None else None,
                        ),
                    )
                    imported += 1
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('legacy_json_imported', ?)",
            (f"imported:{imported}:{_utc_now()}",),
        )

    def _apply_retention(self, connection: sqlite3.Connection, visitor_id: str, tab_id: str) -> None:
        if self.retention_days > 0:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=self.retention_days)).isoformat(timespec="seconds")
            connection.execute("DELETE FROM snapshots WHERE last_seen < ?", (cutoff,))
        if self.max_per_tab > 0:
            connection.execute(
                """
                DELETE FROM snapshots
                WHERE visitor_id = ? AND tab_id = ? AND sequence NOT IN (
                    SELECT sequence FROM snapshots
                    WHERE visitor_id = ? AND tab_id = ?
                    ORDER BY sequence DESC LIMIT ?
                )
                """,
                (visitor_id, tab_id, visitor_id, tab_id, self.max_per_tab),
            )

    def record_snapshot(
        self, *, visitor_id: str, tab_id: str, reason: str, path: str,
        state_a: PlannerState, state_b: Optional[PlannerState],
    ) -> None:
        self._ensure_initialized()
        now = _utc_now()
        panel_a = _serialize_state(state_a)
        panel_b = _serialize_state(state_b)
        state_hash = hashlib.sha256(_json_dump({"panel_a": panel_a, "panel_b": panel_b}).encode("utf-8")).hexdigest()
        summary_a = _state_summary(state_a)
        summary_b = _state_summary(state_b)
        summary = {
            "mode": summary_a["mode"],
            "compare_enabled": state_b is not None,
            "panel_a": summary_a,
            "panel_b": summary_b,
        }

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            latest = connection.execute(
                """SELECT sequence, state_hash FROM snapshots
                   WHERE visitor_id = ? AND tab_id = ? ORDER BY sequence DESC LIMIT 1""",
                (visitor_id, tab_id),
            ).fetchone()
            if latest is not None and latest["state_hash"] == state_hash:
                connection.execute(
                    "UPDATE snapshots SET last_seen = ?, reason = ?, path = ?, summary_json = ? WHERE sequence = ?",
                    (now, reason, path, _json_dump(summary), latest["sequence"]),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO snapshots(
                        snapshot_id, visitor_id, tab_id, created_at, last_seen,
                        reason, path, state_hash, summary_json, panel_a_json, panel_b_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()), visitor_id, tab_id, now, now, reason, path, state_hash,
                        _json_dump(summary), _json_dump(panel_a), _json_dump(panel_b) if panel_b is not None else None,
                    ),
                )
            self._apply_retention(connection, visitor_id, tab_id)

    @staticmethod
    def _decode_row(row: sqlite3.Row) -> dict:
        return {
            "visitor_id": row["visitor_id"],
            "tab_id": row["tab_id"],
            "snapshot_id": row["snapshot_id"],
            "created_at": row["created_at"],
            "last_seen": row["last_seen"],
            "reason": row["reason"],
            "path": row["path"],
            "summary": json.loads(row["summary_json"]),
            "panel_a": json.loads(row["panel_a_json"]),
            "panel_b": json.loads(row["panel_b_json"]) if row["panel_b_json"] else None,
        }

    def list_snapshots(
        self, *, limit: Optional[int] = None, offset: int = 0, visitor_id: Optional[str] = None,
    ) -> list[dict]:
        self._ensure_initialized()
        clauses, params = [], []
        if visitor_id is not None:
            clauses.append("visitor_id = ?")
            params.append(visitor_id)
        sql = "SELECT * FROM snapshots"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY last_seen DESC, sequence DESC"
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend((max(0, int(limit)), max(0, int(offset))))
        with self._connect() as connection:
            return [self._decode_row(row) for row in connection.execute(sql, params).fetchall()]

    def count_snapshots(self, visitor_id: Optional[str] = None) -> int:
        self._ensure_initialized()
        sql, params = "SELECT COUNT(*) FROM snapshots", ()
        if visitor_id is not None:
            sql, params = sql + " WHERE visitor_id = ?", (visitor_id,)
        with self._connect() as connection:
            return int(connection.execute(sql, params).fetchone()[0])

    def delete_visitor(self, visitor_id: str) -> int:
        self._ensure_initialized()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute("DELETE FROM snapshots WHERE visitor_id = ?", (visitor_id,))
            deleted = max(0, cursor.rowcount)
        with self._connect() as connection:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return deleted

    def healthcheck(self) -> bool:
        try:
            self._ensure_initialized()
            with self._connect() as connection:
                return connection.execute("SELECT 1").fetchone()[0] == 1
        except (OSError, sqlite3.Error):
            return False
