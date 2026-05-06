"""Anonymous visitor tracking and snapshot persistence."""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from data import GPUS, MODELS
from state import PlannerState


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_default(obj):
    # PlannerState carries frozensets (Project.requires) that asdict leaves in place;
    # JSON has no set type, so coerce to a sorted list for stable hashes/diffs.
    if isinstance(obj, (frozenset, set)):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _serialize_state(state: Optional[PlannerState]) -> Optional[dict]:
    if state is None:
        return None
    return asdict(state)


def _gpu_summary(state: Optional[PlannerState]) -> str:
    if state is None or not state.gpus:
        return ""
    return ", ".join(f"{gp.count}x {GPUS[gp.gpu_type].name}" for gp in state.gpus)


def _model_summary(state: Optional[PlannerState]) -> str:
    if state is None or not state.models:
        return ""
    parts = []
    for assignment in state.models:
        parts.append(
            f"{MODELS[assignment.model_key].name} ({assignment.gpu_count} GPU, {assignment.prec.upper()}, "
            f"D {assignment.tp}/{assignment.pp}/{assignment.dp})"
        )
    return ", ".join(parts)


def _state_summary(state: Optional[PlannerState]) -> dict:
    if state is None:
        return {
            "mode": None,
            "gpu_summary": "",
            "model_summary": "",
            "gpu_pool_count": 0,
            "model_count": 0,
        }
    return {
        "mode": state.mode,
        "gpu_summary": _gpu_summary(state),
        "model_summary": _model_summary(state),
        "gpu_pool_count": len(state.gpus),
        "model_count": len(state.models),
    }


class SnapshotStore:
    """Append-only JSON-backed snapshot storage for anonymous usage tracking."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def _empty(self) -> dict:
        return {"version": 1, "visitors": {}}

    def _load(self) -> dict:
        if not self.path.exists():
            return self._empty()
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return self._empty()

    def _save(self, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
        temp_path.replace(self.path)

    def record_snapshot(
        self,
        *,
        visitor_id: str,
        tab_id: str,
        reason: str,
        path: str,
        state_a: PlannerState,
        state_b: Optional[PlannerState],
    ) -> None:
        now = _utc_now()
        panel_a = _serialize_state(state_a)
        panel_b = _serialize_state(state_b)
        state_hash = hashlib.sha256(
            json.dumps({"panel_a": panel_a, "panel_b": panel_b}, sort_keys=True, default=_json_default).encode("utf-8")
        ).hexdigest()

        summary_a = _state_summary(state_a)
        summary_b = _state_summary(state_b)

        with self._lock:
            payload = self._load()
            visitor = payload["visitors"].setdefault(
                visitor_id,
                {"visitor_id": visitor_id, "first_seen": now, "last_seen": now, "tabs": {}},
            )
            visitor["last_seen"] = now
            tab = visitor["tabs"].setdefault(
                tab_id,
                {
                    "tab_id": tab_id,
                    "first_seen": now,
                    "last_seen": now,
                    "last_path": path,
                    "snapshots": [],
                },
            )
            tab["last_seen"] = now
            tab["last_path"] = path

            if tab["snapshots"] and tab["snapshots"][-1]["state_hash"] == state_hash:
                tab["snapshots"][-1]["last_seen"] = now
                tab["snapshots"][-1]["reason"] = reason
                tab["snapshots"][-1]["path"] = path
                self._save(payload)
                return

            tab["snapshots"].append(
                {
                    "snapshot_id": str(uuid.uuid4()),
                    "created_at": now,
                    "last_seen": now,
                    "reason": reason,
                    "path": path,
                    "state_hash": state_hash,
                    "panel_a": panel_a,
                    "panel_b": panel_b,
                    "summary": {
                        "mode": summary_a["mode"],
                        "compare_enabled": state_b is not None,
                        "panel_a": summary_a,
                        "panel_b": summary_b,
                    },
                }
            )
            self._save(payload)

    def list_snapshots(self) -> list[dict]:
        with self._lock:
            payload = self._load()

        rows = []
        for visitor_id, visitor in payload.get("visitors", {}).items():
            for tab_id, tab in visitor.get("tabs", {}).items():
                for snapshot in tab.get("snapshots", []):
                    rows.append(
                        {
                            "visitor_id": visitor_id,
                            "tab_id": tab_id,
                            "snapshot_id": snapshot["snapshot_id"],
                            "created_at": snapshot["created_at"],
                            "last_seen": snapshot.get("last_seen", snapshot["created_at"]),
                            "reason": snapshot["reason"],
                            "path": snapshot["path"],
                            "summary": snapshot["summary"],
                            "panel_a": snapshot["panel_a"],
                            "panel_b": snapshot["panel_b"],
                        }
                    )

        rows.sort(key=lambda row: row["last_seen"], reverse=True)
        return rows
