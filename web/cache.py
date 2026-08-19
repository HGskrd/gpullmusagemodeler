"""Small process-local caches for expensive presentation derivations."""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

from scenarios import serialize_scenario
from state import PlannerState

ChartCacheKey = tuple[int, str, str]


@dataclass(frozen=True)
class CachedJson:
    body: bytes
    etag: str


class ScopedRevisionCache:
    """Bounded cache whose per-scope key is (revision, panel, mode)."""

    def __init__(self, max_scopes: int = 5000, max_entries_per_scope: int = 8):
        self._max_scopes = max(1, int(max_scopes))
        self._max_entries_per_scope = max(1, int(max_entries_per_scope))
        self._scopes: OrderedDict[str, OrderedDict[ChartCacheKey, CachedJson]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, scope_id: str, key: ChartCacheKey) -> Optional[CachedJson]:
        with self._lock:
            entries = self._scopes.get(scope_id)
            if entries is None or key not in entries:
                return None
            value = entries.pop(key)
            entries[key] = value
            self._scopes.move_to_end(scope_id)
            return value

    def put(self, scope_id: str, key: ChartCacheKey, body: bytes) -> CachedJson:
        value = CachedJson(body=body, etag=hashlib.sha256(body).hexdigest())
        with self._lock:
            entries = self._scopes.setdefault(scope_id, OrderedDict())
            entries.pop(key, None)
            entries[key] = value
            while len(entries) > self._max_entries_per_scope:
                entries.popitem(last=False)
            self._scopes.move_to_end(scope_id)
            while len(self._scopes) > self._max_scopes:
                self._scopes.popitem(last=False)
        return value


class FingerprintCache:
    """Thread-safe bounded LRU for scenario-derived Python values."""

    def __init__(self, max_entries: int = 256):
        self._max_entries = max(1, int(max_entries))
        self._entries: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, fingerprint: str) -> Any:
        with self._lock:
            if fingerprint not in self._entries:
                return None
            value = self._entries.pop(fingerprint)
            self._entries[fingerprint] = value
            return value

    def put(self, fingerprint: str, value: Any) -> Any:
        with self._lock:
            self._entries.pop(fingerprint, None)
            self._entries[fingerprint] = value
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
        return value


def combined_revision(state_a: PlannerState, state_b: Optional[PlannerState]) -> int:
    """Pair A/B revisions into one collision-free non-negative cache revision."""
    a_revision = max(0, int(state_a.revision))
    b_revision = max(0, int(state_b.revision) + 1) if state_b is not None else 0
    total = a_revision + b_revision
    return total * (total + 1) // 2 + b_revision


def scenario_fingerprint(state: PlannerState) -> str:
    """Stable content fingerprint; revision and generated UIDs do not affect it."""
    cached = getattr(state, "_scenario_fingerprint_cache", None)
    if cached is not None and cached[0] == state.revision:
        return cached[1]
    panel = serialize_scenario(state, None)["panel_a"]
    if isinstance(panel, dict):
        gpu_ids = {str(row.get("uid")): index for index, row in enumerate(panel.get("gpus", []))}
        for row in panel.get("gpus", []):
            row.pop("uid", None)
        for row in panel.get("models", []):
            row.pop("uid", None)
            row["gpu_uid"] = gpu_ids.get(str(row.get("gpu_uid")), row.get("gpu_uid"))
        for row in panel.get("projects", []):
            row.pop("uid", None)
    encoded = json.dumps(panel, sort_keys=True, separators=(",", ":")).encode()
    fingerprint = hashlib.sha256(encoded).hexdigest()
    setattr(state, "_scenario_fingerprint_cache", (state.revision, fingerprint))
    return fingerprint
