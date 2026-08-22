"""Process-local registry of planner state, keyed by browser-tab scope.

The planner keeps every visitor's state in this process -- there is no
external store, and ``WEB_CONCURRENCY`` is pinned to 1 because of it. This is
the module's only global mutable state and its only threading concern, which
is why it lives here rather than in ``state.py`` alongside the dataclasses.

If a shared store is ever needed, this module is the seam to replace; nothing
else reaches into ``_states`` directly.
"""

from __future__ import annotations

import copy
import os
import threading
import time
import weakref
from typing import Optional

from data import normalize_precision
from state import (
    CORPO_CLOUD_DEFAULT,
    DEFAULT_AUTO_MODEL_STRATEGY,
    PlannerState,
    create_default_scenario,
    normalize_auto_strategy,
    normalize_corpo_cloud,
    normalize_day_shape,
    normalize_embedding_doc_distribution,
    normalize_plot_mode,
    normalize_projects,
)


def _env_nonnegative_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


_states: dict[str, PlannerState] = {}
_compare_states: dict[str, PlannerState] = {}
_state_last_seen: dict[str, float] = {}
_state_guard = threading.RLock()
_scope_locks: weakref.WeakValueDictionary[str, threading.RLock] = weakref.WeakValueDictionary()
_STATE_TTL_SECONDS = _env_nonnegative_int("PLANNER_STATE_TTL_SECONDS", 86400)
_STATE_MAX_SCOPES = max(1, _env_nonnegative_int("PLANNER_STATE_MAX_SCOPES", 5000))


def get_scope_lock(session_id: str) -> threading.RLock:
    """Return the process-local lock serializing one browser-tab scope."""
    with _state_guard:
        lock = _scope_locks.get(session_id)
        if lock is None:
            lock = threading.RLock()
            _scope_locks[session_id] = lock
        return lock


def allow_visitor_scope(session_id: str, visitor_id: str, max_scopes: int) -> bool:
    """Return whether a visitor may reuse/create this tab scope under its cap."""
    prefix = f"{visitor_id}:"
    with _state_guard:
        _prune_states_locked(time.monotonic(), preserve=session_id)
        if session_id in _states or session_id in _compare_states:
            return True
        existing = sum(1 for key in _states if key.startswith(prefix))
        return existing < max(1, int(max_scopes))


def _prune_states_locked(now: float, preserve: Optional[str] = None) -> None:
    if _STATE_TTL_SECONDS > 0:
        stale = [
            key
            for key, touched in _state_last_seen.items()
            if key != preserve and now - touched > _STATE_TTL_SECONDS
        ]
        for key in stale:
            _states.pop(key, None)
            _compare_states.pop(key, None)
            _state_last_seen.pop(key, None)

    excess = len(_states) - _STATE_MAX_SCOPES
    if excess > 0:
        oldest = sorted(
            (touched, key) for key, touched in _state_last_seen.items() if key != preserve
        )[:excess]
        for _, key in oldest:
            _states.pop(key, None)
            _compare_states.pop(key, None)
            _state_last_seen.pop(key, None)


def reset_state(session_id: str, *, blank: bool = False) -> PlannerState:
    with _state_guard:
        previous_revision = max(
            getattr(_states.get(session_id), "revision", -1),
            getattr(_compare_states.get(session_id), "revision", -1),
        )
        state, compare = (PlannerState(), None) if blank else create_default_scenario()
        state.revision = max(state.revision, previous_revision + 1)
        if compare is not None:
            compare.revision = max(compare.revision, previous_revision + 1)
        _states[session_id] = state
        if compare is None:
            _compare_states.pop(session_id, None)
        else:
            _compare_states[session_id] = compare
        _state_last_seen[session_id] = time.monotonic()
        _prune_states_locked(_state_last_seen[session_id], preserve=session_id)
        return state


def delete_visitor_states(visitor_id: str) -> int:
    prefix = f"{visitor_id}:"
    with _state_guard:
        keys = {key for key in (*_states.keys(), *_compare_states.keys()) if key.startswith(prefix)}
        for key in keys:
            _states.pop(key, None)
            _compare_states.pop(key, None)
            _state_last_seen.pop(key, None)
        return len(keys)


def _normalize_loaded_state(s: PlannerState) -> PlannerState:
    for am in s.models:
        am.prec = normalize_precision(getattr(am, "prec", "bf16"))
    s.mode = normalize_plot_mode(s.mode)
    s.projection_day_shape = normalize_day_shape(s.projection_day_shape)
    s.corpo_cloud = normalize_corpo_cloud(getattr(s, "corpo_cloud", CORPO_CLOUD_DEFAULT))
    if not hasattr(s, "auto_excluded"):
        s.auto_excluded = []
    if not hasattr(s, "auto_mode"):
        s.auto_mode = False
    s.auto_strategy = normalize_auto_strategy(
        getattr(s, "auto_strategy", DEFAULT_AUTO_MODEL_STRATEGY)
    )
    normalize_embedding_doc_distribution(s)
    normalize_projects(s)
    return s


def get_state(session_id: str) -> PlannerState:
    now = time.monotonic()
    with _state_guard:
        _prune_states_locked(now, preserve=session_id)
        if session_id not in _states:
            state, compare = create_default_scenario()
            _states[session_id] = state
            if compare is not None:
                _compare_states[session_id] = compare
        _state_last_seen[session_id] = now
        _prune_states_locked(now, preserve=session_id)
        s = _states[session_id]
    return _normalize_loaded_state(s)


def get_compare_state(session_id: str) -> Optional[PlannerState]:
    with _state_guard:
        state = _compare_states.get(session_id)
        if state is not None:
            _state_last_seen[session_id] = time.monotonic()
    if state is not None:
        _normalize_loaded_state(state)
    return state


def duplicate_compare_state(session_id: str) -> PlannerState:
    # Clone the current primary configuration so panel B starts from panel A.
    with _state_guard:
        _compare_states[session_id] = copy.deepcopy(get_state(session_id))
        _compare_states[session_id].touch()
        _state_last_seen[session_id] = time.monotonic()
        return _compare_states[session_id]


def clear_compare_state(session_id: str) -> bool:
    with _state_guard:
        return _compare_states.pop(session_id, None) is not None


def replace_scope_states(
    session_id: str, state_a: PlannerState, state_b: Optional[PlannerState]
) -> None:
    with _state_guard:
        previous_revision = max(
            getattr(_states.get(session_id), "revision", -1),
            getattr(_compare_states.get(session_id), "revision", -1),
        )
        next_revision = previous_revision + 1
        state_a.revision = max(state_a.revision, next_revision)
        if state_b is not None:
            state_b.revision = max(state_b.revision, next_revision)
        _states[session_id] = state_a
        if state_b is None:
            _compare_states.pop(session_id, None)
        else:
            _compare_states[session_id] = state_b
        _state_last_seen[session_id] = time.monotonic()
        _prune_states_locked(_state_last_seen[session_id], preserve=session_id)
