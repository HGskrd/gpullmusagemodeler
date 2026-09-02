"""Application services coordinating state mutation, placement, and scenarios."""

from __future__ import annotations

import json
import math
from typing import Any, Optional

from data import MODELS, PRECISION_LABELS, PRECISIONS
from placement import (
    _best_precision_need,
    _finite_gpu_need,
    _min_gpu_count_for_pool,
    _retune_model,
    retune_models,
)
from scenarios import deserialize_scenario as _deserialize_scenario
from state import (
    DEFAULT_SCENARIO_PATH,
    GpuPool,
    PlannerState,
    add_model_assignment,
    configure_default_scenario_factory,
)
from state import (
    change_gpu_qty as _change_gpu_qty,
)
from state import (
    set_model_gpu_count as _set_model_gpu_count,
)
from state import (
    set_model_gpu_pool as _set_model_gpu_pool,
)
from state import (
    set_model_prec as _set_model_prec,
)
from state import (
    set_model_spec as _set_model_spec,
)


def deserialize_scenario(payload: Any) -> tuple[PlannerState, Optional[PlannerState]]:
    state, compare = _deserialize_scenario(payload)
    retune_models(state, preserve_existing=True)
    if compare is not None:
        retune_models(compare, preserve_existing=True)
    return state, compare


def create_default_scenario() -> tuple[PlannerState, Optional[PlannerState]]:
    with DEFAULT_SCENARIO_PATH.open("r", encoding="utf-8") as handle:
        return deserialize_scenario(json.load(handle))


def create_default_state() -> PlannerState:
    state, _compare = create_default_scenario()
    return state


def change_gpu_qty(state: PlannerState, gpu_uid: int, delta: int) -> None:
    for assignment in _change_gpu_qty(state, gpu_uid, delta):
        _retune_model(state, assignment)


def add_model(state: PlannerState, model_key: str) -> None:
    if not state.gpus:
        raise ValueError("Add a GPU pool before adding a model.")
    if model_key not in MODELS or MODELS[model_key].hidden:
        raise ValueError("Invalid model key.")
    model = MODELS[model_key]

    def fit_needs(pool: GpuPool) -> tuple[dict[str, float], dict[str, float]]:
        available = state.free_gpu_for_pool(pool.uid)
        needs_now = {
            precision: _min_gpu_count_for_pool(
                model,
                pool.gpu,
                state.mu,
                state.profiled_non_kv_gb,
                precision,
                available,
            )
            for precision in PRECISIONS
        }
        needs_full = {
            precision: _min_gpu_count_for_pool(
                model,
                pool.gpu,
                state.mu,
                state.profiled_non_kv_gb,
                precision,
                pool.count,
            )
            for precision in PRECISIONS
        }
        return needs_now, needs_full

    def sort_key(pool: GpuPool) -> tuple[bool, float, bool, float, bool, float, int, int]:
        available = state.free_gpu_for_pool(pool.uid)
        needs_now, needs_full = fit_needs(pool)
        best_now = _finite_gpu_need(*needs_now.values())
        best_full = _finite_gpu_need(*needs_full.values())
        native_now = needs_now[model.native_precision_key]
        return (
            math.isinf(native_now),
            native_now,
            math.isinf(best_now),
            best_now,
            math.isinf(best_full),
            best_full,
            -available,
            -pool.count,
        )

    pool = min(state.gpus, key=sort_key)
    available = state.free_gpu_for_pool(pool.uid)
    needs_now, needs_full = fit_needs(pool)
    best_full = _finite_gpu_need(*needs_full.values())
    if math.isinf(best_full):
        labels = ", ".join(PRECISION_LABELS[precision] for precision in PRECISIONS)
        raise ValueError(
            f"{model.name} does not fit on any configured GPU pool under the current memory cap in {labels}."
        )

    native_precision = model.native_precision_key
    native_now = needs_now[native_precision]
    # _best_precision_need returns None when nothing fits; the fallback below
    # resolves it, so the binding is optional until then.
    selected_precision: Optional[str]
    if not math.isinf(native_now):
        selected_precision = native_precision
        gpu_count = int(native_now)
    else:
        selected_precision, best_now = _best_precision_need(needs_now)
        if selected_precision is not None and not math.isinf(best_now):
            gpu_count = int(best_now)
        else:
            selected_precision, _ = _best_precision_need(needs_full)
            selected_precision = selected_precision or native_precision
            gpu_count = available

    assignment = add_model_assignment(
        state,
        model_key,
        pool.uid,
        gpu_count,
        selected_precision,
    )
    _retune_model(state, assignment)


def add_models(state: PlannerState, model_keys: list[str]) -> list[str]:
    existing = {assignment.model_key for assignment in state.models}
    added = []
    for model_key in model_keys:
        if model_key in existing:
            continue
        add_model(state, model_key)
        existing.add(model_key)
        added.append(model_key)
    return added


def set_model_prec(state: PlannerState, model_uid: int, precision: str) -> None:
    assignment = _set_model_prec(state, model_uid, precision)
    if assignment is not None:
        _retune_model(state, assignment, preserve_existing=True)


def set_model_spec(state: PlannerState, model_uid: int, method: str, spec_k: int) -> None:
    assignment = _set_model_spec(state, model_uid, method, spec_k)
    if assignment is not None:
        _retune_model(state, assignment, preserve_existing=True)


def set_model_gpu_count(state: PlannerState, model_uid: int, count: int) -> None:
    assignment = _set_model_gpu_count(state, model_uid, count)
    if assignment is not None:
        _retune_model(state, assignment)


def set_model_gpu_pool(state: PlannerState, model_uid: int, gpu_uid: int) -> None:
    assignment = _set_model_gpu_pool(state, model_uid, gpu_uid)
    if assignment is not None:
        _retune_model(state, assignment)


configure_default_scenario_factory(create_default_scenario)
