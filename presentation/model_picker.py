"""View-model rows for the "+ Add model" picker: will it fit, is it good enough?

Every number here is derived from the same functions the planner uses when a
row is clicked (``planner_service.plan_model_add`` for pool/precision/GPU
count, ``calc.default_strategy`` + ``calc.compute_memory`` for KV headroom,
``data.effective_quality`` for quality). Nothing is re-derived in JS; the
template renders these values as data attributes and the client only filters
and sorts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from calc import compute_memory, default_strategy
from data import GPU, PRECISION_LABELS, Model, effective_quality, required_quality
from planner_service import plan_model_add
from state import PlannerState

# A placement that fits is flagged "tight" when the per-GPU memory left for KV
# cache (gpu mem x mu - weights - profiled non-KV) is below this share of the
# per-GPU memory cap (gpu mem x mu). Below ~15 % the replica can hold only a
# few long-context sequences, so concurrency is likely to be KV-bound.
TIGHT_KV_HEADROOM_FRACTION = 0.15

FIT_FITS = "fits"
FIT_TIGHT = "tight"
FIT_NEEDS = "needs"
FIT_NONE = "none"
FIT_NO_GPU = "nogpu"


@dataclass(frozen=True)
class PickerRow:
    model: Model
    precision_label: str
    weights_gb: Optional[float]
    fit: str
    fit_label: str
    fit_title: str
    passes_fit: bool
    quality: Optional[float]
    passes_quality: bool

    @property
    def search_text(self) -> str:
        m = self.model
        return f"{m.name} {m.cat} {m.size_label} {m.key}".lower()


def panel_quality_floor(state: PlannerState) -> Optional[float]:
    """Quality floor for the "Meets quality floor" chip, or None without use cases.

    Definition: the *least* demanding use case's required general quality,
    i.e. ``min(required_quality(difficulty, min_success_rate, quality_floor))``
    over the panel's use cases. A model below it cannot clear any use case on
    general quality alone, so dimming it is safe; passing it is a necessary,
    not sufficient, condition (domain weights and capability gates are applied
    later by routing and are not checked here).
    """
    if not state.projects:
        return None
    return min(
        required_quality(p.difficulty, p.min_success_rate, quality_floor=p.quality_floor)
        for p in state.projects
    )


def _kv_headroom_bytes(
    state: PlannerState, model: Model, gpu: GPU, precision: str, gpu_count: int
) -> Optional[float]:
    """Per-GPU bytes left for KV cache at the layout the planner defaults to."""
    tp, pp, _dp = default_strategy(
        model, gpu_count, gpu, state.mu, state.profiled_non_kv_gb, precision
    )
    memory = compute_memory(
        model,
        tp,
        pp,
        gpu,
        state.mu,
        state.profiled_non_kv_gb,
        precision,
        state.decode_efficiency,
    )
    return None if memory is None else memory.kv_reserved


def picker_row(
    state: PlannerState, model: Model, quality_floor: Optional[float], is_llm: bool
) -> PickerRow:
    quality = effective_quality(model) if is_llm else None
    passes_quality = quality is None or quality_floor is None or quality + 1e-9 >= quality_floor
    if not state.gpus:
        return PickerRow(
            model,
            PRECISION_LABELS[model.native_precision_key],
            model.gpu_weight_bytes(model.native_precision_key) / 1e9,
            FIT_NO_GPU,
            "add a GPU first",
            "Add a GPU pool to check fit",
            True,
            quality,
            passes_quality,
        )
    plan = plan_model_add(state, model)
    if plan is None:
        precision = model.native_precision_key
        return PickerRow(
            model,
            PRECISION_LABELS[precision],
            model.gpu_weight_bytes(precision) / 1e9,
            FIT_NONE,
            "doesn't fit",
            "Does not fit any GPU pool at any precision under the memory cap",
            False,
            quality,
            passes_quality,
        )
    gpu = plan.pool.gpu
    weights_gb = model.gpu_weight_bytes(plan.precision) / 1e9
    if not plan.fits_now:
        free = state.free_gpu_for_pool(plan.pool.uid)
        return PickerRow(
            model,
            PRECISION_LABELS[plan.precision],
            weights_gb,
            FIT_NEEDS,
            f"needs {plan.gpus_needed} GPU{'' if plan.gpus_needed == 1 else 's'}",
            f"Needs {plan.gpus_needed} × {gpu.name}; {free} free",
            False,
            quality,
            passes_quality,
        )
    headroom = _kv_headroom_bytes(state, model, gpu, plan.precision, plan.gpu_count)
    cap = gpu.mem * state.mu
    noun = "GPU" if plan.gpu_count == 1 else "GPUs"
    if headroom is not None and headroom < TIGHT_KV_HEADROOM_FRACTION * cap:
        left_gb = headroom / 1e9
        return PickerRow(
            model,
            PRECISION_LABELS[plan.precision],
            weights_gb,
            FIT_TIGHT,
            f"tight · {left_gb:.1f} GB KV left",
            f"Fits on {plan.gpu_count} × {gpu.name}, {left_gb:.1f} GB per GPU left for KV cache",
            True,
            quality,
            passes_quality,
        )
    return PickerRow(
        model,
        PRECISION_LABELS[plan.precision],
        weights_gb,
        FIT_FITS,
        f"fits on {plan.gpu_count}",
        f"Fits on {plan.gpu_count} free {gpu.name} {noun}",
        True,
        quality,
        passes_quality,
    )


def picker_rows(
    state: PlannerState, groups: dict[str, list[Model]], kind: Optional[str]
) -> dict[str, list[PickerRow]]:
    """Rows for one picker kind, keeping catalog grouping and order.

    Quality (and the quality-floor chip) applies to LLMs only.
    """
    is_llm = kind == "llm"
    floor = panel_quality_floor(state) if is_llm else None
    return {
        cat: [picker_row(state, m, floor, is_llm) for m in models] for cat, models in groups.items()
    }
