"""Show capacity for models without a compatible published quality score."""

from calc import (
    EMBEDDING_BATCH_SIZES,
    _embedding_doc_dist_for_state,
    _iter_resolved_models,
    compute_embedding_distribution,
    compute_realtime_max_users,
)
from data import EMBEDDING_DOC_BUCKETS, PUBLISHED_ASR_WER, PUBLISHED_EMBEDDING_QUALITY


def capacity_without_quality(state, mode: str, suffix: str = "") -> list[dict]:
    rows = []
    capacity: float
    for am, gpu in _iter_resolved_models(state):
        model = am.model
        if gpu is None or am.gpu_count <= 0:
            continue
        if mode == "asrquality" and model.is_asr_model and model.key not in PUBLISHED_ASR_WER:
            capacity = compute_realtime_max_users(
                model,
                (am.tp, am.pp, am.dp),
                gpu,
                state.mu,
                state.profiled_non_kv_gb,
                am.prec,
                state.decode_efficiency,
            )
            unit = "concurrent streams"
        elif (
            mode == "embedquality"
            and model.is_embedding_model
            and model.key not in PUBLISHED_EMBEDDING_QUALITY
        ):
            results = [
                compute_embedding_distribution(
                    model,
                    (am.prefill_tp, am.prefill_pp, am.prefill_dp),
                    batch,
                    _embedding_doc_dist_for_state(state),
                    EMBEDDING_DOC_BUCKETS,
                    gpu,
                    state.mu,
                    state.profiled_non_kv_gb,
                    am.prec,
                    state.prefill_efficiency,
                )
                for batch in EMBEDDING_BATCH_SIZES
            ]
            capacity = max((r.rps for r in results if r is not None), default=0)
            unit = "docs/s"
        else:
            continue
        rows.append(
            {
                "model_key": model.key,
                "label": model.name + suffix,
                "capacity": capacity,
                "unit": unit,
                "color": model.color,
                "hardware": f"{am.gpu_count}× {gpu.name}",
                "note": "Capacity estimate only; no compatible published quality score is available.",
            }
        )
    return rows
