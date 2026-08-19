"""Chart.js presentation models built from estimator results."""

from __future__ import annotations

import math
from typing import Optional, cast

from calc import (
    DATA_BATCH_SIZES,
    EMBEDDING_BATCH_SIZES,
    PROCESSING_PARETO_COLORS,
    REALTIME_USER_SWEEP,
    USER_EXP_FRACTIONS,
    USER_EXP_SWEEP,
    EfficiencyParams,
    SpecRuntime,
    _embedding_doc_dist_for_state,
    _is_decode_pareto_model,
    _iter_resolved_models,
    avg_dist,
    compute_data,
    compute_decode,
    compute_embedding_distribution,
    compute_realtime_capacity,
    compute_realtime_max_users,
    compute_user_experience,
    embedding_doc_stats,
    resolve_spec_runtime,
    spec_optimization_for,
    spec_runtime_for,
    strategy_label,
)
from data import (
    ASR_WER_LANGUAGE_LABELS,
    ASR_WER_LANGUAGE_SOURCES,
    ASR_WER_LANGUAGES,
    ASR_WER_PLACEHOLDER,
    BATCH_SIZES,
    DIST_PRESETS,
    EMBEDDING_DECONTAMINATED_BEIR_SOURCES,
    EMBEDDING_DOC_BUCKETS,
    EMBEDDING_QUALITY_PLACEHOLDER,
    EMBEDDING_QUALITY_SOURCES,
    GPU,
    INPUT_BUCKETS,
    OUTPUT_BUCKETS,
    PUBLISHED_ASR_WER,
    PUBLISHED_EMBEDDING_DECONTAMINATED_BEIR,
    PUBLISHED_EMBEDDING_QUALITY,
    Model,
)
from deployment import Deployment


def _style(model: Model, is_b: bool) -> dict:
    """Shared line-series style; builders override only genuine chart differences."""
    return {
        "borderColor": model.color,
        "backgroundColor": model.color + "12",
        "borderWidth": 1.5 if is_b else 2,
        "borderDash": [5, 3] if is_b else [],
        "fill": False,
        "tension": 0.3,
        "pointRadius": 2.5,
        "spanGaps": False,
    }


def chart_embedding_quality(state, panel_suffix: str = "") -> list[dict]:
    """Peak docs/s vs published retrieval quality, one dot per embedding model.

    Each model emits a single point — x = peak docs/s (max over the standard
    batch sweep at the current workload distribution), y = decontaminated BEIR
    quality when sourced, otherwise the existing catalog quality fallback in
    [0, 1]. Bytes-per-doc and vec/s are attached to the point so the front-end
    can encode storage cost via dot size and surface multi-vector blowup in the
    tooltip.
    """
    datasets = []
    is_b = panel_suffix != ""
    doc_dist = _embedding_doc_dist_for_state(state)

    for am, gpu in _iter_resolved_models(state):
        model = am.model
        profile = getattr(model, "embedding_profile", None)
        if profile is None:
            continue
        fallback_quality = PUBLISHED_EMBEDDING_QUALITY.get(model.key)
        if fallback_quality is None:
            continue

        stats = embedding_doc_stats(model, doc_dist, EMBEDDING_DOC_BUCKETS, am.prec)

        best = None
        for bs in EMBEDDING_BATCH_SIZES:
            result = compute_embedding_distribution(
                model,
                (am.prefill_tp, am.prefill_pp, am.prefill_dp),
                bs,
                doc_dist,
                EMBEDDING_DOC_BUCKETS,
                gpu,
                state.mu,
                state.profiled_non_kv_gb,
                am.prec,
                state.prefill_efficiency,
            )
            if result is None:
                continue
            if best is None or result.rps > best.rps:
                best = result
                best_bs = bs
        if best is None:
            continue

        is_placeholder = model.key in EMBEDDING_QUALITY_PLACEHOLDER
        decontaminated_beir = PUBLISHED_EMBEDDING_DECONTAMINATED_BEIR.get(model.key)
        uses_decontaminated_beir = decontaminated_beir is not None
        quality = decontaminated_beir if uses_decontaminated_beir else fallback_quality
        bytes_per_doc = stats.mean_output_bytes_per_input
        point = {
            "x": best.rps,
            "y": quality,
            "docs_per_second": best.rps,
            "tokens_per_second": best.tps,
            "vectors_per_second": best.vectors_per_second,
            "vectors_per_input": best.vectors_per_input,
            "output_mb_s": best.output_mb_s,
            "bytes_per_doc": bytes_per_doc,
            "seq_len": best.seq_len,
            "peak_batch": best_bs,
            "max_batch": best.max_batch,
            "mode": profile.mode_label,
            "quality": quality,
            "quality_metric": "Decontaminated BEIR nDCG@10"
            if uses_decontaminated_beir
            else "Published retrieval nDCG@10 fallback",
            "source": (
                EMBEDDING_DECONTAMINATED_BEIR_SOURCES.get(model.key, "")
                if uses_decontaminated_beir
                else EMBEDDING_QUALITY_SOURCES.get(model.key, "")
            ),
            "published_quality": fallback_quality,
            "published_quality_source": EMBEDDING_QUALITY_SOURCES.get(model.key, ""),
            "decontaminated_beir_quality": decontaminated_beir,
            "decontaminated_beir_source": EMBEDDING_DECONTAMINATED_BEIR_SOURCES.get(model.key, ""),
            "uses_decontaminated_beir": uses_decontaminated_beir,
            "placeholder": is_placeholder,
        }

        datasets.append(
            {
                "label": _label(am, model, panel_suffix, include_prefill=True),
                "data": [point],
                **_style(model, is_b),
                "backgroundColor": (model.color + "12") if is_placeholder else (model.color + "AA"),
                "showLine": False,
                "tension": 0,
                "pointRadius": 5,
                "_isEmbeddingQuality": True,
                "_placeholder": is_placeholder,
            }
        )
    return datasets


def chart_embedding_throughput(
    state, batch_sizes: Optional[list[int]] = None, panel_suffix: str = ""
) -> list[dict]:
    datasets = []
    is_b = panel_suffix != ""
    batch_sizes = batch_sizes or EMBEDDING_BATCH_SIZES
    doc_dist = _embedding_doc_dist_for_state(state)

    for am, gpu in _iter_resolved_models(state):
        model = am.model
        profile = getattr(model, "embedding_profile", None)
        if profile is None:
            continue

        stats = embedding_doc_stats(model, doc_dist, EMBEDDING_DOC_BUCKETS, am.prec)
        pts = []
        for bs in batch_sizes:
            result = compute_embedding_distribution(
                model,
                (am.prefill_tp, am.prefill_pp, am.prefill_dp),
                bs,
                doc_dist,
                EMBEDDING_DOC_BUCKETS,
                gpu,
                state.mu,
                state.profiled_non_kv_gb,
                am.prec,
                state.prefill_efficiency,
            )
            if result is None:
                pts.append(
                    {
                        "x": bs,
                        "y": None,
                        "seq_len": round(stats.mean_seq_len),
                        "p50_seq_len": stats.p50_seq_len,
                        "p90_seq_len": stats.p90_seq_len,
                        "p99_seq_len": stats.p99_seq_len,
                        "mode": profile.mode_label,
                        "max_batch": 0,
                    }
                )
                continue
            pts.append(
                {
                    "x": bs,
                    "y": result.rps,
                    "rps": result.rps,
                    "tps": result.tps,
                    "vectors_per_second": result.vectors_per_second,
                    "vectors_per_input": result.vectors_per_input,
                    "output_mb_s": result.output_mb_s,
                    "seq_len": result.seq_len,
                    "p50_seq_len": result.p50_seq_len,
                    "p90_seq_len": result.p90_seq_len,
                    "p99_seq_len": result.p99_seq_len,
                    "mode": profile.mode_label,
                    "max_batch": result.max_batch,
                }
            )

        datasets.append(
            {
                "label": _label(am, model, panel_suffix, include_prefill=True),
                "data": pts,
                **_style(model, is_b),
                "_isEmbedding": True,
            }
        )
    return datasets


def chart_data_processing(
    state, batch_sizes: Optional[list[int]] = None, panel_suffix: str = ""
) -> list[dict]:
    datasets = []
    is_b = panel_suffix != ""
    il, ol = state.task_il, state.task_ol
    batch_sizes = batch_sizes or DATA_BATCH_SIZES

    for am, gpu in _iter_resolved_models(state):
        model = am.model
        if getattr(model, "embedding_profile", None) is not None:
            continue
        pts = []
        for bs in batch_sizes:
            result = compute_data(
                model,
                (am.prefill_tp, am.prefill_pp, am.prefill_dp),
                (am.tp, am.pp, am.dp),
                bs,
                il,
                ol,
                gpu,
                state.mu,
                state.profiled_non_kv_gb,
                am.prec,
                state.prefix_hit_rate,
                state.prefill_efficiency,
                state.decode_efficiency,
                spec_runtime_for(state, am, model),
            )
            pts.append({"x": bs, "y": result.tps if result else None})
        datasets.append(
            {
                "label": _label(am, model, panel_suffix),
                "data": pts,
                **_style(model, is_b),
                "fill": not is_b,
            }
        )

    agg = []
    for bs in batch_sizes:
        total = 0
        for am, gpu in _iter_resolved_models(state):
            model = am.model
            if getattr(model, "embedding_profile", None) is not None:
                continue
            result = compute_data(
                model,
                (am.prefill_tp, am.prefill_pp, am.prefill_dp),
                (am.tp, am.pp, am.dp),
                bs,
                il,
                ol,
                gpu,
                state.mu,
                state.profiled_non_kv_gb,
                am.prec,
                state.prefix_hit_rate,
                state.prefill_efficiency,
                state.decode_efficiency,
                spec_runtime_for(state, am, model),
            )
            if result:
                total += result.tps
        agg.append({"x": bs, "y": total or None})
    datasets.append(
        {
            "label": f"Node total{panel_suffix}",
            "data": agg,
            "borderColor": "#ddd",
            "borderWidth": 2,
            "borderDash": [5, 3],
            "fill": False,
            "tension": 0.3,
            "pointRadius": 1.5,
            "spanGaps": False,
            "_isAggregate": True,
        }
    )
    return datasets


def chart_realtime_capacity(
    state, batch_sizes: Optional[list[int]] = None, panel_suffix: str = ""
) -> list[dict]:
    datasets = []
    is_b = panel_suffix != ""
    batch_sizes = batch_sizes or REALTIME_USER_SWEEP

    for am, gpu in _iter_resolved_models(state):
        model = am.model
        profile = getattr(model, "realtime_profile", None)
        if profile is None:
            continue

        max_users = compute_realtime_max_users(
            model,
            (am.tp, am.pp, am.dp),
            gpu,
            state.mu,
            state.profiled_non_kv_gb,
            am.prec,
            state.decode_efficiency,
        )
        pts = []
        for users in batch_sizes:
            result = compute_realtime_capacity(
                model,
                (am.tp, am.pp, am.dp),
                users,
                gpu,
                state.mu,
                state.profiled_non_kv_gb,
                am.prec,
                state.decode_efficiency,
            )
            if result is None:
                pts.append(
                    {
                        "x": users,
                        "y": None,
                        "users": users,
                        "max_users": max_users,
                        "required_tps": profile.tokens_per_second,
                        "target_delay_ms": profile.target_delay_ms,
                    }
                )
                continue

            pts.append(
                {
                    "x": users,
                    "y": result.realtime_factor,
                    "users": users,
                    "max_users": max_users,
                    "per_user_tps": result.per_user_tps,
                    "required_tps": result.required_tps,
                    "total_tps": result.total_tps,
                    "step_ms": result.step_ms,
                    "max_slots": result.max_slots,
                    "target_delay_ms": profile.target_delay_ms,
                }
            )
        if pts:
            datasets.append(
                {
                    "label": _label(am, model, panel_suffix),
                    "data": pts,
                    **_style(model, is_b),
                    "_isRealtime": True,
                }
            )
    return datasets


def embedding_quality_axis_range(
    datasets: list[dict], margin_ratio: float = 0.08, min_margin: float = 0.01
) -> dict[str, float]:
    values: list[float] = []
    for dataset in datasets:
        for point in dataset.get("data", []):
            quality = point.get("quality", point.get("y"))
            if isinstance(quality, (int, float)) and math.isfinite(float(quality)):
                values.append(float(quality))

    if not values:
        return {"y_min": 0.0, "y_max": 1.0}

    lo = min(values)
    hi = max(values)
    span = hi - lo
    margin = max(span * max(margin_ratio, 0.0), max(min_margin, 0.0))
    if span <= 1e-9:
        margin = max(margin, 0.02)

    return {
        "y_min": round(max(0.0, lo - margin), 4),
        "y_max": round(min(1.0, hi + margin), 4),
    }


def _sample_user_exp_curve(points: list[dict], target_rps: float) -> Optional[dict]:
    if not points or target_rps <= 0 or target_rps > points[-1]["arrival_rps"]:
        return None
    if target_rps <= points[0]["arrival_rps"]:
        point = points[0]
        return {
            "arrival_rps": round(target_rps * 100) / 100,
            "response_s": point["response_s"],
            "inflight": round(target_rps * point["response_s"], 1),
            "ttft_ms": point["ttft_ms"],
            "decode_step_ms": point["decode_step_ms"],
        }

    left = points[0]
    right = points[-1]
    for candidate in points[1:]:
        if target_rps <= candidate["arrival_rps"]:
            right = candidate
            break
        left = candidate

    span = right["arrival_rps"] - left["arrival_rps"]
    t = 0.0 if span <= 0 else (target_rps - left["arrival_rps"]) / span
    response_s = left["response_s"] + (right["response_s"] - left["response_s"]) * t
    ttft_ms = left["ttft_ms"] + (right["ttft_ms"] - left["ttft_ms"]) * t
    decode_step_ms = left["decode_step_ms"] + (right["decode_step_ms"] - left["decode_step_ms"]) * t
    return {
        "arrival_rps": round(target_rps * 100) / 100,
        "response_s": round(response_s * 100) / 100,
        "inflight": round(target_rps * response_s, 1),
        "ttft_ms": round(ttft_ms, 1),
        "decode_step_ms": round(decode_step_ms, 1),
    }


def _spec_chart_selection(state, am, model: Model) -> tuple[Optional[SpecRuntime], dict]:
    method = getattr(am, "spec_method", "off") or "off"
    auto = method != "off" and getattr(am, "spec_k", 0) == 0
    selection = spec_optimization_for(state, am, model) if auto else None
    runtime = (
        selection.runtime
        if selection is not None and selection.beneficial
        else None
        if selection is not None
        else resolve_spec_runtime(
            model,
            method,
            getattr(am, "spec_k", 0),
            getattr(state, "spec_acceptance", 0.0),
            am.prec,
        )
    )
    disclosed_runtime = selection.runtime if selection is not None else runtime
    meta = {
        "spec_method": method,
        "spec_k": disclosed_runtime.k if disclosed_runtime is not None else 0,
        "spec_alpha": disclosed_runtime.alpha if disclosed_runtime is not None else 0.0,
        "spec_auto": auto,
        "spec_speedup": selection.speedup
        if selection is not None
        else (disclosed_runtime.probe_speedup if disclosed_runtime is not None else 1.0),
        "spec_beneficial": selection.beneficial if selection is not None else runtime is not None,
        "spec_probe_concurrency": selection.probe_concurrency if selection is not None else 0,
        "spec_reason": selection.reason if selection is not None else "",
    }
    return runtime, meta


def chart_aggregate(
    state,
    batch_sizes: Optional[list[int]] = None,
    panel_suffix: str = "",
    *,
    deployment: Deployment,
) -> list[dict]:
    datasets = []
    eff = state.decode_efficiency
    is_b = panel_suffix != ""
    deployed = deployment.decode
    batch_sizes = batch_sizes or BATCH_SIZES

    agg = []
    for bs in batch_sizes:
        total = 0
        for am in deployed:
            model = am.model
            if getattr(model, "embedding_profile", None) is not None:
                continue
            gpu = am.gpu_spec
            if gpu is None:
                continue
            result = compute_decode(
                model,
                am.tp,
                am.pp,
                bs,
                am.dp,
                gpu,
                state.mu,
                state.profiled_non_kv_gb,
                am.prec,
                state.in_dist,
                state.out_dist,
                eff,
                spec_runtime_for(state, am, model),
            )
            if result:
                total += result.tps
        agg.append({"x": bs, "y": total or None})
    datasets.append(
        {
            "label": f"Node total{panel_suffix}",
            "data": agg,
            "borderColor": "#ddd",
            "backgroundColor": "rgba(255,255,255,0.04)",
            "borderWidth": 2.5,
            "borderDash": [5, 3] if is_b else [],
            "fill": not is_b,
            "tension": 0.3,
            "pointRadius": 2.5,
            "spanGaps": False,
            "_isAggregate": True,
        }
    )

    for am in deployed:
        model = am.model
        if getattr(model, "embedding_profile", None) is not None:
            continue
        gpu = am.gpu_spec
        if gpu is None:
            continue
        pts = []
        for bs in batch_sizes:
            result = compute_decode(
                model,
                am.tp,
                am.pp,
                bs,
                am.dp,
                gpu,
                state.mu,
                state.profiled_non_kv_gb,
                am.prec,
                state.in_dist,
                state.out_dist,
                eff,
                spec_runtime_for(state, am, model),
            )
            pts.append({"x": bs, "y": result.tps if result else None})
        datasets.append(
            {
                "label": f"{model.name}{panel_suffix}",
                "data": pts,
                "borderColor": model.color + ("44" if is_b else "77"),
                "borderWidth": 1,
                "borderDash": [4, 2] if is_b else [],
                "fill": False,
                "tension": 0.3,
                "pointRadius": 1.5,
                "spanGaps": False,
            }
        )
    return datasets


def chart_decode(
    state,
    batch_sizes: Optional[list[int]] = None,
    panel_suffix: str = "",
    *,
    deployment: Deployment,
) -> list[dict]:
    datasets = []
    eff = state.decode_efficiency
    is_b = panel_suffix != ""
    batch_sizes = batch_sizes or BATCH_SIZES

    for am in deployment.decode:
        model = am.model
        if not _is_decode_pareto_model(model):
            continue
        gpu = am.gpu_spec
        if gpu is None:
            continue
        spec, spec_meta = _spec_chart_selection(state, am, model)
        pts = []
        for bs in batch_sizes:
            result = compute_decode(
                model,
                am.tp,
                am.pp,
                bs,
                am.dp,
                gpu,
                state.mu,
                state.profiled_non_kv_gb,
                am.prec,
                state.in_dist,
                state.out_dist,
                eff,
                spec,
            )
            pts.append(
                {
                    "x": bs,
                    "y": result.tps if result else None,
                    **spec_meta,
                    "spec_speedup": result.spec_speedup if result else spec_meta["spec_speedup"],
                }
            )
        datasets.append(
            {
                "label": _label(am, model, panel_suffix, include_prefill=True, spec_meta=spec_meta),
                "data": pts,
                **spec_meta,
                **_style(model, is_b),
                "fill": not is_b,
            }
        )
    return datasets


def _label(
    am,
    model: Model,
    panel_suffix: str = "",
    include_prefill: bool = False,
    spec_meta: Optional[dict] = None,
) -> str:
    decode_label = strategy_label(am.tp, am.pp, am.dp)
    spec_suffix = ""
    if spec_meta is not None:
        if spec_meta["spec_method"] == "off":
            spec_suffix = " · Spec off"
        elif spec_meta["spec_beneficial"]:
            auto = " Auto→" if spec_meta["spec_auto"] else " "
            spec_suffix = (
                f" · {spec_meta['spec_method'].upper()}{auto}k={spec_meta['spec_k']} "
                f"α={spec_meta['spec_alpha']:.0%}"
            )
        else:
            spec_suffix = (
                f" · Spec off (Auto {spec_meta['spec_method'].upper()} @"
                f"{spec_meta['spec_probe_concurrency']}, best k={spec_meta['spec_k']})"
            )
    if include_prefill:
        prefill_label = strategy_label(am.prefill_tp, am.prefill_pp, am.prefill_dp)
        if prefill_label != decode_label:
            return f"{model.name} P {prefill_label} / D {decode_label} {am.prec.upper()}{spec_suffix}{panel_suffix}"
    return f"{model.name} {decode_label} {am.prec.upper()}{spec_suffix}{panel_suffix}"


def chart_user_experience(state, panel_suffix: str = "") -> list[dict]:
    datasets = []
    is_b = panel_suffix != ""

    for am, gpu in _iter_resolved_models(state):
        model = am.model
        if getattr(model, "embedding_profile", None) is not None:
            continue
        points = _user_exp_curve(
            model,
            (am.prefill_tp, am.prefill_pp, am.prefill_dp),
            (am.tp, am.pp, am.dp),
            gpu,
            state.mu,
            state.profiled_non_kv_gb,
            am.prec,
            state.in_dist,
            state.out_dist,
            state.prefix_hit_rate,
            state.prefill_efficiency,
            state.decode_efficiency,
            spec_runtime_for(state, am, model),
        )
        datasets.append(
            {
                "label": _label(am, model, panel_suffix, include_prefill=True),
                "data": points,
                "borderColor": model.color,
                "borderWidth": 1.5 if is_b else 2,
                "borderDash": [5, 3] if is_b else [],
                "fill": False,
                "tension": 0.3,
                "pointRadius": 3,
                "showLine": True,
                "spanGaps": False,
            }
        )
    return datasets


def chart_pareto(state, panel_suffix: str = "", *, deployment: Deployment) -> list[dict]:
    datasets = []
    eff = state.decode_efficiency
    is_b = panel_suffix != ""

    for am in deployment.decode:
        model = am.model
        if not _is_decode_pareto_model(model):
            continue
        gpu = am.gpu_spec
        if gpu is None:
            continue
        spec, spec_meta = _spec_chart_selection(state, am, model)
        pts = []
        for bs in BATCH_SIZES:
            result = compute_decode(
                model,
                am.tp,
                am.pp,
                bs,
                am.dp,
                gpu,
                state.mu,
                state.profiled_non_kv_gb,
                am.prec,
                state.in_dist,
                state.out_dist,
                eff,
                spec,
            )
            if result:
                pts.append(
                    {
                        "x": result.lat,
                        "y": result.tps,
                        "bs": bs,
                        **spec_meta,
                        "spec_speedup": result.spec_speedup,
                    }
                )
        if pts:
            datasets.append(
                {
                    "label": _label(am, model, panel_suffix, spec_meta=spec_meta),
                    "data": pts,
                    **spec_meta,
                    "borderColor": model.color,
                    "backgroundColor": model.color + "AA",
                    "borderWidth": 1.5 if is_b else 2,
                    "borderDash": [5, 3] if is_b else [],
                    "showLine": True,
                    "tension": 0.3,
                    "pointRadius": 3.5,
                }
            )
    return datasets


def _user_exp_curve(
    m: Model,
    prefill_strat: tuple[int, int, int],
    decode_strat: tuple[int, int, int],
    g: GPU,
    mu: float,
    profiled_non_kv_gb: float,
    prec: str,
    in_dist: list[int],
    out_dist: list[int],
    prefix_hit_rate: float,
    prefill_eff: EfficiencyParams,
    decode_eff: EfficiencyParams,
    spec: Optional[SpecRuntime] = None,
) -> list[dict]:
    points: list[dict] = []
    for users in USER_EXP_SWEEP:
        result = compute_user_experience(
            m,
            prefill_strat,
            decode_strat,
            users,
            g,
            mu,
            profiled_non_kv_gb,
            prec,
            in_dist,
            out_dist,
            prefix_hit_rate,
            prefill_eff,
            decode_eff,
            spec,
        )
        if not result or result.arrival_rps <= 0:
            continue
        point = {
            "x": result.arrival_rps,
            "y": result.response_s,
            "arrival_rps": result.arrival_rps,
            "response_s": result.response_s,
            "inflight": result.inflight,
            "ttft_ms": result.ttft_ms,
            "decode_step_ms": result.decode_step_ms,
        }
        if points and point["arrival_rps"] <= points[-1]["arrival_rps"]:
            continue
        points.append(point)
    return points


def chart_asr_quality(state, panel_suffix: str = "") -> list[dict]:
    """Max realtime streams vs published WER, one dot per benchmark/language.

    Concurrency is benchmark-independent in the capacity model, so every dot
    for a given model sits at the same height. WER is static catalog data;
    see PUBLISHED_ASR_WER in data.py.
    """
    datasets = []
    is_b = panel_suffix != ""

    for am, gpu in _iter_resolved_models(state):
        model = am.model
        profile = getattr(model, "realtime_profile", None)
        if profile is None:
            continue
        wer_by_language = PUBLISHED_ASR_WER.get(model.key)
        if not wer_by_language:
            continue

        max_users = compute_realtime_max_users(
            model,
            (am.tp, am.pp, am.dp),
            gpu,
            state.mu,
            state.profiled_non_kv_gb,
            am.prec,
            state.decode_efficiency,
        )
        if max_users <= 0:
            continue

        is_placeholder = model.key in ASR_WER_PLACEHOLDER
        pts = []
        sources = ASR_WER_LANGUAGE_SOURCES.get(model.key, {})
        for language in ASR_WER_LANGUAGES:
            wer = wer_by_language.get(language)
            if wer is None:
                continue
            pts.append(
                {
                    "x": wer,
                    "y": max_users,
                    "language": ASR_WER_LANGUAGE_LABELS.get(language, language),
                    "source": sources.get(language, ""),
                    "wer": wer,
                    "max_users": max_users,
                    "placeholder": is_placeholder,
                    "asr_mode": "streaming"
                    if getattr(profile, "streaming", True)
                    else "non-streaming",
                }
            )
        if not pts:
            continue
        pts.sort(key=lambda p: cast(float, p["x"]))

        datasets.append(
            {
                "label": _label(am, model, panel_suffix),
                "data": pts,
                **_style(model, is_b),
                "backgroundColor": (model.color + "12") if is_placeholder else (model.color + "AA"),
                "showLine": True,
                "tension": 0,
                "pointRadius": 5,
                "_isAsrQuality": True,
                "_placeholder": is_placeholder,
                "_asrStreaming": bool(getattr(profile, "streaming", True)),
                "_modelKey": model.key,
                "_assignmentUid": am.uid,
                "_seriesId": f"asrquality:{'b' if is_b else 'a'}:{am.uid}:{model.key}",
            }
        )
    return datasets


def chart_user_pareto(
    state,
    batch_sizes: Optional[list[int]] = None,
    panel_suffix: str = "",
    *,
    deployment: Deployment,
) -> list[dict]:
    datasets = []
    eff = state.decode_efficiency
    is_b = panel_suffix != ""
    batch_sizes = batch_sizes or BATCH_SIZES

    for am in deployment.decode:
        model = am.model
        if not _is_decode_pareto_model(model):
            continue
        gpu = am.gpu_spec
        if gpu is None:
            continue
        spec, spec_meta = _spec_chart_selection(state, am, model)
        pts = []
        for users in batch_sizes:
            result = compute_decode(
                model,
                am.tp,
                am.pp,
                users,
                am.dp,
                gpu,
                state.mu,
                state.profiled_non_kv_gb,
                am.prec,
                state.in_dist,
                state.out_dist,
                eff,
                spec,
            )
            if result:
                pts.append(
                    {
                        "x": users,
                        "y": round((result.tps / users) * 100) / 100,
                        "users": users,
                        "total_tps": result.tps,
                        "lat": result.lat,
                        "spec_speedup": result.spec_speedup,
                        **{k: v for k, v in spec_meta.items() if k != "spec_speedup"},
                    }
                )
        if pts:
            datasets.append(
                {
                    "label": _label(am, model, panel_suffix, spec_meta=spec_meta),
                    "data": pts,
                    **spec_meta,
                    "borderColor": model.color,
                    "backgroundColor": model.color + "AA",
                    "borderWidth": 1.5 if is_b else 2,
                    "borderDash": [5, 3] if is_b else [],
                    "showLine": True,
                    "tension": 0.3,
                    "pointRadius": 3.5,
                }
            )
    return datasets


def chart_processing_pareto(
    state, batch_sizes: Optional[list[int]] = None, panel_suffix: str = ""
) -> list[dict]:
    datasets = []
    is_b = panel_suffix != ""
    batch_sizes = batch_sizes or DATA_BATCH_SIZES
    deployed = list(_iter_resolved_models(state))

    for idx, (preset_name, preset) in enumerate(DIST_PRESETS.items()):
        in_len = avg_dist(preset["in"], INPUT_BUCKETS)
        out_len = avg_dist(preset["out"], OUTPUT_BUCKETS)
        tokens_per_req = in_len + out_len
        pts = []
        for bs in batch_sizes:
            total_tps = 0
            for am, gpu in deployed:
                if getattr(am.model, "embedding_profile", None) is not None:
                    continue
                result = compute_data(
                    am.model,
                    (am.prefill_tp, am.prefill_pp, am.prefill_dp),
                    (am.tp, am.pp, am.dp),
                    bs,
                    in_len,
                    out_len,
                    gpu,
                    state.mu,
                    state.profiled_non_kv_gb,
                    am.prec,
                    state.prefix_hit_rate,
                    state.prefill_efficiency,
                    state.decode_efficiency,
                    spec_runtime_for(state, am, am.model),
                )
                if result:
                    total_tps += result.tps
            total_rps = (total_tps / tokens_per_req) if tokens_per_req > 0 else 0.0
            pts.append(
                {
                    "x": bs,
                    "y": round(total_rps * 100) / 100 if total_tps else None,
                    "rps": round(total_rps * 100) / 100 if total_tps else None,
                    "tps": total_tps or None,
                    "in_len": in_len,
                    "out_len": out_len,
                    "workload": preset_name,
                }
            )

        color = PROCESSING_PARETO_COLORS[idx % len(PROCESSING_PARETO_COLORS)]
        datasets.append(
            {
                "label": f"{preset_name}{panel_suffix}",
                "data": pts,
                "borderColor": color,
                "backgroundColor": color + "12",
                "borderWidth": 1.5 if is_b else 2,
                "borderDash": [5, 3] if is_b else [],
                "fill": False,
                "tension": 0.3,
                "pointRadius": 2.5,
                "spanGaps": False,
            }
        )
    return datasets


def compute_user_exp_table(state) -> list[dict]:
    """Build the sampled user-experience table presentation model."""
    rows = []
    for am, gpu in _iter_resolved_models(state):
        model = am.model
        if getattr(model, "embedding_profile", None) is not None:
            continue
        points = _user_exp_curve(
            model,
            (am.prefill_tp, am.prefill_pp, am.prefill_dp),
            (am.tp, am.pp, am.dp),
            gpu,
            state.mu,
            state.profiled_non_kv_gb,
            am.prec,
            state.in_dist,
            state.out_dist,
            state.prefix_hit_rate,
            state.prefill_efficiency,
            state.decode_efficiency,
            spec_runtime_for(state, am, model),
        )
        if not points:
            continue
        peak = points[-1]
        cells: list[dict | None] = []
        for frac in USER_EXP_FRACTIONS:
            sample = _sample_user_exp_curve(points, peak["arrival_rps"] * frac)
            if sample is None:
                cells.append(None)
                continue
            cells.append(
                {
                    "lat": round(sample["decode_step_ms"], 1),
                    "resp_s": round(sample["response_s"], 2),
                    "ttft_ms": round(sample["ttft_ms"], 1),
                }
            )
        rows.append(
            {
                "model": model,
                "config": f"{strategy_label(am.tp, am.pp, am.dp)} {am.prec.upper()}",
                "prec": am.prec,
                "peak_rps": round(peak["arrival_rps"] * 100) / 100,
                "peak_resp_s": round(peak["response_s"] * 100) / 100,
                "peak_inflight": round(peak["inflight"], 1),
                "cells": cells,
            }
        )
    return rows
