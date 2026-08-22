"""Plain-text projection report.

The copy-to-clipboard report behind ``/api/projection-report``. Pure: it
takes planner state and returns a string, with no request context, so it is
characterized by golden fixtures rather than exercised through the client.
"""

from __future__ import annotations

import cloud_policy
from calc import strategy_label
from data import (
    MODELS,
    PRECISION_LABELS,
    effective_quality,
    quality_to_aa_intelligence,
)
from engine.economics import compute_revenue_projection
from presentation.formatting import fmt_money, fmt_num
from presentation.model_cards import get_model_info
from state import AUTO_MODEL_STRATEGY_LABELS, PlannerState


def _fmt_pct(value: float, decimals: int = 0) -> str:
    return f"{float(value or 0.0):.{decimals}f}%"


def _projection_diagnostic(row: dict) -> str:
    proj = row["project"]
    difficulty_index = quality_to_aa_intelligence(float(proj.difficulty))
    slo = round(float(row["min_success_rate"]) * 100)
    if row["served_pct"] > 99.5:
        msg = "Fully served internally"
        if row["any_suboptimal"]:
            msg += " via a stretched model; extra tokens spent"
        return msg
    if row["cap_blocked_for_project"]:
        caps = ", ".join(row["requires"]) or "required capabilities"
        return f"No deployed model supplies {caps}; {round(row['leaked_pct'] + row['destroyed_pct'])}% cannot be served on-prem."
    if row.get("quality_floor_blocked_for_project"):
        mix = row.get("quality_mix_label", row.get("quality_domain_label", "General"))
        return f"No deployed model clears quality Q {row['quality_floor']:.2f} for {mix}."
    if row["slo_blocked_for_project"]:
        return f"No deployed model meets the {slo}% SLO at difficulty index {difficulty_index:.1f}."
    if row["served"] > 0 and row["spilled"] > 0:
        msg = f"Add GPUs; capacity saturated at difficulty index {difficulty_index:.1f}, {round(row['spilled_pct'])}% spills to cloud ({fmt_money(row['value_spilled'])}/day leaking)"
        if row["any_suboptimal"]:
            msg += "; some served via a stretched model"
        return msg
    if row["destroyed"] > 0 and row["cloud_blocked"]:
        return f"No compatible cloud model; {round(row['destroyed_pct'])}% of demand is shelved."
    if row["leaked"] > 0 and not row["has_compatible"]:
        return f"No compatible model deployed; {round(row['leaked_pct'])}% flees to cloud ({fmt_money(row['value_leaked'])}/day leaking)."
    if row["leaked"] > 0:
        return f"On-prem $/M exceeds the ceiling; {fmt_money(row['value_leaked'])}/day leaks."
    if row["destroyed"] > 0:
        return f"Cloud ({row['cloud_label']}: ${row['cloud_pm']:.2f}/M) exceeds WTP; {round(row['destroyed_pct'])}% is shelved."
    return "No routed demand."


def _format_projection_report_for_state(state: PlannerState, label: str) -> str:
    p = compute_revenue_projection(state)
    policy = cloud_policy.summary()
    f = p["fates"]
    lines = [
        label,
        "=" * len(label),
        "",
        "Deployment",
        f"- GPUs: {sum(g.count for g in state.gpus)} total, {sum(m.gpu_count for m in state.models)} assigned",
        f"- Auto model selection: {'on' if state.auto_mode else 'off'}",
        f"- Auto strategy: {AUTO_MODEL_STRATEGY_LABELS.get(getattr(state, 'auto_strategy', ''), AUTO_MODEL_STRATEGY_LABELS['balanced'])}",
        f"- gpu_mem_util: {state.mu:.2f}",
        f"- Profiled non-KV runtime memory: {state.profiled_non_kv_gb:g} GB/GPU",
        f"- Empirical prefix-token reuse (portfolio average): {state.prefix_hit_rate * 100:.0f}%",
    ]
    if state.spec_acceptance > 0:
        lines.append(
            f"- Speculative acceptance override: {state.spec_acceptance * 100:.0f}% per-token alpha for all drafters"
        )
    gateway = cloud_policy.corpo_presets().get(getattr(state, "corpo_cloud", ""), {})
    lines.append(
        f"- Cloud gateway: {gateway.get('label', getattr(state, 'corpo_cloud', 'current'))}"
    )
    if state.auto_excluded:
        excluded = [MODELS[key].name if key in MODELS else key for key in state.auto_excluded]
        lines.append(f"- Excluded from auto: {', '.join(excluded)}")
    if policy["active"]:
        lines.append(
            f"- Cloud policy: active; {policy['allowed_count']}/{policy['total_count']} models allowed, "
            f"{len(policy['overridden'])} price override(s)"
        )

    if state.gpus:
        lines.append("- GPU pools:")
        for gp in state.gpus:
            g = gp.gpu
            free = state.free_gpu_for_pool(gp.uid)
            cost = (
                f"${gp.cost_per_gpu_hour:.2f}/GPU-hr" if gp.cost_per_gpu_hour > 0 else "TCO not set"
            )
            lines.append(
                f"  - {g.name}: {gp.count} GPUs ({free} free), {g.bw_tbs:.1f} TB/s, {g.vendor_label}, {cost}"
            )

    if state.models:
        lines.extend(["", "Deployed Models"])
        for am in state.models:
            model = am.model
            gp = state.find_gpu(am.gpu_uid)
            gpu_name = gp.gpu.name if gp else "No GPU pool"
            prec = PRECISION_LABELS.get(am.prec, am.prec.upper())
            if model.is_embedding_model:
                ep = model.embedding_profile
                lines.append(
                    f"- {model.name}: {model.size_label}, {prec}, {ep.mode_label}, {ep.output_dim}d"
                    f"{f' / late {ep.late_interaction_dim}d' if ep.late_interaction_dim else ''}; "
                    f"{gpu_name} x{am.gpu_count}; E {strategy_label(am.prefill_tp, am.prefill_pp, am.prefill_dp)}"
                )
                continue
            moe = ""
            if model.is_moe:
                moe = f", {model.active_params / 1e9:.1f}B active"
            lines.append(
                f"- {model.name}: {model.size_label}, {prec}, Q {effective_quality(model):.2f} effective "
                f"(raw {model.quality:.2f}, conf {model.quality_confidence:.0%}), eta {model.token_efficiency:.2f}x{moe}; "
                f"{gpu_name} x{am.gpu_count}; P {strategy_label(am.prefill_tp, am.prefill_pp, am.prefill_dp)}, "
                f"D {strategy_label(am.tp, am.pp, am.dp)}"
            )
            spec_info = get_model_info(state, am).get("spec")
            if spec_info is not None:
                spec = spec_info
                auto_label = (
                    f"Auto selected k={spec['k']}"
                    if am.spec_k == 0 and spec["active"]
                    else f"Auto kept spec off (best candidate k={spec['k']})"
                    if am.spec_k == 0
                    else f"manual k={spec['k']}"
                )
                lines.append(
                    f"  - Spec decoding: {spec['profile'].label}, {auto_label}, "
                    f"alpha {spec['alpha']:.2f} ({spec['alpha_source']}), tau ~{spec['tau']:.2f} tok/cycle, "
                    f"+{spec['draft_gb']:.1f} GB draft weights, modeled {spec['speedup']:.2f}x "
                    f"at {spec['probe_bs']} users; lossless verification; "
                    f"draft/verification/KV/scheduler costs are modeled and speedup erodes as batch turns compute-bound. "
                    f"Source: {spec['profile'].source}."
                )

    lines.extend(
        [
            "",
            "Economic Impact",
            f"- Owner revenue: {fmt_money(p['value_served_day'])}/day captured on your GPUs",
            f"- Owner margin: {fmt_money(p['margin_day'])}/day after {fmt_money(p['cost_day'])}/day cluster cost"
            if p["cost_day"] > 0
            else "- Owner margin: set TCO $/GPU-hr to see",
            f"- Demand: {fmt_num(f['total_tokens'])} tokens/day across {len(state.projects)} use cases",
            f"  baseline {fmt_num(p['baseline_tokens_day'])} + latent active {fmt_num(p['latent_active_tokens_day'])}",
            f"- Served internally: {_fmt_pct(f['served_pct'])} ({fmt_num(f['served_tokens'])} tok)",
            f"- Spilled to cloud: {_fmt_pct(f['spilled_pct'])} ({fmt_num(f['spilled_tokens'])} tok)",
            f"- Leaked to cloud: {_fmt_pct(f['leaked_pct'])} ({fmt_num(f['leaked_tokens'])} tok)",
            f"- Cloud spend lost from on-prem: {fmt_money(p['value_cloud_day'])}/day",
            f"- Destroyed: {_fmt_pct(f['destroyed_pct'])} ({fmt_num(f['destroyed_tokens'])} tok)",
            f"- Token coverage: {_fmt_pct(p['token_coverage'] * 100)}",
            f"- Value capture: {_fmt_pct(p['value_capture_rate'] * 100)}",
            f"- Revenue multiple: {p['revenue_multiple']:.2f}x"
            if p["cost_day"] > 0
            else "- Revenue multiple: set TCO $/GPU-hr to see",
            f"- CO2: {p['co2_kg_day_total']:.1f} kg/day"
            if p["co2_kg_day_total"] > 0
            else "- CO2: set GPU TDP data to see",
        ]
    )

    if p.get("recommendations"):
        lines.extend(["", "Best Next GPU"])
        for idx, rec in enumerate(p["recommendations"][:3], 1):
            lines.append(
                f"{idx}. +{rec['added_gpus']} {rec['gpu_name']} to {rec['model_name']}: "
                f"{fmt_money(rec['margin_gain_day'])}/day margin, "
                f"{fmt_money(rec['cloud_reduced_day'])}/day cloud avoided, "
                f"{fmt_money(rec['destroyed_reduced_day'])}/day destroyed demand recovered, "
                f"+{fmt_num(rec['served_gain_tokens'])} tok/day served"
            )

    if p["models"]:
        lines.extend(["", "Internal User Price ($/1M tokens)"])
        for m in p["models"]:
            if m["runnable"] and m["internal_pm"] > 0:
                blended = f"${m['internal_pm']:.2f}/M blended"
                input_price = f"in ${m['internal_input_pm']:.2f}/M"
                output_price = f"out ${m['internal_output_pm']:.2f}/M"
                price = f"{input_price}, {output_price}, {blended}"
            elif m["runnable"]:
                price = "price unavailable — set TCO $/GPU-hr"
            else:
                price = "not runnable"
            lines.append(f"- {m['name']}: {price}")

    if p["projects"]:
        lines.extend(["", "Per Use Case"])
        for row in p["projects"]:
            proj = row["project"]
            cloud = (
                "Cloud ref: blocked"
                if row["cloud_blocked"]
                else f"Cloud ref: {row['cloud_label']} at ${row['cloud_pm']:.2f}/M"
            )
            fate = (
                f"{_fmt_pct(row['served_pct'])} served, "
                f"{_fmt_pct(row['spilled_pct'] + row['leaked_pct'])} to cloud, "
                f"{_fmt_pct(row['destroyed_pct'])} destroyed"
            )
            lines.append(
                f"- {proj.name}: {row.get('quality_mix_label', row.get('quality_domain_label', 'General'))} quality mix, "
                f"AA difficulty index {quality_to_aa_intelligence(proj.difficulty):.1f}, SLO {round(row['min_success_rate'] * 100)}%, "
                f"floor Q {row.get('quality_floor', 0.0):.2f}, "
                f"empirical prefix reuse {row['prefix_hit_rate'] * 100:.0f}%, "
                f"WTP ${proj.wtp_per_m:.2f}/M; {cloud}; {fate}."
            )
            parts = []
            if row["any_served"]:
                parts.extend(
                    [
                        f"{fmt_money(row['value_served'])}/day served",
                        f"margin {fmt_money(row['margin_day'])}/day",
                    ]
                )
            if row["value_spilled"] + row["value_leaked"] > 0:
                parts.append(
                    f"{fmt_money(row['value_spilled'] + row['value_leaked'])}/day paid to cloud"
                )
            if row["value_destroyed"] > 0:
                parts.append(f"{fmt_money(row['value_destroyed'])}/day destroyed")
            if parts:
                lines.append(f"  {', '.join(parts)}.")
            lines.append(f"  {_projection_diagnostic(row)}")
            if row["latent_unlocked"]:
                lines.append(
                    f"  Latent pool active: +{fmt_num(row['latent_active_tokens'])} tok/day "
                    f"({row['latent_activation_pct']:.0f}% of pool) at ${row['cheapest_effective_pm']:.2f}/M."
                )

    if p["models"]:
        lines.extend(["", "Supply"])
        for m in p["models"]:
            status = m.get("status") or (
                "SATURATED"
                if m["saturated"]
                else ("IDLE" if m["runnable"] and m["utilization"] < 0.05 else "OK")
            )
            price = f"${m['internal_pm']:.2f}/M" if m["runnable"] and m["internal_pm"] > 0 else "-"
            lines.append(
                f"- {m['name']}: Q {m['effective_quality'] * 100:.0f}% effective, {m['gpu_count']} GPUs, "
                f"cap {fmt_num(m['daily_tokens_cap'])} tok/day, placed {fmt_num(m['served_tokens'])}, "
                f"util {m['utilization'] * 100:.0f}%, internal {price}, {status}"
            )

    return "\n".join(lines)


def format_projection_report(state_a: PlannerState, state_b: PlannerState | None) -> str:
    title = "GPU/LLM Usage Modeler report"
    parts = [title, "=" * len(title), ""]
    parts.append(
        _format_projection_report_for_state(state_a, "Config A" if state_b else "Current Config")
    )
    if state_b:
        parts.extend(["", "", _format_projection_report_for_state(state_b, "Config B")])
    parts.extend(
        [
            "",
            "Notes",
            "Roofline estimates; continuous-batching approximation; separate prefill/decode efficiency knobs; KV capacity anchored to requested accelerator memory minus weights and profiled non-KV runtime memory.",
        ]
    )
    return "\n".join(parts).strip() + "\n"
