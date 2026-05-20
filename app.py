"""Flask application for vLLM Multi-Model Planner."""

import json
import math
import os
import uuid
from collections import defaultdict
from pathlib import Path

from flask import Flask, g, jsonify, make_response, redirect, render_template, request, session, url_for

from data import (
    GPUS,
    MODELS,
    DIST_PRESETS,
    TASK_PRESETS,
    INPUT_BUCKETS,
    OUTPUT_BUCKETS,
    BATCH_SIZES,
    DAY_SHAPES,
    CLOUD_MODELS,
    CORPO_CLOUD_PRESETS,
    PROJECT_PRESETS,
    MODEL_CAPABILITIES,
    CAPABILITY_LABELS,
    SCALE_MODELS,
    COUNTRIES,
    CARBON_INTENSITY_HOURLY,
    PRECISIONS,
    PRECISION_LABELS,
    PRECISION_DESCRIPTIONS,
    models_by_category,
    gpu_cards_by_vendor,
    gpus_by_vendor,
    required_quality,
    effective_quality,
    success_rate,
)
from calc import (
    avg_dist,
    chart_processing_pareto,
    chart_user_pareto,
    compute_revenue_projection,
    get_decode_bs,
    get_processing_pareto_bs,
    dist_percentile,
    effective_prefill_length,
    normalize_dist,
    strategy_label,
    valid_strategies,
)
from state import (
    AUTO_MODEL_STRATEGIES,
    AUTO_MODEL_STRATEGY_LABELS,
    PlannerState,
    VISIBLE_PLOT_MODES,
    add_gpu,
    add_model,
    add_project,
    add_use_case_def,
    auto_exclude_model,
    auto_reallow_model,
    auto_select_models,
    change_gpu_qty,
    clear_compare_state,
    create_default_state,
    duplicate_compare_state,
    get_compare_state,
    get_model_info,
    get_model_infos,
    get_state,
    get_use_case_defs,
    remove_gpu,
    remove_model,
    remove_project,
    remove_use_case_def,
    retune_models,
    normalize_plot_mode,
    normalize_auto_strategy,
    replace_project_set,
    replace_use_case_defs,
    serialize_project_set,
    serialize_use_case_defs,
    project_scale_config,
    format_scale_value,
    scale_decimals,
    set_dist_preset,
    set_dist_value,
    set_gpu_cost,
    set_model_gpu_count,
    set_model_gpu_pool,
    set_model_prec,
    set_model_strat,
    set_prefix_hit_rate,
    set_project_batch_eligible,
    set_project_capability,
    set_project_dist_preset,
    set_project_field,
    set_project_kind,
    set_project_name,
    set_project_scale_value,
    set_use_case_def_capability,
    set_use_case_def_field,
    set_projection_choice,
    set_projection_pct,
    set_projection_toggle,
)
from tracking import SnapshotStore


BASE_DIR = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_dotenv(BASE_DIR / ".env")


app = Flask(__name__)
app.secret_key = os.environ.get("PLANNER_SECRET_KEY", "vllm-planner-dev-key")

VISITOR_COOKIE = "planner_vid"
TAB_PARAM = "tab_id"
ADMIN_SESSION_KEY = "planner_admin_ok"
SNAPSHOT_STORE = SnapshotStore(BASE_DIR / "instance" / "planner_snapshots.json")


USE_CASE_DETAILS = {
    "classify": {
        "summary": "High-volume, low-latency categorization, routing, tagging, and extraction work where the answer is usually a compact label or short structured record.",
        "examples": (
            "Ticket triage, document labels, compliance flags, PII detection, support intent routing.",
            "Offline backfills where a large archive becomes worth processing only if token cost falls far enough.",
        ),
        "why": (
            "Difficulty is low because correctness usually depends on pattern recognition and schema adherence, not deep reasoning.",
            "Short input and output shapes keep decode pressure low; the economic constraint is usually price per million tokens.",
            "Batch eligibility reflects the fact that most classification queues can be shifted away from peak hours.",
        ),
        "routing": (
            "Small and mid-sized models should win when their quality clears the SLO, because token price matters more than frontier reasoning.",
            "Large latent demand makes this a good probe for whether cheaper internal serving creates new work instead of just replacing cloud spend.",
        ),
    },
    "summarize": {
        "summary": "Long-document compression where users need faithful summaries, extracted takeaways, or briefings from sizeable source material.",
        "examples": (
            "Contracts, research papers, meeting transcripts, policy documents, incident reports.",
            "Periodic knowledge-base digestion or archive summarization jobs.",
        ),
        "why": (
            "The long-document input preset stresses prefill and KV capacity more than a normal chat workload.",
            "The long output preset assumes the answer is not just a label; the model must produce a useful narrative artifact.",
            "The long-context requirement prevents routing to models that cannot safely ingest the source material.",
        ),
        "routing": (
            "Models with strong long-context behavior and acceptable token efficiency should beat tiny models that summarize cheaply but unreliably.",
            "Batch eligibility means capacity planning can often use night-batching levers without hurting user experience.",
        ),
    },
    "chatbot": {
        "summary": "Interactive assistant traffic for customer or employee support, with tool calls and a strict quality floor.",
        "examples": (
            "Customer-service bot, IT helpdesk agent, HR policy assistant, product support copilot.",
            "Short-turn interactive sessions where the answer must be good enough on the first try.",
        ),
        "why": (
            "The strict SLO models a user-facing workflow where bad answers create escalation cost.",
            "Tool use is required because real support flows often need retrieval, ticket lookup, account actions, or workflow APIs.",
            "The chat-shaped distribution keeps this closer to regular interactive prompt and response lengths.",
        ),
        "routing": (
            "Capacity should prioritize low latency and reliable quality over the absolute cheapest token path.",
            "Because it is not batch eligible, this preset competes for daytime peak capacity.",
        ),
    },
    "email_corrector": {
        "summary": "High-frequency writing assistance for short business messages, where scale is usually driven by headcount and message volume.",
        "examples": (
            "Email correction, tone adjustment, short reply drafting, grammar fixes, translation polish.",
            "Employee productivity copilots embedded in mail or chat tools.",
        ),
        "why": (
            "Difficulty is modest: the model mostly rewrites or corrects rather than solving deep tasks.",
            "Chat-shaped input/output keeps the workload interactive and short-turn.",
            "Scale should be set from staff count, adoption rate, and messages per employee rather than inherited from the preset.",
        ),
        "routing": (
            "Smaller models should clear this workload when the SLO is reasonable, making price and latency central.",
            "Because demand follows people rather than documents, large organizations can make this small-looking kind dominate volume.",
        ),
    },
    "coding": {
        "summary": "Developer-assistant traffic for code explanation, edits, generation, and repository-aware workflows.",
        "examples": (
            "IDE assistant, code review helper, test generation, bug diagnosis, migration support.",
            "Longer prompts that include files, logs, stack traces, or design constraints.",
        ),
        "why": (
            "Higher difficulty reflects the need for multi-step reasoning, syntax precision, and domain context.",
            "Tool and long-context gates represent repository search, file inspection, and large prompt windows.",
            "The code-shaped input and output distributions produce more decode work than short chat or classification tasks.",
        ),
        "routing": (
            "Quality failures are expensive, so cheap models may be filtered even when they look attractive on raw throughput.",
            "Token efficiency matters because coding models can produce long intermediate reasoning or verbose patches.",
        ),
    },
    "meeting_notes": {
        "summary": "Transcript-to-summary workflows for meetings, calls, and interviews that can usually tolerate delayed processing.",
        "examples": (
            "Meeting minutes, action-item extraction, call summaries, interview digests.",
            "Team-wide transcription backfills or daily note generation.",
        ),
        "why": (
            "Long-document input captures transcript ingestion; long-document output captures useful summaries.",
            "The long-context gate prevents routing to models that cannot safely hold the source material.",
            "Batch eligibility reflects the fact that most summaries can be delivered minutes later or overnight.",
        ),
        "routing": (
            "This kind often benefits from night batching because immediacy is less important than cost.",
            "Scale comes from recorded hours and transcript length, not from the task definition itself.",
        ),
    },
    "evals": {
        "summary": "Offline evaluation, grading, judging, and scoring workloads where many prompts are processed in batches.",
        "examples": (
            "Model eval suites, regression checks, judge-model scoring, safety review queues, benchmark runs.",
            "RAG-style inputs that end in short verdicts, scores, labels, or pass/fail judgments.",
        ),
        "why": (
            "Moderate difficulty and a high SLO model judge workloads where consistency matters more than creative generation.",
            "The RAG input plus classification output shape captures long evidence with compact decisions.",
            "Batch eligibility is central: evals can usually wait for cheaper off-peak GPU capacity.",
        ),
        "routing": (
            "This preset should expose whether the cluster has spare batch capacity after interactive demand is served.",
            "Latent demand models eval coverage that teams skip until per-token cost is low enough.",
        ),
    },
    "inbox_archive": {
        "summary": "Large personal or organizational inbox archives turned into searchable, summarized, or queryable knowledge bases.",
        "examples": (
            "A decade of executive email, team inbox backfills, legal or discovery-oriented mail analysis.",
            "One-time corpus digestion followed by much smaller incremental updates.",
        ),
        "why": (
            "The scale driver is corpus size: mailboxes, retained years, attachments, and cleanup policy.",
            "RAG-style input with long-form output reflects retrieval-backed synthesis over many messages.",
            "Large latent demand models the work that organizations postpone until unit cost drops.",
        ),
        "routing": (
            "The preset stresses batch economics more than interactive latency.",
            "It should be added at a scale that reflects the corpus, not the number of daily users.",
        ),
    },
    "longctx": {
        "summary": "Analytical workflows over very large context windows, where the prompt itself is the expensive object.",
        "examples": (
            "Log and trace analysis, legal discovery, financial filings, multi-document comparison, technical due diligence.",
            "Large source packs where retrieval is not enough and the model must reason across the full context.",
        ),
        "why": (
            "High difficulty and long-context gating force routing toward models that can both ingest and reason over large inputs.",
            "Long input and output shapes stress memory, prefill, and sustained decode capacity.",
            "The smaller base demand reflects a high-value workflow that is costly enough to stay constrained.",
        ),
        "routing": (
            "KV capacity can dominate here; a model that is cheap per token but memory-starved may still be the wrong fit.",
            "Latent demand represents analyses that become practical only when internal cost drops below the unlock threshold.",
        ),
    },
    "research": {
        "summary": "High-value agentic research where the model gathers information, reasons through tradeoffs, and produces a substantial answer.",
        "examples": (
            "Market research, technical investigation, strategy briefs, due diligence, multi-source synthesis.",
            "Agent loops that combine tools, retrieval, reasoning, and long-form synthesis.",
        ),
        "why": (
            "The highest difficulty and SLO reserve this workload for models with strong reasoning behavior.",
            "Tool and reasoning requirements encode the fact that this is an agent workflow, not plain autocomplete.",
            "RAG input plus long-document output models substantial source context and a detailed final deliverable.",
        ),
        "routing": (
            "Low volume but high WTP means this can justify expensive frontier or large open-weight models.",
            "If no deployed model clears the SLO, demand should leak to cloud rather than be counted as served.",
        ),
    },
}


def _new_id() -> str:
    return str(uuid.uuid4())


def _visitor_id() -> str:
    visitor_id = getattr(g, "visitor_id", None)
    if visitor_id:
        return visitor_id

    visitor_id = request.cookies.get(VISITOR_COOKIE)
    if not visitor_id:
        visitor_id = _new_id()
    g.visitor_id = visitor_id
    return visitor_id


def _tab_id(optional: bool = False) -> str | None:
    tab_id = (
        request.headers.get("X-Tab-ID")
        or request.form.get(TAB_PARAM)
    )
    if tab_id:
        return tab_id
    return None if optional else "default"


def _scope_id() -> str:
    return f"{_visitor_id()}:{_tab_id()}"


@app.after_request
def _set_identity_cookie(response):
    visitor_id = getattr(g, "visitor_id", None)
    if visitor_id and request.cookies.get(VISITOR_COOKIE) != visitor_id:
        response.set_cookie(
            VISITOR_COOKIE,
            visitor_id,
            max_age=60 * 60 * 24 * 365,
            httponly=True,
            samesite="Lax",
            secure=request.is_secure,
        )
    return response


def _admin_password() -> str | None:
    return os.environ.get("PLANNER_ADMIN_PASSWORD")


def _is_admin_authenticated() -> bool:
    return bool(session.get(ADMIN_SESSION_KEY))


def _state(panel: str = "A") -> PlannerState | None:
    if panel == "B":
        return get_compare_state(_scope_id())
    return get_state(_scope_id())


def _request_state() -> PlannerState | None:
    return _state(request.values.get("panel", "A"))


def _template_context() -> dict:
    return {
        "GPUS": GPUS,
        "MODELS": MODELS,
        "DIST_PRESETS": DIST_PRESETS,
        "TASK_PRESETS": TASK_PRESETS,
        "DAY_SHAPES": DAY_SHAPES,
        "CLOUD_MODELS": CLOUD_MODELS,
        "CORPO_CLOUD_PRESETS": CORPO_CLOUD_PRESETS,
        "PROJECT_PRESETS": PROJECT_PRESETS,
        "MODEL_CAPABILITIES": MODEL_CAPABILITIES,
        "CAPABILITY_LABELS": CAPABILITY_LABELS,
        "SCALE_MODELS": SCALE_MODELS,
        "COUNTRIES": COUNTRIES,
        "VISIBLE_PLOT_MODES": VISIBLE_PLOT_MODES,
        "INPUT_BUCKETS": INPUT_BUCKETS,
        "OUTPUT_BUCKETS": OUTPUT_BUCKETS,
        "BATCH_SIZES": BATCH_SIZES,
        "PRECISIONS": PRECISIONS,
        "PRECISION_LABELS": PRECISION_LABELS,
        "PRECISION_DESCRIPTIONS": PRECISION_DESCRIPTIONS,
        "AUTO_MODEL_STRATEGIES": AUTO_MODEL_STRATEGIES,
        "models_by_category": models_by_category,
        "gpu_cards_by_vendor": gpu_cards_by_vendor,
        "gpus_by_vendor": gpus_by_vendor,
        "normalize_dist": normalize_dist,
        "avg_dist": avg_dist,
        "dist_percentile": dist_percentile,
        "get_model_info": get_model_info,
        "get_model_infos": get_model_infos,
        "compute_revenue_projection": compute_revenue_projection,
        "valid_strategies": valid_strategies,
        "effective_prefill_length": effective_prefill_length,
        "strategy_label": strategy_label,
        "required_quality": required_quality,
        "effective_quality": effective_quality,
        "success_rate": success_rate,
        "project_scale_config": project_scale_config,
        "format_scale_value": format_scale_value,
        "scale_decimals": scale_decimals,
        "math": math,
    }


def _record_snapshot(reason: str, state_a: PlannerState, state_b: PlannerState | None, path: str | None = None):
    SNAPSHOT_STORE.record_snapshot(
        visitor_id=_visitor_id(),
        tab_id=_tab_id() or "default",
        reason=reason,
        path=path or request.path,
        state_a=state_a,
        state_b=state_b,
    )


def _htmx_response(state_a=None):
    # Always render the canonical A/B pair. Callers may pass the mutated panel,
    # which is panel B for compare-side edits.
    sa = get_state(_scope_id())
    sb = get_compare_state(_scope_id())
    resp = make_response(render_template("partials/htmx_response.html", A=sa, B=sb, **_template_context()))
    resp.headers["HX-Trigger"] = "refreshChart"
    return resp


def _tracked_htmx_response(reason: str, state_a: PlannerState | None = None):
    # Always render and snapshot the canonical A/B pair. Callers may pass the
    # mutated panel, which is panel B for compare-side edits.
    sa = get_state(_scope_id())
    sb = get_compare_state(_scope_id())
    _record_snapshot(reason, sa, sb)
    resp = make_response(render_template("partials/htmx_response.html", A=sa, B=sb, **_template_context()))
    resp.headers["HX-Trigger"] = "refreshChart"
    return resp


def _fmt_pct(value: float, decimals: int = 0) -> str:
    return f"{float(value or 0.0):.{decimals}f}%"


def _projection_diagnostic(row: dict) -> str:
    proj = row["project"]
    elo = round(float(proj.difficulty) * 3000)
    slo = round(float(row["min_success_rate"]) * 100)
    if row["served_pct"] > 99.5:
        msg = "Fully served internally"
        if row["any_suboptimal"]:
            msg += " via a stretched model; extra tokens spent"
        return msg
    if row["cap_blocked_for_project"]:
        caps = ", ".join(row["requires"]) or "required capabilities"
        return f"No deployed model supplies {caps}; {round(row['leaked_pct'] + row['destroyed_pct'])}% cannot be served on-prem."
    if row["slo_blocked_for_project"]:
        return f"No deployed model meets the {slo}% SLO at ELO {elo}."
    if row["served"] > 0 and row["spilled"] > 0:
        msg = f"Add GPUs; capacity saturated at ELO {elo}, {round(row['spilled_pct'])}% spills to cloud ({fmt_money(row['value_spilled'])}/day leaking)"
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
        f"- Prefix hit rate: {state.prefix_hit_rate * 100:.0f}%",
    ]
    if state.auto_excluded:
        excluded = [MODELS[key].name if key in MODELS else key for key in state.auto_excluded]
        lines.append(f"- Excluded from auto: {', '.join(excluded)}")

    if state.gpus:
        lines.append("- GPU pools:")
        for gp in state.gpus:
            g = gp.gpu
            free = state.free_gpu_for_pool(gp.uid)
            cost = f"${gp.cost_per_gpu_hour:.2f}/GPU-hr" if gp.cost_per_gpu_hour > 0 else "TCO not set"
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
            moe = ""
            if model.is_moe:
                moe = f", {model.active_params / 1e9:.1f}B active"
            lines.append(
                f"- {model.name}: {model.size_label}, {prec}, Q {effective_quality(model):.2f} effective "
                f"(raw {model.quality:.2f}, conf {model.quality_confidence:.0%}), eta {model.token_efficiency:.2f}x{moe}; "
                f"{gpu_name} x{am.gpu_count}; P {strategy_label(am.prefill_tp, am.prefill_pp, am.prefill_dp)}, "
                f"D {strategy_label(am.tp, am.pp, am.dp)}"
            )

    lines.extend([
        "",
        "Economic Impact",
        f"- Owner revenue: {fmt_money(p['value_served_day'])}/day captured on your GPUs",
        f"- Owner margin: {fmt_money(p['margin_day'])}/day after {fmt_money(p['cost_day'])}/day cluster cost" if p["cost_day"] > 0 else "- Owner margin: set TCO $/GPU-hr to see",
        f"- Demand: {fmt_num(f['total_tokens'])} tokens/day across {len(state.projects)} use cases",
        f"  baseline {fmt_num(p['baseline_tokens_day'])} + latent active {fmt_num(p['latent_active_tokens_day'])}",
        f"- Served internally: {_fmt_pct(f['served_pct'])} ({fmt_num(f['served_tokens'])} tok)",
        f"- Spilled to cloud: {_fmt_pct(f['spilled_pct'])} ({fmt_num(f['spilled_tokens'])} tok)",
        f"- Leaked to cloud: {_fmt_pct(f['leaked_pct'])} ({fmt_num(f['leaked_tokens'])} tok)",
        f"- Cloud outflow: {fmt_money(p['value_cloud_day'])}/day",
        f"- Destroyed: {_fmt_pct(f['destroyed_pct'])} ({fmt_num(f['destroyed_tokens'])} tok)",
        f"- Token coverage: {_fmt_pct(p['token_coverage'] * 100)}",
        f"- Value capture: {_fmt_pct(p['value_capture_rate'] * 100)}",
        f"- Revenue multiple: {p['revenue_multiple']:.2f}x" if p["cost_day"] > 0 else "- Revenue multiple: set TCO $/GPU-hr to see",
        f"- CO2: {p['co2_kg_day_total']:.1f} kg/day" if p["co2_kg_day_total"] > 0 else "- CO2: set GPU TDP data to see",
    ])

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
            cloud = "Cloud ref: blocked" if row["cloud_blocked"] else f"Cloud ref: {row['cloud_label']} at ${row['cloud_pm']:.2f}/M"
            fate = (
                f"{_fmt_pct(row['served_pct'])} served, "
                f"{_fmt_pct(row['spilled_pct'] + row['leaked_pct'])} to cloud, "
                f"{_fmt_pct(row['destroyed_pct'])} destroyed"
            )
            lines.append(
                f"- {proj.name}: ELO {round(proj.difficulty * 3000)}, SLO {round(row['min_success_rate'] * 100)}%, "
                f"floor Q {row.get('quality_floor', 0.0):.2f}, "
                f"WTP ${proj.wtp_per_m:.2f}/M; {cloud}; {fate}."
            )
            if row["any_served"]:
                parts = [
                    f"{fmt_money(row['value_served'])}/day served",
                    f"margin {fmt_money(row['margin_day'])}/day",
                ]
                if row["value_spilled"] + row["value_leaked"] > 0:
                    parts.append(f"{fmt_money(row['value_spilled'] + row['value_leaked'])}/day to cloud")
                if row["value_destroyed"] > 0:
                    parts.append(f"{fmt_money(row['value_destroyed'])}/day destroyed")
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
            status = m.get("status") or ("SATURATED" if m["saturated"] else ("IDLE" if m["runnable"] and m["utilization"] < 0.05 else "OK"))
            price = f"${m['internal_pm']:.2f}/M" if m["runnable"] and m["internal_pm"] > 0 else "-"
            lines.append(
                f"- {m['name']}: Q {m['effective_quality'] * 100:.0f}% effective, {m['gpu_count']} GPUs, "
                f"cap {fmt_num(m['daily_tokens_cap'])} tok/day, placed {fmt_num(m['served_tokens'])}, "
                f"util {m['utilization'] * 100:.0f}%, internal {price}, {status}"
            )

    return "\n".join(lines)


def _format_projection_report(state_a: PlannerState, state_b: PlannerState | None) -> str:
    title = "vLLM multi-model planner report"
    parts = [title, "=" * len(title), ""]
    parts.append(_format_projection_report_for_state(state_a, "Config A" if state_b else "Current Config"))
    if state_b:
        parts.extend(["", "", _format_projection_report_for_state(state_b, "Config B")])
    parts.extend([
        "",
        "Notes",
        "Roofline estimates; vLLM continuous batching; separate prefill/decode efficiency knobs; KV capacity anchored to requested GPU memory minus weights and profiled non-KV runtime memory.",
    ])
    return "\n".join(parts).strip() + "\n"


@app.route("/")
def index():
    tab_id = _tab_id(optional=True)
    if tab_id is None:
        return render_template("index.html", A=create_default_state(), B=None, **_template_context())

    sa = get_state(_scope_id())
    sb = get_compare_state(_scope_id())
    _record_snapshot("open", sa, sb)
    return render_template("index.html", A=sa, B=sb, **_template_context())


@app.route("/explainer")
def explainer():
    return render_template("explainer.html")


@app.route("/use-cases")
def use_cases():
    s = get_state(_scope_id())
    return render_template(
        "use_cases.html",
        state=s,
        USE_CASE_DEFS=get_use_case_defs(s),
        use_case_details=USE_CASE_DETAILS,
        **_template_context(),
    )


def _use_case_library_response(reason: str | None = None):
    s = get_state(_scope_id())
    if reason:
        _record_snapshot(reason, s, get_compare_state(_scope_id()), path="/use-cases")
    return render_template(
        "partials/use_case_library.html",
        state=s,
        USE_CASE_DEFS=get_use_case_defs(s),
        use_case_details=USE_CASE_DETAILS,
        **_template_context(),
    )


@app.route("/use-cases/library")
def use_cases_library():
    return _use_case_library_response()


@app.route("/use-cases/definition/add", methods=["POST"])
def use_case_definition_add():
    try:
        s = get_state(_scope_id())
        add_use_case_def(s)
        return _use_case_library_response("use_case_def_add")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/use-cases/definition/remove", methods=["POST"])
def use_case_definition_remove():
    try:
        s = get_state(_scope_id())
        remove_use_case_def(s, request.form.get("key", ""))
        return _use_case_library_response("use_case_def_remove")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/use-cases/definition/set", methods=["POST"])
def use_case_definition_set():
    try:
        s = get_state(_scope_id())
        key = request.form.get("key", "")
        field_name = request.form.get("field", "")
        raw_value = request.form.get("value", "")
        if field_name == "capability":
            set_use_case_def_capability(s, key, request.form.get("cap", ""), raw_value in ("on", "true", "1"))
        elif field_name == "batch_eligible":
            set_use_case_def_field(s, key, "batch_eligible", raw_value in ("on", "true", "1"))
        elif field_name == "tokens_day_m":
            set_use_case_def_field(s, key, "tokens_day", float(raw_value or 0.0) * 1e6)
        elif field_name == "latent_jobs_day_m":
            set_use_case_def_field(s, key, "latent_jobs_day", float(raw_value or 0.0) * 1e6)
        elif field_name == "wtp_per_m_cents":
            set_use_case_def_field(s, key, "wtp_per_m", float(raw_value or 0.0) / 100.0)
        elif field_name == "unlock_price_per_m_cents":
            set_use_case_def_field(s, key, "unlock_price_per_m", float(raw_value or 0.0) / 100.0)
        elif field_name == "min_success_rate_pct":
            set_use_case_def_field(s, key, "min_success_rate", float(raw_value or 0.0) / 100.0)
        elif field_name == "quality_floor_pct":
            set_use_case_def_field(s, key, "quality_floor", float(raw_value or 0.0) / 100.0)
        elif field_name == "difficulty_elo":
            set_use_case_def_field(s, key, "difficulty", max(0.0, min(1.0, float(raw_value or 0.0) / 3000.0)))
        elif field_name == "scale_value":
            set_use_case_def_field(s, key, "scale_value", float(raw_value or 0.0))
        elif field_name == "scale_token_multiplier":
            set_use_case_def_field(s, key, "scale_token_multiplier", float(raw_value or 0.0))
        elif field_name in {"scale_max", "scale_step"}:
            set_use_case_def_field(s, key, field_name, float(raw_value or 0.0))
        elif field_name in {"name", "scale_hint", "scale_model", "scale_label", "scale_unit", "scale_formula", "in_pre", "out_pre"}:
            set_use_case_def_field(s, key, field_name, raw_value)
        elif field_name in {"tokens_day", "wtp_per_m", "difficulty", "min_success_rate", "quality_floor", "latent_jobs_day", "unlock_price_per_m"}:
            set_use_case_def_field(s, key, field_name, float(raw_value or 0.0))
        return _use_case_library_response("use_case_def_set")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/use-cases/export")
def use_case_definition_export():
    try:
        s = get_state(_scope_id())
        body = json.dumps(serialize_use_case_defs(s), indent=2, sort_keys=True) + "\n"
        resp = make_response(body)
        resp.headers["Content-Type"] = "application/json; charset=utf-8"
        resp.headers["Content-Disposition"] = 'attachment; filename="use-case-library.json"'
        return resp
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/use-cases/import", methods=["POST"])
def use_case_definition_import():
    try:
        s = get_state(_scope_id())
        raw = request.form.get("json", "")
        if not raw.strip():
            return jsonify({"error": "Choose a use-case JSON file first."}), 400
        replace_use_case_defs(s, json.loads(raw))
        retune_models(s, preserve_existing=False)
        return _use_case_library_response("use_case_def_import")
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Invalid JSON: {e.msg}"}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/session/sync")
def session_sync():
    sa = get_state(_scope_id())
    sb = get_compare_state(_scope_id())
    _record_snapshot("open", sa, sb, path="/")
    return _htmx_response(sa)


@app.route("/gpu/add", methods=["POST"])
def gpu_add():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        gpu_type = request.form.get("gpu_type")
        if gpu_type not in GPUS:
            return jsonify({"error": "Invalid GPU type"}), 400
        
        count = int(request.form.get("count", 8))
        if count <= 0:
            return jsonify({"error": "Count must be positive"}), 400
            
        add_gpu(s, gpu_type, count)
        return _tracked_htmx_response("gpu_add", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/gpu/qty", methods=["POST"])
def gpu_qty():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        uid = int(request.form.get("uid"))
        if "count" in request.form:
            gp = s.find_gpu(uid)
            if gp is None:
                return jsonify({"error": "GPU pool not found"}), 404
            count = max(0, int(float(request.form.get("count") or 0)))
            change_gpu_qty(s, uid, count - gp.count)
        else:
            delta = int(request.form.get("delta"))
            change_gpu_qty(s, uid, delta)
        return _tracked_htmx_response("gpu_qty", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/gpu/remove", methods=["POST"])
def gpu_remove():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        uid = int(request.form.get("uid"))
        remove_gpu(s, uid)
        return _tracked_htmx_response("gpu_remove", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/gpu/cost", methods=["POST"])
def gpu_cost():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        uid = int(request.form.get("uid"))
        cost = float(request.form.get("value", 0))
        set_gpu_cost(s, uid, cost)
        return _tracked_htmx_response("gpu_cost", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model/add", methods=["POST"])
def model_add():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        model_key = request.form.get("model_key")
        if model_key not in MODELS or MODELS[model_key].hidden:
            return jsonify({"error": "Invalid model key"}), 400
            
        add_model(s, model_key)
        return _tracked_htmx_response("model_add", s)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model/auto", methods=["POST"])
def model_auto():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        strategy = normalize_auto_strategy(
            request.form.get("strategy") or request.form.get("auto_strategy") or getattr(s, "auto_strategy", None)
        )
        auto_select_models(s, strategy)
        return _tracked_htmx_response("model_auto", s)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model/auto-exclude", methods=["POST"])
def model_auto_exclude():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        uid = int(request.form.get("uid"))
        auto_exclude_model(s, uid)
        return _tracked_htmx_response("model_auto_exclude", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model/auto-reallow", methods=["POST"])
def model_auto_reallow():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        model_key = request.form.get("key", "")
        auto_reallow_model(s, model_key)
        return _tracked_htmx_response("model_auto_reallow", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model/remove", methods=["POST"])
def model_remove():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        uid = int(request.form.get("uid"))
        remove_model(s, uid)
        return _tracked_htmx_response("model_remove", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model/prec", methods=["POST"])
def model_prec():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        uid = int(request.form.get("uid"))
        prec = request.form.get("prec")
        if prec not in PRECISIONS:
            return jsonify({"error": "Invalid precision"}), 400
            
        set_model_prec(s, uid, prec)
        return _tracked_htmx_response("model_prec", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model/count", methods=["POST"])
def model_count():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        uid = int(request.form.get("uid"))
        count = int(request.form.get("count"))
        if count < 0:
            return jsonify({"error": "Count cannot be negative"}), 400
            
        set_model_gpu_count(s, uid, count)
        return _tracked_htmx_response("model_count", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model/strat", methods=["POST"])
def model_strat():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        uid = int(request.form.get("uid"))
        phase = request.form.get("phase", "decode")
        tp = int(request.form.get("tp"))
        pp = int(request.form.get("pp", 1))
        dp = int(request.form.get("dp"))
        
        # Validate the strategy before setting
        am = s.find_model(uid)
        if am is None:
            return jsonify({"error": "Model not found"}), 404
            
        gp = s.find_gpu(am.gpu_uid)
        if gp is None:
            return jsonify({"error": "GPU not found"}), 404
            
        model = MODELS[am.model_key]
        valid = valid_strategies(model, am.gpu_count, gp.gpu, s.mu, s.profiled_non_kv_gb, am.prec)
        strategy = (tp, pp, dp)
        if strategy not in valid:
            return jsonify({"error": "Invalid strategy for this model/GPU combination"}), 400
            
        set_model_strat(s, uid, tp, pp, dp, phase)
        return _tracked_htmx_response("model_strat", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model/gpu_pool", methods=["POST"])
def model_gpu_pool():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        uid = int(request.form.get("uid"))
        gpu_uid = int(request.form.get("gpu_uid"))
        set_model_gpu_pool(s, uid, gpu_uid)
        return _tracked_htmx_response("model_gpu_pool", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/dist/preset", methods=["POST"])
def dist_preset():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        kind = request.form.get("kind")
        preset = request.form.get("preset")
        set_dist_preset(s, kind, preset)
        retune_models(s, preserve_existing=False)
        return _tracked_htmx_response("dist_preset", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/dist/slide", methods=["POST"])
def dist_slide():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        kind = request.form.get("kind")
        index = int(request.form.get("index"))
        value = int(request.form.get("value"))
        set_dist_value(s, kind, index, value)
        retune_models(s, preserve_existing=False)
        return _tracked_htmx_response("dist_slide", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/settings/mu", methods=["POST"])
def settings_mu():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        value = int(request.form.get("value"))
        s.mu = value / 100
        retune_models(s, preserve_existing=False)
        return _tracked_htmx_response("settings_mu", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/settings/non-kv", methods=["POST"])
def settings_non_kv():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        value = float(request.form.get("value"))
        s.profiled_non_kv_gb = max(0.0, value)
        retune_models(s, preserve_existing=False)
        return _tracked_htmx_response("settings_non_kv", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/settings/eff", methods=["POST"])
def settings_eff():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        key = request.form.get("key")
        value = int(request.form.get("value")) / 100
        if hasattr(s, key):
            setattr(s, key, value)
            retune_models(s, preserve_existing=False)
        return _tracked_htmx_response("settings_eff", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/settings/int", methods=["POST"])
def settings_int():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        key = request.form.get("key")
        value = int(request.form.get("value"))
        if hasattr(s, key):
            setattr(s, key, value)
            retune_models(s, preserve_existing=False)
        return _tracked_htmx_response("settings_int", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/settings/prefix-hit", methods=["POST"])
def settings_prefix_hit():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        value = int(request.form.get("value"))
        set_prefix_hit_rate(s, value / 100)
        retune_models(s, preserve_existing=False)
        return _tracked_htmx_response("settings_prefix_hit", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/projection/pct", methods=["POST"])
def projection_pct():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        key = request.form.get("key", "")
        value = int(request.form.get("value")) / 100
        set_projection_pct(s, key, value)
        return _tracked_htmx_response("projection_pct", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/projection/choice", methods=["POST"])
def projection_choice():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        key = request.form.get("key", "")
        value = request.form.get("value", "")
        set_projection_choice(s, key, value)
        return _tracked_htmx_response("projection_choice", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/projection/toggle", methods=["POST"])
def projection_toggle():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        key = request.form.get("key", "")
        # HTMX sends a form-encoded value only when the checkbox is checked.
        value = request.form.get("value") in ("on", "true", "1")
        set_projection_toggle(s, key, value)
        return _tracked_htmx_response("projection_toggle", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/project/add", methods=["POST"])
def project_add():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        preset_key = request.form.get("preset") or None
        add_project(s, preset_key)
        retune_models(s, preserve_existing=False)
        return _tracked_htmx_response("project_add", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/project/add-all", methods=["POST"])
def project_add_all():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        for preset in get_use_case_defs(s):
            add_project(s, str(preset["key"]))
        retune_models(s, preserve_existing=False)
        return _tracked_htmx_response("project_add_all", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/project/remove", methods=["POST"])
def project_remove():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        uid = int(request.form.get("uid", "0"))
        remove_project(s, uid)
        retune_models(s, preserve_existing=False)
        return _tracked_htmx_response("project_remove", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/project/set", methods=["POST"])
def project_set():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        uid = int(request.form.get("uid", "0"))
        field_name = request.form.get("field", "")
        raw_value = request.form.get("value", "")
        if field_name == "kind":
            set_project_kind(s, uid, raw_value)
            retune_models(s, preserve_existing=False)
        elif field_name == "name":
            set_project_name(s, uid, raw_value)
        elif field_name == "batch_eligible":
            set_project_batch_eligible(s, uid, raw_value in ("on", "true", "1"))
        elif field_name == "tokens_day_m":
            # slider gives millions of tokens/day; persist in tokens/day
            set_project_field(s, uid, "tokens_day", float(raw_value or 0.0) * 1e6)
            retune_models(s, preserve_existing=False)
        elif field_name == "scale_value":
            set_project_scale_value(s, uid, float(raw_value or 0.0))
            retune_models(s, preserve_existing=False)
        elif field_name == "wtp_per_m_cents":
            # slider gives cents per M tokens; persist as $/M tokens
            set_project_field(s, uid, "wtp_per_m", float(raw_value or 0.0) / 100.0)
        elif field_name == "min_success_rate_pct":
            # slider gives whole-number percent; persist as 0..1 fraction
            set_project_field(s, uid, "min_success_rate", float(raw_value or 0.0) / 100.0)
        elif field_name == "quality_floor_pct":
            set_project_field(s, uid, "quality_floor", float(raw_value or 0.0) / 100.0)
        elif field_name == "difficulty_pct":
            # legacy: slider gave whole-number percent; persist as 0..1 fraction
            set_project_field(s, uid, "difficulty", float(raw_value or 0.0) / 100.0)
        elif field_name == "difficulty_elo":
            # slider/input gives ELO on a 0..3000 axis; persist as 0..1 fraction
            elo = float(raw_value or 0.0)
            set_project_field(s, uid, "difficulty", max(0.0, min(1.0, elo / 3000.0)))
        elif field_name == "capability":
            # `cap` field carries the capability name; `value` is on/off
            cap_name = request.form.get("cap", "")
            set_project_capability(s, uid, cap_name, raw_value in ("on", "true", "1"))
        elif field_name == "latent_jobs_day_m":
            # slider gives millions of latent tokens/day; persist as tokens/day
            set_project_field(s, uid, "latent_jobs_day", float(raw_value or 0.0) * 1e6)
        elif field_name == "unlock_price_per_m_cents":
            # slider gives cents per M tokens; persist as $/M tokens
            set_project_field(s, uid, "unlock_price_per_m", float(raw_value or 0.0) / 100.0)
        elif field_name == "in_pre":
            set_project_dist_preset(s, uid, "in", raw_value)
            retune_models(s, preserve_existing=False)
        elif field_name == "out_pre":
            set_project_dist_preset(s, uid, "out", raw_value)
            retune_models(s, preserve_existing=False)
        elif field_name in ("tokens_day", "wtp_per_m", "difficulty", "min_success_rate", "quality_floor", "latent_jobs_day", "unlock_price_per_m"):
            set_project_field(s, uid, field_name, float(raw_value or 0.0))
            if field_name == "tokens_day":
                retune_models(s, preserve_existing=False)
        return _tracked_htmx_response("project_set", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/project/export")
def project_export():
    try:
        s = _request_state()
        if s is None:
            return jsonify({"error": "No state found"}), 404
        payload = serialize_project_set(s)
        body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        resp = make_response(body)
        resp.headers["Content-Type"] = "application/json; charset=utf-8"
        resp.headers["Content-Disposition"] = 'attachment; filename="use-cases.json"'
        return resp
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/project/import", methods=["POST"])
def project_import():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        raw = request.form.get("json", "")
        if not raw.strip():
            return jsonify({"error": "Choose a use-case JSON file first."}), 400
        payload = json.loads(raw)
        replace_project_set(s, payload)
        retune_models(s, preserve_existing=False)
        return _tracked_htmx_response("project_import", s)
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Invalid JSON: {e.msg}"}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/mode", methods=["POST"])
def set_mode():
    try:
        mode = normalize_plot_mode(request.form.get("mode"))
        sa = get_state(_scope_id())
        sb = get_compare_state(_scope_id())
        if sa:
            sa.mode = mode
        if sb:
            sb.mode = mode
        return _tracked_htmx_response("mode", sa)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/compare/duplicate", methods=["POST"])
def compare_duplicate():
    try:
        duplicate_compare_state(_scope_id())
        return _tracked_htmx_response("compare_duplicate")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/compare/close", methods=["POST"])
def compare_close():
    try:
        clear_compare_state(_scope_id())
        return _tracked_htmx_response("compare_close")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/task/preset", methods=["POST"])
def task_preset():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        key = request.form.get("key")
        tp = TASK_PRESETS.get(key)
        if tp:
            s.task_il = tp["i"]
            s.task_ol = tp["o"]
            retune_models(s, preserve_existing=False)
        return _tracked_htmx_response("task_preset", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/task/length", methods=["POST"])
def task_length():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        length = 2 ** int(request.form.get("value"))
        kind = request.form.get("kind")
        if kind == "in":
            s.task_il = length
        else:
            s.task_ol = length
        retune_models(s, preserve_existing=False)
        return _tracked_htmx_response("task_length", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chart-data")
def chart_data():
    try:
        sa = get_state(_scope_id())
        sb = get_compare_state(_scope_id())
        mode = sa.mode
        states = [sa] + ([sb] if sb else [])

        if mode == "processingpareto":
            batch_sizes = get_processing_pareto_bs(states)
            datasets = chart_processing_pareto(sa, batch_sizes)
            if sb:
                datasets += chart_processing_pareto(sb, batch_sizes, " (B)")
            return jsonify({"type": "line", "datasets": datasets, "mode": mode, "x_max": batch_sizes[-1]})

        batch_sizes = get_decode_bs(states)
        datasets = chart_user_pareto(sa, batch_sizes)
        if sb:
            datasets += chart_user_pareto(sb, batch_sizes, " (B)")
        return jsonify({"type": "line", "datasets": datasets, "mode": mode, "x_max": batch_sizes[-1]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/projection-report")
def projection_report():
    try:
        sa = get_state(_scope_id())
        sb = get_compare_state(_scope_id())
        return jsonify({"text": _format_projection_report(sa, sb)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/picker/gpu")
def picker_gpu():
    return render_template("partials/gpu_picker.html", panel=request.args.get("panel", "A"), gpu_cards_by_vendor=gpu_cards_by_vendor())


@app.route("/picker/model")
def picker_model():
    return render_template("partials/model_picker.html", panel=request.args.get("panel", "A"), models_by_category=models_by_category())


@app.route("/picker/project")
def picker_project():
    panel = request.args.get("panel", "A")
    s = _state(panel) or get_state(_scope_id())
    context = _template_context()
    context["PROJECT_PRESETS"] = get_use_case_defs(s)
    return render_template("partials/project_picker.html", panel=panel, **context)


@app.route("/admin", methods=["GET"])
def admin():
    if not _admin_password():
        return (
            "Set PLANNER_ADMIN_PASSWORD to enable /admin.",
            503,
        )
    if not _is_admin_authenticated():
        return render_template("admin_login.html", error=None)
    all_snapshots = SNAPSHOT_STORE.list_snapshots()
    visitor_map = defaultdict(list)
    for s in all_snapshots:
        visitor_map[s["visitor_id"]].append(s)
    visitors = sorted(
        [{"visitor_id": vid, "snapshots": snaps} for vid, snaps in visitor_map.items()],
        key=lambda v: v["snapshots"][0]["last_seen"],
        reverse=True,
    )
    return render_template("admin.html", visitors=visitors, total_snapshots=len(all_snapshots), GPUS=GPUS, MODELS=MODELS)


@app.route("/admin/login", methods=["POST"])
def admin_login():
    password = _admin_password()
    if not password:
        return (
            "Set PLANNER_ADMIN_PASSWORD to enable /admin.",
            503,
        )
    if request.form.get("password") != password:
        return render_template("admin_login.html", error="Invalid password."), 401

    session[ADMIN_SESSION_KEY] = True
    return redirect(url_for("admin"))


@app.route("/admin/logout", methods=["POST"])
def admin_logout():
    session.pop(ADMIN_SESSION_KEY, None)
    return redirect(url_for("admin"))


@app.template_filter("fmt_num")
def fmt_num(n):
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return str(int(n))


@app.template_filter("fmt_money")
def fmt_money(value):
    value = float(value or 0.0)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1e9:
        return f"{sign}${value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{sign}${value / 1e6:.2f}M"
    if value >= 1e3:
        return f"{sign}${value / 1e3:.1f}k"
    if value >= 100:
        return f"{sign}${value:,.0f}"
    if value >= 1:
        return f"{sign}${value:,.2f}"
    return f"{sign}${value:,.3f}"


@app.template_filter("log2int")
def log2int(n):
    return int(math.log2(n)) if n > 0 else 0


if __name__ == "__main__":
    app.run(debug=True, port=5014)
