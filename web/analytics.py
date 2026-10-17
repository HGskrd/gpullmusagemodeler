"""Visitor analytics: request context, slim event recording, admin payload.

What is stored, per visitor: a first-touch row (Cloudflare country/region/
city/timezone headers, referrer *host*, UTM tags, entry path, device class
from the User-Agent, first Accept-Language tag, and an internal flag for the
admin's own browser). Per action: a small event with the whitelisted form
fields and a compact summary of the affected planner panel (GPU pools, model
assignments, whether each fits). No IP address or full URL is stored; agent
API callers without a cookie are grouped by a daily-rotating salted hash.

Tracking never breaks a request: storage errors are logged and swallowed.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlsplit

from flask import current_app, g, request, session

from data import GPUS, MODELS
from placement import _assignment_memories
from state import PlannerState
from web.config import ADMIN_SESSION_KEY
from web.middleware import _tab_id, _visitor_id

SESSION_GAP_MS = 30 * 60 * 1000
_TOUCH_INTERVAL_MS = 10 * 60 * 1000
_CACHE_LIMIT = 50_000

# ---------------------------------------------------------------------------
# User-Agent classification
# ---------------------------------------------------------------------------

# User-triggered AI fetchers and scripts: counted, as "agent" visitors.
_AGENT_PATTERNS = (
    ("Claude-User", re.compile(r"Claude-User", re.I)),
    ("ChatGPT-User", re.compile(r"ChatGPT-User", re.I)),
    ("Perplexity-User", re.compile(r"Perplexity-User", re.I)),
    ("MistralAI-User", re.compile(r"MistralAI-User", re.I)),
    ("Gemini", re.compile(r"Google-Gemini|Gemini-User", re.I)),
    ("python-httpx", re.compile(r"python-httpx", re.I)),
    ("python-requests", re.compile(r"python-requests", re.I)),
    ("aiohttp", re.compile(r"aiohttp", re.I)),
    ("curl", re.compile(r"^curl/", re.I)),
    ("node", re.compile(r"node-fetch|undici|axios|node\.js", re.I)),
    ("Go", re.compile(r"Go-http-client", re.I)),
)
# Indexers and link-preview crawlers: not people, not tracked at all.
_CRAWLER_RE = re.compile(
    r"bot\b|bot/|crawler|spider|crawl|slurp|facebookexternalhit|embedly|preview|"
    r"headless|lighthouse|pingdom|uptime|monitor|scrapy|httrack|wget",
    re.I,
)


def classify_user_agent(ua: str) -> dict:
    ua = ua or ""
    for name, pattern in _AGENT_PATTERNS:
        if pattern.search(ua):
            return {"device": "agent", "os": None, "agent": name, "crawler": False}
    if not ua or _CRAWLER_RE.search(ua):
        return {"device": "bot", "os": None, "agent": None, "crawler": True}
    if re.search(r"iPad|Tablet|Android(?!.*Mobile)", ua):
        device = "tablet"
    elif re.search(r"Mobi|iPhone|iPod|Android.*Mobile", ua):
        device = "mobile"
    else:
        device = "desktop"
    if re.search(r"iPhone|iPad|iPod", ua):
        os_name = "iOS"
    elif "Android" in ua:
        os_name = "Android"
    elif "Windows" in ua:
        os_name = "Windows"
    elif "Mac OS X" in ua or "Macintosh" in ua:
        os_name = "macOS"
    elif "CrOS" in ua:
        os_name = "ChromeOS"
    elif "Linux" in ua:
        os_name = "Linux"
    else:
        os_name = None
    return {"device": device, "os": os_name, "agent": None, "crawler": False}


# ---------------------------------------------------------------------------
# Referrer channels (computed at read time so the rules can evolve)
# ---------------------------------------------------------------------------

_CHANNEL_RULES = (
    (
        "AI assistant",
        re.compile(
            r"(^|\.)(chatgpt\.com|openai\.com|claude\.ai|perplexity\.ai|gemini\.google\.com|copilot\.microsoft\.com|chat\.mistral\.ai|kagi\.com/assistant|phind\.com|you\.com)$"
        ),
    ),
    (
        "Search",
        re.compile(
            r"(^|\.)(google\.[a-z.]+|bing\.com|duckduckgo\.com|yahoo\.[a-z.]+|baidu\.com|yandex\.[a-z.]+|ecosia\.org|search\.brave\.com|kagi\.com|startpage\.com|naver\.com)$"
        ),
    ),
    ("Hacker News", re.compile(r"(^|\.)news\.ycombinator\.com$")),
    ("Reddit", re.compile(r"(^|\.)(reddit\.com|redd\.it)$")),
    ("X / Twitter", re.compile(r"(^|\.)(x\.com|twitter\.com|t\.co)$")),
    ("LinkedIn", re.compile(r"(^|\.)(linkedin\.com|lnkd\.in)$")),
    ("GitHub", re.compile(r"(^|\.)github\.(com|io)$")),
    ("Hugging Face", re.compile(r"(^|\.)huggingface\.co$")),
    ("Bluesky / Mastodon", re.compile(r"(^|\.)(bsky\.app|mastodon\.[a-z.]+|fosstodon\.org)$")),
    ("Discord / Slack", re.compile(r"(^|\.)(discord\.com|discordapp\.com|slack\.com)$")),
    ("YouTube", re.compile(r"(^|\.)(youtube\.com|youtu\.be)$")),
)
_UTM_CHANNELS = {
    "linkedin": "LinkedIn",
    "twitter": "X / Twitter",
    "x": "X / Twitter",
    "reddit": "Reddit",
    "hn": "Hacker News",
    "hackernews": "Hacker News",
    "github": "GitHub",
    "newsletter": "Newsletter",
    "email": "Newsletter",
    "chatgpt.com": "AI assistant",
    "perplexity": "AI assistant",
}


def channel_for(visitor: dict) -> str:
    if visitor.get("device") == "agent":
        return "Agent / API"
    utm = (visitor.get("utm_source") or "").lower()
    if utm in _UTM_CHANNELS:
        return _UTM_CHANNELS[utm]
    host = (visitor.get("referrer_host") or "").lower()
    if not host:
        return "Campaign" if utm else "Direct"
    for label, pattern in _CHANNEL_RULES:
        if pattern.search(host):
            return label
    return "Other sites"


# ---------------------------------------------------------------------------
# Catalog classification used by the admin pages
# ---------------------------------------------------------------------------


def gpu_class(key: str) -> str:
    gpu = GPUS[key]
    name = gpu.name
    if key.startswith("MAC_") or re.search(r"DGX Spark|Ryzen AI Max|Jetson", name):
        return "unified"
    if re.search(r"GeForce|Radeon RX|Laptop", name):
        return "consumer"
    if re.search(
        r"RTX PRO|RTX A\d|RTX 6000 Ada|Radeon PRO|Radeon AI PRO|Arc Pro|DGX Station|Blackhole p1",
        name,
    ):
        return "workstation"
    return "datacenter"


def model_kind(model) -> str:
    if getattr(model, "is_asr_model", False):
        return "asr"
    if getattr(model, "is_embedding_model", False):
        return "emb"
    if getattr(model, "is_decision_model", False):
        return "decision"
    return "llm"


def catalog_payload() -> dict:
    return {
        "gpus": {
            key: {
                "name": gpu.name,
                "cls": gpu_class(key),
                "vram": round(gpu.mem / 1e9),
                "vendor": gpu.vendor,
            }
            for key, gpu in GPUS.items()
        },
        "models": {
            key: {
                "name": model.name,
                "kind": model_kind(model),
                "b": round(model.total_params / 1e9, 2),
            }
            for key, model in MODELS.items()
        },
    }


# ---------------------------------------------------------------------------
# Request context and visitor identity
# ---------------------------------------------------------------------------

_known_lock = threading.Lock()
_touched: OrderedDict[str, int] = OrderedDict()
_internal: set[str] = set()


def _now_ms() -> int:
    return int(time.time() * 1000)


def _header(name: str, limit: int = 64) -> Optional[str]:
    value = (request.headers.get(name) or "").strip()
    return value[:limit] or None


def _ua_info() -> dict:
    cached = getattr(g, "analytics_ua", None)
    if cached is None:
        cached = classify_user_agent(request.headers.get("User-Agent", ""))
        g.analytics_ua = cached
    return cached


def _agent_visitor_id() -> str:
    """Pseudonymous id for cookie-less agents: salted, rotating daily, no IP kept."""
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    raw = "|".join(
        (
            str(current_app.secret_key),
            day,
            request.remote_addr or "",
            request.headers.get("User-Agent", ""),
        )
    )
    return "agent-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def analytics_visitor_id() -> str:
    if _ua_info()["device"] == "agent" and not request.cookies.get("planner_vid"):
        return _agent_visitor_id()
    return _visitor_id()


def request_context() -> dict:
    ua = _ua_info()
    country = _header("CF-IPCountry", 2)
    if country and (not country.isalpha() or country.upper() in {"XX", "T1"}):
        country = None
    landing = request.method == "GET"
    ref_host = None
    referrer = request.headers.get("Referer", "") if landing else ""
    if referrer:
        try:
            host = (urlsplit(referrer).hostname or "").lower()
        except ValueError:
            host = ""
        if host and host != (request.host or "").split(":")[0].lower():
            ref_host = host[:120]
    lang = (request.headers.get("Accept-Language") or "").split(",")[0].split(";")[0].strip()
    return {
        "country": country.upper() if country else None,
        "region": _header("CF-Region"),
        "city": _header("CF-IPCity"),
        "timezone": _header("CF-Timezone"),
        "lang": lang[:24] or None,
        "device": ua["device"],
        "os": ua["os"],
        "agent": ua["agent"],
        "referrer_host": ref_host,
        "utm_source": (request.args.get("utm_source") or "")[:64] or None,
        "utm_medium": (request.args.get("utm_medium") or "")[:64] or None,
        "utm_campaign": (request.args.get("utm_campaign") or "")[:64] or None,
        "entry_path": request.path[:120] if landing else None,
    }


def _enabled() -> bool:
    return bool(current_app.config.get("TRACKING_ENABLED")) and not _ua_info()["crawler"]


def _store():
    return current_app.extensions["snapshot_store"]


def _safely(action, *args, **kwargs) -> None:
    try:
        action(*args, **kwargs)
    except Exception:  # noqa: BLE001 - analytics must never fail a request
        current_app.logger.warning("Visitor analytics write failed", exc_info=True)


def touch_visitor(visitor_id: Optional[str] = None, overrides: Optional[dict] = None) -> None:
    """Record first-touch context once, then refresh last_seen at most every 10 min."""
    if not current_app.config.get("TRACKING_ENABLED") or (
        overrides is None and _ua_info()["crawler"]
    ):
        return
    visitor_id = visitor_id or analytics_visitor_id()
    now = _now_ms()
    with _known_lock:
        last = _touched.get(visitor_id)
        if last is not None and now - last < _TOUCH_INTERVAL_MS:
            return
        _touched[visitor_id] = now
        _touched.move_to_end(visitor_id)
        while len(_touched) > _CACHE_LIMIT:
            _touched.popitem(last=False)
    _safely(_store().touch_visitor, visitor_id, request_context() | (overrides or {}), now_ms=now)


def mark_internal_if_admin() -> None:
    if not current_app.config.get("TRACKING_ENABLED") or not session.get(ADMIN_SESSION_KEY):
        return
    visitor_id = _visitor_id()
    with _known_lock:
        if visitor_id in _internal:
            return
        _internal.add(visitor_id)
    _safely(_store().mark_internal, visitor_id)


# ---------------------------------------------------------------------------
# Event recording
# ---------------------------------------------------------------------------

_FORM_FIELDS = (
    "gpu_type",
    "count",
    "delta",
    "uid",
    "gpu_uid",
    "model_key",
    "prec",
    "strategy",
    "method",
    "spec_k",
    "tp",
    "pp",
    "dp",
    "phase",
    "kind",
    "preset",
    "index",
    "key",
    "field",
    "cap",
    "mode",
    "value",
)


def _form_subset() -> dict:
    out: dict[str, Any] = {}
    for name in _FORM_FIELDS:
        if name in request.form:
            out[name] = str(request.form.get(name, ""))[:64]
    keys = [key for key in request.form.getlist("model_key") if key]
    if len(keys) > 1:
        out["model_keys"] = [key[:64] for key in keys[:24]]
    elif request.form.get("model_keys"):
        out["model_keys"] = [
            key.strip()[:64] for key in request.form["model_keys"].split(",")[:24] if key.strip()
        ]
    return out


def plan_summary(state: Optional[PlannerState]) -> Optional[dict]:
    """Compact GPU pools + model assignments, with whether each model fits."""
    if state is None:
        return None
    models = []
    for am in state.models:
        fits = None
        need_gb = None
        try:
            model = MODELS[am.model_key]
            need_gb = round(model.gpu_weight_bytes(am.prec) / 1e9, 1)
            pool = state.find_gpu(am.gpu_uid)
            if pool is None or am.gpu_count <= 0:
                fits = False
            else:
                _, decode = _assignment_memories(state, am, pool.gpu)
                fits = decode is not None
        except Exception:  # noqa: BLE001 - a summary must not fail the request
            fits = None
        models.append([am.uid, am.model_key, am.prec, am.gpu_uid, am.gpu_count, fits, need_gb])
    return {
        "g": [
            [gp.uid, gp.gpu_type, gp.count, round(float(gp.cost_per_gpu_hour or 0.0), 2)]
            for gp in state.gpus
        ],
        "m": models,
        "mode": state.mode,
    }


# Reasons whose repeats within a short window collapse into one event.
_COALESCE_BY = {
    "gpu_qty": ("uid",),
    "gpu_cost": ("uid",),
    "model_count": ("uid",),
    "model_strat": ("uid",),
    "dist_slide": ("kind", "index"),
    "settings_mu": (),
    "settings_non_kv": (),
    "settings_eff": ("key",),
    "settings_int": ("key",),
    "projection_pct": ("key",),
    "project_set": ("uid", "field"),
    "task_length": ("kind",),
    "use_case_def_set": ("key", "field"),
}


def record_event(
    reason: str,
    state_a: Optional[PlannerState] = None,
    state_b: Optional[PlannerState] = None,
    *,
    extra: Optional[dict] = None,
    visitor_id: Optional[str] = None,
    tab_id: Optional[str] = None,
) -> None:
    if not _enabled():
        return
    try:
        panel = "B" if request.values.get("panel") == "B" else "A"
        form = _form_subset() if request.method == "POST" else {}
        detail: dict[str, Any] = {"p": panel}
        if form:
            detail["f"] = form
        mutated = state_b if panel == "B" else state_a
        plan = plan_summary(mutated)
        if plan is not None:
            detail["plan"] = plan
        if state_b is not None:
            detail["cmp"] = True
        if extra:
            detail["x"] = extra
        coalesce = None
        if reason in _COALESCE_BY:
            parts = [reason, panel, *(form.get(name, "") for name in _COALESCE_BY[reason])]
            coalesce = ":".join(parts)
        vid = visitor_id or analytics_visitor_id()
        touch_visitor(vid)
        _store().record_event(
            visitor_id=vid,
            tab_id=tab_id or _tab_id() or "default",
            event_type=reason,
            detail=detail,
            coalesce_key=coalesce,
        )
    except Exception:  # noqa: BLE001 - analytics must never fail a request
        current_app.logger.warning("Visitor analytics event failed", exc_info=True)


def record_api_call(response) -> None:
    """Log one agent-API request: endpoint, status, and the scenario it asked about.

    The API is stateless and cookie-free, so callers are always grouped under the
    pseudonymous agent id, even from a browser that has a planner cookie.
    """
    if not current_app.config.get("TRACKING_ENABLED"):
        return
    try:
        extra: dict[str, Any] = {
            "endpoint": f"{request.method} {request.path}"[:80],
            "status": response.status_code,
        }
        payload = request.get_json(silent=True) if request.is_json else None
        scenario = payload.get("scenario") if isinstance(payload, dict) else None
        panel = scenario.get("panel_a") if isinstance(scenario, dict) else None
        if isinstance(panel, dict):
            gpus = [
                [str(row.get("gpu_type", ""))[:64], row.get("count")]
                for row in (panel.get("gpus") or [])[:8]
                if isinstance(row, dict)
            ]
            models = [
                [str(row.get("model_key", ""))[:64], str(row.get("prec", ""))[:16]]
                for row in (panel.get("models") or [])[:8]
                if isinstance(row, dict)
            ]
            extra["gpus"] = gpus
            extra["models"] = models
            unknown = [key for key, _ in gpus if key and key not in GPUS]
            unknown += [key for key, _ in models if key and key not in MODELS]
            if unknown:
                extra["unknown"] = unknown[:8]
        result = getattr(g, "analytics_api_result", None)
        if result is not None:
            extra["fits"] = result
        if response.status_code >= 400:
            body = response.get_json(silent=True) if response.is_json else None
            if isinstance(body, dict) and body.get("error"):
                extra["error"] = str(body["error"])[:200]
        vid = _agent_visitor_id()
        touch_visitor(
            vid, {"device": "agent", "os": None, "agent": _ua_info()["agent"] or "API client"}
        )
        _store().record_event(
            visitor_id=vid,
            tab_id="api",
            event_type="api_call",
            detail={"x": extra},
        )
    except Exception:  # noqa: BLE001
        current_app.logger.warning("Visitor analytics API log failed", exc_info=True)


# ---------------------------------------------------------------------------
# Admin payload: visitors -> sessions -> normalized events
# ---------------------------------------------------------------------------


def _plan_maps(plan: Optional[dict]):
    if not plan:
        return {}, {}
    gpus = {
        row[0]: {"key": row[1], "qty": row[2], "cost": row[3] if len(row) > 3 else 0}
        for row in plan.get("g", [])
    }
    models = {
        row[0]: {
            "key": row[1],
            "prec": row[2],
            "gpu_uid": row[3],
            "count": row[4],
            "fits": row[5],
            "need_gb": row[6],
        }
        for row in plan.get("m", [])
    }
    return gpus, models


def _int(value, default=None):
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _fit_result(plan: Optional[dict]) -> Optional[dict]:
    gpus, models = _plan_maps(plan)
    if not models:
        return None
    failing = [m for m in models.values() if m["fits"] is False]
    if failing:
        first = failing[0]
        pool = gpus.get(first["gpu_uid"])
        vram = 0.0
        if pool and pool["key"] in GPUS:
            vram = GPUS[pool["key"]].mem / 1e9 * max(0, first["count"] or 0)
        return {
            "fits": False,
            "model": first["key"],
            "prec": first["prec"],
            "gpu": pool["key"] if pool else None,
            "qty": first["count"],
            "need_gb": first["need_gb"],
            "vram_gb": round(vram),
        }
    need = sum(m["need_gb"] or 0 for m in models.values())
    vram = sum(GPUS[p["key"]].mem / 1e9 * p["qty"] for p in gpus.values() if p["key"] in GPUS)
    return {"fits": True, "need_gb": round(need), "vram_gb": round(vram)}


_FIT_SENSITIVE = {
    "gpu_add",
    "gpu_qty",
    "gpu_remove",
    "model_add",
    "model_add_many",
    "model_auto",
    "model_remove",
    "model_prec",
    "model_count",
    "model_gpu_pool",
    "model_strat",
    "model_spec",
    "settings_mu",
    "settings_non_kv",
    "scenario_import",
}
_ALWAYS_REPORT_FIT = {"model_add", "model_add_many", "model_auto", "model_prec", "model_count"}

# Events that carry a plan snapshot without the visitor having changed it.
_VIEW_ONLY = {
    "open",
    "econ_view",
    "use_cases_view",
    "explainer_view",
    "export_report",
    "scenario_export",
    "use_case_export",
}

_PASSTHROUGH = {
    "session_reset",
    "compare_duplicate",
    "compare_close",
    "scenario_import",
    "export_report",
    "scenario_export",
    "use_case_export",
    "use_cases_view",
    "explainer_view",
}


def _normalize(raw_type: str, detail: dict, prev: dict) -> list[dict]:
    """Translate one stored event into display events, using the previous plan per panel."""
    panel = detail.get("p", "A")
    form = detail.get("f", {})
    extra = detail.get("x", {})
    plan = detail.get("plan")
    prev_plan = prev.get(panel)
    prev_gpus, prev_models = _plan_maps(prev_plan)
    gpus, models = _plan_maps(plan)
    out: list[dict] = []

    def model_from_uid(uid):
        uid = _int(uid)
        return models.get(uid) or prev_models.get(uid)

    if raw_type == "open":
        out.append({"type": "open", "path": extra.get("path", "/")})
    elif raw_type == "gpu_add":
        key = form.get("gpu_type")
        qty = next((p["qty"] for p in gpus.values() if p["key"] == key), _int(form.get("count"), 8))
        out.append(
            {
                "type": "compare_gpu" if panel == "B" else "gpu_add",
                "gpu": key,
                "qty": qty,
                "side": panel,
            }
        )
    elif raw_type in ("gpu_qty", "gpu_remove", "gpu_cost"):
        uid = _int(form.get("uid"))
        pool = gpus.get(uid) or prev_gpus.get(uid)
        if pool:
            if raw_type == "gpu_qty":
                out.append(
                    {
                        "type": "compare_gpu" if panel == "B" else "gpu_qty",
                        "gpu": pool["key"],
                        "qty": pool["qty"],
                        "side": panel,
                    }
                )
            elif raw_type == "gpu_remove":
                out.append({"type": "gpu_remove", "gpu": pool["key"]})
            else:
                out.append({"type": "gpu_cost", "gpu": pool["key"], "usd_hr": pool["cost"]})
    elif raw_type in ("model_add", "model_add_many", "model_auto"):
        added = (
            [m for uid, m in models.items() if uid not in prev_models]
            if prev_plan
            else list(models.values())
        )
        if raw_type == "model_add" and form.get("model_key"):
            added = [m for m in added if m["key"] == form["model_key"]] or added
        for m in added:
            out.append(
                {
                    "type": "model_add",
                    "model": m["key"],
                    "prec": m["prec"],
                    "auto": raw_type == "model_auto",
                    "side": panel,
                }
            )
    elif raw_type == "model_add_rejected":
        error = extra.get("error", "")
        if "does not fit" in error:
            reason = "fits_nowhere"
        elif "Add a GPU pool" in error:
            reason = "no_gpu"
        else:
            reason = "invalid"
        out.append(
            {
                "type": "model_rejected",
                "model": extra.get("model") or form.get("model_key"),
                "error": error,
                "reason": reason,
            }
        )
    elif raw_type == "model_remove":
        m = prev_models.get(_int(form.get("uid")))
        if m:
            out.append({"type": "model_remove", "model": m["key"]})
    elif raw_type == "model_prec":
        m = model_from_uid(form.get("uid"))
        if m:
            out.append({"type": "model_prec", "model": m["key"], "prec": m["prec"]})
    elif raw_type == "model_count":
        m = model_from_uid(form.get("uid"))
        if m:
            out.append({"type": "model_count", "model": m["key"], "count": m["count"]})
    elif raw_type in (
        "model_gpu_pool",
        "model_strat",
        "model_spec",
        "model_auto_exclude",
        "model_auto_reallow",
    ):
        m = model_from_uid(form.get("uid"))
        what = {
            "model_gpu_pool": "moved to another GPU pool",
            "model_strat": f"TP/PP/DP {form.get('tp', '?')}/{form.get('pp', '1')}/{form.get('dp', '?')}",
            "model_spec": f"speculative decoding: {form.get('method', 'off')}",
            "model_auto_exclude": "excluded from auto-pick",
            "model_auto_reallow": "re-allowed in auto-pick",
        }[raw_type]
        out.append(
            {"type": "model_tune", "model": (m or {}).get("key") or form.get("key"), "what": what}
        )
    elif raw_type == "mode":
        out.append({"type": "mode", "mode": form.get("mode") or (plan or {}).get("mode")})
    elif raw_type in ("dist_preset", "dist_slide", "task_preset", "task_length"):
        out.append(
            {
                "type": "workload",
                "what": raw_type.replace("_", " "),
                "detail": form.get("preset") or form.get("key") or form.get("kind") or "",
            }
        )
    elif raw_type.startswith("settings_"):
        out.append(
            {
                "type": "settings",
                "what": raw_type[len("settings_") :],
                "key": form.get("key", ""),
                "value": form.get("value", ""),
            }
        )
    elif raw_type.startswith("projection_"):
        out.append(
            {
                "type": "projection",
                "what": raw_type[len("projection_") :],
                "key": form.get("key", ""),
            }
        )
    elif raw_type in ("project_add", "project_add_all"):
        out.append(
            {
                "type": "use_case_add",
                "use_case": form.get("preset")
                or ("all presets" if raw_type == "project_add_all" else "custom"),
            }
        )
    elif raw_type.startswith("project_") or raw_type.startswith("use_case_def"):
        out.append({"type": "use_case_edit", "what": raw_type, "field": form.get("field", "")})
    elif raw_type == "econ_view":
        out.append({"type": "econ_view", "page": extra.get("page", "")})
    elif raw_type == "api_call":
        call = {
            "type": "api_call",
            "endpoint": extra.get("endpoint", ""),
            "status": extra.get("status", 200),
        }
        if extra.get("gpus"):
            call["gpu"], call["qty"] = extra["gpus"][0][0], extra["gpus"][0][1]
        if extra.get("models"):
            call["model"], call["prec"] = extra["models"][0][0], extra["models"][0][1]
        if "fits" in extra:
            call["fits"] = extra["fits"]
        if extra.get("unknown"):
            call["unknown"] = extra["unknown"]
        if extra.get("error"):
            call["error"] = extra["error"]
        out.append(call)
    elif raw_type in _PASSTHROUGH:
        out.append({"type": raw_type})
    else:
        out.append({"type": "other", "what": raw_type})

    if plan is not None:
        prev[panel] = plan
    return out


def _final(plan: Optional[dict]) -> Optional[dict]:
    if plan is None:
        return None
    gpus, models = _plan_maps(plan)
    fit = None
    if models:
        fit = all(m["fits"] is not False for m in models.values())
    return {
        "gpus": [{"key": p["key"], "qty": p["qty"]} for p in gpus.values()],
        "models": [
            {"key": m["key"], "prec": m["prec"], "fits": m["fits"]} for m in models.values()
        ],
        "fit": fit,
    }


def build_admin_payload(store, *, days: int) -> dict:
    now = _now_ms()
    since = now - days * 86_400_000
    raw = store.list_analytics(since_ms=since)
    by_visitor: dict[str, list[dict]] = {}
    for event in raw["events"]:
        by_visitor.setdefault(event["visitor_id"], []).append(event)

    visitors = []
    for row in raw["visitors"]:
        vid = row["visitor_id"]
        events = by_visitor.pop(vid, [])
        visitors.append(_visitor_payload(row, events))
    # Events whose visitor row was pruned or never written still count.
    for vid, events in by_visitor.items():
        visitors.append(
            _visitor_payload(
                {"visitor_id": vid, "first_seen": events[0]["t"], "last_seen": events[-1]["t"]},
                events,
            )
        )

    return {
        "meta": {
            "now": now,
            "start": since,
            "days": days,
            "truncated": raw["truncated"],
            "status": store.analytics_status(),
        },
        "catalog": catalog_payload(),
        "visitors": visitors,
    }


def _visitor_payload(row: dict, events: list[dict]) -> dict:
    sessions: list[dict] = []
    current = None
    prev: dict[str, dict] = {}
    last_fit: Optional[bool] = None
    for event in events:
        if current is None or event["t"] - current["end"] > SESSION_GAP_MS:
            current = {
                "id": f"{row['visitor_id'][:6]}-{len(sessions) + 1}",
                "start": event["t"],
                "end": event["t"],
                "tabs": [],
                "events": [],
                "_plan": None,
            }
            sessions.append(current)
            prev = {}
            last_fit = None
        current["end"] = event["t"]
        if event["tab_id"] not in current["tabs"]:
            current["tabs"].append(event["tab_id"])
        detail = event["detail"]
        for item in _normalize(event["type"], detail, prev):
            item["t"] = event["t"]
            current["events"].append(item)
        plan = detail.get("plan")
        # The page-load ("open") plan is the default or a restored scenario, not
        # a choice made in this visit; a visit's final plan starts at its first edit.
        if plan is not None and detail.get("p", "A") == "A" and event["type"] not in _VIEW_ONLY:
            current["_plan"] = plan
            if event["type"] in _FIT_SENSITIVE:
                result = _fit_result(plan)
                if result is not None and (
                    event["type"] in _ALWAYS_REPORT_FIT or result["fits"] != last_fit
                ):
                    current["events"].append({"type": "result", "t": event["t"], **result})
                last_fit = result["fits"] if result else None
    for s in sessions:
        s["final"] = _final(s.pop("_plan"))
        s["entry"] = next((e.get("path") for e in s["events"] if e["type"] == "open"), None)
    first = row.get("first_seen") or (events[0]["t"] if events else _now_ms())
    last = max(row.get("last_seen") or first, sessions[-1]["end"] if sessions else first)
    if not sessions:
        sessions = [
            {
                "id": f"{row['visitor_id'][:6]}-1",
                "start": first,
                "end": last,
                "tabs": [],
                "entry": row.get("entry_path"),
                "events": [{"type": "open", "path": row.get("entry_path") or "/", "t": first}],
                "final": None,
            }
        ]
    utm = "&".join(
        f"utm_{k}={row[f'utm_{k}']}"
        for k in ("source", "medium", "campaign")
        if row.get(f"utm_{k}")
    )
    return {
        "id": row["visitor_id"],
        "first": first,
        "last": last,
        "country": row.get("country"),
        "region": row.get("region"),
        "city": row.get("city"),
        "tz": row.get("timezone"),
        "lang": row.get("lang"),
        "device": row.get("device") or "unknown",
        "os": row.get("os"),
        "ua": row.get("agent"),
        "channel": channel_for(row),
        "ref": row.get("referrer_host"),
        "utm": utm or None,
        "entry": row.get("entry_path") or "/",
        "internal": bool(row.get("internal")),
        "sessions": sessions,
    }


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

# Full-page GETs: their first hit carries the referrer, UTM tags and country.
_PAGE_EVENTS = {
    "planner.index": None,
    "planner.explainer": "explainer_view",
    "use_cases.use_cases": "use_cases_view",
    "econ.index": "econ_view",
    "econ.variant": "econ_view",
}


def _after_request(response):
    try:
        mark_internal_if_admin()
        endpoint = request.endpoint or ""
        if request.blueprint == "agent_api":
            if request.method != "HEAD":
                record_api_call(response)
        elif request.method == "GET" and response.status_code == 200 and endpoint in _PAGE_EVENTS:
            touch_visitor()
            event = _PAGE_EVENTS[endpoint]
            if event:
                page = (request.view_args or {}).get("slug") or ""
                record_event(event, extra={"page": page} if page else None, tab_id="page")
    except Exception:  # noqa: BLE001
        current_app.logger.warning("Visitor analytics hook failed", exc_info=True)
    return response


def register_analytics(application) -> None:
    application.after_request(_after_request)
