"""Stateless JSON API for agents; never accesses the browser state registry."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from typing import Any

from flask import Blueprint, current_app, jsonify, request, send_file
from jsonschema import Draft202012Validator
from werkzeug.exceptions import BadRequest, UnsupportedMediaType

import cloud_policy
from data import (
    DIST_PRESETS,
    EMBEDDING_DOC_BUCKETS,
    EMBEDDING_DOC_PRESETS,
    GPUS,
    INPUT_BUCKETS,
    MODELS,
    OPEN_MODEL_ARCHITECTURE_CAPTURED_AT,
    OPEN_MODEL_ARCHITECTURE_SOURCES,
    OUTPUT_BUCKETS,
    PRECISION_SPECS,
    PREVIEW_ASSUMPTIONS,
    PROJECT_PRESETS,
    artifact_options,
)
from data.groups import _model_kind
from engine.economics import compute_revenue_projection
from planner_service import deserialize_scenario
from presentation.model_cards import get_model_infos
from presentation.reports import format_projection_report
from scenarios import serialize_scenario
from state import PlannerState
from web.agent_contract import CALCULATE_REQUEST, EXAMPLE_REQUEST, openapi_document
from web.api import _build_chart_payload
from web.config import BASE_DIR

agent_api_bp = Blueprint("agent_api", __name__)
_validator = Draft202012Validator(CALCULATE_REQUEST)
ESTIMATE_WARNING = (
    "Closed-form capacity estimates, not benchmark results. Calibrate against the exact "
    "model, precision, serving runtime, hardware topology and workload before deployment."
)


def _json_safe(value: Any) -> Any:
    """Translate estimator objects and non-finite unavailable values into strict JSON."""
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [_json_safe(item) for item in sorted(value)]
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _validate_request(payload: Any) -> None:
    try:
        json.dumps(payload, allow_nan=False)
    except (ValueError, OverflowError, RecursionError) as error:
        raise ValueError("Request must contain finite JSON values with bounded nesting.") from error
    validation_error = next(_validator.iter_errors(payload), None)
    if validation_error:
        path = ".".join(str(part) for part in validation_error.absolute_path)
        raise ValueError(f"{path or '$'}: {validation_error.message[:300]}")
    for label in ("panel_a", "panel_b"):
        panel = payload["scenario"].get(label)
        if panel is None:
            continue
        known_projects = {"custom", *(preset["key"] for preset in PROJECT_PRESETS)}
        known_projects.update(row.get("key") for row in panel.get("use_case_defs", []))
        for idx, project in enumerate(panel.get("projects", [])):
            kind = project.get("kind_key") or project.get("kind") or "custom"
            if kind not in known_projects:
                raise ValueError(f"scenario.{label}.projects.{idx}.kind_key: unknown use-case key.")
        for idx, row in enumerate(panel.get("models", [])):
            model = MODELS[row["model_key"]]
            method = row.get("spec_method", "off")
            allowed = {"off", *(profile.method for profile in model.speculative_profiles)}
            if _model_kind(model) == "llm":
                allowed.add("ngram")
            if method not in allowed:
                raise ValueError(f"scenario.{label}.models.{idx}.spec_method: unsupported method.")


def _panel_result(state: PlannerState) -> dict:
    diagnostics = []
    for info in get_model_infos(state):
        assignment = info["am"]
        diagnostics.append(
            {
                "model_key": assignment.model_key,
                "assignment_uid": assignment.uid,
                "gpu_uid": assignment.gpu_uid,
                "gpu_count": assignment.gpu_count,
                "precision": assignment.prec,
                "artifact_id": assignment.artifact_id,
                "execution": info["execution"],
                "kv_format": info["kv_format"],
                "recurrent_format": info["recurrent_format"],
                "weight_storage": info["weight_memory"],
                **{key: info[key] for key in ("runnable", "prefill_fits", "decode_fits")},
                "decode_max_slots": info["decode_max_slots"],
                **({"decision": info["decision"]} if assignment.model.is_decision_model else {}),
                "prefill_max_batch": info["prefill_max_batch"],
                "prefill_memory": info["prefill_mem"],
                "decode_memory": info["decode_mem"],
                "warnings": list(
                    dict.fromkeys(
                        info["precision_alerts"] + info["prefill_alerts"] + info["decode_alerts"]
                    )
                ),
                "architecture_note": assignment.model.architecture_note,
                "preview_assumptions": {
                    key: PREVIEW_ASSUMPTIONS[key]
                    for key in (
                        f"model:{assignment.model_key}",
                        f"gpu:{info['gpu'].key}" if info["gpu"] else "",
                    )
                    if key in PREVIEW_ASSUMPTIONS
                },
                "system_ram_weight_bytes_per_copy": assignment.model.system_ram_weight_bytes,
            }
        )
    return {
        "projection": compute_revenue_projection(state, include_recommendations=False),
        "models": diagnostics,
    }


@agent_api_bp.get("/llms.txt")
def llms_txt():
    return send_file(BASE_DIR / "llms.txt", mimetype="text/plain")


@agent_api_bp.get("/api/v1/docs.md")
def documentation():
    return send_file(BASE_DIR / "docs" / "api.md", mimetype="text/markdown")


@agent_api_bp.get("/api/v1/openapi.json")
def openapi():
    return jsonify(openapi_document())


@agent_api_bp.get("/api/v1/example")
def example():
    return jsonify(EXAMPLE_REQUEST)


@agent_api_bp.get("/api/v1/catalog")
def catalog():
    models = []
    for key, model in MODELS.items():
        if model.hidden:
            continue
        models.append(
            {
                "key": key,
                "name": model.name,
                "kind": _model_kind(model),
                **({"decision_profile": model.decision_profile} if model.is_decision_model else {}),
                "total_params": model.total_params,
                "active_params": model.active_params,
                "max_context_tokens": model.max_context_tokens,
                "native_precision": model.native_precision_key,
                "artifacts": artifact_options(model),
                "weight_bytes_by_precision": {
                    prec: model.gpu_weight_bytes(prec) for prec in PRECISION_SPECS
                },
                "system_ram_weight_bytes_per_copy": model.system_ram_weight_bytes,
                "architecture_note": model.architecture_note,
                "source": OPEN_MODEL_ARCHITECTURE_SOURCES.get(key),
            }
        )
    return jsonify(
        _json_safe(
            {
                "api_version": "1",
                "models": models,
                "gpus": list(GPUS.values()),
                "precisions": PRECISION_SPECS,
                "preview_assumptions": PREVIEW_ASSUMPTIONS,
                "model_sources_captured_at": OPEN_MODEL_ARCHITECTURE_CAPTURED_AT,
                "use_cases": PROJECT_PRESETS,
                "distributions": {
                    "input_token_buckets": INPUT_BUCKETS,
                    "output_token_buckets": OUTPUT_BUCKETS,
                    "presets": DIST_PRESETS,
                    "embedding_document_token_buckets": EMBEDDING_DOC_BUCKETS,
                    "embedding_document_presets": EMBEDDING_DOC_PRESETS,
                },
                "defaults": serialize_scenario(PlannerState(), None)["panel_a"],
                "cloud_policy": cloud_policy.summary(),
                "limits": {
                    "request_bytes": current_app.config["MAX_CONTENT_LENGTH"],
                    "requests_per_minute_per_ip": current_app.config["REQUEST_RATE_LIMIT"],
                },
                "warnings": [ESTIMATE_WARNING],
            }
        )
    )


@agent_api_bp.post("/api/v1/calculate")
def calculate():
    if not request.is_json:
        raise UnsupportedMediaType("Use Content-Type: application/json.")
    try:
        payload = request.get_json()
    except (BadRequest, RecursionError) as error:
        raise BadRequest("Request body must be valid JSON with bounded nesting.") from error
    _validate_request(payload)
    state_a, state_b = deserialize_scenario(payload["scenario"])
    result = {
        "api_version": "1",
        "scenario": serialize_scenario(state_a, state_b),
        "results": {
            "panel_a": _panel_result(state_a),
            "panel_b": _panel_result(state_b) if state_b is not None else None,
        },
        "warnings": [
            ESTIMATE_WARNING,
            "Null numeric results mean unavailable or non-finite, never zero.",
            "Inspect the returned scenario: import normalization and topology retuning may adjust inputs.",
        ],
    }
    include = payload.get("include", [])
    if "report" in include:
        result["report"] = format_projection_report(state_a, state_b)
    if "charts" in include:
        result["charts"] = _build_chart_payload(state_a, state_b)
    return jsonify(_json_safe(result))
