"""Versioned agent API contract, shared by validation and OpenAPI discovery."""

from __future__ import annotations

import cloud_policy
from data import (
    COUNTRIES,
    DAY_SHAPES,
    DIST_PRESETS,
    EMBEDDING_DOC_BUCKETS,
    EMBEDDING_DOC_PRESETS,
    GPUS,
    INPUT_BUCKETS,
    MODELS,
    OUTPUT_BUCKETS,
    PRECISIONS,
    QUALITY_DOMAIN_LABELS,
)
from state import ALLOWED_AUTO_MODEL_STRATEGIES, ALLOWED_CAPABILITIES, ALLOWED_PLOT_MODES


def obj(properties: dict, required: tuple[str, ...] = ()) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def number(lo: float = 0, hi: float = 1e12, *, integer: bool = False) -> dict:
    return {"type": "integer" if integer else "number", "minimum": lo, "maximum": hi}


def choices(values) -> dict:
    return {"type": "string", "enum": sorted(values)}


def array(items: dict, maximum: int, minimum: int = 0) -> dict:
    return {"type": "array", "items": items, "minItems": minimum, "maxItems": maximum}


TEXT = {"type": "string", "maxLength": 1024}
UID = number(0, 2**53 - 1, integer=True)
SCALE_KIND = obj(
    {
        **dict.fromkeys(("model", "label", "unit", "formula"), TEXT),
        **dict.fromkeys(("token_multiplier", "min", "max", "step"), number()),
    }
)
PROJECT_FIELDS = {
    "name": TEXT,
    "difficulty": number(0, 1),
    "tokens_day": number(),
    "scale_value": number(),
    "scale_kind": SCALE_KIND,
    "wtp_per_m": number(0, 200),
    "requires": array(choices(ALLOWED_CAPABILITIES), len(ALLOWED_CAPABILITIES)),
    "min_success_rate": number(0.5, 1),
    "quality_floor": number(0, 1),
    "quality_domain": choices(QUALITY_DOMAIN_LABELS),
    "quality_weights": {
        "type": "object",
        "properties": dict.fromkeys(QUALITY_DOMAIN_LABELS, number(0, 1e6)),
        "additionalProperties": False,
    },
    "batch_eligible": {"type": "boolean"},
    "prefix_hit_rate": number(0, 1),
    "latent_jobs_day": number(),
    "unlock_price_per_m": number(0, 200),
    "in_pre": choices(DIST_PRESETS),
    "out_pre": choices(DIST_PRESETS),
}
PROJECT = obj(
    {
        **PROJECT_FIELDS,
        "uid": UID,
        "kind_key": TEXT,
        "kind": TEXT,
        "definition": obj(PROJECT_FIELDS),
        "scale": obj({"value": number(), "tokens_day": number(), "latent_jobs_day": number()}),
    }
)
GPU_POOL = obj(
    {
        "uid": UID,
        "gpu_type": choices(GPUS),
        "count": number(1, 100_000, integer=True),
        "cost_per_gpu_hour": number(0, 1_000_000),
        "country": choices(COUNTRIES),
    },
    ("uid", "gpu_type", "count"),
)
MODEL_ASSIGNMENT = obj(
    {
        "uid": UID,
        "model_key": choices(MODELS),
        "gpu_uid": UID,
        "gpu_count": number(0, 100_000, integer=True),
        "prec": choices(PRECISIONS),
        "artifact_id": {"type": "string", "maxLength": 512},
        "execution_backend": choices(("legacy", "auto", "native", "w4a16", "expanded")),
        "kv_precision": choices(("legacy", "auto", "bf16", "fp8")),
        "recurrent_precision": choices(("legacy", "auto", "bf16", "fp32")),
        "serving_runtime": choices(("generic", "vllm", "sglang", "trtllm")),
        "runtime_version": {"type": "string", "maxLength": 64},
        **dict.fromkeys(
            ("tp", "pp", "dp", "prefill_tp", "prefill_pp", "prefill_dp"),
            number(1, 100_000, integer=True),
        ),
        "spec_method": TEXT,
        "spec_k": number(0, 32, integer=True),
    },
    ("model_key", "gpu_uid", "gpu_count", "prec"),
)
PANEL_FIELDS = {
    "gpus": array(GPU_POOL, 64),
    "models": array(MODEL_ASSIGNMENT, 512),
    "projects": array(PROJECT, 256),
    "use_case_defs": array(obj({**PROJECT_FIELDS, "key": TEXT, "scale_hint": TEXT}), 256),
    "auto_excluded": array(choices(MODELS), 512),
    "auto_mode": {"type": "boolean"},
    "auto_strategy": choices(ALLOWED_AUTO_MODEL_STRATEGIES),
    "in_dist": array(number(0, 1_000_000, integer=True), len(INPUT_BUCKETS), len(INPUT_BUCKETS)),
    "out_dist": array(number(0, 1_000_000, integer=True), len(OUTPUT_BUCKETS), len(OUTPUT_BUCKETS)),
    "embedding_doc_dist": array(
        number(0, 1_000_000, integer=True), len(EMBEDDING_DOC_BUCKETS), len(EMBEDDING_DOC_BUCKETS)
    ),
    "in_pre": choices(["", *DIST_PRESETS]),
    "out_pre": choices(["", *DIST_PRESETS]),
    "embedding_doc_pre": choices(["", *EMBEDDING_DOC_PRESETS]),
    "mode": choices(ALLOWED_PLOT_MODES),
    "projection_day_shape": choices(DAY_SHAPES),
    "corpo_cloud": choices(cloud_policy.corpo_presets()),
    "projection_night_batching": {"type": "boolean"},
    "decode_sched_budget": number(1, 10_000_000, integer=True),
    "task_il": number(1, 10_000_000, integer=True),
    "task_ol": number(0, 10_000_000, integer=True),
}
for _key, _lo, _hi in (
    ("mu", 0.01, 1),
    ("profiled_non_kv_gb", 0, 4096),
    ("kv_slack", 0, 1),
    ("moe_imbalance", 0.1, 10),
    ("pd_interference", 0, 1),
    ("prefix_hit_rate", 0, 1),
    ("spec_acceptance", 0, 0.99),
    ("projection_demand_level", 0.05, 1.2),
    ("projection_night_discount", 0, 0.8),
    ("projection_batch_eligible", 0, 1),
    ("projection_elasticity", 0, 4),
):
    PANEL_FIELDS[_key] = number(_lo, _hi)
for _phase in ("prefill", "decode"):
    for _knob in ("bw_eff", "comp_eff", "overhead", "paged_oh", "ar_overlap"):
        PANEL_FIELDS[f"{_phase}_{_knob}"] = number(0.01 if _knob.endswith("eff") else 0, 1)
PANEL = obj(PANEL_FIELDS)
SCENARIO = obj(
    {
        "type": {"const": "gpullm-scenario"},
        "version": {"enum": [1, 2], "type": "integer"},
        "panel_a": PANEL,
        "panel_b": {"anyOf": [PANEL, {"type": "null"}]},
    },
    ("type", "version", "panel_a"),
)
CALCULATE_REQUEST = obj(
    {
        "scenario": SCENARIO,
        "include": {**array(choices(("report", "charts")), 2), "uniqueItems": True},
    },
    ("scenario",),
)
EXAMPLE_REQUEST = {
    "scenario": {
        "type": "gpullm-scenario",
        "version": 1,
        "panel_a": {
            "gpus": [{"uid": 1, "gpu_type": "H100", "count": 1}],
            "models": [{"model_key": "qwen38-27b", "gpu_uid": 1, "gpu_count": 1, "prec": "fp8"}],
            "projects": [
                {
                    "kind_key": "custom",
                    "name": "Agent chat",
                    "tokens_day": 1_000_000,
                    "difficulty": 0.2,
                    "wtp_per_m": 10,
                    "in_pre": "Chat",
                    "out_pre": "Chat",
                }
            ],
        },
        "panel_b": None,
    },
    "include": ["report"],
}


def openapi_document() -> dict:
    """Expose the same JSON Schema that validates requests (OpenAPI 3.1)."""
    error = {
        "description": "JSON error; correct the request or back off on 429.",
        "content": {"application/json": {"schema": obj({"error": TEXT}, ("error",))}},
    }
    paths = {}
    for path, operation, summary in (
        ("/api/v1/catalog", "getCatalog", "Discover model/GPU keys, presets, units and defaults"),
        ("/api/v1/example", "getExample", "Get a ready-to-submit calculation request"),
        ("/api/v1/openapi.json", "getOpenAPI", "Get this API contract"),
    ):
        paths[path] = {
            "get": {
                "operationId": operation,
                "summary": summary,
                "responses": {"200": {"description": "JSON discovery document"}},
            }
        }
    paths["/api/v1/calculate"] = {
        "post": {
            "operationId": "calculateScenario",
            "summary": "Calculate an isolated scenario without saving browser state",
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": CALCULATE_REQUEST}},
            },
            "responses": {
                "200": {
                    "description": "Estimates, normalized scenario and requested report/charts",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["api_version", "scenario", "results", "warnings"],
                                "properties": {
                                    "api_version": {"const": "1"},
                                    "scenario": SCENARIO,
                                    "results": {"type": "object"},
                                    "warnings": {"type": "array", "items": {"type": "string"}},
                                    "report": {"type": "string"},
                                    "charts": {"type": "object"},
                                },
                            }
                        }
                    },
                },
                **dict.fromkeys(("400", "404", "405", "413", "415", "429", "500"), error),
            },
        }
    }
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "GPU/LLM Usage Modeler API",
            "version": "1.0.0",
            "description": "Stateless closed-form estimates. See /api/v1/docs.md for semantics and units.",
        },
        "servers": [{"url": "/"}],
        "paths": paths,
    }
