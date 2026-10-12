# Calculator API v1

The API is available alongside the browser calculator, at the same origin. Use
`http://localhost:5014` for the documented local setup. No switch, API key, cookie,
JavaScript or `X-Tab-ID` is needed. Deployment operators can restrict access at
their reverse proxy. This API estimates deployments; it does not invoke models,
provision GPUs or send requests to cloud providers.

Each calculation builds fresh planner state. It does not read or modify browser
sessions, persist snapshots or populate browser response caches. Ordinary web
server access logging and per-IP rate limiting still apply.

## Quick start

```sh
BASE=http://localhost:5014
curl --fail-with-body "$BASE/api/v1/example" -o request.json
curl --fail-with-body "$BASE/api/v1/calculate" \
  -H 'Content-Type: application/json' --data-binary @request.json
```

The example is one H100 with Qwen 3.8 27B at FP8 and a custom chat use case.
Discover valid identifiers from `GET /api/v1/catalog` before constructing your
own request. GPU and model keys are case-sensitive. The catalog describes the
installed version; it is not a live pricing or availability feed.

## Endpoints

| Method | Path | Result |
| --- | --- | --- |
| GET | `/llms.txt` | Concise agent entry point, served as plain text |
| GET | `/api/v1/docs.md` | This guide |
| GET | `/api/v1/openapi.json` | OpenAPI 3.1 contract, with the input validation schema |
| GET | `/api/v1/catalog` | Models, GPUs, precision assumptions, use cases, distributions, defaults and configured limits |
| GET | `/api/v1/example` | Complete calculation request, ready to POST |
| POST | `/api/v1/calculate` | Normalized scenario, structured estimates and optional report/charts |

## Request

The body has two fields: required `scenario` and optional `include`. Unknown
fields, incorrect JSON types, invalid identifiers, out-of-range values and
non-finite numbers are rejected with HTTP 400. Consult the OpenAPI schema for
all allowed fields and bounds.

```json
{
  "scenario": {
    "type": "gpullm-scenario",
    "version": 1,
    "panel_a": {
      "gpus": [{"uid": 1, "gpu_type": "H100", "count": 1}],
      "models": [
        {"model_key": "qwen38-27b", "gpu_uid": 1, "gpu_count": 1, "prec": "fp8"}
      ],
      "projects": [
        {"kind_key": "custom", "name": "Chat", "tokens_day": 1000000,
         "difficulty": 0.2, "wtp_per_m": 10, "in_pre": "Chat", "out_pre": "Chat"}
      ]
    },
    "panel_b": null
  },
  "include": ["report", "charts"]
}
```

- `scenario` uses the browser's version 1 export format. Wrap an exported
  scenario in `{"scenario": <export>}`. Current canonical keys are required;
  legacy aliases and retired models are not accepted by v1 validation.
- `panel_a` is required. `panel_b` is optional or null; supply it to compare two
  independent configurations. Empty panels are valid and return no supply or
  demand. Each panel defaults to empty `gpus`, `models` and `projects`, not the
  browser's demo scenario.
- GPU pools require a unique integer `uid`, `gpu_type` and `count`. Optional
  `cost_per_gpu_hour` overrides catalog TCO in USD per accelerator-hour;
  `country` defaults to `FR` for carbon estimates. GPU units and count increments
  follow the catalog, including rack-scale entries.
- Assignments require `model_key`, `gpu_uid`, `gpu_count` and `prec`. The pool
  reference must exist in the same panel. `gpu_count: 0` keeps a parked model.
  Optional `tp`, `pp`, `dp` and `prefill_tp`, `prefill_pp`, `prefill_dp` request
  parallel layouts; each defaults to 1. The same scenario import/retuning used
  by the browser adjusts count increments, allocations and invalid topologies.
  Inspect the returned scenario and fit diagnostics rather than assuming the
  requested layout was retained. A model that cannot fit may return HTTP 200
  with `runnable: false`; HTTP success alone does not establish viability.
- `projects` describe daily demand and economic constraints. Use a catalog
  `kind_key` for its preset definition and scale, or `custom` with explicit
  fields. A preset's `scale_value` is in its own `scale_kind` units, not tokens.
  For a custom workload, `tokens_day` is daily useful-token demand; explicit
  `scale_value` takes precedence. Exported nested `definition` and `scale`
  objects are also accepted. Optional `use_case_defs` supplies a custom library.
- Distribution presets (`in_pre` / `out_pre`) belong on each project. With
  projects present, the engine derives the aggregate workload and prefix-cache
  hit rate from them. Without projects, panel `in_dist` / `out_dist` are weight
  arrays aligned to the catalog's token buckets; omitted arrays use Chat.
  Panel preset labels alone do not change these arrays. Panel `task_il` and
  `task_ol` are reference-task sizes, not a replacement for the distributions.
- Optional panel efficiency settings have the same names as browser scenario
  exports. The catalog provides their defaults. `mode` selects the chart;
  allowed values are in the schema. `auto_mode` and `auto_strategy` are retained
  scenario settings; this endpoint evaluates supplied assignments and does not
  run the interactive automatic model-selection workflow.
- `include` defaults to `[]`. Add `report` for the browser's text report and
  `charts` for its chart payload (A/B on the same axes). These can be more
  expensive than the core calculation. Core projections omit marginal GPU
  recommendations; the optional text report computes them as in the browser.

## Response and units

`api_version` is `"1"`. `scenario` is the complete normalized export and can be
resubmitted or imported into the browser. Internal UIDs are regenerated per
request; use returned assignment and pool identifiers together.

`results.panel_a` and optional `results.panel_b` each contain:

- `projection`: capacity, demand outcomes, cost, margin, utilization, carbon,
  and per-model/per-project economics from the planner engine. Check `ready`,
  `has_supply`, `has_demand` and the diagnostics. Capacity can still be useful
  without projects; demand/economic outcomes require a workload.
- `models`: assignment identifiers, effective precision, prefill/decode fit,
  runnable status, memory breakdowns, slot/batch limits, architecture notes,
  system-RAM weight requirements and warnings.

Top-level `warnings` always disclose the estimator's limitations. `report` and
`charts` are present only if requested. Numeric results that are unavailable
or non-finite are encoded as JSON `null`, never as `NaN` or `Infinity`.

Units follow the engine and are **not interchangeable**:

- Currency is USD. `*_day` cost/value/margin fields are USD/day; `*_pm` and
  `cost_per_m_served` are USD per million tokens. GPU hourly TCO is per device.
- Capacity and demand `*_tokens_day`, `total_cap_tokens_day` and all `fates`
  token counts are tokens/day; `fates.*_pct` is percent in the range 0–100.
  `token_coverage`, `value_capture_rate` and `utilization` are fractions.
- Model `prefill_memory` / `decode_memory` byte fields describe **one GPU rank on the busiest pipeline stage**,
  not a replica or cluster total. `kv_per_token` is token-growing KV bytes per
  sequence token on that rank and excludes fixed recurrent state. `*_slots` and batch limits are
  counts. `system_ram_weight_bytes_per_copy` is separate from GPU VRAM.
- Catalog GPU `mem` is bytes; `bw` and interconnect bandwidth are bytes/second;
  `bf16`, `fp8`, `fp4` are FLOP/second rooflines; `tdp_watts` is watts.
  `profiled_non_kv_gb` uses decimal GB per GPU (1 GB = 1e9 bytes).
- `co2_kg_day_total` is kg CO2/day. Charts retain the browser's mode-specific
  axis fields and units. Do not infer units from a generic `x` or `y` alone.

Precision choices describe modeled execution assumptions, not a guarantee that
a runnable checkpoint or optimized kernel exists. Catalog sources and per-model
warnings distinguish proxies and architecture omissions. These are closed-form
estimates, not benchmarks; validate with your runtime, topology and workload.

## Errors and limits

All `/api/v1/` failures use `{"error": "message"}`. Validation messages include
an input path where available. HTTP 400 means invalid JSON or scenario; 404 an
unknown endpoint; 405 an unsupported method; 413 an oversized body; 415 a missing
or unsupported JSON content type; 429 a rate limit; 500 an unexpected server
error whose internal details are not exposed. On 429, respect `Retry-After`.

The API shares the configured mutation-request budget per IP (default 600/minute)
and total request-size limit (default 2 MiB). Current values are in the catalog.
Each panel permits up to 64 GPU pools, 512 assignments and 256 use cases/custom
definitions. Requests are synchronous; keep scenarios small and avoid parallel
sweeps that overload the host. POSTs are safe to retry because they have no
persistent scenario side effects. API responses use `Cache-Control: no-store`.
