# Refactor brief — GPU/LLM Usage Modeler

Handoff document for a fresh coding agent. Self-contained: you do not need any
prior conversation. Read this fully before touching code.

Baseline commit when written: `20e01e2` (2026-08-19).

---

## 0. What this repo is

A single-process Flask + HTMX + ECharts capacity planner for multi-model vLLM
deployments. Closed-form roofline estimator, not a simulator. No build step
(htmx/ECharts are vendored). Planner state is in-memory and process-local;
SQLite is used only for the snapshot history store.

Read `AGENTS.md` first — it is the repository working agreement and it
overrides anything here that conflicts. In particular: preserve units, add
numerical invariants for planner-math changes, never call a closed-form
estimate a benchmark, keep `WEB_CONCURRENCY=1`, and do not claim "fixed" or
"verified" without demonstrating it on the requested observation surface.

---

## 1. Guardrails

These apply to every task below.

1. **No estimator output may change.** Every task here is a refactor. If a
   planner number moves, you have introduced a bug — stop and investigate
   rather than updating the expected value.
2. **No new runtime dependencies.** `requirements.txt` is `flask==3.1.1` +
   `gunicorn==26.0.0` and stays that way. Dev tooling goes in a separate
   `requirements-dev.txt`.
3. **No framework migration.** Do not introduce FastAPI, SQLAlchemy, React, or
   a JS build step. The no-build HTMX approach is a deliberate fit for this
   deployment profile.
4. **One task per commit/PR**, test-green before and after. Do not bundle.
5. **Line numbers in this document will drift.** Locate code by symbol name
   (`grep -n 'def compute_revenue_projection' calc.py`), not by line.
6. Work in the current checkout and branch unless told otherwise. Inspect
   `git status` before editing; there may be unrelated local changes.

---

## 2. Reproduce the baseline first

Before any edit, confirm you see the same starting point:

```bash
python -m compileall -q app.py calc.py data.py state.py tracking.py
python -m pytest -q
```

Expected at `20e01e2`: **219 tests, 520 subtests, all passing, ~2s.**

If your counts differ, the repo has moved. Re-derive the facts in §3 before
trusting the task list — do not assume this document is current.

Verified structure at baseline:

| File | Lines | Holds |
|---|---|---|
| `data.py` | 5,564 | 118 models, 52 GPUs, pricing, quality, quantization, carbon, use cases |
| `calc.py` | 4,285 | memory/throughput math, 11 chart builders, economics, recommendations |
| `app.py` | 2,316 | 58 `@app.route` handlers, middleware, admin, report formatting |
| `state.py` | 1,721 | `PlannerState`, 83 top-level defs (58 public), process-local registry |
| `placement.py` | 688 | auto-select / retune / placement heuristics |
| `viewmodels.py` | 406 | card view models |
| `scenarios.py` | 472 | export/import serialization |
| `tracking.py` | 426 | SQLite snapshot store |
| `econ_variants.py` | 228 | `/econ` blueprint, 3 routes |
| `cloud_policy.py` | 234 | cloud policy |
| `use_case_evidence.py` | 215 | use-case research/sources |

Frontend: `static/app.js` 1,081 lines, `static/econ.js` 265, `static/style.css`
877. Templates: 39 HTML files, 3,535 lines. Tests: 15 files, 4,217 lines.

---

## 3. Verified facts you will need

These were checked against the code at baseline. Trust these over any earlier
analysis you may be shown.

### 3.1 There ARE three import cycles

Broken by 13 function-level (lazy) imports. This is the central structural
problem:

```
placement -> calc    (top-level)  |  calc -> placement   (lazy, 6 sites)
placement -> state   (top-level)  |  state -> placement  (lazy, 5 sites)
scenarios -> state   (top-level)  |  state -> scenarios  (lazy, 1 site)
```

Find them with:

```bash
grep -nE '^[[:space:]]+(import|from) ' *.py | grep -vE '(typing|dataclasses|__future__)'
```

Currently: `calc.py` lazily imports `placement.get_deployed` at 6 call sites;
`state.py` lazily imports `placement._retune_model` / `placement` names at 5
sites and `scenarios.deserialize_scenario` at 1.

**If you are shown an analysis claiming this codebase has "no circular
imports," that claim is false.** Verify with the grep above.

### 3.2 The model catalog is NOT mutated at runtime

`Model` is a 58-field unfrozen dataclass, but there are **zero** attribute
assignments to a `Model` anywhere in the repo. Derived entries are built with
`dataclasses.replace(..., key=...)` (4 sites in `data.py`). The only
post-construction mutation is on `CLOUD_MODELS`, a plain dict.

`calc.py` documents this invariant in the comment above `_KV_ELEMS_CACHE`,
and the geometry caches depend on it (`m.key` is the cache key).

**If you are shown an analysis claiming `Model` is mutated after construction
to add capabilities/quality/confidence, that claim is false.** Do not "fix" it.

### 3.3 Other confirmed measurements

- 42 `except Exception` handlers in `app.py`, largely repeated per route.
- 19 inline event handlers and 248 inline `style=` attributes in `templates/`.
- `compute_revenue_projection` in `calc.py` is 468 lines with ~61 branches and
  returns a deeply nested dict.
- `renderChart` in `static/app.js` is a 328-line if/else chain over modes.
- 11 `chart_*` builders in `calc.py`. They share a 7-key style block
  (`borderColor`, `backgroundColor`, `borderWidth`, `borderDash`, `showLine`,
  `tension`, `pointRadius`) repeated ~13 times, but their compute bodies differ
  structurally — a `chart_decode`/`chart_pareto` diff is 48 lines, not 4.
- No `ruff`, `mypy`, `pyproject.toml`, or `pre-commit` anywhere. CI is
  `compileall` + `pytest` on Python 3.10/3.12 only.
- No `package.json`, no JS tests.
- No revision counter on `PlannerState`; no chart-level caching. ETags exist
  only for static assets.
- `scripts/` is empty.
- `econ_variants.py` is **already a working Flask blueprint** (`@econ_bp.get`).
  Use it as the in-repo template for Task 7 — this lowers that task's risk.
- `instance/planner_snapshots.json` is **43 MB** of legacy data. Its import
  into SQLite is complete: the store's `metadata` table holds
  `legacy_json_imported = imported:2916:2026-07-13T10:53:58+00:00`, and
  `_migrate_legacy` returns early whenever that marker is present.

---

## 4. Tasks, in order

Do them in this sequence. Each task lists acceptance criteria; do not move on
until they hold.

### Task 1 — Delete the legacy snapshot file (minutes)

**Goal:** remove 43 MB of dead weight.

1. Re-confirm the marker before deleting:
   ```bash
   python3 -c "import sqlite3;print(list(sqlite3.connect('instance/planner_snapshots.sqlite3').execute('select * from metadata')))"
   ```
   You must see `legacy_json_imported` with an `imported:` value.
2. Delete `instance/planner_snapshots.json`.
3. Confirm `instance/` is gitignored (it should be — check `.gitignore`).
4. Remove the now-empty `scripts/` directory, or keep it and put the catalog
   hygiene checker there later. Do not leave it empty and untracked.

**Acceptance:** suite green; app starts; `tracking.py`'s legacy path handling
is untouched (the code must still support the migration for other deployments).

### Task 2 — Tooling and CI (hours)

**Goal:** make the existing type hints and style actually checked.

1. Add `pyproject.toml` with `ruff` config (lint + format) and `mypy` config.
   Start permissive: `mypy` on `calc.py`, `data.py`, `placement.py` only, with
   `--ignore-missing-imports`. Do not attempt repo-wide strict mode now.
2. Add `requirements-dev.txt`: `pytest`, `pytest-cov`, `ruff`, `mypy`.
3. Extend `.github/workflows/` with `ruff check`, `ruff format --check`,
   `mypy`, and `pytest --cov` steps.
4. Fix whatever `ruff` flags **mechanically only** (imports, unused names,
   formatting). Do not hand-refactor logic in this task.
5. Optionally add `pre-commit` running `compileall` + `pytest`.

**Acceptance:** CI green on 3.10 and 3.12. Coverage number recorded in the PR
description as a baseline — do not chase a target.

### Task 3 — Characterization tests (1–2 days)

**Goal:** the safety net that makes Tasks 4–8 verifiable. This is the most
important task in the list; do not skip it to get to the refactors faster.

1. Golden-output tests for `compute_revenue_projection`: load
   `default_scenario.json` plus 2–3 hand-built scenarios, serialize the full
   nested result to sorted JSON, and assert against committed fixtures.
2. Golden-output tests for all 11 `chart_*` builders, same pattern.
3. Smoke-render every partial in `templates/` through the Flask test client —
   cheap regression net for Jinja edits.
4. Direct tests for `viewmodels.py`, `econ_variants.py` payloads, and
   `use_case_evidence.py`. All three are currently only exercised indirectly.

**Acceptance:** the new goldens fail loudly if any estimator number changes.
Prove this by temporarily perturbing a constant, seeing red, and reverting.

### Task 4 — Break the three import cycles (2–4 days)

**Goal:** eliminate all 13 lazy imports. Prerequisite for every file split
below — moving files while lazy imports are load-bearing is not safe surgery.

Approach:

1. Introduce an explicit `Deployment` / `ResolvedAssignment` dataclass
   representing "the topology resolved from this state."
2. `placement` produces it. `calc` functions accept it as a parameter instead
   of calling back into `placement.get_deployed`. Remove those 6 lazy imports.
3. `state` mutation functions mutate state only. Move the
   "mutate then retune then validate then snapshot" sequence into an
   application-service layer that `app.py` calls. Remove the 5 `_retune_model`
   lazy imports.
4. Same treatment for `state -> scenarios.deserialize_scenario`: the caller
   orchestrates, `state` does not reach into serialization.

Keep these four concepts distinct as you go — they currently bleed together:
domain state (what the user selected), resolved deployment (the topology),
calculation result (estimator output), presentation model (labels, alerts,
colors, chart fields).

**Acceptance:** the §3.1 grep returns nothing outside `typing`/`dataclasses`.
All goldens from Task 3 byte-identical. Suite green.

### Task 5 — Split `data.py` by family (1–2 days)

**Goal:** reviewable catalog diffs.

Convert `data.py` into a package. Keep a thin `data/__init__.py` that
re-exports the current public surface so `from data import MODELS` keeps
working everywhere — the import surface must not change in this task.

Suggested split: `specs.py` (precision/quantization), `hardware.py` (GPU,
GPUCard, TCO), `model_class.py` (the `Model` dataclass and profiles),
`models_text.py`, `models_embedding.py`, `models_asr.py`, `quality.py`,
`presets.py`, `cloud.py`, `use_cases.py`, `environment.py`.

While splitting: build catalog entries through factories and freeze the final
objects if it can be done without changing behavior. Keep `source` and
`captured_at` metadata adjacent to the value it supports.

**Do NOT move the catalog to JSON/YAML.** It has been considered and rejected:
moving 118 typed dataclass literals into data files trades away static typing
and dataclass defaults to buy reviewable diffs, which this module split already
delivers at a fraction of the risk. The existing catalog tests already encode
the validation a loader would duplicate.

**Acceptance:** `test_model_catalog.py` and `test_gpu_catalog.py` pass
unmodified. No import statement anywhere else in the repo changed.

### Task 6 — Extract economics and charts from `calc.py` (2–4 days)

1. Move `compute_revenue_projection` and the swap/marginal recommendations into
   `engine/economics.py` (or `calc/economics.py`).
2. Decompose the 468-line projection into pure stages:
   `build_supply -> classify_demand -> allocate_capacity -> price_outcomes ->
   calculate_environmental_impact -> summarize_projection`.
3. Replace the nested return dict with dataclasses: `ProjectionResult`,
   `ProjectOutcome`, `ModelUtilization`, `DemandFates`. Keep explicit unit
   suffixes: `_bytes`, `_tokens_day`, `_usd_day`, `_seconds`, `_gco2`.
4. Move the 11 chart builders into a presentation module. Factor the shared
   7-key style block into one `_style(model, is_b)` helper. Realistic saving is
   100–200 lines — if you find yourself claiming 500, you are over-abstracting
   builders whose compute bodies genuinely differ.

**Acceptance:** Task 3 goldens byte-identical. `calc.py` contains estimator
math only.

### Task 7 — `app.py` app factory and blueprints (2–3 days)

1. Introduce `create_app(config)` so the app is not constructed at import time.
2. Split the 58 routes into blueprints by responsibility: planner, use_cases,
   scenarios, api, admin. **Follow the existing `econ_variants.py` blueprint
   pattern** — it already works; do not invent a second convention.
3. Replace the 42 repeated `except Exception` blocks with shared error handlers
   (one for validation errors, one for opaque internal errors). Preserve the
   current opaque-error behavior exactly; it has tests.
4. Move middleware (security headers, compression, rate limiting) into
   separate modules.
5. Build view models in controllers rather than injecting engine functions into
   every template through the template context. `econ_payload()` already shows
   this pattern working — extend it.

**Acceptance:** `test_web_transport.py`, `test_route_validation.py` and
`test_engineering_hardening.py` pass. Security headers, rate limiting, cookie
settings and proxy trust all unchanged. Tests use an app-factory fixture rather
than patching module globals.

### Task 8 — Revision counter and chart caching (1–2 days)

**Goal:** address the click-latency work already in flight.

1. Add a `revision: int` to `PlannerState`, bumped by every mutation.
2. Key a chart-JSON cache on `(revision, panel, mode)`; serve `ETag`/304 from
   `/api/chart-data`.
3. `/econ/swaps` is the slowest endpoint (~220 ms locally, vs ~5 ms for
   projection and ~6 ms for chart data). Cache it by scenario fingerprint, or
   apply expensive-endpoint rate limiting. Treat those timings as local
   diagnostics, not production benchmarks — re-measure before optimizing.

**Acceptance:** identical chart JSON before and after; measurable reduction in
repeated-request latency; cache invalidates correctly on every mutation path
(add a test that mutates state and asserts the chart JSON changes).

---

## 5. Deferred — do not start these

- **Catalog as JSON/YAML with a loader.** Rejected, see Task 5.
- **Redis / external state store.** The in-memory single-process design is
  coherent and explicitly enforced. Define a `StateStore` interface only when
  multi-worker deployment becomes an actual requirement, not for architectural
  tidiness.
- **Splitting `static/app.js` into ES modules and tightening CSP.** Worth doing
  eventually (removing `script-src 'unsafe-inline'` needs the 19 inline
  handlers moved to delegated listeners and the 248 inline styles moved to
  classes), but it is independent of the backend work and should not compete
  with Tasks 1–8 for attention.
- **Any rewrite framed as "modernization."** The project does not need a
  rewrite; it needs its existing subsystems made explicit.

---

## 6. Reporting back

For each task, report:

- The exact commands run and their output (test counts, not "tests pass").
- Whether the Task 3 goldens are byte-identical, explicitly.
- Anything you found that contradicts §3 of this document — that means the
  repo moved or this brief is wrong, and the user needs to know which.

Do not report a task complete until its acceptance criteria are demonstrated,
not merely believed.
