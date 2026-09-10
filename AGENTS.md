# Repository working agreements

## Scope and delivery

- Preserve unrelated user changes and inspect `git status` before editing.
- Use the current checkout and branch unless the user explicitly requests a new branch or worktree.
- Keep one writer in this checkout. Use a separate worktree for concurrent implementation and one integration owner before merging.
- Treat local edits, commit, push, merge, Docker deployment, and production verification as separate delivery states.

## Run and verify

- Use the project virtual environment when present; install with `python -m pip install -r requirements.txt` only when dependencies are missing.
- Run locally with `python app.py`; the documented default is port `5014`.
- For executable code or catalog changes, run the following once against the final edits before handoff (independent read-only checks may run concurrently):
  - `python -m compileall -q app.py calc.py data engine presentation web state.py tracking.py`
  - `python -m ruff check . && python -m ruff format --check . && python -m mypy`
  - `python -m pytest -q`
- `compileall` exits 0 for paths it cannot list, so a stale path in that
  command silently checks nothing. Keep it aligned with the module layout
  and with the same command in `README.md` and `.github/workflows/ci.yml`.
- For UI changes and picker-visible catalog additions, exercise the actual planner flow in a browser in addition to route tests.
- Format only files touched by the task before validation. Rerun checks only after relevant edits, failures, or new evidence; do not repeat a passing suite after browser cleanup or a status read.
- For documentation/instruction-only changes, review the diff, verify referenced paths/commands, and run `git diff --check`; application tests and browser runs are unnecessary.
- If a check fails in pre-existing dirty work, establish its cause from the initial diff or a focused reproduction, report it, and preserve that work. Do not broaden the task to repair unrelated failures.

## Efficient catalog additions

- Treat a single-model addition as a bounded catalog task. Start with one analogous
  entry and its tests; expand into planner internals only when a required field or
  observed result cannot be explained by that pattern.
- Batch initial status, relevant dirty diffs, and targeted reads. Use `rg -n` to
  locate symbols, then read the needed ranges once; avoid repeated whole-file
  dumps and broad searches over every ASR/realtime reference.
- Fetch the official model card, config, and checkpoint metadata together where
  possible. Pin one revision and reuse the results. Consult implementation code
  or the technical report for missing or conflicting fields. Stop researching
  once the required catalog fields and workload assumptions are supported; label
  unavailable facts and conservative proxies instead of inventing precision or
  reverse-engineering unneeded detail. Never download weights for metadata.
- For ASR, the usual edit surface is `data/models_asr.py`, `data/models.py`
  (`MODEL_ORDER`), `data/asr_support.py` (profile and quality data), and
  `data/model_sources.py`; use `tests/test_model_catalog.py` and
  `tests/test_realtime_capacity.py` for focused coverage. Read `data/groups.py`
  and `data/model_class.py` only for routing or schema questions.
- Make one coherent edit, format touched files, run focused tests while iterating,
  then the final verification matrix and one representative planner report.
  Retain source checks, key/cache invariants, units, and proxy caveats.
- Browser acceptance: start with a blank test scenario, add a suitable GPU, select
  the new model in its correct tab, and inspect assignment, model card, and the
  relevant chart. Avoid selecting the model before resetting and doing it again.
  Prefer the in-app Browser. Batch known actions with supported browser APIs and
  stable observed locators, waiting for UI updates; inspect a compact result at
  meaningful checkpoints instead of a separate click/snapshot-file/search cycle
  for every control. Stop after this flow passes and close owned test resources.
- Report the result, source/assumption caveat, checks, and any outstanding failure
  concisely. Further research or verification needs a concrete unresolved question.

## Local tooling recovery

- On this Mac the legacy path
  `/Users/ishanbaichoo/Documents/ProgrammingProjects/gpullmusagemodeler` is a
  symlink to `/Users/ishanbaichoo/Documents/OngoingProjects/gpu-calculator`.
  Use the physical checkout for new project setup and command working directories.
- If a tool reports a symlinked writable-root initialization failure, changing
  `cwd` alone may not repair the session's configured roots. Record the failure
  once and use an approved outside-sandbox command when available; do not retry
  the same failed initialization through multiple wrappers. If the patch tool
  has that same failure, use a narrow `git apply` patch or an exact, guarded file
  edit through the approved command path, then review the diff. Never bypass an
  approval rejection. If the in-app Browser fails for this reason, use the
  documented headless Playwright fallback and close it afterward.
- A durable sandbox repair requires starting the project/session with the physical
  directory as its writable root; repository instructions cannot change an
  already-running session's sandbox configuration.

## Catalog and planner correctness

- Preserve units and distinguish global totals, per-replica values, memory capacity, bandwidth, compute, context limits, latency, and throughput.
- Add numerical invariants for planner-math changes and focused catalog tests for model or hardware entries.
- For current model, hardware, pricing, or runtime claims, use primary sources, record source dates, distinguish facts from inference, and label preview assumptions.
- Do not represent a closed-form estimate as a benchmark result. State calibration requirements and uncertainty.

## Catalog invariants

- `MODELS[key].key` must equal `key`. `calc.py` keys its geometry caches
  (`_KV_ELEMS_CACHE`, `_KV_BYTES_CACHE`, `_LINEAR_STATE_CACHE`,
  `_REPLICA_KV_CACHE`) on `m.key`, so a derived entry built with
  `dataclasses.replace()` that omits `key=` silently returns the parent's
  KV-cache numbers. Guarded by
  `test_every_model_key_matches_its_catalog_key`.
- `Model` is never mutated after construction. Build variants with
  `dataclasses.replace(..., key=...)`, not attribute assignment.
- The catalog is the `data/` package, not a single `data.py`. Family entries
  live in `data/models_text.py`, `data/models_embedding.py` and
  `data/models_asr.py`; `data/models.py` assembles them. There are two
  insertion points in `data/models.py`: the `MODEL_ORDER` tuple that fixes
  picker order, and the later `MODELS.update({...})` block for entries
  derived with `replace()`. Check both.
- Picker placement is derived, not declared: `_model_kind()` routes on
  `is_asr_model` (ASR) and `is_embedding_model` (embedding), defaulting to
  LLM; `hidden` removes an entry from the picker entirely. ASR workload
  profiles use `streaming` to distinguish realtime from non-realtime models;
  `is_realtime_only` remains only as a backward-compatible alias. An otherwise correct
  entry lands in the wrong tab if these are unset.

## Catalog and hardware changes

- Model entries need: parameter count, active params for MoE, layers, hidden
  size, attention and KV heads, context length, attention variant (MLA, sliding
  window, linear attention, recurrence), weight precision and supported
  quantizations, and whether all layers quantize alike.
- Hardware entries need: VRAM, memory bandwidth, relevant precision throughput,
  vendor/category/form factor, availability, and interconnect assumptions where
  the planner uses them. Make mobile/workstation/datacenter/embedded explicit
  the way existing entries do.
- Cloud entries need: provider slug, API id, input/output price, context tier,
  and whether the model is API-only.
- After a catalog edit, generate a planner report exercising the new entry and
  check the output is plausible.

## Math changes

- State the current model in plain terms before changing a formula.
- Check units, per-GPU versus cluster totals, runtime non-KV memory versus KV
  cache, quantization granularity and partial-layer quantization, MoE active
  versus total params, prefix cache hit rate, batching, speculative decoding,
  and cliff effects.
- Where a mechanism is too detailed for closed form, use a conservative
  approximation and say what is not simulated.
- Run a planner report before and after; report both.

## Deployment and completion

- Keep `WEB_CONCURRENCY=1` while planner state is process-local.
- Run Docker build/deploy or modify production only when explicitly requested, then verify the named target and health/user flow.
- Do not say "fixed", "verified", or "deployed" unless the requested observation surface demonstrates the expected behavior.
