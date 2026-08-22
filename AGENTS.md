# Repository working agreements

## Scope and delivery

- Preserve unrelated user changes and inspect `git status` before editing.
- Use the current checkout and branch unless the user explicitly requests a new branch or worktree.
- Keep one writer in this checkout. Use a separate worktree for concurrent implementation and one integration owner before merging.
- Treat local edits, commit, push, merge, Docker deployment, and production verification as separate delivery states.

## Run and verify

- Install with `python -m pip install -r requirements.txt`.
- Run locally with `python app.py`; the documented default is port `5014`.
- Before handoff, run:
  - `python -m compileall -q app.py calc.py data engine presentation web state.py tracking.py`
  - `python -m ruff check . && python -m ruff format --check . && python -m mypy`
  - `python -m pytest -q`
- `compileall` exits 0 for paths it cannot list, so a stale path in that
  command silently checks nothing. Keep it aligned with the module layout
  and with the same command in `README.md` and `.github/workflows/ci.yml`.
- For UI changes, exercise the actual planner flow in a browser in addition to route tests.

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
  `is_realtime_only` (ASR) and `is_embedding_model` (embedding), defaulting to
  LLM; `hidden` removes an entry from the picker entirely. An otherwise correct
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
