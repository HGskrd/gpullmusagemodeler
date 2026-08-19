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
  - `python -m compileall -q app.py calc.py data.py state.py tracking.py`
  - `python -m pytest -q`
- For UI changes, exercise the actual planner flow in a browser in addition to route tests.

## Catalog and planner correctness

- Preserve units and distinguish global totals, per-replica values, memory capacity, bandwidth, compute, context limits, latency, and throughput.
- Add numerical invariants for planner-math changes and focused catalog tests for model or hardware entries.
- For current model, hardware, pricing, or runtime claims, use primary sources, record source dates, distinguish facts from inference, and label preview assumptions.
- Do not represent a closed-form estimate as a benchmark result. State calibration requirements and uncertainty.

## Deployment and completion

- Keep `WEB_CONCURRENCY=1` while planner state is process-local.
- Run Docker build/deploy or modify production only when explicitly requested, then verify the named target and health/user flow.
- Do not say "fixed", "verified", or "deployed" unless the requested observation surface demonstrates the expected behavior.
