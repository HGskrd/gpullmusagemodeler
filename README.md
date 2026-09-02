# GPU/LLM Usage Modeler

A Flask web application for planning and modeling multi-model LLM deployments across GPUs and AI accelerators. It lets you configure accelerator pools, LLM workloads, and traffic distributions to project infrastructure costs and throughput.

## Accuracy and Scope

This project is a closed-form capacity estimator, not a request-level simulator. It combines published hardware rooflines and model architecture metadata with explicit efficiency, runtime-memory, batching, topology, prefix-cache, and workload-shape assumptions. Results are most useful for comparing scenarios and identifying capacity constraints; they are not a substitute for benchmarking the exact model, quantization, serving runtime/version (for example vLLM), hardware topology, and service-level objective you intend to deploy.

Model-fit routing and same-hardware swap recommendations use weighted task-quality profiles across coding, reasoning, long-context, multilingual, vision, and general capability. Domain evidence is blended with a weighted geometric mean so one strong axis cannot fully hide a weak required axis. Every missing model/domain pair falls back to the confidence-adjusted global quality score; benchmark names, raw scores, crosswalks, and sources remain explicit because vendor harnesses are not interchangeable.

Before using a result for procurement or financial planning:

1. Review the pre-filled amortized GPU-hour TCO and replace it with your actual quote or internal chargeback rate when available.
2. Match the input/output distributions and interactive-versus-batch mix to the real workload.
3. Calibrate bandwidth efficiency, compute efficiency, non-KV runtime memory, and prefix-cache hit rate against representative serving-runtime measurements.
4. Review model and hardware provenance, confidence, preview status, and context-window limits in the UI.
5. Treat maximum-throughput points separately from latency-constrained interactive capacity.

For vLLM deployments, useful calibration signals include request counts, prompt and generation tokens, KV-cache usage, prefix-cache hits, time to first token, inter-token latency, and request throughput. Use analogous scheduler and cache metrics for other runtimes. Keep a before/after planner report with the benchmark fixture whenever changing a catalog entry or formula.

Hybrid/recurrent architectures are modeled with separate token-growing attention KV and fixed recurrent-state traffic. The default decode path conservatively reads and writes recurrent state once per target block; runtime-specific buffering such as ReplaySSM still requires calibration. Kimi K3 prefill capacity also includes a conservative Block AttnRes activation bound. MoE estimates do not yet charge expert-parallel dispatch/combine collectives, and the planner surfaces that omission wherever a MoE model is selected.

Decoder prefill charges the causal attention triangle (each prompt query attends only to positions at or before it); bidirectional encoder prefill (embedding models) charges the full rectangle. Absorbed-MLA decode scores the joint latent+rope key, so its per-row attention FLOPs follow the latent geometry rather than `4 × head_dim`. Prefix-cache hits remove prefill compute but not KV residency: capacity is charged at the full prompt length per sequence without simulating prefix sharing across concurrent requests. Speculative-decoding verification re-reads the sequence KV for every drafted position — conservative versus fused verify kernels that load KV once per cycle — and attached drafters (EAGLE-3, DFlash, DSpark, draft models) are additionally charged their own prompt forward in prefill, while MTP and n-gram drafting are not. Decode slot counts are average-occupancy figures under steady-state continuous batching (`input + output/2` resident KV); a fully synchronized cohort peaks at `input + output`. Reported TTFT is the batch-prefill makespan (the last user in the batch), not the mean.

Cloud entries may define an input-length threshold and a second set of long-context input, cached-input, and output rates. The calculator switches tiers only when the request input is strictly above the published threshold.

## Requirements

- Python 3.10+
- Docker and Docker Compose, for container deployment

By default the app listens on `0.0.0.0:5014`, so it is reachable from the local network or from a VPS public interface when firewall rules allow it.

## Docker Deployment

1. **Clone the repository**

   ```bash
   git clone https://github.com/HGskrd/gpullmusagemodeler.git
   cd gpullmusagemodeler
   ```

2. **Create an environment file**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` before deploying. Set a strong `PLANNER_SECRET_KEY`, and set `PLANNER_ADMIN_PASSWORD` if you want to enable `/admin`.
   For an internet-facing deployment, terminate TLS at a reverse proxy and set `PLANNER_SECURE_COOKIES=true` and `PLANNER_BEHIND_PROXY=true`. Only enable `PLANNER_BEHIND_PROXY` when the proxy sets/overwrites `X-Forwarded-For` itself; if clients can inject that header, they can spoof IPs to evade the per-IP rate limits.

3. **Start the app**

   ```bash
   docker compose up --build -d
   ```

   The app will be available at `http://<server-ip>:5014`.

4. **View logs or stop the app**

   ```bash
   docker compose logs -f
   docker compose down
   ```

The Compose setup stores planner snapshots in the named Docker volume `gpullmusagemodeler_planner-instance`, mounted at `/app/instance` inside the container.

The interactive planner state is currently held in process memory. Keep `WEB_CONCURRENCY=1` in production and use `GUNICORN_THREADS` for request concurrency. The shipped Docker command refuses a `WEB_CONCURRENCY` value other than `1`, because separate worker processes do not share memory and would make a browser session appear to swap between different planner states. `compose.yaml` pins it to `1`. If you use a different WSGI command or process manager, you must likewise configure exactly one process.

### Manual Docker Run

```bash
docker build -t gpullmusagemodeler .
docker run -d \
  --name gpullmusagemodeler \
  --restart unless-stopped \
  --env-file .env \
  -p 5014:5014 \
  -v gpullmusagemodeler-instance:/app/instance \
  gpullmusagemodeler
```

## Non-Docker Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/HGskrd/gpullmusagemodeler.git
   cd gpullmusagemodeler
   ```

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and set your values (see `.env.example` for the required variables).

5. **Run the app**

   ```bash
   python app.py
   ```

   The app will be available at `http://localhost:5014` and `http://<your-lan-ip>:5014`.

To change the bind address or port, edit `.env`:

```bash
HOST=0.0.0.0
PORT=5014
```

For local debugging, add `DEBUG=1` or `FLASK_DEBUG=1`.

## Environment Variables

| Variable | Description |
|---|---|
| `HOST` | Bind address for `python app.py`; defaults to `0.0.0.0` |
| `PORT` | HTTP port; defaults to `5014` |
| `PLANNER_ADMIN_PASSWORD` | Password for the admin interface |
| `PLANNER_SECRET_KEY` | Flask session signing key; set this to a strong random value before deployment |
| `WEB_CONCURRENCY` | Gunicorn worker count used by the shipped Docker command; must stay at `1` while planner state is in process memory |
| `GUNICORN_THREADS` | Gunicorn thread count; defaults to `4` |
| `GUNICORN_TIMEOUT` | Gunicorn request timeout in seconds; defaults to `120` |
| `PLANNER_TRACKING_ENABLED` | Persist complete planner scenario snapshots; defaults to `true` |
| `PLANNER_SNAPSHOT_RETENTION_DAYS` | Delete snapshots older than this many days; defaults to `90`; `0` keeps the full history indefinitely |
| `PLANNER_SNAPSHOT_MAX_PER_TAB` | Maximum snapshots retained per browser tab; defaults to `250`; `0` keeps every snapshot |
| `PLANNER_ADMIN_PAGE_SIZE` | Snapshots rendered per admin page; defaults to `100` |
| `PLANNER_STATE_TTL_SECONDS` | Idle lifetime for an in-memory planner scope; defaults to `86400` |
| `PLANNER_STATE_MAX_SCOPES` | Maximum active in-memory browser scopes; defaults to `5000` |
| `PLANNER_MAX_TABS_PER_VISITOR` | Maximum active tab scopes accepted per visitor; defaults to `64` |
| `PLANNER_MAX_IMPORT_BYTES` | Maximum scenario/use-case JSON import size; defaults to `1048576` |
| `PLANNER_MAX_REQUEST_BYTES` | Maximum total HTTP request size; defaults to `2097152` |
| `PLANNER_RATE_LIMIT_PER_MINUTE` | Per-IP mutation-request limit; defaults to `600` |
| `PLANNER_RATE_LIMIT_MAX_IDENTITIES` | Maximum source addresses tracked for rate limiting; defaults to `20000`. Expired windows are swept first; if the map is still over the cap the least recently active entries are dropped |
| `PLANNER_ADMIN_LOGIN_ATTEMPTS_PER_MINUTE` | Per-IP admin login attempt limit; defaults to `10` |
| `PLANNER_SECURE_COOKIES` | Require HTTPS-only session and visitor cookies; enable behind an HTTPS reverse proxy |
| `PLANNER_BEHIND_PROXY` | Trust `X-Forwarded-For` and `X-Forwarded-Proto` from exactly one reverse-proxy hop; defaults to `false`. Only enable when the proxy overwrites these headers, otherwise clients can spoof IPs to evade rate limits |
| `PLANNER_CLOUD_POLICY` | Optional path to a JSON policy that restricts corporate-cloud models, adds gateway presets, and overrides negotiated input/output prices; invalid policies fail at startup |
| `DEBUG` / `FLASK_DEBUG` | Enable Flask debug mode for local development |

### Corporate Cloud Policy

Set `PLANNER_CLOUD_POLICY` to a JSON file when procurement exposes only part of the public cloud catalog or negotiated prices differ from catalog prices. All sections are optional; model keys must already exist in the `data` package.

```json
{
  "allowed_models": ["gemini-flash-lite", "gemini-pro"],
  "price_overrides": {
    "gemini-pro": {
      "in_per_m": 1.0,
      "cached_in_per_m": 0.1,
      "out_per_m": 8.0
    }
  },
  "corpo_presets": {
    "negotiated": {
      "label": "Negotiated gateway",
      "models": ["gemini-flash-lite", "gemini-pro"]
    }
  }
}
```

The policy is validated and loaded once at startup. Unknown models, invalid prices, malformed sections, or custom presets outside the allowlist stop startup with a clear error. Restart the app after changing the file.

## Response Compression and Security Headers

HTML, JSON, and HTMX responses over 1 KB are gzipped when the client advertises
`Accept-Encoding: gzip`. Static assets are compressed separately and cached in
memory keyed by file mtime and size, because Flask serves them in passthrough
mode that the response-level compressor cannot touch. Compressed static
responses carry a `-gzip`-suffixed ETag so a shared cache keeps the encoded and
identity variants apart; revalidation of either variant still returns `304`.

Every response carries `Content-Security-Policy`, `X-Content-Type-Options`,
`Referrer-Policy`, and `X-Frame-Options`. The policy resolves all fetchable
content to `'self'` — no template references an external origin — while keeping
`'unsafe-inline'` for scripts and styles, which the templates still rely on.
Tighten those two directives if you remove the inline script block in
`templates/base.html` and the inline `style` attributes.

## Scenario Data and Privacy

The planner stores complete A/B scenario snapshots—including hardware, model assignments, topology, costs, workload shapes, and use-case economics—so administrators can inspect how planning decisions evolve. Snapshot persistence is enabled by default, bounded to 90 days and 250 snapshots per browser tab; set either limit to `0` for unlimited history. Stored rows omit use-case definitions that match the built-in presets and reattach them on read, so custom libraries are preserved without duplicating preset content in every row.

Snapshots are stored transactionally in `instance/planner_snapshots.sqlite3`; the legacy JSON file is imported on first use when present. The calculator discloses this behavior and provides a **Delete my scenarios** action that removes the current visitor's persisted snapshots and in-memory state. Set `PLANNER_TRACKING_ENABLED=false` when a deployment should not retain scenarios, or configure the optional retention limits above.

Complete scenarios can be exported and imported as versioned JSON from the calculator. Treat these files as potentially sensitive infrastructure-planning data.

## Module Layout

The import graph is acyclic and every module imports downward only, with one
recorded exception noted below. `tests/test_architecture.py` enforces both.

| Layer | Modules | Owns |
|---|---|---|
| `web/` | `planner`, `use_cases`, `scenarios`, `api`, `admin`, `econ` | The 61 route handlers, one blueprint per responsibility |
| | `helpers`, `middleware`, `config`, `cache`, `session_store` | Form coercion and the HTMX envelope; security headers, rate limiting and compression; environment and constants; derived-response caches; the process-local state registry |
| `presentation/` | `charts`, `econ`, `model_cards`, `reports`, `formatting` | Chart series, economics payloads, model card view models, the plain-text report, number formatting |
| `planner_service.py` | | Orchestration: mutate, retune, validate |
| `engine/`, `calc.py`, `placement.py` | `economics`, `deployment.py` | Estimator math, topology resolution, projection economics |
| `state.py`, `scenarios.py` | | `PlannerState` and its mutators; scenario serialization |
| `data/` | 13 catalog modules | Models, GPUs, pricing, quality, presets, use cases |

`app.py` is the composition root only: it builds `create_app()`, registers the
blueprints, middleware, error handlers and template filters, and holds no
routes or business logic.

Two invariants are worth knowing before editing:

- **Revision bumps.** Derived caches key on `PlannerState.revision`. Mutators in
  `state.py` carry `@bumps_revision`; code outside it (notably `placement.py`)
  must bump explicitly. `tests/test_revision_invariant.py` enforces both.
- **Catalog order.** `data/models.py` builds `MODELS` by walking `MODEL_ORDER`,
  so a family entry missing from that tuple never reaches the picker. The module
  raises at import if that happens.
- **No lazy imports.** Function-level imports are how the previous cycles were
  hidden, so they are rejected outside `typing`/`dataclasses`.
- **Release retirement.** Superseded open releases live in
  `data/model_archive.py`, outside `MODELS`. Each record pins its successor,
  source revision, date, and migration safety; only footprint-compatible
  aliases are rewritten during scenario import.
- **Native precision.** A model card's first precision option is its released
  checkpoint format, backed by the actual BF16/FP8/FP4 calculation key. Exact
  artifact profiles override weight bytes and mixed-compute shares; generic
  conversions remain explicitly labeled estimates.

One upward dependency remains and is recorded as an exception in
`tests/test_architecture.py`: `state.py` calls `avg_dist`,
`resolve_spec_runtime` and `valid_strategies` from `calc.py` inside its
mutators. Clearing it means moving those calls up into `planner_service.py`.

## Development and Validation

Run the full regression suite and source compilation checks before changing planner math or catalog data:

```bash
python -m compileall -q app.py calc.py data engine presentation web state.py tracking.py
python -m pytest -q
```

Planner-math changes should include numerical invariants for units, global versus per-replica totals, memory accounting, context limits, and latency semantics. Web changes should include route tests for validation, state isolation, authentication, persistence, and retention. The GitHub Actions workflow runs these checks on Python 3.10 and 3.12.
