# GPU/LLM Usage Modeler

A Flask web application for planning and modeling GPU capacity for multi-model vLLM deployments. It lets you configure GPU pools, LLM workloads, and traffic distributions to project infrastructure costs and throughput.

## Accuracy and Scope

This project is a closed-form capacity estimator, not a request-level simulator. It combines published hardware rooflines and model architecture metadata with explicit efficiency, runtime-memory, batching, topology, prefix-cache, and workload-shape assumptions. Results are most useful for comparing scenarios and identifying capacity constraints; they are not a substitute for benchmarking the exact model, quantization, vLLM version, hardware topology, and service-level objective you intend to deploy.

Model-fit routing and same-hardware swap recommendations can use sparse, sourced quality anchors for coding, reasoning, long-context, multilingual, and vision workloads. Every missing model/domain pair falls back to the existing global quality score; benchmark names and sources remain explicit because vendor harnesses are not interchangeable.

Before using a result for procurement or financial planning:

1. Review the pre-filled amortized GPU-hour TCO and replace it with your actual quote or internal chargeback rate when available.
2. Match the input/output distributions and interactive-versus-batch mix to the real workload.
3. Calibrate bandwidth efficiency, compute efficiency, non-KV runtime memory, and prefix-cache hit rate against representative vLLM measurements.
4. Review model and hardware provenance, confidence, preview status, and context-window limits in the UI.
5. Treat maximum-throughput points separately from latency-constrained interactive capacity.

Useful vLLM calibration signals include request counts, prompt and generation tokens, KV-cache usage, prefix-cache hits, time to first token, inter-token latency, and request throughput. Keep a before/after planner report with the benchmark fixture whenever changing a catalog entry or formula.

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
| `PLANNER_ADMIN_LOGIN_ATTEMPTS_PER_MINUTE` | Per-IP admin login attempt limit; defaults to `10` |
| `PLANNER_SECURE_COOKIES` | Require HTTPS-only session and visitor cookies; enable behind an HTTPS reverse proxy |
| `PLANNER_BEHIND_PROXY` | Trust `X-Forwarded-For` and `X-Forwarded-Proto` from exactly one reverse-proxy hop; defaults to `false`. Only enable when the proxy overwrites these headers, otherwise clients can spoof IPs to evade rate limits |
| `PLANNER_CLOUD_POLICY` | Optional path to a JSON policy that restricts corporate-cloud models, adds gateway presets, and overrides negotiated input/output prices; invalid policies fail at startup |
| `DEBUG` / `FLASK_DEBUG` | Enable Flask debug mode for local development |

### Corporate Cloud Policy

Set `PLANNER_CLOUD_POLICY` to a JSON file when procurement exposes only part of the public cloud catalog or negotiated prices differ from catalog prices. All sections are optional; model keys must already exist in `data.py`.

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

## Scenario Data and Privacy

The planner stores complete A/B scenario snapshots—including hardware, model assignments, topology, costs, workload shapes, and use-case economics—so administrators can inspect how planning decisions evolve. Snapshot persistence is enabled by default, bounded to 90 days and 250 snapshots per browser tab; set either limit to `0` for unlimited history. Stored rows omit use-case definitions that match the built-in presets and reattach them on read, so custom libraries are preserved without duplicating preset content in every row.

Snapshots are stored transactionally in `instance/planner_snapshots.sqlite3`; the legacy JSON file is imported on first use when present. The calculator discloses this behavior and provides a **Delete my scenarios** action that removes the current visitor's persisted snapshots and in-memory state. Set `PLANNER_TRACKING_ENABLED=false` when a deployment should not retain scenarios, or configure the optional retention limits above.

Complete scenarios can be exported and imported as versioned JSON from the calculator. Treat these files as potentially sensitive infrastructure-planning data.

## Development and Validation

Run the full regression suite and source compilation checks before changing planner math or catalog data:

```bash
python -m compileall -q app.py calc.py data.py state.py tracking.py
python -m pytest -q
```

Planner-math changes should include numerical invariants for units, global versus per-replica totals, memory accounting, context limits, and latency semantics. Web changes should include route tests for validation, state isolation, authentication, persistence, and retention. The GitHub Actions workflow runs these checks on Python 3.10 and 3.12.
