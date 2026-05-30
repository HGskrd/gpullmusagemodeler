# GPU/LLM Usage Modeler

A Flask web application for planning and modeling GPU capacity for multi-model vLLM deployments. It lets you configure GPU pools, LLM workloads, and traffic distributions to project infrastructure costs and throughput.

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

The interactive planner state is currently held in process memory. Keep `WEB_CONCURRENCY=1` in production and use `GUNICORN_THREADS` for request concurrency. Running more than one worker process can make a browser session appear to swap between different planner states because separate workers do not share memory.

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
| `WEB_CONCURRENCY` | Gunicorn worker process count for Docker/systemd deployments; keep this at `1` while planner state is in process memory |
| `GUNICORN_THREADS` | Gunicorn thread count; defaults to `4` |
| `GUNICORN_TIMEOUT` | Gunicorn request timeout in seconds; defaults to `120` |
| `DEBUG` / `FLASK_DEBUG` | Enable Flask debug mode for local development |
