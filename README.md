# GPU/LLM Usage Modeler

A Flask web application for planning and modeling GPU capacity for multi-model vLLM deployments. It lets you configure GPU pools, LLM workloads, and traffic distributions to project infrastructure costs and throughput.

## Requirements

- Python 3.10+

## Setup

1. **Clone the repository**

   ```bash
   git clone <repo-url>
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
   flask run
   ```

   The app will be available at `http://127.0.0.1:5000`.

## Environment Variables

| Variable | Description |
|---|---|
| `PLANNER_ADMIN_PASSWORD` | Password for the admin interface |
