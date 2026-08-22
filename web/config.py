"""Environment loading and process-wide planner configuration.

Imported before any blueprint, so ``.env`` is applied and the cloud policy is
configured before route modules read these constants. Living here rather than
in ``app.py`` lets blueprints read configuration without importing the
application module back and creating a cycle.
"""

from __future__ import annotations

import os
from pathlib import Path

import cloud_policy
from tracking import SnapshotStore

# The repository root: this module sits one level down in web/, and the
# snapshot store and .env path are both resolved against the root.
BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_dotenv(BASE_DIR / ".env")

cloud_policy.configure_from_env()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_positive_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _enforce_single_worker() -> None:
    # Planner state and per-scope locks live in this process; multiple gunicorn
    # workers would silently split visitor state across processes. Refuse to
    # boot rather than serve inconsistent sessions.
    raw = os.environ.get("WEB_CONCURRENCY", "").strip()
    if raw not in {"", "1"}:
        raise RuntimeError(
            f"WEB_CONCURRENCY={raw!r} is not supported: planner state and per-scope locks "
            "are held in a single process. Run one worker (WEB_CONCURRENCY=1) and use "
            "GUNICORN_THREADS for request concurrency."
        )


ADMIN_SESSION_KEY = "planner_admin_ok"

SNAPSHOT_STORE = SnapshotStore(BASE_DIR / "instance" / "planner_snapshots.json")

TRACKING_ENABLED = _env_bool("PLANNER_TRACKING_ENABLED", True)

MAX_IMPORT_BYTES = _env_positive_int("PLANNER_MAX_IMPORT_BYTES", 1024 * 1024)

MAX_TABS_PER_VISITOR = _env_positive_int("PLANNER_MAX_TABS_PER_VISITOR", 64)

ADMIN_PAGE_SIZE = min(_env_positive_int("PLANNER_ADMIN_PAGE_SIZE", 100), 500)

REQUEST_RATE_LIMIT = _env_positive_int("PLANNER_RATE_LIMIT_PER_MINUTE", 600)

ADMIN_LOGIN_RATE_LIMIT = _env_positive_int("PLANNER_ADMIN_LOGIN_ATTEMPTS_PER_MINUTE", 10)

RATE_LIMIT_MAX_IDENTITIES = _env_positive_int("PLANNER_RATE_LIMIT_MAX_IDENTITIES", 20000)

EFFICIENCY_SETTING_BOUNDS = {
    "prefill_bw_eff": (0.50, 1.00),
    "prefill_comp_eff": (0.50, 1.00),
    "prefill_overhead": (0.00, 0.25),
    "prefill_paged_oh": (0.00, 0.25),
    "prefill_ar_overlap": (0.00, 0.80),
    "decode_bw_eff": (0.50, 1.00),
    "decode_comp_eff": (0.50, 1.00),
    "decode_overhead": (0.00, 0.25),
    "decode_paged_oh": (0.00, 0.25),
    "decode_ar_overlap": (0.00, 0.80),
    "kv_slack": (0.00, 0.10),
    "moe_imbalance": (1.00, 2.00),
    "pd_interference": (0.00, 1.00),
    "spec_acceptance": (0.00, 0.99),
}

INTEGER_SETTING_BOUNDS = {"decode_sched_budget": (2048, 65536)}
