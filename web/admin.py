"""Administrative routes: the snapshot browser and its login."""

import hmac
import os
from collections import defaultdict

from flask import (
    Blueprint,
    current_app,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from data import (
    GPUS,
    MODELS,
)
from web.config import (
    ADMIN_PAGE_SIZE,
    ADMIN_SESSION_KEY,
)
from web.helpers import (
    _bounded_int,
)

admin_bp = Blueprint("admin", __name__)


def _admin_password() -> str | None:
    return os.environ.get("PLANNER_ADMIN_PASSWORD")


def _admin_configuration_error() -> str | None:
    if not _admin_password():
        return "Set PLANNER_ADMIN_PASSWORD to enable /admin."
    if not current_app.config.get("PLANNER_SECRET_CONFIGURED", False):
        return "Set PLANNER_SECRET_KEY to enable /admin securely."
    return None


def _is_admin_authenticated() -> bool:
    return bool(session.get(ADMIN_SESSION_KEY))


@admin_bp.route("/admin", methods=["GET"])
def admin():
    configuration_error = _admin_configuration_error()
    if configuration_error:
        return configuration_error, 503
    if not _is_admin_authenticated():
        return render_template("admin_login.html", error=None)
    page = _bounded_int(request.args.get("page", 1), name="page", lo=1, hi=1_000_000)
    page_size = min(
        _bounded_int(request.args.get("per_page", ADMIN_PAGE_SIZE), name="per_page", lo=1, hi=500),
        ADMIN_PAGE_SIZE,
    )
    store = current_app.extensions["snapshot_store"]
    total_snapshots = store.count_snapshots()
    all_snapshots = store.list_snapshots(limit=page_size, offset=(page - 1) * page_size)
    visitor_map = defaultdict(list)
    for s in all_snapshots:
        visitor_map[s["visitor_id"]].append(s)
    visitors = sorted(
        [{"visitor_id": vid, "snapshots": snaps} for vid, snaps in visitor_map.items()],
        key=lambda v: v["snapshots"][0]["last_seen"],
        reverse=True,
    )
    return render_template(
        "admin.html",
        visitors=visitors,
        total_snapshots=total_snapshots,
        page=page,
        page_size=page_size,
        GPUS=GPUS,
        MODELS=MODELS,
    )


@admin_bp.route("/admin/login", methods=["POST"])
def admin_login():
    password = _admin_password()
    configuration_error = _admin_configuration_error()
    if configuration_error:
        return configuration_error, 503
    if not hmac.compare_digest(request.form.get("password", ""), password or ""):
        return render_template("admin_login.html", error="Invalid password."), 401

    session[ADMIN_SESSION_KEY] = True
    return redirect(url_for("admin.admin"))


@admin_bp.route("/admin/logout", methods=["POST"])
def admin_logout():
    session.pop(ADMIN_SESSION_KEY, None)
    return redirect(url_for("admin.admin"))
