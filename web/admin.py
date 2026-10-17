"""Administrative routes: visitor insight pages, the snapshot browser, login."""

import hmac
import os
from collections import defaultdict
from functools import wraps

from flask import (
    Blueprint,
    current_app,
    jsonify,
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
from web.analytics import build_admin_payload, mark_internal_if_admin
from web.config import (
    ADMIN_PAGE_SIZE,
    ADMIN_SESSION_KEY,
)
from web.helpers import (
    _bounded_int,
)

admin_bp = Blueprint("admin", __name__)

# (endpoint, template, nav label, page title, subtitle)
INSIGHT_PAGES = (
    (
        "admin",
        "overview",
        "Overview",
        "Visitors",
        "Who uses the calculator, and what they plan with it",
    ),
    (
        "audience",
        "audience",
        "Audience & geo",
        "Audience & geography",
        "Where people are, and how they found the calculator",
    ),
    (
        "personas",
        "personas",
        "Personas",
        "Personas",
        "Distinct kinds of people, derived from what they do",
    ),
    ("people", "people", "People", "People", "One profile per visitor, each visit read as a story"),
    (
        "journeys",
        "journeys",
        "Journeys & friction",
        "Journeys & friction",
        "Where people get stuck, and what they do about it",
    ),
    (
        "demand",
        "demand",
        "Demand & gaps",
        "Demand & gaps",
        "What people want to run, and what the catalog is missing",
    ),
)


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


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        configuration_error = _admin_configuration_error()
        if configuration_error:
            return configuration_error, 503
        if not _is_admin_authenticated():
            if request.path.endswith(".json"):
                return jsonify({"error": "Admin login required."}), 401
            return render_template("admin_login.html", error=None)
        return view(*args, **kwargs)

    return wrapped


def _render_insight(endpoint: str):
    _, template, _, title, subtitle = next(p for p in INSIGHT_PAGES if p[0] == endpoint)
    return render_template(
        f"admin/{template}.html",
        pages=INSIGHT_PAGES,
        page=endpoint,
        title=title,
        subtitle=subtitle,
        tracking_enabled=current_app.config.get("TRACKING_ENABLED", False),
    )


@admin_bp.route("/admin", methods=["GET"])
@admin_required
def admin():
    return _render_insight("admin")


@admin_bp.route("/admin/audience", methods=["GET"])
@admin_required
def audience():
    return _render_insight("audience")


@admin_bp.route("/admin/personas", methods=["GET"])
@admin_required
def personas():
    return _render_insight("personas")


@admin_bp.route("/admin/people", methods=["GET"])
@admin_required
def people():
    return _render_insight("people")


@admin_bp.route("/admin/journeys", methods=["GET"])
@admin_required
def journeys():
    return _render_insight("journeys")


@admin_bp.route("/admin/demand", methods=["GET"])
@admin_required
def demand():
    return _render_insight("demand")


@admin_bp.route("/admin/data.json", methods=["GET"])
@admin_required
def data():
    days = _bounded_int(request.args.get("days", 90), name="days", lo=1, hi=366)
    store = current_app.extensions["snapshot_store"]
    response = jsonify(build_admin_payload(store, days=days))
    response.headers["Cache-Control"] = "no-store"
    return response


@admin_bp.route("/admin/snapshots", methods=["GET"])
@admin_required
def snapshots():
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
    # Exclude this browser from visitor analytics from now on.
    mark_internal_if_admin()
    return redirect(url_for("admin.admin"))


@admin_bp.route("/admin/logout", methods=["POST"])
def admin_logout():
    session.pop(ADMIN_SESSION_KEY, None)
    return redirect(url_for("admin.admin"))
