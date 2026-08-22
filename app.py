"""Flask application for the GPU/LLM Usage Modeler."""

import json
import os
import secrets

from flask import (
    Flask,
    current_app,
    jsonify,
    request,
)
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

from presentation.formatting import fmt_money, fmt_num, log2int
from web.admin import admin_bp
from web.api import api_bp
from web.cache import FingerprintCache, ScopedRevisionCache
from web.config import (
    ADMIN_LOGIN_RATE_LIMIT,
    MAX_TABS_PER_VISITOR,
    RATE_LIMIT_MAX_IDENTITIES,
    REQUEST_RATE_LIMIT,
    SNAPSHOT_STORE,
    TRACKING_ENABLED,
    _enforce_single_worker,
    _env_bool,
    _env_positive_int,
)
from web.econ import econ_bp
from web.middleware import (
    register_middleware,
)
from web.planner import planner_bp
from web.scenarios import scenarios_bp
from web.use_cases import use_cases_bp

_configured_secret = os.environ.get("PLANNER_SECRET_KEY", "").strip()


# Fetchable resources are local. 'unsafe-inline' stays for scripts (one inline
# block in base.html) and styles (~250 inline style attributes across the
# templates); HTMX expression evaluation is disabled in base.html so its
# controls remain compatible with the no-unsafe-eval policy.


# The after_request compressor deliberately skips direct_passthrough responses,
# which is every file served by Flask's static handler — so the vendor bundles
# (~1.1 MB of JS/CSS) went out uncompressed. Compress those once and keep the
# result keyed by (mtime, size) so a redeploy or edit invalidates it.


def _handle_validation_error(error: ValueError):
    if isinstance(error, json.JSONDecodeError):
        message = f"Invalid JSON: {error.msg}"
    else:
        message = str(error)
    return jsonify({"error": message}), 400


def _handle_internal_error(error: Exception):
    if isinstance(error, HTTPException):
        return error
    current_app.logger.exception("Unhandled error in endpoint %s", request.endpoint)
    return jsonify({"error": "Unexpected server error."}), 500


def create_app(config: dict | None = None) -> Flask:
    """Build a configured application without constructing it during import."""
    _enforce_single_worker()
    application = Flask(__name__)
    configured_secret = os.environ.get("PLANNER_SECRET_KEY", "").strip()
    application.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=_env_bool("PLANNER_SECURE_COOKIES", False),
        MAX_CONTENT_LENGTH=_env_positive_int("PLANNER_MAX_REQUEST_BYTES", 2 * 1024 * 1024),
        PLANNER_SECRET_CONFIGURED=bool(configured_secret),
        TRACKING_ENABLED=TRACKING_ENABLED,
        MAX_TABS_PER_VISITOR=MAX_TABS_PER_VISITOR,
        REQUEST_RATE_LIMIT=REQUEST_RATE_LIMIT,
        ADMIN_LOGIN_RATE_LIMIT=ADMIN_LOGIN_RATE_LIMIT,
        RATE_LIMIT_MAX_IDENTITIES=RATE_LIMIT_MAX_IDENTITIES,
        CHART_CACHE_MAX_SCOPES=_env_positive_int("PLANNER_CHART_CACHE_MAX_SCOPES", 5000),
        SWAP_CACHE_MAX_ENTRIES=_env_positive_int("PLANNER_SWAP_CACHE_MAX_ENTRIES", 256),
    )
    if config:
        application.config.update(config)
    explicit_secret = str(application.config.get("SECRET_KEY") or configured_secret).strip()
    application.secret_key = explicit_secret or secrets.token_urlsafe(48)
    application.extensions["snapshot_store"] = application.config.get(
        "SNAPSHOT_STORE", SNAPSHOT_STORE
    )
    application.extensions["chart_json_cache"] = ScopedRevisionCache(
        max_scopes=application.config["CHART_CACHE_MAX_SCOPES"]
    )
    application.extensions["swap_recommendation_cache"] = FingerprintCache(
        max_entries=application.config["SWAP_CACHE_MAX_ENTRIES"]
    )
    if explicit_secret:
        application.config["PLANNER_SECRET_CONFIGURED"] = True
    else:
        application.logger.warning(
            "PLANNER_SECRET_KEY is unset; generated an ephemeral key for this process. "
            "Sessions and admin logins will not survive a restart. Set it before deploying."
        )

    if _env_bool("PLANNER_BEHIND_PROXY", False):
        application.wsgi_app = ProxyFix(application.wsgi_app, x_for=1, x_proto=1)

    for blueprint in (
        planner_bp,
        use_cases_bp,
        scenarios_bp,
        api_bp,
        admin_bp,
        econ_bp,
    ):
        application.register_blueprint(blueprint)

    register_middleware(application)
    application.register_error_handler(ValueError, _handle_validation_error)
    application.register_error_handler(Exception, _handle_internal_error)
    application.add_template_filter(fmt_num)
    application.add_template_filter(fmt_money)
    application.add_template_filter(log2int)
    return application


if __name__ == "__main__":
    host = os.environ.get("HOST") or os.environ.get("FLASK_RUN_HOST") or "0.0.0.0"
    # Keep this aligned with .env.example, compose.yaml, the Dockerfile, and README.
    port = int(os.environ.get("PORT") or os.environ.get("FLASK_RUN_PORT") or "5014")
    debug = _env_bool("FLASK_DEBUG", _env_bool("DEBUG", False))
    create_app().run(host=host, port=port, debug=debug)
