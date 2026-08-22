"""Request identity, locking, rate limiting, security, and compression."""

from __future__ import annotations

import gzip
import re
import threading
import time
import uuid
from collections import deque
from pathlib import Path

from flask import Flask, current_app, g, jsonify, request
from werkzeug.security import safe_join

from web.session_store import (
    allow_visitor_scope,
    get_scope_lock,
)

VISITOR_COOKIE = "planner_vid"
TAB_PARAM = "tab_id"
_TAB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
RATE_WINDOW_SECONDS = 60.0
RATE_SWEEP_INTERVAL_SECONDS = 60.0
_rate_lock = threading.Lock()
_rate_windows: dict[tuple[str, str], deque[float]] = {}
_rate_last_sweep = 0.0

CONTENT_SECURITY_POLICY = "; ".join(
    (
        "default-src 'self'",
        "script-src 'self' 'unsafe-inline'",
        "style-src 'self' 'unsafe-inline'",
        "img-src 'self' data:",
        "font-src 'self'",
        "connect-src 'self'",
        "form-action 'self'",
        "base-uri 'self'",
        "frame-ancestors 'none'",
        "object-src 'none'",
    )
)

_STATIC_COMPRESSIBLE_SUFFIXES = {".css", ".js", ".json", ".map", ".svg", ".txt"}
_GZIP_ETAG_MARKER = '-gzip"'
_STATIC_GZIP_MIN_BYTES = 1024
_static_gzip_cache: dict[str, tuple[float, int, bytes]] = {}
_static_gzip_lock = threading.Lock()

_UNSCOPED_ENDPOINTS = {
    "static",
    "planner.explainer",
    "admin.admin",
    "admin.admin_login",
    "admin.admin_logout",
    "api.healthz",
    "scenarios.session_data_delete",
}


def _new_id() -> str:
    return str(uuid.uuid4())


def _visitor_id() -> str:
    visitor_id = getattr(g, "visitor_id", None)
    if visitor_id:
        return visitor_id
    visitor_id = request.cookies.get(VISITOR_COOKIE, "")
    try:
        valid = str(uuid.UUID(visitor_id)) == visitor_id.lower()
    except (ValueError, AttributeError):
        valid = False
    if not valid:
        visitor_id = _new_id()
    g.visitor_id = visitor_id
    return visitor_id


def _tab_id(optional: bool = False) -> str | None:
    tab_id = request.headers.get("X-Tab-ID") or request.form.get(TAB_PARAM)
    if tab_id and _TAB_ID_RE.fullmatch(tab_id):
        return tab_id
    return None if optional else "default"


def _scope_id() -> str:
    cached = getattr(g, "planner_scope_id", None)
    if cached:
        return cached
    scope_id = f"{_visitor_id()}:{_tab_id()}"
    g.planner_scope_id = scope_id
    return scope_id


def _sweep_rate_windows(now: float) -> None:
    """Drop expired identities and cap active rate-window memory."""
    global _rate_last_sweep
    _rate_last_sweep = now
    stale = [
        key
        for key, window in _rate_windows.items()
        if not window or now - window[-1] >= RATE_WINDOW_SECONDS
    ]
    for key in stale:
        del _rate_windows[key]

    cap = current_app.config["RATE_LIMIT_MAX_IDENTITIES"]
    excess = len(_rate_windows) - cap
    if excess > 0:
        oldest = sorted(_rate_windows, key=lambda key: _rate_windows[key][-1])[:excess]
        for key in oldest:
            del _rate_windows[key]


def _rate_limit(bucket: str, limit: int) -> bool:
    identity = request.remote_addr or "unknown"
    now = time.monotonic()
    key = (bucket, identity)
    with _rate_lock:
        if now - _rate_last_sweep >= RATE_SWEEP_INTERVAL_SECONDS:
            _sweep_rate_windows(now)
        window = _rate_windows.get(key)
        if window is None:
            window = _rate_windows[key] = deque()
        while window and now - window[0] >= RATE_WINDOW_SECONDS:
            window.popleft()
        if len(window) >= limit:
            return False
        window.append(now)
        return True


def _protect_and_lock_request_scope():
    if request.endpoint == "admin.admin_login" and not _rate_limit(
        "admin-login", current_app.config["ADMIN_LOGIN_RATE_LIMIT"]
    ):
        return jsonify({"error": "Too many login attempts. Try again later."}), 429
    if request.method not in {"GET", "HEAD", "OPTIONS"} and not _rate_limit(
        "mutation", current_app.config["REQUEST_RATE_LIMIT"]
    ):
        return jsonify({"error": "Too many updates. Try again shortly."}), 429
    if request.endpoint and request.endpoint not in _UNSCOPED_ENDPOINTS:
        visitor_id = _visitor_id()
        visitor_lock = get_scope_lock(f"visitor:{visitor_id}")
        visitor_lock.acquire()
        scope_id = _scope_id()
        if not allow_visitor_scope(
            scope_id, visitor_id, current_app.config["MAX_TABS_PER_VISITOR"]
        ):
            visitor_lock.release()
            return jsonify({"error": "Too many active tabs for this planner session."}), 429
        scope_lock = get_scope_lock(scope_id)
        scope_lock.acquire()
        g.planner_scope_locks = (scope_lock, visitor_lock)
    return None


def _release_request_scope_lock(_error=None):
    for lock in getattr(g, "planner_scope_locks", ()):
        lock.release()


def _set_identity_cookie(response):
    if getattr(g, "suppress_identity_cookie", False):
        return response
    visitor_id = getattr(g, "visitor_id", None)
    if visitor_id and request.cookies.get(VISITOR_COOKIE) != visitor_id:
        response.set_cookie(
            VISITOR_COOKIE,
            visitor_id,
            max_age=60 * 60 * 24 * 365,
            httponly=True,
            samesite="Lax",
            secure=current_app.config["SESSION_COOKIE_SECURE"] or request.is_secure,
        )
    return response


def _set_security_headers(response):
    response.headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault("X-Frame-Options", "DENY")
    return response


def _compress_large_text_responses(response):
    if (
        request.method == "HEAD"
        or response.status_code < 200
        or response.status_code >= 300
        or response.direct_passthrough
        or response.headers.get("Content-Encoding")
        or "no-transform" in response.headers.get("Cache-Control", "")
    ):
        return response
    content_type = response.mimetype or ""
    if not (
        content_type.startswith("text/")
        or content_type in {"application/json", "application/javascript", "image/svg+xml"}
    ):
        return response
    response.vary.add("Accept-Encoding")
    if request.accept_encodings["gzip"] <= 0:
        return response
    body = response.get_data()
    if len(body) < 1024:
        return response
    response.set_data(gzip.compress(body, compresslevel=4))
    response.headers["Content-Encoding"] = "gzip"
    return response


def _static_gzip_payload(path: Path) -> bytes | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    if stat.st_size < _STATIC_GZIP_MIN_BYTES:
        return None
    key = str(path)
    with _static_gzip_lock:
        cached = _static_gzip_cache.get(key)
        if cached and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
            return cached[2]
    try:
        compressed = gzip.compress(path.read_bytes(), compresslevel=4)
    except OSError:
        return None
    with _static_gzip_lock:
        _static_gzip_cache[key] = (stat.st_mtime, stat.st_size, compressed)
    return compressed


def _mark_gzip_etag(response) -> None:
    etag = response.headers.get("ETag")
    if etag and etag.endswith('"') and not etag.endswith(_GZIP_ETAG_MARKER):
        response.headers["ETag"] = etag[:-1] + _GZIP_ETAG_MARKER


def _serve_static(filename: str):
    if_none_match = request.headers.get("If-None-Match", "")
    if _GZIP_ETAG_MARKER in if_none_match:
        request.environ["HTTP_IF_NONE_MATCH"] = if_none_match.replace(_GZIP_ETAG_MARKER, '"')
    response = current_app.send_static_file(filename)
    if (
        request.method == "HEAD"
        or Path(filename).suffix.lower() not in _STATIC_COMPRESSIBLE_SUFFIXES
    ):
        return response
    response.vary.add("Accept-Encoding")
    if request.accept_encodings["gzip"] <= 0:
        return response
    if response.status_code == 304:
        _mark_gzip_etag(response)
        return response
    if response.status_code != 200:
        return response
    full_path = safe_join(current_app.static_folder or "", filename)
    payload = _static_gzip_payload(Path(full_path)) if full_path else None
    if payload is None:
        return response
    response.direct_passthrough = False
    response.set_data(payload)
    response.headers["Content-Encoding"] = "gzip"
    _mark_gzip_etag(response)
    return response


def register_middleware(application: Flask) -> None:
    """Register middleware in the same effective order as the legacy app."""
    application.before_request(_protect_and_lock_request_scope)
    application.teardown_request(_release_request_scope_lock)
    application.after_request(_set_identity_cookie)
    application.after_request(_set_security_headers)
    application.after_request(_compress_large_text_responses)
    application.view_functions["static"] = _serve_static
