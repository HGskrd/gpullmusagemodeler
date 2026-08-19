"""Shared isolated Flask application fixture."""

from __future__ import annotations

from typing import Any

from app import create_app


def create_test_app(**config: Any):
    defaults = {
        "TESTING": True,
        "TRACKING_ENABLED": False,
        "SECRET_KEY": "test-secret",
    }
    defaults.update(config)
    return create_app(defaults)
