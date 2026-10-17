"""Keep HTML and its cached CSS/JavaScript on the same asset revision."""

from hashlib import sha256
from pathlib import Path

from flask import Flask


def register_asset_versions(application: Flask) -> None:
    """Fingerprint assets once per app startup, including across container builds."""
    root = Path(application.static_folder or "static")
    versions = {
        path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()[:16]
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".css", ".js"}
    }

    @application.url_defaults
    def asset_version(endpoint: str, values: dict) -> None:
        if endpoint == "static" and (version := versions.get(values.get("filename", ""))):
            values.setdefault("v", version)
