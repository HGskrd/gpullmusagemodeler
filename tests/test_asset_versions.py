"""Browser cache invalidation must follow CSS/JS content, not file timestamps."""

import os
import re
from hashlib import sha256
from urllib.parse import parse_qs, urlsplit

from app_factory import create_test_app
from flask import Flask, url_for

from web.assets import register_asset_versions


def asset_url(root):
    app = Flask(__name__, static_folder=str(root))
    register_asset_versions(app)
    with app.test_request_context():
        return url_for("static", filename="style.css")


def test_content_changes_invalidate_asset_urls_even_with_identical_timestamps(tmp_path):
    asset = tmp_path / "style.css"
    asset.write_text("old styles")
    old_url = asset_url(tmp_path)
    original_stat = asset.stat()
    asset.write_text("new styles")
    os.utime(asset, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    new_url = asset_url(tmp_path)
    assert new_url != old_url
    assert asset_url(tmp_path) == new_url


def test_rendered_planner_uses_versions_of_the_served_assets():
    client = create_test_app().test_client()
    html = client.get("/").text
    urls = re.findall(r'(?:href|src)="(/static/[^"?]+\.(?:css|js)\?v=[0-9a-f]+)"', html)
    assert len(urls) == 6
    for url in urls:
        response = client.get(url)
        assert response.status_code == 200
        expected = sha256(response.data).hexdigest()[:16]
        assert parse_qs(urlsplit(url).query)["v"] == [expected]
