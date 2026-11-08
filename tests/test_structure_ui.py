"""Execute asynchronous UI paths without network or browser timing dependence."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node executes the UI code")


def run_js(source):
    subprocess.run(["node", "-e", source], capture_output=True, text=True, check=True, timeout=10)


def test_latest_chart_request_owns_rendering_and_mutations_invalidate_pending_requests():
    source = (ROOT / "static/app.js").read_text()
    refresh = source[
        source.index("let chartRequestEpoch =") : source.index(
            "function updateChartAccessibleSummary"
        )
    ]
    run_js(
        """
const assert = require('node:assert/strict');
const pending = [], rendered = [], handlers = {};
const document = {getElementById: () => ({}), addEventListener: (n,f) => handlers[n] = f};
const withTabUrl = x => x, requestHeaders = () => ({}), renderChart = x => rendered.push(x);
const fetch = () => new Promise(resolve => pending.push(resolve));
const flush = () => new Promise(resolve => setImmediate(resolve));
"""
        + refresh
        + """
(async () => {
  refreshChart(); refreshChart();
  pending[1]({ok:true, json:async () => 'new'}); await flush();
  pending[0]({ok:true, json:async () => 'old'}); await flush();
  assert.deepEqual(rendered, ['new']);
  refreshChart(); handlers['htmx:beforeRequest']();
  pending[2]({ok:true, json:async () => 'before mutation'}); await flush();
  assert.deepEqual(rendered, ['new']);
})().catch(error => {console.error(error); process.exitCode = 1;});
"""
    )


def test_failed_admin_load_reveals_error_and_removes_loading_indicator():
    source = (ROOT / "static/admin/admin.js").read_text()
    start = source[
        source.index("  function start(render)") : source.index("  // ---------- tooltip")
    ]
    run_js(
        """
const assert = require('node:assert/strict');
const content = {hidden:true}, loading = {remove() {this.removed = true;}};
const document = {getElementById: id => ({content, loading})[id]};
const load = () => Promise.reject(new Error('offline'));
let rendered = false;
"""
        + start
        + """
start(() => {rendered = true;});
setImmediate(() => {
  assert.equal(content.hidden, false);
  assert.equal(loading.removed, true);
  assert.equal(rendered, false);
});
"""
    )
