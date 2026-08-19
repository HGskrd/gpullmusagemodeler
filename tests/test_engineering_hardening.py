import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from collections import deque
from pathlib import Path
from unittest.mock import patch

import app as app_module
from tracking import SnapshotStore


class EngineeringHardeningTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = SnapshotStore(Path(self.tempdir.name) / "snapshots.sqlite3")
        self.original_store = app_module.SNAPSHOT_STORE
        self.original_tracking = app_module.TRACKING_ENABLED
        app_module.SNAPSHOT_STORE = self.store
        app_module.TRACKING_ENABLED = False
        app_module.app.config.update(TESTING=True)
        self.client = app_module.app.test_client()
        self.visitor_id = str(uuid.uuid4())
        self.tab_id = str(uuid.uuid4())
        self.client.set_cookie(app_module.VISITOR_COOKIE, self.visitor_id)
        self.headers = {"X-Tab-ID": self.tab_id}

    def tearDown(self):
        app_module.SNAPSHOT_STORE = self.original_store
        app_module.TRACKING_ENABLED = self.original_tracking
        self.tempdir.cleanup()

    def test_arbitrary_setting_key_is_rejected_without_corrupting_state(self):
        response = self.client.post(
            "/settings/int",
            data={"key": "projects", "value": "1", "tab_id": self.tab_id},
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get("/session/sync", headers=self.headers).status_code, 200)

    def test_invalid_numeric_and_distribution_inputs_return_400(self):
        response = self.client.post(
            "/settings/non-kv",
            data={"value": "NaN", "tab_id": self.tab_id},
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 400)
        response = self.client.post(
            "/dist/slide",
            data={"kind": "in", "index": "0", "value": "-1", "tab_id": self.tab_id},
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 400)

    def test_full_scenario_export_reset_and_import(self):
        exported = self.client.get("/scenario/export", headers=self.headers)
        self.assertEqual(exported.status_code, 200)
        payload = exported.get_json()
        self.assertEqual(payload["type"], "gpullm-scenario")
        self.assertEqual(len(payload["panel_a"]["models"]), 3)
        self.assertEqual([gpu["gpu_type"] for gpu in payload["panel_a"]["gpus"]], ["H100"])
        self.assertEqual(
            [model["model_key"] for model in payload["panel_a"]["models"]],
            ["ms32", "g26", "q35"],
        )
        self.assertIsNotNone(payload["panel_b"])
        self.assertEqual(
            [model["model_key"] for model in payload["panel_b"]["models"]],
            ["g31", "q27", "nem3no"],
        )
        self.assertEqual(len(payload["panel_a"]["projects"]), 19)
        self.assertEqual(len(payload["panel_b"]["projects"]), 19)
        self.assertAlmostEqual(payload["panel_a"]["projects"][0]["prefix_hit_rate"], 0.10)
        self.assertAlmostEqual(payload["panel_b"]["projects"][0]["prefix_hit_rate"], 0.10)

        reset = self.client.post(
            "/session/reset", data={"tab_id": self.tab_id}, headers=self.headers
        )
        self.assertEqual(reset.status_code, 200)
        blank = self.client.get("/scenario/export", headers=self.headers).get_json()
        self.assertEqual(blank["panel_a"]["models"], [])
        self.assertIsNone(blank["panel_b"])

        restored = self.client.post(
            "/scenario/import",
            data={"json": json.dumps(payload), "tab_id": self.tab_id},
            headers=self.headers,
        )
        self.assertEqual(restored.status_code, 200)
        roundtrip = self.client.get("/scenario/export", headers=self.headers).get_json()
        self.assertEqual(len(roundtrip["panel_a"]["gpus"]), len(payload["panel_a"]["gpus"]))
        self.assertEqual(len(roundtrip["panel_a"]["models"]), len(payload["panel_a"]["models"]))
        self.assertEqual(
            roundtrip["panel_a"]["models"][0]["tp"], payload["panel_a"]["models"][0]["tp"]
        )

    def test_delete_session_data_removes_state_snapshots_and_cookie(self):
        app_module.TRACKING_ENABLED = True
        self.assertEqual(self.client.get("/session/sync", headers=self.headers).status_code, 200)
        self.assertEqual(self.store.count_snapshots(self.visitor_id), 1)

        response = self.client.delete("/session/data")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["deleted"])
        self.assertEqual(self.store.count_snapshots(self.visitor_id), 0)
        self.assertIn("planner_vid=;", response.headers.get("Set-Cookie", ""))

    def test_healthz_uses_transactional_store(self):
        app_module.TRACKING_ENABLED = True
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_per_visitor_active_tab_cap_is_enforced(self):
        with patch.object(app_module, "MAX_TABS_PER_VISITOR", 1):
            self.assertEqual(
                self.client.get("/session/sync", headers=self.headers).status_code, 200
            )
            other = {"X-Tab-ID": str(uuid.uuid4())}
            self.assertEqual(self.client.get("/session/sync", headers=other).status_code, 429)

    def test_admin_is_disabled_without_explicit_secret(self):
        self.assertNotEqual(app_module.app.secret_key, "vllm-planner-dev-key")
        with patch.dict("os.environ", {"PLANNER_ADMIN_PASSWORD": "strong"}):
            with patch.object(app_module, "_configured_secret", ""):
                response = self.client.get("/admin")
        self.assertEqual(response.status_code, 503)
        self.assertIn(b"PLANNER_SECRET_KEY", response.data)

    def test_admin_rejects_invalid_pagination(self):
        with patch.dict("os.environ", {"PLANNER_ADMIN_PASSWORD": "strong"}):
            with patch.object(app_module, "_configured_secret", "strong-secret"):
                with self.client.session_transaction() as session:
                    session[app_module.ADMIN_SESSION_KEY] = True
                response = self.client.get("/admin?page=not-a-number")
        self.assertEqual(response.status_code, 400)

    def test_multi_worker_startup_is_refused(self):
        repo_root = Path(app_module.__file__).resolve().parent
        env = dict(os.environ, WEB_CONCURRENCY="2")
        refused = subprocess.run(
            [sys.executable, "-c", "import app"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("RuntimeError", refused.stderr)
        self.assertIn("WEB_CONCURRENCY", refused.stderr)

        allowed = subprocess.run(
            [sys.executable, "-c", "import app"],
            cwd=repo_root,
            env=dict(os.environ, WEB_CONCURRENCY="1"),
            capture_output=True,
            text=True,
        )
        self.assertEqual(allowed.returncode, 0, allowed.stderr)

    def test_rate_limiter_sees_forwarded_ip_when_behind_proxy(self):
        repo_root = Path(app_module.__file__).resolve().parent
        script = (
            "import app; "
            "app.app.test_client().post('/admin/login', data={'password':'x'}, "
            "headers={'X-Forwarded-For':'203.0.113.7'}); "
            "print(sorted(str(k) for k in app._rate_windows))"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo_root,
            env=dict(os.environ, PLANNER_BEHIND_PROXY="true"),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("203.0.113.7", result.stdout)
        self.assertNotIn("127.0.0.1", result.stdout)

    def test_forwarded_header_is_ignored_without_proxy_flag(self):
        repo_root = Path(app_module.__file__).resolve().parent
        script = (
            "import app; "
            "app.app.test_client().post('/admin/login', data={'password':'x'}, "
            "headers={'X-Forwarded-For':'203.0.113.7'}); "
            "print(sorted(str(k) for k in app._rate_windows))"
        )
        env = dict(os.environ)
        env.pop("PLANNER_BEHIND_PROXY", None)
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("127.0.0.1", result.stdout)
        self.assertNotIn("203.0.113.7", result.stdout)

    def test_expired_rate_windows_are_swept(self):
        original = dict(app_module._rate_windows)
        original_sweep = app_module._rate_last_sweep
        try:
            now = time.monotonic()
            app_module._rate_windows.clear()
            for i in range(50):
                # Last seen well outside the 60s window.
                app_module._rate_windows[("mutation", f"198.51.100.{i}")] = deque(
                    [now - app_module.RATE_WINDOW_SECONDS - 5.0]
                )
            app_module._rate_windows[("mutation", "203.0.113.1")] = deque([now])

            app_module._sweep_rate_windows(now)

            self.assertEqual(list(app_module._rate_windows), [("mutation", "203.0.113.1")])
        finally:
            app_module._rate_windows.clear()
            app_module._rate_windows.update(original)
            app_module._rate_last_sweep = original_sweep

    def test_active_rate_windows_are_capped(self):
        original = dict(app_module._rate_windows)
        original_cap = app_module.RATE_LIMIT_MAX_IDENTITIES
        original_sweep = app_module._rate_last_sweep
        try:
            now = time.monotonic()
            app_module._rate_windows.clear()
            app_module.RATE_LIMIT_MAX_IDENTITIES = 10
            # All still active, so expiry alone cannot get under the cap.
            for i in range(40):
                app_module._rate_windows[("mutation", f"198.51.100.{i}")] = deque(
                    [now - (40 - i) * 0.1]
                )

            app_module._sweep_rate_windows(now)

            self.assertEqual(len(app_module._rate_windows), 10)
            # The survivors are the most recently active.
            self.assertIn(("mutation", "198.51.100.39"), app_module._rate_windows)
            self.assertNotIn(("mutation", "198.51.100.0"), app_module._rate_windows)
        finally:
            app_module.RATE_LIMIT_MAX_IDENTITIES = original_cap
            app_module._rate_windows.clear()
            app_module._rate_windows.update(original)
            app_module._rate_last_sweep = original_sweep

    def test_security_headers_are_set_on_every_response(self):
        for path in ("/", "/static/app.js", "/healthz"):
            with self.subTest(path=path):
                response = self.client.get(path, headers={"X-Tab-ID": str(uuid.uuid4())})
                self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
                self.assertEqual(response.headers["X-Frame-Options"], "DENY")
                self.assertEqual(response.headers["Referrer-Policy"], "same-origin")
                csp = response.headers["Content-Security-Policy"]
                self.assertIn("default-src 'self'", csp)
                self.assertIn("frame-ancestors 'none'", csp)
                self.assertIn("object-src 'none'", csp)
                self.assertIn("base-uri 'self'", csp)
                self.assertIn("form-action 'self'", csp)

    def test_default_port_matches_the_deployment_config(self):
        # app.py's __main__ default drifted from .env.example/compose/Dockerfile
        # once already; pin them together.
        repo_root = Path(app_module.__file__).resolve().parent
        source = (repo_root / "app.py").read_text(encoding="utf-8")
        self.assertIn('or "5014"', source)
        self.assertIn("PORT=5014", (repo_root / ".env.example").read_text(encoding="utf-8"))

    def test_visitor_cookie_is_samesite_lax_and_httponly(self):
        self.assertEqual(app_module.app.config["SESSION_COOKIE_SAMESITE"], "Lax")
        self.assertTrue(app_module.app.config["SESSION_COOKIE_HTTPONLY"])
        response = app_module.app.test_client().get("/")
        cookie = response.headers.get("Set-Cookie", "")
        self.assertIn(app_module.VISITOR_COOKIE, cookie)
        self.assertIn("SameSite=Lax", cookie)
        self.assertIn("HttpOnly", cookie)

    def test_fmt_num_uses_b_unit_above_a_billion(self):
        # Supply capacities in the tens of billions must not render as "31795.8M".
        self.assertEqual(app_module.fmt_num(31_795_800_000), "31.8B")
        self.assertEqual(app_module.fmt_num(1_048_576), "1.0M")
        self.assertEqual(app_module.fmt_num(131_072), "131.1k")
        self.assertEqual(app_module.fmt_num(999), "999")

    def test_prefix_panel_reflects_workload_distribution_basis(self):
        # The model card probes prefill capacity at max(task_il, avg in-dist);
        # the prefix-reuse panel must quote the same basis, not task_il alone.
        from calc import avg_dist, effective_prefill_length
        from data import INPUT_BUCKETS
        from planner_service import create_default_state

        state = create_default_state()
        basis = max(state.task_il, avg_dist(state.in_dist, INPUT_BUCKETS))
        effective = effective_prefill_length(basis, state.prefix_hit_rate)
        response = self.client.get("/", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn(f"Effective prefill input: {effective:,} / {basis:,} tok", html)
        self.assertIn("Portfolio prefix-token reuse", html)
        self.assertNotIn('hx-post="/settings/prefix-hit"', html)

    def test_cloud_only_use_case_shows_money_paid_to_cloud(self):
        from state import PlannerState, Project, replace_scope_states

        state = PlannerState(
            projects=[
                Project(
                    1,
                    "Cloud-only workload",
                    0.1,
                    1_000_000,
                    100.0,
                    min_success_rate=0.5,
                    prefix_hit_rate=0.25,
                )
            ]
        )
        replace_scope_states(f"{self.visitor_id}:{self.tab_id}", state, None)

        response = self.client.get("/", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Cloud spend lost / day", html)
        # The KPI tile and the per-use-case table both surface the cloud outflow.
        from engine.economics import compute_revenue_projection

        projection = compute_revenue_projection(state)
        self.assertGreater(projection["value_cloud_day"], 0)
        self.assertIn(app_module.fmt_money(projection["value_cloud_day"]), html)
        self.assertIn("To cloud", html)

    def test_unserved_sublabel_is_neutral_without_demand(self):
        self.client.post("/session/reset", data={"tab_id": self.tab_id}, headers=self.headers)
        response = self.client.get("/", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("No demand routed yet", html)
        self.assertNotIn("no compatible option within the price ceiling", html)


if __name__ == "__main__":
    unittest.main()
