import json
import tempfile
import unittest
import uuid
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
            "/settings/int", data={"key": "projects", "value": "1", "tab_id": self.tab_id},
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get("/session/sync", headers=self.headers).status_code, 200)

    def test_invalid_numeric_and_distribution_inputs_return_400(self):
        response = self.client.post(
            "/settings/non-kv", data={"value": "NaN", "tab_id": self.tab_id}, headers=self.headers,
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

        reset = self.client.post("/session/reset", data={"tab_id": self.tab_id}, headers=self.headers)
        self.assertEqual(reset.status_code, 200)
        blank = self.client.get("/scenario/export", headers=self.headers).get_json()
        self.assertEqual(blank["panel_a"]["models"], [])

        restored = self.client.post(
            "/scenario/import", data={"json": json.dumps(payload), "tab_id": self.tab_id}, headers=self.headers,
        )
        self.assertEqual(restored.status_code, 200)
        roundtrip = self.client.get("/scenario/export", headers=self.headers).get_json()
        self.assertEqual(len(roundtrip["panel_a"]["gpus"]), len(payload["panel_a"]["gpus"]))
        self.assertEqual(len(roundtrip["panel_a"]["models"]), len(payload["panel_a"]["models"]))
        self.assertEqual(roundtrip["panel_a"]["models"][0]["tp"], payload["panel_a"]["models"][0]["tp"])

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
            self.assertEqual(self.client.get("/session/sync", headers=self.headers).status_code, 200)
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


if __name__ == "__main__":
    unittest.main()
