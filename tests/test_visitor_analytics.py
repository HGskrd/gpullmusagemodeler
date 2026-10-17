"""Visitor analytics: request context, event capture, storage, admin payload."""

import sqlite3
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from app_factory import create_test_app

from data import GPUS
from tracking import SnapshotStore
from web.analytics import build_admin_payload, channel_for, classify_user_agent, gpu_class
from web.config import ADMIN_SESSION_KEY
from web.middleware import VISITOR_COOKIE
from web.session_store import get_state

CHROME_MAC = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
)
IPHONE = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 19_0 like Mac OS X) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/19.0 Mobile/15E148 Safari/604.1"
)


class ClassificationTests(unittest.TestCase):
    def test_user_agents(self):
        self.assertEqual(classify_user_agent(CHROME_MAC)["device"], "desktop")
        self.assertEqual(classify_user_agent(CHROME_MAC)["os"], "macOS")
        self.assertEqual(classify_user_agent(IPHONE)["device"], "mobile")
        agent = classify_user_agent("Mozilla/5.0 (compatible; Claude-User/1.0)")
        self.assertEqual((agent["device"], agent["agent"]), ("agent", "Claude-User"))
        self.assertTrue(
            classify_user_agent("Googlebot/2.1 (+http://www.google.com/bot.html)")["crawler"]
        )
        self.assertTrue(classify_user_agent("")["crawler"])

    def test_channels(self):
        self.assertEqual(channel_for({"referrer_host": "www.google.com"}), "Search")
        self.assertEqual(channel_for({"referrer_host": "news.ycombinator.com"}), "Hacker News")
        self.assertEqual(channel_for({"referrer_host": "old.reddit.com"}), "Reddit")
        self.assertEqual(channel_for({"referrer_host": "chatgpt.com"}), "AI assistant")
        self.assertEqual(channel_for({"utm_source": "linkedin"}), "LinkedIn")
        self.assertEqual(channel_for({}), "Direct")
        self.assertEqual(channel_for({"referrer_host": "example.org"}), "Other sites")
        self.assertEqual(channel_for({"device": "agent"}), "Agent / API")

    def test_every_gpu_has_a_class(self):
        allowed = {"consumer", "unified", "workstation", "datacenter"}
        for key in GPUS:
            with self.subTest(gpu=key):
                self.assertIn(gpu_class(key), allowed)
        self.assertEqual(gpu_class("RTX5090"), "consumer")
        self.assertEqual(gpu_class("MAC_STUDIO_M5_MAX"), "unified")
        self.assertEqual(gpu_class("DGX_SPARK"), "unified")
        self.assertEqual(gpu_class("RTXPRO6000_BSE"), "workstation")
        self.assertEqual(gpu_class("H100"), "datacenter")


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = SnapshotStore(Path(self.tempdir.name) / "snapshots.sqlite3")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_repeats_coalesce_within_window_only(self):
        kw = {"visitor_id": "v", "tab_id": "t", "event_type": "gpu_qty"}
        self.store.record_event(**kw, detail={"n": 1}, coalesce_key="gpu_qty:A:1", now_ms=1_000)
        self.store.record_event(**kw, detail={"n": 2}, coalesce_key="gpu_qty:A:1", now_ms=5_000)
        self.store.record_event(**kw, detail={"n": 3}, coalesce_key="gpu_qty:A:2", now_ms=6_000)
        self.store.record_event(**kw, detail={"n": 4}, coalesce_key="gpu_qty:A:2", now_ms=60_000)
        events = self.store.list_analytics(since_ms=0)["events"]
        self.assertEqual([e["detail"]["n"] for e in events], [2, 3, 4])

    def test_first_touch_context_is_kept(self):
        self.store.touch_visitor("v", {"country": "FR", "referrer_host": "a.com"}, now_ms=1_000)
        self.store.touch_visitor("v", {"country": "US", "referrer_host": "b.com"}, now_ms=9_000)
        (row,) = self.store.list_analytics(since_ms=0)["visitors"]
        self.assertEqual((row["country"], row["referrer_host"]), ("FR", "a.com"))
        self.assertEqual((row["first_seen"], row["last_seen"]), (1_000, 9_000))

    def test_delete_visitor_removes_analytics(self):
        self.store.touch_visitor("v", {"country": "FR"})
        self.store.record_event(visitor_id="v", tab_id="t", event_type="open", detail={})
        self.store.record_event(visitor_id="w", tab_id="t", event_type="open", detail={})
        self.store.delete_visitor("v")
        data = self.store.list_analytics(since_ms=0)
        self.assertEqual({e["visitor_id"] for e in data["events"]}, {"w"})
        self.assertEqual({row["visitor_id"] for row in data["visitors"]}, {"w"})

    def test_retention_prunes_old_events_and_visitors(self):
        self.store.retention_days = 1
        day = 86_400_000
        self.store.record_event(
            visitor_id="old", tab_id="t", event_type="open", detail={}, now_ms=0
        )
        self.store._event_writes = 0
        self.store.record_event(
            visitor_id="new", tab_id="t", event_type="open", detail={}, now_ms=3 * day
        )
        data = self.store.list_analytics(since_ms=0)
        self.assertEqual({e["visitor_id"] for e in data["events"]}, {"new"})
        self.assertEqual({row["visitor_id"] for row in data["visitors"]}, {"new"})


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db = Path(self.tempdir.name) / "snapshots.sqlite3"
        self.store = SnapshotStore(self.db)
        self.app = create_test_app(SNAPSHOT_STORE=self.store, TRACKING_ENABLED=True)
        self.client = self.app.test_client()
        self.visitor_id = str(uuid.uuid4())
        self.tab = str(uuid.uuid4())
        self.client.set_cookie(VISITOR_COOKIE, self.visitor_id)
        self.headers = {"X-Tab-ID": self.tab, "User-Agent": CHROME_MAC}

    def tearDown(self):
        self.tempdir.cleanup()

    def _post(self, path, **data):
        response = self.client.post(path, data={**data, "tab_id": self.tab}, headers=self.headers)
        return response

    def _state(self):
        return get_state(f"{self.visitor_id}:{self.tab}")

    def test_landing_context_and_planner_actions_reach_the_admin_payload(self):
        landing = self.client.get(
            "/?utm_source=linkedin&utm_campaign=launch",
            headers={
                "User-Agent": CHROME_MAC,
                "CF-IPCountry": "DE",
                "CF-IPCity": "Berlin",
                "CF-Timezone": "Europe/Berlin",
                "Referer": "https://www.linkedin.com/feed/",
                "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
            },
            environ_base={"REMOTE_ADDR": "203.0.113.77"},
        )
        self.assertEqual(landing.status_code, 200)
        self.client.get("/session/sync", headers=self.headers)
        self._post("/session/reset")
        self._post("/gpu/add", gpu_type="H100", count="8")
        self._post("/model/add", model_key="qwen38-27b")
        uid = self._state().models[0].uid
        self._post("/model/prec", uid=str(uid), prec="fp8")
        gpu_uid = self._state().gpus[0].uid
        self._post("/gpu/qty", uid=str(gpu_uid), count="4")
        self._post("/gpu/qty", uid=str(gpu_uid), count="2")

        payload = build_admin_payload(self.store, days=7)
        (visitor,) = payload["visitors"]
        self.assertEqual(visitor["country"], "DE")
        self.assertEqual(visitor["city"], "Berlin")
        self.assertEqual(visitor["tz"], "Europe/Berlin")
        self.assertEqual(visitor["lang"], "de-DE")
        self.assertEqual(visitor["channel"], "LinkedIn")
        self.assertEqual(visitor["device"], "desktop")
        self.assertIn("utm_source=linkedin", visitor["utm"])
        (session,) = visitor["sessions"]
        types = [e["type"] for e in session["events"]]
        self.assertEqual(types[:3], ["open", "session_reset", "gpu_add"])
        self.assertIn("model_add", types)
        self.assertIn("model_prec", types)
        self.assertIn("result", types)
        gpu_add = next(e for e in session["events"] if e["type"] == "gpu_add")
        self.assertEqual((gpu_add["gpu"], gpu_add["qty"]), ("H100", 8))
        model_add = next(e for e in session["events"] if e["type"] == "model_add")
        self.assertEqual(model_add["model"], "qwen38-27b")
        # Two rapid count edits on the same pool collapse into one event.
        self.assertEqual(types.count("gpu_qty"), 1)
        self.assertEqual(next(e for e in session["events"] if e["type"] == "gpu_qty")["qty"], 2)
        self.assertEqual(session["final"]["gpus"], [{"key": "H100", "qty": 2}])
        self.assertEqual(session["final"]["models"][0]["key"], "qwen38-27b")

        # Nothing identifying beyond the documented fields: no IP address.
        with sqlite3.connect(self.db) as connection:
            dump = "\n".join(connection.iterdump())
        self.assertNotIn("203.0.113.77", dump)

    def test_rejected_model_is_logged_as_friction(self):
        self.client.get("/session/sync", headers=self.headers)
        self._post("/session/reset")
        self._post("/gpu/add", gpu_type="T4", count="1")
        response = self._post("/model/add", model_key="kimi-k3")
        self.assertEqual(response.status_code, 400)
        payload = build_admin_payload(self.store, days=7)
        events = payload["visitors"][0]["sessions"][0]["events"]
        rejected = [e for e in events if e["type"] == "model_rejected"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["model"], "kimi-k3")

    def test_default_scenario_is_not_counted_as_a_plan(self):
        # A visitor who only loads the page has the default scenario in their
        # "open" snapshot; that must not read as a plan they chose.
        self.client.get("/", headers={"User-Agent": CHROME_MAC})
        self.client.get("/session/sync", headers=self.headers)
        (visitor,) = build_admin_payload(self.store, days=7)["visitors"]
        self.assertEqual([e["type"] for e in visitor["sessions"][0]["events"]], ["open"])
        self.assertIsNone(visitor["sessions"][0]["final"])

    def test_crawlers_are_not_tracked(self):
        self.client.get(
            "/", headers={"User-Agent": "Googlebot/2.1 (+http://www.google.com/bot.html)"}
        )
        self.assertEqual(build_admin_payload(self.store, days=7)["visitors"], [])

    def test_economics_page_view_is_an_event(self):
        self.client.get("/econ/", headers={"User-Agent": CHROME_MAC})
        events = build_admin_payload(self.store, days=7)["visitors"][0]["sessions"][0]["events"]
        self.assertIn("econ_view", [e["type"] for e in events])

    def test_tracking_disabled_records_nothing(self):
        self.app.config["TRACKING_ENABLED"] = False
        self.client.get("/", headers={"User-Agent": CHROME_MAC, "CF-IPCountry": "FR"})
        self._post("/gpu/add", gpu_type="H100", count="8")
        self.assertEqual(self.store.analytics_status()["events"], 0)
        self.assertEqual(self.store.analytics_status()["visitors"], 0)


class AdminRouteTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = SnapshotStore(Path(self.tempdir.name) / "snapshots.sqlite3")
        self.app = create_test_app(SNAPSHOT_STORE=self.store, TRACKING_ENABLED=True)
        self.client = self.app.test_client()
        self.env = patch.dict("os.environ", {"PLANNER_ADMIN_PASSWORD": "strong"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tempdir.cleanup()

    def test_data_requires_login(self):
        self.assertEqual(self.client.get("/admin/data.json").status_code, 401)
        page = self.client.get("/admin/people")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b'action="/admin/login"', page.data)

    def test_pages_and_data_render_when_logged_in(self):
        with self.client.session_transaction() as session:
            session[ADMIN_SESSION_KEY] = True
        for path in (
            "/admin",
            "/admin/audience",
            "/admin/personas",
            "/admin/people",
            "/admin/journeys",
            "/admin/demand",
            "/admin/snapshots",
        ):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)
        response = self.client.get("/admin/data.json?days=30")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        body = response.get_json()
        self.assertIn("H100", body["catalog"]["gpus"])
        self.assertEqual(body["meta"]["days"], 30)
        self.assertEqual(self.client.get("/admin/data.json?days=0").status_code, 400)

    def test_login_marks_the_browser_internal(self):
        visitor_id = str(uuid.uuid4())
        self.client.set_cookie(VISITOR_COOKIE, visitor_id)
        self.client.post("/admin/login", data={"password": "strong"})
        (row,) = self.store.list_analytics(since_ms=0)["visitors"]
        self.assertEqual((row["visitor_id"], row["internal"]), (visitor_id, 1))


if __name__ == "__main__":
    unittest.main()
