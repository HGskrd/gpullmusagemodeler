import unittest
import uuid
from unittest.mock import patch

from app_factory import create_test_app

import planner_service
import web.api as web_api
import web.econ as econ_routes
import web.session_store as session_store
from scenarios import serialize_scenario
from state import PlannerState, add_gpu, set_gpu_cost, set_project_name


class PlannerRevisionTests(unittest.TestCase):
    def test_top_level_and_nested_mutations_advance_revision(self):
        state = PlannerState()
        self.assertEqual(state.revision, 0)

        state.mu = 0.85
        after_setting = state.revision
        self.assertGreater(after_setting, 0)

        add_gpu(state, "H100", 8)
        after_add = state.revision
        self.assertGreater(after_add, after_setting)

        set_gpu_cost(state, state.gpus[0].uid, 3.25)
        self.assertGreater(state.revision, after_add)

    def test_child_only_project_edit_advances_revision(self):
        state, _ = planner_service.create_default_scenario()
        before = state.revision
        set_project_name(state, state.projects[0].uid, "Revised workload")
        self.assertGreater(state.revision, before)

    def test_revision_is_not_part_of_exported_scenario(self):
        state = PlannerState()
        state.touch()
        payload = serialize_scenario(state, None)
        self.assertNotIn("revision", payload["panel_a"])


class DerivedResponseCacheTests(unittest.TestCase):
    def setUp(self):
        self.app = create_test_app()
        self.client = self.app.test_client()
        self.tab_id = str(uuid.uuid4())
        self.headers = {"X-Tab-ID": self.tab_id}
        self.client.get("/", headers=self.headers)

    def _state(self):
        scope_id = next(key for key in session_store._states if key.endswith(f":{self.tab_id}"))
        return session_store.get_state(scope_id)

    def test_chart_json_is_cached_revalidated_and_invalidated_by_mutation(self):
        original = web_api._build_chart_payload
        with patch.object(web_api, "_build_chart_payload", wraps=original) as build:
            first = self.client.get("/api/chart-data", headers=self.headers)
            second = self.client.get("/api/chart-data", headers=self.headers)
            conditional = self.client.get(
                "/api/chart-data",
                headers={**self.headers, "If-None-Match": first.headers["ETag"]},
            )

            self.assertEqual(build.call_count, 1)
            self.assertEqual(first.data, second.data)
            self.assertEqual(first.headers["ETag"], second.headers["ETag"])
            self.assertEqual(conditional.status_code, 304)
            self.assertEqual(conditional.data, b"")

            state = self._state()
            previous_revision = state.revision
            state.mode = "processingpareto"
            changed = self.client.get("/api/chart-data", headers=self.headers)

            self.assertGreater(state.revision, previous_revision)
            self.assertEqual(build.call_count, 2)
            self.assertNotEqual(changed.data, first.data)
            self.assertNotEqual(changed.headers["ETag"], first.headers["ETag"])

    def test_chart_etag_revalidates_the_gzip_representation(self):
        headers = {**self.headers, "Accept-Encoding": "gzip"}
        first = self.client.get("/api/chart-data", headers=headers)
        conditional = self.client.get(
            "/api/chart-data",
            headers={**headers, "If-None-Match": first.headers["ETag"]},
        )

        self.assertEqual(first.headers["Content-Encoding"], "gzip")
        self.assertEqual(conditional.status_code, 304)

    def test_swap_recommendations_are_cached_by_scenario_fingerprint(self):
        original = econ_routes.compute_swap_recs
        with patch.object(econ_routes, "compute_swap_recs", wraps=original) as compute:
            first = self.client.get("/econ/swaps?panel=A&view=table", headers=self.headers)
            second = self.client.get("/econ/swaps?panel=A&view=table", headers=self.headers)

            self.assertEqual(first.status_code, 200)
            self.assertEqual(first.data, second.data)
            self.assertEqual(compute.call_count, 1)

            self._state().projection_demand_level = 0.72
            third = self.client.get("/econ/swaps?panel=A&view=table", headers=self.headers)

            self.assertEqual(third.status_code, 200)
            self.assertEqual(compute.call_count, 2)


if __name__ == "__main__":
    unittest.main()
