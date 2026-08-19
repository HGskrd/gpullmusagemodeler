import json
import unittest
import uuid
from unittest.mock import patch

from app_factory import create_test_app
from characterization_support import (
    FIXTURE_DIR,
    canonical_json,
    chart_outputs,
    chart_state,
    projection_outputs,
)
from flask import render_template

import app as app_module
from econ_variants import _chart_payload, econ_payload
from use_case_evidence import (
    USE_CASE_RESEARCH_CAPTURED_AT,
    USE_CASE_SOURCES,
    enrich_use_case_details,
)
from viewmodels import get_model_info, get_model_infos


class GoldenOutputTests(unittest.TestCase):
    def assert_matches_fixture(self, name: str, actual) -> None:
        expected = (FIXTURE_DIR / name).read_text(encoding="utf-8")
        self.assertEqual(canonical_json(actual), expected)

    def test_revenue_projection_goldens(self):
        self.assert_matches_fixture("projections.json", projection_outputs())

    def test_all_chart_builder_goldens(self):
        outputs = chart_outputs()
        self.assertEqual(
            set(outputs),
            {
                "chart_decode",
                "chart_pareto",
                "chart_user_pareto",
                "chart_aggregate",
                "chart_data_processing",
                "chart_embedding_throughput",
                "chart_embedding_quality",
                "chart_processing_pareto",
                "chart_user_experience",
                "chart_realtime_capacity",
                "chart_asr_quality",
            },
        )
        self.assert_matches_fixture("charts.json", outputs)


class PartialRenderSmokeTests(unittest.TestCase):
    def setUp(self):
        self.app = create_test_app()
        self.client = self.app.test_client()

    def test_every_partial_renders_through_a_flask_request(self):
        environment = self.app.jinja_env
        expected = {name for name in environment.list_templates() if name.startswith("partials/")}
        seen: set[str] = set()
        loader = environment.loader
        original_get_source = loader.get_source

        def tracking_get_source(jinja_environment, template):
            seen.add(template)
            return original_get_source(jinja_environment, template)

        environment.cache.clear()
        tab_id = f"characterization-{uuid.uuid4().hex}"
        headers = {"X-Tab-ID": tab_id}
        original_healthz = self.app.view_functions["api.healthz"]
        with patch.object(loader, "get_source", side_effect=tracking_get_source):
            try:
                responses = [
                    self.client.get(route, headers=headers)
                    for route in (
                        "/",
                        "/use-cases",
                        "/picker/gpu?panel=A",
                        "/picker/model?panel=A&kind=text",
                        "/picker/project?panel=A",
                        "/econ/supply",
                        "/econ/swaps?panel=A&view=table",
                    )
                ]
                responses.append(self.client.post("/compare/duplicate", headers=headers))

                state = chart_state()
                state.mode = "data"

                def render_task_partial():
                    return render_template(
                        "partials/task.html",
                        state=state,
                        panel="A",
                        **app_module._template_context(),
                    )

                self.app.view_functions["api.healthz"] = render_task_partial
                environment.cache.clear()
                responses.append(self.client.get("/healthz"))

                def render_distribution_partial():
                    return render_template(
                        "partials/distribution.html",
                        dist=state.in_dist,
                        dist_label="Input",
                        pre=state.in_pre,
                        panel="A",
                        kind="in",
                        buckets=app_module.INPUT_BUCKETS,
                        **app_module._template_context(),
                    )

                self.app.view_functions["api.healthz"] = render_distribution_partial
                environment.cache.clear()
                responses.append(self.client.get("/healthz"))
            finally:
                self.app.view_functions["api.healthz"] = original_healthz

        self.assertTrue(all(response.status_code == 200 for response in responses))
        self.assertTrue(expected <= seen, expected - seen)


class DirectPresentationModuleTests(unittest.TestCase):
    def test_viewmodels_cover_text_embedding_and_realtime_assignments(self):
        state = chart_state()
        infos = get_model_infos(state)
        self.assertEqual(len(infos), len(state.models))
        self.assertEqual(
            [get_model_info(state, assignment)["model"].key for assignment in state.models],
            [info["model"].key for info in infos],
        )
        self.assertTrue(any(info["embedding"] is not None for info in infos))
        self.assertTrue(any(info["realtime"] is not None for info in infos))
        self.assertTrue(any(info["decode_max_slots"] > 0 for info in infos))

    def test_economics_payload_is_json_safe_and_matches_chart_builder(self):
        state = chart_state()
        payload = econ_payload(state)
        charts = json.loads(payload["charts_json"])
        rebuilt = _chart_payload(payload["p"], payload["model_tokens"])
        self.assertEqual(charts, rebuilt)
        self.assertIs(payload["f"], payload["p"]["fates"])
        self.assertNotIn("<", payload["charts_json"])

    def test_use_case_evidence_enrichment_is_non_destructive_and_sourced(self):
        original = {"coding": {"custom": "kept"}, "private": {"summary": "Local"}}
        enriched = enrich_use_case_details(original)
        self.assertEqual(original, {"coding": {"custom": "kept"}, "private": {"summary": "Local"}})
        self.assertEqual(enriched["coding"]["custom"], "kept")
        self.assertIn(enriched["coding"]["confidence"], {"low", "medium", "high"})
        self.assertEqual(enriched["private"]["summary"], "Local")
        self.assertRegex(USE_CASE_RESEARCH_CAPTURED_AT, r"^\d{4}-\d{2}-\d{2}$")
        for details in enriched.values():
            for source_id in details.get("source_ids", ()):
                self.assertIn(source_id, USE_CASE_SOURCES)


if __name__ == "__main__":
    unittest.main()
