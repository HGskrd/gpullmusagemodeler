import unittest
import uuid

import app as app_module
from calc import (
    _portfolio_domain_quality,
    _swap_candidate_shortlist,
    compute_revenue_projection,
)
from data import (
    MODELS,
    MODEL_DOMAIN_QUALITY_ANCHORS,
    QUALITY_DOMAINS,
    effective_quality,
    model_domain_anchor,
    normalize_quality_domain,
)
from scenarios import deserialize_scenario, serialize_scenario
from state import (
    GpuPool,
    ModelAssignment,
    PlannerState,
    Project,
    _normalize_use_case_def,
)


class DomainQualityCatalogTests(unittest.TestCase):
    def test_domain_taxonomy_and_sparse_fallback_are_well_formed(self):
        self.assertEqual(
            set(QUALITY_DOMAINS),
            {"general", "coding", "reasoning", "long_context", "multilingual", "vision"},
        )
        self.assertEqual(normalize_quality_domain("NOT-A-DOMAIN"), "general")

        for model_key, domains in MODEL_DOMAIN_QUALITY_ANCHORS.items():
            self.assertIn(model_key, MODELS)
            for domain, anchor in domains.items():
                with self.subTest(model=model_key, domain=domain):
                    self.assertIn(domain, QUALITY_DOMAINS)
                    self.assertNotEqual(domain, "general")
                    self.assertGreaterEqual(anchor.quality, 0.0)
                    self.assertLessEqual(anchor.quality, 1.0)
                    self.assertGreater(anchor.raw_score, 0.0)
                    self.assertTrue(anchor.benchmark)
                    self.assertTrue(anchor.source.startswith("http"))
                    self.assertGreater(anchor.confidence, 0.0)
                    self.assertLessEqual(anchor.confidence, 1.0)

        unanchored = MODELS["l8"]
        self.assertIsNone(model_domain_anchor(unanchored, "long_context"))
        self.assertEqual(effective_quality(unanchored, "long_context"), effective_quality(unanchored))

    def test_representative_anchors_change_only_the_requested_domain(self):
        model = MODELS["k25"]

        self.assertAlmostEqual(model_domain_anchor(model, "coding").raw_score, 76.8)
        self.assertNotEqual(effective_quality(model, "coding"), effective_quality(model))
        self.assertEqual(effective_quality(model, "general"), effective_quality(model))

    def test_use_case_definition_normalizes_domain_with_builtin_migration(self):
        self.assertEqual(
            _normalize_use_case_def({"key": "coding", "name": "Coding"})["quality_domain"],
            "coding",
        )
        self.assertEqual(
            _normalize_use_case_def(
                {"key": "custom-domain", "name": "Custom", "quality_domain": "invalid"}
            )["quality_domain"],
            "general",
        )


class DomainQualityPlannerTests(unittest.TestCase):
    def _state(self, domain="coding"):
        return PlannerState(
            gpus=[GpuPool(1, "H100", 2, cost_per_gpu_hour=0.01)],
            models=[ModelAssignment(2, "q27", 1, 1, 1, 1, "bf16")],
            projects=[
                Project(
                    3,
                    "Domain workload",
                    difficulty=0.30,
                    tokens_day=100_000,
                    wtp_per_m=100.0,
                    min_success_rate=0.50,
                    quality_floor=0.0,
                    quality_domain=domain,
                )
            ],
        )

    def test_projection_uses_and_exposes_project_domain_quality(self):
        state = self._state("coding")

        projection = compute_revenue_projection(state, include_recommendations=False)
        row = projection["projects"][0]
        served = row["per_model_served"][0]

        self.assertEqual(row["quality_domain"], "coding")
        self.assertEqual(row["quality_domain_label"], "Coding")
        self.assertEqual(served["quality_anchor"], "SWE-bench Verified")
        self.assertAlmostEqual(served["effective_quality"], effective_quality(MODELS["q27"], "coding"))
        self.assertNotEqual(served["effective_quality"], effective_quality(MODELS["q27"]))

    def test_portfolio_quality_and_shortlist_follow_active_domain(self):
        state = self._state("coding")
        coding_quality, mix, anchored = _portfolio_domain_quality(MODELS["q27"], state.projects)

        self.assertEqual(mix, "Coding 100%")
        self.assertGreater(anchored, 0.99)
        self.assertAlmostEqual(coding_quality, effective_quality(MODELS["q27"], "coding"))

        shortlist = _swap_candidate_shortlist(MODELS["ds3"], state, "llm", 3)
        self.assertTrue(
            any(model_domain_anchor(candidate, "coding") is not None for _, candidate, _ in shortlist)
        )
        strongest_anchored_quality = max(
            effective_quality(model, "coding")
            for model in MODELS.values()
            if (
                not model.hidden
                and model.embedding_profile is None
                and model.realtime_profile is None
                and model_domain_anchor(model, "coding") is not None
            )
        )
        self.assertTrue(any(abs(score - strongest_anchored_quality) < 1e-9 for _, _, score in shortlist))

    def test_scenario_round_trip_preserves_domain_and_legacy_defaults_general(self):
        state = self._state("reasoning")
        restored, _ = deserialize_scenario(serialize_scenario(state, None))
        self.assertEqual(restored.projects[0].quality_domain, "reasoning")

        legacy_payload = serialize_scenario(self._state(), None)
        legacy_payload["panel_a"]["projects"][0].pop("quality_domain", None)
        restored_legacy, _ = deserialize_scenario(legacy_payload)
        self.assertEqual(restored_legacy.projects[0].quality_domain, "general")

    def test_swap_card_labels_portfolio_fit_and_domain_mix(self):
        rec = {
            "current_name": "Current",
            "candidate_name": "Candidate",
            "gpu_count": 2,
            "gpu_name": "H100",
            "current_quality": 0.60,
            "candidate_quality": 0.75,
            "quality_mix": "Coding 100%",
            "margin_gain_day": 10.0,
            "cloud_reduced_day": 2.0,
            "destroyed_reduced_day": 3.0,
        }
        with app_module.app.test_request_context("/econ/swaps"):
            html = app_module.render_template(
                "partials/econ/swaps.html",
                swap_recs=[rec],
                gpu_recs=[],
                view="cards",
                panel="A",
            )

        self.assertIn("portfolio fit", html)
        self.assertIn("Coding 100%", html)

    def test_economics_routes_render_domain_aware_swap_output(self):
        original_tracking = app_module.TRACKING_ENABLED
        app_module.TRACKING_ENABLED = False
        app_module.app.config.update(TESTING=True)
        client = app_module.app.test_client()
        client.set_cookie(app_module.VISITOR_COOKIE, str(uuid.uuid4()))
        headers = {"X-Tab-ID": str(uuid.uuid4())}
        try:
            for path in ("/econ/", "/econ/flow", "/econ/dashboard", "/econ/brief", "/econ/supply"):
                with self.subTest(path=path):
                    self.assertEqual(client.get(path, headers=headers).status_code, 200)
            swaps = client.get("/econ/swaps?panel=A&view=cards", headers=headers)
            self.assertEqual(swaps.status_code, 200)
            self.assertIn("portfolio fit", swaps.get_data(as_text=True))
        finally:
            app_module.TRACKING_ENABLED = original_tracking


if __name__ == "__main__":
    unittest.main()
