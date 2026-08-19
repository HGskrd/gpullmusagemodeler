import unittest
import uuid

import app as app_module
from calc import (
    _portfolio_domain_quality,
    _swap_candidate_shortlist,
    compute_revenue_projection,
)
from data import (
    MODEL_DOMAIN_QUALITY_ANCHORS,
    MODELS,
    QUALITY_DOMAINS,
    effective_quality,
    model_domain_anchor,
    model_profile_quality,
    normalize_quality_domain,
    normalize_quality_weights,
    swebench_pro_to_coding_quality,
)
from placement import _model_serves_project
from planner_service import create_default_state, deserialize_scenario
from scenarios import serialize_scenario
from state import (
    GpuPool,
    ModelAssignment,
    PlannerState,
    Project,
    _normalize_use_case_def,
)


class DomainQualityCatalogTests(unittest.TestCase):
    def test_quality_weights_normalize_and_fall_back_to_legacy_domain(self):
        self.assertEqual(normalize_quality_weights(None, "coding"), {"coding": 1.0})
        self.assertEqual(normalize_quality_weights({}, "reasoning"), {"reasoning": 1.0})
        self.assertEqual(
            normalize_quality_weights({"coding": 7, "reasoning": 2, "invalid": 1}, "general"),
            {"coding": 0.7, "reasoning": 0.2, "general": 0.1},
        )
        self.assertEqual(
            normalize_quality_weights({"coding": -1, "reasoning": 0}, "general"),
            {"general": 1.0},
        )

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
        self.assertEqual(
            effective_quality(unanchored, "long_context"), effective_quality(unanchored)
        )

    def test_representative_anchors_change_only_the_requested_domain(self):
        model = MODELS["k25"]

        self.assertAlmostEqual(model_domain_anchor(model, "coding").raw_score, 76.8)
        self.assertNotEqual(effective_quality(model, "coding"), effective_quality(model))
        self.assertEqual(effective_quality(model, "general"), effective_quality(model))

    def test_kimi_k3_release_anchors_cover_reported_core_domains(self):
        model = MODELS["kimi-k3"]

        self.assertAlmostEqual(model_domain_anchor(model, "coding").raw_score, 67.5)
        self.assertAlmostEqual(model_domain_anchor(model, "reasoning").raw_score, 93.5)
        self.assertAlmostEqual(model_domain_anchor(model, "long_context").raw_score, 74.7)
        self.assertAlmostEqual(model_domain_anchor(model, "vision").raw_score, 81.6)
        self.assertIsNone(model_domain_anchor(model, "multilingual"))

    def test_weighted_profile_uses_each_axis_and_sparse_global_fallback(self):
        model = MODELS["laguna-s-2-1"]
        weights = {"coding": 0.70, "reasoning": 0.20, "long_context": 0.10}
        score = model_profile_quality(model, weights, "coding")

        self.assertGreater(score, 0.72)
        self.assertLess(score, effective_quality(model, "coding"))
        self.assertIsNone(model_domain_anchor(model, "reasoning"))

    def test_swe_pro_crosswalk_keeps_laguna_close_to_glm52(self):
        laguna = swebench_pro_to_coding_quality(59.4)
        glm52 = swebench_pro_to_coding_quality(62.1)

        self.assertAlmostEqual(laguna, 0.7935277031511743)
        self.assertAlmostEqual(glm52, 0.8109831062928664)
        self.assertLess(laguna, glm52)
        self.assertLess(glm52 - laguna, 0.02)

    def test_use_case_definition_normalizes_domain_with_builtin_migration(self):
        coding = _normalize_use_case_def({"key": "coding", "name": "Coding"})
        self.assertEqual(coding["quality_domain"], "coding")
        self.assertAlmostEqual(coding["quality_weights"]["coding"], 0.70)
        self.assertAlmostEqual(coding["quality_weights"]["reasoning"], 0.20)
        self.assertAlmostEqual(coding["quality_weights"]["long_context"], 0.10)
        self.assertEqual(
            _normalize_use_case_def(
                {"key": "custom-domain", "name": "Custom", "quality_domain": "invalid"}
            )["quality_domain"],
            "general",
        )


class DomainQualityPlannerTests(unittest.TestCase):
    def test_laguna_clears_repository_agent_vector_without_global_proxy_cliff(self):
        project = Project(
            99,
            "Repository coding agent",
            difficulty=0.55,
            tokens_day=1.2e9,
            wtp_per_m=4.0,
            requires=frozenset({"tools", "ctx_128k"}),
            min_success_rate=0.85,
            quality_floor=0.70,
            quality_domain="coding",
            quality_weights={"coding": 0.70, "reasoning": 0.20, "long_context": 0.10},
        )

        self.assertTrue(_model_serves_project(MODELS["laguna-s-2-1"], project))

    def test_equal_value_routing_protects_harder_coding_work_before_generic_chat(self):
        state = create_default_state()
        state.gpus = [GpuPool(1, "H100", 6, cost_per_gpu_hour=1.32)]
        state.models = [ModelAssignment(2, "laguna-s-2-1", 1, 6, 2, 1, "bf16", pp=3)]
        state.projects = [
            project
            for project in state.projects
            if project.kind_key in {"chatbot", "coding", "meeting_notes"}
        ]

        projection = compute_revenue_projection(state, include_recommendations=False)
        rows = {row["project"].kind_key: row for row in projection["projects"]}

        self.assertGreater(rows["coding"]["served_pct"], 0.0)
        self.assertLess(rows["chatbot"]["served_pct"], 1e-9)
        self.assertEqual(
            rows["coding"]["quality_mix_label"],
            "Coding 70% · Reasoning 20% · Long context 10%",
        )
        self.assertEqual(
            [
                component["domain"]
                for component in rows["coding"]["per_model_served"][0]["quality_components"]
            ],
            ["coding", "reasoning", "long_context"],
        )

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
        self.assertAlmostEqual(
            served["effective_quality"], effective_quality(MODELS["q27"], "coding")
        )
        self.assertNotEqual(served["effective_quality"], effective_quality(MODELS["q27"]))

    def test_portfolio_quality_and_shortlist_follow_active_domain(self):
        state = self._state("coding")
        coding_quality, mix, anchored = _portfolio_domain_quality(MODELS["q27"], state.projects)

        self.assertEqual(mix, "Coding 100%")
        self.assertGreater(anchored, 0.99)
        self.assertAlmostEqual(coding_quality, effective_quality(MODELS["q27"], "coding"))

        shortlist = _swap_candidate_shortlist(MODELS["ds3"], state, "llm", 3)
        self.assertTrue(
            any(
                model_domain_anchor(candidate, "coding") is not None
                for _, candidate, _ in shortlist
            )
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
        self.assertTrue(
            any(abs(score - strongest_anchored_quality) < 1e-9 for _, _, score in shortlist)
        )

    def test_scenario_round_trip_preserves_domain_and_legacy_defaults_general(self):
        state = self._state("reasoning")
        restored, _ = deserialize_scenario(serialize_scenario(state, None))
        self.assertEqual(restored.projects[0].quality_domain, "reasoning")

        legacy_payload = serialize_scenario(self._state(), None)
        legacy_payload["panel_a"]["projects"][0].pop("quality_domain", None)
        legacy_payload["panel_a"]["projects"][0].pop("quality_weights", None)
        legacy_payload["panel_a"]["projects"][0].get("definition", {}).pop("quality_weights", None)
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
