import unittest
from unittest.mock import patch

from calc import avg_dist
from data import DIST_PRESETS, INPUT_BUCKETS, OUTPUT_BUCKETS
from econ_variants import econ_payload
from engine.economics import (
    DemandFates,
    ModelUtilization,
    ProjectionResult,
    ProjectOutcome,
    allocate_capacity,
    build_supply,
    calculate_environmental_impact,
    classify_demand,
    compute_revenue_projection,
    latent_activation_share,
    price_outcomes,
    summarize_projection,
)
from state import GpuPool, ModelAssignment, PlannerState, Project, _sync_aggregate_distribution


class RevenueProjectionTests(unittest.TestCase):
    def test_typed_projection_stages_round_trip_to_legacy_payload(self):
        state = PlannerState(
            gpus=[GpuPool(1, "H100", 2, cost_per_gpu_hour=1.0)],
            models=[ModelAssignment(2, "q27", 1, 1, 1, 1, "bf16")],
            projects=[Project(3, "Coding", 0.45, 20_000_000, 4.0)],
        )

        supplied = build_supply(state)
        classified = classify_demand(supplied)
        allocated = allocate_capacity(classified)
        priced = price_outcomes(allocated)
        impacted = calculate_environmental_impact(priced)
        result = summarize_projection(impacted, include_recommendations=False)

        self.assertIsInstance(result, ProjectionResult)
        self.assertIsInstance(result.fates, DemandFates)
        self.assertTrue(all(isinstance(row, ProjectOutcome) for row in result.projects))
        self.assertTrue(all(isinstance(row, ModelUtilization) for row in result.models))
        self.assertEqual(
            result.to_dict(),
            compute_revenue_projection(state, include_recommendations=False),
        )

    def test_main_economics_payload_defers_expansion_recommendations(self):
        state = PlannerState()

        with patch(
            "econ_variants.compute_revenue_projection",
            wraps=compute_revenue_projection,
        ) as projection:
            payload = econ_payload(state)

        projection.assert_called_once_with(state, include_recommendations=False)
        self.assertEqual(payload["p"]["recommendations"], [])

    def test_prefix_reuse_is_prompt_token_weighted_across_use_cases(self):
        low = Project(
            1,
            "Short input",
            0.1,
            1_000_000,
            10.0,
            prefix_hit_rate=0.0,
            in_pre="Classify",
            out_pre="Long doc",
        )
        high = Project(
            2,
            "Long input",
            0.1,
            1_000_000,
            10.0,
            prefix_hit_rate=1.0,
            in_pre="Long doc",
            out_pre="Classify",
        )
        state = PlannerState(projects=[low, high])

        _sync_aggregate_distribution(state)

        def prompt_weight(project):
            input_len = avg_dist(DIST_PRESETS[project.in_pre]["in"], INPUT_BUCKETS)
            output_len = avg_dist(DIST_PRESETS[project.out_pre]["out"], OUTPUT_BUCKETS)
            return project.tokens_day * input_len / (input_len + output_len)

        expected = prompt_weight(high) / (prompt_weight(low) + prompt_weight(high))
        self.assertAlmostEqual(state.prefix_hit_rate, expected)

    def test_cloud_price_uses_each_use_cases_prefix_reuse(self):
        state = PlannerState(
            projects=[
                Project(
                    1, "No reuse", 0.1, 1_000_000, 100.0, min_success_rate=0.5, prefix_hit_rate=0.0
                ),
                Project(
                    2,
                    "High reuse",
                    0.1,
                    1_000_000,
                    100.0,
                    min_success_rate=0.5,
                    prefix_hit_rate=0.8,
                ),
            ]
        )

        projection = compute_revenue_projection(state, include_recommendations=False)
        rows = {row["name"]: row for row in projection["projects"]}

        self.assertLess(rows["High reuse"]["cloud_pm"], rows["No reuse"]["cloud_pm"])
        self.assertEqual(rows["No reuse"]["prefix_hit_rate"], 0.0)
        self.assertEqual(rows["High reuse"]["prefix_hit_rate"], 0.8)
        self.assertAlmostEqual(
            projection["value_cloud_day"],
            projection["value_spilled_day"] + projection["value_leaked_day"],
        )

    def test_projection_exposes_distinct_coverage_metrics(self):
        state = PlannerState(
            gpus=[GpuPool(1, "H100", 2, cost_per_gpu_hour=1.0)],
            models=[ModelAssignment(2, "q27", 1, 1, 1, 1, "bf16")],
            projects=[
                Project(
                    3,
                    "Coding",
                    difficulty=0.55,
                    tokens_day=20_000_000,
                    wtp_per_m=4.0,
                    min_success_rate=0.80,
                    quality_floor=0.60,
                )
            ],
        )

        projection = compute_revenue_projection(state)

        self.assertIn("token_coverage", projection)
        self.assertIn("value_capture_rate", projection)
        self.assertIn("revenue_multiple", projection)
        self.assertEqual(projection["coverage"], projection["revenue_multiple"])
        self.assertGreaterEqual(projection["token_coverage"], 0.0)
        self.assertLessEqual(projection["token_coverage"], 1.0)

    def test_zero_capacity_assignment_is_not_runnable(self):
        state = PlannerState(
            gpus=[GpuPool(1, "H100", 1, cost_per_gpu_hour=1.0)],
            models=[ModelAssignment(2, "q122", 1, 1, 1, 1, "bf16")],
            projects=[
                Project(
                    3,
                    "Long context",
                    difficulty=0.70,
                    tokens_day=10_000_000,
                    wtp_per_m=8.0,
                    requires=frozenset({"ctx_128k"}),
                    min_success_rate=0.80,
                    quality_floor=0.70,
                )
            ],
        )

        projection = compute_revenue_projection(state)

        self.assertEqual(projection["models"][0]["daily_tokens_cap"], 0.0)
        self.assertFalse(projection["models"][0]["runnable"])
        self.assertEqual(projection["models"][0]["status"], "NOT RUNNABLE")

    def test_latent_activation_is_smooth_around_unlock_price(self):
        self.assertAlmostEqual(latent_activation_share(1.0, 1.0), 0.5)
        self.assertGreater(latent_activation_share(0.5, 1.0), 0.95)
        self.assertLess(latent_activation_share(2.0, 1.0), 0.15)

    def test_latent_demand_is_reported_separately_from_baseline(self):
        state = PlannerState(
            gpus=[GpuPool(1, "H100", 2, cost_per_gpu_hour=1.0)],
            models=[ModelAssignment(2, "q27", 1, 1, 1, 1, "bf16")],
            projects=[
                Project(
                    3,
                    "Summaries",
                    difficulty=0.25,
                    tokens_day=10_000_000,
                    wtp_per_m=4.0,
                    min_success_rate=0.80,
                    quality_floor=0.60,
                    latent_jobs_day=10_000_000,
                    unlock_price_per_m=10.0,
                )
            ],
        )

        projection = compute_revenue_projection(state)

        self.assertEqual(projection["baseline_tokens_day"], 10_000_000)
        self.assertGreater(projection["latent_active_tokens_day"], 0.0)
        self.assertAlmostEqual(
            projection["fates"]["total_tokens"],
            projection["baseline_tokens_day"] + projection["latent_active_tokens_day"],
        )


if __name__ == "__main__":
    unittest.main()
