import unittest

from state import (
    AUTO_MODEL_STRATEGIES,
    DEFAULT_AUTO_MODEL_STRATEGY,
    GpuPool,
    PlannerState,
    Project,
    auto_select_models,
    normalize_auto_strategy,
)


def _strategy_fixture() -> PlannerState:
    return PlannerState(
        gpus=[GpuPool(1, "H100", 8, cost_per_gpu_hour=1.0)],
        projects=[
            Project(
                10,
                "Classification",
                difficulty=0.10,
                tokens_day=100_000_000,
                wtp_per_m=0.25,
                min_success_rate=0.80,
                quality_floor=0.35,
            ),
            Project(
                11,
                "Reasoning",
                difficulty=0.75,
                tokens_day=100_000_000,
                wtp_per_m=20.0,
                requires=frozenset({"tools", "reasoning"}),
                min_success_rate=0.85,
                quality_floor=0.75,
            ),
        ],
    )


class AutoSelectionTests(unittest.TestCase):
    def test_all_declared_auto_strategies_select_models(self):
        assignments = {}
        for strategy, _, _ in AUTO_MODEL_STRATEGIES:
            state = _strategy_fixture()
            auto_select_models(state, strategy)

            self.assertTrue(state.auto_mode)
            self.assertEqual(state.auto_strategy, strategy)
            self.assertTrue(state.models)
            assignments[strategy] = tuple((m.model_key, m.gpu_count, m.prec) for m in state.models)

        self.assertGreater(len(set(assignments.values())), 1)

    def test_lean_strategy_leaves_capacity_unassigned(self):
        balanced = _strategy_fixture()
        lean = _strategy_fixture()

        auto_select_models(balanced, "balanced")
        auto_select_models(lean, "lean")

        self.assertLess(
            sum(m.gpu_count for m in lean.models),
            sum(m.gpu_count for m in balanced.models),
        )

    def test_unknown_auto_strategy_falls_back_to_default(self):
        state = _strategy_fixture()

        auto_select_models(state, "unknown")

        self.assertEqual(normalize_auto_strategy("unknown"), DEFAULT_AUTO_MODEL_STRATEGY)
        self.assertEqual(state.auto_strategy, DEFAULT_AUTO_MODEL_STRATEGY)


if __name__ == "__main__":
    unittest.main()
