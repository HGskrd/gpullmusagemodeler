"""Every mutation path must advance PlannerState.revision.

Chart JSON, ETags and the swap-recommendation cache are all keyed on
`revision` (directly, or through `web.cache.combined_revision`). A mutation that
does not advance it serves a stale body under a matching ETag, which no other
test in the suite would notice.

Two layers guard that:

* a structural check, so a new `set_*` / `add_*` mutator added to `state.py`
  without `@bumps_revision` fails immediately; and
* a behavioural check that drives every real entry point — including the ones
  in `placement.py` and `planner_service.py`, which are *not* covered by the
  decorator and bump only as a side effect of what they happen to assign.
"""

from __future__ import annotations

import inspect
import unittest

import placement
import planner_service
import state as state_module
from web.cache import combined_revision

# Names with these prefixes mutate planner state and must carry @bumps_revision.
_MUTATOR_PREFIXES = ("add_", "remove_", "set_", "change_", "auto_")

# Registry and lookup helpers that share a prefix but own no PlannerState field.
_NON_MUTATORS = {
    "allow_visitor_scope",
    "configure_default_scenario_factory",
}


def _public_state_functions():
    for name, function in vars(state_module).items():
        if name.startswith("_") or not inspect.isfunction(function):
            continue
        if function.__module__ != "state":
            continue
        yield name, function


class BumpsRevisionCoverageTests(unittest.TestCase):
    def test_every_state_mutator_carries_the_decorator(self):
        missing = sorted(
            name
            for name, function in _public_state_functions()
            if name.startswith(_MUTATOR_PREFIXES)
            and name not in _NON_MUTATORS
            and not getattr(function, "__bumps_revision__", False)
        )
        self.assertEqual(
            missing,
            [],
            f"state.py mutators missing @bumps_revision; derived caches would go stale: {missing}",
        )

    def test_the_decorator_is_not_applied_to_read_only_helpers(self):
        misapplied = sorted(
            name
            for name, function in _public_state_functions()
            if getattr(function, "__bumps_revision__", False)
            and not name.startswith(_MUTATOR_PREFIXES)
        )
        self.assertEqual(misapplied, [])

    def test_the_decorator_bumps_even_when_the_body_changes_nothing(self):
        @state_module.bumps_revision
        def noop(_state):
            return None

        state = state_module.PlannerState()
        before = state.revision
        noop(state)
        self.assertGreater(state.revision, before)


def _mutations():
    """(label, callable(state)) for every public mutation entry point.

    Each callable is applied to a fresh default scenario, so arguments can be
    read off that scenario's first GPU pool, model and project.
    """
    S = state_module

    def gpu(state):
        return state.gpus[0].uid

    def model(state):
        return state.models[0].uid

    def project(state):
        return state.projects[0].uid

    def use_case(state):
        return state.use_case_defs[0]["key"]

    return [
        # --- GPU pools -------------------------------------------------
        ("state.add_gpu", lambda s: S.add_gpu(s, "H100", 8)),
        ("state.change_gpu_qty", lambda s: S.change_gpu_qty(s, gpu(s), 2)),
        ("state.set_gpu_cost", lambda s: S.set_gpu_cost(s, gpu(s), 4.5)),
        ("state.remove_gpu", lambda s: S.remove_gpu(s, gpu(s))),
        # --- model assignments -----------------------------------------
        (
            "state.add_model_assignment",
            lambda s: S.add_model_assignment(s, "q35", gpu(s), 1, "bf16"),
        ),
        ("state.set_model_prec", lambda s: S.set_model_prec(s, model(s), "fp8")),
        ("state.set_model_spec", lambda s: S.set_model_spec(s, model(s), "ngram", 4)),
        ("state.set_model_gpu_count", lambda s: S.set_model_gpu_count(s, model(s), 1)),
        ("state.set_model_gpu_pool", lambda s: S.set_model_gpu_pool(s, model(s), gpu(s))),
        ("state.set_model_strat", lambda s: S.set_model_strat(s, model(s), 2, 1, 1)),
        ("state.auto_exclude_model", lambda s: S.auto_exclude_model(s, model(s))),
        ("state.auto_reallow_model", lambda s: S.auto_reallow_model(s, s.models[0].model_key)),
        ("state.remove_model", lambda s: S.remove_model(s, model(s))),
        # --- projects ---------------------------------------------------
        ("state.add_project", lambda s: S.add_project(s)),
        ("state.set_project_name", lambda s: S.set_project_name(s, project(s), "Renamed")),
        ("state.set_project_kind", lambda s: S.set_project_kind(s, project(s), "coding")),
        (
            "state.set_project_field",
            lambda s: S.set_project_field(s, project(s), "difficulty", 0.8),
        ),
        ("state.set_project_scale_value", lambda s: S.set_project_scale_value(s, project(s), 42.0)),
        (
            "state.set_project_dist_preset",
            lambda s: S.set_project_dist_preset(s, project(s), "in", "Chat"),
        ),
        (
            "state.set_project_batch_eligible",
            lambda s: S.set_project_batch_eligible(s, project(s), True),
        ),
        (
            "state.set_project_capability",
            lambda s: S.set_project_capability(s, project(s), "tools", True),
        ),
        ("state.remove_project", lambda s: S.remove_project(s, project(s))),
        # --- use-case definitions --------------------------------------
        ("state.add_use_case_def", lambda s: S.add_use_case_def(s)),
        (
            "state.set_use_case_def_field",
            lambda s: S.set_use_case_def_field(s, use_case(s), "difficulty", 0.7),
        ),
        (
            "state.set_use_case_def_capability",
            lambda s: S.set_use_case_def_capability(s, use_case(s), "tools", True),
        ),
        ("state.remove_use_case_def", lambda s: S.remove_use_case_def(s, use_case(s))),
        # --- distributions and global settings -------------------------
        ("state.set_dist_preset", lambda s: S.set_dist_preset(s, "in", "Code")),
        ("state.set_dist_value", lambda s: S.set_dist_value(s, "in", 0, 12)),
        ("state.set_spec_acceptance", lambda s: S.set_spec_acceptance(s, 0.7)),
        ("state.set_projection_choice", lambda s: S.set_projection_choice(s, "day_shape", "flat")),
        ("state.set_projection_pct", lambda s: S.set_projection_pct(s, "demand_level", 0.4)),
        (
            "state.set_projection_toggle",
            lambda s: S.set_projection_toggle(s, "night_batching", True),
        ),
        # --- scalar attribute writes ------------------------------------
        ("PlannerState.mode", lambda s: setattr(s, "mode", "processingpareto")),
        ("PlannerState.mu", lambda s: setattr(s, "mu", 0.77)),
        ("PlannerState.corpo_cloud", lambda s: setattr(s, "corpo_cloud", "advocated")),
        # --- placement (outside the decorator) --------------------------
        ("placement.auto_select_models", lambda s: placement.auto_select_models(s, "coverage")),
        (
            "placement.retune_models",
            lambda s: (
                setattr(s.models[0], "gpu_count", s.models[0].gpu_count + 1),
                placement.retune_models(s, preserve_existing=False),
            ),
        ),
        # --- application services (outside the decorator) ---------------
        ("planner_service.add_model", lambda s: planner_service.add_model(s, "q35")),
        # A key the default scenario does not already hold: add_models skips
        # duplicates, and a skipped add is correctly a no-op.
        (
            "planner_service.add_models",
            lambda s: planner_service.add_models(s, ["nvidia-nemotron-3-embed-8b"]),
        ),
        ("planner_service.change_gpu_qty", lambda s: planner_service.change_gpu_qty(s, gpu(s), 1)),
        (
            "planner_service.set_model_prec",
            lambda s: planner_service.set_model_prec(s, model(s), "fp8"),
        ),
        (
            "planner_service.set_model_spec",
            lambda s: planner_service.set_model_spec(s, model(s), "ngram", 4),
        ),
        (
            "planner_service.set_model_gpu_count",
            lambda s: planner_service.set_model_gpu_count(s, model(s), 1),
        ),
        (
            "planner_service.set_model_gpu_pool",
            lambda s: planner_service.set_model_gpu_pool(s, model(s), gpu(s)),
        ),
    ]


class MutationEntryPointTests(unittest.TestCase):
    def test_every_mutation_entry_point_advances_the_revision(self):
        for label, mutate in _mutations():
            with self.subTest(entry_point=label):
                state = planner_service.create_default_state()
                before = state.revision
                mutate(state)
                self.assertGreater(
                    state.revision,
                    before,
                    f"{label} mutated state without advancing revision; "
                    "cached chart JSON and ETags would go stale",
                )

    def test_every_mutation_entry_point_changes_the_cache_revision(self):
        """The cache key, not just the counter, has to move for A and for B."""
        for label, mutate in _mutations():
            with self.subTest(entry_point=label):
                panel_a = planner_service.create_default_state()
                panel_b = planner_service.create_default_state()
                before = combined_revision(panel_a, panel_b)
                mutate(panel_a)
                self.assertNotEqual(combined_revision(panel_a, panel_b), before, label)

                before_b = combined_revision(panel_a, panel_b)
                mutate(panel_b)
                self.assertNotEqual(combined_revision(panel_a, panel_b), before_b, label)

    def test_the_entry_point_inventory_covers_every_decorated_mutator(self):
        """Guard against a new @bumps_revision mutator that nobody exercises."""
        covered = {
            label.split(".", 1)[1] for label, _ in _mutations() if label.startswith("state.")
        }
        decorated = {
            name
            for name, function in _public_state_functions()
            if getattr(function, "__bumps_revision__", False)
        }
        self.assertEqual(
            sorted(decorated - covered),
            [],
            "decorated mutators with no entry in _mutations(); add one so the "
            "behavioural check actually exercises them",
        )


class NestedMutationHazardTests(unittest.TestCase):
    """Document the hole the checks above exist to catch."""

    def test_appending_to_a_state_list_in_place_does_not_bump_on_its_own(self):
        state = planner_service.create_default_state()
        before = state.revision
        state.models.append(state.models[0])
        self.assertEqual(
            state.revision,
            before,
            "in-place list mutation now bumps on its own; if that is deliberate, "
            "relax this test — but the mutators must still not rely on it",
        )
        state.touch()
        self.assertGreater(state.revision, before)

    def test_reassigning_a_state_list_does_bump(self):
        state = planner_service.create_default_state()
        before = state.revision
        state.models = list(state.models[:1])
        self.assertGreater(state.revision, before)


if __name__ == "__main__":
    unittest.main()
