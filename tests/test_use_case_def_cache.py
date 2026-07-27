"""Staleness guards for the normalized use-case definition cache.

normalize_use_case_defs() memoizes its result and is skipped while the cache
still points at the same list object. Reassigning state.use_case_defs
invalidates it implicitly; mutating a definition in place does not, so every
in-place edit path has to invalidate explicitly. These tests pin that contract.
"""

import copy
import unittest
from dataclasses import asdict
from typing import Any
from unittest.mock import patch

import state as state_module
from scenarios import replace_use_case_defs, serialize_scenario
from state import (
    PlannerState,
    _find_use_case_def,
    add_use_case_def,
    get_use_case_defs,
    normalize_use_case_defs,
    remove_use_case_def,
    set_use_case_def_capability,
    set_use_case_def_field,
)


class UseCaseDefCacheTests(unittest.TestCase):
    def find(self, state: PlannerState, key: str) -> dict[str, Any]:
        item = _find_use_case_def(state, key)
        assert item is not None, f"use-case def {key!r} should exist"
        return item

    def test_repeated_reads_normalize_only_once(self):
        state = PlannerState()
        get_use_case_defs(state)

        with patch(
            "state._normalize_use_case_def",
            wraps=state_module._normalize_use_case_def,
        ) as normalize:
            for _ in range(50):
                get_use_case_defs(state)
                _find_use_case_def(state, "coding")

        normalize.assert_not_called()

    def test_field_edit_is_visible_immediately(self):
        state = PlannerState()
        original = self.find(state, "coding")["wtp_per_m"]

        set_use_case_def_field(state, "coding", "wtp_per_m", original + 3.0)

        self.assertAlmostEqual(self.find(state, "coding")["wtp_per_m"], original + 3.0)

    def test_raised_scale_value_widens_scale_max_on_next_read(self):
        state = PlannerState()
        beyond = float(self.find(state, "coding")["scale_kind"]["max"]) * 4.0

        set_use_case_def_field(state, "coding", "scale_value", beyond)

        refreshed = self.find(state, "coding")
        self.assertAlmostEqual(refreshed["scale_value"], beyond)
        self.assertGreaterEqual(float(refreshed["scale_kind"]["max"]), beyond)

    def test_capability_edit_is_visible_immediately(self):
        state = PlannerState()

        set_use_case_def_capability(state, "coding", "images", True)
        self.assertIn("images", self.find(state, "coding")["requires"])

        set_use_case_def_capability(state, "coding", "images", False)
        self.assertNotIn("images", self.find(state, "coding")["requires"])

    def test_added_definition_is_findable(self):
        state = PlannerState()
        get_use_case_defs(state)

        item = add_use_case_def(state)

        self.assertIsNotNone(_find_use_case_def(state, item["key"]))

    def test_removed_definition_is_not_findable(self):
        state = PlannerState()
        get_use_case_defs(state)

        remove_use_case_def(state, "coding")

        self.assertIsNone(_find_use_case_def(state, "coding"))

    def test_replacing_the_library_invalidates_implicitly(self):
        state = PlannerState()
        get_use_case_defs(state)

        replace_use_case_defs(
            state,
            {"use_cases": [{"key": "only_one", "name": "Only one", "tokens_day": 1e6}]},
        )

        self.assertIsNone(_find_use_case_def(state, "coding"))
        self.assertIsNotNone(_find_use_case_def(state, "only_one"))

    def test_cache_never_reaches_serialized_output(self):
        state = PlannerState()
        get_use_case_defs(state)

        self.assertNotIn(state_module._UCD_CACHE_ATTR, asdict(state))
        self.assertNotIn(
            state_module._UCD_CACHE_ATTR,
            serialize_scenario(state, None)["panel_a"],
        )

    def test_deepcopied_state_keeps_a_valid_cache(self):
        state = PlannerState()
        get_use_case_defs(state)
        clone = copy.deepcopy(state)

        # deepcopy memoizes, so the clone's marker must still point at the
        # clone's own list rather than the original's.
        cached = getattr(clone, state_module._UCD_CACHE_ATTR)
        self.assertIs(cached[0], clone.use_case_defs)
        self.assertIsNot(cached[0], state.use_case_defs)

        set_use_case_def_field(clone, "coding", "wtp_per_m", 42.0)
        self.assertAlmostEqual(self.find(clone, "coding")["wtp_per_m"], 42.0)
        self.assertNotAlmostEqual(self.find(state, "coding")["wtp_per_m"], 42.0)

    def test_normalization_stays_idempotent(self):
        state = PlannerState()
        normalize_use_case_defs(state)
        first = [dict(d) for d in state.use_case_defs]

        state_module.invalidate_use_case_defs(state)
        normalize_use_case_defs(state)

        self.assertEqual(first, [dict(d) for d in state.use_case_defs])


if __name__ == "__main__":
    unittest.main()
