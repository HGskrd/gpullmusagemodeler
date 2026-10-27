"""The "+ Add model" picker: search, fit indicator, quality floor, no "Add all"."""

from __future__ import annotations

import copy
import re
import unittest
import uuid

from app_factory import create_test_app

import web.session_store as session_store
from data import MODELS, models_by_kind, required_quality
from planner_service import add_model, plan_model_add
from presentation.model_picker import (
    FIT_FITS,
    FIT_NEEDS,
    FIT_NO_GPU,
    FIT_NONE,
    TIGHT_KV_HEADROOM_FRACTION,
    panel_quality_floor,
    picker_row,
    picker_rows,
)
from state import GpuPool, ModelAssignment, PlannerState

SMALL = "l8"  # Llama 3.1 8B, ~16 GB BF16
HUGE = "qwen38-2.4t-a95b"  # multi-TB checkpoint


def _row_html(html: str, model_key: str) -> str:
    match = re.search(
        r'<button[^>]*data-model-row[^>]*"model_key":"'
        + re.escape(model_key)
        + r'"[^>]*>(.*?)</button>',
        html,
        re.S,
    )
    assert match, f"row for {model_key} not rendered"
    return match.group(0)


class ModelPickerRouteTests(unittest.TestCase):
    def setUp(self):
        self.app = create_test_app()
        self.client = self.app.test_client()
        tab_id = str(uuid.uuid4())
        self.headers = {"X-Tab-ID": tab_id}
        self.client.get("/", headers=self.headers)
        self.scope = next(scope for scope in session_store._states if scope.endswith(tab_id))

    def _use(self, state: PlannerState) -> None:
        session_store._states[self.scope] = state

    def _picker(self, kind: str = "llm") -> str:
        response = self.client.get(f"/picker/model?panel=A&kind={kind}", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        return response.get_data(as_text=True)

    def test_renders_search_filters_and_no_add_all(self):
        self._use(PlannerState(gpus=[GpuPool(1, "H100", 8)]))
        html = self._picker()
        self.assertIn("data-model-search", html)
        self.assertIn("No matching models", html)
        self.assertIn('data-model-filter="fit"', html)
        self.assertIn('data-model-filter="quality"', html)
        self.assertNotIn("Add all", html)
        self.assertNotIn("/model/add-many", html)
        # Rows remain plain HTMX add buttons.
        self.assertIn('hx-post="/model/add"', _row_html(html, SMALL))

    def test_quality_chip_is_llm_only(self):
        self._use(PlannerState(gpus=[GpuPool(1, "H100", 8)]))
        self.assertNotIn('data-model-filter="quality"', self._picker("embedding"))

    def test_small_model_fits_and_huge_model_does_not(self):
        self._use(PlannerState(gpus=[GpuPool(1, "H100", 8)]))
        html = self._picker()
        small = _row_html(html, SMALL)
        self.assertIn("fit-fits", small)
        self.assertIn("fits on 1", small)
        self.assertIn('data-pass-fit="1"', small)
        self.assertIn("16.0 GB BF16", small)
        huge = _row_html(html, HUGE)
        self.assertIn("doesn&#39;t fit", huge)
        self.assertIn('data-pass-fit="0"', huge)

    def test_no_gpu_state(self):
        self._use(PlannerState(gpus=[]))
        html = self._picker()
        self.assertIn("add a GPU first", _row_html(html, SMALL))
        self.assertRegex(html, r'data-model-filter="fit"[^>]*disabled')

    def test_adding_from_picker_still_works(self):
        self._use(PlannerState(gpus=[GpuPool(1, "H100", 8)]))
        response = self.client.post(
            "/model/add", headers=self.headers, data={"panel": "A", "model_key": SMALL}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual([am.model_key for am in session_store._states[self.scope].models], [SMALL])


class ModelPickerRowTests(unittest.TestCase):
    def test_needs_more_gpus_than_free(self):
        state = PlannerState(
            gpus=[GpuPool(1, "H100", 4)],
            models=[ModelAssignment(2, SMALL, 1, 4, 4, 1, "bf16")],
        )
        row = picker_row(state, MODELS["llama33-70b"], None, True)
        self.assertEqual(row.fit, FIT_NEEDS)
        self.assertFalse(row.passes_fit)
        self.assertRegex(row.fit_label, r"^needs \d+ GPUs?$")

    def test_no_gpu_and_none(self):
        self.assertEqual(picker_row(PlannerState(), MODELS[SMALL], None, True).fit, FIT_NO_GPU)
        state = PlannerState(gpus=[GpuPool(1, "L4", 1)])
        self.assertEqual(picker_row(state, MODELS[HUGE], None, True).fit, FIT_NONE)

    def test_fit_row_matches_what_add_model_does(self):
        state = PlannerState(gpus=[GpuPool(1, "H100", 8)])
        for key in (SMALL, "llama33-70b"):
            plan = plan_model_add(state, MODELS[key])
            assert plan is not None
            added = copy.deepcopy(state)
            add_model(added, key)
            am = added.models[-1]
            self.assertEqual(
                (am.gpu_uid, am.gpu_count, am.prec), (1, plan.gpu_count, plan.precision)
            )
            row = picker_row(state, MODELS[key], None, True)
            self.assertIn(row.fit, {FIT_FITS, "tight"})
            self.assertAlmostEqual(
                row.weights_gb, MODELS[key].gpu_weight_bytes(plan.precision) / 1e9
            )
        self.assertGreater(TIGHT_KV_HEADROOM_FRACTION, 0)

    def test_quality_floor_is_least_demanding_use_case(self):
        state = PlannerState(gpus=[GpuPool(1, "H100", 8)])
        self.assertIsNone(panel_quality_floor(state))
        base = session_store.get_state(f"picker-{uuid.uuid4().hex}")
        self.assertTrue(base.projects)
        expected = min(
            required_quality(p.difficulty, p.min_success_rate, quality_floor=p.quality_floor)
            for p in base.projects
        )
        self.assertAlmostEqual(panel_quality_floor(base), expected)
        rows = picker_rows(base, models_by_kind()["llm"], "llm")
        flat = [row for group in rows.values() for row in group]
        for row in flat:
            assert row.quality is not None
            self.assertEqual(row.passes_quality, row.quality + 1e-9 >= expected)
        # Non-LLM kinds carry no quality and never fail the quality chip.
        embed = picker_rows(base, models_by_kind()["embedding"], "embedding")
        self.assertTrue(
            all(r.quality is None and r.passes_quality for g in embed.values() for r in g)
        )


if __name__ == "__main__":
    unittest.main()
