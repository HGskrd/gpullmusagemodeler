"""Task contracts survive persistence and constrain routing and native capacity."""

import pytest
from test_use_case_modalities import scenario

from data import MODELS
from data.task_primitives import supports_task
from engine.economics import (
    _native_deployment_capacity,
    _project_workload_profile,
    compute_revenue_projection,
)
from placement import _model_serves_project
from scenarios import deserialize_scenario, serialize_scenario
from state import set_use_case_def_field


def test_task_contract_round_trip_and_boolean_options():
    state, project = scenario("ticket_routing", "pplx-decider-flash-v1.2")
    for field, value in (
        ("task_primitive", "boolean"),
        ("questions_per_request", 3),
        ("options_per_question", 8),
    ):
        set_use_case_def_field(state, "ticket_routing", field, value)
    restored, _ = deserialize_scenario(serialize_scenario(state, None))
    assert restored.projects[0].task_primitive == "boolean"
    assert restored.projects[0].questions_per_request == 3
    assert restored.projects[0].options_per_question == 8
    profile = _project_workload_profile(project, {})
    assert profile["options_per_question"] == 2
    assert _model_serves_project(MODELS["pplx-decider-flash-v1.2"], project)
    assert not _model_serves_project(MODELS["intern-decision-0.8b"], project)


def test_unsupported_extraction_never_receives_classification_capacity():
    state, project = scenario("ticket_routing", "pplx-decider-flash-v1.2")
    set_use_case_def_field(state, "ticket_routing", "task_primitive", "span_extraction")
    assert not _model_serves_project(state.models[0].model, project)
    profile = _project_workload_profile(project, {})
    assert _native_deployment_capacity(state, state.models[0], state.gpus[0].gpu, profile) == (0, 0)
    row = compute_revenue_projection(state, include_recommendations=False)["projects"][0]
    assert row["served"] == 0
    assert row["cloud_blocked"]
    assert row["task_primitive"] == "span_extraction"


def test_question_count_reaches_per_question_runtime():
    state, project = scenario("ticket_routing", "startlux-decision-v1-4b")
    profile = _project_workload_profile(project, {})
    single = _native_deployment_capacity(state, state.models[0], state.gpus[0].gpu, profile)
    project.questions_per_request = 3
    triple = _native_deployment_capacity(
        state, state.models[0], state.gpus[0].gpu, _project_workload_profile(project, {})
    )
    assert triple[1] == pytest.approx(single[1] / 3, rel=0.01)
    assert triple[0] == pytest.approx(triple[1] * 86400 * profile["in_len"])


def test_legacy_defaults_and_explicit_capabilities():
    _, project = scenario("ticket_routing", "intern-decision-0.8b")
    assert (
        project.task_primitive,
        project.questions_per_request,
        project.options_per_question,
    ) == ("choice", 1, 4)
    assert supports_task(MODELS["pplx-decider-flash-v1.2"], "score")
    assert not supports_task(MODELS["pplx-decider-flash-v1.2"], "span_extraction")


def test_flash13_locate_catalog_capacity_and_round_trip():
    state, project = scenario("ticket_routing", "pplx-decider-flash-v1.3")
    set_use_case_def_field(state, "ticket_routing", "task_primitive", "span_extraction")
    set_use_case_def_field(state, "ticket_routing", "questions_per_request", 4)
    assert _model_serves_project(state.models[0].model, project)
    profile = _project_workload_profile(project, {})
    daily, rps = _native_deployment_capacity(state, state.models[0], state.gpus[0].gpu, profile)
    assert daily > 0 and rps > 0
    assert daily == pytest.approx(rps * 86400 * profile["in_len"])
    row = compute_revenue_projection(state, include_recommendations=False)["projects"][0]
    assert row["served"] > 0
    restored, _ = deserialize_scenario(serialize_scenario(state, None))
    assert restored.projects[0].task_primitive == "span_extraction"
    assert restored.models[0].model.key == "pplx-decider-flash-v1.3"
    assert _native_deployment_capacity(
        state, state.models[0], state.gpus[0].gpu, {**profile, "in_len": 32769}
    ) == (0, 0)
