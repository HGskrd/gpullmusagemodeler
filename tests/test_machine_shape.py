"""Pool machine shape (GPUs per machine, network) and shared prefill/decode layouts."""

from data import GPUS
from placement import retune_models
from planner_service import add_model
from presentation.econ import econ_payload
from presentation.model_cards import get_model_info, tech_scorecard
from presentation.placement_view import machine_options, placement_view
from state import PlannerState, add_gpu, pool_gpu, set_gpu_machines, set_model_strat


def _deepseek_on_16_rtx_pro() -> PlannerState:
    s = PlannerState()
    add_gpu(s, "RTXPRO6000_BSE", 16)
    add_model(s, "deepseek-v4.1-flash")
    s.models[0].gpu_count = 16
    retune_models(s, preserve_existing=False)
    return s


def test_pool_without_override_uses_catalog_gpu():
    s = _deepseek_on_16_rtx_pro()
    assert s.gpus[0].gpu is GPUS["RTXPRO6000_BSE"]
    assert s.gpus[0].machine_count == 2


def test_override_sets_node_size_and_caps_network_at_peer_bandwidth():
    base = GPUS["RTXPRO6000_BSE"]
    g = pool_gpu("RTXPRO6000_BSE", 4, 400.0)
    assert g.node_size == 4
    assert g.inter_node_bw == 50e9
    # 800 Gb/s = 100 GB/s exceeds the card's 64 GB/s PCIe peer path.
    capped = pool_gpu("RTXPRO6000_BSE", 0, 800.0)
    assert capped.node_size == base.node_size
    assert capped.inter_node_bw == base.scale_up_collective_bw


def test_machine_shape_change_retunes_layout_within_one_machine():
    s = _deepseek_on_16_rtx_pro()
    set_gpu_machines(s, s.gpus[0].uid, 4, 0.0)
    retune_models(s, preserve_existing=False)
    am = s.models[0]
    assert am.tp * am.pp <= 4


def test_layout_picker_sets_both_phases_and_keeps_capacity():
    s = _deepseek_on_16_rtx_pro()
    am = s.models[0]
    set_model_strat(s, am.uid, 4, 4, 1, "both")
    assert (am.tp, am.pp, am.dp) == (am.prefill_tp, am.prefill_pp, am.prefill_dp) == (4, 4, 1)
    assert tech_scorecard(s, econ_payload(s))[0]["capacity"] > 0


def test_shape_grid_deltas_are_relative_to_deployed_layout():
    s = _deepseek_on_16_rtx_pro()
    am = s.models[0]
    set_model_strat(s, am.uid, 4, 4, 1, "both")
    mo = machine_options(s, s.gpus[0])
    cur = mo["current"]
    assert cur["own_layouts"] == "TP4×PP4"
    assert not cur["own_is_pick"]
    assert cur["tps"] > cur["own_tps"]
    assert abs(cur["delta"] - (cur["tps"] / cur["own_tps"] - 1)) < 1e-9


def test_machine_map_marks_split_copies_and_single_machine():
    s = _deepseek_on_16_rtx_pro()
    am = s.models[0]
    set_model_strat(s, am.uid, 4, 4, 1, "both")
    pv = placement_view(s, get_model_info(s, am))
    assert pv["cur"]["span"] == 2 and pv["cur"]["partners"] == ["M2", "M1"]
    set_model_strat(s, am.uid, 8, 1, 2, "both")
    pv = placement_view(s, get_model_info(s, am))
    assert pv["cur"]["span"] == 1 and pv["cur"]["partners"] == ["", ""]
