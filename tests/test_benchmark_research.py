"""Coverage, partitioning and resumability of the offline research queue."""

import json
from dataclasses import replace

import pytest

from benchmark_feasibility import ResearchGrid, screen_pair
from benchmark_research import completed_tasks, iter_tasks, main
from calc import EfficiencyParams, compute_data, compute_memory, valid_strategies
from data import GPUS, MODELS, artifact_options, configured_model
from data.hardware_picker import purchasable_gpu_keys


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_full_catalog_product_preserves_every_pair_and_units():
    tasks = list(iter_tasks(queue="all"))
    pairs = {(row["hardware"]["key"], row["model"]["key"]) for row in tasks}
    assert pairs == {(gpu, model) for gpu in GPUS for model in MODELS}
    assert len({row["task_id"] for row in tasks}) == len(GPUS) * len(MODELS)
    row = next(row for row in tasks if row["hardware"]["key"] == "H100")
    assert row["hardware"]["memory_bytes_per_device"] == GPUS["H100"].mem
    assert row["hardware"]["peak_memory_bandwidth_bytes_per_second"] == GPUS["H100"].bw
    assert all(
        not artifact["key"].startswith("estimate:")
        for row in tasks
        for artifact in row["model"]["artifacts"]
    )


def test_visible_scope_and_estimated_artifacts():
    tasks = list(iter_tasks(visible_only=True, queue="all"))
    assert len(tasks) == len(purchasable_gpu_keys()) * sum(not m.hidden for m in MODELS.values())
    assert all(row["hardware"]["picker_visible"] and not row["model"]["hidden"] for row in tasks)
    rows = list(iter_tasks(hardware_keys=["H100"], model_keys=["qwen38-27b"]))
    expanded = list(
        iter_tasks(
            hardware_keys=["H100"],
            model_keys=["qwen38-27b"],
            include_estimated_artifacts=True,
        )
    )
    assert len(expanded[0]["model"]["artifacts"]) > len(rows[0]["model"]["artifacts"])
    assert expanded[0]["input_fingerprint"] != rows[0]["input_fingerprint"]


def test_ids_stay_stable_but_changed_catalog_inputs_invalidate_completion(monkeypatch):
    selection = {"hardware_keys": ["H100"], "model_keys": ["qwen38-27b"]}
    before = next(iter_tasks(**selection))
    monkeypatch.setitem(GPUS, "H100", replace(GPUS["H100"], bw=123.0))
    after = next(iter_tasks(**selection))
    assert before["task_id"] == after["task_id"]
    assert before["input_fingerprint"] != after["input_fingerprint"]
    assert after == next(iter_tasks(**selection))


def test_shards_cover_selection_and_resume_retries_errors_and_changed_inputs(tmp_path):
    common = ["--hardware", "H100", "--hardware", "RTX4090", "--model", "qwen38-27b"]
    whole = tmp_path / "whole.jsonl"
    assert main([*common, "--output", str(whole)]) == 0
    rows = read_rows(whole)
    shards = []
    for index in range(3):
        path = tmp_path / f"shard-{index}.jsonl"
        main([*common, "--output", str(path), "--shard-count", "3", "--shard-index", str(index)])
        shards.extend(read_rows(path))
    assert sorted(shards, key=lambda row: row["task_id"]) == rows
    results = tmp_path / "results.jsonl"
    results.write_text(
        "\n".join(
            json.dumps(dict(row, status=status)) for row, status in zip(rows, ["found", "error"])
        )
    )
    remaining = tmp_path / "remaining.jsonl"
    main([*common, "--completed", str(results), "--output", str(remaining)])
    assert read_rows(remaining) == rows[1:]
    manifest = json.loads(remaining.with_suffix(".jsonl.manifest.json").read_text())
    assert manifest["counts"] == {
        "selected_pairs": 2,
        "shard_pairs": 2,
        "skipped_completed": 1,
        "written": 1,
    }
    refreshed = tmp_path / "refreshed.jsonl"
    main(
        [
            *common,
            "--completed",
            str(results),
            "--include-estimated-artifacts",
            "--output",
            str(refreshed),
        ]
    )
    assert len(read_rows(refreshed)) == 2
    with pytest.raises(SystemExit):
        main([*common, "--output", str(whole)])
    assert read_rows(whole) == rows


@pytest.mark.parametrize("status", ["found", "not_found", "unsupported", "error"])
def test_terminal_statuses(tmp_path, status):
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps({"task_id": "job", "input_fingerprint": "hash", "status": status}))
    assert completed_tasks(path) == ({("job", "hash")} if status != "error" else set())


@pytest.mark.parametrize(
    "line",
    [
        '{"status":"found"}',
        "[]",
        "invalid json",
        '{"task_id":[],"input_fingerprint":"x","status":"found"}',
    ],
)
def test_malformed_completions_fail_with_line_number(tmp_path, line):
    path = tmp_path / "bad.jsonl"
    path.write_text("\n" + line)
    with pytest.raises(ValueError, match=r"bad.jsonl:2:"):
        completed_tasks(path)


@pytest.mark.parametrize(
    "extra",
    [
        ["--model", "typo"],
        ["--hardware", "typo"],
        ["--shard-count", "0"],
        ["--shard-index", "-1"],
        ["--shard-count", "2", "--shard-index", "2"],
    ],
)
def test_invalid_arguments_do_not_create_output(tmp_path, extra):
    path = tmp_path / "invalid.jsonl"
    with pytest.raises(SystemExit):
        main(["--output", str(path), *extra])
    assert not path.exists()


def native_artifacts(key):
    return [option for option in artifact_options(MODELS[key]) if option["native"]]


def test_default_queue_rejects_pairs_that_do_not_fit_requested_gpu_bound():
    grid = ResearchGrid(gpu_counts=(1,), batch_sizes=(1,))
    common = {"hardware_keys": ["H100", "RTX3060_12"], "model_keys": ["qwen38-27b"], "grid": grid}
    plausible = list(iter_tasks(**common))
    rejected = list(iter_tasks(**common, queue="exploratory"))
    assert [row["hardware"]["key"] for row in plausible] == ["H100"]
    assert [row["hardware"]["key"] for row in rejected] == ["RTX3060_12"]
    assert rejected[0]["screening"]["configurations"] == []
    assert rejected[0]["screening"]["rejected_grid_cells"]


def test_feasible_topologies_and_predictions_match_calculator():
    grid = ResearchGrid(gpu_counts=(1, 4), batch_sizes=(1, 16))
    key = "qwen38-27b"
    artifact = native_artifacts(key)[0]
    model = configured_model(key, artifact["key"], "auto", "auto", "auto")
    gpu = GPUS["H100"]
    result = screen_pair("H100", key, [artifact], grid)
    assert len(result["configurations"]) == 4
    for config in result["configurations"]:
        strat = config["tp"], config["pp"], config["dp"]
        assert config["gpu_count"] == strat[0] * strat[1] * strat[2]
        assert strat in valid_strategies(
            model, config["gpu_count"], gpu, 0.9, 4.0, artifact["precision"]
        )
        batch = config["workload"]["batch_size"]
        prediction = compute_data(
            model,
            strat,
            strat,
            batch,
            2048,
            128,
            gpu,
            0.9,
            4.0,
            artifact["precision"],
            0.0,
            EfficiencyParams(),
            EfficiencyParams(),
        )
        assert prediction is not None
        assert config["planner_estimate"]["requests_per_second"] == prediction.rps
        assert config["runtime_evidence"] == "estimated"
    assert result["runtime_support_verified"] is False


def test_weights_fit_alone_does_not_establish_workload_fit(monkeypatch):
    key = "qwen38-27b"
    artifact = native_artifacts(key)[0]
    model = configured_model(key, artifact["key"], "auto", "auto", "auto")
    precision = artifact["precision"]
    tiny_headroom = (model.gpu_weight_bytes(precision) + 4e9 + 1024) / 0.9
    gpu = replace(GPUS["H100"], mem=tiny_headroom)
    monkeypatch.setitem(GPUS, "H100", gpu)
    assert compute_memory(model, 1, 1, gpu, 0.9, 4.0, precision, EfficiencyParams()) is not None
    result = screen_pair("H100", key, [artifact], ResearchGrid(gpu_counts=(1,), batch_sizes=(1,)))
    assert not result["configurations"]
    assert result["rejected_grid_cells"] == {"workload_memory_or_capacity": 1}


def test_native_fp4_is_rejected_on_hopper_but_auto_fallback_is_checked():
    artifacts = [
        artifact
        for artifact in artifact_options(MODELS["nemotron35-lightning"])
        if artifact["precision"] == "nvfp4"
    ]
    native = screen_pair(
        "H100",
        "nemotron35-lightning",
        artifacts,
        ResearchGrid(gpu_counts=(1,), batch_sizes=(1,), execution_backends=("native",)),
    )
    assert native["rejected_grid_cells"] == {"execution_unsupported": 1}
    auto = screen_pair(
        "H100", "nemotron35-lightning", artifacts, ResearchGrid(gpu_counts=(1,), batch_sizes=(1,))
    )
    assert auto["configurations"][0]["execution_path"] == "w4a16"


@pytest.mark.parametrize(
    "key,metric",
    [
        ("d1-2b", "decisions_per_second"),
        ("embeddinggemma-2", "inputs_per_second"),
        ("voxtral-realtime-mini-4b", "aggregate_audio_seconds_per_second"),
    ],
)
def test_modalities_use_their_own_workload_and_units(key, metric):
    result = screen_pair(
        "H100", key, native_artifacts(key), ResearchGrid(gpu_counts=(1,), batch_sizes=(1,))
    )
    config = result["configurations"][0]
    assert metric in config["planner_estimate"]
    assert config["planner_estimate"][metric] > 0
    if MODELS[key].is_asr_model:
        assert "input_tokens" not in config["workload"]
    else:
        assert config["workload"]["output_tokens"] == 0


def test_exploration_is_bounded_and_context_is_not_silently_truncated():
    common = {
        "hardware_keys": ["H100"],
        "model_keys": ["qwen38-27b", "embeddinggemma-2", "d1-2b"],
        "grid": ResearchGrid(gpu_counts=(1,), batch_sizes=(1,), input_tokens=10**9),
    }
    assert list(iter_tasks(**common)) == []
    sample = list(iter_tasks(**common, queue="exploratory", exploratory_limit=2))
    assert len(sample) == 2
    assert sample == list(iter_tasks(**common, queue="exploratory", exploratory_limit=2))
    assert all(
        row["screening"]["rejected_grid_cells"]
        == {"context_exceeded": len(row["model"]["artifacts"])}
        for row in sample
    )


@pytest.mark.parametrize(
    "extra",
    [
        ["--gpu-count", "0"],
        ["--batch-size", "-1"],
        ["--input-tokens", "0"],
        ["--memory-utilization", "1.1"],
        ["--non-kv-gb", "nan"],
        ["--exploratory-limit", "0"],
    ],
)
def test_invalid_grid_does_not_create_output(tmp_path, extra):
    path = tmp_path / "invalid.jsonl"
    with pytest.raises(SystemExit):
        main(["--output", str(path), *extra])
    assert not path.exists()


def test_purchase_constraints_are_normalized_and_deduplicated(monkeypatch):
    monkeypatch.setitem(GPUS, "H100", replace(GPUS["H100"], min_count=4, count_multiple=4))
    result = screen_pair(
        "H100",
        "qwen38-27b",
        native_artifacts("qwen38-27b"),
        ResearchGrid(gpu_counts=(1, 2, 4), batch_sizes=(1,)),
    )
    assert result["effective_gpu_counts"] == [4]
    assert len(result["configurations"]) == 1
    assert result["configurations"][0]["gpu_count"] == 4


def test_changed_grid_invalidates_resume_identity():
    selection = {"hardware_keys": ["H100"], "model_keys": ["qwen38-27b"]}
    first = next(iter_tasks(**selection, grid=ResearchGrid(gpu_counts=(1,))))
    changed = next(iter_tasks(**selection, grid=ResearchGrid(gpu_counts=(1,), input_tokens=512)))
    assert first["task_id"] == changed["task_id"]
    assert first["input_fingerprint"] != changed["input_fingerprint"]
