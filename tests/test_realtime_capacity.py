import copy
import unittest

from calc import (
    chart_asr_quality,
    chart_embedding_quality,
    chart_realtime_capacity,
    chart_user_pareto,
    compute_realtime_capacity,
    compute_revenue_projection,
    embedding_quality_axis_range,
    get_decode_bs,
)
from data import ASR_WER_PLACEHOLDER, DIST_PRESETS, MODELS, PUBLISHED_ASR_WER
from state import (
    GpuPool,
    ModelAssignment,
    PlannerState,
    Project,
    VISIBLE_PLOT_MODES,
    normalize_plot_mode,
    retune_models,
)


class RealtimeCapacityTests(unittest.TestCase):
    def _state(self) -> PlannerState:
        state = PlannerState(
            gpus=[GpuPool(1, "H100", 1, cost_per_gpu_hour=1.0)],
            models=[ModelAssignment(2, "voxtral-realtime-mini-4b", 1, 1, 1, 1, "bf16")],
        )
        retune_models(state, preserve_existing=False)
        return state

    def test_voxtral_realtime_capacity_ignores_use_case_distribution(self):
        state = self._state()
        am = state.models[0]
        gpu = state.gpus[0].gpu
        model = MODELS[am.model_key]

        base = compute_realtime_capacity(
            model,
            (am.tp, am.pp, am.dp),
            8,
            gpu,
            state.mu,
            state.profiled_non_kv_gb,
            am.prec,
            state.decode_efficiency,
        )

        state.in_dist = copy.deepcopy(DIST_PRESETS["Long doc"]["in"])
        state.out_dist = copy.deepcopy(DIST_PRESETS["Long doc"]["out"])
        shifted = compute_realtime_capacity(
            model,
            (am.tp, am.pp, am.dp),
            8,
            gpu,
            state.mu,
            state.profiled_non_kv_gb,
            am.prec,
            state.decode_efficiency,
        )

        self.assertIsNotNone(base)
        self.assertIsNotNone(shifted)
        self.assertEqual(base.realtime_factor, shifted.realtime_factor)
        self.assertEqual(base.max_slots, shifted.max_slots)

    def test_realtime_chart_reports_concurrent_stream_capacity(self):
        state = self._state()

        datasets = chart_realtime_capacity(state, [1, 8, 16])

        self.assertEqual(len(datasets), 1)
        point = datasets[0]["data"][1]
        self.assertEqual(point["users"], 8)
        self.assertGreater(point["required_tps"], 0.0)
        self.assertIn("max_users", point)

    def test_user_pareto_excludes_asr_models(self):
        text_only = PlannerState(
            gpus=[GpuPool(1, "H100", 2, cost_per_gpu_hour=1.0)],
            models=[ModelAssignment(3, "q08", 1, 1, 1, 1, "bf16")],
        )
        mixed = PlannerState(
            gpus=[GpuPool(1, "H100", 2, cost_per_gpu_hour=1.0)],
            models=[
                ModelAssignment(2, "voxtral-realtime-mini-4b", 1, 1, 1, 1, "bf16"),
                ModelAssignment(3, "q08", 1, 1, 1, 1, "bf16"),
            ],
        )
        retune_models(text_only, preserve_existing=False)
        retune_models(mixed, preserve_existing=False)

        batch_sizes = get_decode_bs([mixed])
        datasets = chart_user_pareto(mixed, batch_sizes)
        labels = [dataset["label"] for dataset in datasets]

        self.assertEqual(batch_sizes, get_decode_bs([text_only]))
        self.assertEqual(len(datasets), 1)
        self.assertTrue(any("Qwen 3.5 0.8B" in label for label in labels))
        self.assertFalse(any("Voxtral" in label for label in labels))

    def test_visible_asr_view_is_quality_max_streams_plot(self):
        visible_modes = [mode for mode, _label in VISIBLE_PLOT_MODES]

        self.assertNotIn("realtime", visible_modes)
        self.assertIn("asrquality", visible_modes)
        self.assertEqual(normalize_plot_mode("realtime"), "asrquality")

    def test_visible_embedding_view_is_quality_plot_only(self):
        visible_modes = [mode for mode, _label in VISIBLE_PLOT_MODES]

        self.assertNotIn("embedding", visible_modes)
        self.assertIn("embedquality", visible_modes)
        self.assertEqual(normalize_plot_mode("embedding"), "embedquality")

    def test_asr_quality_chart_uses_sourced_wer_points(self):
        new_asr_models = [
            "nvidia-nemotron-speech-streaming-0.6b",
            "nvidia-nemotron-3.5-asr-streaming-0.6b",
            "nvidia-parakeet-unified-0.6b",
            "nvidia-parakeet-realtime-eou-120m",
            "nvidia-multitalker-parakeet-streaming-0.6b",
            "kyutai-stt-1b-en-fr",
            "kyutai-stt-2.6b-en",
            "moonshine-streaming-tiny",
            "moonshine-streaming-small",
            "moonshine-streaming-medium",
            "fun-asr-nano-2512",
            "granite-4.0-1b-speech",
            "nvidia-parakeet-tdt-0.6b-v3",
        ]
        state = PlannerState(
            gpus=[GpuPool(1, "H100", 2, cost_per_gpu_hour=1.0)],
            models=[
                ModelAssignment(2, "voxtral-realtime-mini-4b", 1, 1, 1, 1, "bf16"),
                ModelAssignment(3, "mimo-v2.5-asr", 1, 1, 1, 1, "bf16"),
                *[
                    ModelAssignment(10 + idx, key, 1, 1, 1, 1, "bf16")
                    for idx, key in enumerate(new_asr_models)
                ],
            ],
        )
        retune_models(state, preserve_existing=False)

        datasets = chart_asr_quality(state)

        self.assertEqual(ASR_WER_PLACEHOLDER, frozenset({
            "kyutai-stt-1b-en-fr",
            "fun-asr-nano-2512",
        }))
        self.assertEqual(PUBLISHED_ASR_WER["voxtral-realtime-mini-4b"]["en"], 4.90)
        self.assertEqual(PUBLISHED_ASR_WER["voxtral-realtime-mini-4b"]["fr_covost"], 9.68)
        self.assertEqual(PUBLISHED_ASR_WER["voxtral-realtime-mini-4b"]["fr_fleurs"], 8.44)
        self.assertEqual(PUBLISHED_ASR_WER["voxtral-realtime-mini-4b"]["fr_mls"], 5.64)
        self.assertEqual(PUBLISHED_ASR_WER["mimo-v2.5-asr"]["en"], 5.73)
        self.assertNotIn("fr_covost", PUBLISHED_ASR_WER["mimo-v2.5-asr"])
        self.assertEqual(PUBLISHED_ASR_WER["nvidia-nemotron-3.5-asr-streaming-0.6b"]["en"], 7.99)
        self.assertEqual(PUBLISHED_ASR_WER["nvidia-nemotron-3.5-asr-streaming-0.6b"]["fr_fleurs"], 9.45)
        self.assertEqual(PUBLISHED_ASR_WER["nvidia-parakeet-tdt-0.6b-v3"]["fr_covost"], 6.38)
        self.assertEqual(PUBLISHED_ASR_WER["nvidia-parakeet-tdt-0.6b-v3"]["fr_fleurs"], 4.76)
        self.assertEqual(PUBLISHED_ASR_WER["nvidia-parakeet-tdt-0.6b-v3"]["fr_mls"], 5.12)
        self.assertEqual(PUBLISHED_ASR_WER["granite-4.0-1b-speech"]["fr_commonvoice"], 7.15)
        self.assertEqual(PUBLISHED_ASR_WER["moonshine-streaming-medium"]["en"], 6.65)

        seen_keys = {ds["_modelKey"] for ds in datasets}
        self.assertIn("voxtral-realtime-mini-4b", seen_keys)
        self.assertIn("mimo-v2.5-asr", seen_keys)
        for key in new_asr_models:
            self.assertIn(key, seen_keys)

        self.assertFalse(next(ds for ds in datasets if ds["_modelKey"] == "granite-4.0-1b-speech")["_asrStreaming"])
        self.assertFalse(next(ds for ds in datasets if ds["_modelKey"] == "nvidia-parakeet-tdt-0.6b-v3")["_asrStreaming"])
        self.assertTrue(next(ds for ds in datasets if ds["_modelKey"] == "nvidia-nemotron-speech-streaming-0.6b")["_asrStreaming"])
        self.assertTrue(next(ds for ds in datasets if ds["_modelKey"] == "nvidia-nemotron-3.5-asr-streaming-0.6b")["_asrStreaming"])
        for dataset in datasets:
            self.assertEqual(dataset["_placeholder"], dataset["_modelKey"] in ASR_WER_PLACEHOLDER)
            self.assertTrue(dataset["showLine"])
            for point in dataset["data"]:
                self.assertGreater(point["max_users"], 0)
                self.assertIn("language", point)
                self.assertIn("source", point)
                self.assertIn(point["asr_mode"], {"streaming", "non-streaming"})

    def test_asr_quality_chart_keeps_duplicate_two_wer_deployments(self):
        state = PlannerState(
            gpus=[
                GpuPool(1, "A10", 1, cost_per_gpu_hour=1.0),
                GpuPool(2, "H100", 1, cost_per_gpu_hour=1.0),
                GpuPool(3, "H200", 1, cost_per_gpu_hour=1.0),
            ],
            models=[
                ModelAssignment(2, "voxtral-realtime-mini-4b", 1, 1, 1, 1, "bf16"),
                ModelAssignment(3, "voxtral-realtime-mini-4b", 2, 1, 1, 1, "bf16"),
                ModelAssignment(4, "voxtral-realtime-mini-4b", 3, 1, 1, 1, "bf16"),
            ],
        )

        datasets = chart_asr_quality(state)

        self.assertEqual(len(datasets), 3)
        self.assertEqual(len({dataset["label"] for dataset in datasets}), 1)
        self.assertEqual(len({dataset["_seriesId"] for dataset in datasets}), 3)
        self.assertEqual(sum(len(dataset["data"]) for dataset in datasets), 12)
        wers = sorted({point["wer"] for dataset in datasets for point in dataset["data"]})
        self.assertEqual(wers, [4.9, 5.64, 8.44, 9.68])
        self.assertGreater(len({dataset["data"][0]["max_users"] for dataset in datasets}), 1)

    def test_embedding_quality_axis_range_tracks_visible_quality_points(self):
        state = PlannerState(
            gpus=[GpuPool(1, "H100", 2, cost_per_gpu_hour=1.0)],
            models=[
                ModelAssignment(2, "mxbai-embed-xsmall-v1", 1, 1, 1, 1, "bf16"),
                ModelAssignment(3, "denseon", 1, 1, 1, 1, "bf16"),
            ],
        )
        retune_models(state, preserve_existing=False)

        datasets = chart_embedding_quality(state)
        axis = embedding_quality_axis_range(datasets)
        points = [point for dataset in datasets for point in dataset["data"]]
        qualities = [point["quality"] for point in points]
        decontaminated_beir = [point["decontaminated_beir_quality"] for point in points]

        self.assertGreater(len(qualities), 1)
        self.assertIn(0.4280, qualities)
        self.assertIn(0.5771, qualities)
        self.assertIn(None, decontaminated_beir)
        self.assertIn(0.5771, decontaminated_beir)
        self.assertTrue(any(point["uses_decontaminated_beir"] for point in points))
        self.assertTrue(any(not point["uses_decontaminated_beir"] for point in points))
        self.assertLess(axis["y_min"], min(qualities))
        self.assertGreater(axis["y_max"], max(qualities))
        self.assertGreater(axis["y_min"], 0.0)
        self.assertLess(axis["y_max"], 1.0)

    def test_realtime_models_do_not_enter_use_case_projection_supply(self):
        state = self._state()
        state.projects = [
            Project(
                3,
                "Easy classification",
                difficulty=0.10,
                tokens_day=10_000_000,
                wtp_per_m=1.0,
                min_success_rate=0.80,
                quality_floor=0.0,
            )
        ]

        projection = compute_revenue_projection(state)

        self.assertFalse(projection["has_supply"])
        self.assertEqual(projection["models"], [])


if __name__ == "__main__":
    unittest.main()
