"""Published release scores; protocols are deliberately not cross-walked."""

RELEASE_EMBEDDING_METRICS: dict[str, dict] = {
    "qwen3-embedding-0.6b": {
        "value": 0.6464,
        "metric": "MTEB Multilingual Retrieval aggregate (Retri. column)",
        "source": "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/README.md",
        "protocol": "Publisher model-card MTEB Multilingual table; full output "
        "dimensions and instruction-aware text embedding. Score is "
        "retrieval aggregate, not Mean(Task), and is not "
        "cross-walked to BEIR, RTEB or other multilingual task "
        "snapshots.",
        "available": True,
        "comparable_to_other_catalog_metrics": False,
        "comparison_group": "qwen3_card_mteb_multilingual_retrieval",
    },
    "qwen3-embedding-4b": {
        "value": 0.696,
        "metric": "MTEB Multilingual Retrieval aggregate (Retri. column)",
        "source": "https://huggingface.co/Qwen/Qwen3-Embedding-4B/blob/5cf2132abc99cad020ac570b19d031efec650f2b/README.md",
        "protocol": "Publisher model-card MTEB Multilingual table; full output "
        "dimensions and instruction-aware text embedding. Score is "
        "retrieval aggregate, not Mean(Task), and is not cross-walked "
        "to BEIR, RTEB or other multilingual task snapshots.",
        "available": True,
        "comparable_to_other_catalog_metrics": False,
        "comparison_group": "qwen3_card_mteb_multilingual_retrieval",
    },
    "qwen3-embedding-8b": {
        "value": 0.7088,
        "metric": "MTEB Multilingual Retrieval aggregate (Retri. column)",
        "source": "https://huggingface.co/Qwen/Qwen3-Embedding-8B/blob/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af/README.md",
        "protocol": "Publisher model-card MTEB Multilingual table; full output "
        "dimensions and instruction-aware text embedding. Score is "
        "retrieval aggregate, not Mean(Task), and is not cross-walked "
        "to BEIR, RTEB or other multilingual task snapshots.",
        "available": True,
        "comparable_to_other_catalog_metrics": False,
        "comparison_group": "qwen3_card_mteb_multilingual_retrieval",
    },
    "nomic-embed-text-v2-moe": {
        "value": 0.5286,
        "metric": "BEIR average nDCG@10",
        "source": "https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe/blob/1066b6599d099fbb93dfcb64f9c37a7c9e503e85/README.md",
        "protocol": "Publisher card BEIR52.86, supported by technical report "
        "section5.3: FlagEmbedding evaluation, "
        "search_query/search_document prefixes, all inputs "
        "truncated512tokens. Dataset/protocol differences from "
        "other BEIR aggregates must remain explicit.",
        "available": True,
        "comparable_to_other_catalog_metrics": False,
        "comparison_group": "nomic_v2_report_beir",
        "protocol_source": "https://arxiv.org/html/2502.07972v1#S5.SS3",
    },
    "harrier-oss-v1-270m": {
        "value": None,
        "metric": "Retrieval aggregate unavailable in pinned official card",
        "source": "https://huggingface.co/microsoft/harrier-oss-v1-270m/blob/31de22b673913c7d658c0f03f792d77c2dcf8ebd/README.md",
        "protocol": "Card publishes multilingual MTEBv2 overall scores only; no "
        "retrieval-only aggregate or model-index found. Do not label "
        "overall66.5/69.0/74.3 as retrieval. Use capacity-only "
        "uncalibrated display for these entries.",
        "available": False,
        "comparable_to_other_catalog_metrics": False,
        "alternate_published_metric": "Multilingual MTEB v2 overall score",
        "alternate_published_value": 0.665,
    },
    "harrier-oss-v1-0.6b": {
        "value": None,
        "metric": "Retrieval aggregate unavailable in pinned official card",
        "source": "https://huggingface.co/microsoft/harrier-oss-v1-0.6b/blob/f9b9dc8d367d443f2479d27aa5d8d2850c0774ee/README.md",
        "protocol": "Card publishes multilingual MTEBv2 overall scores only; no "
        "retrieval-only aggregate or model-index found. Do not label "
        "overall66.5/69.0/74.3 as retrieval. Use capacity-only "
        "uncalibrated display for these entries.",
        "available": False,
        "comparable_to_other_catalog_metrics": False,
        "alternate_published_metric": "Multilingual MTEB v2 overall score",
        "alternate_published_value": 0.69,
    },
    "harrier-oss-v1-27b": {
        "value": None,
        "metric": "Retrieval aggregate unavailable in pinned official card",
        "source": "https://huggingface.co/microsoft/harrier-oss-v1-27b/blob/0c0fc62f6d8af9e8604cb818c412301b103a0093/README.md",
        "protocol": "Card publishes multilingual MTEBv2 overall scores only; no "
        "retrieval-only aggregate or model-index found. Do not label "
        "overall66.5/69.0/74.3 as retrieval. Use capacity-only "
        "uncalibrated display for these entries.",
        "available": False,
        "comparable_to_other_catalog_metrics": False,
        "alternate_published_metric": "Multilingual MTEB v2 overall score",
        "alternate_published_value": 0.743,
    },
    "jina-code-embeddings-0.5b": {
        "value": 0.7872,
        "metric": "MTEB Code AVG (technical report Table2)",
        "source": "https://arxiv.org/html/2508.21290v1#S5",
        "protocol": "Paper Table2 published MTEB Code AVG, not "
        "overall25-benchmarkaverage. AppendixC: "
        "FP16,batch8,8192tokens; task-specific code "
        "query/document prefixes. Table reports percentages "
        "without explicitly specifying one uniform underlying "
        "metric, so do not relabel as BEIR nDCG@10.",
        "available": True,
        "comparable_to_other_catalog_metrics": False,
        "checkpoint_source": "https://huggingface.co/jinaai/jina-code-embeddings-0.5b/blob/4db235132dafbe56a8b9c5f59b59795ecf58a4a7/README.md",
        "comparison_group": "jina_code_report_mteb_code_avg",
    },
    "jina-code-embeddings-1.5b": {
        "value": 0.7894,
        "metric": "MTEB Code AVG (technical report Table2)",
        "source": "https://arxiv.org/html/2508.21290v1#S5",
        "protocol": "Paper Table2 published MTEB Code AVG, not "
        "overall25-benchmarkaverage. AppendixC: "
        "FP16,batch8,8192tokens; task-specific code "
        "query/document prefixes. Table reports percentages "
        "without explicitly specifying one uniform underlying "
        "metric, so do not relabel as BEIR nDCG@10.",
        "available": True,
        "comparable_to_other_catalog_metrics": False,
        "checkpoint_source": "https://huggingface.co/jinaai/jina-code-embeddings-1.5b/blob/39aeb4fb9b60f930934c78ae5d749a46287c248a/README.md",
        "comparison_group": "jina_code_report_mteb_code_avg",
    },
    "qwen3-vl-embedding-2b": {
        "value": 0.6712,
        "metric": "MMTEB Retrieval aggregate (Retri. column)",
        "source": "https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/blob/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda/README.md",
        "protocol": "Publisher model-card MMTEB text table. This is text "
        "retrieval quality, not the separate MMEB-V2 multimodal "
        "average. No cross-walk to another catalog "
        "benchmark/protocol.",
        "available": True,
        "comparable_to_other_catalog_metrics": False,
        "comparison_group": "qwen3_vl_card_mmteb_retrieval",
    },
    "qwen3-vl-embedding-8b": {
        "value": 0.6941,
        "metric": "MMTEB Retrieval aggregate (Retri. column)",
        "source": "https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B/blob/2c4565515e0f265c6511776e7193b22c0968ddc7/README.md",
        "protocol": "Publisher model-card MMTEB text table. This is text "
        "retrieval quality, not the separate MMEB-V2 multimodal "
        "average. No cross-walk to another catalog "
        "benchmark/protocol.",
        "available": True,
        "comparable_to_other_catalog_metrics": False,
        "comparison_group": "qwen3_vl_card_mmteb_retrieval",
    },
    "vultronretrieverflash-qwen3.5-0.8b": {
        "value": 0.5616,
        "metric": "ViDoRe V3 mean nDCG@10 (10 tasks)",
        "source": "https://huggingface.co/vultr/VultronRetrieverFlash-Qwen3.5-0.8B/blob/6f6b6757e9ea3ad93f4f2cbd839bc6f2f0b93739/README.md",
        "protocol": "Publisher official MTEB late-interaction "
        "evaluation,320dimensions,1792visualtokens;8public+2privateViDoReV3tasks. "
        "Visual document retrieval quality; not "
        "comparable to text BEIR/MMTEB and not a "
        "measurement of planner text-input throughput.",
        "available": True,
        "comparable_to_other_catalog_metrics": False,
        "comparison_group": "vultron_vidore_v3_10_tasks_320d_1792visual",
    },
    "vultronretrievercore-qwen3.5-4.5b": {
        "value": 0.6357,
        "metric": "ViDoRe V3 mean nDCG@10 (10 tasks)",
        "source": "https://huggingface.co/vultr/VultronRetrieverCore-Qwen3.5-4.5B/blob/b0857609ad6db741009a7026e02cf30bd384f54d/README.md",
        "protocol": "Publisher official MTEB late-interaction "
        "evaluation,320dimensions,1792visualtokens;8public+2privateViDoReV3tasks. "
        "Visual document retrieval quality; not "
        "comparable to text BEIR/MMTEB and not a "
        "measurement of planner text-input throughput.",
        "available": True,
        "comparable_to_other_catalog_metrics": False,
        "comparison_group": "vultron_vidore_v3_10_tasks_320d_1792visual",
    },
    "vultronretrieverprime-qwen3.5-8b": {
        "value": 0.6426,
        "metric": "ViDoRe V3 mean nDCG@10 (10 tasks)",
        "source": "https://huggingface.co/vultr/VultronRetrieverPrime-Qwen3.5-8B/blob/ca8bc3c904d845761a942965e4c27734a3e45ac1/README.md",
        "protocol": "Publisher official MTEB late-interaction "
        "evaluation,320dimensions,1792visualtokens;8public+2privateViDoReV3tasks. "
        "Visual document retrieval quality; not "
        "comparable to text BEIR/MMTEB and not a "
        "measurement of planner text-input throughput.",
        "available": True,
        "comparable_to_other_catalog_metrics": False,
        "comparison_group": "vultron_vidore_v3_10_tasks_320d_1792visual",
    },
}

RELEASE_ASR_WER_ROWS: list[dict] = [
    {
        "model_key": "qwen3-asr-0.6b",
        "language": "en",
        "inference_mode": "offline",
        "metric": "WER",
        "score_percent": 2.11,
        "protocol_id": "qwen2026_librispeech_clean",
        "dataset": "LibriSpeech test-clean",
        "source_url": "https://huggingface.co/Qwen/Qwen3-ASR-0.6B/blob/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/README.md",
        "score_date": None,
        "notes": "Qwen vendor model card ASR Inference Mode Performance table; BF16 vLLM greedy with "
        "max_new_tokens1024 and no forced language in general evaluation. Streaming-specific "
        "chunk/rollback settings are not specified in this benchmark table; do not claim this "
        "score calibrates proposed30s/8x throughput profile.",
        "crosswalk_to_existing_en_aggregate": False,
    },
    {
        "model_key": "qwen3-asr-0.6b",
        "language": "en",
        "inference_mode": "offline",
        "metric": "WER",
        "score_percent": 4.55,
        "protocol_id": "qwen2026_librispeech_other",
        "dataset": "LibriSpeech test-other",
        "source_url": "https://huggingface.co/Qwen/Qwen3-ASR-0.6B/blob/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/README.md",
        "score_date": None,
        "notes": "Qwen vendor model card ASR Inference Mode Performance table; BF16 vLLM greedy with "
        "max_new_tokens1024 and no forced language in general evaluation. Streaming-specific "
        "chunk/rollback settings are not specified in this benchmark table; do not claim this "
        "score calibrates proposed30s/8x throughput profile.",
        "crosswalk_to_existing_en_aggregate": False,
    },
    {
        "model_key": "qwen3-asr-0.6b",
        "language": "en",
        "inference_mode": "offline",
        "metric": "WER",
        "score_percent": 4.39,
        "protocol_id": "qwen2026_fleurs_en",
        "dataset": "FLEURS English",
        "source_url": "https://huggingface.co/Qwen/Qwen3-ASR-0.6B/blob/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/README.md",
        "score_date": None,
        "notes": "Qwen vendor model card ASR Inference Mode Performance table; BF16 vLLM greedy with "
        "max_new_tokens1024 and no forced language in general evaluation. Streaming-specific "
        "chunk/rollback settings are not specified in this benchmark table; do not claim this "
        "score calibrates proposed30s/8x throughput profile.",
        "crosswalk_to_existing_en_aggregate": False,
    },
    {
        "model_key": "qwen3-asr-1.7b",
        "language": "en",
        "inference_mode": "offline",
        "metric": "WER",
        "score_percent": 1.63,
        "protocol_id": "qwen2026_librispeech_clean",
        "dataset": "LibriSpeech test-clean",
        "source_url": "https://huggingface.co/Qwen/Qwen3-ASR-0.6B/blob/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/README.md",
        "score_date": None,
        "notes": "Qwen vendor model card ASR Inference Mode Performance table; BF16 vLLM greedy with "
        "max_new_tokens1024 and no forced language in general evaluation. Streaming-specific "
        "chunk/rollback settings are not specified in this benchmark table; do not claim this "
        "score calibrates proposed30s/8x throughput profile.",
        "crosswalk_to_existing_en_aggregate": False,
    },
    {
        "model_key": "qwen3-asr-1.7b",
        "language": "en",
        "inference_mode": "offline",
        "metric": "WER",
        "score_percent": 3.38,
        "protocol_id": "qwen2026_librispeech_other",
        "dataset": "LibriSpeech test-other",
        "source_url": "https://huggingface.co/Qwen/Qwen3-ASR-0.6B/blob/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/README.md",
        "score_date": None,
        "notes": "Qwen vendor model card ASR Inference Mode Performance table; BF16 vLLM greedy with "
        "max_new_tokens1024 and no forced language in general evaluation. Streaming-specific "
        "chunk/rollback settings are not specified in this benchmark table; do not claim this "
        "score calibrates proposed30s/8x throughput profile.",
        "crosswalk_to_existing_en_aggregate": False,
    },
    {
        "model_key": "qwen3-asr-1.7b",
        "language": "en",
        "inference_mode": "offline",
        "metric": "WER",
        "score_percent": 3.35,
        "protocol_id": "qwen2026_fleurs_en",
        "dataset": "FLEURS English",
        "source_url": "https://huggingface.co/Qwen/Qwen3-ASR-0.6B/blob/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/README.md",
        "score_date": None,
        "notes": "Qwen vendor model card ASR Inference Mode Performance table; BF16 vLLM greedy with "
        "max_new_tokens1024 and no forced language in general evaluation. Streaming-specific "
        "chunk/rollback settings are not specified in this benchmark table; do not claim this "
        "score calibrates proposed30s/8x throughput profile.",
        "crosswalk_to_existing_en_aggregate": False,
    },
    {
        "model_key": "qwen3-asr-0.6b-streaming",
        "language": "en",
        "inference_mode": "streaming",
        "metric": "WER",
        "score_percent": 2.54,
        "protocol_id": "qwen2026_librispeech_clean",
        "dataset": "LibriSpeech test-clean",
        "source_url": "https://huggingface.co/Qwen/Qwen3-ASR-0.6B/blob/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/README.md",
        "score_date": None,
        "notes": "Qwen vendor model card ASR Inference Mode Performance table; BF16 vLLM greedy with "
        "max_new_tokens1024 and no forced language in general evaluation. Streaming-specific "
        "chunk/rollback settings are not specified in this benchmark table; do not claim this "
        "score calibrates proposed30s/8x throughput profile.",
        "crosswalk_to_existing_en_aggregate": False,
    },
    {
        "model_key": "qwen3-asr-0.6b-streaming",
        "language": "en",
        "inference_mode": "streaming",
        "metric": "WER",
        "score_percent": 6.27,
        "protocol_id": "qwen2026_librispeech_other",
        "dataset": "LibriSpeech test-other",
        "source_url": "https://huggingface.co/Qwen/Qwen3-ASR-0.6B/blob/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/README.md",
        "score_date": None,
        "notes": "Qwen vendor model card ASR Inference Mode Performance table; BF16 vLLM greedy with "
        "max_new_tokens1024 and no forced language in general evaluation. Streaming-specific "
        "chunk/rollback settings are not specified in this benchmark table; do not claim this "
        "score calibrates proposed30s/8x throughput profile.",
        "crosswalk_to_existing_en_aggregate": False,
    },
    {
        "model_key": "qwen3-asr-0.6b-streaming",
        "language": "en",
        "inference_mode": "streaming",
        "metric": "WER",
        "score_percent": 5.38,
        "protocol_id": "qwen2026_fleurs_en",
        "dataset": "FLEURS English",
        "source_url": "https://huggingface.co/Qwen/Qwen3-ASR-0.6B/blob/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/README.md",
        "score_date": None,
        "notes": "Qwen vendor model card ASR Inference Mode Performance table; BF16 vLLM greedy with "
        "max_new_tokens1024 and no forced language in general evaluation. Streaming-specific "
        "chunk/rollback settings are not specified in this benchmark table; do not claim this "
        "score calibrates proposed30s/8x throughput profile.",
        "crosswalk_to_existing_en_aggregate": False,
    },
    {
        "model_key": "qwen3-asr-1.7b-streaming",
        "language": "en",
        "inference_mode": "streaming",
        "metric": "WER",
        "score_percent": 1.95,
        "protocol_id": "qwen2026_librispeech_clean",
        "dataset": "LibriSpeech test-clean",
        "source_url": "https://huggingface.co/Qwen/Qwen3-ASR-0.6B/blob/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/README.md",
        "score_date": None,
        "notes": "Qwen vendor model card ASR Inference Mode Performance table; BF16 vLLM greedy with "
        "max_new_tokens1024 and no forced language in general evaluation. Streaming-specific "
        "chunk/rollback settings are not specified in this benchmark table; do not claim this "
        "score calibrates proposed30s/8x throughput profile.",
        "crosswalk_to_existing_en_aggregate": False,
    },
    {
        "model_key": "qwen3-asr-1.7b-streaming",
        "language": "en",
        "inference_mode": "streaming",
        "metric": "WER",
        "score_percent": 4.51,
        "protocol_id": "qwen2026_librispeech_other",
        "dataset": "LibriSpeech test-other",
        "source_url": "https://huggingface.co/Qwen/Qwen3-ASR-0.6B/blob/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/README.md",
        "score_date": None,
        "notes": "Qwen vendor model card ASR Inference Mode Performance table; BF16 vLLM greedy with "
        "max_new_tokens1024 and no forced language in general evaluation. Streaming-specific "
        "chunk/rollback settings are not specified in this benchmark table; do not claim this "
        "score calibrates proposed30s/8x throughput profile.",
        "crosswalk_to_existing_en_aggregate": False,
    },
    {
        "model_key": "qwen3-asr-1.7b-streaming",
        "language": "en",
        "inference_mode": "streaming",
        "metric": "WER",
        "score_percent": 4.02,
        "protocol_id": "qwen2026_fleurs_en",
        "dataset": "FLEURS English",
        "source_url": "https://huggingface.co/Qwen/Qwen3-ASR-0.6B/blob/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/README.md",
        "score_date": None,
        "notes": "Qwen vendor model card ASR Inference Mode Performance table; BF16 vLLM greedy with "
        "max_new_tokens1024 and no forced language in general evaluation. Streaming-specific "
        "chunk/rollback settings are not specified in this benchmark table; do not claim this "
        "score calibrates proposed30s/8x throughput profile.",
        "crosswalk_to_existing_en_aggregate": False,
    },
    {
        "model_key": "whisper-large-v3",
        "language": "en",
        "inference_mode": "offline",
        "metric": "WER",
        "score_percent": 4.08,
        "protocol_id": "qwen2026_fleurs_en",
        "dataset": "FLEURS English",
        "source_url": "https://huggingface.co/Qwen/Qwen3-ASR-0.6B/blob/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/README.md",
        "score_date": None,
        "notes": "Whisper-large-v3 baseline evaluated and published by Qwen, not an OpenAI-published "
        "benchmark. Same Qwen Public Datasets table as Qwen scores; generation settings for "
        "baseline not separately documented.",
        "crosswalk_to_existing_en_aggregate": False,
    },
    {
        "model_key": "whisper-large-v3",
        "language": "en",
        "inference_mode": "offline",
        "metric": "WER",
        "score_percent": 1.51,
        "protocol_id": "qwen2026_librispeech_clean",
        "dataset": "LibriSpeech test-clean",
        "source_url": "https://huggingface.co/Qwen/Qwen3-ASR-0.6B/blob/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/README.md",
        "score_date": None,
        "notes": "Whisper-large-v3 baseline from Qwen vendor evaluation; baseline generation settings "
        "not separately documented.",
        "crosswalk_to_existing_en_aggregate": False,
    },
    {
        "model_key": "whisper-large-v3",
        "language": "en",
        "inference_mode": "offline",
        "metric": "WER",
        "score_percent": 3.97,
        "protocol_id": "qwen2026_librispeech_other",
        "dataset": "LibriSpeech test-other",
        "source_url": "https://huggingface.co/Qwen/Qwen3-ASR-0.6B/blob/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/README.md",
        "score_date": None,
        "notes": "Whisper-large-v3 baseline from Qwen vendor evaluation; baseline generation settings "
        "not separately documented.",
        "crosswalk_to_existing_en_aggregate": False,
    },
    {
        "model_key": "ark-asr-0.6b",
        "language": "en",
        "inference_mode": "offline",
        "metric": "WER",
        "score_percent": 1.53,
        "protocol_id": "hfopen_legacy_librispeech_clean",
        "dataset": "LibriSpeech test-clean",
        "source_url": "https://huggingface.co/Edge0/ARK-ASR-0.6B/blob/45776b56d58cdfb2e2eb632f7e110f38684633e0/.eval_results/open_asr_leaderboard.yaml",
        "score_date": "2026-06-17",
        "notes": "Exact checkpoint-published evaluation YAML. Normalization/generation protocol not "
        "completely restated in YAML; do not merge with Qwen-vendor evaluations or changed "
        "October aggregate.",
        "crosswalk_to_existing_en_aggregate": False,
    },
    {
        "model_key": "ark-asr-0.6b",
        "language": "en",
        "inference_mode": "offline",
        "metric": "WER",
        "score_percent": 3.51,
        "protocol_id": "hfopen_legacy_librispeech_other",
        "dataset": "LibriSpeech test-other",
        "source_url": "https://huggingface.co/Edge0/ARK-ASR-0.6B/blob/45776b56d58cdfb2e2eb632f7e110f38684633e0/.eval_results/open_asr_leaderboard.yaml",
        "score_date": "2026-06-17",
        "notes": "Exact checkpoint-published evaluation YAML. Normalization/generation protocol not "
        "completely restated in YAML; do not merge with Qwen-vendor evaluations or changed "
        "October aggregate.",
        "crosswalk_to_existing_en_aggregate": False,
    },
    {
        "model_key": "ark-asr-0.6b",
        "language": "en",
        "inference_mode": "offline",
        "metric": "WER",
        "score_percent": 5.53,
        "protocol_id": "hfopen_legacy_mean_asreported_2026-06-17",
        "dataset": "HF Open ASR mean_wer as reported in checkpoint evaluation YAML",
        "source_url": "https://huggingface.co/Edge0/ARK-ASR-0.6B/blob/45776b56d58cdfb2e2eb632f7e110f38684633e0/.eval_results/open_asr_leaderboard.yaml",
        "score_date": "2026-06-17",
        "notes": "Retain as source-specific aggregate only. ARK README and YAML aggregates differ "
        "(3BREADME5.04 vsYAML4.76); dataset lists differ, so do not relabel as a shared English "
        "aggregate.",
        "crosswalk_to_existing_en_aggregate": False,
    },
    {
        "model_key": "ark-asr-3b",
        "language": "en",
        "inference_mode": "offline",
        "metric": "WER",
        "score_percent": 1.03,
        "protocol_id": "hfopen_legacy_librispeech_clean",
        "dataset": "LibriSpeech test-clean",
        "source_url": "https://huggingface.co/Edge0/ARK-ASR-3B/blob/1e28271b79edc97635783bea65abc89195a09ed3/.eval_results/open_asr_leaderboard.yaml",
        "score_date": "2026-06-23",
        "notes": "Exact checkpoint-published evaluation YAML. Normalization/generation protocol not "
        "completely restated in YAML; do not merge with Qwen-vendor evaluations or changed "
        "October aggregate.",
        "crosswalk_to_existing_en_aggregate": False,
    },
    {
        "model_key": "ark-asr-3b",
        "language": "en",
        "inference_mode": "offline",
        "metric": "WER",
        "score_percent": 2.35,
        "protocol_id": "hfopen_legacy_librispeech_other",
        "dataset": "LibriSpeech test-other",
        "source_url": "https://huggingface.co/Edge0/ARK-ASR-3B/blob/1e28271b79edc97635783bea65abc89195a09ed3/.eval_results/open_asr_leaderboard.yaml",
        "score_date": "2026-06-23",
        "notes": "Exact checkpoint-published evaluation YAML. Normalization/generation protocol not "
        "completely restated in YAML; do not merge with Qwen-vendor evaluations or changed "
        "October aggregate.",
        "crosswalk_to_existing_en_aggregate": False,
    },
    {
        "model_key": "ark-asr-3b",
        "language": "en",
        "inference_mode": "offline",
        "metric": "WER",
        "score_percent": 4.76,
        "protocol_id": "hfopen_legacy_mean_asreported_2026-06-23",
        "dataset": "HF Open ASR mean_wer as reported in checkpoint evaluation YAML",
        "source_url": "https://huggingface.co/Edge0/ARK-ASR-3B/blob/1e28271b79edc97635783bea65abc89195a09ed3/.eval_results/open_asr_leaderboard.yaml",
        "score_date": "2026-06-23",
        "notes": "Retain as source-specific aggregate only. ARK README and YAML aggregates differ "
        "(3BREADME5.04 vsYAML4.76); dataset lists differ, so do not relabel as a shared English "
        "aggregate.",
        "crosswalk_to_existing_en_aggregate": False,
    },
    {
        "model_key": "moss-transcribe-diarize-0.9b",
        "language": "en",
        "inference_mode": "offline",
        "metric": "WER",
        "score_percent": 1.64,
        "protocol_id": "hfopen_legacy_librispeech_clean",
        "dataset": "LibriSpeech test-clean",
        "source_url": "https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize/blob/704aa4a9c304e8520be88901e0d1960158ef5b15/.eval_results/open_asr_leaderboard.yaml",
        "score_date": "2026-07-25",
        "notes": "Exact checkpoint-published evaluation YAML. Normalization/generation protocol not "
        "completely restated in YAML; do not merge with Qwen-vendor evaluations or changed "
        "October aggregate.",
        "crosswalk_to_existing_en_aggregate": False,
    },
    {
        "model_key": "moss-transcribe-diarize-0.9b",
        "language": "en",
        "inference_mode": "offline",
        "metric": "WER",
        "score_percent": 4.15,
        "protocol_id": "hfopen_legacy_librispeech_other",
        "dataset": "LibriSpeech test-other",
        "source_url": "https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize/blob/704aa4a9c304e8520be88901e0d1960158ef5b15/.eval_results/open_asr_leaderboard.yaml",
        "score_date": "2026-07-25",
        "notes": "Exact checkpoint-published evaluation YAML. Normalization/generation protocol not "
        "completely restated in YAML; do not merge with Qwen-vendor evaluations or changed "
        "October aggregate.",
        "crosswalk_to_existing_en_aggregate": False,
    },
    {
        "model_key": "moss-transcribe-diarize-0.9b",
        "language": "en",
        "inference_mode": "offline",
        "metric": "WER",
        "score_percent": 5.17,
        "protocol_id": "hfopen_legacy_mean_asreported_2026-07-25",
        "dataset": "HF Open ASR mean_wer as reported in checkpoint evaluation YAML",
        "source_url": "https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize/blob/704aa4a9c304e8520be88901e0d1960158ef5b15/.eval_results/open_asr_leaderboard.yaml",
        "score_date": "2026-07-25",
        "notes": "Retain as source-specific aggregate only. ARK README and YAML aggregates differ "
        "(3BREADME5.04 vsYAML4.76); dataset lists differ, so do not relabel as a shared English "
        "aggregate.",
        "crosswalk_to_existing_en_aggregate": False,
    },
    {
        "model_key": "granite-speech-5.0-2b",
        "language": "en",
        "inference_mode": "offline",
        "metric": "WER",
        "score_percent": 5.68,
        "protocol_id": "hfopen_20261021_9public_2private_mean",
        "dataset": "HF Open ASR October21 2026;9public+2private English sets",
        "source_url": "https://huggingface.co/ibm-granite/granite-speech-5.0-2b/blob/c684620bb54fc89d14ba27256b23bbc9e09ba764/assets/granite-5.0-2b-vs-open-english.png",
        "score_date": "2026-10-21",
        "notes": "Exact labeled legend/average in official IBM card image, visually inspected. Changed "
        "dataset protocol includesURGENT2024+Monsoon+private tests; incompatible with older "
        "English mean_wer.",
        "crosswalk_to_existing_en_aggregate": False,
    },
    {
        "model_key": "granite-speech-5.0-2b",
        "language": "en",
        "inference_mode": "offline",
        "metric": "WER",
        "score_percent": 6.02,
        "protocol_id": "ibm_granite5_commonvoice_en",
        "dataset": "CommonVoice English; IBM vendor evaluation",
        "source_url": "https://huggingface.co/ibm-granite/granite-speech-5.0-2b/blob/c684620bb54fc89d14ba27256b23bbc9e09ba764/assets/granite-speech-5.0-2b-wer-cv.png",
        "score_date": None,
        "notes": "Exact labeled bar in IBM model card image; CommonVoice version/subset is not "
        "explicitly given by this plot. Do not merge with other CommonVoice versions.",
        "crosswalk_to_existing_en_aggregate": False,
    },
    {
        "model_key": "granite-speech-5.0-2b",
        "language": "fr",
        "inference_mode": "offline",
        "metric": "WER",
        "score_percent": 6.49,
        "protocol_id": "ibm_granite5_commonvoice_fr",
        "dataset": "CommonVoice French; IBM vendor evaluation",
        "source_url": "https://huggingface.co/ibm-granite/granite-speech-5.0-2b/blob/c684620bb54fc89d14ba27256b23bbc9e09ba764/assets/granite-speech-5.0-2b-wer-cv.png",
        "score_date": None,
        "notes": "Exact labeled bar in IBM model card image; CommonVoice version/subset is not "
        "explicitly given by this plot. Do not map to CoVoST/FLEURS/MLS or an unspecified "
        "French aggregate.",
        "crosswalk_to_existing_en_aggregate": False,
    },
    {
        "model_key": "ark-asr-3b",
        "language": "en",
        "inference_mode": "offline",
        "metric": "WER",
        "score_percent": 5.92,
        "protocol_id": "hfopen_20261021_9public_2private_mean",
        "dataset": "HF Open ASR October21 2026;9public+2private English sets",
        "source_url": "https://huggingface.co/ibm-granite/granite-speech-5.0-2b/blob/c684620bb54fc89d14ba27256b23bbc9e09ba764/assets/granite-5.0-2b-vs-open-english.png",
        "score_date": "2026-10-21",
        "notes": "Competitor score in official IBM comparison figure, exact legend value. "
        "Fine-tuning/serving details not specified in plot; use dated protocol, not historical "
        "English aggregate.",
        "crosswalk_to_existing_en_aggregate": False,
    },
    {
        "model_key": "moss-transcribe-diarize-0.9b",
        "language": "en",
        "inference_mode": "offline",
        "metric": "WER",
        "score_percent": 6.17,
        "protocol_id": "hfopen_20261021_9public_2private_mean",
        "dataset": "HF Open ASR October21 2026;9public+2private English sets",
        "source_url": "https://huggingface.co/ibm-granite/granite-speech-5.0-2b/blob/c684620bb54fc89d14ba27256b23bbc9e09ba764/assets/granite-5.0-2b-vs-open-english.png",
        "score_date": "2026-10-21",
        "notes": "Competitor score in official IBM comparison figure, exact legend value. "
        "Fine-tuning/serving details not specified in plot; use dated protocol, not historical "
        "English aggregate.",
        "crosswalk_to_existing_en_aggregate": False,
    },
]
