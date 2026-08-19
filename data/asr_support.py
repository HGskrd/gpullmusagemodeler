"""ASR profiles and published word-error-rate evidence."""

from dataclasses import replace

from .model_class import RealtimeProfile

RWKV7_G1_HEAD_DIM = 64
RWKV7_G1_CONTEXT = 8192
RWKV7_G1_BASE_CAPABILITIES = frozenset({"reasoning"})
RWKV7_G1_TOOL_CAPABILITIES = frozenset({"tools", "reasoning"})
VOXTRAL_REALTIME_PROFILE = RealtimeProfile(
    label="Realtime ASR",
    tokens_per_second=12.5,
    audio_ms_per_token=80.0,
    target_delay_ms=480,
    state_tokens=8192,
    source="mistralai/Voxtral-Mini-4B-Realtime-2602",
    note="Realtime stream demand uses 6 delay tokens over 480 ms, i.e. 12.5 streaming ticks/sec. Each 80 ms tick also runs 4 causal audio-encoder tokens.",
    audio_encoder_params=0.97e9,
    audio_tokens_per_step=4,
    audio_attention_layers=32,
    audio_attention_heads=32,
    audio_attention_head_dim=64,
    audio_attention_window=750,
)
MIMO_V25_ASR_PROFILE = RealtimeProfile(
    label="ASR",
    tokens_per_second=6.25,
    audio_ms_per_token=160.0,
    target_delay_ms=1600,
    state_tokens=8192,
    source="XiaomiMiMo/MiMo-V2.5-ASR + XiaomiMiMo/MiMo-Audio-Tokenizer",
    note="ASR stream demand uses the published four-frame patch grouping (6.25 groups/sec). The catalog records the ASR encoder's 16-layer, 64-head × 16-dim local geometry; end-to-end tokenizer/adaptor work remains a conservative proxy.",
    audio_encoder_params=1.2e9,
    audio_tokens_per_step=1,
    audio_attention_layers=16,
    audio_attention_heads=64,
    audio_attention_head_dim=16,
    audio_attention_window=25,
)
GEMMA_4_E2B_ASR_PROFILE = RealtimeProfile(
    label="Offline Multilingual ASR",
    tokens_per_second=25.0,
    audio_ms_per_token=40.0,
    target_delay_ms=30000,
    state_tokens=750,
    source="google/gemma-4-E2B-it + Gemma 4 Technical Report",
    note="Gemma 4 accepts up to 30 seconds of audio as 750 40 ms tokens. E2B uses Google's frozen 305M-parameter USM-style encoder with two downsampling convolutions and 12 Conformer layers; offline capacity remains a conservative one-audio-token-per-planner-step proxy.",
    audio_encoder_params=305e6,
    audio_attention_layers=12,
    audio_attention_heads=8,
    audio_attention_head_dim=128,
    audio_attention_window=13,
    streaming=False,
)
GEMMA_4_E4B_ASR_PROFILE = replace(
    GEMMA_4_E2B_ASR_PROFILE,
    source="google/gemma-4-E4B-it + Gemma 4 Technical Report",
    note="Gemma 4 accepts up to 30 seconds of audio as 750 40 ms tokens. E4B uses Google's frozen 305M-parameter USM-style encoder with two downsampling convolutions and 12 Conformer layers; offline capacity remains a conservative one-audio-token-per-planner-step proxy.",
)
GEMMA_4_12B_ASR_PROFILE = RealtimeProfile(
    label="Offline Multilingual ASR",
    tokens_per_second=25.0,
    audio_ms_per_token=40.0,
    target_delay_ms=30000,
    state_tokens=750,
    source="google/gemma-4-12B-it + Gemma 4 Technical Report",
    note="Gemma 4 12B accepts up to 30 seconds of 16 kHz audio as 750 40 ms (640-sample) vectors projected directly into the LLM embedding space. It has no separate audio encoder; offline capacity uses one projected audio token per planner step.",
    streaming=False,
)
NEMOTRON_SPEECH_STREAMING_PROFILE = RealtimeProfile(
    label="Streaming ASR",
    tokens_per_second=1000.0 / 560.0,
    audio_ms_per_token=560.0,
    target_delay_ms=560,
    state_tokens=70,
    source="nvidia/nemotron-speech-streaming-en-0.6b",
    note="Cache-aware FastConformer-RNNT profile uses the 560 ms streaming chunk setting; the cached left context is 70 80 ms frames.",
)
NEMOTRON_35_ASR_STREAMING_PROFILE = RealtimeProfile(
    label="Multilingual Streaming ASR",
    tokens_per_second=1000.0 / 560.0,
    audio_ms_per_token=560.0,
    target_delay_ms=560,
    state_tokens=56,
    source="nvidia/nemotron-3.5-asr-streaming-0.6b",
    note="Prompt-conditioned cache-aware FastConformer-RNNT profile uses the 560 ms streaming chunk setting from att_context_size [56,6]; the cached left context is 56 80 ms frames.",
)
PARAKEET_UNIFIED_STREAMING_PROFILE = RealtimeProfile(
    label="Streaming ASR",
    tokens_per_second=1000.0 / 560.0,
    audio_ms_per_token=560.0,
    target_delay_ms=560,
    state_tokens=70,
    source="nvidia/parakeet-unified-en-0.6b",
    note="Unified FastConformer-RNNT profile uses the published 0.56 s streaming latency point with 5.6 s left context.",
)
PARAKEET_REALTIME_EOU_PROFILE = RealtimeProfile(
    label="Streaming ASR + EOU",
    tokens_per_second=6.25,
    audio_ms_per_token=160.0,
    target_delay_ms=160,
    state_tokens=70,
    source="nvidia/parakeet_realtime_eou_120m-v1",
    note="Voice-agent streaming profile uses the 160 ms setting from the model card and keeps the 70-frame cache-aware left context.",
)
MULTITALKER_PARAKEET_STREAMING_PROFILE = RealtimeProfile(
    label="Streaming Multitalker ASR",
    tokens_per_second=1000.0 / 1120.0,
    audio_ms_per_token=1120.0,
    target_delay_ms=1120,
    state_tokens=70,
    source="nvidia/multitalker-parakeet-streaming-0.6b-v1",
    note="Multitalker profile uses the 1.12 s setting; NVIDIA documents one ASR model instance per active speaker, so planner users should scale assignments by target speaker count.",
)
KYUTAI_STT_1B_PROFILE = RealtimeProfile(
    label="Streaming ASR",
    tokens_per_second=12.5,
    audio_ms_per_token=80.0,
    target_delay_ms=500,
    state_tokens=750,
    source="kyutai/stt-1b-en_fr",
    note="Kyutai delayed-streams profile uses the published 12.5 Hz Mimi frame rate, 32 audio tokens per frame, and 0.5 s text delay. Capacity remains a conservative shared-trunk proxy until parallel codebook streams and the depformer are modeled separately.",
)
KYUTAI_STT_2_6B_PROFILE = RealtimeProfile(
    label="Streaming ASR",
    tokens_per_second=12.5,
    audio_ms_per_token=80.0,
    target_delay_ms=2500,
    state_tokens=375,
    source="kyutai/stt-2.6b-en",
    note="Kyutai delayed-streams profile uses the published 12.5 Hz Mimi frame rate, 32 audio tokens per frame, and 2.5 s text delay; the Transformers config uses a 375-token sliding window. Capacity remains a conservative shared-trunk proxy until parallel codebook streams and the depformer are modeled separately.",
)
MOONSHINE_STREAMING_TINY_PROFILE = RealtimeProfile(
    label="Streaming ASR",
    tokens_per_second=12.5,
    audio_ms_per_token=80.0,
    target_delay_ms=80,
    state_tokens=16,
    source="UsefulSensors/moonshine-streaming-tiny",
    note="Moonshine Streaming uses 50 Hz frontend features with stride-4 downsampling and bounded 16-frame encoder windows; the first/last layers add 80 ms lookahead.",
)
MOONSHINE_STREAMING_SMALL_PROFILE = RealtimeProfile(
    label="Streaming ASR",
    tokens_per_second=12.5,
    audio_ms_per_token=80.0,
    target_delay_ms=80,
    state_tokens=16,
    source="UsefulSensors/moonshine-streaming-small",
    note="Moonshine Streaming uses 50 Hz frontend features with stride-4 downsampling and bounded 16-frame encoder windows; the first/last layers add 80 ms lookahead.",
)
MOONSHINE_STREAMING_MEDIUM_PROFILE = RealtimeProfile(
    label="Streaming ASR",
    tokens_per_second=12.5,
    audio_ms_per_token=80.0,
    target_delay_ms=80,
    state_tokens=16,
    source="UsefulSensors/moonshine-streaming-medium",
    note="Moonshine Streaming uses 50 Hz frontend features with stride-4 downsampling and bounded 16-frame encoder windows; the first/last layers add 80 ms lookahead.",
)
FUN_ASR_NANO_PROFILE = RealtimeProfile(
    label="Realtime ASR",
    tokens_per_second=6.25,
    audio_ms_per_token=160.0,
    target_delay_ms=160,
    state_tokens=8192,
    source="FunAudioLLM/Fun-ASR-Nano-2512",
    note="Fun-ASR-Nano is published as a low-latency realtime 800M ASR model; planner timing uses a conservative 160 ms streaming tick because no comparable chunk table is published. The 50-block SenseVoice encoder/adaptor is not yet separately simulated, so this is an explicitly conservative proxy.",
)
GRANITE_4_1B_SPEECH_PROFILE = RealtimeProfile(
    label="Offline ASR",
    tokens_per_second=1.0,
    audio_ms_per_token=1000.0,
    target_delay_ms=30000,
    state_tokens=8192,
    source="ibm-granite/granite-4.0-1b-speech",
    note="Granite Speech is modeled as high-throughput chunked ASR rather than native streaming; one planner tick approximates one audio-second equivalent of chunk processing.",
    streaming=False,
)
COHERE_TRANSCRIBE_PROFILE = RealtimeProfile(
    label="Offline Multilingual ASR",
    tokens_per_second=1.0,
    audio_ms_per_token=1000.0,
    target_delay_ms=30000,
    state_tokens=8192,
    source="Cohere Transcribe 03-2026 docs",
    note="Cohere publishes Transcribe as a 2B Conformer ASR model for 14 languages. No comparable local streaming table is published, so planner timing uses the same offline one-audio-second tick as other chunked ASR entries.",
    streaming=False,
)
PARAKEET_TDT_06B_V3_PROFILE = RealtimeProfile(
    label="Offline ASR",
    tokens_per_second=1.0,
    audio_ms_per_token=1000.0,
    target_delay_ms=30000,
    state_tokens=8192,
    source="nvidia/parakeet-tdt-0.6b-v3",
    note="Parakeet TDT v3 is modeled as high-throughput non-streaming ASR; one planner tick approximates one audio-second equivalent of chunk processing.",
    streaming=False,
)


# ---------------------------------------------------------------------------
# Published ASR quality: word error rate (WER) by benchmark/language, in
# percent; lower is better. Used by the "ASR Quality" plot (max streams vs
# WER). Capacity is benchmark-independent in this closed-form model, so each
# benchmark point for a model sits at the same max-stream height.
#
# Sources:
# - https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602
#   FLEURS row for 480 ms streaming WER.
# - https://mimo.xiaomi.com/mimo-v2-5-asr
#   Xiaomi-published Open ASR English average. MiMo is documented as
#   Chinese-English; no primary French WER was found.
# - https://artificialanalysis.ai/speech-to-text/batch#error-rate
#   AA-WER v2 is an English batch score over AA-AgentTalk, VoxPopuli-Cleaned-AA,
#   and Earnings22-Cleaned-AA. The closest source-backed French public rows
#   used here are CoVoST for short-form prompted speech, FLEURS for formal
#   read-speech, and MLS for long-form speech. VoxPopuli-fr would be the closest
#   French parliamentary match, but the current public model table used below
#   does not publish matching rows for this catalog's models.
# - NVIDIA, Kyutai, Moonshine, IBM, and Hugging Face model cards / Open ASR
#   leaderboard rows for the added open/self-hosted ASR models below.
# - https://huggingface.co/datasets/Steveeeeeeen/multilingual_evals
#   Hugging Face Open ASR multilingual CSV for French CoVoST/MLS/FLEURS rows
#   where the catalog model has a current public result.
# - https://arxiv.org/abs/2603.11243
#   IBM Granite 4.0 Speech paper, CommonVoice French full-AR WER.
# - https://arxiv.org/abs/2607.02770
#   Google Gemma 4 Technical Report, Tables 7-8, FLEURS English and French WER.
# French rows are filled only when a source-backed French WER was found; models
# that are English-only or have no published French table intentionally omit the
# French benchmark keys unless the whole model is marked placeholder.
# ---------------------------------------------------------------------------
ASR_WER_LANGUAGES: tuple[str, ...] = (
    "en",
    "fr_covost",
    "fr_fleurs",
    "fr_mls",
    "fr_commonvoice",
)
ASR_WER_LANGUAGE_LABELS: dict[str, str] = {
    "en": "English",
    # Legacy aggregate label kept for imports/tests; not included in
    # ASR_WER_LANGUAGES because the chart now plots the component French rows.
    "fr": "French aggregate",
    "fr_covost": "French CoVoST",
    "fr_fleurs": "French FLEURS",
    "fr_mls": "French MLS",
    "fr_commonvoice": "French Common Voice",
}
ASR_WER_LANGUAGE_SOURCES: dict[str, dict[str, str]] = {
    "voxtral-realtime-mini-4b": {
        "en": "Mistral FLEURS benchmark, 480 ms streaming delay, English WER.",
        "fr": "Legacy aggregate mean over French CoVoST, FLEURS, and MLS rows.",
        "fr_covost": "Hugging Face Open ASR multilingual eval, French CoVoST row; closest current source-backed proxy for AA-AgentTalk-style short prompted speech.",
        "fr_fleurs": "Hugging Face Open ASR multilingual eval, French FLEURS row; closest current source-backed French formal/read-speech proxy.",
        "fr_mls": "Hugging Face Open ASR multilingual eval, French MLS row; closest current source-backed long-form French proxy.",
    },
    "mimo-v2.5-asr": {
        "en": "Xiaomi MiMo General English Recognition Open ASR average WER.",
    },
    "gemma-4-e2b-asr": {
        "en": "Google Gemma 4 Technical Report Table 7, FLEURS English WER.",
        "fr_fleurs": "Google Gemma 4 Technical Report Table 7, FLEURS French WER.",
    },
    "gemma-4-e4b-asr": {
        "en": "Google Gemma 4 Technical Report Table 7, FLEURS English WER.",
        "fr_fleurs": "Google Gemma 4 Technical Report Table 7, FLEURS French WER.",
    },
    "gemma-4-12b-unified-asr": {
        "en": "Google Gemma 4 Technical Report Table 8, FLEURS English WER.",
        "fr_fleurs": "Google Gemma 4 Technical Report Table 8, FLEURS French WER.",
    },
    "nvidia-nemotron-speech-streaming-0.6b": {
        "en": "NVIDIA comparison table, HuggingFace OpenASR average WER at 0.56 s streaming latency.",
    },
    "nvidia-nemotron-3.5-asr-streaming-0.6b": {
        "en": "NVIDIA Nemotron 3.5 ASR model card, FLEURS English WER at 560 ms LangID streaming chunk.",
        "fr_fleurs": "NVIDIA Nemotron 3.5 ASR model card, FLEURS French WER at 560 ms LangID streaming chunk.",
    },
    "nvidia-parakeet-unified-0.6b": {
        "en": "NVIDIA comparison table, HuggingFace OpenASR average WER at 0.56 s streaming latency.",
    },
    "nvidia-parakeet-realtime-eou-120m": {
        "en": "NVIDIA model card, HuggingFace OpenASR average WER at 160 ms streaming setting.",
    },
    "nvidia-multitalker-parakeet-streaming-0.6b": {
        "en": "NVIDIA model card, single-speaker-mode HuggingFace OpenASR average WER.",
    },
    "kyutai-stt-1b-en-fr": {
        "en": "Placeholder: Kyutai publishes model latency and throughput, but no text WER table was found.",
        "fr": "Placeholder legacy aggregate: Kyutai publishes model latency and throughput, but no text WER table was found.",
        "fr_covost": "Placeholder French CoVoST proxy: Kyutai publishes model latency and throughput, but no text WER table was found.",
    },
    "kyutai-stt-2.6b-en": {
        "en": "Kyutai Hugging Face evaluation, HuggingFace OpenASR mean WER.",
    },
    "moonshine-streaming-tiny": {
        "en": "Useful Sensors Moonshine Streaming model card, Open ASR average WER.",
    },
    "moonshine-streaming-small": {
        "en": "Useful Sensors Moonshine Streaming model card, Open ASR average WER.",
    },
    "moonshine-streaming-medium": {
        "en": "Useful Sensors Moonshine Streaming model card, Open ASR average WER.",
    },
    "fun-asr-nano-2512": {
        "en": "Placeholder: FunAudioLLM publishes realtime capability and parameter count, but no comparable OpenASR WER was found.",
    },
    "granite-4.0-1b-speech": {
        "en": "IBM Granite model card, HuggingFace OpenASR average WER.",
        "fr": "Legacy aggregate alias for IBM Granite 4.0 Speech paper, CommonVoice French full-AR WER.",
        "fr_commonvoice": "IBM Granite 4.0 Speech paper, CommonVoice French full-AR WER.",
    },
    "nvidia-parakeet-tdt-0.6b-v3": {
        "en": "NVIDIA model card, HuggingFace OpenASR average WER.",
        "fr": "Legacy aggregate mean over French CoVoST, FLEURS, and MLS rows.",
        "fr_covost": "Hugging Face Open ASR multilingual eval, French CoVoST row; closest current source-backed proxy for AA-AgentTalk-style short prompted speech.",
        "fr_fleurs": "Hugging Face Open ASR multilingual eval, French FLEURS row; closest current source-backed French formal/read-speech proxy.",
        "fr_mls": "Hugging Face Open ASR multilingual eval, French MLS row; closest current source-backed long-form French proxy.",
    },
}
PUBLISHED_ASR_WER: dict[str, dict[str, float]] = {
    "voxtral-realtime-mini-4b": {
        "en": 4.90,
        "fr": 7.92,
        "fr_covost": 9.68,
        "fr_fleurs": 8.44,
        "fr_mls": 5.64,
    },
    "mimo-v2.5-asr": {
        "en": 5.73,
    },
    "gemma-4-e2b-asr": {
        "en": 8.0,
        "fr_fleurs": 10.1,
    },
    "gemma-4-e4b-asr": {
        "en": 6.5,
        "fr_fleurs": 8.0,
    },
    "gemma-4-12b-unified-asr": {
        "en": 6.3,
        "fr_fleurs": 8.1,
    },
    "nvidia-nemotron-speech-streaming-0.6b": {
        "en": 7.09,
    },
    "nvidia-nemotron-3.5-asr-streaming-0.6b": {
        "en": 7.99,
        "fr_fleurs": 9.45,
    },
    "nvidia-parakeet-unified-0.6b": {
        "en": 6.52,
    },
    "nvidia-parakeet-realtime-eou-120m": {
        "en": 9.30,
    },
    "nvidia-multitalker-parakeet-streaming-0.6b": {
        "en": 7.44,
    },
    "kyutai-stt-1b-en-fr": {
        "en": 7.00,
        "fr": 7.50,
        "fr_covost": 7.50,
    },
    "kyutai-stt-2.6b-en": {
        "en": 6.40,
    },
    "moonshine-streaming-tiny": {
        "en": 12.01,
    },
    "moonshine-streaming-small": {
        "en": 7.84,
    },
    "moonshine-streaming-medium": {
        "en": 6.65,
    },
    "fun-asr-nano-2512": {
        "en": 7.00,
    },
    "granite-4.0-1b-speech": {
        "en": 5.52,
        "fr": 7.15,
        "fr_commonvoice": 7.15,
    },
    "nvidia-parakeet-tdt-0.6b-v3": {
        "en": 6.34,
        "fr": 5.42,
        "fr_covost": 6.38,
        "fr_fleurs": 4.76,
        "fr_mls": 5.12,
    },
}
ASR_WER_PLACEHOLDER: frozenset[str] = frozenset(
    {
        "kyutai-stt-1b-en-fr",
        "fun-asr-nano-2512",
    }
)

# Backward-compatible aliases for older chart/import names.
ASR_WER_BENCHMARKS = ASR_WER_LANGUAGES
ASR_WER_BENCHMARK_LABELS = ASR_WER_LANGUAGE_LABELS
ASR_WER_BENCHMARK_SOURCES = ASR_WER_LANGUAGE_SOURCES
STREAMING_WER_LANGUAGES = ASR_WER_LANGUAGES
STREAMING_WER_LANGUAGE_LABELS = ASR_WER_LANGUAGE_LABELS
STREAMING_ASR_WER = PUBLISHED_ASR_WER
STREAMING_ASR_WER_PLACEHOLDER = ASR_WER_PLACEHOLDER


# ---------------------------------------------------------------------------
# Published embedding retrieval quality, in [0, 1]. Used by the "Embedding
# Quality" plot (quality vs docs/s, one dot per model). Scores are published
