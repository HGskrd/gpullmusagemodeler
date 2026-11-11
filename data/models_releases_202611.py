"""November 11 source-pinned corrections and bounded model/artifact additions.

Exact tensor storage is separate from conservative traffic and runtime estimates.
See docs/catalog-audit-2026-11-11.md for source semantics and validation.
"""

from dataclasses import replace

from .model_class import (
    DECISION_PROFILES,
    MODEL_ARTIFACT_PROFILES,
    MODEL_QUANTIZATION_PROFILES,
    DecisionProfile,
    EmbeddingProfile,
    Model,
    RealtimeProfile,
    _artifact_profile,
)
from .model_sources import OPEN_MODEL_ARCHITECTURE_SOURCES, ModelArchitectureSource
from .models_decision import DECISION_MODELS
from .models_embedding import EMBEDDING_MODELS
from .models_lfm3 import LFM3_MODELS
from .models_releases_202610 import RELEASE_MODELS
from .models_text import TEXT_MODELS
from .specs import QuantizationScheme

CAPTURED_AT = "2026-11-11"
NOVEMBER_MODELS: dict[str, Model] = {}


def _profile(
    key,
    precision,
    repo,
    rev,
    size,
    counts,
    *,
    shares=None,
    scheme=None,
    active_bytes=None,
    default=True,
    note="",
    runtime="",
):
    pair, profile = _artifact_profile(
        model_key=key,
        precision_key=precision,
        label=(
            scheme.encoding
            if scheme
            else {"bf16": "Official BF16", "fp8": "Official FP8", "nvfp4": "Official NVFP4 / FP8"}[
                precision
            ]
        ),
        source_repo=repo,
        source_revision=rev,
        total_weight_bytes=size,
        storage_format_counts=counts,
        compute_precision_shares=shares or {"bf16": 1.0},
        quantized=()
        if precision == "bf16"
        else ("Checkpoint-specific mixed operators; see scheme.",),
        retained=(
            "Retained embeddings, heads, norms and modality/auxiliary tensors as documented.",
        ),
        notes=note,
        quant_algo=scheme.encoding if scheme else precision,
        captured_at=CAPTURED_AT,
        kv_cache_format="BF16",
        scheme=scheme,
        storage_evidence="Pinned tensor headers/index; exact subset where explicitly documented.",
    )
    model = NOVEMBER_MODELS.get(key) or LFM3_MODELS.get(key) or RELEASE_MODELS.get(key)
    profile = replace(
        profile,
        active_weight_bytes_per_param_override=(
            active_bytes / (model.active_weight_params or model.active_params)
            if active_bytes and model
            else None
        ),
        runtime_evidence=runtime
        or "Kernel support and throughput require runtime-specific calibration.",
    )
    if default:
        MODEL_QUANTIZATION_PROFILES[pair] = profile
    MODEL_ARTIFACT_PROFILES[(key, profile.artifact_id)] = profile


def _source(key, repo, rev, note):
    OPEN_MODEL_ARCHITECTURE_SOURCES[key] = ModelArchitectureSource(
        "https://huggingface.co/" + repo,
        rev,
        f"Captured {CAPTURED_AT}. {note} Throughput and quality are uncalibrated.",
    )


for (
    key,
    repo,
    rev,
    total,
    stored,
    active,
    ple,
    lookup,
    layers,
    hidden,
    kv,
    local,
    producers,
    local_producers,
) in (
    (
        "ge2",
        "google/gemma-4-E2B-it",
        "4adf15748de9e1a0097b6274cee9c15836c41c37",
        5_123_178_051,
        5_123_178_979,
        2_298_639_651,
        2_348_810_240,
        17_920,
        35,
        1536,
        1,
        28,
        15,
        12,
    ),
    (
        "ge4",
        "google/gemma-4-E4B-it",
        "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c",
        7_996_156_490,
        7_996_157_418,
        4_699_496_746,
        2_818_572_288,
        21_504,
        42,
        2560,
        2,
        35,
        24,
        20,
    ),
):
    note = (
        "Effective E-size is not resident parameters. Exact complete BF16 checkpoint includes "
        "vision/audio towers, PLE and calibration scalars. PLE remains GPU-resident BF16. "
        "Text-active work excludes PLE and inactive towers; retained consumer K/V projections "
        "make compute conservative. Shared KV reserves unique producers, charges every "
        "consumer read and conservatively duplicates the producer pool across PP stages."
    )
    NOVEMBER_MODELS[key] = replace(
        TEXT_MODELS[key],
        total_params=total,
        active_params=active,
        active_weight_params=active,
        layers=layers,
        hidden_dim=hidden,
        num_heads=8,
        kv_heads=kv,
        head_dim=256,
        local_attention_layers=local,
        local_attention_window=512,
        local_attention_heads=8,
        local_attention_head_dim=256,
        local_kv_heads=kv,
        local_kv_head_dim=256,
        global_kv_heads=kv,
        global_head_dim=512,
        kv_layers=producers,
        local_kv_layers=local_producers,
        conditional_memory_params=ple,
        conditional_memory_weight_bytes=ple * 2,
        conditional_memory_lookup_bytes_per_token=lookup,
        conditional_memory_layers=tuple(range(layers)),
        attention_label=f"{local} SWA 512 + {layers - local} global · {producers} shared KV producers",
        architecture_note=note,
    )
    _profile(
        key, "bf16", repo, rev, stored * 2, {"BF16": stored}, active_bytes=active * 2, note=note
    )
    _source(key, repo, rev, note)
NOVEMBER_MODELS["g12"] = replace(TEXT_MODELS["g12"], max_context_tokens=262_144)

for key, repo, rev, size, total, active, draft_bytes, f32 in (
    (
        "q08",
        "Qwen/Qwen3.5-0.8B",
        "2fc06364715b967f1860aea9cf38778875588b17",
        1_705_977_024,
        852_985_920,
        752_393_024,
        40_905_728,
        2592,
    ),
    (
        "q2",
        "Qwen/Qwen3.5-2B",
        "15852e8c16360a2fea060d615a32b45270f8a8fc",
        4_426_488_512,
        2_213_241_664,
        1_881_825_088,
        121_656_320,
        2592,
    ),
    (
        "q4",
        "Qwen/Qwen3.5-4B",
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        9_078_538_752,
        4_539_265_536,
        4_205_751_296,
        241_199_104,
        3840,
    ),
    (
        "q9",
        "Qwen/Qwen3.5-9B",
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        18_819_635_168,
        9_409_813_744,
        8_953_803_264,
        486_581_248,
        3840,
    ),
):
    base = TEXT_MODELS[key]
    mtp = base.speculative_profiles[0]
    note = (
        "Exact target tensor subset retains vision and excludes optional MTP; loader must omit "
        "drafter until enabled. Text traffic excludes vision; acceptance remains an unmeasured prior."
    )
    NOVEMBER_MODELS[key] = replace(
        base,
        total_params=total,
        active_params=active,
        active_weight_params=active,
        architecture_note=note,
        speculative_profiles=(
            replace(
                mtp,
                exact_weight_bytes=draft_bytes,
                draft_params=draft_bytes / 2,
                active_params=draft_bytes / 2,
                note=mtp.note + " Exact BF16 resident bytes pinned.",
            ),
        ),
    )
    _profile(
        key,
        "bf16",
        repo,
        rev,
        size,
        {"BF16": total - f32, "F32": f32},
        active_bytes=active * 2 + f32 * 2,
        note=note,
    )
    _source(key, repo, rev, note)

ml4_note = (
    "72 executed DSA layers; nine reuse layer35 KV, giving63 producers.44 full indexers; "
    "other layers reuse selected indices. Single compressed MLA latent is replicated across TP. "
    "Cache bound charges RoPE64 on all producers (54 RoPE/9 NoPE in source) and256 bytes "
    "per BF16 index key; runtime FP8 cache/scales are not assumed. Full artifact includes "
    "vision and4.20GB inactive DSpark weights; speedup disabled. Vision latency unmodeled."
)
NOVEMBER_MODELS["mistral-large-4"] = Model(
    key="mistral-large-4",
    name="Mistral Large 4 1T-A52B",
    cat="Mistral",
    color="#7A3B18",
    total_params=1_053_161_553_184,
    active_params=52e9,
    is_moe=True,
    layers=72,
    hidden_dim=6144,
    num_heads=64,
    kv_heads=1,
    head_dim=128,
    attention_value_head_dim=128,
    is_mla=True,
    mla_kv_dim=512,
    mla_rope_dim=64,
    mla_tp_supported=True,
    attention_layers=72,
    kv_layers=63,
    local_kv_layers=0,
    sparse_attention_top_k=2048,
    sparse_indexer_heads=32,
    sparse_indexer_head_dim=128,
    sparse_indexer_layers=44,
    sparse_indexer_compression_ratio=1,
    sparse_indexer_cache_bytes_per_token=256,
    moe_intermediate_dim=6144,
    moe_routed_experts=128,
    moe_active_experts=4,
    moe_shared_experts=1,
    max_context_tokens=1_048_576,
    native_precision="fp8",
    quality_confidence=0.0,
    native_precision_label="Official FP8 + BF16 islands",
    extra_capabilities=frozenset({"images", "reasoning"}),
    attention_label="72 DSA layers · 63 MLA cache producers · 44 indexers",
    architecture_note=ml4_note,
    native_precision_note=ml4_note,
)
for precision, repo, rev, size, counts, shares, active in (
    (
        "fp8",
        "mistralai/Mistral-Large-4-1T-A52B",
        "af26e7869af601441d5cdf75a1272469ddc3d059",
        1_060_548_427_328,
        {"F32": 63_844_608, "F8_E4M3": 1_046_030_057_472, "BF16": 7_131_495_712},
        {"fp8": 0.93642623069059, "bf16": 0.06357376930941001},
        55_677_023_808,
    ),
    (
        "nvfp4",
        "mistralai/Mistral-Large-4-1T-A52B-NVFP4",
        "bb6d00fe249b0e876f0cffc6b85014be02de3e68",
        610_030_457_408,
        {"F32": 1_082_880, "BF16": 7_131_495_712, "F8_E4M3": 81_172_365_312, "U8": 514_590_769_152},
        {"nvfp4": 0.6145071747484334, "fp8": 0.3219190559421566, "bf16": 0.06357376930941001},
        41_598_337_248,
    ),
):
    _profile(
        "mistral-large-4",
        precision,
        repo,
        rev,
        size,
        counts,
        shares=shares,
        active_bytes=active,
        note=ml4_note,
    )
_source(
    "mistral-large-4",
    "mistralai/Mistral-Large-4-1T-A52B",
    "af26e7869af601441d5cdf75a1272469ddc3d059",
    ml4_note,
)

for size_label, rev, params, size, dim in (
    ("0.6b", "6d1fddab0a18478acc0a51edd1e81333fe441095", 595_370_176, 2_381_480_704, 1024),
    ("9b", "dab89ff203fc943c4ca16e51c5882ff92790d758", 8_401_083_632, 33_604_334_528, 2048),
):
    key = f"pplx-embed-v2-context-{size_label}"
    repo = f"perplexity-ai/{key}"
    note = (
        "One full-document hybrid forward emits contextual chunk vectors.512 formatted positions "
        "per chunk is an explicit workload assumption, not a vendor chunk limit. Prefixes/separators "
        "consume input positions. Source neither autochunks nor truncates. FP32 arrays contain "
        "INT8-valued outputs; packed INT8 storage is not assumed. Checkpoint residency is FP32; "
        "BF16 compute is an uncalibrated serving proxy. Image execution unmodeled."
    )
    NOVEMBER_MODELS[key] = replace(
        EMBEDDING_MODELS[f"pplx-embed-v2-late-{size_label}"],
        key=key,
        name=f"Perplexity Embed v2 Context {size_label.upper()}",
        total_params=params,
        active_params=params,
        prefill_active_params=params,
        active_weight_params=params,
        max_context_tokens=262_144,
        bf16_weight_bytes_per_param=4.0,
        quality_confidence=0.0,
        architecture_note=note,
        native_precision_label="FP32 checkpoint / BF16 compute proxy",
        native_precision_note=note,
        attention_label="Noncausal global + causal linear · contextual chunk pooling",
        embedding_profile=EmbeddingProfile(
            "Contextual document chunks",
            "single",
            dim,
            262_144,
            repo,
            note,
            vector_bytes_per_elem=4,
            storage_format="FP32 (INT8-valued)",
            pooling="joint-document chunk pooling",
            contextual_chunk_tokens=512,
        ),
    )
    _profile(
        key,
        "bf16",
        repo,
        rev,
        size,
        {"F32": params},
        scheme=QuantizationScheme("FP32 storage", (), "none"),
        active_bytes=size,
        note=note,
    )
    _source(key, repo, rev, note)

ctc_key = "granite-speech-5.0-470m-turboctc"
ctc_repo = "ibm-granite/" + ctc_key
ctc_rev = "91dcb0605daf65f88dbb04fef015c423241d941d"
ctc_note = (
    "Encoder-only multilingual CTC, zero autoregressive decoder/cache.16kHz input,100Hz features "
    "stacked to50Hz; first two blocks subsample to25/12.5Hz. Staged bound charges subsampling "
    "blocks at input rate. Tied CTC output executes twice but is resident once. Feature extraction, "
    "softmax and exact runtime buffering unmodeled. Seven languages; CER is not WER."
)
NOVEMBER_MODELS[ctc_key] = Model(
    key=ctc_key,
    name="Granite Speech 5.0 470M TurboCTC Multilingual",
    cat="Audio",
    color="#2875AA",
    total_params=472_993_792,
    active_params=472_993_792,
    is_moe=False,
    layers=16,
    hidden_dim=1024,
    num_heads=8,
    kv_heads=8,
    head_dim=128,
    is_mla=False,
    attention_layers=0,
    kv_layers=0,
    max_context_tokens=262_144,
    capabilities_override=frozenset(),
    quality_confidence=0.0,
    architecture_note=ctc_note,
    attention_label="16 audio encoder blocks · CTC · no transcript KV",
    realtime_profile=RealtimeProfile(
        label="Offline multilingual CTC",
        tokens_per_second=12.5,
        audio_ms_per_token=80,
        target_delay_ms=30_000,
        state_tokens=0,
        source=ctc_repo,
        note=ctc_note,
        audio_encoder_params=472_993_792,
        audio_attention_layers=16,
        audio_attention_heads=8,
        audio_attention_head_dim=128,
        audio_attention_window=128,
        streaming=False,
        audio_chunk_seconds=30,
        audio_frame_rate=50,
        audio_encoder_stages=(
            (27_772_032, 50.0, 1),
            (27_443_328, 25.0, 1),
            (434_572_032, 12.5, 14),
        ),
    ),
)
_profile(
    ctc_key,
    "bf16",
    ctc_repo,
    ctc_rev,
    946_249_856,
    {"BF16": 472_862_720, "F32": 131_072, "I64": 16},
    note=ctc_note,
)
_source(ctc_key, ctc_repo, ctc_rev, ctc_note)

decider_key = "pplx-decider-v1.2"
decider_repo = "perplexity-ai/" + decider_key
decider_rev = "cce0f3f80033e9d89d80831d5281fd72124aacce"
decider_note = (
    "Single joint-question forward; noncausal global attention and causal linear state. "
    "Zero generated tokens. Includes separate255x5120 BF16 readout.262K architectural positions "
    "versus32K training coverage; long-context transfer uncalibrated. Splitting repeats state "
    "and changes joint-question semantics. No FP8 fast-path speedup assumed."
)
NOVEMBER_MODELS[decider_key] = replace(
    DECISION_MODELS["pplx-decider-v1.1-27b"],
    key=decider_key,
    name="Perplexity Decider v1.2 27B",
    total_params=26_086_635_760,
    active_params=26_086_635_760,
    attention_label="16 noncausal full + 48 causal linear · joint forward",
    architecture_note=decider_note,
    quality_confidence=0.0,
)
DECISION_PROFILES[decider_key] = DecisionProfile(
    262_144,
    decider_repo,
    decider_note,
    causal_attention=False,
    forward_per_question=False,
    max_options=255,
)
_profile(
    decider_key,
    "bf16",
    decider_repo,
    decider_rev,
    52_173_271_520,
    {"BF16": 26_086_635_760},
    note=decider_note,
)
_source(decider_key, decider_repo, decider_rev, decider_note)

_profile(
    "lfm3-vl-2b",
    "nvfp4",
    "LiquidAI/LFM3-VL-2B-MLX-4bit",
    "d0d7edaf1b462161720898e46687c6fdaae6caee",
    1_590_866_944,
    {"BF16": 297_794_048, "U32": 248_819_712},
    active_bytes=1_212_106_240,
    scheme=QuantizationScheme("MLX affine INT4", (32,), "BF16", "BF16 additive bias"),
    note="MLX-VLM >=0.7.11 on Apple Silicon. Group32 affine INT4 text linears, INT8 tied "
    "embedding and retained BF16 vision. BF16 compute proxy, not NVFP4 arithmetic. "
    "Source quantization-aware distillation is not independent artifact quality calibration.",
    runtime="MLX-VLM >=0.7.11 on Apple Silicon; other serving runtimes unsupported.",
)
for precision, suffix, rev, size, counts, scheme, shares, traffic in (
    (
        "fp8",
        "FP8",
        "0b1acaa19bae10fec482600ada7c9defcb7c4884",
        35_516_330_752,
        {"BF16": 2395478400, "F8_E4M3": 30725373952},
        QuantizationScheme("E4M3", (128, 128), "BF16"),
        {"fp8": 0.9625201195677168, "bf16": 0.037479880432283286},
        33_123_494_656,
    ),
    (
        "fp8",
        "INT8",
        "879049b69c5f64384f3067095924b63e9125e84f",
        35_992_669_888,
        {"I64": 728, "I32": 7681343488, "BF16": 2633645056},
        QuantizationScheme("INT8", (128,), "BF16"),
        {"bf16": 1.0},
        33_599_833_792,
    ),
    (
        "nvfp4",
        "INT4",
        "d834b414cf1c47e42640de33c147b7cf8a9321ef",
        22_550_315_872,
        {"I32": 3960693464, "BF16": 3353771008},
        QuantizationScheme("INT4", (32,), "BF16", "INT8 packed in I32"),
        {"bf16": 1.0},
        20_157_479_776,
    ),
):
    _profile(
        "nanbeige43-33b",
        precision,
        "Nanbeige/Nanbeige4.3-33B-" + suffix,
        rev,
        size,
        counts,
        shares=shares,
        scheme=scheme,
        active_bytes=traffic,
        default=suffix != "INT8",
        note="Exact mixed tensor storage; embeddings/head retained BF16. "
        "Integer weight-only formats use BF16 compute, not FP4 peaks. Traffic is a "
        "header-derived bound; published BF16 quality is not a quantized quality measurement.",
    )
