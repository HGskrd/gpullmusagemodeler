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
    SpeculativeProfile,
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
CAPTURED_AT_NOV22 = "2026-11-22"
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
    captured_at=CAPTURED_AT,
    label=None,
):
    pair, profile = _artifact_profile(
        model_key=key,
        precision_key=precision,
        label=label
        or (
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
        captured_at=captured_at,
        kv_cache_format="BF16",
        scheme=scheme,
        storage_evidence="Pinned tensor headers/index; exact subset where explicitly documented.",
    )
    model = (
        NOVEMBER_MODELS.get(key)
        or LFM3_MODELS.get(key)
        or RELEASE_MODELS.get(key)
        or TEXT_MODELS.get(key)
    )
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


def _source(key, repo, rev, note, *, captured_at=CAPTURED_AT):
    OPEN_MODEL_ARCHITECTURE_SOURCES[key] = ModelArchitectureSource(
        "https://huggingface.co/" + repo,
        rev,
        f"Captured {captured_at}. {note} Throughput and quality are uncalibrated.",
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

# ---------------------------------------------------------------------------
# 2026-11-22 biweekly audit additions (source-pinned; no weights downloaded)
# ---------------------------------------------------------------------------

_ink2s_note = (
    "48-layer multimodal MoE (594.05B total / 17B active; 8 of 384 routed experts, "
    "intermediate 2304, no shared experts). Hybrid 40 SWA-1024 + 8 global layers with "
    "Q48/KV8/head_dim 128, Block AttnRes (6 layers/block), SConv kernel 4, 1M context. "
    "Checkpoint includes 23-layer ViT vision tower and 5-layer MTP/DSpark tensors; "
    "speculative speedup and vision latency remain unmodeled. Compute shares are "
    "conservative operator-role proxies."
)
NOVEMBER_MODELS["inkling-2-small"] = Model(
    key="inkling-2-small",
    name="Inkling 2 Small 594B-A17B",
    cat="Thinking Machines",
    color="#D63B59",
    total_params=594_050_672_720,
    active_params=17e9,
    active_weight_params=17e9,
    is_moe=True,
    layers=48,
    hidden_dim=4608,
    num_heads=48,
    kv_heads=8,
    head_dim=128,
    is_mla=False,
    attention_layers=48,
    kv_layers=48,
    local_attention_layers=40,
    local_kv_layers=40,
    local_attention_window=1024,
    local_attention_heads=48,
    local_kv_heads=8,
    local_kv_head_dim=128,
    global_kv_heads=8,
    global_head_dim=128,
    attention_residual_block_size=6,
    moe_intermediate_dim=2304,
    moe_routed_experts=384,
    moe_active_experts=8,
    moe_shared_experts=0,
    max_context_tokens=1_048_576,
    native_precision="nvfp4",
    native_precision_label="Official NVFP4 / MXFP8 · 390.1 GB",
    native_precision_note=_ink2s_note,
    extra_capabilities=frozenset({"images", "reasoning"}),
    quality_confidence=0.0,
    attention_label="40 SWA 1024 + 8 global · Block AttnRes 6 · 8/384 MoE",
    architecture_note=_ink2s_note,
)
_profile(
    "inkling-2-small",
    "nvfp4",
    "thinkingmachines/Inkling-2-Small",
    "885d78bfca61a7e9a498e33cfa7ad35850533b8c",
    390_056_448_320,
    None,
    shares={"nvfp4": 0.82, "fp8": 0.16, "bf16": 0.02},
    note=_ink2s_note,
    captured_at=CAPTURED_AT_NOV22,
    label="Official NVFP4 / MXFP8",
)
_profile(
    "inkling-2-small",
    "fp8",
    "thinkingmachines/Inkling-2-Small-MXFP8",
    "14f2cb29a5d0f4c1fda8f645fa339e936c2147a2",
    619_379_896_320,
    {
        "F32": 534_528,
        "BF16": 6_981_765_120,
        "F8_E4M3": 587_068_342_272,
        "U8": 18_345_885_696,
    },
    shares={"fp8": 0.96, "bf16": 0.04},
    scheme=QuantizationScheme("MXFP8", (1, 32), "UE8M0"),
    note="Exact 48-shard MXFP8 checkpoint (619.380 GB indexed; BF16 embeddings/head/vision/attention, MXFP8 experts).",
    captured_at=CAPTURED_AT_NOV22,
    label="Official MXFP8",
)
_source(
    "inkling-2-small",
    "thinkingmachines/Inkling-2-Small",
    "885d78bfca61a7e9a498e33cfa7ad35850533b8c",
    _ink2s_note,
    captured_at=CAPTURED_AT_NOV22,
)

_ink2m_note = (
    "Looped multimodal MoE: 24 physical layers (6 prelude + 12 core looped 3 passes + 6 coda = "
    "48 logical executed layers; 40 SWA-1024 + 8 global logical visits). Per vLLM recipe, each "
    "loop visit maintains its own KV and short-conv state (48 logical KV layers total). "
    "26.39B physical total / 3.4B unique active / 5.4B effective active across 3 passes "
    "(parameter_passes=5.4/3.4). Mixed NVFP4/MXFP8 checkpoint (19.526 GB indexed) includes "
    "23-layer ViT vision tower and 3-layer DSpark/MTP weights; speculative speedup, short-conv "
    "state, inter-pass rank-64 gating, and vision latency are unmodeled proxies."
)
NOVEMBER_MODELS["inkling-2-mini"] = Model(
    key="inkling-2-mini",
    name="Inkling 2 Mini 26B-A3.4B (3 loops)",
    cat="Thinking Machines",
    color="#E6647B",
    total_params=26_388_646_440,
    active_params=3.4e9,
    active_weight_params=3.4e9,
    parameter_passes=5.4 / 3.4,
    is_moe=True,
    layers=48,
    hidden_dim=2304,
    num_heads=24,
    kv_heads=8,
    head_dim=128,
    is_mla=False,
    attention_layers=48,
    kv_layers=48,
    local_attention_layers=40,
    local_kv_layers=40,
    local_attention_window=1024,
    local_attention_heads=24,
    local_kv_heads=8,
    local_kv_head_dim=128,
    global_kv_heads=8,
    global_head_dim=128,
    attention_residual_block_size=6,
    moe_intermediate_dim=1152,
    moe_routed_experts=128,
    moe_active_experts=8,
    moe_shared_experts=0,
    max_context_tokens=1_048_576,
    native_precision="nvfp4",
    native_precision_label="Official NVFP4 / MXFP8 · 19.5 GB",
    native_precision_note=_ink2m_note,
    extra_capabilities=frozenset({"images", "reasoning"}),
    quality_confidence=0.0,
    attention_label="3 loops · 48 logical / 24 physical layers · 40 SWA 1024 + 8 global",
    architecture_note=_ink2m_note,
)
_profile(
    "inkling-2-mini",
    "nvfp4",
    "thinkingmachines/Inkling-2-Mini",
    "b85c3cfa615fc413cc9af3379f491b963f3b586a",
    19_525_870_752,
    None,
    shares={"nvfp4": 0.77, "fp8": 0.16, "bf16": 0.07},
    note=_ink2m_note,
    captured_at=CAPTURED_AT_NOV22,
    label="Official NVFP4 / MXFP8",
)
_source(
    "inkling-2-mini",
    "thinkingmachines/Inkling-2-Mini",
    "b85c3cfa615fc413cc9af3379f491b963f3b586a",
    _ink2m_note,
    captured_at=CAPTURED_AT_NOV22,
)

_decider13_key = "pplx-decider-v1.3"
_decider13_repo = "perplexity-ai/" + _decider13_key
_decider13_rev = "9c2ebb27470d3dec2ad7ab727727e40d142c7213"
_decider13_note = (
    "Qwen3.8-27B hybrid backbone (16 noncausal full + 48 causal linear); default separate "
    "decision sequences and joint Locate span-extraction questions. Max default length 32,768 "
    "formatted positions; Locate requires text-only state (<=128 questions) and disables "
    "fast=True. Conservatively keeps FP32 Locate adapter (1,867,665,408 B) and span head "
    "(13,171,348 B) resident alongside BF16 backbone (52,170,660,320 B) and BF16 readout "
    "(2,611,200 B). BF16 compute is a planning proxy; CPU span decoding and vision latency "
    "are unmodeled."
)
NOVEMBER_MODELS[_decider13_key] = replace(
    NOVEMBER_MODELS["pplx-decider-v1.2"],
    key=_decider13_key,
    name="Perplexity Decider v1.3 + Locate (27B)",
    total_params=26_556_844_949,
    active_params=26_556_844_949,
    active_weight_params=26_556_844_949,
    max_context_tokens=32_768,
    attention_label="16 noncausal full + 48 causal linear · decisions / Locate spans",
    architecture_note=_decider13_note,
    native_precision_label="BF16 backbone + FP32 Locate adapter/head",
    native_precision_note=_decider13_note,
    quality_confidence=0.0,
)
DECISION_PROFILES[_decider13_key] = DecisionProfile(
    32_768,
    f"https://huggingface.co/{_decider13_repo}/blob/{_decider13_rev}/README.md",
    _decider13_note,
    causal_attention=False,
    forward_per_question=True,
    max_options=255,
    task_primitives=("choice", "boolean", "score", "span_extraction"),
)
_profile(
    _decider13_key,
    "bf16",
    _decider13_repo,
    _decider13_rev,
    54_054_108_276,
    {"BF16": 26_086_635_760, "F32": 470_209_189},
    note=_decider13_note,
    captured_at=CAPTURED_AT_NOV22,
    label="Native BF16 + resident FP32 Locate",
)
_source(
    _decider13_key,
    _decider13_repo,
    _decider13_rev,
    _decider13_note,
    captured_at=CAPTURED_AT_NOV22,
)

_sov_key = "sov-transcribe-1"
_sov_repo = "salesforce-sovereign-intelligence/SOV-Transcribe-1"
_sov_rev = "505d8fae243924c0d8acdc622f7f0abab344eba3"
_sov_note = (
    "Non-streaming multilingual ASR combining Whisper large-v3 audio encoder (32L, d=1280, "
    "20 heads, ~637M params) + projector + Gemma 4 26B-A4B MoE decoder (30L: 25 SWA-1024 + "
    "5 global, 8/128 routed experts, shared K=V on global layers). 16 kHz audio yields 375 "
    "audio tokens per 30 s window (12.5 tok/s, 80 ms/tok); tested up to 30 min. Companion "
    "5-layer BF16 DFlash drafter (859.4 MB) uses an explicit conservative unmeasured acceptance "
    "prior (publisher reports ~2-3x wall-clock speedup on single requests, not a measured tau). "
    "No comparable inline WER score published in card; capacity-only surface."
)
NOVEMBER_MODELS[_sov_key] = Model(
    key=_sov_key,
    name="Salesforce SOV-Transcribe-1 26.6B-A4.6B",
    cat="Audio",
    color="#00A1E0",
    total_params=26_630_656_030,
    active_params=4.6e9,
    active_weight_params=4.6e9,
    is_moe=True,
    layers=30,
    hidden_dim=2816,
    num_heads=16,
    kv_heads=2,
    head_dim=512,
    is_mla=False,
    attention_layers=30,
    kv_layers=30,
    local_attention_layers=25,
    local_kv_layers=25,
    local_attention_window=1024,
    local_attention_heads=16,
    local_attention_head_dim=256,
    local_kv_heads=8,
    local_kv_head_dim=256,
    global_kv_heads=2,
    global_head_dim=512,
    shared_key_value=True,
    moe_routed_experts=128,
    moe_active_experts=8,
    max_context_tokens=262_144,
    native_precision="bf16",
    native_precision_label="Official BF16 · 53.3 GB",
    native_precision_note=_sov_note,
    capabilities_override=frozenset(),
    quality_confidence=0.0,
    attention_label="Whisper large-v3 encoder + Gemma 4 26B-A4B (25 SWA 1k + 5 global)",
    architecture_note=_sov_note,
    realtime_profile=RealtimeProfile(
        label="Offline multilingual SALM ASR",
        tokens_per_second=12.5,
        audio_ms_per_token=80.0,
        target_delay_ms=30_000,
        state_tokens=375,
        source=_sov_repo,
        note=_sov_note,
        audio_encoder_params=637e6,
        audio_attention_layers=32,
        audio_attention_heads=20,
        audio_attention_head_dim=64,
        streaming=False,
        audio_chunk_seconds=30.0,
        audio_frame_rate=50.0,
        audio_embedding_rate=12.5,
        decoder_params=4.0e9,
    ),
    speculative_profiles=(
        SpeculativeProfile(
            label="SOV-Transcribe-1 DFlash (430M, prior)",
            method="dflash",
            draft_params=429_689_088,
            draft_layers=5,
            parallel_draft=True,
            default_k=15,
            acceptance_alpha=0.6,
            kv_overhead=0.0,
            source="https://huggingface.co/salesforce-sovereign-intelligence/SOV-Transcribe-1-DFlash/tree/8c0b3f3a98507d2e3dd35b1bee0f147fa3bded05",
            note="Captured 2026-11-22. Official 5-layer BF16 DFlash drafter (859,378,176 B, block_size=16, 15 drafted tokens). Publisher reports ~2-3x single-request wall-clock speedup without a measured mean acceptance length; alpha=0.60 is an explicitly conservative unmeasured planning prior.",
            exact_weight_bytes=859_378_176,
            active_params=429_689_088,
            supported_ks=(15,),
            proposed_tokens=15,
            verify_anchor_positions=1,
            draft_compute_precision="bf16",
            draft_kv_heads=8,
            draft_head_dim=128,
            draft_kv_bytes=2.0,
            draft_window=2048,
        ),
    ),
)
_profile(
    _sov_key,
    "bf16",
    _sov_repo,
    _sov_rev,
    53_283_115_580,
    {"BF16": 26_619_754_270, "F32": 10_901_760},
    note=_sov_note,
    captured_at=CAPTURED_AT_NOV22,
)
_source(_sov_key, _sov_repo, _sov_rev, _sov_note, captured_at=CAPTURED_AT_NOV22)

_nt_key = "nvidia-nemotron-3.5-transcribe-30b-a4b"
_nt_repo = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
_nt_rev = "c17667641642c4265ff2b7037dd101b7d4567634"
_nt_note = (
    "Offline 40-language SALM ASR combining an ~800M Parallel Expert Encoder (32-layer d=1280 "
    "Transformer ASR encoder + speaker diarization encoder) with the Nemotron-3.5-Lightning "
    "hybrid Mamba-2/Transformer MoE backbone (52 layers: 23 Mamba-2 + 6 GQA Q32/KV2/d128 + "
    "23 MoE 6/128 routed + 1 shared). 33.70B total / 4.3B active (3.5B LLM + ~0.8B encoder); "
    "65,536-token context, recordings up to 1 hour in 30 s encoder chunks. Released in BF16 "
    "(67.495 GB), ModelOpt FP8 (34.667 GB), and NVFP4 (21.553 GB). Bundled 2-layer MTP "
    "speedup is left disabled pending measured acceptance calibration."
)
NOVEMBER_MODELS[_nt_key] = Model(
    key=_nt_key,
    name="NVIDIA Nemotron 3.5 Transcribe 30B-A4B",
    cat="Audio",
    color="#76B900",
    total_params=33_697_781_029,
    active_params=4.3e9,
    active_weight_params=4.3e9,
    is_moe=True,
    layers=52,
    hidden_dim=2688,
    num_heads=32,
    kv_heads=2,
    head_dim=128,
    is_mla=False,
    attention_layers=6,
    kv_layers=6,
    linear_attention_layers=23,
    linear_attention_heads=64,
    linear_attention_head_dim=64,
    linear_attention_k_heads=64,
    linear_attention_k_head_dim=64,
    linear_attention_conv_kernel=4,
    moe_intermediate_dim=1856,
    moe_routed_experts=128,
    moe_active_experts=6,
    moe_shared_experts=1,
    max_context_tokens=65_536,
    native_precision="bf16",
    native_precision_label="Official BF16 · 67.5 GB",
    native_precision_note=_nt_note,
    capabilities_override=frozenset(),
    quality_confidence=0.0,
    attention_label="32L audio encoder + 23 Mamba-2 / 6 GQA / 23 MoE (6/128)",
    architecture_note=_nt_note,
    realtime_profile=RealtimeProfile(
        label="Offline 40-language SALM ASR",
        tokens_per_second=12.5,
        audio_ms_per_token=80.0,
        target_delay_ms=30_000,
        state_tokens=375,
        source=_nt_repo,
        note=_nt_note,
        audio_encoder_params=800e6,
        audio_attention_layers=32,
        audio_attention_heads=10,
        audio_attention_head_dim=128,
        streaming=False,
        audio_chunk_seconds=30.0,
        audio_frame_rate=100.0,
        audio_embedding_rate=12.5,
        decoder_params=3.5e9,
    ),
)
_profile(
    _nt_key,
    "bf16",
    _nt_repo,
    _nt_rev,
    67_495_450_710,
    {"BF16": 33_646_888_256, "F32": 50_260_472, "U8": 632_302, "I64": 1},
    note=_nt_note,
    captured_at=CAPTURED_AT_NOV22,
)
_profile(
    _nt_key,
    "fp8",
    "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8",
    "2b6209f89bec0da576d69183bfdabb3b6c05ade2",
    34_666_946_014,
    None,
    shares={"fp8": 0.94, "bf16": 0.06},
    note="Official ModelOpt FP8 checkpoint (34.667 GB indexed; retains BF16 embeddings/gates/conv1d/lm_head/audio encoder components).",
    captured_at=CAPTURED_AT_NOV22,
)
_profile(
    _nt_key,
    "nvfp4",
    "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4",
    "66c6adc4020179567074d0399cda2a4eeb2bb347",
    21_553_247_662,
    None,
    shares={"nvfp4": 0.78, "fp8": 0.14, "bf16": 0.08},
    note="Official ModelOpt NVFP4 checkpoint (21.553 GB indexed; mixed NVFP4/FP8/BF16 with retained BF16 encoder/head/gates).",
    captured_at=CAPTURED_AT_NOV22,
)
_source(_nt_key, _nt_repo, _nt_rev, _nt_note, captured_at=CAPTURED_AT_NOV22)

# Pinned NVIDIA NVFP4 artifacts for existing local models
_profile(
    "nemotron35-super-vl",
    "nvfp4",
    "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4",
    "ae1368329b1fc76b71534c5d04b87b35d9cc6e15",
    75_010_189_968,
    None,
    shares={"nvfp4": 0.80, "fp8": 0.12, "bf16": 0.08},
    note="Official NVIDIA QAD NVFP4 release (75.010 GB indexed across 88 shards; mixed NVFP4 experts, FP8 attention/Mamba projections, BF16 vision/embeddings/head). Compute shares are conservative operator-role proxies.",
    captured_at=CAPTURED_AT_NOV22,
    label="Official NVFP4 (QAD)",
)
_profile(
    "qwen38-27b",
    "nvfp4",
    "nvidia/Qwen3.8-27B-NVFP4-QAD",
    "cd46ab575a5a687f0bc3e54251016fc2cc1a6aef",
    19_898_956_392,
    None,
    shares={"nvfp4": 0.80, "fp8": 0.10, "bf16": 0.10},
    default=False,
    note="Official NVIDIA ModelOpt QAD NVFP4 release (19.899 GB indexed across 64 shards; distinct from earlier 21.921 GB PTQ NVFP4 artifact). Compute shares are conservative operator-role proxies.",
    captured_at=CAPTURED_AT_NOV22,
    label="NVIDIA NVFP4 (QAD)",
)

# Official NVIDIA external draft models for Qwen 3.8 27B (released Nov 17, 2026)
NOVEMBER_MODELS["qwen38-27b"] = replace(
    TEXT_MODELS["qwen38-27b"],
    speculative_profiles=(
        *TEXT_MODELS["qwen38-27b"].speculative_profiles,
        SpeculativeProfile(
            label="NVIDIA Qwen3.5-4B Draft (SPEED-Bench)",
            method="draft_model",
            draft_params=4_205_751_296,
            draft_layers=32,
            parallel_draft=False,
            default_k=15,
            acceptance_alpha=0.8298540648911450,
            kv_overhead=0.0,
            source="https://huggingface.co/nvidia/Qwen3.5-4B-Qwen3.8-27B-Draft/tree/bc5ad93d03d96a9c1ef49748e00ec5130a7080de",
            note="Captured 2026-11-22. Official NVIDIA BF16 standalone draft model (8,411,502,592 B, 4.21B params). Alpha fitted to SPEED-Bench qualitative mean tau=5.58 at block size 16 (long-context 32k tau=5.11); trained at 12,288 context. Transfer to other workloads requires calibration.",
            exact_weight_bytes=8_411_502_592,
            active_params=4_205_751_296,
            supported_ks=(15,),
            proposed_tokens=15,
            verify_anchor_positions=1,
            draft_compute_precision="bf16",
            draft_kv_heads=4,
            draft_head_dim=256,
            draft_kv_bytes=2.0,
        ),
        SpeculativeProfile(
            label="NVIDIA Qwen3.5-0.8B Draft (SPEED-Bench)",
            method="draft_model",
            draft_params=752_393_024,
            draft_layers=24,
            parallel_draft=False,
            default_k=15,
            acceptance_alpha=0.7473120036366827,
            kv_overhead=0.0,
            source="https://huggingface.co/nvidia/Qwen3.5-0.8B-Qwen3.8-27B-Draft/tree/6afca85a17ef55a7b89ec0be38cf62983f68d23b",
            note="Captured 2026-11-22. Official NVIDIA BF16 standalone draft model (1,504,786,048 B, 0.75B params). Alpha fitted to SPEED-Bench qualitative mean tau=3.92 at block size 16 (long-context 32k tau=3.60); trained at 12,288 context. Transfer to other workloads requires calibration.",
            exact_weight_bytes=1_504_786_048,
            active_params=752_393_024,
            supported_ks=(15,),
            proposed_tokens=15,
            verify_anchor_positions=1,
            draft_compute_precision="bf16",
            draft_kv_heads=2,
            draft_head_dim=256,
            draft_kv_bytes=2.0,
        ),
        SpeculativeProfile(
            label="NVIDIA Llama-3.2-1B Draft (SPEED-Bench)",
            method="draft_model",
            draft_params=1_481_705_472,
            draft_layers=16,
            parallel_draft=False,
            default_k=15,
            acceptance_alpha=0.7087887720283081,
            kv_overhead=0.0,
            source="https://huggingface.co/nvidia/Llama-3.2-1B-Qwen3.8-27B-Draft/tree/a01c05f41e2cdaf1920433647f69021e03aca523",
            note="Captured 2026-11-22. Official NVIDIA BF16 cross-family standalone draft model (2,963,410,944 B, 1.48B params, 128K context). Alpha fitted to SPEED-Bench qualitative mean tau=3.42 at block size 16 (long-context 32k tau=3.07). Transfer to other workloads requires calibration.",
            exact_weight_bytes=2_963_410_944,
            active_params=1_481_705_472,
            supported_ks=(15,),
            proposed_tokens=15,
            verify_anchor_positions=1,
            draft_compute_precision="bf16",
            draft_kv_heads=8,
            draft_head_dim=64,
            draft_kv_bytes=2.0,
        ),
    ),
)
