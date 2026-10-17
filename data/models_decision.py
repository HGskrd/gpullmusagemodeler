"""Pinned single-forward decision checkpoints; no autoregressive generation."""

from .model_class import (
    DECISION_PROFILES,
    MODEL_QUANTIZATION_PROFILES,
    DecisionProfile,
    Model,
    _artifact_profile,
)

DECISION_MODELS: dict[str, Model] = {}

DECISION_MODELS["d1-3b"] = Model(
    key="d1-3b",
    name="Liquid d1 3B",
    cat="LiquidAI",
    color="#0F766E",
    total_params=3123483888,
    active_params=3123483888,
    is_moe=False,
    layers=30,
    num_heads=32,
    kv_heads=8,
    head_dim=64,
    is_mla=False,
    hidden_dim=2048,
    attention_layers=8,
    kv_layers=8,
    max_context_tokens=32768,
    quality_confidence=0.0,
    capabilities_override=frozenset(),
    attention_label="8 full attention + 22 linear/conv · single forward",
    native_precision_label="Native BF16",
    architecture_note="Text-only throughput proxy; vision weights included, image preprocessing unmodeled. Capacity is a closed-form text-only estimate, not a measured benchmark. Quality uncalibrated; no generation economics.",
)
DECISION_PROFILES["d1-3b"] = DecisionProfile(
    32768,
    "https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/README.md",
    "Text-only throughput proxy; vision weights included, image preprocessing unmodeled.",
    causal_attention=True,
)
MODEL_QUANTIZATION_PROFILES.update(
    [
        _artifact_profile(
            model_key="d1-3b",
            precision_key="bf16",
            label="Native checkpoint",
            source_repo="LiquidAI/d1-3B",
            source_revision="051bcc464b01b9f92942b364d9586b0ef5912432",
            total_weight_bytes=6246967776,
            compute_precision_shares={"bf16": 1.0},
            quantized=(),
            retained=("Entire native checkpoint; modality components and heads included.",),
            notes="Pinned Hub physical tensor dtype inventory. BF16 compute is an uncalibrated proxy; native FP32 tensors retain their full storage.",
            quant_algo="Unquantized",
            kv_cache_format="BF16",
            captured_at="2026-10-17",
        )
    ]
)

DECISION_MODELS["d1-omni-600m"] = Model(
    key="d1-omni-600m",
    name="Liquid d1 Omni 600M",
    cat="LiquidAI",
    color="#7C3AED",
    total_params=587161089,
    active_params=587161089,
    is_moe=False,
    layers=16,
    num_heads=16,
    kv_heads=8,
    head_dim=64,
    is_mla=False,
    hidden_dim=1024,
    attention_layers=6,
    kv_layers=6,
    max_context_tokens=16384,
    quality_confidence=0.0,
    capabilities_override=frozenset(),
    attention_label="6 full attention + 10 linear/conv · single forward",
    native_precision_label="Native FP32 storage · BF16 compute proxy",
    architecture_note="Experimental. Released combined-position cap 16,384; text-only BF16 compute proxy with native FP32 storage. Image/audio encoder latency unmodeled. Capacity is a closed-form text-only estimate, not a measured benchmark. Quality uncalibrated; no generation economics.",
)
DECISION_PROFILES["d1-omni-600m"] = DecisionProfile(
    16384,
    "https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/README.md",
    "Experimental. Released combined-position cap 16,384; text-only BF16 compute proxy with native FP32 storage. Image/audio encoder latency unmodeled.",
    causal_attention=False,
)
MODEL_QUANTIZATION_PROFILES.update(
    [
        _artifact_profile(
            model_key="d1-omni-600m",
            precision_key="bf16",
            label="Native checkpoint",
            source_repo="LiquidAI/d1-omni-600M",
            source_revision="02b55d7076f15129e59ab3f94783f32c4b088674",
            total_weight_bytes=2348644356,
            compute_precision_shares={"bf16": 1.0},
            quantized=(),
            retained=("Entire native checkpoint; modality components and heads included.",),
            notes="Pinned Hub physical tensor dtype inventory. BF16 compute is an uncalibrated proxy; native FP32 tensors retain their full storage.",
            quant_algo="Unquantized",
            kv_cache_format="BF16",
            captured_at="2026-10-17",
        )
    ]
)

DECISION_MODELS["pplx-decider-v1.1-27b"] = Model(
    key="pplx-decider-v1.1-27b",
    name="Perplexity Decider v1.1 27B",
    cat="perplexity-ai",
    color="#2563EB",
    total_params=26086635760,
    active_params=26086635760,
    is_moe=False,
    layers=64,
    num_heads=24,
    kv_heads=4,
    head_dim=256,
    is_mla=False,
    hidden_dim=5120,
    attention_layers=16,
    kv_layers=16,
    max_context_tokens=262144,
    quality_confidence=0.0,
    capabilities_override=frozenset(),
    attention_label="16 full attention + 48 linear/conv · single forward",
    native_precision_label="Native BF16",
    architecture_note="Noncausal full-attention layers, native linear layers, calibrated readout. Config positional ceiling 262,144; production long-context performance uncalibrated. Custom inference and checkpoint access may be required. Capacity is a closed-form text-only estimate, not a measured benchmark. Quality uncalibrated; no generation economics.",
    linear_attention_layers=48,
    linear_attention_heads=48,
    linear_attention_head_dim=128,
    linear_attention_k_heads=16,
    linear_attention_k_head_dim=128,
    linear_attention_conv_kernel=4,
)
DECISION_PROFILES["pplx-decider-v1.1-27b"] = DecisionProfile(
    262144,
    "https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b/blob/14c9d90137962d300f3de95633f7b6535e80c5ce/README.md",
    "Noncausal full-attention layers, native linear layers, calibrated readout. Config positional ceiling 262,144; production long-context performance uncalibrated. Custom inference and checkpoint access may be required.",
    causal_attention=False,
)
MODEL_QUANTIZATION_PROFILES.update(
    [
        _artifact_profile(
            model_key="pplx-decider-v1.1-27b",
            precision_key="bf16",
            label="Native checkpoint",
            source_repo="perplexity-ai/pplx-decider-v1.1-27b",
            source_revision="14c9d90137962d300f3de95633f7b6535e80c5ce",
            total_weight_bytes=52173271520,
            compute_precision_shares={"bf16": 1.0},
            quantized=(),
            retained=("Entire native checkpoint; modality components and heads included.",),
            notes="Pinned Hub physical tensor dtype inventory. BF16 compute is an uncalibrated proxy; native FP32 tensors retain their full storage.",
            quant_algo="Unquantized",
            kv_cache_format="BF16",
            captured_at="2026-10-17",
        )
    ]
)

DECISION_MODELS["clef"] = Model(
    key="clef",
    name="Cloudflare Clef",
    cat="Cloudflare",
    color="#B45309",
    total_params=27356728560,
    active_params=27356728560,
    is_moe=False,
    layers=64,
    num_heads=24,
    kv_heads=4,
    head_dim=256,
    is_mla=False,
    hidden_dim=5120,
    attention_layers=16,
    kv_layers=16,
    max_context_tokens=16384,
    quality_confidence=0.0,
    capabilities_override=frozenset(),
    attention_label="16 full attention + 48 linear/conv · single forward",
    native_precision_label="Native BF16",
    architecture_note="Released encode_record default max_length 16,384; text-only estimate, image latency unmodeled. Capacity is a closed-form text-only estimate, not a measured benchmark. Quality uncalibrated; no generation economics.",
    linear_attention_layers=48,
    linear_attention_heads=48,
    linear_attention_head_dim=128,
    linear_attention_k_heads=16,
    linear_attention_k_head_dim=128,
    linear_attention_conv_kernel=4,
)
DECISION_PROFILES["clef"] = DecisionProfile(
    16384,
    "https://huggingface.co/Cloudflare/clef/blob/0b331204bb13fbd2ca93a64df1956f5b55478ce5/README.md",
    "Released encode_record default max_length 16,384; text-only estimate, image latency unmodeled.",
    causal_attention=True,
)
MODEL_QUANTIZATION_PROFILES.update(
    [
        _artifact_profile(
            model_key="clef",
            precision_key="bf16",
            label="Native checkpoint",
            source_repo="Cloudflare/clef",
            source_revision="0b331204bb13fbd2ca93a64df1956f5b55478ce5",
            total_weight_bytes=54713457120,
            compute_precision_shares={"bf16": 1.0},
            quantized=(),
            retained=("Entire native checkpoint; modality components and heads included.",),
            notes="Pinned Hub physical tensor dtype inventory. BF16 compute is an uncalibrated proxy; native FP32 tensors retain their full storage.",
            quant_algo="Unquantized",
            kv_cache_format="BF16",
            captured_at="2026-10-17",
        )
    ]
)

DECISION_MODELS["clef-flash"] = Model(
    key="clef-flash",
    name="Cloudflare Clef Flash",
    cat="Cloudflare",
    color="#0F766E",
    total_params=9409813744,
    active_params=9409813744,
    is_moe=False,
    layers=32,
    num_heads=16,
    kv_heads=4,
    head_dim=256,
    is_mla=False,
    hidden_dim=4096,
    attention_layers=8,
    kv_layers=8,
    max_context_tokens=16384,
    quality_confidence=0.0,
    capabilities_override=frozenset(),
    attention_label="8 full attention + 24 linear/conv · single forward",
    native_precision_label="Native BF16",
    architecture_note="Released encode_record default max_length 16,384; text-only estimate, image latency unmodeled. Capacity is a closed-form text-only estimate, not a measured benchmark. Quality uncalibrated; no generation economics.",
    linear_attention_layers=24,
    linear_attention_heads=32,
    linear_attention_head_dim=128,
    linear_attention_k_heads=16,
    linear_attention_k_head_dim=128,
    linear_attention_conv_kernel=4,
)
DECISION_PROFILES["clef-flash"] = DecisionProfile(
    16384,
    "https://huggingface.co/Cloudflare/clef-flash/blob/8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3/README.md",
    "Released encode_record default max_length 16,384; text-only estimate, image latency unmodeled.",
    causal_attention=True,
)
MODEL_QUANTIZATION_PROFILES.update(
    [
        _artifact_profile(
            model_key="clef-flash",
            precision_key="bf16",
            label="Native checkpoint",
            source_repo="Cloudflare/clef-flash",
            source_revision="8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3",
            total_weight_bytes=18819627488,
            compute_precision_shares={"bf16": 1.0},
            quantized=(),
            retained=("Entire native checkpoint; modality components and heads included.",),
            notes="Pinned Hub physical tensor dtype inventory. BF16 compute is an uncalibrated proxy; native FP32 tensors retain their full storage.",
            quant_algo="Unquantized",
            kv_cache_format="BF16",
            captured_at="2026-10-17",
        )
    ]
)

DECISION_MODELS["mapika-decider-2b"] = Model(
    key="mapika-decider-2b",
    name="Mapika Decider 2B",
    cat="Mapika",
    color="#7C3AED",
    total_params=1881825088,
    active_params=1881825088,
    is_moe=False,
    layers=24,
    num_heads=8,
    kv_heads=2,
    head_dim=256,
    is_mla=False,
    hidden_dim=2048,
    attention_layers=6,
    kv_layers=6,
    max_context_tokens=32768,
    quality_confidence=0.0,
    capabilities_override=frozenset(),
    attention_label="6 full attention + 18 linear/conv · single forward",
    native_precision_label="Native BF16",
    architecture_note="Released 32k serving cap includes state, questions and answer options. Capacity is a closed-form text-only estimate, not a measured benchmark. Quality uncalibrated; no generation economics.",
    linear_attention_layers=18,
    linear_attention_heads=16,
    linear_attention_head_dim=128,
    linear_attention_k_heads=16,
    linear_attention_k_head_dim=128,
    linear_attention_conv_kernel=4,
)
DECISION_PROFILES["mapika-decider-2b"] = DecisionProfile(
    32768,
    "https://huggingface.co/Mapika/decider-2b/blob/ff99089b6810691bd79134ec6febb5d9afa6e619/README.md",
    "Released 32k serving cap includes state, questions and answer options.",
    causal_attention=True,
)
MODEL_QUANTIZATION_PROFILES.update(
    [
        _artifact_profile(
            model_key="mapika-decider-2b",
            precision_key="bf16",
            label="Native checkpoint",
            source_repo="Mapika/decider-2b",
            source_revision="ff99089b6810691bd79134ec6febb5d9afa6e619",
            total_weight_bytes=3763650176,
            compute_precision_shares={"bf16": 1.0},
            quantized=(),
            retained=("Entire native checkpoint; modality components and heads included.",),
            notes="Pinned Hub physical tensor dtype inventory. BF16 compute is an uncalibrated proxy; native FP32 tensors retain their full storage.",
            quant_algo="Unquantized",
            kv_cache_format="BF16",
            captured_at="2026-10-17",
        )
    ]
)

DECISION_MODELS["mapika-decider-4b"] = Model(
    key="mapika-decider-4b",
    name="Mapika Decider 4B",
    cat="Mapika",
    color="#2563EB",
    total_params=4205751296,
    active_params=4205751296,
    is_moe=False,
    layers=32,
    num_heads=16,
    kv_heads=4,
    head_dim=256,
    is_mla=False,
    hidden_dim=2560,
    attention_layers=8,
    kv_layers=8,
    max_context_tokens=32768,
    quality_confidence=0.0,
    capabilities_override=frozenset(),
    attention_label="8 full attention + 24 linear/conv · single forward",
    native_precision_label="Native BF16",
    architecture_note="Released 32k serving cap includes state, questions and answer options. Capacity is a closed-form text-only estimate, not a measured benchmark. Quality uncalibrated; no generation economics.",
    linear_attention_layers=24,
    linear_attention_heads=32,
    linear_attention_head_dim=128,
    linear_attention_k_heads=16,
    linear_attention_k_head_dim=128,
    linear_attention_conv_kernel=4,
)
DECISION_PROFILES["mapika-decider-4b"] = DecisionProfile(
    32768,
    "https://huggingface.co/Mapika/decider-4b/blob/99d7fd5bfcc4d11ba37239dafe0ed653803819d3/README.md",
    "Released 32k serving cap includes state, questions and answer options.",
    causal_attention=True,
)
MODEL_QUANTIZATION_PROFILES.update(
    [
        _artifact_profile(
            model_key="mapika-decider-4b",
            precision_key="bf16",
            label="Native checkpoint",
            source_repo="Mapika/decider-4b",
            source_revision="99d7fd5bfcc4d11ba37239dafe0ed653803819d3",
            total_weight_bytes=8411502592,
            compute_precision_shares={"bf16": 1.0},
            quantized=(),
            retained=("Entire native checkpoint; modality components and heads included.",),
            notes="Pinned Hub physical tensor dtype inventory. BF16 compute is an uncalibrated proxy; native FP32 tensors retain their full storage.",
            quant_algo="Unquantized",
            kv_cache_format="BF16",
            captured_at="2026-10-17",
        )
    ]
)

DECISION_MODELS["mapika-decider-35b-a3b"] = Model(
    key="mapika-decider-35b-a3b",
    name="Mapika Decider 35B A3B",
    cat="Mapika",
    color="#B45309",
    total_params=34660610688,
    active_params=3000000000.0,
    is_moe=True,
    layers=40,
    num_heads=16,
    kv_heads=2,
    head_dim=256,
    is_mla=False,
    hidden_dim=2048,
    attention_layers=10,
    kv_layers=10,
    max_context_tokens=32768,
    quality_confidence=0.0,
    capabilities_override=frozenset(),
    attention_label="10 full attention + 30 linear/conv · single forward",
    native_precision_label="Native BF16",
    architecture_note="Released 32k serving cap. Rounded 3B active-parameter proxy; expert routing and multimodal latency uncalibrated. Capacity is a closed-form text-only estimate, not a measured benchmark. Quality uncalibrated; no generation economics.",
    linear_attention_layers=30,
    linear_attention_heads=32,
    linear_attention_head_dim=128,
    linear_attention_k_heads=16,
    linear_attention_k_head_dim=128,
    linear_attention_conv_kernel=4,
    moe_routed_experts=256,
    moe_active_experts=8,
)
DECISION_PROFILES["mapika-decider-35b-a3b"] = DecisionProfile(
    32768,
    "https://huggingface.co/Mapika/decider-35b-a3b/blob/91470e65478656eb68fde8c0ccdfe575d69d0309/README.md",
    "Released 32k serving cap. Rounded 3B active-parameter proxy; expert routing and multimodal latency uncalibrated.",
    causal_attention=True,
)
MODEL_QUANTIZATION_PROFILES.update(
    [
        _artifact_profile(
            model_key="mapika-decider-35b-a3b",
            precision_key="bf16",
            label="Native checkpoint",
            source_repo="Mapika/decider-35b-a3b",
            source_revision="91470e65478656eb68fde8c0ccdfe575d69d0309",
            total_weight_bytes=69321221376,
            compute_precision_shares={"bf16": 1.0},
            quantized=(),
            retained=("Entire native checkpoint; modality components and heads included.",),
            notes="Pinned Hub physical tensor dtype inventory. BF16 compute is an uncalibrated proxy; native FP32 tensors retain their full storage.",
            quant_algo="Unquantized",
            kv_cache_format="BF16",
            captured_at="2026-10-17",
        )
    ]
)

DECISION_MODELS["kev-27b"] = Model(
    key="kev-27b",
    name="Kev 27B",
    cat="jaredpalmer",
    color="#0F766E",
    total_params=25624600064,
    active_params=25624600064,
    is_moe=False,
    layers=64,
    num_heads=24,
    kv_heads=4,
    head_dim=256,
    is_mla=False,
    hidden_dim=5120,
    attention_layers=16,
    kv_layers=16,
    max_context_tokens=65536,
    quality_confidence=0.0,
    capabilities_override=frozenset(),
    attention_label="16 full attention + 48 linear/conv · single forward",
    native_precision_label="Native BF16",
    architecture_note="Conservative 65,536-position validated serving envelope; pointer head, prefill only. Questions consume the same input budget. Capacity is a closed-form text-only estimate, not a measured benchmark. Quality uncalibrated; no generation economics.",
    linear_attention_layers=48,
    linear_attention_heads=48,
    linear_attention_head_dim=128,
    linear_attention_k_heads=16,
    linear_attention_k_head_dim=128,
    linear_attention_conv_kernel=4,
)
DECISION_PROFILES["kev-27b"] = DecisionProfile(
    65536,
    "https://huggingface.co/jaredpalmer/kev-27b/blob/af0e6d551bdc2cc724f3e9d7a8bee1cd4fb8f7bf/README.md",
    "Conservative 65,536-position validated serving envelope; pointer head, prefill only. Questions consume the same input budget.",
    causal_attention=True,
)
MODEL_QUANTIZATION_PROFILES.update(
    [
        _artifact_profile(
            model_key="kev-27b",
            precision_key="bf16",
            label="Native checkpoint",
            source_repo="jaredpalmer/kev-27b",
            source_revision="af0e6d551bdc2cc724f3e9d7a8bee1cd4fb8f7bf",
            total_weight_bytes=51249200128,
            compute_precision_shares={"bf16": 1.0},
            quantized=(),
            retained=("Entire native checkpoint; modality components and heads included.",),
            notes="Pinned Hub physical tensor dtype inventory. BF16 compute is an uncalibrated proxy; native FP32 tensors retain their full storage.",
            quant_algo="Unquantized",
            kv_cache_format="BF16",
            captured_at="2026-10-17",
        )
    ]
)

DECISION_MODELS["intern-decision-0.8b"] = Model(
    key="intern-decision-0.8b",
    name="Intern-Decision 0.8B",
    cat="internlm",
    color="#7C3AED",
    total_params=852985920,
    active_params=852985920,
    is_moe=False,
    layers=24,
    num_heads=8,
    kv_heads=2,
    head_dim=256,
    is_mla=False,
    hidden_dim=1024,
    attention_layers=6,
    kv_layers=6,
    max_context_tokens=8192,
    quality_confidence=0.0,
    capabilities_override=frozenset(),
    attention_label="6 full attention + 18 linear/conv · single forward",
    native_precision_label="Native BF16",
    architecture_note="DecisionEngine default max_length 8,192 includes image positions. Scored fields are not generated tokens. Capacity is a closed-form text-only estimate, not a measured benchmark. Quality uncalibrated; no generation economics.",
    linear_attention_layers=18,
    linear_attention_heads=16,
    linear_attention_head_dim=128,
    linear_attention_k_heads=16,
    linear_attention_k_head_dim=128,
    linear_attention_conv_kernel=4,
)
DECISION_PROFILES["intern-decision-0.8b"] = DecisionProfile(
    8192,
    "https://huggingface.co/internlm/Intern-Decision-0.8B/blob/85a0cc5a99d67ea8d56dfe98115689212867171d/README.md",
    "DecisionEngine default max_length 8,192 includes image positions. Scored fields are not generated tokens.",
    causal_attention=True,
)
MODEL_QUANTIZATION_PROFILES.update(
    [
        _artifact_profile(
            model_key="intern-decision-0.8b",
            precision_key="bf16",
            label="Native checkpoint",
            source_repo="internlm/Intern-Decision-0.8B",
            source_revision="85a0cc5a99d67ea8d56dfe98115689212867171d",
            total_weight_bytes=1705977024,
            compute_precision_shares={"bf16": 1.0},
            quantized=(),
            retained=("Entire native checkpoint; modality components and heads included.",),
            notes="Pinned Hub physical tensor dtype inventory. BF16 compute is an uncalibrated proxy; native FP32 tensors retain their full storage.",
            quant_algo="Unquantized",
            kv_cache_format="BF16",
            captured_at="2026-10-17",
        )
    ]
)

DECISION_MODELS["intern-decision-2b"] = Model(
    key="intern-decision-2b",
    name="Intern-Decision 2B",
    cat="internlm",
    color="#2563EB",
    total_params=2213241664,
    active_params=2213241664,
    is_moe=False,
    layers=24,
    num_heads=8,
    kv_heads=2,
    head_dim=256,
    is_mla=False,
    hidden_dim=2048,
    attention_layers=6,
    kv_layers=6,
    max_context_tokens=8192,
    quality_confidence=0.0,
    capabilities_override=frozenset(),
    attention_label="6 full attention + 18 linear/conv · single forward",
    native_precision_label="Native BF16",
    architecture_note="DecisionEngine default max_length 8,192 includes image positions. Scored fields are not generated tokens. Capacity is a closed-form text-only estimate, not a measured benchmark. Quality uncalibrated; no generation economics.",
    linear_attention_layers=18,
    linear_attention_heads=16,
    linear_attention_head_dim=128,
    linear_attention_k_heads=16,
    linear_attention_k_head_dim=128,
    linear_attention_conv_kernel=4,
)
DECISION_PROFILES["intern-decision-2b"] = DecisionProfile(
    8192,
    "https://huggingface.co/internlm/Intern-Decision-2B/blob/8797836c65fc91a2435b1fb6850b5f0aabd75cc3/README.md",
    "DecisionEngine default max_length 8,192 includes image positions. Scored fields are not generated tokens.",
    causal_attention=True,
)
MODEL_QUANTIZATION_PROFILES.update(
    [
        _artifact_profile(
            model_key="intern-decision-2b",
            precision_key="bf16",
            label="Native checkpoint",
            source_repo="internlm/Intern-Decision-2B",
            source_revision="8797836c65fc91a2435b1fb6850b5f0aabd75cc3",
            total_weight_bytes=4426488512,
            compute_precision_shares={"bf16": 1.0},
            quantized=(),
            retained=("Entire native checkpoint; modality components and heads included.",),
            notes="Pinned Hub physical tensor dtype inventory. BF16 compute is an uncalibrated proxy; native FP32 tensors retain their full storage.",
            quant_algo="Unquantized",
            kv_cache_format="BF16",
            captured_at="2026-10-17",
        )
    ]
)

DECISION_MODELS["intern-decision-4b"] = Model(
    key="intern-decision-4b",
    name="Intern-Decision 4B",
    cat="internlm",
    color="#B45309",
    total_params=4539265536,
    active_params=4539265536,
    is_moe=False,
    layers=32,
    num_heads=16,
    kv_heads=4,
    head_dim=256,
    is_mla=False,
    hidden_dim=2560,
    attention_layers=8,
    kv_layers=8,
    max_context_tokens=8192,
    quality_confidence=0.0,
    capabilities_override=frozenset(),
    attention_label="8 full attention + 24 linear/conv · single forward",
    native_precision_label="Native BF16",
    architecture_note="DecisionEngine default max_length 8,192 includes image positions. Scored fields are not generated tokens. Capacity is a closed-form text-only estimate, not a measured benchmark. Quality uncalibrated; no generation economics.",
    linear_attention_layers=24,
    linear_attention_heads=32,
    linear_attention_head_dim=128,
    linear_attention_k_heads=16,
    linear_attention_k_head_dim=128,
    linear_attention_conv_kernel=4,
)
DECISION_PROFILES["intern-decision-4b"] = DecisionProfile(
    8192,
    "https://huggingface.co/internlm/Intern-Decision-4B/blob/1e65a988dacbcf00b84aee7366f9e47563407000/README.md",
    "DecisionEngine default max_length 8,192 includes image positions. Scored fields are not generated tokens.",
    causal_attention=True,
)
MODEL_QUANTIZATION_PROFILES.update(
    [
        _artifact_profile(
            model_key="intern-decision-4b",
            precision_key="bf16",
            label="Native checkpoint",
            source_repo="internlm/Intern-Decision-4B",
            source_revision="1e65a988dacbcf00b84aee7366f9e47563407000",
            total_weight_bytes=9078538752,
            compute_precision_shares={"bf16": 1.0},
            quantized=(),
            retained=("Entire native checkpoint; modality components and heads included.",),
            notes="Pinned Hub physical tensor dtype inventory. BF16 compute is an uncalibrated proxy; native FP32 tensors retain their full storage.",
            quant_algo="Unquantized",
            kv_cache_format="BF16",
            captured_at="2026-10-17",
        )
    ]
)


DECISION_MODELS["pplx-decider-flash-v1.2"] = Model(
    key="pplx-decider-flash-v1.2",
    name="Perplexity Decider Flash v1.2 (4B)",
    cat="perplexity-ai",
    color="#0891B2",
    total_params=4539918336,
    active_params=4539918336,
    is_moe=False,
    layers=32,
    num_heads=16,
    kv_heads=4,
    head_dim=256,
    is_mla=False,
    hidden_dim=2560,
    attention_layers=8,
    kv_layers=8,
    max_context_tokens=262144,
    quality_confidence=0.0,
    capabilities_override=frozenset(),
    attention_label="8 noncausal full attention + 24 causal linear · joint single forward",
    native_precision_label="Native BF16",
    architecture_note="Qwen3.5-4B backbone with joint per-question marker readout; noncausal full attention and native causal linear layers. Released max_length defaults to 262,144 positions, including images and question/option templates; training covered about 32k. Longer inputs require task validation. Recursive multi-question splitting, image encoder latency, compilation, CUDA graphs and optional FP8 fast-path gains are not simulated. Text-only closed-form capacity estimate, not a benchmark; quality uncalibrated and no generation economics.",
    linear_attention_layers=24,
    linear_attention_heads=32,
    linear_attention_head_dim=128,
    linear_attention_k_heads=16,
    linear_attention_k_head_dim=128,
    linear_attention_conv_kernel=4,
)
DECISION_PROFILES["pplx-decider-flash-v1.2"] = DecisionProfile(
    262144,
    "https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2/blob/6d771ad7b5ecff5e93b8fc7b90d4c5ccc341954d/README.md",
    "Qwen3.5-4B backbone with joint per-question marker readout; noncausal full attention and native causal linear layers. Released max_length defaults to 262,144 positions, including images and question/option templates; training covered about 32k. Longer inputs require task validation. Recursive multi-question splitting, image encoder latency, compilation, CUDA graphs and optional FP8 fast-path gains are not simulated. Text-only closed-form capacity estimate, not a benchmark; quality uncalibrated and no generation economics.",
    causal_attention=False,
)
MODEL_QUANTIZATION_PROFILES.update(
    [
        _artifact_profile(
            model_key="pplx-decider-flash-v1.2",
            precision_key="bf16",
            label="Native BF16 + decision head",
            source_repo="perplexity-ai/pplx-decider-flash-v1.2",
            source_revision="6d771ad7b5ecff5e93b8fc7b90d4c5ccc341954d",
            total_weight_bytes=9079836672,
            compute_precision_shares={"bf16": 1.0},
            quantized=(),
            retained=("Entire BF16 backbone including vision; separate 255×2560 BF16 readout.",),
            notes="Pinned backbone index: 9,078,531,072 bytes; separate readout tensor: 1,305,600 bytes. No released quantized artifact or optional runtime fast-path speedup is assumed.",
            quant_algo="Unquantized",
            kv_cache_format="BF16",
            captured_at="2026-10-17",
        )
    ]
)
