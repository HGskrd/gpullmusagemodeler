"""Roofline throughput estimation engine for the GPU/LLM Usage Modeler."""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Optional, cast

import cloud_policy
from data import (
    ASR_WER_LANGUAGE_LABELS,
    ASR_WER_LANGUAGE_SOURCES,
    ASR_WER_LANGUAGES,
    ASR_WER_PLACEHOLDER,
    BATCH_SIZES,
    CORPO_CLOUD_DEFAULT,
    DAY_SHAPES,
    DEFAULT_COUNTRY,
    DIST_PRESETS,
    EMBEDDING_DECONTAMINATED_BEIR_SOURCES,
    EMBEDDING_DOC_BUCKETS,
    EMBEDDING_QUALITY_PLACEHOLDER,
    EMBEDDING_QUALITY_SOURCES,
    GPU,
    INPUT_BUCKETS,
    MODELS,
    OUTPUT_BUCKETS,
    PUBLISHED_ASR_WER,
    PUBLISHED_EMBEDDING_DECONTAMINATED_BEIR,
    PUBLISHED_EMBEDDING_QUALITY,
    QUALITY_DOMAIN_LABELS,
    Bucket,
    Model,
    SpeculativeProfile,
    carbon_intensity_avg,
    effective_quality,
    model_domain_anchor,
    model_profile_quality,
    model_profile_success_rate,
    normalize_precision,
    normalize_quality_domain,
    normalize_quality_weights,
    quality_weights_label,
    required_quality,
    success_rate,
)
from deployment import Deployment

# Wall-clock accelerator draw as a fraction of published board TDP during saturated
# inference. The 0.6–0.8 anchor comes primarily from vLLM measurements on H100/MI300
# and is a transparent starting point, not a portable runtime/hardware guarantee.
GPU_POWER_UTILIZATION = 0.70
LATENT_UNLOCK_STEEPNESS = 4.0
MARGINAL_RECOMMENDATION_LIMIT = 5

INTER_NODE_COLLECTIVE_BW = 25e9
DATA_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]  # Fixed to match BATCH_SIZES
EMBEDDING_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
USER_EXP_SWEEP = [
    1,
    2,
    4,
    8,
    12,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    192,
    256,
    384,
    512,
    768,
    1024,
]
USER_EXP_FRACTIONS = [0.50, 0.75, 0.90, 0.95]
REALTIME_USER_SWEEP = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    12288,
    16384,
]
MAX_REALTIME_USERS = 262_144
UNBOUNDED_BATCH = 1_000_000_000
LONG_CTX_DCP_SEQ = 32768
BATCH_AXIS_HEADROOM = 0.12
PROCESSING_PARETO_COLORS = ["#3266ad", "#1D9E75", "#BA7517", "#7F77DD", "#D85A30", "#A32D2D"]
NIGHT_HOURS = frozenset({22, 23, 0, 1, 2, 3, 4, 5})
NVIDIA_FP4_GPU_KEYS = frozenset(
    {
        "RTXPRO6000_BSE",
        "RTXPRO6000_BW_WS",
        "RTXPRO5000_BW_72",
        "RTX5090",
        "DGX_SPARK",
        "GB200",
        "B200",
        "B300",
        "GB300",
        "DGX_STATION_GB300",
        "JETSON_AGX_THOR",
    }
)
MXFP4_GPU_KEYS = NVIDIA_FP4_GPU_KEYS | frozenset(
    {"MI350X", "MI355X", "MI455X", "HELIOS_MI455X", "MI400"}
)

# Speculative decoding has a real control-plane cost even when draft weights are
# tiny: launch the draft/verify work, schedule the block, and synchronize the
# rejection sampler.  These are deliberately small, fixed closed-form terms
# (microseconds per speculative cycle), not a claim to simulate a specific
# runtime.  Keeping them explicit prevents MTP from being treated as free.
SPEC_DRAFT_LAUNCH_OVERHEAD_S = 3e-6
SPEC_SCHEDULER_OVERHEAD_S = 4e-6
SPEC_REJECTION_SYNC_OVERHEAD_S = 3e-6
SPEC_AUTO_K_PROBE_CONCURRENCY = 32
SPEC_MIN_BENEFICIAL_SPEEDUP = 1.01


@dataclass
class EfficiencyParams:
    bw_eff: float = 0.80
    comp_eff: float = 0.75
    overhead: float = 0.08
    kv_slack: float = 0.02
    paged_oh: float = 0.10
    ar_overlap: float = 0.30
    moe_imbalance: float = 1.15
    sched_budget: int = 16384
    pd_interference: float = 0.0  # Added for UI


@dataclass(frozen=True)
class SpecRuntime:
    """Resolved speculative-decoding configuration for one deployment."""

    profile: SpeculativeProfile
    k: int  # speculative tokens proposed per cycle
    alpha: float  # per-token acceptance probability
    tau: float  # expected tokens emitted per cycle (accepted prefix + bonus token)
    passes: int  # draft forward passes per cycle
    draft_weight_bytes: float  # drafter weight footprint read per draft pass
    draft_active_params: float  # per-token active drafter parameters (MoE-aware)
    auto_selected: bool = False
    probe_speedup: float = 1.0


@dataclass(frozen=True)
class SpecOptimization:
    """Result of comparing all feasible draft depths with speculative decode off."""

    runtime: Optional[SpecRuntime]
    selected_k: int
    speedup: float
    beneficial: bool
    reason: str
    probe_concurrency: int


def spec_acceptance_len(alpha: float, k: int) -> float:
    """Expected tokens per cycle for a chain of k drafts: accepted prefix + bonus.

    Real drafters verify trees (EAGLE-3) or blocks (DFlash), so profile alphas are
    fitted to measured acceptance lengths at their default k; this chain formula is
    the documented conservative approximation when k moves off that default.
    """
    k = max(int(k), 0)
    alpha = min(max(float(alpha), 0.0), 1.0)
    if alpha >= 1.0:
        return float(k + 1)
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


def spec_finite_output_tau(alpha: float, k: int, output_tokens: int) -> float:
    """Expected emitted tokens/cycle including final partial-cycle waste.

    A cycle emits one through k+1 tokens. This small renewal recurrence prevents
    one-token and short responses from receiving an asymptotic long-generation
    speculative speedup.
    """
    n = max(int(output_tokens), 1)
    k = min(max(int(k), 0), n - 1)
    alpha = min(max(float(alpha), 0.0), 1.0)
    if k <= 0:
        return 1.0
    expected_cycles = [0.0] * (n + 1)
    for remaining in range(1, n + 1):
        cycle_k = min(k, remaining - 1)
        total = 1.0
        for emitted in range(1, cycle_k + 1):
            probability = (alpha ** (emitted - 1)) * (1.0 - alpha)
            total += probability * expected_cycles[remaining - emitted]
        total += (alpha**cycle_k) * expected_cycles[max(remaining - cycle_k - 1, 0)]
        expected_cycles[remaining] = total
    return n / expected_cycles[n]


def resolve_spec_runtime(
    m: Model,
    method: str,
    spec_k: int,
    alpha_override: float,
    prec: str,
) -> Optional[SpecRuntime]:
    if not method or method == "off":
        return None
    profile = next((p for p in m.available_spec_profiles if p.method == method), None)
    if profile is None:
        return None
    requested_k = min(max(int(spec_k if spec_k > 0 else profile.default_k), 1), 32)
    supported_ks = tuple(
        sorted({int(k) for k in getattr(profile, "supported_ks", ()) if 1 <= int(k) <= 32})
    )
    k = (
        min(supported_ks, key=lambda candidate: (abs(candidate - requested_k), candidate))
        if supported_ks
        else requested_k
    )
    calibrated_alphas = dict(getattr(profile, "acceptance_alpha_by_k", ()))
    profile_alpha = calibrated_alphas.get(k, profile.acceptance_alpha)
    alpha = min(max(float(alpha_override if alpha_override > 0 else profile_alpha), 0.0), 1.0)
    passes = 1 if profile.parallel_draft else k
    # Draft weights share the target's served precision: scale by the target's
    # average bytes/param at this precision (mixed-precision LUTs included).
    avg_bpp = m.weight_bytes(prec) / m.total_params if m.total_params > 0 else 2.0
    draft_weight_bytes = (
        getattr(profile, "exact_weight_bytes", 0.0)
        if getattr(profile, "exact_weight_bytes", 0.0) > 0
        else profile.draft_params * avg_bpp
    )
    return SpecRuntime(
        profile=profile,
        k=k,
        alpha=alpha,
        tau=spec_acceptance_len(alpha, k),
        passes=passes,
        draft_weight_bytes=draft_weight_bytes,
        draft_active_params=getattr(profile, "active_params", 0.0) or profile.draft_params,
    )


@dataclass
class MemoryResult:
    requested: float
    weights: float
    profiled_non_kv: float
    kv_reserved: float
    kv_budget: float
    kv_per_token: float


@dataclass
class DecodeResult:
    tps: int
    lat: float
    step_ms: float
    max_slots: int
    spec_tau: float = 0.0  # tokens emitted per speculative cycle; 0 when spec is off
    spec_speedup: float = 1.0  # per-token latency ratio vs the same topology without spec


@dataclass
class PrefillResult:
    tps: int
    service_time: float
    rps: float
    max_batch: int


@dataclass
class DataResult:
    rps: float
    tps: int
    prefill_frac: float


@dataclass
class EmbeddingResult:
    rps: float
    tps: int
    vectors_per_second: int
    output_mb_s: float
    service_time: float
    max_batch: int
    seq_len: int
    vectors_per_input: float
    p50_seq_len: int = 0
    p90_seq_len: int = 0
    p99_seq_len: int = 0
    output_bytes_per_input: float = 0.0


@dataclass(frozen=True)
class EmbeddingDocStats:
    mean_seq_len: float
    p50_seq_len: int
    p90_seq_len: int
    p99_seq_len: int
    mean_vectors_per_input: float
    mean_output_bytes_per_input: float
    mean_scratch_bytes_per_input: float


@dataclass
class UserExperienceResult:
    arrival_rps: float
    decode_step_ms: float
    ttft_ms: float
    response_s: float
    inflight: float


@dataclass
class RealtimeCapacityResult:
    users: int
    realtime_factor: float
    per_user_tps: float
    total_tps: int
    step_ms: float
    max_slots: int
    required_tps: float


@dataclass
class DeploymentPeakResult:
    tps: int
    rps: float
    batch_size: int
    prefill_frac: float


@dataclass
class CommBreakdown:
    dense_tp: float = 0.0
    pp_boundary: float = 0.0
    tp_cross_node: bool = False
    pp_cross_node_boundaries: int = 0
    ep_advisory: bool = False
    # Expert parallelism is not a planner topology dimension.  Keep this explicit
    # instead of silently charging TP as if it were EP; callers can disclose it.
    expert_parallel_unmodeled: bool = False
    dcp_advisory: bool = False

    @property
    def total(self) -> float:
        return self.dense_tp + self.pp_boundary


def factors(n: int) -> list[int]:
    return [i for i in range(1, n + 1) if n > 0 and n % i == 0]


def factor_triples(n: int) -> list[tuple[int, int, int]]:
    triples = []
    for tp in factors(n):
        rem = n // tp
        for pp in factors(rem):
            triples.append((tp, pp, rem // pp))
    return triples


def strategy_label(tp: int, pp: int, dp: int) -> str:
    return f"TP{tp}xPP{pp}xDP{dp}"


def kv_bytes_per_token(m: Model, prec: str) -> float:
    """Initial KV allocation slope for one token, before any sliding-window cap.

    Keep this readout on the same canonical path as sequence-level budgeting so MLA's
    joint latent representation and local/global layer splits cannot drift apart.
    """
    return kv_cache_bytes_for_sequence(m, 1.0, prec)


def _split_attention_layers(total_layers: int, local_layers: int) -> tuple[int, int]:
    local = min(max(local_layers, 0), max(total_layers, 0))
    return max(total_layers - local, 0), local


def _local_context_tokens(m: Model, seq_len: float) -> float:
    if m.local_attention_window <= 0:
        return max(seq_len, 0.0)
    return min(max(seq_len, 0.0), float(m.local_attention_window))


def _kv_projection_count(m: Model) -> int:
    return 1 if m.shared_key_value else 2


# Model geometry is fixed at import time: every Model lives in data.MODELS under
# a unique key and is never mutated or rebuilt at runtime (the one derived entry,
# gemma-4-e2b-asr, is created with replace(..., key=...) and so gets its own key).
# That makes m.key a sound cache key for the pure geometry helpers below, which
# the projection search calls tens of thousands of times per request.
# Model itself is an unfrozen dataclass and therefore unhashable, so these cannot
# be functools.lru_cache'd on the instance directly.
_KV_ELEMS_CACHE: dict[tuple[str, bool], int] = {}
_KV_BYTES_CACHE: dict[tuple[str, str, bool], float] = {}
_PP_PEAK_CACHE: dict[tuple[str, int], float] = {}
_LINEAR_STATE_CACHE: dict[tuple[str, str, int], float] = {}
_REPLICA_KV_CACHE: dict[tuple[str, float, str, int, int], float] = {}
# seq_len is a continuous key, so this one needs a ceiling; the others are bounded
# by catalog size (~115 models x a handful of precisions and ranks).
_REPLICA_KV_CACHE_MAX = 100_000


def _kv_heads_for_layer(m: Model, global_layer: bool = False) -> int:
    if global_layer and m.global_kv_heads > 0:
        return m.global_kv_heads
    if not global_layer and m.local_attention_layers > 0:
        return m.local_kv_head_count
    return m.kv_heads


def _kv_elems_per_layer(m: Model, global_layer: bool = False) -> int:
    cache_key = (m.key, global_layer)
    cached = _KV_ELEMS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    _KV_ELEMS_CACHE[cache_key] = value = _kv_elems_per_layer_uncached(m, global_layer)
    return value


def _kv_elems_per_layer_uncached(m: Model, global_layer: bool = False) -> int:
    if m.is_mla:
        return m.mla_kv_dim + m.mla_rope_dim
    if global_layer and m.global_kv_heads > 0:
        heads = _kv_heads_for_layer(m, global_layer=True)
        head_dim = m.global_head_dim or m.head_dim
    elif not global_layer and m.local_attention_layers > 0:
        heads = _kv_heads_for_layer(m, global_layer=False)
        head_dim = m.local_kv_head_size
    else:
        heads = m.kv_heads
        head_dim = m.head_dim
    return _kv_projection_count(m) * heads * head_dim


def _kv_bytes_per_layer(m: Model, prec: str, global_layer: bool = False) -> float:
    cache_key = (m.key, prec, global_layer)
    cached = _KV_BYTES_CACHE.get(cache_key)
    if cached is not None:
        return cached
    value = _kv_elems_per_layer(m, global_layer=global_layer) * m.kv_cache_bytes_per_elem(prec)
    _KV_BYTES_CACHE[cache_key] = value
    return value


def linear_attention_state_bytes(m: Model, prec: str) -> float:
    layers = m.linear_attention_layer_count
    if layers <= 0:
        return 0.0

    bpe = m.kv_cache_bytes_per_elem(prec)
    heads = m.linear_attention_head_count
    head_dim = m.linear_attention_head_size
    k_heads = m.linear_attention_k_head_count
    k_head_dim = m.linear_attention_k_head_size
    conv_len = m.linear_attention_kernel_size - 1

    recurrent_elems = heads * head_dim * head_dim
    conv_elems = conv_len * ((heads * head_dim) + (2 * k_heads * k_head_dim))
    return layers * (recurrent_elems + conv_elems) * bpe


def _head_aligned_tp_shards(heads: int, tp: int) -> int:
    """Conservative number of head-aligned state shards represented by the schema."""
    return max(math.gcd(max(int(heads), 1), max(int(tp), 1)), 1)


def per_tp_linear_attention_state_bytes(m: Model, prec: str, tp: int) -> float:
    """Per-rank recurrent/conv state using the model's explicit linear-head schema.

    Linear recurrent state is partitioned over its query/value heads, while the
    convolution K/V state is partitioned over its (possibly different) K-head axis.
    Using head-aligned shard counts avoids over-sharding state when TP exceeds or is
    not divisible into either schema dimension.
    """
    cache_key = (m.key, prec, int(tp))
    cached = _LINEAR_STATE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    _LINEAR_STATE_CACHE[cache_key] = value = _per_tp_linear_attention_state_bytes_uncached(
        m, prec, tp
    )
    return value


def _per_tp_linear_attention_state_bytes_uncached(m: Model, prec: str, tp: int) -> float:
    layers = m.linear_attention_layer_count
    if layers <= 0:
        return 0.0

    bpe = m.kv_cache_bytes_per_elem(prec)
    heads = m.linear_attention_head_count
    head_dim = m.linear_attention_head_size
    k_heads = m.linear_attention_k_head_count
    k_head_dim = m.linear_attention_k_head_size
    conv_len = m.linear_attention_kernel_size - 1
    head_shards = _head_aligned_tp_shards(heads, tp)
    k_head_shards = _head_aligned_tp_shards(k_heads, tp)

    recurrent_elems = heads * head_dim * head_dim / head_shards
    conv_elems = conv_len * (
        (heads * head_dim / head_shards) + (2 * k_heads * k_head_dim / k_head_shards)
    )
    return layers * (recurrent_elems + conv_elems) * bpe


def kv_cache_bytes_for_sequence(m: Model, seq_len: float, prec: str) -> float:
    seq = max(float(seq_len), 0.0)
    full_layers, local_layers = _split_attention_layers(m.kv_layer_count, m.local_attention_layers)
    return full_layers * seq * _kv_bytes_per_layer(
        m, prec, global_layer=True
    ) + local_layers * _local_context_tokens(m, seq) * _kv_bytes_per_layer(
        m, prec, global_layer=False
    )


def per_replica_kv_cache_bytes(m: Model, seq_len: float, prec: str, pp: int, tp: int) -> float:
    cache_key = (m.key, float(seq_len), prec, int(pp), int(tp))
    cached = _REPLICA_KV_CACHE.get(cache_key)
    if cached is not None:
        return cached
    value = _per_replica_kv_cache_bytes_uncached(m, seq_len, prec, pp, tp)
    if len(_REPLICA_KV_CACHE) >= _REPLICA_KV_CACHE_MAX:
        _REPLICA_KV_CACHE.clear()
    _REPLICA_KV_CACHE[cache_key] = value
    return value


def per_replica_token_kv_cache_bytes(
    m: Model, seq_len: float, prec: str, pp: int, tp: int
) -> float:
    """Per-rank, token-growing attention KV for one sequence.

    Recurrent/linear-attention state is deliberately excluded.  The two kinds of
    state have the same residency lifetime but different decode traffic: attention
    reads the existing KV once per verified query position, while a fused recurrent
    block loads one initial state and stores one final state for the whole block.
    """
    pp = max(pp, 1)
    pp_fraction = _pp_peak_fraction(m, pp)
    if m.is_mla:
        token_cache = kv_cache_bytes_for_sequence(m, seq_len, prec) / kv_shards(m, tp)
    else:
        full_layers, local_layers = _split_attention_layers(
            m.kv_layer_count, m.local_attention_layers
        )
        full_cache = (
            full_layers * max(float(seq_len), 0.0) * _kv_bytes_per_layer(m, prec, global_layer=True)
        )
        local_cache = (
            local_layers
            * _local_context_tokens(m, seq_len)
            * _kv_bytes_per_layer(m, prec, global_layer=False)
        )
        token_cache = full_cache / kv_shards_for_heads(
            _kv_heads_for_layer(m, global_layer=True), tp
        ) + local_cache / kv_shards_for_heads(_kv_heads_for_layer(m, global_layer=False), tp)
    return token_cache * pp_fraction


def per_replica_recurrent_state_bytes(m: Model, prec: str, pp: int, tp: int) -> float:
    """Per-rank fixed recurrent/conv state resident for one sequence."""
    return per_tp_linear_attention_state_bytes(m, prec, tp) * _pp_peak_fraction(m, max(pp, 1))


def _per_replica_kv_cache_bytes_uncached(
    m: Model, seq_len: float, prec: str, pp: int, tp: int
) -> float:
    return per_replica_token_kv_cache_bytes(
        m, seq_len, prec, pp, tp
    ) + per_replica_recurrent_state_bytes(m, prec, pp, tp)


def attention_residual_scratch_bytes(
    m: Model, seq_len: float, prec: str, pp: int, tp: int
) -> float:
    """Conservative per-rank prefill scratch for block attention residuals.

    AttnRes retains several full-width activation sources across layer blocks.  The
    catalog records their count but not kernel liveness/checkpointing details, so the
    closed-form guard assumes every source is live in BF16.  Tokens are sequence-
    sharded across TP ranks and source-producing layers are sharded across PP stages.
    This is intentionally a capacity guard, not a claim about a specific runtime's
    allocator peak; users can calibrate remaining non-KV memory from measurements.
    """
    sources = max(int(getattr(m, "attention_residual_source_count", 0)), 0)
    if sources <= 0 or seq_len <= 0:
        return 0.0
    activation_bpe = max(2.0, float(m.kv_cache_bytes_per_elem(prec)))
    return (
        sources
        * max(float(seq_len), 0.0)
        * max(int(m.hidden_size), 1)
        * activation_bpe
        * _pp_peak_fraction(m, max(pp, 1))
        / max(int(tp), 1)
    )


def _pp_peak_fraction(m: Model, pp: int) -> float:
    """Conservative fraction of layer-distributed work/memory on the busiest PP stage."""
    cache_key = (m.key, int(pp))
    cached = _PP_PEAK_CACHE.get(cache_key)
    if cached is not None:
        return cached
    layers = max(int(m.layers), 1)
    stages = min(max(int(pp), 1), layers)
    _PP_PEAK_CACHE[cache_key] = value = math.ceil(layers / stages) / layers
    return value


def _pp_bubble_multiplier(pp: int, microbatches: int) -> float:
    """Classic pipeline fill/drain tax: (microbatches + stages - 1) / microbatches."""
    stages = max(int(pp), 1)
    microbatches = max(int(microbatches), 1)
    return 1.0 + (stages - 1) / microbatches


def context_supported(m: Model, input_tokens: float, output_tokens: float = 0.0) -> bool:
    limit = max(int(getattr(m, "max_context_tokens", 0) or 0), 1)
    return max(float(input_tokens), 0.0) + max(float(output_tokens), 0.0) <= limit


def _linear_attention_work(m: Model, seq_len: float) -> float:
    layers = m.linear_attention_layer_count
    if layers <= 0:
        return 0.0
    heads = m.linear_attention_head_count
    head_dim = m.linear_attention_head_size
    return layers * heads * head_dim * head_dim * max(seq_len, 0.0)


def _sparse_attention_context(m: Model, seq_len: float) -> float:
    seq = max(seq_len, 0.0)
    top_k = max(int(m.sparse_attention_top_k), 0)
    return min(seq, float(top_k)) if top_k > 0 else seq


def _sparse_indexer_work(m: Model, pr: int, seq_len: float, *, prefill: bool) -> float:
    """Closed-form DSA indexer work under the estimator's attention conventions.

    The lightweight indexer performs one QK-style dot product over the available
    context. During prefill all query positions run that scan; decode has one scan
    per active sequence. IndexShare is represented by the explicit number of
    layers that evaluate an indexer, with other layers reusing their selected KV
    positions.
    """
    layers = max(int(m.sparse_indexer_layers), 0)
    heads = max(int(m.sparse_indexer_heads), 0)
    head_dim = max(int(m.sparse_indexer_head_dim), 0)
    if layers <= 0 or heads <= 0 or head_dim <= 0:
        return 0.0
    seq = max(seq_len, 0.0)
    query_positions = seq if prefill else 1.0
    # One QK matmul at two FLOPs per multiply-add.
    return 2 * pr * layers * heads * head_dim * query_positions * seq


def _decode_attention_work(m: Model, pr: int, avg_seq: float, pp: int) -> float:
    full_layers, local_layers = _split_attention_layers(
        m.attention_layer_count, m.local_attention_layers
    )
    full_width = m.attention_query_head_count * m.head_dim
    local_width = m.local_attention_head_count * m.local_attention_head_size
    full_work = full_layers * full_width * _sparse_attention_context(m, avg_seq)
    local_work = local_layers * local_width * _local_context_tokens(m, avg_seq)
    linear_work = _linear_attention_work(m, 1.0)
    # QK and AV are each one matrix multiply (2 FLOPs per multiply-add).
    dot_product_work = 4 * pr * (full_work + local_work)
    indexer_work = _sparse_indexer_work(m, pr, avg_seq, prefill=False)
    recurrent_work = 2 * pr * linear_work
    return (dot_product_work + indexer_work + recurrent_work) * _pp_peak_fraction(m, pp)


def _realtime_audio_encoder_work(profile, pr: int, pp: int) -> float:
    audio_params = max(float(getattr(profile, "audio_encoder_params", 0.0)), 0.0)
    audio_tokens = max(int(getattr(profile, "audio_tokens_per_step", 1)), 1)
    pp = max(pp, 1)

    # The normal decoder step already accounts for one full model pass. Voxtral
    # realtime expands every streaming text tick into several causal audio
    # encoder tokens, so add only those extra encoder-token passes here.
    extra_token_passes = max(audio_tokens - 1, 0)
    ffn_work = 2 * audio_params * extra_token_passes * pr / pp

    layers = max(int(getattr(profile, "audio_attention_layers", 0)), 0)
    heads = max(int(getattr(profile, "audio_attention_heads", 0)), 0)
    head_dim = max(int(getattr(profile, "audio_attention_head_dim", 0)), 0)
    window = max(int(getattr(profile, "audio_attention_window", 0)), audio_tokens)
    attention_work = 4 * pr * layers * heads * head_dim * audio_tokens * window / pp
    return ffn_work + attention_work


def _prefill_attention_work(m: Model, pr: int, seq_len: int, pp: int) -> float:
    seq = max(float(seq_len), 0.0)
    full_layers, local_layers = _split_attention_layers(
        m.attention_layer_count, m.local_attention_layers
    )
    full_width = m.attention_query_head_count * m.head_dim
    local_width = m.local_attention_head_count * m.local_attention_head_size
    full_work = full_layers * full_width * seq * _sparse_attention_context(m, seq)
    local_work = local_layers * local_width * seq * _local_context_tokens(m, seq)
    linear_work = _linear_attention_work(m, seq)
    dot_product_work = 4 * pr * (full_work + local_work)
    indexer_work = _sparse_indexer_work(m, pr, seq, prefill=True)
    recurrent_work = 2 * pr * linear_work
    return (dot_product_work + indexer_work + recurrent_work) * _pp_peak_fraction(m, pp)


def gpu_supports_mxfp4(g: GPU) -> bool:
    return g.fp4 is not None and g.key in MXFP4_GPU_KEYS


def gpu_supports_nvfp4(g: GPU) -> bool:
    return g.fp4 is not None and g.key in NVIDIA_FP4_GPU_KEYS


def gpu_flops(g: GPU, prec: str) -> float:
    prec = normalize_precision(prec)
    if prec == "bf16":
        return g.bf16
    if prec == "fp8":
        return g.fp8
    if prec == "mxfp4" and gpu_supports_mxfp4(g):
        assert g.fp4 is not None
        return g.fp4
    if prec == "nvfp4" and gpu_supports_nvfp4(g):
        assert g.fp4 is not None
        return g.fp4

    # Non-native FP4 paths still benefit from compressed weight traffic, but the
    # matmul path usually pays dequant/packing overhead and cannot claim FP4 peak.
    fallback = g.fp8 if g.fp8 > 0 else g.bf16
    return fallback * (0.75 if prec == "mxfp4" else 0.65)


def model_gpu_flops(g: GPU, m: Model, prec: str) -> float:
    profile = m.quantization_profile(prec)
    if profile is None or not profile.compute_precision_shares:
        return gpu_flops(g, prec)

    denom = 0.0
    for profile_prec, share in profile.compute_precision_shares.items():
        share = max(float(share), 0.0)
        if share <= 0:
            continue
        denom += share / max(gpu_flops(g, profile_prec), 1e-9)
    return (1.0 / denom) if denom > 0 else gpu_flops(g, prec)


def normalize_dist(dist: list[int]) -> list[float]:
    total = sum(dist) or 1
    return [v / total for v in dist]


def avg_dist(dist: list[int], buckets: list[Bucket]) -> int:
    weights = normalize_dist(dist)
    return round(sum(bucket.length * weights[i] for i, bucket in enumerate(buckets)))


def dist_percentile(dist: list[int], buckets: list[Bucket], pct: float) -> int:
    pct = min(max(pct, 0.0), 1.0)
    cdf = 0.0
    for share, bucket in zip(normalize_dist(dist), buckets):
        cdf += share
        if pct <= cdf + 1e-9:
            return bucket.length
    return buckets[-1].length if buckets else 0


def _aligned_dist(dist: list[int], buckets: list[Bucket]) -> list[int]:
    values = []
    for i in range(len(buckets)):
        raw = dist[i] if i < len(dist) else 0
        values.append(max(int(raw or 0), 0))
    if not any(values) and values:
        values[0] = 1
    return values


def dist_stats(dist: list[int], buckets: list[Bucket]) -> tuple[float, float]:
    weights = normalize_dist(_aligned_dist(dist, buckets))
    mean = sum(bucket.length * weights[i] for i, bucket in enumerate(buckets))
    var = sum(((bucket.length - mean) ** 2) * weights[i] for i, bucket in enumerate(buckets))
    return mean, math.sqrt(max(var, 0.0))


def dist_share_leq(dist: list[int], buckets: list[Bucket], limit: int) -> float:
    weights = normalize_dist(dist)
    return sum(weights[i] for i, bucket in enumerate(buckets) if bucket.length <= limit)


def _paged_kv_pressure(avg_seq: float, heterogeneity: float, short_share: float) -> float:
    avg_blocks = max(avg_seq / 16.0, 1.0)
    block_pressure = min(1.6, math.log2(avg_blocks + 1.0) / 6.0)
    mix_pressure = min(1.5, heterogeneity * (0.6 + short_share))
    return min(2.0, 0.45 + 0.35 * block_pressure + 0.35 * mix_pressure + 0.25 * short_share)


def decode_paged_oh(in_dist: list[int], out_dist: list[int], eff: EfficiencyParams) -> float:
    in_mean, in_std = dist_stats(in_dist, INPUT_BUCKETS)
    out_mean, out_std = dist_stats(out_dist, OUTPUT_BUCKETS)
    avg_seq = in_mean + out_mean / 2.0
    seq_std = math.sqrt(in_std**2 + (out_std / 2.0) ** 2)
    heterogeneity = min(1.5, seq_std / max(avg_seq, 1.0))
    short_share = 0.65 * dist_share_leq(in_dist, INPUT_BUCKETS, 1024)
    short_share += 0.35 * dist_share_leq(out_dist, OUTPUT_BUCKETS, 128)
    return eff.paged_oh * _paged_kv_pressure(avg_seq, heterogeneity, short_share)


def fixed_paged_oh(seq_len: float, eff: EfficiencyParams, scale: float = 1.0) -> float:
    short_share = 1.0 / (1.0 + (seq_len / 2048.0))
    return eff.paged_oh * scale * _paged_kv_pressure(seq_len, 0.0, short_share)


def effective_prefill_length(seq_len: int, prefix_hit_rate: float) -> int:
    hit_rate = min(max(prefix_hit_rate, 0.0), 1.0)
    miss_rate = 1.0 - hit_rate
    if seq_len <= 0 or miss_rate <= 0:
        return 0
    return max(1, math.ceil(seq_len * miss_rate))


def profiled_non_kv_bytes(tp: int, profiled_non_kv_gb: float) -> float:
    base = max(profiled_non_kv_gb, 0.0) * 1e9
    if tp <= 1:
        return base
    # Wider TP tends to need more per-GPU scratch and collective buffering.
    return base * (1.0 + 0.12 * math.log2(tp))


def per_gpu_weight_budget(g: GPU, mu: float, profiled_non_kv_gb: float, tp: int = 1) -> float:
    return max(g.mem * mu - profiled_non_kv_bytes(tp, profiled_non_kv_gb), 0.0)


def tp_supported(m: Model, tp: int) -> bool:
    if tp < 1 or m.num_heads % tp != 0:
        return False
    # MLA sharding constraints are model/runtime-specific. Be conservative until
    # we model them explicitly instead of allowing any TP because kv_heads == 1.
    if m.is_mla and not m.mla_tp_supported:
        return tp == 1
    kv_head_counts = {max(1, m.kv_heads)}
    if m.global_kv_heads > 0:
        kv_head_counts.add(m.global_kv_heads)
    if m.local_attention_layers > 0:
        kv_head_counts.add(m.local_kv_head_count)
    return all(
        tp <= heads and heads % tp == 0 or tp > heads and tp % heads == 0
        for heads in kv_head_counts
    )


def kv_duplication_groups_for_heads(kv_heads: int, tp: int) -> int:
    kv_heads = max(1, kv_heads)
    if tp <= kv_heads:
        return 1
    return tp // kv_heads


def kv_shards_for_heads(kv_heads: int, tp: int) -> int:
    return max(1, tp // kv_duplication_groups_for_heads(kv_heads, tp))


def kv_duplication_groups(m: Model, tp: int) -> int:
    if not tp_supported(m, tp):
        return 1
    return kv_duplication_groups_for_heads(m.kv_heads, tp)


def kv_shards(m: Model, tp: int) -> int:
    return kv_shards_for_heads(m.kv_heads, tp)


def compute_memory(
    m: Model,
    tp: int,
    pp: int,
    g: GPU,
    mu: float,
    profiled_non_kv_gb: float,
    prec: str,
    eff: EfficiencyParams,
    spec: Optional[SpecRuntime] = None,
) -> Optional[MemoryResult]:
    requested = g.mem * mu
    pp_fraction = _pp_peak_fraction(m, pp)
    weights = m.weight_bytes(prec) * pp_fraction / tp
    if spec is not None:
        # Drafter weights are resident whenever speculative decoding is enabled,
        # so they shrink the KV budget for prefill and decode alike.
        weights += spec.draft_weight_bytes * pp_fraction / tp
    profiled_non_kv = profiled_non_kv_bytes(tp, profiled_non_kv_gb)
    non_kv = weights + profiled_non_kv
    if non_kv > requested:
        return None
    kv_reserved = requested - non_kv
    kv_budget = kv_reserved * (1 - eff.kv_slack)
    return MemoryResult(
        requested=requested,
        weights=weights,
        profiled_non_kv=profiled_non_kv,
        kv_reserved=kv_reserved,
        kv_budget=kv_budget,
        kv_per_token=kv_bytes_per_token(m, prec) * pp_fraction / kv_shards(m, tp),
    )


def valid_strategies(
    m: Model,
    gpu_count: int,
    g: GPU,
    mu: float,
    profiled_non_kv_gb: float,
    prec: str,
    spec: Optional[SpecRuntime] = None,
) -> list[tuple[int, int, int]]:
    if gpu_count <= 0:
        return []

    result = []
    for tp, pp, dp in factor_triples(gpu_count):
        if pp > m.layers or not tp_supported(m, tp):
            continue
        budget = per_gpu_weight_budget(g, mu, profiled_non_kv_gb, tp)
        if budget <= 0:
            continue
        resident_weights = m.weight_bytes(prec)
        if spec is not None:
            resident_weights += spec.draft_weight_bytes
        if resident_weights * _pp_peak_fraction(m, pp) / tp <= budget:
            result.append((tp, pp, dp))

    return sorted(
        result,
        key=lambda s: (
            -s[2],
            s[0] > g.node_size,
            -min(s[0], g.node_size),
            s[1],
            s[0],
        ),
    )


def default_strategy(
    m: Model,
    gpu_count: int,
    g: GPU,
    mu: float,
    profiled_non_kv_gb: float,
    prec: str,
    spec: Optional[SpecRuntime] = None,
) -> tuple[int, int, int]:
    candidates = valid_strategies(m, gpu_count, g, mu, profiled_non_kv_gb, prec, spec)
    if not candidates:
        return (max(gpu_count, 1), 1, 1)

    best = candidates[0]
    best_score = None
    requested = g.mem * mu
    for tp, pp, dp in candidates:
        profiled_non_kv = profiled_non_kv_bytes(tp, profiled_non_kv_gb)
        resident_weights = m.weight_bytes(prec) + (
            spec.draft_weight_bytes if spec is not None else 0.0
        )
        kv_headroom = max(
            0.0, requested - (resident_weights * _pp_peak_fraction(m, pp) / tp) - profiled_non_kv
        )
        score = (
            1 if tp <= g.node_size else 0,
            min(tp, g.node_size),
            dp,
            -pp,
            kv_headroom,
        )
        if best_score is None or score > best_score:
            best = (tp, pp, dp)
            best_score = score
    return best


def _eff_collective_bw(tp: int, g: GPU) -> float:
    if tp <= g.node_size:
        return g.scale_up_collective_bw
    return INTER_NODE_COLLECTIVE_BW


def _pp_boundary_counts(tp: int, pp: int, g: GPU) -> tuple[int, int]:
    if pp <= 1:
        return 0, 0
    if tp > g.node_size:
        return 0, pp - 1

    intra = 0
    cross = 0
    node_idx = 0
    used_on_node = 0
    prev_node = None
    for _ in range(pp):
        if used_on_node + tp > g.node_size:
            node_idx += 1
            used_on_node = 0
        if prev_node is not None:
            if node_idx == prev_node:
                intra += 1
            else:
                cross += 1
        prev_node = node_idx
        used_on_node += tp
    return intra, cross


def _dense_tp_oh(
    tp: int, pp: int, batch_tokens: float, m: Model, g: GPU, bw_eff: float, overlap: float
) -> float:
    if tp <= 1:
        return 0.0
    collective_bw = _eff_collective_bw(tp, g) * bw_eff
    msg = batch_tokens * m.hidden_size * 2
    stage_layers = m.layers / pp
    # A standard tensor-parallel transformer block has one row-parallel reduction
    # after attention and one after the MLP. The ring factor below is per collective.
    collectives_per_layer = 2
    comm_time = stage_layers * collectives_per_layer * (msg * 2 * (tp - 1) / (tp * collective_bw))
    latency = stage_layers * collectives_per_layer * 3e-6
    return (comm_time + latency) * (1 - overlap)


def _pp_boundary_oh(
    tp: int, pp: int, batch_tokens: float, m: Model, g: GPU, bw_eff: float
) -> tuple[float, int]:
    intra, cross = _pp_boundary_counts(tp, pp, g)
    if intra + cross <= 0:
        return 0.0, 0
    msg = batch_tokens * m.hidden_size * 2
    intra_time = intra * (msg / (g.scale_up_collective_bw * bw_eff))
    cross_time = cross * (msg / (INTER_NODE_COLLECTIVE_BW * bw_eff))
    latency = (intra + cross) * 3e-6
    return intra_time + cross_time + latency, cross


def communication_breakdown(
    m: Model,
    tp: int,
    pp: int,
    batch_tokens: float,
    avg_seq: float,
    g: GPU,
    eff: EfficiencyParams,
) -> CommBreakdown:
    pp_boundary, pp_cross = _pp_boundary_oh(tp, pp, batch_tokens, m, g, eff.bw_eff)
    return CommBreakdown(
        dense_tp=_dense_tp_oh(tp, pp, batch_tokens, m, g, eff.bw_eff, eff.ar_overlap),
        pp_boundary=pp_boundary,
        tp_cross_node=tp > g.node_size,
        pp_cross_node_boundaries=pp_cross,
        ep_advisory=m.is_moe and (tp * pp > g.node_size),
        expert_parallel_unmodeled=m.is_moe,
        dcp_advisory=(
            getattr(m, "embedding_profile", None) is None
            and avg_seq >= LONG_CTX_DCP_SEQ
            and (tp > 1 or kv_duplication_groups(m, tp) > 1)
        ),
    )


def _moe_tail_multiplier(m: Model, eff: EfficiencyParams) -> float:
    return eff.moe_imbalance if m.is_moe else 1.0


def _active_weight_bytes(m: Model, prec: str) -> float:
    return m.active_weight_bytes(prec)


def _decode_step_time(
    m: Model,
    tp: int,
    pp: int,
    pr: int,
    g: GPU,
    prec: str,
    avg_seq: float,
    eff: EfficiencyParams,
    paged_oh: float = 0.0,
    extra_flops: float = 0.0,
    spec: Optional[SpecRuntime] = None,
) -> float:
    aw = _active_weight_bytes(m, prec)
    pp_fraction = _pp_peak_fraction(m, pp)
    wt = (aw * pp_fraction / tp) / (g.effective_bw * eff.bw_eff)
    base_token_kv_read_bytes = pr * per_replica_token_kv_cache_bytes(m, avg_seq, prec, pp, tp)
    recurrent_state_bytes = pr * per_replica_recurrent_state_bytes(m, prec, pp, tp)
    # A speculative verification pass forwards the k drafted positions. The
    # already-available target logit verifies the first draft and the final
    # forwarded position supplies the bonus token, so this is k (not k+1)
    # target positions in steady state. Target weights are reused once, while
    # KV reads, activation FLOPs, and collective payloads scale with k.
    verify_positions = max(spec.k, 1) if spec is not None else 1
    # A fused recurrent kernel advances all verified positions from one initial
    # state, then persists one final state.  Charge one read + one write per block.
    # This is conservative for ReplaySSM-style buffered stores and, unlike the old
    # combined-KV path, does not multiply fixed recurrent state by speculative k.
    state_traffic_bytes = 2.0 * recurrent_state_bytes
    kv_traffic_bytes = base_token_kv_read_bytes * verify_positions + state_traffic_bytes
    kv_time = kv_traffic_bytes / (g.effective_bw * eff.bw_eff)
    bt = wt + kv_time

    wf = 2 * m.active_params * pr * pp_fraction
    af = _decode_attention_work(m, pr, avg_seq, pp)
    flops = wf + af
    if spec is not None:
        # Verification reuses weights, but performs k positions of target work.
        flops *= verify_positions
    ct = (flops + max(extra_flops, 0.0)) / (model_gpu_flops(g, m, prec) * tp * eff.comp_eff)

    comm = communication_breakdown(m, tp, pp, pr * verify_positions, avg_seq, g, eff)
    # ``pr`` is the number of sequences in one continuous decode batch, not a
    # count of independent pipeline microbatches.  Treating it as the latter
    # makes the fill/drain multiplier shrink with concurrency and can therefore
    # claim that every user's autoregressive token latency improves when more
    # users arrive.  A decode iteration has a dependency barrier before those
    # sequences can request their next token, so model its end-to-end traversal
    # across all pipeline stages.  This is a conservative closed-form latency
    # approximation; it intentionally does not turn aggregate PP utilization
    # into a per-request latency reduction.
    base = max(bt, ct) * max(int(pp), 1)
    stage = base + comm.total
    if spec is not None:
        # Draft stage, sequential with verification: one pass for parallel
        # (block-diffusion) drafters, k autoregressive passes otherwise. The
        # drafter reads its own weights and its small KV cache (the profile's
        # fraction of the target per-sequence KV read) each pass. Draft
        # Draft collective payload/layer geometry is not available in the
        # catalog. Approximate it by scaling the target collective time by the
        # drafter/target active-parameter ratio. This stays small for MTP while
        # avoiding the previous assumption that draft collectives were free.
        # Autoregressive drafters process one position per pass; a parallel
        # block drafter processes all k positions in its pass. Keep that
        # distinction in both KV traffic and compute instead of treating a
        # length-k block as one token of work.
        draft_positions = spec.k if spec.profile.parallel_draft else 1
        draft_bt = (
            spec.draft_weight_bytes * pp_fraction / tp
            + (base_token_kv_read_bytes + recurrent_state_bytes)
            * spec.profile.kv_overhead
            * draft_positions
        ) / (g.effective_bw * eff.bw_eff)
        draft_ct = (2 * spec.draft_active_params * pr * pp_fraction * draft_positions) / (
            model_gpu_flops(g, m, prec) * tp * eff.comp_eff
        )
        stage += max(draft_bt, draft_ct) * max(int(pp), 1) * spec.passes
        draft_ratio = min(max(spec.draft_active_params / max(m.active_params, 1.0), 0.0), 1.0)
        # ``comm`` already carries k verification positions. Both an AR drafter
        # (k one-position passes) and a parallel drafter (one k-position pass)
        # communicate k draft positions in total, so do not multiply by passes.
        stage += comm.total * draft_ratio
        stage += (
            SPEC_SCHEDULER_OVERHEAD_S
            + SPEC_REJECTION_SYNC_OVERHEAD_S
            + SPEC_DRAFT_LAUNCH_OVERHEAD_S * spec.passes
        )
    step = stage * (1 + eff.overhead + paged_oh)
    return step * _moe_tail_multiplier(m, eff)


def _compute_decode_core(
    m: Model,
    tp: int,
    pp: int,
    bs: int,
    dp: int,
    g: GPU,
    mu: float,
    profiled_non_kv_gb: float,
    prec: str,
    avg_in: float,
    avg_out: float,
    eff: EfficiencyParams,
    paged_oh: float = 0.0,
    extra_flops: float = 0.0,
    spec: Optional[SpecRuntime] = None,
) -> Optional[DecodeResult]:
    if not context_supported(m, avg_in, avg_out):
        return None
    mem = compute_memory(m, tp, pp, g, mu, profiled_non_kv_gb, prec, eff, spec)
    if mem is None:
        return None

    dp = max(int(dp), 1)
    output_tokens = max(int(round(avg_out)), 1)
    active_spec = spec
    if spec is not None:
        effective_k = min(spec.k, output_tokens - 1)
        if effective_k <= 0:
            active_spec = None
        else:
            active_spec = SpecRuntime(
                profile=spec.profile,
                k=effective_k,
                alpha=spec.alpha,
                tau=spec_finite_output_tau(spec.alpha, effective_k, output_tokens),
                passes=1 if spec.profile.parallel_draft else effective_k,
                draft_weight_bytes=spec.draft_weight_bytes,
                draft_active_params=spec.draft_active_params,
                auto_selected=spec.auto_selected,
                probe_speedup=spec.probe_speedup,
            )
    active_replicas = min(max(int(bs), 0), dp)
    if active_replicas <= 0:
        return None
    base_load, extra_replicas = divmod(int(bs), active_replicas)
    replica_loads = [base_load + (1 if i < extra_replicas else 0) for i in range(active_replicas)]
    pr = max(replica_loads)
    avg_seq = avg_in + avg_out / 2.0
    avg_kv = per_replica_kv_cache_bytes(m, avg_seq, prec, pp, tp)
    if active_spec is not None:
        # The drafter keeps its own small KV cache per sequence on top of the
        # target's, which shrinks the number of slots that fit the KV budget.
        avg_kv *= 1.0 + active_spec.profile.kv_overhead
    max_slots = int(mem.kv_budget / avg_kv) if avg_kv > 0 else 0
    if eff.sched_budget > 0:
        max_slots = min(max_slots, eff.sched_budget)
    if pr > max_slots:
        return None

    # Sum independently loaded replica throughput. For realtime audio, callers pass
    # per-request extra work so uneven replicas receive proportional encoder work.
    replica_cycles = [
        _decode_step_time(
            m, tp, pp, load, g, prec, avg_seq, eff, paged_oh, extra_flops * load, active_spec
        )
        for load in replica_loads
    ]
    tau = active_spec.tau if active_spec is not None else 1.0
    total_tps = sum(
        load * tau / cycle for load, cycle in zip(replica_loads, replica_cycles) if cycle > 0
    )
    slowest_cycle = max(replica_cycles)
    # step_ms/lat stay per-token: with spec, one cycle emits tau tokens.
    per_token = slowest_cycle / tau
    spec_speedup = 1.0
    if spec is not None:
        baseline_step = _decode_step_time(
            m, tp, pp, pr, g, prec, avg_seq, eff, paged_oh, extra_flops * pr
        )
        spec_speedup = baseline_step / per_token if per_token > 0 else 1.0
    return DecodeResult(
        tps=round(total_tps),
        lat=round(per_token * 1e5) / 100,
        step_ms=round(per_token * 1e5) / 100,
        max_slots=max_slots * dp,
        spec_tau=tau if spec is not None else 0.0,
        spec_speedup=round(spec_speedup * 100) / 100,
    )


def compute_decode(
    m: Model,
    tp: int,
    pp: int,
    bs: int,
    dp: int,
    g: GPU,
    mu: float,
    profiled_non_kv_gb: float,
    prec: str,
    in_dist: list[int],
    out_dist: list[int],
    eff: EfficiencyParams,
    spec: Optional[SpecRuntime] = None,
) -> Optional[DecodeResult]:
    avg_in = avg_dist(in_dist, INPUT_BUCKETS)
    avg_out = avg_dist(out_dist, OUTPUT_BUCKETS)
    return _compute_decode_core(
        m,
        tp,
        pp,
        bs,
        dp,
        g,
        mu,
        profiled_non_kv_gb,
        prec,
        avg_in,
        avg_out,
        eff,
        paged_oh=decode_paged_oh(in_dist, out_dist, eff),
        spec=spec,
    )


def optimize_spec_k(
    m: Model,
    method: str,
    alpha_override: float,
    prec: str,
    tp: int,
    pp: int,
    dp: int,
    g: GPU,
    mu: float,
    profiled_non_kv_gb: float,
    in_dist: list[int],
    out_dist: list[int],
    eff: EfficiencyParams,
    probe_concurrency: int = SPEC_AUTO_K_PROBE_CONCURRENCY,
) -> SpecOptimization:
    """Select a fixed deployment k at a declared concurrency probe.

    Auto is a deployment decision, not a per-chart-point trick: the returned k
    is selected once and should be held across the workload curve. The search
    includes speculative decoding off, bounds k by both 32 and finite output
    length, and ignores candidates that do not fit the selected topology.
    """
    profile_runtime = resolve_spec_runtime(m, method, 1, alpha_override, prec)
    requested_probe = max(int(probe_concurrency), 1)
    if profile_runtime is None:
        return SpecOptimization(None, 0, 1.0, False, "method unavailable", requested_probe)

    output_tokens = max(int(round(avg_dist(out_dist, OUTPUT_BUCKETS))), 1)
    max_k = min(32, output_tokens - 1)
    if max_k < 1:
        return SpecOptimization(None, 0, 1.0, False, "output is too short for drafting", 1)

    # Discover capacity at one user first so an Auto @ 32 probe degrades to the
    # largest feasible declared operating point rather than failing wholesale.
    baseline_one = compute_decode(
        m,
        tp,
        pp,
        1,
        dp,
        g,
        mu,
        profiled_non_kv_gb,
        prec,
        in_dist,
        out_dist,
        eff,
    )
    if baseline_one is None or baseline_one.max_slots < 1:
        return SpecOptimization(None, 0, 1.0, False, "baseline topology is infeasible", 1)
    effective_probe = min(requested_probe, baseline_one.max_slots)
    baseline = compute_decode(
        m,
        tp,
        pp,
        effective_probe,
        dp,
        g,
        mu,
        profiled_non_kv_gb,
        prec,
        in_dist,
        out_dist,
        eff,
    )
    if baseline is None or baseline.tps <= 0:
        return SpecOptimization(
            None, 0, 1.0, False, "baseline probe is infeasible", effective_probe
        )

    best_runtime: Optional[SpecRuntime] = None
    best_speedup = 0.0
    profile_supported_ks = tuple(getattr(profile_runtime.profile, "supported_ks", ()))
    supported_ks = tuple(sorted({int(k) for k in profile_supported_ks if 1 <= int(k) <= max_k}))
    if profile_supported_ks and not supported_ks:
        return SpecOptimization(
            None,
            0,
            1.0,
            False,
            "no calibrated k fits the output length",
            effective_probe,
        )
    candidate_ks = supported_ks or tuple(range(1, max_k + 1))
    for k in candidate_ks:
        runtime = resolve_spec_runtime(m, method, k, alpha_override, prec)
        candidate = compute_decode(
            m,
            tp,
            pp,
            effective_probe,
            dp,
            g,
            mu,
            profiled_non_kv_gb,
            prec,
            in_dist,
            out_dist,
            eff,
            runtime,
        )
        if candidate is None:
            continue
        speedup = candidate.tps / baseline.tps
        if best_runtime is None or speedup > best_speedup:
            best_runtime = runtime
            best_speedup = speedup

    if best_runtime is None:
        return SpecOptimization(
            None, 0, 1.0, False, "no speculative k fits the topology", effective_probe
        )

    beneficial = best_speedup >= SPEC_MIN_BENEFICIAL_SPEEDUP
    selected = replace(
        best_runtime,
        auto_selected=True,
        probe_speedup=best_speedup,
    )
    reason = (
        f"best k at {effective_probe} users"
        if beneficial
        else f"spec off is faster at {effective_probe} users"
    )
    return SpecOptimization(
        selected,
        selected.k,
        best_speedup,
        beneficial,
        reason,
        effective_probe,
    )


def compute_decode_capacity(
    m: Model,
    tp: int,
    pp: int,
    dp: int,
    g: GPU,
    mu: float,
    profiled_non_kv_gb: float,
    prec: str,
    in_dist: list[int],
    out_dist: list[int],
    eff: EfficiencyParams,
    spec: Optional[SpecRuntime] = None,
) -> int:
    result = compute_decode(
        m, tp, pp, max(dp, 1), dp, g, mu, profiled_non_kv_gb, prec, in_dist, out_dist, eff, spec
    )
    return result.max_slots if result else 0


def compute_prefill(
    m: Model,
    tp: int,
    pp: int,
    bs: int,
    dp: int,
    seq_len: int,
    g: GPU,
    mu: float,
    profiled_non_kv_gb: float,
    prec: str,
    eff: EfficiencyParams,
    spec: Optional[SpecRuntime] = None,
) -> Optional[PrefillResult]:
    if not context_supported(m, seq_len):
        return None
    mem = compute_memory(m, tp, pp, g, mu, profiled_non_kv_gb, prec, eff, spec)
    if mem is None:
        return None
    if seq_len <= 0:
        # A 100% prefix hit removes prefill work. Use the planner's finite sentinel
        # instead of infinity so charts/JSON remain numerically well-defined.
        return PrefillResult(
            tps=0, service_time=0.0, rps=float(UNBOUNDED_BATCH), max_batch=UNBOUNDED_BATCH
        )

    pr = math.ceil(bs / dp)
    seq_kv = per_replica_kv_cache_bytes(m, seq_len, prec, pp, tp)
    if spec is not None:
        # Reserve the drafter's prompt/hidden-state cache consistently with
        # decode capacity. Some attached drafters reuse target hidden states;
        # the profile overhead is the catalog's measured/estimated aggregate.
        seq_kv *= 1.0 + spec.profile.kv_overhead
    # AttnRes activations are temporary rather than KV, but they scale with
    # batch×sequence and therefore must participate in the prefill fit check.
    per_sequence_memory = seq_kv + attention_residual_scratch_bytes(m, seq_len, prec, pp, tp)
    max_per_replica = int(mem.kv_budget / per_sequence_memory) if per_sequence_memory > 0 else 0
    if pr > max_per_replica:
        return None

    pp_fraction = _pp_peak_fraction(m, pp)
    ffn = 2 * m.active_params * pr * seq_len * pp_fraction
    att = _prefill_attention_work(m, pr, seq_len, pp)
    tf = ffn + att
    ct = tf / (model_gpu_flops(g, m, prec) * tp * eff.comp_eff)

    aw = _active_weight_bytes(m, prec)
    mt = (aw * pp_fraction / tp) / (g.effective_bw * eff.bw_eff)

    comm = communication_breakdown(m, tp, pp, pr * seq_len, seq_len, g, eff)
    base = max(ct, mt) * _pp_bubble_multiplier(pp, pr)
    t = (base + comm.total) * (1 + eff.overhead * 1.3 + fixed_paged_oh(seq_len, eff, 0.35))
    t *= _moe_tail_multiplier(m, eff)
    rps = bs / t if t > 0 else 0.0
    return PrefillResult(
        tps=round(rps * seq_len),
        service_time=t,
        rps=rps,
        max_batch=max_per_replica * dp,
    )


def embedding_sequence_length(m: Model, requested_seq_len: int) -> int:
    profile = getattr(m, "embedding_profile", None)
    if profile is None:
        return 0
    max_len = max(int(profile.max_sequence_length or 0), 1)
    return max(1, min(max(int(requested_seq_len or 0), 1), max_len))


def embedding_vectors_per_input(m: Model, seq_len: int) -> int:
    profile = getattr(m, "embedding_profile", None)
    if profile is None:
        return 0
    if profile.supports_late_interaction:
        max_vectors = int(profile.document_length or profile.max_sequence_length or seq_len)
        token_vectors = max(1, min(max(int(seq_len), 1), max_vectors))
        return token_vectors + (1 if profile.supports_single_vector else 0)
    return 1


def embedding_output_bytes_per_input(m: Model, seq_len: int) -> float:
    profile = getattr(m, "embedding_profile", None)
    if profile is None:
        return 0.0
    vectors = embedding_vectors_per_input(m, seq_len)
    dim = (
        int(profile.late_interaction_dim or profile.output_dim)
        if profile.supports_late_interaction
        else int(profile.output_dim)
    )
    return vectors * max(dim, 1) * max(float(profile.vector_bytes_per_elem), 0.25)


def _embedding_weighted_sequences(
    m: Model,
    doc_dist: list[int],
    buckets: list[Bucket] = EMBEDDING_DOC_BUCKETS,
) -> list[tuple[float, int, Bucket]]:
    values = _aligned_dist(doc_dist, buckets)
    weights = normalize_dist(values)
    return [
        (weights[i], embedding_sequence_length(m, bucket.length), bucket)
        for i, bucket in enumerate(buckets)
        if weights[i] > 0
    ]


def _embedding_percentile(weighted_sequences: list[tuple[float, int, Bucket]], pct: float) -> int:
    pct = min(max(pct, 0.0), 1.0)
    cdf = 0.0
    for share, seq_len, _bucket in weighted_sequences:
        cdf += share
        if pct <= cdf + 1e-9:
            return seq_len
    return weighted_sequences[-1][1] if weighted_sequences else 0


def embedding_doc_stats(
    m: Model,
    doc_dist: list[int],
    buckets: list[Bucket] = EMBEDDING_DOC_BUCKETS,
    prec: str = "bf16",
) -> EmbeddingDocStats:
    weighted = _embedding_weighted_sequences(m, doc_dist, buckets)
    if not weighted:
        return EmbeddingDocStats(0.0, 0, 0, 0, 0.0, 0.0, 0.0)

    mean_seq = sum(share * seq for share, seq, _bucket in weighted)
    mean_vectors = sum(
        share * embedding_vectors_per_input(m, seq) for share, seq, _bucket in weighted
    )
    mean_output = sum(
        share * embedding_output_bytes_per_input(m, seq) for share, seq, _bucket in weighted
    )
    mean_scratch = sum(
        share * embedding_scratch_bytes_per_input(m, seq, prec) for share, seq, _bucket in weighted
    )
    return EmbeddingDocStats(
        mean_seq_len=mean_seq,
        p50_seq_len=_embedding_percentile(weighted, 0.50),
        p90_seq_len=_embedding_percentile(weighted, 0.90),
        p99_seq_len=_embedding_percentile(weighted, 0.99),
        mean_vectors_per_input=mean_vectors,
        mean_output_bytes_per_input=mean_output,
        mean_scratch_bytes_per_input=mean_scratch,
    )


def embedding_scratch_bytes_per_input(m: Model, seq_len: int, prec: str) -> float:
    """Approximate inference scratch for one encoder item.

    Embedding inference does not reserve decode KV slots. The remaining memory after
    weights is instead used as a batch-sized activation/output work buffer. Flash attention
    keeps the attention side close to linear memory, so this intentionally models a compact
    forward buffer instead of training-style saved activations.
    """
    hidden = max(m.hidden_size, m.attention_query_head_count * max(m.head_dim, 1), 1)
    bpe = m.kv_cache_bytes_per_elem(prec)
    activation = seq_len * hidden * bpe * 4.0
    return activation + embedding_output_bytes_per_input(m, seq_len)


def compute_embedding(
    m: Model,
    strat: tuple[int, int, int],
    bs: int,
    seq_len: int,
    g: GPU,
    mu: float,
    profiled_non_kv_gb: float,
    prec: str,
    eff: EfficiencyParams,
) -> Optional[EmbeddingResult]:
    if getattr(m, "embedding_profile", None) is None:
        return None

    tp, pp, dp = strat
    seq = embedding_sequence_length(m, seq_len)
    mem = compute_memory(m, tp, pp, g, mu, profiled_non_kv_gb, prec, eff)
    if mem is None:
        return None

    pr = math.ceil(bs / max(dp, 1))
    scratch_per = embedding_scratch_bytes_per_input(m, seq, prec)
    max_per_replica = int(mem.kv_budget / scratch_per) if scratch_per > 0 else UNBOUNDED_BATCH
    if pr > max_per_replica:
        return None

    pp_fraction = _pp_peak_fraction(m, pp)
    ffn = 2 * m.active_params * pr * seq * pp_fraction
    att = _prefill_attention_work(m, pr, seq, pp)
    ct = (ffn + att) / (model_gpu_flops(g, m, prec) * max(tp, 1) * eff.comp_eff)

    aw = _active_weight_bytes(m, prec)
    mt = (aw * pp_fraction / max(tp, 1)) / (g.effective_bw * eff.bw_eff)
    output_time = (embedding_output_bytes_per_input(m, seq) * pr) / (g.effective_bw * eff.bw_eff)
    comm = communication_breakdown(m, tp, pp, pr * seq, seq, g, eff)

    t = (max(ct, mt) + output_time + comm.total) * (
        1 + eff.overhead * 1.2 + fixed_paged_oh(seq, eff, 0.20)
    )
    if t <= 0:
        return None

    rps = bs / t
    vectors_per_input = embedding_vectors_per_input(m, seq)
    output_bytes = embedding_output_bytes_per_input(m, seq)
    output_bps = rps * output_bytes
    return EmbeddingResult(
        rps=round(rps * 100) / 100,
        tps=round(rps * seq),
        vectors_per_second=round(rps * vectors_per_input),
        output_mb_s=round((output_bps / 1e6) * 100) / 100,
        service_time=t,
        max_batch=max_per_replica * max(dp, 1),
        seq_len=seq,
        vectors_per_input=vectors_per_input,
        p50_seq_len=seq,
        p90_seq_len=seq,
        p99_seq_len=seq,
        output_bytes_per_input=output_bytes,
    )


def compute_embedding_distribution(
    m: Model,
    strat: tuple[int, int, int],
    bs: int,
    doc_dist: list[int],
    buckets: list[Bucket],
    g: GPU,
    mu: float,
    profiled_non_kv_gb: float,
    prec: str,
    eff: EfficiencyParams,
) -> Optional[EmbeddingResult]:
    if getattr(m, "embedding_profile", None) is None:
        return None

    tp, pp, dp = strat
    stats = embedding_doc_stats(m, doc_dist, buckets, prec)
    if stats.mean_seq_len <= 0:
        return None

    mem = compute_memory(m, tp, pp, g, mu, profiled_non_kv_gb, prec, eff)
    if mem is None:
        return None

    pr = math.ceil(bs / max(dp, 1))
    max_per_replica = (
        int(mem.kv_budget / stats.mean_scratch_bytes_per_input)
        if stats.mean_scratch_bytes_per_input > 0
        else UNBOUNDED_BATCH
    )
    if pr > max_per_replica:
        return None

    pp_fraction = _pp_peak_fraction(m, pp)
    ffn = 2 * m.active_params * pr * stats.mean_seq_len * pp_fraction
    att = sum(
        share * _prefill_attention_work(m, pr, seq, pp)
        for share, seq, _bucket in _embedding_weighted_sequences(m, doc_dist, buckets)
    )
    ct = (ffn + att) / (model_gpu_flops(g, m, prec) * max(tp, 1) * eff.comp_eff)

    aw = _active_weight_bytes(m, prec)
    mt = (aw * pp_fraction / max(tp, 1)) / (g.effective_bw * eff.bw_eff)
    output_time = (stats.mean_output_bytes_per_input * pr) / (g.effective_bw * eff.bw_eff)
    comm = communication_breakdown(m, tp, pp, pr * stats.mean_seq_len, stats.mean_seq_len, g, eff)

    t = (max(ct, mt) + output_time + comm.total) * (
        1 + eff.overhead * 1.2 + fixed_paged_oh(stats.mean_seq_len, eff, 0.20)
    )
    if t <= 0:
        return None

    rps = bs / t
    output_bps = rps * stats.mean_output_bytes_per_input
    return EmbeddingResult(
        rps=round(rps * 100) / 100,
        tps=round(rps * stats.mean_seq_len),
        vectors_per_second=round(rps * stats.mean_vectors_per_input),
        output_mb_s=round((output_bps / 1e6) * 100) / 100,
        service_time=t,
        max_batch=max_per_replica * max(dp, 1),
        seq_len=round(stats.mean_seq_len),
        vectors_per_input=stats.mean_vectors_per_input,
        p50_seq_len=stats.p50_seq_len,
        p90_seq_len=stats.p90_seq_len,
        p99_seq_len=stats.p99_seq_len,
        output_bytes_per_input=stats.mean_output_bytes_per_input,
    )


def compute_embedding_capacity(
    m: Model,
    strat: tuple[int, int, int],
    seq_len: int,
    g: GPU,
    mu: float,
    profiled_non_kv_gb: float,
    prec: str,
    eff: EfficiencyParams,
) -> int:
    result = compute_embedding(
        m, strat, max(strat[2], 1), seq_len, g, mu, profiled_non_kv_gb, prec, eff
    )
    return result.max_batch if result else 0


def compute_embedding_distribution_capacity(
    m: Model,
    strat: tuple[int, int, int],
    doc_dist: list[int],
    buckets: list[Bucket],
    g: GPU,
    mu: float,
    profiled_non_kv_gb: float,
    prec: str,
    eff: EfficiencyParams,
) -> int:
    result = compute_embedding_distribution(
        m,
        strat,
        max(strat[2], 1),
        doc_dist,
        buckets,
        g,
        mu,
        profiled_non_kv_gb,
        prec,
        eff,
    )
    return result.max_batch if result else 0


def compute_data(
    m: Model,
    prefill_strat: tuple[int, int, int],
    decode_strat: tuple[int, int, int],
    bs: int,
    in_len: int,
    out_len: int,
    g: GPU,
    mu: float,
    profiled_non_kv_gb: float,
    prec: str,
    prefix_hit_rate: float,
    prefill_eff: EfficiencyParams,
    decode_eff: EfficiencyParams,
    spec: Optional[SpecRuntime] = None,
) -> Optional[DataResult]:
    if not context_supported(m, in_len, out_len):
        return None
    # One assignment owns one GPU pool and one memory reservation. Independent P/D
    # layouts would require two explicitly allocated pools; accepting them here would
    # count the same GPUs and VRAM twice.
    if prefill_strat != decode_strat:
        return None
    prefill_tp, prefill_pp, prefill_dp = prefill_strat
    decode_tp, decode_pp, decode_dp = decode_strat
    pf_in = effective_prefill_length(in_len, prefix_hit_rate)
    pf = compute_prefill(
        m,
        prefill_tp,
        prefill_pp,
        bs,
        prefill_dp,
        pf_in,
        g,
        mu,
        profiled_non_kv_gb,
        prec,
        prefill_eff,
        spec,
    )
    if pf is None:
        return None

    dec = _compute_decode_core(
        m,
        decode_tp,
        decode_pp,
        bs,
        decode_dp,
        g,
        mu,
        profiled_non_kv_gb,
        prec,
        in_len,
        out_len,
        decode_eff,
        paged_oh=fixed_paged_oh(in_len + out_len / 2.0, decode_eff),
        spec=spec,
    )
    if dec is None:
        return None

    decode_time = out_len * dec.step_ms / 1000
    interference = min(max(max(prefill_eff.pd_interference, decode_eff.pd_interference), 0.0), 1.0)
    overlap_time = max(pf.service_time, decode_time)
    total_time = overlap_time + ((pf.service_time + decode_time) - overlap_time) * interference
    rps = bs / total_time if total_time > 0 else 0.0
    return DataResult(
        rps=round(rps * 100) / 100,
        tps=round(rps * (in_len + out_len)),
        prefill_frac=pf.service_time / total_time if total_time > 0 else 0.0,
    )


def compute_data_capacity(
    m: Model,
    prefill_strat: tuple[int, int, int],
    decode_strat: tuple[int, int, int],
    in_len: int,
    out_len: int,
    g: GPU,
    mu: float,
    profiled_non_kv_gb: float,
    prec: str,
    prefix_hit_rate: float,
    prefill_eff: EfficiencyParams,
    decode_eff: EfficiencyParams,
    spec: Optional[SpecRuntime] = None,
) -> int:
    if not context_supported(m, in_len, out_len):
        return 0
    if prefill_strat != decode_strat:
        return 0
    prefill_tp, prefill_pp, prefill_dp = prefill_strat
    decode_tp, decode_pp, decode_dp = decode_strat
    pf_in = effective_prefill_length(in_len, prefix_hit_rate)
    pf = compute_prefill(
        m,
        prefill_tp,
        prefill_pp,
        max(prefill_dp, 1),
        prefill_dp,
        pf_in,
        g,
        mu,
        profiled_non_kv_gb,
        prec,
        prefill_eff,
        spec,
    )
    if pf is None:
        return 0

    dec = _compute_decode_core(
        m,
        decode_tp,
        decode_pp,
        max(decode_dp, 1),
        decode_dp,
        g,
        mu,
        profiled_non_kv_gb,
        prec,
        in_len,
        out_len,
        decode_eff,
        paged_oh=fixed_paged_oh(in_len + out_len / 2.0, decode_eff),
        spec=spec,
    )
    if dec is None:
        return 0

    return min(pf.max_batch, dec.max_slots)


def compute_user_experience(
    m: Model,
    prefill_strat: tuple[int, int, int],
    decode_strat: tuple[int, int, int],
    bs: int,
    g: GPU,
    mu: float,
    profiled_non_kv_gb: float,
    prec: str,
    in_dist: list[int],
    out_dist: list[int],
    prefix_hit_rate: float,
    prefill_eff: EfficiencyParams,
    decode_eff: EfficiencyParams,
    spec: Optional[SpecRuntime] = None,
) -> Optional[UserExperienceResult]:
    if prefill_strat != decode_strat:
        return None
    decode_tp, decode_pp, decode_dp = decode_strat
    prefill_tp, prefill_pp, prefill_dp = prefill_strat
    dec = compute_decode(
        m,
        decode_tp,
        decode_pp,
        bs,
        decode_dp,
        g,
        mu,
        profiled_non_kv_gb,
        prec,
        in_dist,
        out_dist,
        decode_eff,
        spec,
    )
    if dec is None:
        return None

    avg_in = avg_dist(in_dist, INPUT_BUCKETS)
    avg_out = avg_dist(out_dist, OUTPUT_BUCKETS)
    pf_in = effective_prefill_length(avg_in, prefix_hit_rate)
    pf = compute_prefill(
        m,
        prefill_tp,
        prefill_pp,
        bs,
        prefill_dp,
        pf_in,
        g,
        mu,
        profiled_non_kv_gb,
        prec,
        prefill_eff,
        spec,
    )
    if pf is None:
        return None

    ttft_ms = pf.service_time * 1000
    decode_time = avg_out * dec.step_ms / 1000
    response_s = (ttft_ms / 1000) + decode_time
    interference = min(max(max(prefill_eff.pd_interference, decode_eff.pd_interference), 0.0), 1.0)
    overlap_time = max(pf.service_time, decode_time)
    cycle_time = overlap_time + ((pf.service_time + decode_time) - overlap_time) * interference
    arrival_rps = bs / cycle_time if cycle_time > 0 else 0.0
    return UserExperienceResult(
        arrival_rps=round(arrival_rps * 100) / 100,
        decode_step_ms=dec.step_ms,
        ttft_ms=round(ttft_ms * 10) / 10,
        response_s=round(response_s * 100) / 100,
        inflight=float(bs),
    )


def compute_realtime_capacity(
    m: Model,
    decode_strat: tuple[int, int, int],
    users: int,
    g: GPU,
    mu: float,
    profiled_non_kv_gb: float,
    prec: str,
    eff: EfficiencyParams,
) -> Optional[RealtimeCapacityResult]:
    profile = getattr(m, "realtime_profile", None)
    if profile is None or users <= 0:
        return None

    tp, pp, dp = decode_strat
    state_tokens = max(float(profile.state_tokens), 1.0)
    # Pass one user's incremental encoder work; the decode core scales it by
    # each unevenly loaded replica's actual user count.
    extra_flops = _realtime_audio_encoder_work(profile, 1, pp)
    result = _compute_decode_core(
        m,
        tp,
        pp,
        users,
        dp,
        g,
        mu,
        profiled_non_kv_gb,
        prec,
        state_tokens,
        0.0,
        eff,
        paged_oh=fixed_paged_oh(state_tokens, eff, 0.5),
        extra_flops=extra_flops,
    )
    if result is None:
        return None

    required_tps = max(float(profile.tokens_per_second), 1e-9)
    per_user_tps = result.tps / max(float(users), 1.0)
    realtime_factor = per_user_tps / required_tps
    return RealtimeCapacityResult(
        users=users,
        realtime_factor=round(realtime_factor * 1000) / 1000,
        per_user_tps=round(per_user_tps * 100) / 100,
        total_tps=result.tps,
        step_ms=result.step_ms,
        max_slots=result.max_slots,
        required_tps=required_tps,
    )


def compute_realtime_max_users(
    m: Model,
    decode_strat: tuple[int, int, int],
    g: GPU,
    mu: float,
    profiled_non_kv_gb: float,
    prec: str,
    eff: EfficiencyParams,
) -> int:
    if getattr(m, "realtime_profile", None) is None:
        return 0

    best = 0
    high = 1
    while high <= MAX_REALTIME_USERS:
        result = compute_realtime_capacity(
            m, decode_strat, high, g, mu, profiled_non_kv_gb, prec, eff
        )
        if result is not None and result.realtime_factor >= 1.0:
            best = high
            high *= 2
            continue
        break

    if high > MAX_REALTIME_USERS:
        return MAX_REALTIME_USERS

    lo = best + 1
    hi = max(best, high - 1)
    while lo <= hi:
        mid = (lo + hi) // 2
        result = compute_realtime_capacity(
            m, decode_strat, mid, g, mu, profiled_non_kv_gb, prec, eff
        )
        if result is not None and result.realtime_factor >= 1.0:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _label(
    am,
    model: Model,
    panel_suffix: str = "",
    include_prefill: bool = False,
    spec_meta: Optional[dict] = None,
) -> str:
    decode_label = strategy_label(am.tp, am.pp, am.dp)
    spec_suffix = ""
    if spec_meta is not None:
        if spec_meta["spec_method"] == "off":
            spec_suffix = " · Spec off"
        elif spec_meta["spec_beneficial"]:
            auto = " Auto→" if spec_meta["spec_auto"] else " "
            spec_suffix = (
                f" · {spec_meta['spec_method'].upper()}{auto}k={spec_meta['spec_k']} "
                f"α={spec_meta['spec_alpha']:.0%}"
            )
        else:
            spec_suffix = (
                f" · Spec off (Auto {spec_meta['spec_method'].upper()} @"
                f"{spec_meta['spec_probe_concurrency']}, best k={spec_meta['spec_k']})"
            )
    if include_prefill:
        prefill_label = strategy_label(am.prefill_tp, am.prefill_pp, am.prefill_dp)
        if prefill_label != decode_label:
            return f"{model.name} P {prefill_label} / D {decode_label} {am.prec.upper()}{spec_suffix}{panel_suffix}"
    return f"{model.name} {decode_label} {am.prec.upper()}{spec_suffix}{panel_suffix}"


def _spec_chart_selection(state, am, model: Model) -> tuple[Optional[SpecRuntime], dict]:
    method = getattr(am, "spec_method", "off") or "off"
    auto = method != "off" and getattr(am, "spec_k", 0) == 0
    selection = spec_optimization_for(state, am, model) if auto else None
    runtime = (
        selection.runtime
        if selection is not None and selection.beneficial
        else None
        if selection is not None
        else resolve_spec_runtime(
            model,
            method,
            getattr(am, "spec_k", 0),
            getattr(state, "spec_acceptance", 0.0),
            am.prec,
        )
    )
    disclosed_runtime = selection.runtime if selection is not None else runtime
    meta = {
        "spec_method": method,
        "spec_k": disclosed_runtime.k if disclosed_runtime is not None else 0,
        "spec_alpha": disclosed_runtime.alpha if disclosed_runtime is not None else 0.0,
        "spec_auto": auto,
        "spec_speedup": selection.speedup
        if selection is not None
        else (disclosed_runtime.probe_speedup if disclosed_runtime is not None else 1.0),
        "spec_beneficial": selection.beneficial if selection is not None else runtime is not None,
        "spec_probe_concurrency": selection.probe_concurrency if selection is not None else 0,
        "spec_reason": selection.reason if selection is not None else "",
    }
    return runtime, meta


def _batch_axis_sweep(capacities: list[int], fallback: list[int]) -> list[int]:
    caps = sorted({c for c in capacities if 0 < c < UNBOUNDED_BATCH})
    sweep = set(fallback)
    if not caps:
        return sorted(sweep)

    target = max(max(sweep, default=1), max(2, math.ceil(caps[-1] * (1 + BATCH_AXIS_HEADROOM))))
    value = 1
    while value <= target:
        sweep.add(value)
        value *= 2

    sweep.update(caps)
    sweep.add(target)
    return sorted(sweep)


def _embedding_doc_dist_for_state(state) -> list[int]:
    dist = getattr(state, "embedding_doc_dist", None)
    if isinstance(dist, list) and len(dist) == len(EMBEDDING_DOC_BUCKETS):
        return dist

    seq_len = max(int(getattr(state, "task_il", 2048) or 2048), 1)
    nearest = min(
        range(len(EMBEDDING_DOC_BUCKETS)),
        key=lambda i: abs(EMBEDDING_DOC_BUCKETS[i].length - seq_len),
    )
    fallback = [0] * len(EMBEDDING_DOC_BUCKETS)
    fallback[nearest] = 100
    return fallback


def _iter_resolved_models(state):
    for am in state.models:
        if am.gpu_count <= 0:
            continue
        gp = state.find_gpu(am.gpu_uid)
        if gp is None:
            continue
        yield am, gp.gpu


def _is_decode_pareto_model(model: Model) -> bool:
    return (
        getattr(model, "embedding_profile", None) is None
        and getattr(model, "realtime_profile", None) is None
    )


def get_decode_bs(
    states: Optional[list] = None, *, deployments: Optional[list[Deployment]] = None
) -> list[int]:
    if not states:
        return list(BATCH_SIZES)
    if deployments is None or len(deployments) != len(states):
        raise ValueError("A resolved deployment is required for each planner state.")

    capacities = []
    for state, deployment in zip(states, deployments):
        eff = state.decode_efficiency
        for am in deployment.decode:
            if not _is_decode_pareto_model(am.model):
                continue
            gpu = am.gpu_spec
            if gpu is None:
                continue
            capacities.append(
                compute_decode_capacity(
                    am.model,
                    am.tp,
                    am.pp,
                    am.dp,
                    gpu,
                    state.mu,
                    state.profiled_non_kv_gb,
                    am.prec,
                    state.in_dist,
                    state.out_dist,
                    eff,
                    spec_runtime_for(state, am, am.model),
                )
            )
    return _batch_axis_sweep(capacities, BATCH_SIZES)


def get_realtime_bs(
    states: Optional[list] = None, *, deployments: Optional[list[Deployment]] = None
) -> list[int]:
    if not states:
        return list(REALTIME_USER_SWEEP)
    if deployments is None or len(deployments) != len(states):
        raise ValueError("A resolved deployment is required for each planner state.")

    capacities = []
    for state, deployment in zip(states, deployments):
        for am in deployment.decode:
            if getattr(am.model, "realtime_profile", None) is None:
                continue
            gpu = am.gpu_spec
            if gpu is None:
                continue
            capacities.append(
                compute_realtime_max_users(
                    am.model,
                    (am.tp, am.pp, am.dp),
                    gpu,
                    state.mu,
                    state.profiled_non_kv_gb,
                    am.prec,
                    state.decode_efficiency,
                )
            )
    return _batch_axis_sweep(capacities, REALTIME_USER_SWEEP)


def get_embedding_bs(states: Optional[list] = None) -> list[int]:
    if not states:
        return list(EMBEDDING_BATCH_SIZES)

    capacities = []
    for state in states:
        doc_dist = _embedding_doc_dist_for_state(state)
        for am, gpu in _iter_resolved_models(state):
            if getattr(am.model, "embedding_profile", None) is None:
                continue
            capacities.append(
                compute_embedding_distribution_capacity(
                    am.model,
                    (am.prefill_tp, am.prefill_pp, am.prefill_dp),
                    doc_dist,
                    EMBEDDING_DOC_BUCKETS,
                    gpu,
                    state.mu,
                    state.profiled_non_kv_gb,
                    am.prec,
                    state.prefill_efficiency,
                )
            )
    return _batch_axis_sweep(capacities, EMBEDDING_BATCH_SIZES)


def get_data_bs(states: Optional[list] = None) -> list[int]:
    if not states:
        return list(DATA_BATCH_SIZES)

    capacities = []
    for state in states:
        for am, gpu in _iter_resolved_models(state):
            if getattr(am.model, "embedding_profile", None) is not None:
                continue
            capacities.append(
                compute_data_capacity(
                    am.model,
                    (am.prefill_tp, am.prefill_pp, am.prefill_dp),
                    (am.tp, am.pp, am.dp),
                    state.task_il,
                    state.task_ol,
                    gpu,
                    state.mu,
                    state.profiled_non_kv_gb,
                    am.prec,
                    state.prefix_hit_rate,
                    state.prefill_efficiency,
                    state.decode_efficiency,
                    spec_runtime_for(state, am, am.model),
                )
            )
    return _batch_axis_sweep(capacities, DATA_BATCH_SIZES)


def spec_runtime_for(state, am, m: Model) -> Optional[SpecRuntime]:
    """Resolve an assignment's speculative-decoding config against global state."""
    method = getattr(am, "spec_method", "off")
    spec_k = getattr(am, "spec_k", 0)
    if method not in ("", "off") and spec_k == 0:
        selection = spec_optimization_for(state, am, m)
        # Off participates in Auto selection. Keep the assignment's method and
        # k=0 sentinel untouched so the UI can disclose why Auto chose Off.
        return selection.runtime if selection.beneficial else None
    return resolve_spec_runtime(
        m,
        method,
        spec_k,
        getattr(state, "spec_acceptance", 0.0),
        am.prec,
    )


def spec_optimization_for(state, am, m: Model) -> SpecOptimization:
    """State adapter for callers that need Auto's k, probe, and warning reason."""
    gpu = getattr(am, "gpu_spec", None)
    if gpu is None:
        gp = state.find_gpu(am.gpu_uid) if hasattr(state, "find_gpu") else None
        gpu = gp.gpu if gp is not None else None
    if gpu is None:
        return SpecOptimization(
            None, 0, 1.0, False, "GPU unavailable", SPEC_AUTO_K_PROBE_CONCURRENCY
        )
    return optimize_spec_k(
        m,
        getattr(am, "spec_method", "off"),
        getattr(state, "spec_acceptance", 0.0),
        am.prec,
        am.tp,
        am.pp,
        am.dp,
        gpu,
        state.mu,
        state.profiled_non_kv_gb,
        state.in_dist,
        state.out_dist,
        state.decode_efficiency,
        getattr(state, "spec_probe_concurrency", SPEC_AUTO_K_PROBE_CONCURRENCY),
    )


def get_processing_pareto_bs(states: Optional[list] = None) -> list[int]:
    if not states:
        return list(DATA_BATCH_SIZES)

    capacities = []
    for state in states:
        for preset in DIST_PRESETS.values():
            in_len = avg_dist(preset["in"], INPUT_BUCKETS)
            out_len = avg_dist(preset["out"], OUTPUT_BUCKETS)
            for am, gpu in _iter_resolved_models(state):
                if getattr(am.model, "embedding_profile", None) is not None:
                    continue
                capacities.append(
                    compute_data_capacity(
                        am.model,
                        (am.prefill_tp, am.prefill_pp, am.prefill_dp),
                        (am.tp, am.pp, am.dp),
                        in_len,
                        out_len,
                        gpu,
                        state.mu,
                        state.profiled_non_kv_gb,
                        am.prec,
                        state.prefix_hit_rate,
                        state.prefill_efficiency,
                        state.decode_efficiency,
                        spec_runtime_for(state, am, am.model),
                    )
                )
    return _batch_axis_sweep(capacities, DATA_BATCH_SIZES)


def _user_exp_curve(
    m: Model,
    prefill_strat: tuple[int, int, int],
    decode_strat: tuple[int, int, int],
    g: GPU,
    mu: float,
    profiled_non_kv_gb: float,
    prec: str,
    in_dist: list[int],
    out_dist: list[int],
    prefix_hit_rate: float,
    prefill_eff: EfficiencyParams,
    decode_eff: EfficiencyParams,
    spec: Optional[SpecRuntime] = None,
) -> list[dict]:
    points: list[dict] = []
    for users in USER_EXP_SWEEP:
        result = compute_user_experience(
            m,
            prefill_strat,
            decode_strat,
            users,
            g,
            mu,
            profiled_non_kv_gb,
            prec,
            in_dist,
            out_dist,
            prefix_hit_rate,
            prefill_eff,
            decode_eff,
            spec,
        )
        if not result or result.arrival_rps <= 0:
            continue
        point = {
            "x": result.arrival_rps,
            "y": result.response_s,
            "arrival_rps": result.arrival_rps,
            "response_s": result.response_s,
            "inflight": result.inflight,
            "ttft_ms": result.ttft_ms,
            "decode_step_ms": result.decode_step_ms,
        }
        if points and point["arrival_rps"] <= points[-1]["arrival_rps"]:
            continue
        points.append(point)
    return points


def _sample_user_exp_curve(points: list[dict], target_rps: float) -> Optional[dict]:
    if not points or target_rps <= 0 or target_rps > points[-1]["arrival_rps"]:
        return None
    if target_rps <= points[0]["arrival_rps"]:
        point = points[0]
        return {
            "arrival_rps": round(target_rps * 100) / 100,
            "response_s": point["response_s"],
            "inflight": round(target_rps * point["response_s"], 1),
            "ttft_ms": point["ttft_ms"],
            "decode_step_ms": point["decode_step_ms"],
        }

    left = points[0]
    right = points[-1]
    for candidate in points[1:]:
        if target_rps <= candidate["arrival_rps"]:
            right = candidate
            break
        left = candidate

    span = right["arrival_rps"] - left["arrival_rps"]
    t = 0.0 if span <= 0 else (target_rps - left["arrival_rps"]) / span
    response_s = left["response_s"] + (right["response_s"] - left["response_s"]) * t
    ttft_ms = left["ttft_ms"] + (right["ttft_ms"] - left["ttft_ms"]) * t
    decode_step_ms = left["decode_step_ms"] + (right["decode_step_ms"] - left["decode_step_ms"]) * t
    return {
        "arrival_rps": round(target_rps * 100) / 100,
        "response_s": round(response_s * 100) / 100,
        "inflight": round(target_rps * response_s, 1),
        "ttft_ms": round(ttft_ms, 1),
        "decode_step_ms": round(decode_step_ms, 1),
    }


def chart_decode(
    state,
    batch_sizes: Optional[list[int]] = None,
    panel_suffix: str = "",
    *,
    deployment: Deployment,
) -> list[dict]:
    datasets = []
    eff = state.decode_efficiency
    is_b = panel_suffix != ""
    batch_sizes = batch_sizes or BATCH_SIZES

    for am in deployment.decode:
        model = am.model
        if not _is_decode_pareto_model(model):
            continue
        gpu = am.gpu_spec
        if gpu is None:
            continue
        spec, spec_meta = _spec_chart_selection(state, am, model)
        pts = []
        for bs in batch_sizes:
            result = compute_decode(
                model,
                am.tp,
                am.pp,
                bs,
                am.dp,
                gpu,
                state.mu,
                state.profiled_non_kv_gb,
                am.prec,
                state.in_dist,
                state.out_dist,
                eff,
                spec,
            )
            pts.append(
                {
                    "x": bs,
                    "y": result.tps if result else None,
                    **spec_meta,
                    "spec_speedup": result.spec_speedup if result else spec_meta["spec_speedup"],
                }
            )
        datasets.append(
            {
                "label": _label(am, model, panel_suffix, include_prefill=True, spec_meta=spec_meta),
                "data": pts,
                **spec_meta,
                "borderColor": model.color,
                "backgroundColor": model.color + "12",
                "borderWidth": 1.5 if is_b else 2,
                "borderDash": [5, 3] if is_b else [],
                "fill": not is_b,
                "tension": 0.3,
                "pointRadius": 2.5,
                "spanGaps": False,
            }
        )
    return datasets


def chart_pareto(state, panel_suffix: str = "", *, deployment: Deployment) -> list[dict]:
    datasets = []
    eff = state.decode_efficiency
    is_b = panel_suffix != ""

    for am in deployment.decode:
        model = am.model
        if not _is_decode_pareto_model(model):
            continue
        gpu = am.gpu_spec
        if gpu is None:
            continue
        spec, spec_meta = _spec_chart_selection(state, am, model)
        pts = []
        for bs in BATCH_SIZES:
            result = compute_decode(
                model,
                am.tp,
                am.pp,
                bs,
                am.dp,
                gpu,
                state.mu,
                state.profiled_non_kv_gb,
                am.prec,
                state.in_dist,
                state.out_dist,
                eff,
                spec,
            )
            if result:
                pts.append(
                    {
                        "x": result.lat,
                        "y": result.tps,
                        "bs": bs,
                        **spec_meta,
                        "spec_speedup": result.spec_speedup,
                    }
                )
        if pts:
            datasets.append(
                {
                    "label": _label(am, model, panel_suffix, spec_meta=spec_meta),
                    "data": pts,
                    **spec_meta,
                    "borderColor": model.color,
                    "backgroundColor": model.color + "AA",
                    "borderWidth": 1.5 if is_b else 2,
                    "borderDash": [5, 3] if is_b else [],
                    "showLine": True,
                    "tension": 0.3,
                    "pointRadius": 3.5,
                }
            )
    return datasets


def chart_user_pareto(
    state,
    batch_sizes: Optional[list[int]] = None,
    panel_suffix: str = "",
    *,
    deployment: Deployment,
) -> list[dict]:
    datasets = []
    eff = state.decode_efficiency
    is_b = panel_suffix != ""
    batch_sizes = batch_sizes or BATCH_SIZES

    for am in deployment.decode:
        model = am.model
        if not _is_decode_pareto_model(model):
            continue
        gpu = am.gpu_spec
        if gpu is None:
            continue
        spec, spec_meta = _spec_chart_selection(state, am, model)
        pts = []
        for users in batch_sizes:
            result = compute_decode(
                model,
                am.tp,
                am.pp,
                users,
                am.dp,
                gpu,
                state.mu,
                state.profiled_non_kv_gb,
                am.prec,
                state.in_dist,
                state.out_dist,
                eff,
                spec,
            )
            if result:
                pts.append(
                    {
                        "x": users,
                        "y": round((result.tps / users) * 100) / 100,
                        "users": users,
                        "total_tps": result.tps,
                        "lat": result.lat,
                        "spec_speedup": result.spec_speedup,
                        **{k: v for k, v in spec_meta.items() if k != "spec_speedup"},
                    }
                )
        if pts:
            datasets.append(
                {
                    "label": _label(am, model, panel_suffix, spec_meta=spec_meta),
                    "data": pts,
                    **spec_meta,
                    "borderColor": model.color,
                    "backgroundColor": model.color + "AA",
                    "borderWidth": 1.5 if is_b else 2,
                    "borderDash": [5, 3] if is_b else [],
                    "showLine": True,
                    "tension": 0.3,
                    "pointRadius": 3.5,
                }
            )
    return datasets


def chart_aggregate(
    state,
    batch_sizes: Optional[list[int]] = None,
    panel_suffix: str = "",
    *,
    deployment: Deployment,
) -> list[dict]:
    datasets = []
    eff = state.decode_efficiency
    is_b = panel_suffix != ""
    deployed = deployment.decode
    batch_sizes = batch_sizes or BATCH_SIZES

    agg = []
    for bs in batch_sizes:
        total = 0
        for am in deployed:
            model = am.model
            if getattr(model, "embedding_profile", None) is not None:
                continue
            gpu = am.gpu_spec
            if gpu is None:
                continue
            result = compute_decode(
                model,
                am.tp,
                am.pp,
                bs,
                am.dp,
                gpu,
                state.mu,
                state.profiled_non_kv_gb,
                am.prec,
                state.in_dist,
                state.out_dist,
                eff,
                spec_runtime_for(state, am, model),
            )
            if result:
                total += result.tps
        agg.append({"x": bs, "y": total or None})
    datasets.append(
        {
            "label": f"Node total{panel_suffix}",
            "data": agg,
            "borderColor": "#ddd",
            "backgroundColor": "rgba(255,255,255,0.04)",
            "borderWidth": 2.5,
            "borderDash": [5, 3] if is_b else [],
            "fill": not is_b,
            "tension": 0.3,
            "pointRadius": 2.5,
            "spanGaps": False,
            "_isAggregate": True,
        }
    )

    for am in deployed:
        model = am.model
        if getattr(model, "embedding_profile", None) is not None:
            continue
        gpu = am.gpu_spec
        if gpu is None:
            continue
        pts = []
        for bs in batch_sizes:
            result = compute_decode(
                model,
                am.tp,
                am.pp,
                bs,
                am.dp,
                gpu,
                state.mu,
                state.profiled_non_kv_gb,
                am.prec,
                state.in_dist,
                state.out_dist,
                eff,
                spec_runtime_for(state, am, model),
            )
            pts.append({"x": bs, "y": result.tps if result else None})
        datasets.append(
            {
                "label": f"{model.name}{panel_suffix}",
                "data": pts,
                "borderColor": model.color + ("44" if is_b else "77"),
                "borderWidth": 1,
                "borderDash": [4, 2] if is_b else [],
                "fill": False,
                "tension": 0.3,
                "pointRadius": 1.5,
                "spanGaps": False,
            }
        )
    return datasets


def chart_data_processing(
    state, batch_sizes: Optional[list[int]] = None, panel_suffix: str = ""
) -> list[dict]:
    datasets = []
    is_b = panel_suffix != ""
    il, ol = state.task_il, state.task_ol
    batch_sizes = batch_sizes or DATA_BATCH_SIZES

    for am, gpu in _iter_resolved_models(state):
        model = am.model
        if getattr(model, "embedding_profile", None) is not None:
            continue
        pts = []
        for bs in batch_sizes:
            result = compute_data(
                model,
                (am.prefill_tp, am.prefill_pp, am.prefill_dp),
                (am.tp, am.pp, am.dp),
                bs,
                il,
                ol,
                gpu,
                state.mu,
                state.profiled_non_kv_gb,
                am.prec,
                state.prefix_hit_rate,
                state.prefill_efficiency,
                state.decode_efficiency,
                spec_runtime_for(state, am, model),
            )
            pts.append({"x": bs, "y": result.tps if result else None})
        datasets.append(
            {
                "label": _label(am, model, panel_suffix),
                "data": pts,
                "borderColor": model.color,
                "backgroundColor": model.color + "12",
                "borderWidth": 1.5 if is_b else 2,
                "borderDash": [5, 3] if is_b else [],
                "fill": not is_b,
                "tension": 0.3,
                "pointRadius": 2.5,
                "spanGaps": False,
            }
        )

    agg = []
    for bs in batch_sizes:
        total = 0
        for am, gpu in _iter_resolved_models(state):
            model = am.model
            if getattr(model, "embedding_profile", None) is not None:
                continue
            result = compute_data(
                model,
                (am.prefill_tp, am.prefill_pp, am.prefill_dp),
                (am.tp, am.pp, am.dp),
                bs,
                il,
                ol,
                gpu,
                state.mu,
                state.profiled_non_kv_gb,
                am.prec,
                state.prefix_hit_rate,
                state.prefill_efficiency,
                state.decode_efficiency,
                spec_runtime_for(state, am, model),
            )
            if result:
                total += result.tps
        agg.append({"x": bs, "y": total or None})
    datasets.append(
        {
            "label": f"Node total{panel_suffix}",
            "data": agg,
            "borderColor": "#ddd",
            "borderWidth": 2,
            "borderDash": [5, 3],
            "fill": False,
            "tension": 0.3,
            "pointRadius": 1.5,
            "spanGaps": False,
            "_isAggregate": True,
        }
    )
    return datasets


def chart_embedding_throughput(
    state, batch_sizes: Optional[list[int]] = None, panel_suffix: str = ""
) -> list[dict]:
    datasets = []
    is_b = panel_suffix != ""
    batch_sizes = batch_sizes or EMBEDDING_BATCH_SIZES
    doc_dist = _embedding_doc_dist_for_state(state)

    for am, gpu in _iter_resolved_models(state):
        model = am.model
        profile = getattr(model, "embedding_profile", None)
        if profile is None:
            continue

        stats = embedding_doc_stats(model, doc_dist, EMBEDDING_DOC_BUCKETS, am.prec)
        pts = []
        for bs in batch_sizes:
            result = compute_embedding_distribution(
                model,
                (am.prefill_tp, am.prefill_pp, am.prefill_dp),
                bs,
                doc_dist,
                EMBEDDING_DOC_BUCKETS,
                gpu,
                state.mu,
                state.profiled_non_kv_gb,
                am.prec,
                state.prefill_efficiency,
            )
            if result is None:
                pts.append(
                    {
                        "x": bs,
                        "y": None,
                        "seq_len": round(stats.mean_seq_len),
                        "p50_seq_len": stats.p50_seq_len,
                        "p90_seq_len": stats.p90_seq_len,
                        "p99_seq_len": stats.p99_seq_len,
                        "mode": profile.mode_label,
                        "max_batch": 0,
                    }
                )
                continue
            pts.append(
                {
                    "x": bs,
                    "y": result.rps,
                    "rps": result.rps,
                    "tps": result.tps,
                    "vectors_per_second": result.vectors_per_second,
                    "vectors_per_input": result.vectors_per_input,
                    "output_mb_s": result.output_mb_s,
                    "seq_len": result.seq_len,
                    "p50_seq_len": result.p50_seq_len,
                    "p90_seq_len": result.p90_seq_len,
                    "p99_seq_len": result.p99_seq_len,
                    "mode": profile.mode_label,
                    "max_batch": result.max_batch,
                }
            )

        datasets.append(
            {
                "label": _label(am, model, panel_suffix, include_prefill=True),
                "data": pts,
                "borderColor": model.color,
                "backgroundColor": model.color + "12",
                "borderWidth": 1.5 if is_b else 2,
                "borderDash": [5, 3] if is_b else [],
                "fill": False,
                "tension": 0.3,
                "pointRadius": 2.5,
                "spanGaps": False,
                "_isEmbedding": True,
            }
        )
    return datasets


def chart_embedding_quality(state, panel_suffix: str = "") -> list[dict]:
    """Peak docs/s vs published retrieval quality, one dot per embedding model.

    Each model emits a single point — x = peak docs/s (max over the standard
    batch sweep at the current workload distribution), y = decontaminated BEIR
    quality when sourced, otherwise the existing catalog quality fallback in
    [0, 1]. Bytes-per-doc and vec/s are attached to the point so the front-end
    can encode storage cost via dot size and surface multi-vector blowup in the
    tooltip.
    """
    datasets = []
    is_b = panel_suffix != ""
    doc_dist = _embedding_doc_dist_for_state(state)

    for am, gpu in _iter_resolved_models(state):
        model = am.model
        profile = getattr(model, "embedding_profile", None)
        if profile is None:
            continue
        fallback_quality = PUBLISHED_EMBEDDING_QUALITY.get(model.key)
        if fallback_quality is None:
            continue

        stats = embedding_doc_stats(model, doc_dist, EMBEDDING_DOC_BUCKETS, am.prec)

        best = None
        for bs in EMBEDDING_BATCH_SIZES:
            result = compute_embedding_distribution(
                model,
                (am.prefill_tp, am.prefill_pp, am.prefill_dp),
                bs,
                doc_dist,
                EMBEDDING_DOC_BUCKETS,
                gpu,
                state.mu,
                state.profiled_non_kv_gb,
                am.prec,
                state.prefill_efficiency,
            )
            if result is None:
                continue
            if best is None or result.rps > best.rps:
                best = result
                best_bs = bs
        if best is None:
            continue

        is_placeholder = model.key in EMBEDDING_QUALITY_PLACEHOLDER
        decontaminated_beir = PUBLISHED_EMBEDDING_DECONTAMINATED_BEIR.get(model.key)
        uses_decontaminated_beir = decontaminated_beir is not None
        quality = decontaminated_beir if uses_decontaminated_beir else fallback_quality
        bytes_per_doc = stats.mean_output_bytes_per_input
        point = {
            "x": best.rps,
            "y": quality,
            "docs_per_second": best.rps,
            "tokens_per_second": best.tps,
            "vectors_per_second": best.vectors_per_second,
            "vectors_per_input": best.vectors_per_input,
            "output_mb_s": best.output_mb_s,
            "bytes_per_doc": bytes_per_doc,
            "seq_len": best.seq_len,
            "peak_batch": best_bs,
            "max_batch": best.max_batch,
            "mode": profile.mode_label,
            "quality": quality,
            "quality_metric": "Decontaminated BEIR nDCG@10"
            if uses_decontaminated_beir
            else "Published retrieval nDCG@10 fallback",
            "source": (
                EMBEDDING_DECONTAMINATED_BEIR_SOURCES.get(model.key, "")
                if uses_decontaminated_beir
                else EMBEDDING_QUALITY_SOURCES.get(model.key, "")
            ),
            "published_quality": fallback_quality,
            "published_quality_source": EMBEDDING_QUALITY_SOURCES.get(model.key, ""),
            "decontaminated_beir_quality": decontaminated_beir,
            "decontaminated_beir_source": EMBEDDING_DECONTAMINATED_BEIR_SOURCES.get(model.key, ""),
            "uses_decontaminated_beir": uses_decontaminated_beir,
            "placeholder": is_placeholder,
        }

        datasets.append(
            {
                "label": _label(am, model, panel_suffix, include_prefill=True),
                "data": [point],
                "borderColor": model.color,
                "backgroundColor": (model.color + "12") if is_placeholder else (model.color + "AA"),
                "borderWidth": 1.5 if is_b else 2,
                "borderDash": [5, 3] if is_b else [],
                "showLine": False,
                "fill": False,
                "tension": 0,
                "pointRadius": 5,
                "spanGaps": False,
                "_isEmbeddingQuality": True,
                "_placeholder": is_placeholder,
            }
        )
    return datasets


def embedding_quality_axis_range(
    datasets: list[dict], margin_ratio: float = 0.08, min_margin: float = 0.01
) -> dict[str, float]:
    values: list[float] = []
    for dataset in datasets:
        for point in dataset.get("data", []):
            quality = point.get("quality", point.get("y"))
            if isinstance(quality, (int, float)) and math.isfinite(float(quality)):
                values.append(float(quality))

    if not values:
        return {"y_min": 0.0, "y_max": 1.0}

    lo = min(values)
    hi = max(values)
    span = hi - lo
    margin = max(span * max(margin_ratio, 0.0), max(min_margin, 0.0))
    if span <= 1e-9:
        margin = max(margin, 0.02)

    return {
        "y_min": round(max(0.0, lo - margin), 4),
        "y_max": round(min(1.0, hi + margin), 4),
    }


def chart_processing_pareto(
    state, batch_sizes: Optional[list[int]] = None, panel_suffix: str = ""
) -> list[dict]:
    datasets = []
    is_b = panel_suffix != ""
    batch_sizes = batch_sizes or DATA_BATCH_SIZES
    deployed = list(_iter_resolved_models(state))

    for idx, (preset_name, preset) in enumerate(DIST_PRESETS.items()):
        in_len = avg_dist(preset["in"], INPUT_BUCKETS)
        out_len = avg_dist(preset["out"], OUTPUT_BUCKETS)
        tokens_per_req = in_len + out_len
        pts = []
        for bs in batch_sizes:
            total_tps = 0
            for am, gpu in deployed:
                if getattr(am.model, "embedding_profile", None) is not None:
                    continue
                result = compute_data(
                    am.model,
                    (am.prefill_tp, am.prefill_pp, am.prefill_dp),
                    (am.tp, am.pp, am.dp),
                    bs,
                    in_len,
                    out_len,
                    gpu,
                    state.mu,
                    state.profiled_non_kv_gb,
                    am.prec,
                    state.prefix_hit_rate,
                    state.prefill_efficiency,
                    state.decode_efficiency,
                    spec_runtime_for(state, am, am.model),
                )
                if result:
                    total_tps += result.tps
            total_rps = (total_tps / tokens_per_req) if tokens_per_req > 0 else 0.0
            pts.append(
                {
                    "x": bs,
                    "y": round(total_rps * 100) / 100 if total_tps else None,
                    "rps": round(total_rps * 100) / 100 if total_tps else None,
                    "tps": total_tps or None,
                    "in_len": in_len,
                    "out_len": out_len,
                    "workload": preset_name,
                }
            )

        color = PROCESSING_PARETO_COLORS[idx % len(PROCESSING_PARETO_COLORS)]
        datasets.append(
            {
                "label": f"{preset_name}{panel_suffix}",
                "data": pts,
                "borderColor": color,
                "backgroundColor": color + "12",
                "borderWidth": 1.5 if is_b else 2,
                "borderDash": [5, 3] if is_b else [],
                "fill": False,
                "tension": 0.3,
                "pointRadius": 2.5,
                "spanGaps": False,
            }
        )
    return datasets


def chart_user_experience(state, panel_suffix: str = "") -> list[dict]:
    datasets = []
    is_b = panel_suffix != ""

    for am, gpu in _iter_resolved_models(state):
        model = am.model
        if getattr(model, "embedding_profile", None) is not None:
            continue
        points = _user_exp_curve(
            model,
            (am.prefill_tp, am.prefill_pp, am.prefill_dp),
            (am.tp, am.pp, am.dp),
            gpu,
            state.mu,
            state.profiled_non_kv_gb,
            am.prec,
            state.in_dist,
            state.out_dist,
            state.prefix_hit_rate,
            state.prefill_efficiency,
            state.decode_efficiency,
            spec_runtime_for(state, am, model),
        )
        datasets.append(
            {
                "label": _label(am, model, panel_suffix, include_prefill=True),
                "data": points,
                "borderColor": model.color,
                "borderWidth": 1.5 if is_b else 2,
                "borderDash": [5, 3] if is_b else [],
                "fill": False,
                "tension": 0.3,
                "pointRadius": 3,
                "showLine": True,
                "spanGaps": False,
            }
        )
    return datasets


def chart_realtime_capacity(
    state, batch_sizes: Optional[list[int]] = None, panel_suffix: str = ""
) -> list[dict]:
    datasets = []
    is_b = panel_suffix != ""
    batch_sizes = batch_sizes or REALTIME_USER_SWEEP

    for am, gpu in _iter_resolved_models(state):
        model = am.model
        profile = getattr(model, "realtime_profile", None)
        if profile is None:
            continue

        max_users = compute_realtime_max_users(
            model,
            (am.tp, am.pp, am.dp),
            gpu,
            state.mu,
            state.profiled_non_kv_gb,
            am.prec,
            state.decode_efficiency,
        )
        pts = []
        for users in batch_sizes:
            result = compute_realtime_capacity(
                model,
                (am.tp, am.pp, am.dp),
                users,
                gpu,
                state.mu,
                state.profiled_non_kv_gb,
                am.prec,
                state.decode_efficiency,
            )
            if result is None:
                pts.append(
                    {
                        "x": users,
                        "y": None,
                        "users": users,
                        "max_users": max_users,
                        "required_tps": profile.tokens_per_second,
                        "target_delay_ms": profile.target_delay_ms,
                    }
                )
                continue

            pts.append(
                {
                    "x": users,
                    "y": result.realtime_factor,
                    "users": users,
                    "max_users": max_users,
                    "per_user_tps": result.per_user_tps,
                    "required_tps": result.required_tps,
                    "total_tps": result.total_tps,
                    "step_ms": result.step_ms,
                    "max_slots": result.max_slots,
                    "target_delay_ms": profile.target_delay_ms,
                }
            )
        if pts:
            datasets.append(
                {
                    "label": _label(am, model, panel_suffix),
                    "data": pts,
                    "borderColor": model.color,
                    "backgroundColor": model.color + "12",
                    "borderWidth": 1.5 if is_b else 2,
                    "borderDash": [5, 3] if is_b else [],
                    "fill": False,
                    "tension": 0.3,
                    "pointRadius": 2.5,
                    "spanGaps": False,
                    "_isRealtime": True,
                }
            )
    return datasets


def chart_asr_quality(state, panel_suffix: str = "") -> list[dict]:
    """Max realtime streams vs published WER, one dot per benchmark/language.

    Concurrency is benchmark-independent in the capacity model, so every dot
    for a given model sits at the same height. WER is static catalog data;
    see PUBLISHED_ASR_WER in data.py.
    """
    datasets = []
    is_b = panel_suffix != ""

    for am, gpu in _iter_resolved_models(state):
        model = am.model
        profile = getattr(model, "realtime_profile", None)
        if profile is None:
            continue
        wer_by_language = PUBLISHED_ASR_WER.get(model.key)
        if not wer_by_language:
            continue

        max_users = compute_realtime_max_users(
            model,
            (am.tp, am.pp, am.dp),
            gpu,
            state.mu,
            state.profiled_non_kv_gb,
            am.prec,
            state.decode_efficiency,
        )
        if max_users <= 0:
            continue

        is_placeholder = model.key in ASR_WER_PLACEHOLDER
        pts = []
        sources = ASR_WER_LANGUAGE_SOURCES.get(model.key, {})
        for language in ASR_WER_LANGUAGES:
            wer = wer_by_language.get(language)
            if wer is None:
                continue
            pts.append(
                {
                    "x": wer,
                    "y": max_users,
                    "language": ASR_WER_LANGUAGE_LABELS.get(language, language),
                    "source": sources.get(language, ""),
                    "wer": wer,
                    "max_users": max_users,
                    "placeholder": is_placeholder,
                    "asr_mode": "streaming"
                    if getattr(profile, "streaming", True)
                    else "non-streaming",
                }
            )
        if not pts:
            continue
        pts.sort(key=lambda p: cast(float, p["x"]))

        datasets.append(
            {
                "label": _label(am, model, panel_suffix),
                "data": pts,
                "borderColor": model.color,
                "backgroundColor": (model.color + "12") if is_placeholder else (model.color + "AA"),
                "borderWidth": 1.5 if is_b else 2,
                "borderDash": [5, 3] if is_b else [],
                "showLine": True,
                "fill": False,
                "tension": 0,
                "pointRadius": 5,
                "spanGaps": False,
                "_isAsrQuality": True,
                "_placeholder": is_placeholder,
                "_asrStreaming": bool(getattr(profile, "streaming", True)),
                "_modelKey": model.key,
                "_assignmentUid": am.uid,
                "_seriesId": f"asrquality:{'b' if is_b else 'a'}:{am.uid}:{model.key}",
            }
        )
    return datasets


def compute_stats_data(state) -> dict:
    il, ol = state.task_il, state.task_ol
    batch_sizes = get_data_bs([state])

    peak_tps = 0
    peak_bs = 0
    for bs in batch_sizes:
        total = 0
        for am, gpu in _iter_resolved_models(state):
            model = am.model
            if getattr(model, "embedding_profile", None) is not None:
                continue
            result = compute_data(
                model,
                (am.prefill_tp, am.prefill_pp, am.prefill_dp),
                (am.tp, am.pp, am.dp),
                bs,
                il,
                ol,
                gpu,
                state.mu,
                state.profiled_non_kv_gb,
                am.prec,
                state.prefix_hit_rate,
                state.prefill_efficiency,
                state.decode_efficiency,
                spec_runtime_for(state, am, model),
            )
            if result:
                total += result.tps
        if total > peak_tps:
            peak_tps = total
            peak_bs = bs

    rps = peak_tps / (il + ol) if (il + ol) > 0 else 0.0
    return {
        "peak_tps": peak_tps,
        "peak_bs": peak_bs,
        "rps": rps,
        "dph": round(rps * 3600),
        "il": il,
        "ol": ol,
    }


def compute_user_exp_table(state) -> list[dict]:
    rows = []
    for am, gpu in _iter_resolved_models(state):
        model = am.model
        if getattr(model, "embedding_profile", None) is not None:
            continue
        points = _user_exp_curve(
            model,
            (am.prefill_tp, am.prefill_pp, am.prefill_dp),
            (am.tp, am.pp, am.dp),
            gpu,
            state.mu,
            state.profiled_non_kv_gb,
            am.prec,
            state.in_dist,
            state.out_dist,
            state.prefix_hit_rate,
            state.prefill_efficiency,
            state.decode_efficiency,
            spec_runtime_for(state, am, model),
        )
        if not points:
            continue
        peak = points[-1]
        cells: list[dict | None] = []
        for frac in USER_EXP_FRACTIONS:
            sample = _sample_user_exp_curve(points, peak["arrival_rps"] * frac)
            if sample is None:
                cells.append(None)
                continue
            cells.append(
                {
                    "lat": round(sample["decode_step_ms"], 1),
                    "resp_s": round(sample["response_s"], 2),
                    "ttft_ms": round(sample["ttft_ms"], 1),
                }
            )
        rows.append(
            {
                "model": model,
                "config": f"{strategy_label(am.tp, am.pp, am.dp)} {am.prec.upper()}",
                "prec": am.prec,
                "peak_rps": round(peak["arrival_rps"] * 100) / 100,
                "peak_resp_s": round(peak["response_s"] * 100) / 100,
                "peak_inflight": round(peak["inflight"], 1),
                "cells": cells,
            }
        )
    return rows


def _workload_profile(state) -> dict:
    """Average in/out lengths from the planner's distributions — a single workload for all models."""
    in_len = avg_dist(state.in_dist, INPUT_BUCKETS)
    out_len = avg_dist(state.out_dist, OUTPUT_BUCKETS)
    return {
        "in_len": in_len,
        "out_len": out_len,
        "tokens_per_request": in_len + out_len,
    }


def _project_workload_profile(project, fallback: dict) -> dict:
    """Average in/out lengths for one project's declared workload shape.

    The capacity model still uses the aggregate workload to estimate shared GPU supply, but
    routing economics need the project's own shape. Otherwise a short classification stream
    inherits the blended portfolio's long-output token tax and can look falsely priced out.
    """
    in_preset = DIST_PRESETS.get(getattr(project, "in_pre", "")) or DIST_PRESETS["Chat"]
    out_preset = DIST_PRESETS.get(getattr(project, "out_pre", "")) or DIST_PRESETS["Chat"]
    in_len: float = avg_dist(in_preset["in"], INPUT_BUCKETS)
    out_len: float = avg_dist(out_preset["out"], OUTPUT_BUCKETS)
    if in_len <= 0 or out_len <= 0:
        in_len = float(fallback["in_len"])
        out_len = float(fallback["out_len"])
    return {
        "in_len": in_len,
        "out_len": out_len,
        "tokens_per_request": max(1.0, in_len + out_len),
    }


def _best_deployment_result_for_model(
    state,
    am,
    gpu: GPU,
    in_len: int,
    out_len: int,
    batch_sizes: list[int],
    prefix_hit_rate: Optional[float] = None,
) -> Optional[DeploymentPeakResult]:
    prefix_rate = (
        state.prefix_hit_rate
        if prefix_hit_rate is None
        else min(max(float(prefix_hit_rate), 0.0), 1.0)
    )
    best: Optional[DeploymentPeakResult] = None
    spec = spec_runtime_for(state, am, am.model)
    for bs in batch_sizes:
        result = compute_data(
            am.model,
            (am.prefill_tp, am.prefill_pp, am.prefill_dp),
            (am.tp, am.pp, am.dp),
            bs,
            in_len,
            out_len,
            gpu,
            state.mu,
            state.profiled_non_kv_gb,
            am.prec,
            prefix_rate,
            state.prefill_efficiency,
            state.decode_efficiency,
            spec,
        )
        if result is None:
            continue

        candidate = DeploymentPeakResult(
            tps=result.tps,
            rps=result.rps,
            batch_size=bs,
            prefill_frac=result.prefill_frac,
        )
        if best is None:
            best = candidate
            continue

        if candidate.tps > best.tps:
            best = candidate
            continue
        if candidate.tps == best.tps and candidate.batch_size < best.batch_size:
            best = candidate
    return best


def _deployment_capacity_for_profile(
    state,
    am,
    gpu: GPU,
    profile: dict,
    peak_factor: float,
    prefix_hit_rate: Optional[float] = None,
) -> tuple[float, float]:
    """Return shape-specific daily token capacity and peak RPS.

    Capacity is recomputed for each workload shape. Routing consumes a shared
    normalized deployment-time fraction, so long and short requests no longer
    spend an interchangeable blended token budget.
    """
    in_len = int(profile["in_len"])
    out_len = int(profile["out_len"])
    prefix_rate = (
        state.prefix_hit_rate
        if prefix_hit_rate is None
        else min(max(float(prefix_hit_rate), 0.0), 1.0)
    )
    cap = compute_data_capacity(
        am.model,
        (am.prefill_tp, am.prefill_pp, am.prefill_dp),
        (am.tp, am.pp, am.dp),
        in_len,
        out_len,
        gpu,
        state.mu,
        state.profiled_non_kv_gb,
        am.prec,
        prefix_rate,
        state.prefill_efficiency,
        state.decode_efficiency,
        spec_runtime_for(state, am, am.model),
    )
    batch_sizes = _batch_axis_sweep([cap], DATA_BATCH_SIZES)
    best = _best_deployment_result_for_model(
        state,
        am,
        gpu,
        in_len,
        out_len,
        batch_sizes,
        prefix_rate,
    )
    peak_rps = best.rps if best and best.rps > 0 else 0.0
    tokens_per_request = max(float(profile["tokens_per_request"]), 1.0)
    daily_tokens = (
        peak_rps * 86400.0 * tokens_per_request / max(float(peak_factor), 1.0)
        if peak_rps > 0
        else 0.0
    )
    return daily_tokens, peak_rps


def _cloud_price_per_m_in_preset(
    difficulty: float,
    min_success: float,
    quality_floor: float,
    profile: dict,
    prefix_hit_rate: float,
    preset_name: str,
    required_capabilities: frozenset[str] = frozenset(),
    quality_domain: str = "general",
) -> tuple[Optional[dict], float]:
    """Cheapest cloud model in the active corpo preset that can serve a project with the
    given (difficulty, min_success_rate). ``quality_domain`` is accepted so the local and
    cloud paths share one project contract; cloud entries currently fall back to their
    global catalog quality until provider-domain anchors are added. Effective $/M is computed apples-to-apples with
    on-prem: sticker price × (1 / token_efficiency). A cloud is eligible only if
    success_rate(cloud.quality, difficulty) ≥ min_success_rate.

    Returns (cloud_info_or_None, effective_price_per_m). None when no compatible cloud
    exists in the catalog — i.e. spillover is *blocked* for this project."""
    in_len = float(profile["in_len"])
    out_len = float(profile["out_len"])
    cached = in_len * min(max(prefix_hit_rate, 0.0), 1.0)
    uncached = max(0.0, in_len - cached)
    tokens_per_req = max(1.0, in_len + out_len)

    best: Optional[tuple[float, dict]] = None
    for key, cloud in cloud_policy.effective_corpo_models(preset_name):
        if not (required_capabilities <= frozenset(cloud.get("capabilities", ()))):
            continue
        cloud_quality = float(cloud.get("quality", 0.5))
        cloud_eff = max(float(cloud.get("token_efficiency", 1.0)), 1e-6)
        if cloud_quality + 1e-9 < quality_floor:
            continue
        cloud_success = success_rate(cloud_quality, difficulty)
        if cloud_success + 1e-9 < min_success:
            continue
        threshold = max(float(cloud.get("long_context_threshold_tokens", 0.0) or 0.0), 0.0)
        long_context_pricing = threshold > 0.0 and in_len > threshold

        def tier_price(field: str, base_field: str) -> float:
            value = cloud.get(field) if long_context_pricing else None
            return max(float(cloud[base_field] if value is None else value), 0.0)

        input_price = tier_price("long_context_in_per_m", "in_per_m")
        cached_input_price = tier_price("long_context_cached_in_per_m", "cached_in_per_m")
        output_price = tier_price("long_context_out_per_m", "out_per_m")
        sticker = (
            (uncached / 1e6) * input_price
            + (cached / 1e6) * cached_input_price
            + ((out_len / cloud_eff) / 1e6) * output_price
        )
        # Token efficiency affects generated tokens, not the fixed prompt payload.
        # Retry-adjust cloud and on-prem routes symmetrically. A route with
        # success probability p consumes 1/p attempts per completed useful task.
        price_pm = sticker / (tokens_per_req / 1e6) / max(cloud_success, 1e-6)
        if best is None or price_pm < best[0]:
            best = (
                price_pm,
                cloud
                | {
                    "key": key,
                    "success_rate": cloud_success,
                    "long_context_pricing_applied": long_context_pricing,
                    "effective_in_per_m": input_price,
                    "effective_cached_in_per_m": cached_input_price,
                    "effective_out_per_m": output_price,
                },
            )

    if best is None:
        return None, math.inf
    return best[1], best[0]


def tokens_per_task(model: Model, task_il: int, task_ol: int) -> float:
    """Output tokens scale by 1/token_efficiency (verbose models emit more to finish a task)."""
    eff = max(float(getattr(model, "token_efficiency", 1.0)), 1e-6)
    return float(task_il) + float(task_ol) / eff


def _actual_token_multiplier(token_efficiency: float, in_len: float, out_len: float) -> float:
    """Actual GPU/cloud tokens consumed per useful workload token.

    Token efficiency is an output-token verbosity proxy. Prompts do not get longer just
    because a model thinks or writes more, so only the output side is scaled.
    """
    eff = max(float(token_efficiency), 1e-6)
    useful = max(float(in_len) + float(out_len), 1.0)
    actual = max(float(in_len), 0.0) + max(float(out_len), 0.0) / eff
    return max(actual / useful, 1e-9)


def latent_activation_share(cheapest_pm: float, unlock_price: float) -> float:
    """Smooth latent-demand activation around the configured unlock price.

    A hard threshold makes portfolio demand jump discontinuously when a model becomes
    barely cheap enough. This curve keeps the same midpoint semantics: at the unlock
    price, half the latent pool is active; materially cheaper routes approach 100%.
    """
    if unlock_price <= 0 or math.isinf(cheapest_pm) or cheapest_pm <= 0:
        return 0.0
    ratio = unlock_price / cheapest_pm
    return min(max(1.0 / (1.0 + math.exp(-LATENT_UNLOCK_STEEPNESS * (ratio - 1.0))), 0.0), 1.0)


def co2_g_per_task(
    gpu: GPU,
    gpu_count: int,
    tokens_per_task_val: float,
    tokens_per_sec: float,
    gco2_per_kwh: float,
    utilization: float = GPU_POWER_UTILIZATION,
) -> float:
    """Grams CO2-eq per task. Energy = cluster_power × tokens_per_task / tokens_per_sec."""
    if tokens_per_sec <= 0 or tokens_per_task_val <= 0:
        return 0.0
    tdp = float(getattr(gpu, "tdp_watts", 0.0))
    if tdp <= 0:
        return 0.0
    cluster_power_w = tdp * gpu_count * utilization
    task_wall_s = tokens_per_task_val / tokens_per_sec
    energy_j = cluster_power_w * task_wall_s
    # 1 kWh = 3.6e6 J; gCO2/kWh × kWh = grams.
    return energy_j * gco2_per_kwh / 3.6e6


def _build_model_supply(state, profile, prefix_hit_rate, peak_factor_eff) -> list[dict]:
    """For each deployed model, compute peak RPS, sustainable tokens/day, and internal $/M."""
    tokens_per_req = max(1.0, profile["tokens_per_request"])
    pool_rate = {gp.uid: gp.cost_per_gpu_hour * 24.0 for gp in state.gpus}
    pool_country = {gp.uid: getattr(gp, "country", DEFAULT_COUNTRY) for gp in state.gpus}
    day_shape = (
        DAY_SHAPES.get(getattr(state, "projection_day_shape", "workday")) or DAY_SHAPES["workday"]
    )
    day_weights = cast(list[float], day_shape["weights"]) or [1.0] * 24
    night_weights = [1.0 if h in NIGHT_HOURS else 0.0 for h in range(24)]
    supply = []
    for am, gpu in _iter_resolved_models(state):
        if (
            getattr(am.model, "is_realtime_only", False)
            or getattr(am.model, "embedding_profile", None) is not None
        ):
            continue
        cap = compute_data_capacity(
            am.model,
            (am.prefill_tp, am.prefill_pp, am.prefill_dp),
            (am.tp, am.pp, am.dp),
            profile["in_len"],
            profile["out_len"],
            gpu,
            state.mu,
            state.profiled_non_kv_gb,
            am.prec,
            prefix_hit_rate,
            state.prefill_efficiency,
            state.decode_efficiency,
            spec_runtime_for(state, am, am.model),
        )
        batch_sizes = _batch_axis_sweep([cap], DATA_BATCH_SIZES)
        best = _best_deployment_result_for_model(
            state, am, gpu, profile["in_len"], profile["out_len"], batch_sizes
        )
        peak_rps = best.rps if (best and best.rps > 0) else 0.0
        # Sustainable daily token capacity: honor peak-hour headroom so we don't promise
        # throughput the day-shape can't actually sustain without thrashing.
        daily_tokens_cap = (
            peak_rps * 3600.0 * 24.0 * tokens_per_req / peak_factor_eff if peak_rps > 0 else 0.0
        )
        gpu_cost_day = pool_rate.get(am.gpu_uid, 0.0) * am.gpu_count
        internal_pm = (gpu_cost_day * 1e6 / daily_tokens_cap) if daily_tokens_cap > 0 else math.inf
        country = pool_country.get(am.gpu_uid, DEFAULT_COUNTRY)
        grid_day = carbon_intensity_avg(country, day_weights)
        grid_night = carbon_intensity_avg(country, night_weights)
        supply.append(
            {
                "am": am,
                "am_uid": am.uid,
                "gpu": gpu,
                "gpu_uid": am.gpu_uid,
                "gpu_count": am.gpu_count,
                "model": am.model,
                "quality": float(am.model.quality),
                "effective_quality": effective_quality(am.model),
                "quality_confidence": min(
                    max(float(getattr(am.model, "quality_confidence", 1.0)), 0.0), 1.0
                ),
                "token_efficiency": max(float(am.model.token_efficiency), 1e-6),
                "peak_rps": peak_rps,
                "daily_tokens_cap": daily_tokens_cap,
                "remaining_cap": daily_tokens_cap,
                "remaining_fraction": 1.0,
                "used_fraction": 0.0,
                "served_tokens": 0.0,
                "gpu_cost_day": gpu_cost_day,
                "internal_pm": internal_pm,
                # These are accumulated with each project's own task shape while routing.
                "served_tasks": 0.0,
                "served_co2_g_day": 0.0,
                "served_co2_g_night": 0.0,
                "country": country,
                "grid_gco2_per_kwh_day": grid_day,
                "grid_gco2_per_kwh_night": grid_night,
                "runnable": peak_rps > 0,
            }
        )
    return supply


def compute_revenue_projection(state, include_recommendations: bool = True) -> dict:
    """Internal-market economics for the current deployment, driven by project-level demand.

    For each project we allocate demand to the cheapest tier-compatible deployed model that
    is also at or below the project's willingness-to-pay (and ≤ cloud price). What can't be
    placed falls into one of three "demand destruction" buckets:
      * spilled — right model exists but saturated → flees to cloud (if WTP allows) else destroyed
      * leaked  — no compatible model, or all compatible too expensive → flees to cloud else destroyed
      * destroyed — cloud reference also above WTP: user shelves the work entirely

    The returned dict powers the Internal market panel."""
    profile = _workload_profile(state)
    prefix_hit_rate = min(max(state.prefix_hit_rate, 0.0), 1.0)
    corpo_cloud = getattr(state, "corpo_cloud", CORPO_CLOUD_DEFAULT)
    day_shape = DAY_SHAPES.get(state.projection_day_shape) or DAY_SHAPES["workday"]
    weights = cast(list[float], day_shape["weights"]) or [1.0]
    mean_w = sum(weights) / len(weights)
    peak_factor = (max(weights) / mean_w) if mean_w > 0 else 1.0

    projects = list(state.projects)
    total_demand = sum(max(0.0, p.tokens_day) for p in projects)
    batch_demand = sum(max(0.0, p.tokens_day) for p in projects if p.batch_eligible)
    batch_share = (batch_demand / total_demand) if total_demand > 0 else 0.0
    night_batching = bool(state.projection_night_batching)
    # Night batching flattens the day shape for batch-eligible demand: effective peak factor
    # is a convex blend of the raw shape (non-batch demand) and a perfectly flat shape.
    if night_batching:
        peak_factor_eff = (1.0 - batch_share) * peak_factor + batch_share * 1.0
    else:
        peak_factor_eff = peak_factor
    peak_factor_eff = max(peak_factor_eff, 1.0)

    supply = _build_model_supply(state, profile, prefix_hit_rate, peak_factor_eff)

    # Project routing — serve higher WTP first. Within the same value tier, protect the
    # harder/scarcer contract before easy generic work; insertion-order/UID must not let
    # customer-service demand consume a coding model before an equally valued repository
    # workload merely because that preset appears first.
    projects_sorted = sorted(
        projects,
        key=lambda p: (
            -float(p.wtp_per_m),
            -required_quality(
                float(getattr(p, "difficulty", 0.5)),
                float(getattr(p, "min_success_rate", 0.85)),
                quality_floor=float(getattr(p, "quality_floor", 0.0)),
            ),
            -len(getattr(p, "requires", frozenset()) or frozenset()),
            p.uid,
        ),
    )
    routed: dict[int, dict] = {}
    for p in projects_sorted:
        difficulty = float(getattr(p, "difficulty", 0.5))
        slo = float(getattr(p, "min_success_rate", 0.85))
        quality_floor = float(getattr(p, "quality_floor", 0.0))
        quality_domain = normalize_quality_domain(getattr(p, "quality_domain", "general"))
        quality_weights = normalize_quality_weights(
            getattr(p, "quality_weights", None),
            quality_domain,
        )
        required_caps = frozenset(getattr(p, "requires", frozenset()) or frozenset())
        project_prefix_hit_rate = min(max(float(getattr(p, "prefix_hit_rate", 0.0)), 0.0), 1.0)
        project_profile = _project_workload_profile(p, profile)
        cloud_info, cloud_pm = _cloud_price_per_m_in_preset(
            difficulty,
            slo,
            quality_floor,
            project_profile,
            project_prefix_hit_rate,
            corpo_cloud,
            required_caps,
            quality_domain,
        )
        cloud_blocked = cloud_info is None
        cloud_details = cloud_info or {}
        wtp = float(p.wtp_per_m)
        total = max(0.0, float(p.tokens_day))

        # Candidate list with capability + success-rate gates. `useful tokens` = work the
        # project needs done; token efficiency affects generated/output tokens only, so the
        # actual GPU tokens burned per useful token depends on this workload's input/output mix.
        candidates: list[dict] = []
        runnable_seen = False
        capability_compatible_seen = False
        floor_compatible_seen = False
        slo_compatible_seen = False
        for me in supply:
            if not me["runnable"]:
                continue
            runnable_seen = True
            if not (required_caps <= me["model"].capabilities):
                continue
            capability_compatible_seen = True
            project_quality = model_profile_quality(me["model"], quality_weights, quality_domain)
            if project_quality + 1e-9 < quality_floor:
                continue
            floor_compatible_seen = True
            sr = model_profile_success_rate(
                me["model"], difficulty, quality_weights, quality_domain
            )
            if sr + 1e-9 < slo:
                continue
            slo_compatible_seen = True
            token_mult = _actual_token_multiplier(
                me["token_efficiency"],
                float(project_profile["in_len"]),
                float(project_profile["out_len"]),
            )
            retry_mult = 1.0 / max(sr, 1e-6)
            project_peak_factor = 1.0 if (night_batching and p.batch_eligible) else peak_factor
            shape_daily_cap, shape_peak_rps = _deployment_capacity_for_profile(
                state,
                me["am"],
                me["gpu"],
                project_profile,
                project_peak_factor,
                project_prefix_hit_rate,
            )
            if shape_daily_cap <= 0:
                continue
            shape_internal_pm = me["gpu_cost_day"] * 1e6 / shape_daily_cap
            project_tpt = tokens_per_task(
                me["model"],
                int(project_profile["in_len"]),
                int(project_profile["out_len"]),
            )
            project_tokens_per_sec = shape_peak_rps * project_tpt
            candidates.append(
                {
                    "me": me,
                    "success_rate": sr,
                    "retry_mult": retry_mult,
                    "token_mult": token_mult * retry_mult,
                    "shape_daily_cap": shape_daily_cap,
                    "shape_peak_rps": shape_peak_rps,
                    "shape_internal_pm": shape_internal_pm,
                    "effective_pm": shape_internal_pm * token_mult * retry_mult,
                    "tokens_per_task": project_tpt,
                    "co2_g_per_task_day": co2_g_per_task(
                        me["gpu"],
                        me["gpu_count"],
                        project_tpt,
                        project_tokens_per_sec,
                        me["grid_gco2_per_kwh_day"],
                    ),
                    "co2_g_per_task_night": co2_g_per_task(
                        me["gpu"],
                        me["gpu_count"],
                        project_tpt,
                        project_tokens_per_sec,
                        me["grid_gco2_per_kwh_night"],
                    ),
                }
            )
        candidates.sort(key=lambda c: c["effective_pm"])

        # Latent demand activates smoothly around the unlock price. This keeps the configured
        # unlock as the midpoint while avoiding discontinuous portfolio demand jumps.
        baseline_tokens = total
        latent_pool = max(0.0, float(getattr(p, "latent_jobs_day", 0.0)))
        unlock_price = float(getattr(p, "unlock_price_per_m", 0.0))
        cheapest_pm = candidates[0]["effective_pm"] if candidates else float("inf")
        latent_activation = (
            latent_activation_share(cheapest_pm, unlock_price) if candidates else 0.0
        )
        latent_active = latent_pool * latent_activation
        latent_unlocked = latent_active > 1.0
        total = baseline_tokens + latent_active

        served = 0.0  # useful tokens delivered (project-perspective)
        per_model_served: list[tuple[dict, float, float, int]] = []
        internal_cost = 0.0
        co2_g_day_project = 0.0
        # Internal price cap: never charge above WTP; if cloud is reachable, also cap at cloud
        # (otherwise the project would just buy from cloud instead of paying us more).
        price_cap = wtp if cloud_blocked else min(wtp, cloud_pm)
        has_affordable_candidate = any(c["effective_pm"] <= price_cap + 1e-9 for c in candidates)
        for c in candidates:
            me = c["me"]
            if me["remaining_fraction"] <= 0:
                continue
            if c["effective_pm"] > price_cap:
                continue
            useful_remaining = total - served
            if useful_remaining <= 0:
                break
            shape_remaining = me["remaining_fraction"] * c["shape_daily_cap"]
            useful_take = min(useful_remaining, shape_remaining / c["token_mult"])
            if useful_take <= 0:
                continue
            actual_take = useful_take * c["token_mult"]
            fraction_used = actual_take / c["shape_daily_cap"]
            me["remaining_fraction"] = max(0.0, me["remaining_fraction"] - fraction_used)
            me["used_fraction"] = min(1.0, me["used_fraction"] + fraction_used)
            me["remaining_cap"] = me["daily_tokens_cap"] * me["remaining_fraction"]
            me["served_tokens"] += actual_take
            per_model_served.append((me, useful_take, actual_take, c["success_rate"]))
            internal_cost += (actual_take / 1e6) * c["shape_internal_pm"]
            tpt_m = c["tokens_per_task"]
            if tpt_m > 0:
                attempt_tasks = actual_take / tpt_m
                co2_day = attempt_tasks * c["co2_g_per_task_day"]
                co2_night = attempt_tasks * c["co2_g_per_task_night"]
                co2_g_day_project += co2_day
                me["served_tasks"] += attempt_tasks
                me["served_co2_g_day"] += co2_day
                me["served_co2_g_night"] += co2_night
            served += useful_take

        unserved = max(0.0, total - served)
        spilled, leaked, destroyed = 0.0, 0.0, 0.0
        if unserved > 0:
            if cloud_blocked:
                # No model in the corpo catalog can serve this tier — there's no cloud to flee
                # to. The work is dropped regardless of WTP.
                destroyed = unserved
            else:
                # "Had a usable home" means a matching model exists below the project price cap.
                # If none was served, the reason can still be capacity exhaustion caused by
                # higher-priority workloads, so classify that as spill instead of wrong-model leak.
                if served > 0 or has_affordable_candidate:
                    spilled = unserved
                else:
                    leaked = unserved
                if cloud_pm > wtp:
                    destroyed = spilled + leaked
                    spilled, leaked = 0.0, 0.0

        # Value of internally served tokens reflects the cheapest substitute (cloud price);
        # when cloud is blocked there's no substitute, so use WTP as the realized value.
        value_basis = wtp if cloud_blocked else cloud_pm
        # useful_t is completed work; retry cost and capacity were already charged.
        value_served = sum(
            (useful_t / 1e6) * value_basis for _, useful_t, _, _sr in per_model_served
        )
        baseline_tokens_per_task = max(float(project_profile["tokens_per_request"]), 1.0)
        tasks_served_day = served / baseline_tokens_per_task
        co2_g_per_task_project = (
            (co2_g_day_project / tasks_served_day) if tasks_served_day > 0 else 0.0
        )
        routed[p.uid] = {
            "project": p,
            "name": p.name,
            "difficulty": difficulty,
            "tokens_day": total,
            "cloud_pm": 0.0 if cloud_blocked else cloud_pm,
            "cloud_label": "blocked — no compatible cloud"
            if cloud_blocked
            else cloud_details["label"],
            "cloud_vendor": "" if cloud_blocked else cloud_details["vendor"],
            "cloud_regions": () if cloud_blocked else cloud_details.get("regions", ()),
            "cloud_grid_gco2_per_kwh": 0.0
            if cloud_blocked
            else cloud_details.get("grid_gco2_per_kwh", 0.0),
            "cloud_price_source": ""
            if cloud_blocked
            else cloud_details.get("price_source", "catalog"),
            "cloud_blocked": cloud_blocked,
            "served": served,
            "spilled": spilled,
            "leaked": leaked,
            "destroyed": destroyed,
            "served_pct": (served / total * 100.0) if total > 0 else 0.0,
            "spilled_pct": (spilled / total * 100.0) if total > 0 else 0.0,
            "leaked_pct": (leaked / total * 100.0) if total > 0 else 0.0,
            "destroyed_pct": (destroyed / total * 100.0) if total > 0 else 0.0,
            "internal_cost_day": internal_cost,
            "quality_floor": quality_floor,
            "quality_domain": quality_domain,
            "quality_domain_label": QUALITY_DOMAIN_LABELS[quality_domain],
            "quality_weights": quality_weights,
            "quality_mix_label": quality_weights_label(quality_weights, quality_domain),
            "prefix_hit_rate": project_prefix_hit_rate,
            "value_served": value_served,
            "value_spilled": (spilled / 1e6) * value_basis,
            "value_leaked": (leaked / 1e6) * value_basis,
            "value_destroyed": (destroyed / 1e6) * value_basis,
            "margin_day": value_served - internal_cost,
            "tasks_served_day": tasks_served_day,
            "co2_kg_day": co2_g_day_project / 1000.0,
            "co2_g_per_task_avg": co2_g_per_task_project,
            "wtp_per_m": wtp,
            "requires": sorted(required_caps),
            "min_success_rate": slo,
            "has_compatible": bool(candidates),
            "cap_blocked_for_project": runnable_seen and not capability_compatible_seen,
            "quality_floor_blocked_for_project": (
                capability_compatible_seen and not floor_compatible_seen
            ),
            "slo_blocked_for_project": floor_compatible_seen and not slo_compatible_seen,
            "capacity_blocked_for_project": slo_compatible_seen and not candidates,
            # True when *any* of the actually-serving candidates isn't a near-perfect fit
            # (success_rate < ~1.0) — used by the UI to flag "served, but via a stretched model".
            "any_suboptimal": any(sr < 0.99 for *_, sr in per_model_served),
            "any_served": served > 0,
            "baseline_tokens_day": baseline_tokens,
            "latent_jobs_day": latent_pool,
            "unlock_price_per_m": unlock_price,
            "latent_unlocked": latent_unlocked,
            "latent_active_tokens": latent_active,
            "latent_activation_pct": latent_activation * 100.0,
            "cheapest_effective_pm": (0.0 if math.isinf(cheapest_pm) else cheapest_pm),
            # Diagnostic hint: cheapest is within ~1.5× of unlock price but not yet under it.
            "latent_close_to_unlock": (
                latent_pool > 0
                and unlock_price > 0
                and latent_activation < 0.50
                and bool(candidates)
                and cheapest_pm <= unlock_price * 1.5 + 1e-9
            ),
            "per_model_served": [
                {
                    "am_uid": me["am_uid"],
                    "name": me["model"].name,
                    "tokens": useful_t,
                    "actual_tokens": actual_t,
                    "success_rate": sr,
                    "effective_quality": model_profile_quality(
                        me["model"], quality_weights, quality_domain
                    ),
                    "quality_anchor": " · ".join(
                        (
                            anchor.benchmark
                            if (anchor := model_domain_anchor(me["model"], domain)) is not None
                            else f"{QUALITY_DOMAIN_LABELS[domain]} global fallback"
                        )
                        for domain in quality_weights
                    ),
                    "quality_components": [
                        {
                            "domain": domain,
                            "label": QUALITY_DOMAIN_LABELS[domain],
                            "weight": weight,
                            "effective_quality": effective_quality(me["model"], domain),
                            "benchmark": (
                                anchor.benchmark
                                if (anchor := model_domain_anchor(me["model"], domain)) is not None
                                else "Global quality fallback"
                            ),
                            "anchored": model_domain_anchor(me["model"], domain) is not None,
                        }
                        for domain, weight in quality_weights.items()
                    ],
                    "retry_mult": 1.0 / max(sr, 1e-6),
                    "color": me["model"].color,
                }
                for me, useful_t, actual_t, sr in per_model_served
            ],
        }

    # Restore the user's original project order for UI stability.
    project_rows = [routed[p.uid] for p in projects if p.uid in routed]

    total_tokens = sum(r["tokens_day"] for r in project_rows)
    total_served = sum(r["served"] for r in project_rows)
    total_spilled = sum(r["spilled"] for r in project_rows)
    total_leaked = sum(r["leaked"] for r in project_rows)
    total_destroyed = sum(r["destroyed"] for r in project_rows)

    value_served = sum(r["value_served"] for r in project_rows)
    value_spilled = sum(r["value_spilled"] for r in project_rows)
    value_leaked = sum(r["value_leaked"] for r in project_rows)
    value_destroyed = sum(r["value_destroyed"] for r in project_rows)
    value_cloud = value_spilled + value_leaked  # money that leaves for the cloud
    value_lost = value_cloud + value_destroyed  # money not captured internally
    value_opportunity = value_served + value_lost

    # Procurement/TCO is a cluster commitment, so idle and unassigned pool GPUs still
    # cost money even though per-model tariffs above allocate only assigned GPU cost.
    cost_day = sum(gp.cost_per_gpu_hour * 24.0 * gp.count for gp in state.gpus)
    cost_per_m_served = (cost_day * 1e6 / total_served) if total_served > 0 else 0.0
    # Day-weighted gCO2/task averaged across the actual routed workload mix.
    _co2_numer = sum(me["served_co2_g_day"] for me in supply)
    co2_kg_day_total = _co2_numer / 1000.0
    _served_attempt_tasks = sum(me["served_tasks"] for me in supply)
    co2_g_per_task_avg = (_co2_numer / _served_attempt_tasks) if _served_attempt_tasks > 0 else 0.0
    margin_day = value_served - cost_day
    revenue_multiple = (value_served / cost_day) if cost_day > 0 else 0.0
    token_coverage = (total_served / total_tokens) if total_tokens > 0 else 0.0
    value_capture_rate = (value_served / value_opportunity) if value_opportunity > 0 else 0.0
    baseline_tokens_total = sum(r["baseline_tokens_day"] for r in project_rows)
    latent_active_tokens_total = sum(r["latent_active_tokens"] for r in project_rows)

    # Per-model "demand fit" rows (what each deployed model actually ended up serving).
    # Note `served_tokens` here is *actual* GPU-tokens consumed (including downgrade waste);
    # project-side `served` is *useful* tokens delivered. Per-model utilization is the actual
    # GPU pressure; the project-side fate bars track the useful work the project got done.
    model_rows = []
    total_cap = sum(me["daily_tokens_cap"] for me in supply)
    total_actual_served = sum(me["served_tokens"] for me in supply)
    total_gpu_weight = sum(max(me["gpu_count"], 0) for me in supply)
    time_utilization = (
        sum(me.get("used_fraction", 0.0) * max(me["gpu_count"], 0) for me in supply)
        / total_gpu_weight
        if total_gpu_weight > 0
        else 0.0
    )
    for me in supply:
        cap = me["daily_tokens_cap"]
        util = me.get("used_fraction", 0.0)
        saturated = cap > 0 and me.get("remaining_fraction", 1.0) <= 0.01
        served_tasks = me["served_tasks"]
        tpt = me["served_tokens"] / served_tasks if served_tasks > 0 else 0.0
        co2_day_g = me["served_co2_g_day"] / served_tasks if served_tasks > 0 else 0.0
        co2_night_g = me["served_co2_g_night"] / served_tasks if served_tasks > 0 else 0.0
        model_rows.append(
            {
                "am_uid": me["am_uid"],
                "model": me["model"],
                "name": me["model"].name,
                "color": me["model"].color,
                "quality": me["quality"],
                "effective_quality": me["effective_quality"],
                "quality_confidence": me["quality_confidence"],
                "token_efficiency": me["token_efficiency"],
                "gpu_count": me["gpu_count"],
                "peak_rps": me["peak_rps"],
                "daily_tokens_cap": cap,
                "served_tokens": me["served_tokens"],
                "utilization": util,
                "internal_pm": 0.0 if math.isinf(me["internal_pm"]) else me["internal_pm"],
                "internal_input_pm": 0.0 if math.isinf(me["internal_pm"]) else me["internal_pm"],
                "internal_output_pm": (
                    0.0
                    if math.isinf(me["internal_pm"])
                    else me["internal_pm"] / max(me["token_efficiency"], 1e-6)
                ),
                "gpu_cost_day": me["gpu_cost_day"],
                "tokens_per_task": tpt,
                "country": me.get("country", DEFAULT_COUNTRY),
                "grid_gco2_per_kwh_day": me.get("grid_gco2_per_kwh_day", 0.0),
                "grid_gco2_per_kwh_night": me.get("grid_gco2_per_kwh_night", 0.0),
                "co2_g_per_task_day": co2_day_g,
                "co2_g_per_task_night": co2_night_g,
                "co2_kg_day": me["served_co2_g_day"] / 1000.0,
                "saturated": saturated,
                "runnable": me["runnable"],
                "status": (
                    "NOT RUNNABLE"
                    if not me["runnable"]
                    else "SATURATED"
                    if saturated
                    else "IDLE"
                    if util < 0.05
                    else "OK"
                ),
            }
        )

    recommendations = (
        _marginal_gpu_recommendations(state, margin_day, value_cloud, value_destroyed, total_served)
        if include_recommendations
        else []
    )

    return {
        "ready": bool(supply) and bool(project_rows),
        "has_supply": bool(supply),
        "has_demand": bool(project_rows),
        "corpo_cloud": corpo_cloud,
        "day_shape_label": day_shape["label"],
        "day_shape_note": day_shape.get("note", ""),
        "peak_factor": peak_factor,
        "peak_factor_eff": peak_factor_eff,
        "batch_share": batch_share,
        "night_batching": night_batching,
        "projects": project_rows,
        "models": model_rows,
        "fates": {
            "total_tokens": total_tokens,
            "served_tokens": total_served,
            "spilled_tokens": total_spilled,
            "leaked_tokens": total_leaked,
            "destroyed_tokens": total_destroyed,
            "served_pct": (total_served / total_tokens * 100.0) if total_tokens > 0 else 0.0,
            "spilled_pct": (total_spilled / total_tokens * 100.0) if total_tokens > 0 else 0.0,
            "leaked_pct": (total_leaked / total_tokens * 100.0) if total_tokens > 0 else 0.0,
            "destroyed_pct": (total_destroyed / total_tokens * 100.0) if total_tokens > 0 else 0.0,
        },
        "value_served_day": value_served,
        "value_spilled_day": value_spilled,
        "value_leaked_day": value_leaked,
        "value_destroyed_day": value_destroyed,
        "value_cloud_day": value_cloud,
        "value_lost_day": value_lost,
        "avoidable_cloud_outflow_day": value_cloud,
        "cost_day": cost_day,
        "cost_per_m_served": cost_per_m_served,
        "co2_kg_day_total": co2_kg_day_total,
        "co2_g_per_task_avg": co2_g_per_task_avg,
        "margin_day": margin_day,
        "coverage": revenue_multiple,
        "revenue_multiple": revenue_multiple,
        "token_coverage": token_coverage,
        "value_capture_rate": value_capture_rate,
        "baseline_tokens_day": baseline_tokens_total,
        "latent_active_tokens_day": latent_active_tokens_total,
        "recommendations": recommendations,
        "total_gpus_used": sum(me["gpu_count"] for me in supply),
        "total_gpus": sum(gp.count for gp in state.gpus),
        "total_cap_tokens_day": total_cap,
        "actual_served_tokens": total_actual_served,
        "utilization": time_utilization,
        "workload_in_len": profile["in_len"],
        "workload_out_len": profile["out_len"],
    }


def _marginal_gpu_recommendations(
    state,
    base_margin: float,
    base_cloud: float,
    base_destroyed: float,
    base_served_tokens: float,
) -> list[dict]:
    """Estimate the best one-GPU expansions for currently deployed models.

    This stays inside calc.py to avoid importing state.py back into the module that state.py
    already imports. It therefore simulates only growth of existing assignments and retunes
    their topology with calc.py's local strategy helper.
    """
    seen: set[tuple[int, int]] = set()
    rows: list[dict] = []
    for am, gpu in _iter_resolved_models(state):
        if getattr(am.model, "embedding_profile", None) is not None:
            continue
        key = (am.uid, am.gpu_uid)
        if key in seen:
            continue
        seen.add(key)

        sim = copy.deepcopy(state)
        sim_am = next((m for m in sim.models if m.uid == am.uid), None)
        sim_gp = next((gp for gp in sim.gpus if gp.uid == am.gpu_uid), None)
        if sim_am is None or sim_gp is None:
            continue

        sim_gp.count += 1
        sim_am.gpu_count += 1
        sim_spec = spec_runtime_for(sim, sim_am, sim_am.model)
        strategy = default_strategy(
            sim_am.model,
            sim_am.gpu_count,
            sim_gp.gpu,
            sim.mu,
            sim.profiled_non_kv_gb,
            sim_am.prec,
            sim_spec,
        )
        if not valid_strategies(
            sim_am.model,
            sim_am.gpu_count,
            sim_gp.gpu,
            sim.mu,
            sim.profiled_non_kv_gb,
            sim_am.prec,
            sim_spec,
        ):
            continue
        sim_am.tp, sim_am.pp, sim_am.dp = strategy
        sim_am.prefill_tp, sim_am.prefill_pp, sim_am.prefill_dp = strategy

        projected = compute_revenue_projection(sim, include_recommendations=False)
        margin_gain = projected["margin_day"] - base_margin
        cloud_reduced = max(0.0, base_cloud - projected["value_cloud_day"])
        destroyed_reduced = max(0.0, base_destroyed - projected["value_destroyed_day"])
        score = margin_gain + 0.25 * cloud_reduced + 0.50 * destroyed_reduced
        if score <= 0 and margin_gain <= 0 and cloud_reduced <= 0 and destroyed_reduced <= 0:
            continue
        rows.append(
            {
                "model_name": sim_am.model.name,
                "gpu_name": sim_gp.gpu.name,
                "gpu_uid": sim_gp.uid,
                "am_uid": sim_am.uid,
                "added_gpus": 1,
                "new_gpu_count": sim_am.gpu_count,
                "margin_gain_day": margin_gain,
                "cloud_reduced_day": cloud_reduced,
                "destroyed_reduced_day": destroyed_reduced,
                "served_gain_tokens": max(
                    0.0, projected["fates"]["served_tokens"] - base_served_tokens
                ),
                "score": score,
            }
        )

    rows.sort(key=lambda r: (-r["score"], -r["margin_gain_day"], r["model_name"]))
    return rows[:MARGINAL_RECOMMENDATION_LIMIT]


def _model_kind_for_swap(model: Model) -> str:
    if getattr(model, "embedding_profile", None) is not None:
        return "embedding"
    if getattr(model, "realtime_profile", None) is not None:
        return "asr"
    return "llm"


def _portfolio_domain_quality(model: Model, projects) -> tuple[float, str, float]:
    """Demand/value-weighted quality across the active workload domain mix.

    The score is used only to shortlist and explain swap candidates; the full
    projection simulation still decides the recommendation economics.
    """
    weights: dict[str, float] = defaultdict(float)
    for project in projects:
        domain = normalize_quality_domain(getattr(project, "quality_domain", "general"))
        project_weights = normalize_quality_weights(
            getattr(project, "quality_weights", None),
            domain,
        )
        demand = max(0.0, float(getattr(project, "tokens_day", 0.0) or 0.0))
        latent = 0.25 * max(0.0, float(getattr(project, "latent_jobs_day", 0.0) or 0.0))
        value = max(0.01, float(getattr(project, "wtp_per_m", 0.0) or 0.0))
        project_weight = (demand + latent) * value
        for component_domain, component_weight in project_weights.items():
            weights[component_domain] += project_weight * component_weight
    if not weights or sum(weights.values()) <= 0:
        return effective_quality(model), QUALITY_DOMAIN_LABELS["general"], 0.0

    total = sum(weights.values())
    score = (
        sum(effective_quality(model, domain) * weight for domain, weight in weights.items()) / total
    )
    anchored = (
        sum(
            weight
            for domain, weight in weights.items()
            if model_domain_anchor(model, domain) is not None
        )
        / total
    )
    ranked = sorted(weights.items(), key=lambda item: (-item[1], item[0]))
    labels = [
        f"{QUALITY_DOMAIN_LABELS[domain]} {weight / total:.0%}" for domain, weight in ranked[:3]
    ]
    return score, " · ".join(labels), anchored


def _swap_candidate_shortlist(current: Model, state, current_kind: str, per_slot_cap: int):
    current_quality, _, _ = _portfolio_domain_quality(current, state.projects)
    candidates = []
    for cand_key, cand in MODELS.items():
        if (
            cand_key == current.key
            or getattr(cand, "hidden", False)
            or _model_kind_for_swap(cand) != current_kind
        ):
            continue
        portfolio_quality, _, _ = _portfolio_domain_quality(cand, state.projects)
        candidates.append((cand_key, cand, portfolio_quality))

    cap = max(1, int(per_slot_cap))
    nearest_n = max(1, cap // 2)
    quality_n = max(1, cap // 4)
    efficient_n = max(1, cap - nearest_n - quality_n)
    selected: list[tuple[str, Model, float]] = []
    seen: set[str] = set()
    groups = (
        sorted(candidates, key=lambda row: (abs(row[2] - current_quality), row[0]))[:nearest_n],
        sorted(
            candidates,
            key=lambda row: (
                -_portfolio_domain_quality(row[1], state.projects)[2],
                -row[2],
                row[0],
            ),
        )[:quality_n],
        sorted(
            candidates,
            key=lambda row: (
                row[1].active_params / max(float(row[1].token_efficiency), 1e-6),
                -row[2],
                row[0],
            ),
        )[:efficient_n],
    )
    for group in groups:
        for row in group:
            if row[0] not in seen:
                seen.add(row[0])
                selected.append(row)
    if len(selected) < cap:
        for row in sorted(candidates, key=lambda item: (abs(item[2] - current_quality), item[0])):
            if row[0] not in seen:
                seen.add(row[0])
                selected.append(row)
            if len(selected) >= cap:
                break
    return selected[:cap]


def _marginal_model_swap_recommendations(
    state,
    base_margin: float,
    base_cloud: float,
    base_destroyed: float,
    base_served_tokens: float,
    per_slot_cap: int = 20,
) -> list[dict]:
    """Estimate the best same-hardware model replacements for current deployments.

    Complement to _marginal_gpu_recommendations: instead of adding a GPU to an
    existing assignment, swap the deployed model for another catalog model of the
    same kind on the same GPUs, retune topology, and rescore the projection.
    Candidates combine the nearest portfolio-domain quality, the strongest domain
    quality, and the smallest inference work size, then are capped per slot to bound
    runtime. Scoring mirrors the GPU expansion recommender.
    """
    rows: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for am, gpu in _iter_resolved_models(state):
        key = (am.uid, am.gpu_uid)
        if key in seen:
            continue
        seen.add(key)
        current = am.model
        current_kind = _model_kind_for_swap(current)
        pool = _swap_candidate_shortlist(current, state, current_kind, per_slot_cap)
        current_portfolio_quality, quality_mix, current_anchor_share = _portfolio_domain_quality(
            current, state.projects
        )
        evaluated = 0
        for cand_key, cand, candidate_portfolio_quality in pool:
            if evaluated >= per_slot_cap:
                break
            sim = copy.deepcopy(state)
            sim_am = next((m for m in sim.models if m.uid == am.uid), None)
            sim_gp = next((gp for gp in sim.gpus if gp.uid == am.gpu_uid), None)
            if sim_am is None or sim_gp is None:
                continue
            sim_am.model_key = cand_key
            # Keep the deployment's precision when the candidate supports it;
            # fall back to bf16 rather than discarding a viable swap outright.
            chosen_prec = None
            chosen_spec = None
            for prec in (sim_am.prec, "bf16"):
                sim_am.prec = prec
                spec = spec_runtime_for(sim, sim_am, sim_am.model)
                if valid_strategies(
                    sim_am.model,
                    sim_am.gpu_count,
                    sim_gp.gpu,
                    sim.mu,
                    sim.profiled_non_kv_gb,
                    prec,
                    spec,
                ):
                    chosen_prec = prec
                    chosen_spec = spec
                    break
            if chosen_prec is None:
                continue
            evaluated += 1
            strategy = default_strategy(
                sim_am.model,
                sim_am.gpu_count,
                sim_gp.gpu,
                sim.mu,
                sim.profiled_non_kv_gb,
                chosen_prec,
                chosen_spec,
            )
            sim_am.tp, sim_am.pp, sim_am.dp = strategy
            sim_am.prefill_tp, sim_am.prefill_pp, sim_am.prefill_dp = strategy

            projected = compute_revenue_projection(sim, include_recommendations=False)
            candidate_portfolio_quality, _, candidate_anchor_share = _portfolio_domain_quality(
                cand, state.projects
            )
            margin_gain = projected["margin_day"] - base_margin
            cloud_reduced = max(0.0, base_cloud - projected["value_cloud_day"])
            destroyed_reduced = max(0.0, base_destroyed - projected["value_destroyed_day"])
            score = margin_gain + 0.25 * cloud_reduced + 0.50 * destroyed_reduced
            if score <= 0 and margin_gain <= 0 and cloud_reduced <= 0 and destroyed_reduced <= 0:
                continue
            rows.append(
                {
                    "current_key": am.model_key,
                    "current_name": current.name,
                    "candidate_key": cand_key,
                    "candidate_name": cand.name,
                    "gpu_name": sim_gp.gpu.name,
                    "gpu_count": sim_am.gpu_count,
                    "current_quality": current_portfolio_quality,
                    "candidate_quality": candidate_portfolio_quality,
                    "current_global_quality": effective_quality(current),
                    "candidate_global_quality": effective_quality(cand),
                    "quality_mix": quality_mix,
                    "current_anchor_share": current_anchor_share,
                    "candidate_anchor_share": candidate_anchor_share,
                    "prec": chosen_prec,
                    "margin_gain_day": margin_gain,
                    "cloud_reduced_day": cloud_reduced,
                    "destroyed_reduced_day": destroyed_reduced,
                    "served_gain_tokens": max(
                        0.0, projected["fates"]["served_tokens"] - base_served_tokens
                    ),
                    "margin_before_day": base_margin,
                    "margin_after_day": projected["margin_day"],
                    "cloud_before_day": base_cloud,
                    "cloud_after_day": projected["value_cloud_day"],
                    "destroyed_before_day": base_destroyed,
                    "destroyed_after_day": projected["value_destroyed_day"],
                    "served_before_tokens": base_served_tokens,
                    "served_after_tokens": projected["fates"]["served_tokens"],
                    "score": score,
                }
            )

    rows.sort(key=lambda r: (-r["score"], -r["margin_gain_day"], r["candidate_name"]))
    return rows[:MARGINAL_RECOMMENDATION_LIMIT]
