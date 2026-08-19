"""Roofline throughput estimation engine for the GPU/LLM Usage Modeler."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

from data import (
    BATCH_SIZES,
    DIST_PRESETS,
    EMBEDDING_DOC_BUCKETS,
    GPU,
    INPUT_BUCKETS,
    OUTPUT_BUCKETS,
    Bucket,
    Model,
    SpeculativeProfile,
    normalize_precision,
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
