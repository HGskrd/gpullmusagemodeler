"""Roofline throughput estimation engine for vLLM multi-model planning."""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from data import (
    GPU,
    Model,
    Bucket,
    CLOUD_MODELS,
    CORPO_CLOUD_DEFAULT,
    DIST_PRESETS,
    EMBEDDING_DOC_BUCKETS,
    INPUT_BUCKETS,
    OUTPUT_BUCKETS,
    BATCH_SIZES,
    TASK_PRESETS,
    DAY_SHAPES,
    DEFAULT_COUNTRY,
    ASR_WER_LANGUAGE_LABELS,
    ASR_WER_LANGUAGE_SOURCES,
    ASR_WER_LANGUAGES,
    ASR_WER_PLACEHOLDER,
    PUBLISHED_ASR_WER,
    EMBEDDING_DECONTAMINATED_BEIR_SOURCES,
    EMBEDDING_QUALITY_PLACEHOLDER,
    EMBEDDING_QUALITY_SOURCES,
    PUBLISHED_EMBEDDING_DECONTAMINATED_BEIR,
    PUBLISHED_EMBEDDING_QUALITY,
    normalize_precision,
    carbon_intensity_avg,
    corpo_cloud_models,
    effective_quality,
    model_success_rate,
    success_rate,
)

# Wall-clock GPU draw as a fraction of published board TDP during vLLM inference.
# vLLM-at-saturation is typically compute- or bandwidth-bound; measured draw on
# H100/MI300 commonly lands in the 0.6–0.8 range. Held central for transparency.
GPU_POWER_UTILIZATION = 0.70
LATENT_UNLOCK_STEEPNESS = 4.0
MARGINAL_RECOMMENDATION_LIMIT = 5

INTER_NODE_COLLECTIVE_BW = 25e9
DATA_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]  # Fixed to match BATCH_SIZES
EMBEDDING_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
USER_EXP_SWEEP = [
    1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
]
USER_EXP_FRACTIONS = [0.50, 0.75, 0.90, 0.95]
REALTIME_USER_SWEEP = [
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384,
]
MAX_REALTIME_USERS = 262_144
UNBOUNDED_BATCH = 1_000_000_000
LONG_CTX_DCP_SEQ = 32768
BATCH_AXIS_HEADROOM = 0.12
PROCESSING_PARETO_COLORS = ["#3266ad", "#1D9E75", "#BA7517", "#7F77DD", "#D85A30", "#A32D2D"]
NIGHT_HOURS = frozenset({22, 23, 0, 1, 2, 3, 4, 5})
NVIDIA_FP4_GPU_KEYS = frozenset({
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
})
MXFP4_GPU_KEYS = NVIDIA_FP4_GPU_KEYS | frozenset({"MI350X", "MI355X", "MI400"})


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
    bpe = m.kv_cache_bytes_per_elem(prec)
    full_layers, local_layers = _split_attention_layers(m.kv_layer_count, m.local_attention_layers)
    if m.is_mla:
        return (full_layers + local_layers) * (m.mla_kv_dim + m.mla_rope_dim) * 2 * bpe
    return (
        (full_layers * _kv_elems_per_layer(m, global_layer=True))
        + (local_layers * _kv_elems_per_layer(m, global_layer=False))
    ) * bpe


def _split_attention_layers(total_layers: int, local_layers: int) -> tuple[int, int]:
    local = min(max(local_layers, 0), max(total_layers, 0))
    return max(total_layers - local, 0), local


def _local_context_tokens(m: Model, seq_len: float) -> float:
    if m.local_attention_window <= 0:
        return max(seq_len, 0.0)
    return min(max(seq_len, 0.0), float(m.local_attention_window))


def _kv_projection_count(m: Model) -> int:
    return 1 if m.shared_key_value else 2


def _kv_elems_per_layer(m: Model, global_layer: bool = False) -> int:
    if m.is_mla:
        return (m.mla_kv_dim + m.mla_rope_dim) * 2
    if global_layer and m.global_kv_heads > 0:
        heads = m.global_kv_heads
        head_dim = m.global_head_dim or m.head_dim
    else:
        heads = m.kv_heads
        head_dim = m.head_dim
    return _kv_projection_count(m) * heads * head_dim


def _kv_bytes_per_layer(m: Model, prec: str, global_layer: bool = False) -> float:
    return _kv_elems_per_layer(m, global_layer=global_layer) * m.kv_cache_bytes_per_elem(prec)


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


def kv_cache_bytes_for_sequence(m: Model, seq_len: float, prec: str) -> float:
    seq = max(float(seq_len), 0.0)
    full_layers, local_layers = _split_attention_layers(m.kv_layer_count, m.local_attention_layers)
    return (
        full_layers * seq * _kv_bytes_per_layer(m, prec, global_layer=True)
        + local_layers * _local_context_tokens(m, seq) * _kv_bytes_per_layer(m, prec, global_layer=False)
    )


def per_replica_kv_cache_bytes(m: Model, seq_len: float, prec: str, pp: int, tp: int) -> float:
    pp = max(pp, 1)
    token_cache = kv_cache_bytes_for_sequence(m, seq_len, prec) / (pp * kv_shards(m, tp))
    linear_state = linear_attention_state_bytes(m, prec) / (pp * max(tp, 1))
    return token_cache + linear_state


def _linear_attention_work(m: Model, seq_len: float) -> float:
    layers = m.linear_attention_layer_count
    if layers <= 0:
        return 0.0
    heads = m.linear_attention_head_count
    head_dim = m.linear_attention_head_size
    return layers * heads * head_dim * head_dim * max(seq_len, 0.0)


def _decode_attention_work(m: Model, pr: int, avg_seq: float, pp: int) -> float:
    full_layers, local_layers = _split_attention_layers(m.attention_layer_count, m.local_attention_layers)
    full_width = m.attention_query_head_count * m.head_dim
    local_width = m.local_attention_head_count * m.head_dim
    full_work = full_layers * full_width * max(avg_seq, 0.0)
    local_work = local_layers * local_width * _local_context_tokens(m, avg_seq)
    linear_work = _linear_attention_work(m, 1.0)
    return 2 * pr * (full_work + local_work + linear_work) / max(pp, 1)


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
    attention_work = 2 * pr * layers * heads * head_dim * audio_tokens * window / pp
    return ffn_work + attention_work


def _prefill_attention_work(m: Model, pr: int, seq_len: int, pp: int) -> float:
    seq = max(float(seq_len), 0.0)
    full_layers, local_layers = _split_attention_layers(m.attention_layer_count, m.local_attention_layers)
    full_width = m.attention_query_head_count * m.head_dim
    local_width = m.local_attention_head_count * m.head_dim
    full_work = full_layers * full_width * seq * seq
    local_work = local_layers * local_width * seq * _local_context_tokens(m, seq)
    linear_work = _linear_attention_work(m, seq)
    return 2 * pr * (full_work + local_work + linear_work) / max(pp, 1)


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
        return g.fp4
    if prec == "nvfp4" and gpu_supports_nvfp4(g):
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
    seq_std = math.sqrt(in_std ** 2 + (out_std / 2.0) ** 2)
    heterogeneity = min(1.5, seq_std / max(avg_seq, 1.0))
    short_share = 0.65 * dist_share_leq(in_dist, INPUT_BUCKETS, 1024)
    short_share += 0.35 * dist_share_leq(out_dist, OUTPUT_BUCKETS, 128)
    return eff.paged_oh * _paged_kv_pressure(avg_seq, heterogeneity, short_share)


def fixed_paged_oh(seq_len: int, eff: EfficiencyParams, scale: float = 1.0) -> float:
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
    kv_heads = max(1, m.kv_heads)
    if tp <= kv_heads:
        return m.kv_heads % tp == 0
    return tp % kv_heads == 0


def kv_duplication_groups(m: Model, tp: int) -> int:
    kv_heads = max(1, m.kv_heads)
    if not tp_supported(m, tp) or tp <= kv_heads:
        return 1
    return tp // kv_heads


def kv_shards(m: Model, tp: int) -> int:
    return max(1, tp // kv_duplication_groups(m, tp))


def compute_memory(
    m: Model,
    tp: int,
    pp: int,
    g: GPU,
    mu: float,
    profiled_non_kv_gb: float,
    prec: str,
    eff: EfficiencyParams,
) -> Optional[MemoryResult]:
    requested = g.mem * mu
    weights = m.weight_bytes(prec) / (tp * pp)
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
        kv_per_token=kv_bytes_per_token(m, prec) / (pp * kv_shards(m, tp)),
    )


def valid_strategies(
    m: Model,
    gpu_count: int,
    g: GPU,
    mu: float,
    profiled_non_kv_gb: float,
    prec: str,
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
        if m.weight_bytes(prec) / (tp * pp) <= budget:
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
) -> tuple[int, int, int]:
    candidates = valid_strategies(m, gpu_count, g, mu, profiled_non_kv_gb, prec)
    if not candidates:
        return (max(gpu_count, 1), 1, 1)

    best = candidates[0]
    best_score = None
    requested = g.mem * mu
    for tp, pp, dp in candidates:
        profiled_non_kv = profiled_non_kv_bytes(tp, profiled_non_kv_gb)
        kv_headroom = max(0.0, requested - (m.weight_bytes(prec) / (tp * pp)) - profiled_non_kv)
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


def _dense_tp_oh(tp: int, pp: int, batch_tokens: int, m: Model, g: GPU, bw_eff: float, overlap: float) -> float:
    if tp <= 1:
        return 0.0
    collective_bw = _eff_collective_bw(tp, g) * bw_eff
    msg = batch_tokens * m.hidden_size * 2
    stage_layers = m.layers / pp
    comm_time = stage_layers * (msg * 2 * (tp - 1) / (tp * collective_bw))
    latency = stage_layers * 3e-6
    return (comm_time + latency) * (1 - overlap)


def _pp_boundary_oh(tp: int, pp: int, batch_tokens: int, m: Model, g: GPU, bw_eff: float) -> tuple[float, int]:
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
    batch_tokens: int,
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
) -> float:
    aw = _active_weight_bytes(m, prec)
    wt = (aw / (tp * pp)) / (g.effective_bw * eff.bw_eff)
    kv_read_bytes = pr * per_replica_kv_cache_bytes(m, avg_seq, prec, pp, tp)
    kv_time = kv_read_bytes / (g.effective_bw * eff.bw_eff)
    bt = wt + kv_time

    wf = 2 * m.active_params * pr / pp
    af = _decode_attention_work(m, pr, avg_seq, pp)
    ct = (wf + af + max(extra_flops, 0.0)) / (model_gpu_flops(g, m, prec) * tp * eff.comp_eff)

    comm = communication_breakdown(m, tp, pp, pr, avg_seq, g, eff)
    step = (max(bt, ct) + comm.total) * (1 + eff.overhead + paged_oh)
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
) -> Optional[DecodeResult]:
    mem = compute_memory(m, tp, pp, g, mu, profiled_non_kv_gb, prec, eff)
    if mem is None:
        return None

    pr = math.ceil(bs / dp)
    avg_seq = avg_in + avg_out / 2.0
    avg_kv = per_replica_kv_cache_bytes(m, avg_seq, prec, pp, tp)
    max_slots = int(mem.kv_budget / avg_kv) if avg_kv > 0 else 0
    if eff.sched_budget > 0:
        max_slots = min(max_slots, eff.sched_budget)
    if pr > max_slots:
        return None

    step = _decode_step_time(m, tp, pp, pr, g, prec, avg_seq, eff, paged_oh, extra_flops)
    return DecodeResult(
        tps=round(pr / step * dp),
        lat=round((step / pr) * 1e5) / 100,
        step_ms=round(step * 1e5) / 100,
        max_slots=max_slots * dp,
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
) -> int:
    result = compute_decode(m, tp, pp, max(dp, 1), dp, g, mu, profiled_non_kv_gb, prec, in_dist, out_dist, eff)
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
) -> Optional[PrefillResult]:
    mem = compute_memory(m, tp, pp, g, mu, profiled_non_kv_gb, prec, eff)
    if mem is None:
        return None
    if seq_len <= 0:
        return PrefillResult(tps=0, service_time=0.0, rps=math.inf, max_batch=UNBOUNDED_BATCH)

    pr = math.ceil(bs / dp)
    seq_kv = per_replica_kv_cache_bytes(m, seq_len, prec, pp, tp)
    max_per_replica = int(mem.kv_budget / seq_kv) if seq_kv > 0 else 0
    if pr > max_per_replica:
        return None

    ffn = 2 * m.active_params * pr * seq_len / pp
    att = _prefill_attention_work(m, pr, seq_len, pp)
    tf = ffn + att
    ct = tf / (model_gpu_flops(g, m, prec) * tp * eff.comp_eff)

    aw = _active_weight_bytes(m, prec)
    mt = (aw / (tp * pp)) / (g.effective_bw * eff.bw_eff)

    comm = communication_breakdown(m, tp, pp, pr * seq_len, seq_len, g, eff)
    t = (max(ct, mt) + comm.total) * (1 + eff.overhead * 1.3 + fixed_paged_oh(seq_len, eff, 0.35))
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
    dim = int(profile.late_interaction_dim or profile.output_dim) if profile.supports_late_interaction else int(profile.output_dim)
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
    mean_vectors = sum(share * embedding_vectors_per_input(m, seq) for share, seq, _bucket in weighted)
    mean_output = sum(share * embedding_output_bytes_per_input(m, seq) for share, seq, _bucket in weighted)
    mean_scratch = sum(share * embedding_scratch_bytes_per_input(m, seq, prec) for share, seq, _bucket in weighted)
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

    ffn = 2 * m.active_params * pr * seq / max(pp, 1)
    att = _prefill_attention_work(m, pr, seq, pp)
    ct = (ffn + att) / (model_gpu_flops(g, m, prec) * max(tp, 1) * eff.comp_eff)

    aw = _active_weight_bytes(m, prec)
    mt = (aw / (max(tp, 1) * max(pp, 1))) / (g.effective_bw * eff.bw_eff)
    output_time = (embedding_output_bytes_per_input(m, seq) * pr) / (g.effective_bw * eff.bw_eff)
    comm = communication_breakdown(m, tp, pp, pr * seq, seq, g, eff)

    t = (max(ct, mt) + output_time + comm.total) * (1 + eff.overhead * 1.2 + fixed_paged_oh(seq, eff, 0.20))
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
    max_per_replica = int(mem.kv_budget / stats.mean_scratch_bytes_per_input) if stats.mean_scratch_bytes_per_input > 0 else UNBOUNDED_BATCH
    if pr > max_per_replica:
        return None

    ffn = 2 * m.active_params * pr * stats.mean_seq_len / max(pp, 1)
    att = sum(
        share * _prefill_attention_work(m, pr, seq, pp)
        for share, seq, _bucket in _embedding_weighted_sequences(m, doc_dist, buckets)
    )
    ct = (ffn + att) / (model_gpu_flops(g, m, prec) * max(tp, 1) * eff.comp_eff)

    aw = _active_weight_bytes(m, prec)
    mt = (aw / (max(tp, 1) * max(pp, 1))) / (g.effective_bw * eff.bw_eff)
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
    result = compute_embedding(m, strat, max(strat[2], 1), seq_len, g, mu, profiled_non_kv_gb, prec, eff)
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
) -> Optional[DataResult]:
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
) -> int:
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
) -> Optional[UserExperienceResult]:
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
    pr = math.ceil(users / max(dp, 1))
    extra_flops = _realtime_audio_encoder_work(profile, pr, pp)
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
        result = compute_realtime_capacity(m, decode_strat, high, g, mu, profiled_non_kv_gb, prec, eff)
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
        result = compute_realtime_capacity(m, decode_strat, mid, g, mu, profiled_non_kv_gb, prec, eff)
        if result is not None and result.realtime_factor >= 1.0:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _label(am, model: Model, panel_suffix: str = "", include_prefill: bool = False) -> str:
    decode_label = strategy_label(am.tp, am.pp, am.dp)
    if include_prefill:
        prefill_label = strategy_label(am.prefill_tp, am.prefill_pp, am.prefill_dp)
        if prefill_label != decode_label:
            return f"{model.name} P {prefill_label} / D {decode_label} {am.prec.upper()}{panel_suffix}"
    return f"{model.name} {decode_label} {am.prec.upper()}{panel_suffix}"


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
    nearest = min(range(len(EMBEDDING_DOC_BUCKETS)), key=lambda i: abs(EMBEDDING_DOC_BUCKETS[i].length - seq_len))
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


def get_decode_bs(states: Optional[list] = None) -> list[int]:
    if not states:
        return list(BATCH_SIZES)

    from state import get_deployed

    capacities = []
    for state in states:
        eff = state.decode_efficiency
        for am in get_deployed(state, phase="decode"):
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
                )
            )
    return _batch_axis_sweep(capacities, BATCH_SIZES)


def get_realtime_bs(states: Optional[list] = None) -> list[int]:
    if not states:
        return list(REALTIME_USER_SWEEP)

    from state import get_deployed

    capacities = []
    for state in states:
        for am in get_deployed(state, phase="decode"):
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
                )
            )
    return _batch_axis_sweep(capacities, DATA_BATCH_SIZES)


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
) -> list[dict]:
    points = []
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


def chart_decode(state, batch_sizes: Optional[list[int]] = None, panel_suffix: str = "") -> list[dict]:
    from state import get_deployed

    datasets = []
    eff = state.decode_efficiency
    is_b = panel_suffix != ""
    batch_sizes = batch_sizes or BATCH_SIZES

    for am in get_deployed(state, phase="decode"):
        model = am.model
        if not _is_decode_pareto_model(model):
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
            )
            pts.append({"x": bs, "y": result.tps if result else None})
        datasets.append({
            "label": _label(am, model, panel_suffix, include_prefill=True),
            "data": pts,
            "borderColor": model.color,
            "backgroundColor": model.color + "12",
            "borderWidth": 1.5 if is_b else 2,
            "borderDash": [5, 3] if is_b else [],
            "fill": not is_b,
            "tension": 0.3,
            "pointRadius": 2.5,
            "spanGaps": False,
        })
    return datasets


def chart_pareto(state, panel_suffix: str = "") -> list[dict]:
    from state import get_deployed

    datasets = []
    eff = state.decode_efficiency
    is_b = panel_suffix != ""

    for am in get_deployed(state, phase="decode"):
        model = am.model
        if not _is_decode_pareto_model(model):
            continue
        gpu = am.gpu_spec
        if gpu is None:
            continue
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
            )
            if result:
                pts.append({"x": result.lat, "y": result.tps, "bs": bs})
        if pts:
            datasets.append({
                "label": _label(am, model, panel_suffix),
                "data": pts,
                "borderColor": model.color,
                "backgroundColor": model.color + "AA",
                "borderWidth": 1.5 if is_b else 2,
                "borderDash": [5, 3] if is_b else [],
                "showLine": True,
                "tension": 0.3,
                "pointRadius": 3.5,
            })
    return datasets


def chart_user_pareto(state, batch_sizes: Optional[list[int]] = None, panel_suffix: str = "") -> list[dict]:
    from state import get_deployed

    datasets = []
    eff = state.decode_efficiency
    is_b = panel_suffix != ""
    batch_sizes = batch_sizes or BATCH_SIZES

    for am in get_deployed(state, phase="decode"):
        model = am.model
        if not _is_decode_pareto_model(model):
            continue
        gpu = am.gpu_spec
        if gpu is None:
            continue
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
            )
            if result:
                pts.append({
                    "x": users,
                    "y": round((result.tps / users) * 100) / 100,
                    "users": users,
                    "total_tps": result.tps,
                    "lat": result.lat,
                })
        if pts:
            datasets.append({
                "label": _label(am, model, panel_suffix),
                "data": pts,
                "borderColor": model.color,
                "backgroundColor": model.color + "AA",
                "borderWidth": 1.5 if is_b else 2,
                "borderDash": [5, 3] if is_b else [],
                "showLine": True,
                "tension": 0.3,
                "pointRadius": 3.5,
            })
    return datasets


def chart_aggregate(state, batch_sizes: Optional[list[int]] = None, panel_suffix: str = "") -> list[dict]:
    from state import get_deployed

    datasets = []
    eff = state.decode_efficiency
    is_b = panel_suffix != ""
    deployed = get_deployed(state, phase="decode")
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
            )
            if result:
                total += result.tps
        agg.append({"x": bs, "y": total or None})
    datasets.append({
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
    })

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
            )
            pts.append({"x": bs, "y": result.tps if result else None})
        datasets.append({
            "label": f"{model.name}{panel_suffix}",
            "data": pts,
            "borderColor": model.color + ("44" if is_b else "77"),
            "borderWidth": 1,
            "borderDash": [4, 2] if is_b else [],
            "fill": False,
            "tension": 0.3,
            "pointRadius": 1.5,
            "spanGaps": False,
        })
    return datasets


def chart_data_processing(state, batch_sizes: Optional[list[int]] = None, panel_suffix: str = "") -> list[dict]:
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
            )
            pts.append({"x": bs, "y": result.tps if result else None})
        datasets.append({
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
        })

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
            )
            if result:
                total += result.tps
        agg.append({"x": bs, "y": total or None})
    datasets.append({
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
    })
    return datasets


def chart_embedding_throughput(state, batch_sizes: Optional[list[int]] = None, panel_suffix: str = "") -> list[dict]:
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
                pts.append({
                    "x": bs,
                    "y": None,
                    "seq_len": round(stats.mean_seq_len),
                    "p50_seq_len": stats.p50_seq_len,
                    "p90_seq_len": stats.p90_seq_len,
                    "p99_seq_len": stats.p99_seq_len,
                    "mode": profile.mode_label,
                    "max_batch": 0,
                })
                continue
            pts.append({
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
            })

        datasets.append({
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
        })
    return datasets


def chart_embedding_quality(state, panel_suffix: str = "") -> list[dict]:
    """Peak docs/s vs published retrieval quality, one dot per embedding model.

    Each model emits a single point — x = peak docs/s (max over the standard
    batch sweep at the current workload distribution), y = catalog quality in
    [0, 1]. Bytes-per-doc and vec/s are attached to the point so the front-end
    can encode storage cost via dot size and surface multi-vector blowup in the
    tooltip. Decontaminated BEIR is included as optional hover context when a
    sourced score is available.
    """
    datasets = []
    is_b = panel_suffix != ""
    doc_dist = _embedding_doc_dist_for_state(state)

    for am, gpu in _iter_resolved_models(state):
        model = am.model
        profile = getattr(model, "embedding_profile", None)
        if profile is None:
            continue
        quality = PUBLISHED_EMBEDDING_QUALITY.get(model.key)
        if quality is None:
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
            "quality_metric": "Published retrieval nDCG@10",
            "source": EMBEDDING_QUALITY_SOURCES.get(model.key, ""),
            "decontaminated_beir_quality": decontaminated_beir,
            "decontaminated_beir_source": EMBEDDING_DECONTAMINATED_BEIR_SOURCES.get(model.key, ""),
            "placeholder": is_placeholder,
        }

        datasets.append({
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
        })
    return datasets


def embedding_quality_axis_range(datasets: list[dict], margin_ratio: float = 0.08, min_margin: float = 0.01) -> dict[str, float]:
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


def chart_processing_pareto(state, batch_sizes: Optional[list[int]] = None, panel_suffix: str = "") -> list[dict]:
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
                )
                if result:
                    total_tps += result.tps
            total_rps = (total_tps / tokens_per_req) if tokens_per_req > 0 else 0.0
            pts.append({
                "x": bs,
                "y": round(total_rps * 100) / 100 if total_tps else None,
                "rps": round(total_rps * 100) / 100 if total_tps else None,
                "tps": total_tps or None,
                "in_len": in_len,
                "out_len": out_len,
                "workload": preset_name,
            })

        color = PROCESSING_PARETO_COLORS[idx % len(PROCESSING_PARETO_COLORS)]
        datasets.append({
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
        })
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
        )
        datasets.append({
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
        })
    return datasets


def chart_realtime_capacity(state, batch_sizes: Optional[list[int]] = None, panel_suffix: str = "") -> list[dict]:
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
                pts.append({
                    "x": users,
                    "y": None,
                    "users": users,
                    "max_users": max_users,
                    "required_tps": profile.tokens_per_second,
                    "target_delay_ms": profile.target_delay_ms,
                })
                continue

            pts.append({
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
            })
        if pts:
            datasets.append({
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
            })
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
            pts.append({
                "x": wer,
                "y": max_users,
                "language": ASR_WER_LANGUAGE_LABELS.get(language, language),
                "source": sources.get(language, ""),
                "wer": wer,
                "max_users": max_users,
                "placeholder": is_placeholder,
                "asr_mode": "streaming" if getattr(profile, "streaming", True) else "non-streaming",
            })
        if not pts:
            continue
        pts.sort(key=lambda p: p["x"])

        datasets.append({
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
        })
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
        )
        if not points:
            continue
        peak = points[-1]
        cells = []
        for frac in USER_EXP_FRACTIONS:
            sample = _sample_user_exp_curve(points, peak["arrival_rps"] * frac)
            if sample is None:
                cells.append(None)
                continue
            cells.append({
                "lat": round(sample["decode_step_ms"], 1),
                "resp_s": round(sample["response_s"], 2),
                "ttft_ms": round(sample["ttft_ms"], 1),
            })
        rows.append({
            "model": model,
            "config": f"{strategy_label(am.tp, am.pp, am.dp)} {am.prec.upper()}",
            "prec": am.prec,
            "peak_rps": round(peak["arrival_rps"] * 100) / 100,
            "peak_resp_s": round(peak["response_s"] * 100) / 100,
            "peak_inflight": round(peak["inflight"], 1),
            "cells": cells,
        })
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
    in_len = avg_dist(in_preset["in"], INPUT_BUCKETS)
    out_len = avg_dist(out_preset["out"], OUTPUT_BUCKETS)
    if in_len <= 0 or out_len <= 0:
        in_len = float(fallback["in_len"])
        out_len = float(fallback["out_len"])
    return {
        "in_len": in_len,
        "out_len": out_len,
        "tokens_per_request": max(1.0, in_len + out_len),
    }


def _stall_curve(load: float) -> float:
    """Map requested load (fraction of peak capacity) → served fraction.
    Below 100% runs clean. 100–115% thrashes (KV pressure, scheduler contention). Above 115% stalls."""
    if load <= 0:
        return 0.0
    if load <= 1.0:
        return load
    if load <= 1.15:
        # Linear decline from 1.0 (at load=1.0) down to 0.70 (at load=1.15).
        return 1.0 - 2.0 * (load - 1.0)
    return 0.55  # stall floor


def _apply_night_batching(weights: list[float], effective_shift: float, night_hours: frozenset) -> tuple[list[float], float]:
    """Move `effective_shift` fraction of each daytime hour's demand into the night hours (evenly).
    Returns (new weights, total fraction of original daily demand shifted)."""
    if effective_shift <= 0 or not weights:
        return list(weights), 0.0
    new = list(weights)
    shifted_total = 0.0
    for h, w in enumerate(weights):
        if h in night_hours:
            continue
        delta = w * effective_shift
        new[h] -= delta
        shifted_total += delta
    night_count = max(1, len(night_hours))
    per_night = shifted_total / night_count
    for h in night_hours:
        new[h] += per_night
    orig_total = sum(weights) or 1.0
    return new, shifted_total / orig_total


def _best_deployment_result_for_model(state, am, gpu: GPU, in_len: int, out_len: int, batch_sizes: list[int]) -> Optional[DeploymentPeakResult]:
    best: Optional[DeploymentPeakResult] = None
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
            state.prefix_hit_rate,
            state.prefill_efficiency,
            state.decode_efficiency,
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


def _cloud_price_per_m_in_preset(
    difficulty: float,
    min_success: float,
    quality_floor: float,
    profile: dict,
    prefix_hit_rate: float,
    preset_name: str,
) -> tuple[Optional[dict], float]:
    """Cheapest cloud model in the active corpo preset that can serve a project with the
    given (difficulty, min_success_rate). Effective $/M is computed apples-to-apples with
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
    for key in corpo_cloud_models(preset_name):
        cloud = CLOUD_MODELS.get(key)
        if cloud is None:
            continue
        cloud_quality = float(cloud.get("quality", 0.5))
        cloud_eff = max(float(cloud.get("token_efficiency", 1.0)), 1e-6)
        if cloud_quality + 1e-9 < quality_floor:
            continue
        if success_rate(cloud_quality, difficulty) + 1e-9 < min_success:
            continue
        sticker = (
            (uncached / 1e6) * cloud["in_per_m"]
            + (cached / 1e6) * cloud["cached_in_per_m"]
            + ((out_len / cloud_eff) / 1e6) * cloud["out_per_m"]
        )
        # Token efficiency affects generated tokens, not the fixed prompt payload.
        price_pm = sticker / (tokens_per_req / 1e6)
        if best is None or price_pm < best[0]:
            best = (price_pm, cloud | {"key": key})

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
    day_shape = DAY_SHAPES.get(getattr(state, "projection_day_shape", "workday")) or DAY_SHAPES["workday"]
    day_weights = day_shape["weights"] or [1.0] * 24
    night_weights = [1.0 if h in NIGHT_HOURS else 0.0 for h in range(24)]
    task_il = int(getattr(state, "task_il", profile["in_len"]))
    task_ol = int(getattr(state, "task_ol", profile["out_len"]))
    supply = []
    for am, gpu in _iter_resolved_models(state):
        if getattr(am.model, "is_realtime_only", False) or getattr(am.model, "embedding_profile", None) is not None:
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
        )
        batch_sizes = _batch_axis_sweep([cap], DATA_BATCH_SIZES)
        best = _best_deployment_result_for_model(
            state, am, gpu, profile["in_len"], profile["out_len"], batch_sizes
        )
        peak_rps = best.rps if (best and best.rps > 0) else 0.0
        # Sustainable daily token capacity: honor peak-hour headroom so we don't promise
        # throughput the day-shape can't actually sustain without thrashing.
        daily_tokens_cap = (
            peak_rps * 3600.0 * 24.0 * tokens_per_req / peak_factor_eff
            if peak_rps > 0 else 0.0
        )
        gpu_cost_day = pool_rate.get(am.gpu_uid, 0.0) * am.gpu_count
        internal_pm = (gpu_cost_day * 1e6 / daily_tokens_cap) if daily_tokens_cap > 0 else math.inf
        tokens_per_sec_peak = peak_rps * tokens_per_req
        tpt = tokens_per_task(am.model, task_il, task_ol)
        country = pool_country.get(am.gpu_uid, DEFAULT_COUNTRY)
        grid_day = carbon_intensity_avg(country, day_weights)
        grid_night = carbon_intensity_avg(country, night_weights)
        co2_task_day = co2_g_per_task(gpu, am.gpu_count, tpt, tokens_per_sec_peak, grid_day)
        co2_task_night = co2_g_per_task(gpu, am.gpu_count, tpt, tokens_per_sec_peak, grid_night)
        supply.append({
            "am": am,
            "am_uid": am.uid,
            "gpu": gpu,
            "gpu_uid": am.gpu_uid,
            "gpu_count": am.gpu_count,
            "model": am.model,
            "quality": float(am.model.quality),
            "effective_quality": effective_quality(am.model),
            "quality_confidence": min(max(float(getattr(am.model, "quality_confidence", 1.0)), 0.0), 1.0),
            "token_efficiency": max(float(am.model.token_efficiency), 1e-6),
            "peak_rps": peak_rps,
            "daily_tokens_cap": daily_tokens_cap,
            "remaining_cap": daily_tokens_cap,
            "served_tokens": 0.0,
            "gpu_cost_day": gpu_cost_day,
            "internal_pm": internal_pm,
            "tokens_per_task": tpt,
            "country": country,
            "grid_gco2_per_kwh_day": grid_day,
            "grid_gco2_per_kwh_night": grid_night,
            "co2_g_per_task_day": co2_task_day,
            "co2_g_per_task_night": co2_task_night,
            "runnable": peak_rps > 0,
        })
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
    tokens_per_req = max(1.0, profile["tokens_per_request"])
    corpo_cloud = getattr(state, "corpo_cloud", CORPO_CLOUD_DEFAULT)
    day_shape = DAY_SHAPES.get(state.projection_day_shape) or DAY_SHAPES["workday"]
    weights = day_shape["weights"] or [1.0]
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

    # Project routing — serve higher WTP first, so demand destruction falls on lower-value
    # workloads when capacity is tight (realistic triage).
    projects_sorted = sorted(projects, key=lambda p: (-float(p.wtp_per_m), p.uid))
    routed: dict[int, dict] = {}
    for p in projects_sorted:
        difficulty = float(getattr(p, "difficulty", 0.5))
        slo = float(getattr(p, "min_success_rate", 0.85))
        quality_floor = float(getattr(p, "quality_floor", 0.0))
        project_profile = _project_workload_profile(p, profile)
        cloud_info, cloud_pm = _cloud_price_per_m_in_preset(
            difficulty, slo, quality_floor, project_profile, prefix_hit_rate, corpo_cloud,
        )
        cloud_blocked = cloud_info is None
        wtp = float(p.wtp_per_m)
        total = max(0.0, float(p.tokens_day))
        required_caps = getattr(p, "requires", frozenset()) or frozenset()

        # Candidate list with capability + success-rate gates. `useful tokens` = work the
        # project needs done; token efficiency affects generated/output tokens only, so the
        # actual GPU tokens burned per useful token depends on this workload's input/output mix.
        candidates: list[dict] = []
        cap_filtered = False
        slo_filtered = False
        for me in supply:
            if not me["runnable"]:
                continue
            if not (required_caps <= me["model"].capabilities):
                cap_filtered = True
                continue
            if me["effective_quality"] + 1e-9 < quality_floor:
                slo_filtered = True
                continue
            sr = model_success_rate(me["model"], difficulty)
            if sr + 1e-9 < slo:
                slo_filtered = True
                continue
            token_mult = _actual_token_multiplier(
                me["token_efficiency"],
                float(project_profile["in_len"]),
                float(project_profile["out_len"]),
            )
            retry_mult = 1.0 / max(sr, 1e-6)
            candidates.append({
                "me": me,
                "success_rate": sr,
                "retry_mult": retry_mult,
                "token_mult": token_mult * retry_mult,
                "effective_pm": me["internal_pm"] * token_mult * retry_mult,
            })
        candidates.sort(key=lambda c: c["effective_pm"])

        # Latent demand activates smoothly around the unlock price. This keeps the configured
        # unlock as the midpoint while avoiding discontinuous portfolio demand jumps.
        baseline_tokens = total
        latent_pool = max(0.0, float(getattr(p, "latent_jobs_day", 0.0)))
        unlock_price = float(getattr(p, "unlock_price_per_m", 0.0))
        cheapest_pm = candidates[0]["effective_pm"] if candidates else float("inf")
        latent_activation = latent_activation_share(cheapest_pm, unlock_price) if candidates else 0.0
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
            if me["remaining_cap"] <= 0:
                continue
            if c["effective_pm"] > price_cap:
                continue
            useful_remaining = total - served
            if useful_remaining <= 0:
                break
            useful_take = min(useful_remaining, me["remaining_cap"] / c["token_mult"])
            if useful_take <= 0:
                continue
            actual_take = useful_take * c["token_mult"]
            me["remaining_cap"] -= actual_take
            me["served_tokens"] += actual_take
            per_model_served.append((me, useful_take, actual_take, c["success_rate"]))
            internal_cost += (actual_take / 1e6) * me["internal_pm"]
            tpt_m = me.get("tokens_per_task", 0.0)
            if tpt_m > 0:
                co2_g_day_project += (actual_take / tpt_m) * me.get("co2_g_per_task_day", 0.0)
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
        value_served = sum((useful_t / 1e6) * value_basis * sr for _, useful_t, _, sr in per_model_served)
        baseline_tokens_per_task = max(float(project_profile["tokens_per_request"]), 1.0)
        tasks_served_day = served / baseline_tokens_per_task
        co2_g_per_task_project = (co2_g_day_project / tasks_served_day) if tasks_served_day > 0 else 0.0
        routed[p.uid] = {
            "project": p,
            "name": p.name,
            "difficulty": difficulty,
            "tokens_day": total,
            "cloud_pm": 0.0 if cloud_blocked else cloud_pm,
            "cloud_label": "blocked — no compatible cloud" if cloud_blocked else cloud_info["label"],
            "cloud_vendor": "" if cloud_blocked else cloud_info["vendor"],
            "cloud_regions": () if cloud_blocked else cloud_info.get("regions", ()),
            "cloud_grid_gco2_per_kwh": 0.0 if cloud_blocked else cloud_info.get("grid_gco2_per_kwh", 0.0),
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
            "cap_blocked_for_project": cap_filtered and not candidates,
            "slo_blocked_for_project": slo_filtered and not candidates,
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

    cost_day = sum(gp.cost_per_gpu_hour * 24.0 * gp.count for gp in state.gpus)
    cost_per_m_served = (cost_day * 1e6 / total_served) if total_served > 0 else 0.0
    # Day-weighted gCO2/task averaged over served demand (0 if nothing served).
    _co2_numer = sum(me["served_tokens"] * me.get("co2_g_per_task_day", 0.0) / max(me.get("tokens_per_task", 0.0), 1e-9) for me in supply)
    co2_kg_day_total = _co2_numer / 1000.0
    co2_g_per_task_avg = (_co2_numer / sum(me["served_tokens"] / max(me.get("tokens_per_task", 0.0), 1e-9) for me in supply)) if total_served > 0 else 0.0
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
    for me in supply:
        cap = me["daily_tokens_cap"]
        util = (me["served_tokens"] / cap) if cap > 0 else 0.0
        saturated = cap > 0 and me["remaining_cap"] <= max(cap * 0.01, 1.0)
        tpt = me.get("tokens_per_task", 0.0)
        co2_day_g = me.get("co2_g_per_task_day", 0.0)
        co2_night_g = me.get("co2_g_per_task_night", 0.0)
        co2_g_day_total = co2_day_g * (me["served_tokens"] / tpt) if tpt > 0 else 0.0
        model_rows.append({
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
                0.0 if math.isinf(me["internal_pm"])
                else me["internal_pm"] / max(me["token_efficiency"], 1e-6)
            ),
            "gpu_cost_day": me["gpu_cost_day"],
            "tokens_per_task": tpt,
            "country": me.get("country", DEFAULT_COUNTRY),
            "grid_gco2_per_kwh_day": me.get("grid_gco2_per_kwh_day", 0.0),
            "grid_gco2_per_kwh_night": me.get("grid_gco2_per_kwh_night", 0.0),
            "co2_g_per_task_day": co2_day_g,
            "co2_g_per_task_night": co2_night_g,
            "co2_kg_day": co2_g_day_total / 1000.0,
            "saturated": saturated,
            "runnable": me["runnable"],
            "status": (
                "NOT RUNNABLE" if not me["runnable"]
                else "SATURATED" if saturated
                else "IDLE" if util < 0.05
                else "OK"
            ),
        })

    recommendations = (
        _marginal_gpu_recommendations(state, margin_day, value_cloud, value_destroyed, total_served)
        if include_recommendations else []
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
        "utilization": (total_actual_served / total_cap) if total_cap > 0 else 0.0,
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
        strategy = default_strategy(
            sim_am.model,
            sim_am.gpu_count,
            sim_gp.gpu,
            sim.mu,
            sim.profiled_non_kv_gb,
            sim_am.prec,
        )
        if not valid_strategies(sim_am.model, sim_am.gpu_count, sim_gp.gpu, sim.mu, sim.profiled_non_kv_gb, sim_am.prec):
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
        rows.append({
            "model_name": sim_am.model.name,
            "gpu_name": sim_gp.gpu.name,
            "gpu_uid": sim_gp.uid,
            "am_uid": sim_am.uid,
            "added_gpus": 1,
            "new_gpu_count": sim_am.gpu_count,
            "margin_gain_day": margin_gain,
            "cloud_reduced_day": cloud_reduced,
            "destroyed_reduced_day": destroyed_reduced,
            "served_gain_tokens": max(0.0, projected["fates"]["served_tokens"] - base_served_tokens),
            "score": score,
        })

    rows.sort(key=lambda r: (-r["score"], -r["margin_gain_day"], r["model_name"]))
    return rows[:MARGINAL_RECOMMENDATION_LIMIT]
