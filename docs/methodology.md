# What the estimates model

These notes used to sit at the bottom of the calculator page. They describe
what the technical estimates (step 3) include, and how to check them against a
real deployment. For the measurement plan, see
[calibration-benchmark.md](calibration-benchmark.md).

## Model scope

- Roofline estimates, with a continuous-batching approximation.
- Separate prefill, decode and embedding-encoder views.
- KV capacity is anchored to the requested per-accelerator memory, minus
  weights, minus the profiled non-KV runtime memory.
- MoE models use activated parameters for core compute, plus an adjustable
  expert-imbalance penalty.
- Empirical per-use-case prefix reuse for prefill-sensitive views.
- An explicit prefill/decode interference blend.
- Paged-KV runtime overhead and residual KV slack are modelled.
- A warning when tensor parallelism crosses nodes.
- Decision models: response time against load comes from a queue model
  (M/D/c, Erlang C halved), built from the estimated capacity and the time
  one request takes alone. It is not measured.

## Calibration

For vLLM deployments, what the model misses shows up in runtime metrics:
running requests, KV-cache usage, prefix-cache hits, prompt tokens and
generation tokens. Async scheduling and CUDA-graph capture sizing also move
real latency and throughput. For other runtimes, use the equivalent scheduler
and cache metrics.

In practice, fit the planner's knobs against representative measurements
rather than refining the closed-form model in isolation.
