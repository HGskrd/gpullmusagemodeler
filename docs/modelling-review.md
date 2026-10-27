# Modelling review: demand timing, latent demand, cloud gateway

Reviewed 2026-10-24 while reworking the Workload step. Points 1–4 were
changed in the engine the same day; point 5 is still open.

## 1. Serving was computed on daily totals (changed)

**Before.** Each deployment's daily capacity was its peak-hour throughput ×
24 h ÷ the day shape's peak/mean ratio (1.89 for "Enterprise workday"). That
treats every off-peak hour as unusable, even when demand is waiting. Overflow
to cloud was also spread over the whole day.

**Effect.** Exact for unsaturated deployments, pessimistic once saturated.
With the workday shape and demand 1.5–3× the old daily capacity, the GPUs
really serve about 20 points more of the demand than reported. For example,
at 2× they serve 71% of it, not 50%.

**Now.** Capacity is booked hour by hour (`engine/economics.py`,
`allocate_capacity`). Each deployment offers 1/24 of its full-load daily
tokens in every hour. Interactive demand arrives on its use case's day shape
and only uses its own hour. `daily_tokens_cap` is the 24 h full-load total,
and utilization is the share of the day actually booked. A deployment shows
SATURATED when at least one hour is fully booked.

## 2. Latent demand and willingness to pay were always on (changed)

**Before.** The sample's entered workload was 16.0B tok/day; the results
used 38.3B. 22.3B was latent demand that activates when the internal price
is low, and use cases were refused when every route cost more than their
willingness to pay (WTP). Those are chargeback concepts. They also drove
step 3's saturation, even with the economics hidden.

**Now.** `PlannerState.price_driven_demand` (default off, "Price-driven
demand" on the Assumptions line):
- **Off (sizing):** demand is the workload entered. Local GPUs serve whatever
  they can; leftovers go to cloud when a qualifying model exists, otherwise
  they are lost.
- **On (chargeback):** the previous behaviour, with latent activation, price
  caps, and destruction when cloud costs more than WTP.

"Equal demand" only matters with price-driven demand on, so it only shows then.

## 3. Night batching did not fill troughs (changed)

**Before.** With the switch on, batch-eligible use cases were treated as
flat over 24 h, but in their own slice of GPU time. They never used the idle
troughs left by interactive traffic. The portfolio figure was a linear blend
(1 − b)·pf + b, which gives 1.36 at 61% batch, where trough-filling could
take it close to 1.0.

**Now.** Batch use cases are routed after all interactive ones and take
whatever hours are still free. The global switch is gone: no planner would
run deferrable work at peak.

## 4. One day shape for the whole portfolio (changed)

**Now.** Each use case has its own "When": Office hours, 24/7, Consumer
evening, Follow the sun, or Batch (any free hour), in the Workload table.
Legacy scenarios give every use case the old portfolio shape, and the
definition library seeds new rows. The "Night batch" arrival curve is not
offered per use case, to avoid confusion with Batch.

## 5. Other findings

- **Cloud capability data (changed).** 23 of 28 cloud models carry no
  capability list, and the gate treated missing data as "no capabilities".
  So every use case needing tools, reasoning or long context was blocked from
  cloud, including Claude Opus 5. Missing data now passes the gate and is
  flagged `capabilities_unverified`.
  - Open: add sourced capability lists to `data/cloud.py`, per the catalog
    rules in AGENTS.md.
- **Wording (changed).** "No model meets the bar" said "no catalog model"; it
  only checks deployed models and the chosen gateway.
- **Preset calibration (open).** Deep research, Contract review, Security
  alert and AML/KYC still need more quality than any model in the default
  gateways (for example, Deep research's quality floor is 0.85; Claude Opus 5
  is 0.84). The October 2026 gateway (Opus 5.5) serves them. Check whether
  those floors and SLOs are intended.
- **Value basis (open, by design).** Value served prices local tokens at the
  cheapest cloud substitute, so changing the gateway changes "value served"
  with no change on the GPUs. That is correct for chargeback but misleading as
  a headline; relabel or move it if it confuses users.
- **Gateway scope.** The gateway is now set for A and B together, so you
  can't compare two gateways as A vs B. Per-config gateways remain in the
  state and in saved scenarios.

## Planner report, sample scenario

Before (cloud on):
- A: demand 38.3B (16.0B entered); served 46%, cloud 24%, lost 29%; all 3
  deployments shown 100% saturated.
- B: demand 61.8B; served 11%, cloud 43%, lost 46%; all saturated.

After (sizing mode, cloud on):
- A: demand 16.0B; served 68%, cloud 16%, lost 16%. Fleet 35% busy over the
  day: Qwen 3.5 88%, Gemma 4 26B 17%, Mistral Small 4 idle because it meets
  few use cases' quality bar.
- B: demand 16.0B; served 47%, cloud 37%, lost 16%. Gemma 4 31B and Qwen 3.8
  27B saturated at peak; Nemotron 36%.

What is still not simulated: queueing within an hour, requests arriving
spread unevenly across the GPUs, and the KV memory taken by prefix caching.
