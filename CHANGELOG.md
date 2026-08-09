# Changelog

- Added Gemma 4 E2B, E4B, and 12B Unified as separate non-streaming multilingual ASR catalog entries while retaining their general-purpose LLM entries. The ASR profiles use Google's published 40 ms audio-token rate, 30-second input limit, E2B/E4B 305M Conformer encoder metadata, encoder-free 12B projection, and English/French FLEURS WER.

All notable changes to this project will be documented in this file.

## 2026-08-09

### Added

- Added the released MiniMax M3 sparse-MoE catalog entry, NVIDIA Nemotron 3 Embed 8B/1B encoder entries, exact official MiniMax M3 and Nemotron 1B NVFP4 artifact footprints, and published seven-token DSpark profiles for Kimi K3 and DeepSeek V4 Flash 0731.
- Added explicit long-context API price tiers for GPT-5.6 and Gemini 3.1 Pro, applied only when input length exceeds each vendor threshold.
- Added conservative Kimi K3 Block AttnRes prefill activation capacity accounting and an explicit warning that expert-parallel MoE dispatch/combine traffic is not yet included.

### Changed

- Renamed the user-facing product from “vLLM multi-model planner” to “GPU/LLM Usage Modeler.” The estimator is runtime-neutral and now describes vLLM as a calibration/runtime example rather than a requirement, which better reflects Tenstorrent, Furiosa, Intel, AMD, NVIDIA, and Apple accelerator profiles.
- Replaced the DeepSeek V4 Flash preview proxy with the official 0731 configuration and 1M context; kept V4 Pro explicitly labeled Preview with dated assumptions.
- Updated Gemini 3.6 Flash, Claude Opus 5, GPT-5.6 Terra/Luna prices, and Gemini Flash-Lite context from current first-party API documentation.
- Corrected MI455X, Helios, and the MI400 compatibility profile to AMD's launched 432 GB / 23.3 TB/s / 5.0 PF dense BF16 / 20.1 PF FP8 / 40.3 PF MXFP4 / 3.6 TB/s scale-up specifications. The unpublished 1.5 kW power input remains labeled as a planner proxy.
- Split token-growing attention KV traffic from fixed recurrent-state traffic so speculative verification no longer multiplies recurrent-state reads by the draft depth.

### Fixed

- GPU vendor names in the hardware picker now scroll with their cards instead of sticking inside the dropdown and visually detaching from the values below.

## 2026-07-27

### Changed

- Replaced the Kimi K3 launch proxy with the exact open-weight architecture from Moonshot AI's config and technical report: 2.78T total / 104.2B active parameters, 93 layers, 69 lower-bounded/full-rank KDA + 24 NoPE Gated MLA layers, 96 heads, 7,168 hidden width, 12-layer Block Attention Residuals, and Stable LatentMoE with 16-of-896 routed experts, two shared experts, a 3,584-wide latent path, normalized aggregation, SiTU-GLU, and Quantile Balancing.
- Added an exact native-MXFP4 artifact profile from K3's 96-shard safetensors index (1.561 TB), retaining the report's BF16 non-expert path in active-weight bandwidth and mixed-compute estimates. Updated K3 to its direct Artificial Analysis 57-point / 130M-token row and added official coding, reasoning, long-context, and vision quality anchors.
- Promoted the local catalog identifier from the launch-only `kimi-k3-preview` key to the final `kimi-k3` key.

## 2026-07-24

### Changed

- Corrected GLM-5.1 to its official 78-layer MLA/DSA geometry and 202,752-token context. GLM-5.1/5.2 throughput now models top-2048 sparse attention plus full-context indexer work; GLM-5.2 evaluates 21 full indexers and reuses their selections across the remaining layers, reproducing the published approximately 2.9× 1M-context per-token FLOP reduction.
- Completed a source-backed model-catalog audit pass. Corrected GLM 4.5–5, Qwen 3.5 MoE, Gemma 4, MiMo V2.5, Laguna M.1, Cohere North Mini Code, LFM, Nemotron, Mistral/Ministral/Devstral, Croissant, DenseOn, and Moonshine planner geometry/context values. Added separate local/global KV-head accounting so hybrid attention models shard the two cache types correctly. Audio architectures whose encoder or parallel-stream work cannot yet be represented are now explicitly labeled as conservative proxies.

## 2026-07-23

### Added

- Added real multi-axis task-quality profiles. Built-in workloads now carry normalized domain weights (for example repository coding is 70% coding, 20% reasoning, and 10% long-context), routing uses a conservative weighted geometric blend, scenario exports preserve the vector, and the use-case editor exposes every component.
- Added GLM-5.2 with its published 744B backbone / 40B-active planner accounting, 78-layer IndexShare DSA configuration, 1M context, native MTP, and sourced reasoning/coding evidence.
- Added provisional, explicitly labeled SWE-Bench Pro → SWE-bench Verified-equivalent calibration for Laguna S 2.1, GLM-5.1, and GLM-5.2. The frozen overlap-cohort fit is `verified% = 40.9508834 + 0.6464964 × pro%`; raw scores and reduced confidence remain attached to each anchor.

### Changed

- Removed Laguna S 2.1's unsourced 95M-output-token proxy (`η=0.11`). Until a directly comparable Artificial Analysis verbosity row exists, the planner uses neutral `η=1.0` instead of allowing an unrelated family proxy to dominate fleet economics.
- Equal-WTP workloads now route harder, more capability-constrained contracts first rather than using preset insertion order as the tie-breaker.
- Model-fit diagnostics distinguish capability, absolute quality-floor, SLO, and capacity rejection, and projection rows expose the weighted quality components and benchmark/fallback used for each.

## 2026-07-22

### Added

- Added researched owned-hardware TCO defaults for every selectable GPU profile. New pools pre-fill an amortized $/GPU-hour value derived from current acquisition pricing, four-year depreciation, host/network allocation, facility/ops uplift, and energy at 80% draw, 1.50 PUE, and $0.20/kWh; imported scenarios with an omitted TCO receive the same default while explicit zero remains a suppression override. The bundled H100 starter scenario now uses the researched $1.32/GPU-hour default.
- Added exploratory presentation variants of the demand & economics projection under `/econ/` (gallery + twin money/token flow sankeys, executive dashboard, narrative decision brief, fleet cockpit). Each page renders the visitor's current Config A read-only via a new `econ_variants` blueprint, with semantic fate colors (green served → amber spill → red leak → dark maroon destroyed) so the team can compare displays and pick a clearer presentation for the main planner page.
- Added a same-hardware model-swap recommender (`_marginal_model_swap_recommendations` in calc.py): for each deployed model it simulates one-for-one catalog replacements on the existing GPUs, retunes topology, and ranks swaps by margin gain, avoided cloud spend, and recovered destroyed demand. Shown on the dashboard, brief, and fleet cockpit in place of GPU-expansion recommendations, with before→after and ±delta views of every metric column.
- Added per-domain model quality for general, coding, reasoning, long-context, multilingual, and vision workloads. Built-in use cases carry a domain, sparse official Qwen 3.5/Kimi K2.5/DeepSeek V3/Gemma 4/GLM-5/North Mini Code anchors retain benchmark provenance, and missing anchors preserve the prior global score. Routing, auto-selection, and model-swap shortlisting now use the active demand/value-weighted domain mix; swap tables disclose portfolio fit, mix, and anchor coverage.
- Made the variant pages interactive: the per-use-case table sorts by any column on repeated header clicks (desc → asc → original), and the value bridge toggles between dollar and token waterfalls. Sankeys pin node order per layer (children stacked beside their parents) so flow bands never cross.
- Added true automatic speculative-depth selection: Auto searches calibrated/supported k values at a disclosed concurrency probe, includes spec-off as a candidate, and holds the chosen deployment depth across charts. Manual controls now expose only supported depths.
- Added speculative-decoding modeling: per-deployment native MTP, EAGLE-3, DFlash, and training-free n-gram selection; measured acceptance lengths where available and explicitly labeled priors elsewhere; finite-output cycle accounting; k-position target KV/compute/communication verification; separate MoE drafter resident-memory and active-compute costs; exact attached-checkpoint bytes; draft KV capacity overhead; and spec-aware topology retuning. Cards expose drafter and k controls, modeled speedup/slowdown, memory, and provenance.
- Expanded the built-in use-case catalog from 10 to 19 scenarios, adding document extraction, enterprise search, contact-center QA, translation, contract review, security investigation, AML/KYC casework, synthetic generation, and catalog enrichment with dated source-backed assumptions.

### Changed

- Rebuilt the planner's Demand & economic impact section on the /econ/ variant designs: the old projection panel is replaced by per-config KPI strips and Flow / Dashboard / Brief / Fleet tabs (twin crossing-free sankeys, $/token value bridge, stacked demand outcomes, prose memo, fleet cards, tariff sheet, model-swap table, who-serves-whom sankey). Chart markup lives in shared partials rendered from `econ_payload()`, initialized by `static/econ.js` across HTMX OOB swaps and session syncs; model-swap recommendations lazy-load via `/econ/swaps` to keep mutation responses fast. The standalone /econ/ pages remain as full-page views of the same partials.
- Corrected dense Qwen 3.5 catalog geometry and added native MTP profiles for 0.8B/2B/4B/9B/27B plus the benchmarked 2B Qwen 3.5 27B DFlash drafter with per-depth acceptance calibration. Corrected Gemma 4 31B geometry and modeled its actual four-layer, 939 MB assistant with a conservative, explicitly unmeasured acceptance prior.
- Speculative decode now includes small explicit launch, scheduling, rejection-sync, and approximate draft-collective costs. Cards, reports, legends, and chart tooltips disclose method, effective k, alpha provenance, modeled speedup, and when Auto keeps the baseline.
- Replaced the user-controlled global prefix-hit slider with catalog-owned empirical prefix-token reuse priors per use case. Shared estimates now use a prompt-token-weighted portfolio average, while routing capacity and cloud cached-input pricing use each workload's own prior.
- Replaced the generated single-panel starter configuration with the bundled six-H100 A/B scenario, including its six-model comparison and 19-use-case workload definitions.
- Recalibrated several use-case token formulas and quality gates, made deep research routable, and separated published workload evidence from low/medium-confidence planner defaults in the use-case UI.

### Fixed

- Removed the unsupported Gemma 4 12B MTP option, replaced Gemma 4's unsupported 80% acceptance assumption with a conservative 40% prior, and prevented unsupported manual k values from being silently snapped across UI changes or imported scenarios.
- Made projected cloud invoice spend explicit in the headline and per-use-case rows as money paid to cloud and lost from on-prem, including workloads routed 100% to cloud.
- Corrected pipeline-parallel decode latency so User Pareto no longer reports higher per-user token speed merely from adding concurrent users; concurrency is a continuous batch, not a count of independent pipeline microbatches.
- Formatted billion-scale token counts with a B unit in `fmt_num` (supply capacity showed "31795.8M") and aligned the live-slider JS formatter to the same units and decimals as the server filter.
- Quoted the prefix-reuse panel's effective prefill length on the same workload basis the planner actually probes and routes on — `max(task input, mean input distribution)` — instead of the raw task input knob, and corrected the caption to state that prefill-sensitive capacity and demand estimates use the knob while pure decode and pareto charts do not.
- Replaced the static "no compatible option within the price ceiling" unserved sub-label with demand-aware copy ("no demand routed yet" at zero demand).

## 2026-07-21

### Added

- Added AMD Instinct MI455X as the selectable Helios accelerator profile; retained the MI400 compatibility key for existing saved plans.
- Added an AMD Helios 72-GPU system profile, Tenstorrent Blackhole p100a/p150 cards and Galaxy server, and FuriosaAI RNGD. Added MI440X and VSORA Jotunn 8 as reference-only cards pending sufficient published planner specs.
- Added Poolside Laguna S 2.1, the 118B-A8B MoE flagship released today, using its published Hugging Face configuration (48 layers, hidden 3072, 12 global + 36 sliding-window-512 layers with per-head gating, 1M context) and a conservative quality proxy above Laguna M.1 pending an Artificial Analysis row.
- Added Poolside Laguna XS 2.1, the latest 33B-A3B MoE release, with its 256K served context window and the XS.2 architecture proxy pending a published configuration; XS.2 remains available for existing scenarios.
- Added an optional deployment-level corporate cloud policy (`PLANNER_CLOUD_POLICY`) with model allowlists, negotiated input/cached-input/output price overrides, custom gateway presets, fail-fast validation, UI status, and report provenance.
- Added current July 2026 cloud/API offerings and dated first-party pricing provenance for OpenAI GPT-5.6, Anthropic Claude 5/4.8, Google Gemini 3.x, Mistral, xAI Grok 4.1 Fast, and DeepSeek V4 families.
- Added a preliminary NVIDIA Vera Rubin NVL72 rack profile and a dated preview-assumptions registry/guard for non-Kimi preview hardware and models.
- Added numerical regression coverage for P/D residency, recurrent-state sharding, TP collectives, pipeline-stage embedding work, full-prefix reuse, cloud policy routing, proxy handling, and catalog provenance.

### Changed

- Enforced one resident TP/PP/DP layout per model assignment so independently tuned prefill/decode layouts cannot double-count the same GPUs and VRAM; automatic retuning now co-locates both phases until explicit disaggregated pools are modeled.
- Corrected hybrid linear-attention state sharding, MLA/sliding-window KV readouts, two-per-layer tensor-parallel reductions, uneven pipeline-stage embedding work, finite full-prefix-hit output, and workload-specific CO₂ accounting.
- Updated the AMD MI400 compatibility profile to the named MI450X/MI455X Helios generation and replaced legacy Mistral entries in active corporate presets.
- Pinned the shipped Compose deployment to one worker and reject incompatible `WEB_CONCURRENCY` values; added opt-in one-hop proxy handling without trusting forwarded host headers.
- Added the previously missing corporate-gateway selector and visible model/negotiated-price tags to the demand projection controls.

## 2026-07-18

### Added

- Anchored Kimi K2.5 and DeepSeek V3 to their Artificial Analysis rows: K2.5 (Reasoning) scores 35 on the Intelligence Index with 87M verbosity tokens (artificialanalysis.ai/models/kimi-k2-5); DeepSeek V3 (Dec '24) scores 14 with 3.3M tokens (artificialanalysis.ai/models/deepseek-v3). Both were the last text models routing on the undiscounted quality=0.5 fallback.
- Updated the Kimi K2.5 catalog entry for its published 256k context window and text/image/video input, matching the AA model page.
- Added catalog coverage guards: `text_models_missing_quality_anchors()` / `cloud_models_missing_quality_anchors()` plus an explicit `AA_QUALITY_PLACEHOLDER` registry, with regression tests asserting every servable model carries a quality anchor and no quality/GPU side-table key dangles.
- Added snapshot-store tests covering bounded-retention defaults, explicit-zero unlimited opt-in, preset-def slimming with restore-on-read, and legacy fat-row reads.

### Changed

- Snapshot retention now defaults to 90 days and 250 snapshots per browser tab instead of unlimited; setting either variable to `0` still keeps the full history.
- Snapshot rows no longer embed copies of the ten built-in use-case presets: stored payloads keep only custom or modified definitions and reattach built-ins on read, shrinking per-row size while leaving the admin viewer and legacy fat rows fully readable. Retention deletions now trigger a WAL checkpoint so space is actually reclaimed.
- Split the state module along its fault lines: `state.py` keeps the state model, CRUD, and session registry (2,720 → 1,460 lines); the placement/auto-selection engine moved to `placement.py`, scenario import/export to `scenarios.py`, and the model view-model builder to `viewmodels.py`. Callers were migrated directly with no re-export shims, and the duplicated loaded-state normalization block in `get_state`/`get_compare_state` is now a single shared helper.
- Extracted the calculator's inline script (~1,260 lines) into cacheable `static/app.js`; `templates/base.html` is now an 84-line skeleton with no inline JavaScript. Removed the dead project-chart code and an always-false scroll-behavior feature detect.
- Vendored HTMX 2.0.4 and Apache ECharts 5.5.1 under `static/vendor/`, removing the CDN dependency and its no-SRI supply-chain exposure.
- Unified the use-cases page on the shared `static/app.js` plus a 50-line `static/use_cases.js`, deleting its drifted copy of the calculator's JavaScript.

## 2026-07-16

### Added

- Added Kimi K3 as an API-launch preview proxy with its published 2.8T size, Kimi Delta Attention, native vision, and 1M-token context; unpublished active-parameter and layer details remain explicitly labeled as assumptions.
- Added the open-weight Inkling 975B-A41B model with exact hybrid-attention config, multimodal capabilities, 1M context, and an artifact-backed NVFP4 storage profile.
- Added Inkling-Small 276B-A12B as a preview entry with conservative architecture assumptions pending its promised weight/config release.

### Changed

- Updated Kimi K3 for its public launch with Stable LatentMoE 16-of-896 expert routing, MXFP4-weight/MXFP8-activation QAT, a revised 60B-active capacity proxy, and the official $0.30 cached-input / $3 input / $15 output API pricing. The layer layout and active parameter count remain labeled estimates until the weights and technical report arrive.

## 2026-07-13

### Added

- Added versioned full-scenario JSON export/import, a blank-start workflow, and a visitor-controlled action that deletes stored scenarios and in-memory state.
- Added transactional SQLite snapshot storage with complete A/B payloads, one-time legacy JSON migration, corruption quarantine, paginated admin reads, optional retention controls, and a health endpoint.
- Added regression coverage for planner numerical invariants, context limits, economics, scenario round trips, input validation, state isolation, persistence, migration, retention, and admin hardening.
- Added CI checks for Python 3.10 and 3.12 and a container health check.

### Changed

- Corrected MLA KV-cache sizing, attention FLOPs, full-step decode latency, uneven data-parallel loading, pipeline-stage capacity and bubble estimates, combined context validation, retry economics, and workload-specific projected capacity.
- Hardened mutable settings with explicit allowlists and finite bounds; added request/import limits, scoped locking and state caps, tab limits, mutation/login throttling, secure cookie controls, and fail-closed admin configuration.
- Pinned the production Flask and Gunicorn versions for reproducible deployments.
- Clarified the planner workflow and scenario-storage behavior, gated incomplete economics when TCO is unset, improved recommendation wording, and strengthened keyboard, screen-reader, focus, contrast, target-size, reduced-motion, and mobile-table behavior.
- Kept complete scenario-history retention enabled and unlimited by default; deployments can opt into age or per-tab limits through environment variables.

## 2026-05-28

### Added

- Added open/self-hosted ASR catalog entries for NVIDIA Nemotron/Parakeet, Kyutai STT, Moonshine Streaming, Fun-ASR-Nano, IBM Granite Speech, and Parakeet TDT.
- Added embedding model catalog entries, document-size workload presets, embedding throughput math, and embedding model cards for dense, hybrid, and late-interaction retrieval models.
- Added Laguna M.1 225B-A23B to the Poolside model catalog with planner proxy assumptions and regression coverage.
- Added model picker tabs for LLM, embedding, and ASR catalogs, including add-all actions for grouped model additions.

### Changed

- Updated visible plot modes to focus ASR and embedding analysis on quality-vs-capacity views.
- Used sourced decontaminated BEIR nDCG@10 for embedding quality where available, with hover details for fallback scores.
- Updated the ASR quality plot to use different point shapes for streaming versus non-streaming ASR profiles.
- Updated memory bars, task panels, and model strategy controls to show embedding encoder workloads separately from decode and prefill paths.

## 2026-05-26

### Added

- Added NVIDIA GB300 NVL72 and DGX Station GB300 Blackwell Ultra hardware profiles, including FP4 throughput, TDP, picker cards, and catalog regression coverage.
- Added set-only GPU pool sizing with `min_count` and `count_multiple` constraints so rack/system profiles snap to valid 72-GPU or 8-GPU deployments.
- Added realtime audio-encoder workload metadata for Voxtral-style streaming audio models.

### Changed

- Updated GB200, B200, and B300 catalog entries to distinguish rack-scale and HGX/DGX system-only profiles, including corrected B300 BF16/FP8 roofline and TDP assumptions.
- Updated GPU picker and GPU count controls to show and enforce system/rack pool-size constraints.
- Updated realtime capacity math to include extra causal audio-encoder work for Voxtral Mini Realtime 4B, and corrected its parameter count to 4.37B.
- Reduced HTMX interaction delays for calculator controls and use-case library edits.
- Made add/remove interactions feel immediate by closing pickers and hiding removed cards while requests are in flight.
- Skipped admin snapshot persistence for high-frequency GPU quantity and cost edits, reducing GPU count update latency from roughly 800 ms to single-digit milliseconds locally.
- Expanded model-card GPU count controls to expose full rack/system assignment sizes such as 72 GPUs and support direct numeric entry.

## 2026-05-25

### Added

- Added Docker and Docker Compose deployment support for running the planner on `0.0.0.0:5014` with persistent instance storage.
- Added environment-based host, port, debug, secret-key, and admin-password configuration examples.

### Changed

- Updated setup documentation to use the `HGskrd/gpullmusagemodeler` repository URL and document both Docker and non-Docker launch paths.
- Updated the Flask entrypoint to read host, port, and debug settings from environment variables.

## 2026-05-22

### Added

- Added an NVIDIA A10 GPU catalog entry with planner specs, picker metadata, TDP data, and regression coverage.

## 2026-05-20

### Added

- Added an RTX A2000 mobile GPU catalog entry.
- Added selectable automatic model-selection strategies for best value per GPU, use-case coverage, quality, lean GPU usage, and throughput.
- Added auto-selection tests covering every declared strategy and fallback behavior.

### Changed

- Updated automatic model selection so the selected strategy is persisted in state, included in reports, and passed through the model panel UI.
- Cleaned up planner navigation with shared calculator, user guide, and use-case tabs.
- Reworked GPU controls to support direct count entry, synced picker state, and faster cost updates.
- Simplified calculator and use-case editor cards by removing duplicate range sliders where numeric inputs already provide direct editing.

## 2026-05-15

### Added

- Added automatic model selection that fits deployed models to configured GPU pools, use-case demand, capability gates, SLOs, and quality floors.
- Added model exclusion and re-allow controls for auto-selected model sets.
- Added a bulk project picker action to add one project for every current use-case definition.
- Added a copyable projection report endpoint and UI action for exporting deployment, routing, economics, supply, and expansion diagnostics.
- Added internal user price readouts for deployed models, split into input and output $/1M token prices.
- Added best-next-GPU recommendations that estimate margin gain, cloud spend avoided, destroyed demand recovered, and served-token uplift.
- Added revenue projection tests for coverage metrics, zero-capacity assignments, and smooth latent-demand activation.

### Changed

- Updated routing quality checks to use confidence-adjusted effective quality plus an explicit per-use-case quality floor.
- Updated cloud and internal effective-price calculations so token efficiency scales output tokens rather than fixed prompt tokens.
- Reworked projection headline metrics around owner revenue, active demand, token coverage, value capture, and revenue multiple.
- Changed latent demand unlocks from a hard threshold to a smooth activation curve around the unlock price.
- Improved per-use-case routing diagnostics for SLO, capability, cloud, price-ceiling, and latent-demand outcomes.

## 2026-05-09

### Added

- Added a dedicated `/use-cases` page for viewing, editing, importing, and exporting reusable use-case definitions.
- Added a use-case library partial with controls for definition metadata, scale model, token multiplier, difficulty, SLO, price ceiling, capability gates, token shape, batch eligibility, and latent-demand economics.
- Added JSON import/export for both the reusable use-case library and the selected calculator use-case set.
- Added organization-scale controls so calculator cards can keep the selected use-case kind separate from the organization's current scale.
- Added scale models for linear, quadratic, network/graph, corpus/backfill, and custom formula demand sizing.
- Added richer built-in use-case definitions, including email correction, meeting notes, and inbox archive workloads.
- Added detailed use-case documentation content with examples, assumptions, routing implications, and token-shape visualizations.
- Added model metadata for mixed attention layouts, including hidden size, local attention, linear attention, CCA-style compressed attention, and attention labels.
- Added planner entries and quality proxies for Kimi Linear 48B, ZAYA1-8B, the legacy ZAYA1 74B proxy, and Laguna XS.2.
- Added GPU catalog entries and ordering updates for Blackwell, AMD MI350/MI355/MI400, Intel Arc Pro B50/B60, and Apple M3 Ultra/M4 Pro profiles.

### Changed

- Refactored project state so selected calculator use cases reference reusable definitions while preserving per-organization scale.
- Simplified calculator use-case cards to show the selected kind summary plus a scale control instead of all definition-editing controls.
- Updated project picker entries to use the editable use-case library rather than the static preset list.
- Moved use-case definition editing out of calculator cards and into the dedicated use-case page.
- Updated routing and state normalization so existing saved projects can infer or preserve their use-case kind.
- Updated aggregate demand syncing to account for use-case scale conversions and imported project sets.
- Updated KV-cache and attention-work calculations to model full attention, local attention windows, linear-attention recurrent state, and MLA tensor-parallel support more accurately.
- Updated model cards to surface attention labels and compressed KV details for MLA and CCA-style models.
- Updated Mistral Medium 3.5 labeling, cloud pricing, and quality assumptions.
- Reworked GPU picker cards into simpler single-action rows with planner-profile notes available as tooltips.
