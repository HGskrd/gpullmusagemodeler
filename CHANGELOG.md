# Changelog

All notable changes to this project will be documented in this file.

## 2026-07-21

### Added

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
