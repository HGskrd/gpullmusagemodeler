# Changelog

All notable changes to this project will be documented in this file.

## 2026-05-28

### Added

- Added open/self-hosted ASR catalog entries for NVIDIA Nemotron/Parakeet, Kyutai STT, Moonshine Streaming, Fun-ASR-Nano, IBM Granite Speech, and Parakeet TDT.
- Added embedding model catalog entries, document-size workload presets, embedding throughput math, and embedding model cards for dense, hybrid, and late-interaction retrieval models.
- Added Laguna M.1 225B-A23B to the Poolside model catalog with planner proxy assumptions and regression coverage.
- Added model picker tabs for LLM, embedding, and ASR catalogs, including add-all actions for grouped model additions.

### Changed

- Updated visible plot modes to focus ASR and embedding analysis on quality-vs-capacity views.
- Added sourced decontaminated BEIR nDCG@10 as embedding quality hover detail where available.
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
