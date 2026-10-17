# Owned-hardware catalog and chooser

Reviewed October 17, 2026. This is a dated catalog, not a live stock or quote feed.

The chooser offers released cards and complete systems sold through retail,
hardware vendors, or resellers. No cloud-only TPU, Inferentia or Trainium entries
are offered. Unverified purchase configurations, reference-only products and the
unreleased 512 GB M5 Ultra configuration are excluded. Compatibility keys remain
in `GPUS`, and imported plans retain their device counts and explicit costs.
`data/hardware_picker.py` records the exclusions and purchase-channel caveats.

The anchored dropdown uses vendor tabs, family rows and inline variant buttons.
Search matches names, memory, architecture and vendor across tabs, switching to
an available match when needed. Older purchasable hardware remains searchable.
Quantity presets appear only above the rows and are remembered per vendor:
eight by default, one for Apple. Clicking a variant adds that quantity once.

## Quantity and memory contracts

Counts retain the existing accelerator-count API and saved-scenario semantics.
System-only profiles normalize to their minimum and multiple (8 for B200/B300,
72 for NVL72, 32 for Galaxy). Hovering a variant shows the actual added count.
Existing pool quantity controls are preserved.

Memory and bandwidth are displayed per device. Aggregate device memory is not
presented as a single allocation. NVLink bridge/domain assumptions remain
separate from pool size. Ryzen AI Max+ 395 models a **96 GB GPU allocation budget
inside a 128 GB system**, before planner runtime reservations. Unified-memory
systems also need room for OS and host allocations; this is not a complete host
memory simulator.

## Sources and numerical assumptions

- [A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf):
  312 dense BF16 TFLOPS for all four included A100 variants. PCIe 80 GB bandwidth
  is 1.935 TB/s; SXM 80 GB is 2.039 TB/s. The PCIe profiles assume PCIe-only
  communication; optional NVLink bridges are not inferred from memory capacity.
- [H100 architecture whitepaper](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c)
  and [H100 specifications](https://www.nvidia.com/en-us/data-center/h100/):
  PCIe 80 GB uses 756 dense BF16 / 1,513 FP8 TFLOPS and 2 TB/s.
  NVL uses 94 GB **per GPU**, 3.9 TB/s, 835.5 dense BF16 / 1,670.5 FP8 TFLOPS
  (half the rounded sparse table) and a two-GPU NVLink domain.
- [H200 specifications](https://www.nvidia.com/en-us/data-center/h200/):
  NVL retains 141 GB / 4.8 TB/s per GPU, uses dense 835.5 BF16 / 1,670.5 FP8
  TFLOPS, 600 W, and a modeled four-GPU bridged NVLink domain.
- [RTX 3060](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3060-3060ti/),
  [RTX 4060 Ti](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4060-4060ti/),
  [RTX 5060 Ti](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/):
  explicit 12 / 16 / 16 GB configurations. Memory bandwidth is 360 / 288 / 448
  GB/s and power is 170 / 165 / 180 W. The x8 PCIe link on the Ti cards is
  reflected in the host-link assumption; none assumes NVLink.
- [GeForce Blackwell architecture whitepaper, Appendix A](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf):
  use **dense FP32-accumulation** tensor rates. RTX 5090 is 209.5 BF16 / 419 FP8 /
  1,676 FP4 TFLOPS. RTX 4090 is 165.15 BF16 / 330.3 FP8; RTX 3090 uses 71.15
  BF16 and the same-rate FP8 fallback. The new lower-end profiles conservatively
  apply these architecture ratios: 3060 25.5 BF16; 4060 Ti 44.125 BF16 / 88.25
  FP8; 5060 Ti's 759 sparse FP4 AI TOPS become 379.5 dense FP4, 94.875 FP8 and
  47.4375 BF16 TFLOPS. Smaller-SKU ratios are proxies, not benchmarks.
- [RTX PRO Blackwell whitepaper, Appendix A](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf):
  workstation RTX PRO 6000 uses dense 503.8 BF16 / 1,007.6 FP8 / 2,015.2 FP4
  TFLOPS. The existing Server Edition and PRO 5000 estimates remove the sparse
  factor and retain their explicitly derived, rounded-headline proxies.
- [AMD Ryzen AI Halo specifications](https://www.amd.com/en/products/processors/desktops/ryzen/ryzen-ai-halo/ryzen-ai-max-plus-395.html),
  [AMD GPU allocation explanation](https://www.amd.com/en/blogs/2025/amd-ryzen-ai-max-395-processor-breakthrough-ai-.html),
  [GMKtec owned-system channel](https://www.gmktec.com/products/amd-ryzen%E2%84%A2-ai-max-395-evo-x2-ai-mini-pc):
  128 GB system memory, up to 96 GB GPU allocation, 256 GB/s peak bandwidth.
  The 200 GB/s sustained bandwidth and 29.7 TFLOPS BF16/FP8 rooflines are
  conservative, uncalibrated proxies; there is no claim of native FP8 execution.
  The 120 W power budget is APU power, not measured whole-PC wall power.
- [Apple M5 technical specifications](https://support.apple.com/en-sg/128107),
  [Apple availability announcement](https://www.apple.com/newsroom/2026/08/apple-introduces-new-mac-studio-with-m5-max-and-m5-ultra/),
  [previous Studio configurations](https://support.apple.com/en-us/122211),
  [Mac mini configurations](https://support.apple.com/en-us/121555): memory-tier
  variants preserve their parent's GPU bin, peak/sustained bandwidth and compute
  proxies. M5 Ultra 256 GB uses the 80-core bin. The 512 GB M5 Ultra option is
  excluded until its announced late-October availability can be reviewed.
- [AMD MI250X module boundaries](https://instinct.docs.amd.com/projects/system-acceptance/en/latest/gpus/mi250.html):
  the visible label now agrees with the planner's 128 GB module, comprising two
  separately addressed 64 GB GCDs. This remains a module-level abstraction.
- [Intel Gaudi 3 specifications](https://cdrdv2-public.intel.com/817488/Gaudi%203%20PCIe%20Product%20Brief_RB_1_V6.pdf):
  show 128 GB per accelerator, not an 8.2 TB rack total.

## Price and runtime confidence

New configurations without a configuration-matched acquisition quote have no
prefilled TCO. In particular, the M5 maximum-memory profiles no longer reuse base
launch prices. Internally zero retains the application's established “exclude
cost” convention; the chooser and pool explicitly say **quote required**.
Do not compare economics until those costs are entered. Existing catalog
estimates retain their July 21 assumptions, documented in the catalog source metadata.

Runtime caveats are separate from purchase channel and specification evidence:
CUDA, ROCm, Metal/MLX, SynapseAI, TT-Metal and Furiosa SDK support are not
interchangeable. A capacity estimate does not certify model/kernel compatibility.
