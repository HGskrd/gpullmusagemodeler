# Tinygrad tinybox configurations

Reviewed **2026-11-11** from the [official configurator](https://tinygrad.org/)
and its [configuration/pricing JavaScript](https://tinygrad.org/assets/tinybox.js?v=red-whale-2).
The Tinygrad tab in **Hardware → Add GPU** includes all three named presets and
all 132 valid custom combinations (104 with GPUs, 28 barebones).

| Preset | Platform | GPUs | Host RAM | Extra storage | PSU capacity | System USD |
| --- | --- | --- | --- | --- | --- | --- |
| red flash | G4 | 6× R9700, 192 GB total | 32 GB DDR4 | None | 1700 W | $22,000 |
| red kimi | G5 | 4× R9700, 128 GB total | 32 GB DDR5 | 4× 4 TB RAID | 1700 W | $30,000 |
| red whale | G5 | 14× R9700, 448 GB total | 32 GB DDR5 | 4 TB RAID | 3400 W | $50,000 |

Prices exclude shipping, tax and import fees. Each includes a 1 TB boot drive.
The site's red flash preset selects its first six-GPU option, which is R9700.

Custom options follow the source's rules:

- G4 starts at $7,000 with 32 GB DDR4; 128 GB adds $3,000.
- G5 starts at $12,000 with 32 GB DDR5; 192 GB adds $8,000.
- The 3400 W tall case adds $1,000 over the standard 1700 W case.
- Extra storage: none, 4 TB RAID (+$2,000), 16 TB RAID (+$6,000),
  or G5-only 4× 4 TB RAID (+$8,000). The latter cannot accompany six or more GPUs.
- GPUs: none, 4× or 6× 9070 XT ($1,000/card), or 4×, 6× or 14× R9700
  ($2,500/card). Six 9070 XTs and fourteen R9700s require the tall case.
- Six R9700s in either case are limited to 210 W/card; fourteen R9700s
  use 210 W/card and half-width PCIe x8 links.
- G4 has dual 1GbE; G5 dual 10GbE. Four-array storage and fourteen GPUs remove
  the OCP expansion option. Storage with six or more GPUs, and four-array storage,
  use USB boot as noted by the vendor.

Quantity on this tab counts **systems**, and adds the corresponding number of
physical GPUs. Exported scenarios preserve the exact configuration. Barebones
prices are inspectable, but cannot be added as inference capacity.

Planner VRAM and bandwidth remain **per GPU**: 16 GB / 640 GB/s for 9070 XT,
32 GB / 640 GB/s for R9700. Host RAM and SSD capacity do not become VRAM;
CPU/SSD offload and storage-bound inference are not simulated. Compute reuses
the existing Radeon catalog, conservatively scaled by the GPU power limit;
this is an uncalibrated estimate, not a tinybox benchmark.

The PCIe collective proxy uses half of theoretical bidirectional link bandwidth:
32 GB/s for G4 x16, 64 GB/s for G5 x16, halved again for fourteen-card x8 builds.
Scale-out conservatively shares one onboard Ethernet port across the system's GPUs;
multi-system performance still requires calibration.
No NVLink or unified multi-GPU memory is implied. Runtime model/precision support
must be checked for tinygrad or ROCm.

TCO apportions complete-system acquisition cost per physical GPU without the
board-only host uplift. Its energy term uses PSU capacity divided by GPU count
as a conservative planning proxy, not measured consumption. Task-energy estimates
continue to use GPU power limits, excluding host overhead. Replace defaults with
measured power and a matching quote for operational decisions.
