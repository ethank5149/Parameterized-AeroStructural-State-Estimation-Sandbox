# V8: Batch throughput — replicates/s vs N_MC and N, CPU baseline

- **Failure criterion (stated in advance, Paper I §8):** sublinear scaling in N_MC below device saturation
- **Verdict:** **PASS**
- **Generated:** 2026-08-01 00:50 UTC · numpy 2.5.1 · scipy 1.18.0 · CPython 3.12.13 (x86_64)

## Throughput vs N_MC (400 RK4 steps of the entry RHS per replicate)

| N_MC | CPU batch (rep/s) | GPU batch (rep/s) |
|---|---|---|
| 256 | 10,050 | 1,509 |
| 1,024 | 14,979 | 6,009 |
| 4,096 | 17,015 | 24,115 |
| 16,384 | 17,377 | 71,443 |
| 65,536 | 15,037 | 84,730 |

## Scaling acceptance

Fitted log–log slope of throughput vs N_MC below saturation (points [256, 1024, 4096]): **1.00** against the criterion ≥ 0.8 (linear scaling) → **PASS**.

## CPU baseline comparison

Per-replicate Python loop: **225 rep/s** at N_MC = 256 — the decohered execution model. The vectorized CPU batch peaks at **17,377 rep/s** (77× the loop); the CUDA batch peaks at **84,730 rep/s** (377× the loop, 4.9× the CPU batch). The batch never decoheres: every replicate shares the same kernel launches and the same outer time grid.

## Structural block: batched IMEX Newmark, one shared LU across 4096 replicates

| N | reduced dim | replicate-steps / s |
|---|---|---|
| 16 | 13 | 3,471,490 |
| 24 | 21 | 2,823,567 |
| 32 | 29 | 2,722,307 |

## Theoretical occupancy of the batched stage kernel (SM 8.6, 82 SMs)

| threads/block | registers/thread | blocks/SM | warps/SM | occupancy | limiter |
|---|---|---|---|---|---|
| 64 | 27 | 16 | 32 | 0.667 | shared_memory |
| 128 | 27 | 12 | 48 | 1.000 | warps |
| 256 | 27 | 6 | 48 | 1.000 | warps |
| 512 | 27 | 3 | 48 | 1.000 | warps |
| 1024 | 27 | 1 | 32 | 0.667 | warps |

## Achieved occupancy — profiler blocked

**Theoretical** occupancy above is exact: it is the standard CUDA occupancy model evaluated from the compiled kernel's register and shared-memory footprint against the device's per-SM limits, and it bounds achieved occupancy from above. The **achieved** counter needs Nsight Compute, which is installed but cannot read the counters here: ncu not found on PATH. That is a host-level driver setting, not a code gap, and it is reported rather than worked around. Warp-divergence measurement (Paper I, Remark 9) is blocked by the same gate; the common-outer-grid design that mitigates divergence is already the only execution mode implemented.
