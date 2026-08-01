# V8: Batch throughput — replicates/s vs N_MC and N, CPU baseline

- **Failure criterion (stated in advance, Paper I §8):** sublinear scaling in N_MC below device saturation
- **Verdict:** **PASS**
- **Generated:** 2026-08-01 01:31 UTC · numpy 2.5.1 · scipy 1.18.0 · CPython 3.12.13 (x86_64)

## Throughput vs N_MC (400 RK4 steps of the entry RHS per replicate)

| N_MC | CPU batch (rep/s) | GPU batch (rep/s) |
|---|---|---|
| 256 | 10,136 | 1,491 |
| 1,024 | 15,187 | 5,920 |
| 4,096 | 17,083 | 23,480 |
| 16,384 | 17,649 | 71,424 |
| 65,536 | 15,374 | 85,029 |

## Scaling acceptance

Fitted log–log slope of throughput vs N_MC below saturation (points [256, 1024, 4096]): **0.99** against the criterion ≥ 0.8 (linear scaling) → **PASS**.

## CPU baseline comparison

Per-replicate Python loop: **229 rep/s** at N_MC = 256 — the decohered execution model. The vectorized CPU batch peaks at **17,649 rep/s** (77× the loop); the CUDA batch peaks at **85,029 rep/s** (372× the loop, 4.8× the CPU batch). The batch never decoheres: every replicate shares the same kernel launches and the same outer time grid.

## Structural block: batched IMEX Newmark, one shared LU across 4096 replicates

| N | reduced dim | replicate-steps / s |
|---|---|---|
| 16 | 13 | 3,652,022 |
| 24 | 21 | 2,620,874 |
| 32 | 29 | 2,358,013 |

## Theoretical occupancy of the batched stage kernel (SM 8.6, 82 SMs)

| threads/block | registers/thread | blocks/SM | warps/SM | occupancy | limiter |
|---|---|---|---|---|---|
| 64 | 27 | 16 | 32 | 0.667 | shared_memory |
| 128 | 27 | 12 | 48 | 1.000 | warps |
| 256 | 27 | 6 | 48 | 1.000 | warps |
| 512 | 27 | 3 | 48 | 1.000 | warps |
| 1024 | 27 | 1 | 32 | 0.667 | warps |

## Achieved occupancy — measured

Nsight Compute reports an achieved occupancy of **0.842** for kernel `rk4_stage`, averaged over 3 launches, against the theoretical bound of 1.000 at the 256-thread block size used. The gap is what the counter exists to expose and the model cannot predict: launch tail, since the grid does not divide evenly across 82 SMs, plus ramp-up and drain at the ends of a short kernel. The kernel is selected by name because a CuPy process also launches its own fill and copy kernels, and averaging over those would report the occupancy of the setup rather than of the workload.
