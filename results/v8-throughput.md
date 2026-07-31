# V8: Batch throughput — replicates/s vs N_MC and N, CPU baseline

- **Failure criterion (stated in advance, Paper I §8):** sublinear scaling in N_MC below device saturation
- **Verdict:** **PASS**
- **Generated:** 2026-07-31 10:44 UTC · numpy 2.5.1 · scipy 1.18.0 · CPython 3.12.13 (x86_64)

## Throughput vs N_MC (400 RK4 steps of the entry RHS per replicate)

| N_MC | CPU batch (rep/s) | GPU batch (rep/s) |
|---|---|---|
| 256 | 10,110 | 1,513 |
| 1,024 | 15,279 | 6,083 |
| 4,096 | 17,283 | 24,509 |
| 16,384 | 17,730 | 71,383 |
| 65,536 | 15,248 | 84,538 |

## Scaling acceptance

Fitted log–log slope of throughput vs N_MC below saturation (points [256, 1024, 4096]): **1.00** against the criterion ≥ 0.8 (linear scaling) → **PASS**.

## CPU baseline comparison

Per-replicate Python loop: **224 rep/s** at N_MC = 256 — the decohered execution model. The vectorized CPU batch peaks at **17,730 rep/s** (79× the loop); the CUDA batch peaks at **84,538 rep/s** (377× the loop, 4.8× the CPU batch). The batch never decoheres: every replicate shares the same kernel launches and the same outer time grid.

## Structural block: batched IMEX Newmark, one shared LU across 4096 replicates

| N | reduced dim | replicate-steps / s |
|---|---|---|
| 16 | 13 | 7,979,008 |
| 24 | 21 | 2,738,016 |
| 32 | 29 | 2,318,138 |

## Achieved occupancy — PENDING instrumentation

The occupancy counter requires Nsight Compute profiling, which is not wired into this environment; the throughput-saturation curve above is its externally observable consequence and is what the failure criterion is evaluated against. Warp-divergence measurement (Paper I, Remark 9) likewise awaits profiler integration; the common-outer-grid design that mitigates it is already the only execution mode implemented.
