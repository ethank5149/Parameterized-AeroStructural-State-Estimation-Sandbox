# V8: Batch throughput — replicates/s vs N_MC and N, CPU baseline

- **Failure criterion (stated in advance, Paper I §8):** sublinear scaling in N_MC below device saturation
- **Verdict:** **PASS**
- **Generated:** 2026-07-31 20:00 UTC · numpy 2.5.1 · scipy 1.18.0 · CPython 3.12.13 (x86_64)

## Throughput vs N_MC (400 RK4 steps of the entry RHS per replicate)

| N_MC | CPU batch (rep/s) | GPU batch (rep/s) |
|---|---|---|
| 256 | 9,736 | 1,479 |
| 1,024 | 14,519 | 5,864 |
| 4,096 | 16,754 | 23,957 |
| 16,384 | 17,468 | 70,830 |
| 65,536 | 14,873 | 84,161 |

## Scaling acceptance

Fitted log–log slope of throughput vs N_MC below saturation (points [256, 1024, 4096]): **1.00** against the criterion ≥ 0.8 (linear scaling) → **PASS**.

## CPU baseline comparison

Per-replicate Python loop: **224 rep/s** at N_MC = 256 — the decohered execution model. The vectorized CPU batch peaks at **17,468 rep/s** (78× the loop); the CUDA batch peaks at **84,161 rep/s** (376× the loop, 4.8× the CPU batch). The batch never decoheres: every replicate shares the same kernel launches and the same outer time grid.

## Structural block: batched IMEX Newmark, one shared LU across 4096 replicates

| N | reduced dim | replicate-steps / s |
|---|---|---|
| 16 | 13 | 5,381,163 |
| 24 | 21 | 2,699,674 |
| 32 | 29 | 2,415,255 |

## Achieved occupancy — PENDING instrumentation

The occupancy counter requires Nsight Compute profiling, which is not wired into this environment; the throughput-saturation curve above is its externally observable consequence and is what the failure criterion is evaluated against. Warp-divergence measurement (Paper I, Remark 9) likewise awaits profiler integration; the common-outer-grid design that mitigates it is already the only execution mode implemented.
