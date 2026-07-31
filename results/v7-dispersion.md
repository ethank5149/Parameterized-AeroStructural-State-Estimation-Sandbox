# V7: Dispersion statistics — CEP/R95 convergence, bootstrap, normality

- **Failure criterion (stated in advance, Paper I §8):** CEP not converging at the 1/sqrt(2 N_MC) rate
- **Verdict:** **PASS**
- **Generated:** 2026-07-31 10:44 UTC · numpy 2.5.1 · scipy 1.18.0 · CPython 3.12.13 (x86_64)

## Full-batch summary (N_MC = 32,000, generic entry model)

| metric | value | 95% bootstrap CI |
|---|---|---|
| CEP | 264.6 m (linear-approximation) | [262.9, 266.2] |
| R95 (scalar, = a95) | 818.1 m | [811.3, 824.7] |
| σ1, σ2 | 334.2, 115.2 m | — |
| aspect σ2/σ1 | 0.345 | — |
| per-σ RSE bound 1/sqrt(2N) | 0.0040 | — |
| empirical containment of R95 ellipse | 0.9503 | target 0.95 |
| Henze–Zirkler | HZ = 1.011, p = 0.249 | — |

## Reading

The footprint is downrange-elongated (aspect 0.345) but inside the CEP validity band, so the linear approximation applies. The R95 ellipse empirically contains 95.0% of impacts (consistent with the 95% design). The Henze–Zirkler p-value of 0.249 does not reject bivariate normality for this batch.

## CEP sampling error vs N_MC (disjoint sub-batches of one 32k draw)

| N_MC | sub-batches | mean CEP (m) | empirical RSE | 1/sqrt(2N) | ratio |
|---|---|---|---|---|---|
| 250 | 128 | 264.2 | 0.0371 | 0.0447 | 0.83 |
| 500 | 64 | 264.4 | 0.0274 | 0.0316 | 0.87 |
| 1000 | 32 | 264.5 | 0.0160 | 0.0224 | 0.72 |
| 2000 | 16 | 264.6 | 0.0091 | 0.0158 | 0.57 |
| 4000 | 8 | 264.6 | 0.0073 | 0.0112 | 0.66 |

## Convergence acceptance

Fitted log–log slope of the empirical CEP relative standard error versus N_MC: **-0.627** (criterion band [-0.75, -0.25], the −1/2 of §6.3), with every empirical RSE within a factor 2 of the 1/sqrt(2N) prediction → **PASS**.

## Elongated-footprint fallback (Remark 10)

Suppressing crossrange dispersions produces aspect ratio 0.023 < 0.25, and the summary switches to the direct order statistic (CEP = 215.7 m, method 'order-statistic') instead of reporting a linear approximation outside its validity band → **PASS**. The linear formula would have claimed 193.6 m.
