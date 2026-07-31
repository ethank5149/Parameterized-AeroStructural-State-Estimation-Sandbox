# V3: Time integration — explicit stability limit and strategy comparison

- **Failure criterion (stated in advance, Paper I §8):** explicit Δt not scaling as N⁻⁴
- **Verdict:** **PASS**
- **Generated:** 2026-07-31 11:15 UTC · numpy 2.5.1 · scipy 1.18.0 · CPython 3.12.13 (x86_64)

## Explicit RK45: achieved step vs the Prop. 2 bound

| N | ω_max (rad/s) | Δt bound (C=3.0/ω_max) | achieved mean Δt | achieved Δt·ω_max | wall (ms) |
|---|---|---|---|---|---|
| 12 | 7.735e+02 | 3.879e-03 | 9.259e-04 | 0.72 | 1.5 |
| 16 | 2.392e+03 | 1.254e-03 | 9.259e-04 | 2.21 | 1.5 |
| 20 | 5.686e+03 | 5.276e-04 | 3.049e-04 | 1.73 | 4.6 |
| 24 | 1.154e+04 | 2.600e-04 | 1.333e-04 | 1.54 | 10.7 |
| 28 | 2.102e+04 | 1.427e-04 | 6.623e-05 | 1.39 | 21.8 |
| 32 | 3.541e+04 | 8.472e-05 | 3.655e-05 | 1.29 | 39.1 |

## Acceptance

Fitted log–log slope of achieved Δt versus N: **-3.57** (criterion: within [-5.0, -3.0], i.e. the N⁻⁴ scaling of Prop. 2) → **PASS**. The achieved Δt·ω_max column shows the integrator is pinned at an O(1) multiple of the stability bound, i.e. steps are stability-limited, not accuracy-limited — the pathology Remark 4 describes.

## Strategy comparison at N = 32 (free vibration of elastic mode 4, T = 0.05 s)

| strategy | Δt | steps | max rel error vs exact modal | wall (ms) |
|---|---|---|---|---|
| explicit RK45 (adaptive) | 3.655e-05 | 1368 | 1.52e-06 | 37.8 |
| IMEX Newmark (Δt = 0.0001) | 1.000e-04 | 500 | 1.79e-04 | 4.8 |
| modal truncation (n_m = 10, exact) | 1.000e-04 | 500 | 3.18e-15 | 5.1 |

## Reading

The IMEX step is 1.2× the explicit stability bound at the same N with no loss of stability (the test suite exercises the same scheme at 10⁴× the bound); its error is the O(Δt²) Newmark dispersion at the excited frequency. Modal truncation with the excited mode retained is exact to the ZOH/rounding floor. Retained-mode translation participation at n_m = 10: 1.000000. Which mitigation is preferable is configuration-dependent; both preserve the fixed-dimension batching argument, the IMEX branch because its factorization is shared across replicates, the modal branch because the basis is.
