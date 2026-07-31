# V4: Ablation — manufactured solutions (MMS leg); FIAT case pending

- **Failure criterion (stated in advance, Paper I §8):** recession disagreement > 5% on the FIAT reference case (unevaluated here — see scope); MMS leg: loss of spectral convergence on Eqs. (3.17)–(3.18)
- **Verdict:** **PASS**
- **Generated:** 2026-07-31 10:43 UTC · numpy 2.5.1 · scipy 1.18.0 · CPython 3.12.13 (x86_64)

## MMS convergence, coupled T/ρ/ṁ system to t = 2.0 s (normalized max-norm errors at t_f)

| N_T | T error / ΔT | ρ error / Δρ | s error (m) |
|---|---|---|---|
| 6 | 1.995e-06 | 7.892e-06 | 2.7e-19 |
| 8 | 1.808e-08 | 7.895e-08 | 2.2e-19 |
| 10 | 8.665e-11 | 3.776e-10 | 1.1e-19 |
| 12 | 1.170e-12 | 5.302e-12 | 2.7e-19 |
| 16 | 2.653e-15 | 1.012e-12 | 5.4e-20 |
| 20 | 3.790e-15 | 2.143e-12 | 1.6e-19 |

## MMS acceptance

Error contracts by **5.3e+08** from N = 6 to N = 20, reaching **3.8e-15** against the 1e-06 criterion (floor set by the 1e-11 time tolerance) → **PASS**. The decay is exponential in N_T until the time-integration floor — the spectral signature that the collocated operators discretize Eqs. (3.17)–(3.18) consistently, grid-velocity advection and pyrolysis sources included. The manufactured fields keep the degree-of-char clip and the kinetics extent clamp strictly inactive, so the sources are exact.

## Supporting closed-form checks

| check | measured | criterion |
|---|---|---|
| first-order kinetics vs exact exponential | 1.6e-13 | < 1e-9 |
| gas-flux operator, polynomial source | 2.6e-18 | < 1e-14 (exact) |
| blowing φ at B' = 1e-17 (naive form returns 0) | 1.000000000000000 | = 1 ± 1e-12 |

## Surface energy balance

Brent solve of Eq. (3.19) with ablating mass fluxes returns T_w = 1995.2 K with residual 1.16e-09 W/m² (**PASS**). The blowing correction enters through the log1p form, so the non-ablating limit is reached without cancellation.

## FIAT reference comparison — PENDING

The stated failure criterion (recession within 5% of a FIAT reference case) requires the external FIAT code or its published reference-case data, neither of which is available in this repository. The comparison harness accepts any tabulated (t, s, T(y)) reference once one is supplied; until then V4 is **partially complete** and is *not* counted as a finished verification task.
