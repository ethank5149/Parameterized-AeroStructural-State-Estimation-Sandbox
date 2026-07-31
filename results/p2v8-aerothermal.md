# II-V8: Aerothermal correlations — implementation verification leg

- **Failure criterion (stated in advance, Paper II §8):** heating disagreement > 5% on published reference conditions (unevaluated here — see scope); implementation leg: any scaling or continuity property failing its closed form
- **Verdict:** **PASS**
- **Generated:** 2026-07-31 21:03 UTC · numpy 2.5.1 · scipy 1.18.0 · CPython 3.12.13 (x86_64)

## Closed-form structure checks

| check | measured | expected | pass |
|---|---|---|---|
| q_conv ratio for R_eff 0.25 vs 1.0 (expect 4^{1/2} = 2) | 2 | 2 | yes |
| enthalpy-potential linearity ratio | 0.5 | 0.5 | yes |
| frozen/catalytic vs equilibrium bracket shift (expect ~1%) | 0.0107165 | 0.011 | yes |
| Lees jump across x = R_eff (expect 0) | 5e-10 | 0 | yes |
| recession at radiative-equilibrium wall (expect 0) | 0 | 0 | yes |

## Opposite-sign radius trade (Paper II §4.2)

With convective heating falling as R_eff^(-1/2) and radiative rising as R_eff^(+1), the total exhibits an interior optimum at R_eff ≈ **0.63 m** on the demonstration corridor → **PASS**. A framework modeling only convection would drive R_eff to the right edge of this sweep — the over-blunting bias the paper warns of. (Radiative component evaluated with a clearly-labeled synthetic velocity-function surrogate; the published Tauber–Sutton table is a required input with provenance, and the implementation refuses to extrapolate it.)

## Published reference cases and FIAT — PENDING

The 5% criterion is stated against published Fay–Riddell reference conditions, which require transcribed equilibrium-air properties (ρ_e μ_e, ρ_w μ_w, h_D at the reference states); the FIAT recession comparison shares the pending status of I-V4. Neither dataset is in this repository, and no synthetic stand-in is presented as either. II-V8 is therefore **partially complete**: implementation verified, reference comparison pending data.
