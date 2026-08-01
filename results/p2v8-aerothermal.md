# II-V8: Aerothermal correlations — implementation verification leg

- **Failure criterion (stated in advance, Paper II §8):** heating disagreement > 5% on published reference conditions (unevaluated here — see scope); implementation leg: any scaling or continuity property failing its closed form
- **Verdict:** **PASS**
- **Generated:** 2026-08-01 00:51 UTC · numpy 2.5.1 · scipy 1.18.0 · CPython 3.12.13 (x86_64)

## Closed-form structure checks

| check | measured | expected | pass |
|---|---|---|---|
| q_conv ratio for R_eff 0.25 vs 1.0 (expect 4^{1/2} = 2) | 2 | 2 | yes |
| enthalpy-potential linearity ratio | 0.5 | 0.5 | yes |
| frozen/catalytic vs equilibrium bracket shift (expect ~1%) | 0.0107165 | 0.011 | yes |
| Lees jump across x = R_eff (expect 0) | 5e-10 | 0 | yes |
| recession at radiative-equilibrium wall (expect 0) | 0 | 0 | yes |

## Opposite-sign radius trade (Paper II §4.2) — on published data

With convective heating falling as R_eff^(−1/2) and the published Tauber–Sutton correlation rising as R_eff^(+0.321), the total heating has an **interior optimum at R_eff ≈ 0.94 m** at V = 12000 m/s, ρ = 0.0003 kg/m³ — inside the source's own 0.3–3 m nose-radius envelope → **PASS**. At this condition radiation (8.40 MW/m²) exceeds convection (5.37 MW/m²), which is the regime Tauber and Sutton wrote the correlation for. A framework modelling only convection would drive R_eff to the top of this range — the over-blunting bias the paper warns of, now demonstrated against published data rather than a surrogate.

## Published velocity function f_E(V), Table 1 (spot check of the transcription)

| V (m/s) | published f_E | interpolant at the node |
|---|---|---|
| 9000 | 1.5 | 1.5 |
| 9250 | 4.3 | 4.3 |
| 9500 | 9.7 | 9.7 |
| 9750 | 19.5 | 19.5 |
| 15000 | 1550 | 1550 |
| 15500 | 1780 | 1780 |
| 16000 | 2040 | 2040 |

## Published correlation at V = 11000 m/s, ρ = 0.0003 kg/m³, r_n = 1 m

| quantity | value | note |
|---|---|---|
| nose-radius exponent a | 0.3778 | source requires a < 1; Paper II §4.2 describes it as ≈ 1.0 |
| radiative flux (Eqs. 1–2) | 3.6014 MW/m² | converted from the source's W/cm² |
| convective flux (Sutton–Graves) | 4.0148 MW/m² | for scale only |
| transcription exact at every node | yes | — |
| validity envelope enforced | yes | 10–16 km/s, 6.66e-5–6.31e-4 kg/m³, r_n 0.3–3 m |

## Published radiative data — now verified

The velocity function is transcribed from Tauber & Sutton, J. Spacecraft and Rockets 28(1), 1991, pp. 40-42, Tables 1 and Eqs. (1)-(4), DOI 10.2514/3.26206; transcribed from the archived PDF in reference/ and cross-checked against the layout extraction. Every tabulated node is reproduced exactly by the interpolant (confirmed), linear interpolation being what the source prescribes, and the stated validity envelope is enforced rather than silently extrapolated → **PASS**. At the sample condition radiative and convective heating are comparable (0.90×), which is the regime the paper was written for.

**Two discrepancies against Paper II §4.2, both recorded rather than reconciled.** First, the paper describes the nose-radius exponent as *a ≈ 1.0*; the source makes it a function of velocity and density, *a = 1.072×10⁶ V^(−1.88) ρ^(−0.325)*, and states explicitly that **a < 1 must always be met**. It evaluates to 0.378 here — the qualitative claim that blunting increases radiative heating survives, but the exponent does not. Second, the source returns **W/cm²**, so an SI framework needs a 10⁴ conversion; omitting it is a four-order-of-magnitude error that still resembles a heat flux. The OCR of the r_n-dependent conditional clauses in Eq. (2) is degraded in the archived scan, so the implementation covers the principal branch and enforces the a < 1 requirement rather than guessing the clauses.

## Fay–Riddell reference cases and FIAT — PENDING

The radiative half of this task is now closed against published data (previous section). What remains is the **convective** half: the 5% criterion is stated against published Fay–Riddell reference conditions, which need transcribed equilibrium-air properties (ρ_e μ_e, ρ_w μ_w, h_D at the reference states). The FIAT recession comparison shares the pending status of I-V4. Neither dataset is in this repository, and no synthetic stand-in is presented as either. II-V8 is therefore **partially complete**: implementation verified, radiative reference verified, convective reference pending data.
