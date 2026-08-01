# II-V8: Aerothermal correlations — implementation verification leg

- **Failure criterion (stated in advance, Paper II §8):** heating disagreement > 5% on published reference conditions (unevaluated here — see scope); implementation leg: any scaling or continuity property failing its closed form
- **Verdict:** **PASS**
- **Generated:** 2026-08-01 01:54 UTC · numpy 2.5.1 · scipy 1.18.0 · CPython 3.12.13 (x86_64)

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

**Two discrepancies against Paper II §4.2, both recorded rather than reconciled.** First, the paper describes the nose-radius exponent as *a ≈ 1.0*; the source makes it a function of velocity and density, *a = 1.072×10⁶ V^(−1.88) ρ^(−0.325)*, and states explicitly that **a < 1 must always be met**. It evaluates to 0.378 here — the qualitative claim that blunting increases radiative heating survives, but the exponent does not. Second, the source returns **W/cm²**, so an SI framework needs a 10⁴ conversion; omitting it is a four-order-of-magnitude error that still resembles a heat flux. The r_n-dependent conditional clauses of Eq. (2) — a ≤ 0.6 on 1 ≤ r_n ≤ 2, a ≤ 0.5 on 2 < r_n ≤ 3 — were unreadable in the archived scan and were supplied separately from the published text; they are implemented as caps with the band edges exactly as printed (first band closed, second half-open) rather than guessed.

## Fay–Riddell leading constant, back-substituted from the original paper

| source statement | back-substitution | constant | q_s here |
|---|---|---|---|
| Eq. (58)/(62), factor 0.67 | 0.67·Pr^(−0.4) | 0.7684 | 1.9347 MW/m² |
| Eq. (63), factor 0.94 | 0.94·Pr^(+0.6) | 0.7654 | 1.9271 MW/m² |
| footnote to Eq. (63) | as printed | 0.7600 | 1.9135 MW/m² |
| *not in the source* | literature / this framework | 0.7630 | 1.9211 MW/m² |

**Exponents: derived, not assumed.** The archived scan of Eq. (63) is too degraded to read its exponents directly, but they are recoverable. Eq. (45) gives q = (Nu/√Re)·√(ρ_w μ_w (du_e/dx)_s)·(h_s − h_w)/Pr, and Eq. (58)/(62) correlates Nu/√Re = 0.67·(ρ_s μ_s / ρ_w μ_w)^0.4·{1 + (Le^0.52 − 1) h_D/h_s} — a *ratio*, because the parameter 'was found to depend only upon the total variation in ρμ across the boundary layer'. Substituting gives 0.4 from the ratio and a residual 0.1 from the √(ρ_w μ_w), i.e. exactly (ρ_e μ_e)^0.4 (ρ_w μ_w)^0.1. The Lewis exponents are confirmed directly: 0.52 equilibrium (Eq. 60/62), 0.63 frozen with a fully catalytic wall (Eq. 65, corroborated by Fig. 4).

**The constant is a different story.** The same substitution shows every constant descending from one fitted number at two significant figures — the 0.67 of Eq. (58)/(62). The printed 0.94 of Eq. (63) is 0.67/0.71 = 0.9437 rounded, and the footnote's Pr^(−0.6) is the correlation's own Pr^(+0.4) against the Pr^(−1) of Eq. (45). Backing the coefficient out three ways spans 1.1%, or 1.11% in flux. The widely quoted 0.763 occurs **nowhere in the source**, but it lies inside that band and is as defensible as any of them. This is a provenance finding, not an accuracy finding: no reading supports a third significant figure, all four constants are named and selectable, and none is presented as *the* source value.

## Fay–Riddell reference cases and FIAT — PENDING

The radiative half of this task is closed against published data, and the convective **coefficients** are now checked against the original Fay & Riddell paper (previous two sections). What remains is the convective **reference case**: the 5% criterion is stated against published Fay–Riddell reference conditions, and those need transcribed equilibrium-air properties (ρ_e μ_e, ρ_w μ_w, h_D at the reference states). The source presents its numerical results in figures (Figs. 2, 3, 4, 7, 8), not tables, so no reference q values are extractable from it at the precision a 5% criterion requires. The FIAT recession comparison shares the pending status of I-V4. Neither dataset is in this repository, and no synthetic stand-in is presented as either. II-V8 is therefore **partially complete**: implementation verified, radiative reference verified, convective coefficients verified, convective reference case pending tabulated data.
