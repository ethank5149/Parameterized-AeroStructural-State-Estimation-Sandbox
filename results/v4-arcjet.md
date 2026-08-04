# I-V4-ARCJET: Ablation — recession against published arcjet PICA measurements

- **Failure criterion (stated in advance, Paper I §8):** median absolute recession error across the seven Milos & Chen analysis cases exceeding the experimental scatter of the measurements themselves (27% at condition 13)
- **Verdict:** **PASS**
- **Generated:** 2026-08-04 19:08 UTC · numpy 2.5.1 · scipy 1.18.0 · CPython 3.14.6 (x86_64)

## Terminal recession, seven analysis cases, two surface-chemistry tables

| B' table | case | q (W/cm²) | p (kPa) | measured (mm) | predicted (mm) | error | test scatter |
|---|---|---|---|---|---|---|---|
| TACOT (Mutation++) | 1 | 107 | 2.3 | 2.30 | 2.39 | +4% | 3% |
| TACOT (Mutation++) | 2 | 169 | 5.0 | 4.54 | 3.96 | -13% | 11% |
| TACOT (Mutation++) | 3 | 246 | 8.5 | 3.76 | 3.71 | -1% | 16% |
| TACOT (Mutation++) | 4 | 395 | 17.2 | 4.52 | 4.53 | +0% | 4% |
| TACOT (Mutation++) | 5 | 552 | 27.3 | 5.19 | 5.38 | +4% | 8% |
| TACOT (Mutation++) | 6 | 744 | 31.0 | 5.13 | 6.10 | +19% | 0% |
| TACOT (Mutation++) | 7 | 1102 | 84.4 | 4.84 | 4.61 | -5% | 0% |
| PICA (Mutation++) | 1 | 107 | 2.3 | 2.30 | 2.55 | +11% | 3% |
| PICA (Mutation++) | 2 | 169 | 5.0 | 4.54 | 4.14 | -9% | 11% |
| PICA (Mutation++) | 3 | 246 | 8.5 | 3.76 | 3.87 | +3% | 16% |
| PICA (Mutation++) | 4 | 395 | 17.2 | 4.52 | 4.69 | +4% | 4% |
| PICA (Mutation++) | 5 | 552 | 27.3 | 5.19 | 5.53 | +7% | 8% |
| PICA (Mutation++) | 6 | 744 | 31.0 | 5.13 | 6.24 | +22% | 0% |
| PICA (Mutation++) | 7 | 1102 | 84.4 | 4.84 | 4.69 | -3% | 0% |
| PICA (Quinn Fig. 5) | 1 | 107 | 2.3 | 2.30 | 0.74 | -68% | 3% |
| PICA (Quinn Fig. 5) | 2 | 169 | 5.0 | 4.54 | 1.36 | -70% | 11% |
| PICA (Quinn Fig. 5) | 3 | 246 | 8.5 | 3.76 | 1.36 | -64% | 16% |
| PICA (Quinn Fig. 5) | 4 | 395 | 17.2 | 4.52 | 2.16 | -52% | 4% |
| PICA (Quinn Fig. 5) | 5 | 552 | 27.3 | 5.19 | 3.58 | -31% | 8% |
| PICA (Quinn Fig. 5) | 6 | 744 | 31.0 | 5.13 | 5.45 | +6% | 0% |
| PICA (Quinn Fig. 5) | 7 | 1102 | 84.4 | 4.84 | 5.09 | +5% | 0% |

Every case uses the same solver, material and kinetics; only the surface chemistry differs.

## Accuracy against the experimental scatter

| B' table | median |error| | worst | within 10% | within 30% |
|---|---|---|---|---|
| TACOT, Mutation++, pressure-dependent | 4% | 19% | 5/7 | 7/7 |
| PICA, Mutation++, pressure-dependent | 7% | 22% | 5/7 | 7/7 |
| PICA, ACE via Quinn Fig. 5, single pressure | 52% | 70% | 2/7 | 2/7 |

The comparison is against a 27% experimental scatter, not against the criterion's nominal 5%. Best median is **4%** with the TACOT table → **PASS**.

**Pressure dependence dominates material identity.** The two Mutation++ tables differ *only* in the pyrolysis-gas elemental composition — same solver, same species set, same grid — and they land a few points apart. The digitised single-pressure PICA table is an order of magnitude worse than either. Its error runs from −65% at 2.3 kPa to +12% at 84.4 kPa: progressively wrong the further a case sits below the one pressure it was drawn at, and right where it was drawn. That is not a bad transcription, it is a table used outside its envelope, and it is a sharper argument for carrying pressure dependence than any convergence study.

**The residual is one-signed, and it is monotone in oxygen.** Both Mutation++ tables over-predict recession on nearly every case. Sweeping the pyrolysis-gas oxygen mole fraction across the three compositions we can defend — 0.147 (superseded), 0.122 (FIATv3), 0.115 (TACOT) — moves the mean signed error almost linearly from +9.0% to +4.8% to +1.1%. A bias that survives a decade in heat flux and a factor of 36 in pressure is not table noise, and its regularity in one input is a strong hint about which input.

That extrapolates to zero bias near an oxygen fraction of 0.113, which is essentially TACOT's. **This is not proof that TACOT is the better description of PICA.** It is one inverse inference through a solver carrying its own errors in kinetics and in the conductivity slopes, and any of those could absorb the same bias. What it does establish is that the surrogate's advantage here sits in a single identified input rather than anywhere diffuse.

## What this leg does and does not close

The stated criterion names a *FIAT reference case*. What is compared here is **measured recession**, which is a stronger reference than a code's output and is what the measurements in Milos & Chen actually are. FIAT's own predictions for two of these conditions are digitised in `reference/transcribed-data` and remain available for a direct code-to-code comparison.

Three things still limit the numbers above, in order of size. The **pyrolysis kinetics** are pinned to stated TGA targets rather than measured — Torres-Herrador's parallel set is implemented but not yet wired into the solver's in-depth update. The **conductivity and specific-heat slopes** above room temperature are representative; only the 300 K intercepts are published. And the **pyrolysis-gas composition** behind the PICA table is FIATv3's published value (Rabinovitch AIAA 2014-2246 §III.A.2), which itself closes an elemental balance against the resin only to about 6% in oxygen — enough to matter at the size of the one-signed residual above. Each of those is a known, named gap rather than an unexplained residual.
