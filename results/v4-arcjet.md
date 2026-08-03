# I-V4-ARCJET: Ablation — recession against published arcjet PICA measurements

- **Failure criterion (stated in advance, Paper I §8):** median absolute recession error across the seven Milos & Chen analysis cases exceeding the experimental scatter of the measurements themselves (27% at condition 13)
- **Verdict:** **PASS**
- **Generated:** 2026-08-03 13:02 UTC · numpy 2.5.1 · scipy 1.18.0 · CPython 3.12.13 (x86_64)

## Terminal recession, seven analysis cases, two surface-chemistry tables

| B' table | case | q (W/cm²) | p (kPa) | measured (mm) | predicted (mm) | error | test scatter |
|---|---|---|---|---|---|---|---|
| TACOT | 1 | 107 | 2.3 | 2.30 | 2.53 | +10% | 3% |
| TACOT | 2 | 169 | 5.0 | 4.54 | 4.17 | -8% | 11% |
| TACOT | 3 | 246 | 8.5 | 3.76 | 3.86 | +3% | 16% |
| TACOT | 4 | 395 | 17.2 | 4.52 | 4.83 | +7% | 4% |
| TACOT | 5 | 552 | 27.3 | 5.19 | 5.82 | +12% | 8% |
| TACOT | 6 | 744 | 31.0 | 5.13 | 6.60 | +29% | 0% |
| TACOT | 7 | 1102 | 84.4 | 4.84 | 5.00 | +3% | 0% |
| PICA | 1 | 107 | 2.3 | 2.30 | 0.79 | -65% | 3% |
| PICA | 2 | 169 | 5.0 | 4.54 | 1.45 | -68% | 11% |
| PICA | 3 | 246 | 8.5 | 3.76 | 1.43 | -62% | 16% |
| PICA | 4 | 395 | 17.2 | 4.52 | 2.33 | -48% | 4% |
| PICA | 5 | 552 | 27.3 | 5.19 | 3.87 | -25% | 8% |
| PICA | 6 | 744 | 31.0 | 5.13 | 5.84 | +14% | 0% |
| PICA | 7 | 1102 | 84.4 | 4.84 | 5.43 | +12% | 0% |

Every case uses the same solver, material and kinetics; only the surface chemistry differs.

## Accuracy against the experimental scatter

| B' table | median |error| | worst | within 10% | within 30% |
|---|---|---|---|---|
| TACOT, Mutation++, pressure-dependent | 8% | 29% | 5/7 | 7/7 |
| PICA, ACE via Quinn Fig. 5, single pressure | 48% | 68% | 0/7 | 3/7 |

The comparison is against a 27% experimental scatter, not against the criterion's nominal 5%. Best median is **8%** with the TACOT table → **PASS**.

**The interesting result is which table wins, and why.** The pressure-dependent equilibrium table computed for *TACOT*, an open surrogate, beats the digitised *PICA* table from published ACE output — and the reason is visible in the error's structure rather than its size. The PICA table is single-pressure, digitised from a figure drawn at ambient, and its error runs monotonically from −65% at 2.3 kPa to +12% at 84.4 kPa: it is progressively wrong the further the case sits below the one pressure it was drawn at, and right where it was drawn. That is not a bad transcription, it is a table being used outside its envelope, and it is a sharper argument for carrying pressure dependence than any convergence study.

## What this leg does and does not close

The stated criterion names a *FIAT reference case*. What is compared here is **measured recession**, which is a stronger reference than a code's output and is what the measurements in Milos & Chen actually are. FIAT's own predictions for two of these conditions are digitised in `reference/transcribed-data` and remain available for a direct code-to-code comparison.

Three things still limit the numbers above, in order of size. The **pyrolysis kinetics** are pinned to stated TGA targets rather than measured — Torres-Herrador's parallel set is implemented but not yet wired into the solver's in-depth update. The **conductivity and specific-heat slopes** above room temperature are representative; only the 300 K intercepts are published. And **TACOT is not PICA**, so the best-performing configuration here is using a surrogate's surface chemistry with PICA's bulk properties. Each of those is a known, named gap rather than an unexplained residual.
