# V6: Guidance — stable vs textbook time-to-go as A_c → 0

- **Failure criterion (stated in advance, Paper I §8):** the conjugate form (Eq. 4.16) losing more than 1 significant digit where the textbook form (Eq. 4.15) loses many
- **Verdict:** **PASS**
- **Generated:** 2026-07-31 10:44 UTC · numpy 2.5.1 · scipy 1.18.0 · CPython 3.12.13 (x86_64)

## Operating point

R_LOS = 10000 m, V̂_c = 1000 m/s, Â_c swept over ±10^{2} … ±10^{-14} m/s². Reference: 50-digit decimal evaluation from the exact binary inputs.

## Significant digits retained (float64, floor ≈ 15.7)

| Â_c (m/s²) | stable Eq. 4.16 | naive Eq. 4.15 |
|---|---|---|
| 1e+02 | 15.7 | 15.7 |
| 1e+01 | 15.7 | 15.1 |
| 1e+00 | 15.7 | 14.9 |
| 1e-01 | 15.7 | 13.3 |
| 1e-02 | 15.7 | 12.7 |
| 1e-03 | 15.7 | 11.7 |
| 1e-04 | 15.7 | 10.7 |
| 1e-05 | 15.7 | 9.6 |
| 1e-06 | 15.7 | 8.6 |
| 1e-07 | 15.7 | 8.7 |
| 1e-08 | 15.7 | 6.5 |
| 1e-09 | 15.7 | 6.1 |
| 1e-10 | 15.7 | 5.0 |
| 1e-11 | 15.7 | 3.4 |
| 1e-12 | 15.7 | 3.4 |
| 1e-13 | 15.7 | 1.6 |
| 1e-14 | 15.7 | 0.9 |
| -1e+01 | 15.7 | 15.7 |
| -1e+00 | 15.7 | 14.4 |
| -1e-01 | 15.7 | 13.9 |
| -1e-02 | 15.7 | 12.3 |
| -1e-03 | 15.7 | 11.8 |
| -1e-04 | 15.7 | 10.5 |
| -1e-05 | 15.7 | 9.6 |
| -1e-06 | 15.7 | 8.4 |
| -1e-07 | 15.7 | 8.5 |
| -1e-08 | 15.7 | 6.5 |
| -1e-09 | 15.7 | 6.1 |
| -1e-10 | 15.7 | 5.0 |
| -1e-11 | 15.7 | 3.4 |
| -1e-12 | 15.7 | 3.4 |
| -1e-13 | 15.7 | 1.6 |
| -1e-14 | 15.7 | 0.9 |

## Significant digits retained (float32, floor ≈ 6.9)

| Â_c (m/s²) | stable Eq. 4.16 | naive Eq. 4.15 |
|---|---|---|
| 1e+02 | 6.9 | 6.9 |
| 1e+01 | 6.9 | 6.3 |
| 1e+00 | 6.9 | 6.2 |
| 1e-01 | 6.9 | 4.9 |
| 1e-02 | 6.9 | 3.7 |
| 1e-03 | 6.9 | 3.0 |
| 1e-04 | 6.9 | 1.6 |
| 1e-05 | 6.9 | 0.7 |
| 1e-06 | 6.9 | 0.0 |
| 1e-07 | 6.9 | 0.0 |
| 1e-08 | 6.9 | 0.0 |
| 1e-09 | 6.9 | 0.0 |
| 1e-10 | 6.9 | 0.0 |
| 1e-11 | 6.9 | 0.0 |
| 1e-12 | 6.9 | 0.0 |
| 1e-13 | 6.9 | 0.0 |
| 1e-14 | 6.9 | 0.0 |
| -1e+01 | 6.9 | 6.9 |
| -1e+00 | 6.9 | 5.5 |
| -1e-01 | 6.9 | 4.9 |
| -1e-02 | 6.9 | 3.5 |
| -1e-03 | 6.9 | 3.0 |
| -1e-04 | 6.9 | 1.6 |
| -1e-05 | 6.9 | 0.7 |
| -1e-06 | 6.9 | 0.0 |
| -1e-07 | 6.9 | 0.0 |
| -1e-08 | 6.9 | 0.0 |
| -1e-09 | 6.9 | 0.0 |
| -1e-10 | 6.9 | 0.0 |
| -1e-11 | 6.9 | 0.0 |
| -1e-12 | 6.9 | 0.0 |
| -1e-13 | 6.9 | 0.0 |
| -1e-14 | 6.9 | 0.0 |

## Acceptance (float64)

Stable form worst case: **15.7** digits (criterion ≥ 14.7, i.e. ≤ 1 digit below the 15.7 precision floor) → **PASS**. Textbook form worst case: **0.9** digits — the catastrophic loss the criterion presupposes.

## Acceptance (float32)

Stable form worst case: **6.9** digits (criterion ≥ 5.9, i.e. ≤ 1 digit below the 6.9 precision floor) → **PASS**. Textbook form worst case: **0.0** digits — the catastrophic loss the criterion presupposes.

## Non-intercept guard and continuity

D < 0 (Â_c = −100): status LINEAR_FALLBACK, t_go = 10.000000 s (= R/V̂_c), no NaN propagation. At Â_c = 0 exactly: t_go = 10 s against the limit R/V̂_c = 10 s. **PASS**.
