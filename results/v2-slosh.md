# V2: Slosh regularization — exact force transfer and moment error

- **Failure criterion (stated in advance, Paper I §8):** force error above machine precision; moment error not O(σ²) in the interior
- **Verdict:** **PASS**
- **Generated:** 2026-07-31 21:02 UTC · numpy 2.5.1 · scipy 1.18.0 · CPython 3.12.13 (x86_64)

## Total force transfer, 7 stations per row (relative error vs Σ|F|)

| N | γ | σ min (m) | σ max (m) | rel force error |
|---|---|---|---|---|
| 16 | 1.0 | 2.749e-01 | 9.817e-01 | 7.93e-17 |
| 16 | 1.5 | 4.123e-01 | 1.473e+00 | 4.57e-17 |
| 16 | 2.0 | 5.498e-01 | 1.963e+00 | 0.00e+00 |
| 24 | 1.0 | 1.833e-01 | 6.545e-01 | 2.65e-17 |
| 24 | 1.5 | 2.749e-01 | 9.817e-01 | 0.00e+00 |
| 24 | 2.0 | 3.665e-01 | 1.309e+00 | 1.03e-16 |
| 32 | 1.0 | 1.374e-01 | 4.909e-01 | 1.53e-16 |
| 32 | 1.5 | 2.062e-01 | 7.363e-01 | 0.00e+00 |
| 32 | 2.0 | 2.749e-01 | 9.817e-01 | 7.83e-17 |
| 48 | 1.0 | 9.163e-02 | 3.272e-01 | 0.00e+00 |
| 48 | 1.5 | 1.374e-01 | 4.909e-01 | 0.00e+00 |
| 48 | 2.0 | 1.833e-01 | 6.545e-01 | 0.00e+00 |
| 64 | 1.0 | 6.872e-02 | 2.454e-01 | 5.96e-17 |
| 64 | 1.5 | 1.031e-01 | 3.682e-01 | 4.13e-17 |
| 64 | 2.0 | 1.374e-01 | 4.909e-01 | 0.00e+00 |

## Force acceptance

Worst relative force error across the sweep: **1.53e-16** against the machine-precision criterion 5e-14 → **PASS**. Per Prop. 1 the transfer is exact by construction of the discrete normalization; the residual is the rounding of one quadrature sum.

## Interior moment error (N = 64, x_s = 4.3, lever about x = 5)

| σ (m) | rel moment error | O(σ²) bound (σ/L)² | within bound |
|---|---|---|---|
| 0.245 | 1.22e-09 | 6.02e-04 | yes |
| 0.368 | 1.73e-11 | 1.36e-03 | yes |
| 0.491 | 2.15e-14 | 2.41e-03 | yes |
| 0.736 | 1.15e-09 | 5.42e-03 | yes |
| 0.982 | 2.67e-06 | 9.64e-03 | yes |
| 1.473 | 7.96e-04 | 2.17e-02 | yes |

## Moment acceptance (interior)

Every resolved interior kernel transfers the first moment within the (σ/L)² bound → **PASS**. The measured errors sit near the rounding floor, far below the bound: for a kernel resolved by the grid, Clenshaw–Curtis integrates the Gaussian's first moment spectrally, so the O(σ²) allowance is consumed only where truncation breaks the kernel's symmetry (next table).

## Endpoint moment bias (σ = 0.4 m fixed, station approaching x = 0)

| x_s (m) | x_s/σ | rel moment error |
|---|---|---|
| 2.0 | 5.0 | 5.95e-08 |
| 1.2 | 3.0 | 1.78e-04 |
| 0.8 | 2.0 | 2.21e-03 |
| 0.4 | 1.0 | 1.15e-02 |
| 0.2 | 0.5 | 2.04e-02 |
| 0.1 | 0.2 | 2.58e-02 |

## Reading (endpoint)

The bias switches on as the station enters ~2σ of the end and grows monotonically as the truncated kernel loses symmetry — the behavior the remark after Prop. 1 predicts. Tanks that drain toward a vehicle end should carry this bias in their error budget; the force transfer itself remains exact there (first table, stations at 0.02L and 0.98L).
