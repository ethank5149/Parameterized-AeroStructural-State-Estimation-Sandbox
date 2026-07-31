# V1: Structural operator — conditioning and free-free frequencies

- **Failure criterion (stated in advance, Paper I §8):** relative frequency error > 1e-06 at N = 32 for the uniform case
- **Verdict:** **PASS**
- **Generated:** 2026-07-31 11:15 UTC · numpy 2.5.1 · scipy 1.18.0 · CPython 3.12.13 (x86_64)

## Conditioning of the reduced stiffness operator

| N | uniform κ₂(K̂) raw | uniform κ elastic | stepped κ₂(K̂) raw | stepped κ elastic |
|---|---|---|---|---|
| 8 | 5.11e+16 | 6.03e+01 | 9.83e+15 | 1.66e+02 |
| 12 | 7.03e+15 | 2.25e+03 | 6.34e+15 | 1.05e+04 |
| 16 | 3.27e+15 | 2.28e+04 | 6.16e+16 | 7.60e+04 |
| 20 | 4.25e+15 | 1.32e+05 | 1.00e+17 | 4.37e+05 |
| 24 | 6.05e+15 | 5.51e+05 | 3.58e+16 | 1.89e+06 |
| 28 | 8.01e+15 | 1.83e+06 | 1.03e+17 | 6.16e+06 |
| 32 | 6.34e+16 | 5.21e+06 | 8.14e+16 | 1.77e+07 |
| 40 | 1.96e+16 | 3.00e+07 | 1.74e+17 | 1.04e+08 |
| 48 | 5.17e+16 | 1.26e+08 | 1.70e+17 | 4.44e+08 |
| 64 | 3.37e+16 | 1.22e+09 | 5.51e+17 | 4.40e+09 |

## Interpretation

The raw κ₂(K̂) is pinned at the reciprocal rounding floor (~1/ε) at every N: the free-free operator retains the two *physical* rigid-body null directions, so its smallest singular value is rounding noise by construction. The informative measurand is the elastic condition number σ₁/σ_{n-2}, whose fitted log–log slope is **8.01** (uniform) and **8.04** (stepped) over N ∈ [8, 64]. Paper I, Remark 3 declined to assert a rate; the measured growth remains of the same O(N⁸) order as the unprojected fourth-derivative operator, i.e. the projection removes the constraint-violating extremal modes but does not flatten the asymptotic rate for these profiles.

## Uniform free-free frequencies vs analytic (first 5 elastic modes)

| N | worst rel err | mode 1 | mode 2 | mode 3 | mode 4 | mode 5 |
|---|---|---|---|---|---|---|
| 8 | 1.153e-01 | 1.204e-03 | 6.914e-02 | 1.153e-01 | — | — |
| 12 | 1.818e-02 | 2.696e-07 | 1.660e-04 | 7.090e-04 | 1.361e-02 | 1.818e-02 |
| 16 | 4.238e-04 | 1.650e-11 | 1.130e-07 | 1.467e-06 | 1.475e-04 | 4.238e-04 |
| 20 | 2.166e-06 | 6.985e-11 | 2.882e-11 | 1.049e-09 | 3.950e-07 | 2.166e-06 |
| 24 | 4.451e-09 | 1.593e-10 | 1.620e-11 | 1.949e-12 | 4.212e-10 | 4.451e-09 |
| 28 | 1.258e-10 | 1.258e-10 | 6.736e-11 | 5.971e-12 | 2.021e-12 | 7.177e-12 |
| 32 | 1.220e-09 | 1.220e-09 | 2.335e-12 | 1.104e-11 | 3.109e-12 | 2.951e-12 |
| 40 | 2.484e-09 | 2.484e-09 | 3.994e-10 | 3.659e-11 | 1.174e-11 | 8.143e-12 |
| 48 | 1.124e-08 | 1.124e-08 | 4.390e-10 | 2.050e-10 | 1.732e-10 | 2.913e-11 |
| 64 | 1.676e-09 | 1.574e-09 | 3.543e-10 | 1.676e-09 | 1.033e-09 | 4.403e-10 |

## Acceptance

Worst relative frequency error at N = 32: **1.220e-09** against the criterion 1e-06 → **PASS**.

## Null-space projection vs row replacement (§3.2 counterexample)

At N = 32, the projected pencil returns a spectrum with relative imaginary contamination 0.0e+00 and rigid eigenvalues [-1.91023712e-06  9.36181303e-08] (relative magnitude ≤ 1.5e-15). The conventional row-replacement treatment of the same problem yields 12 eigenvalues with non-negligible imaginary parts (largest |Im λ| = 1.479e+08 against spectral scale 1.795e+07) — the spurious complex modes that manifest as growth in time integration.
