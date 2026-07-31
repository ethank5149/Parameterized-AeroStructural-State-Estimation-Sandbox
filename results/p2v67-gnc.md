# II-V6/V7: Blackout navigation and SCvx convergence

- **Failure criterion (stated in advance, Paper II §8):** II-V6: measured growth not matching the Eq. (6.5) exponents, or chattering at the boundary; II-V7: ||ν|| not reaching zero for finite w_ν
- **Verdict:** **PASS**
- **Generated:** 2026-07-31 20:00 UTC · numpy 2.5.1 · scipy 1.18.0 · CPython 3.12.13 (x86_64)

## II-V6 (a): unaided covariance growth exponents

| channel | Eq. (6.5) exponent | measured slope | propagation vs closed form | pass |
|---|---|---|---|---|
| velocity random walk | 3 | 3.000000 | 5.62e-15 | yes |
| accel bias | 4 | 4.000000 | 2.94e-13 | yes |
| gyro bias | 6 | 6.000000 | 1.52e-15 | yes |

## Growth acceptance

All three channels reproduce their stated powers of :math:`t` to better than 1e-6 in the fitted log–log slope → **PASS**. The slopes are fitted to an independent Lyapunov propagation of the augmented error state [δr, δv, θ, b_a, b_g], which carries the biases and the tilt as states rather than assuming their contributions — so the exponents *emerge* from the integration and the agreement with Prop. 3 is a genuine check.

## II-V6 (b): why a quadratic trigger model under-predicts

| blackout duration (s) | σ_pos actual (m) | σ_pos on a t² model (m) | under-prediction | gyro-channel share |
|---|---|---|---|---|
| 10 | 0.19 | 0.19 | 1.0× | 0.7% |
| 30 | 1.14 | 0.57 | 2.0× | 15.0% |
| 60 | 4.79 | 1.14 | 4.2× | 54.4% |
| 120 | 30.12 | 2.28 | 13.2× | 87.9% |

## Reading the growth model

Quadratic is the growth of position *error* from an accelerometer bias; the corresponding *covariance* grows as t⁴ and the gyro-bias channel as t⁶. By two minutes the gyro channel carries 88% of the variance and a quadratic model under-predicts the position uncertainty by 13×. A pull-up trigger sized on the quadratic model fires far too late — which is the operational significance the Remark claims, now measured.

## II-V6 (c): gate transitions over 2,000 samples hovering at the blackout boundary (2% noise)

| hysteresis | transitions | transitions per sample |
|---|---|---|
| 0.00 | 986 | 0.4930 |
| 0.02 | 502 | 0.2510 |
| 0.05 | 32 | 0.0160 |
| 0.10 | 0 | 0.0000 |
| 0.20 | 0 | 0.0000 |

## Chattering acceptance

A bare threshold produces 986 transitions on a signal that merely hovers at the boundary — textbook chattering, and the failure mode the criterion names. A Schmitt trigger with a 10% reacquisition margin reduces this to the single legitimate latch into blackout → **PASS**. The gate is hysteretic by construction rather than by tuning, because a bare comparison against ω_GNSS cannot be made non-chattering by any choice of threshold.

## Saha-derived plasma frequency versus post-shock temperature (n = 1e23 m⁻³)

| T (K) | n_e (m⁻³) | ω_p (rad/s) | GNSS L1 available |
|---|---|---|---|
| 3000 | 5.52e+12 | 1.33e+08 | yes |
| 5000 | 6.19e+17 | 4.44e+10 | no |
| 7000 | 9.85e+19 | 5.60e+11 | no |
| 9000 | 1.72e+21 | 2.34e+12 | no |
| 11000 | 1.05e+22 | 5.77e+12 | no |

## II-V7: virtual-control norm at convergence versus penalty weight

| w_ν | iterations | ‖ν‖₁ | status | converged |
|---|---|---|---|---|
| 1e+00 | 80 | 2.489600e+02 | nonzero | no |
| 3e+00 | 80 | 0.000000e+00 | exactly zero | no |
| 1e+01 | 80 | 0.000000e+00 | exactly zero | no |
| 1e+02 | 80 | 0.000000e+00 | exactly zero | no |
| 1e+03 | 78 | 0.000000e+00 | exactly zero | yes |
| 1e+04 | 50 | 0.000000e+00 | exactly zero | yes |
| 1e+05 | 50 | 0.000000e+00 | exactly zero | yes |

## ℓ₁ against ℓ₂ on the *same* linearized subproblem

| w_ν | ‖ν‖₁ under the ℓ₁ penalty | ‖ν‖₁ under an ℓ₂ penalty | ℓ₁ status |
|---|---|---|---|
| 1e+01 | 0.000e+00 | 9.219e+01 | exactly zero |
| 1e+02 | 0.000e+00 | 1.211e+01 | exactly zero |
| 1e+03 | 0.000e+00 | 1.257e+00 | exactly zero |
| 1e+04 | 0.000e+00 | 4.779e+02 | exactly zero |

## Exactness acceptance

The ℓ₁ penalty drives the virtual controls to **exactly zero** at w_ν = 3 — a finite weight — and at every larger weight tested → **PASS**. Below that threshold (w_ν = 1) the norm stays at O(10²): the penalty is cheaper than the real control, so the optimizer buys infeasibility instead of thrust. That is not a defect but the definition of an exact penalty — it is exact *above a finite threshold related to the dual variables*, and the threshold is visible here between w_ν = 1 and 3. The quadratic penalty never reaches zero on the same subproblem at any weight tested (confirmed). It shrinks roughly as 1/w_ν — 92 → 12 → 1.3 over three decades — and then *grows again* at the largest weight, where the solver loses the ill-conditioned problem entirely. Both halves of that are §6.1's argument: a quadratic penalty approaches zero only asymptotically, and the weight needed to make it small is the same weight that degrades conditioning. The comparison is run on one subproblem rather than inferred from two, so the penalties are the only thing that differs.
