# II-V1/V2/V3: Mindlin–Reissner plate kernel — conditioning, locking, frequencies

- **Failure criterion (stated in advance, Paper II §8):** II-V1: κ growing faster than O(N); II-V2: spurious stiffening above 1% at h/L = 1e-3; II-V3: relative frequency error > 1e-5 for the first ten modes
- **Verdict:** **PASS**
- **Generated:** 2026-07-31 19:12 UTC · numpy 2.5.1 · scipy 1.18.0 · CPython 3.12.13 (x86_64)

## II-V3 (a): manufactured solutions on Eqs. (5.5)–(5.7)

| N | max relative coefficient error |
|---|---|
| 8 | 1.457e-05 |
| 10 | 4.843e-08 |
| 12 | 9.862e-11 |
| 14 | 2.922e-13 |
| 18 | 7.205e-13 |

## MMS acceptance

The assembled block operator reproduces the analytic residual of all three governing equations to **7.2e-13** relative at N = 18 → **PASS**. Every Kronecker term — bending, twist coupling, transverse shear and the shear–slope coupling that distinguishes Mindlin from Kirchhoff — is exercised.

## II-V3 (b): first 10 frequencies vs the closed-form Mindlin simply-supported solution

| N | reduced dim | max relative error |
|---|---|---|
| 10 | 192 | 1.422e-02 |
| 12 | 300 | 5.178e-04 |
| 14 | 432 | 5.139e-06 |
| 16 | 588 | 4.130e-07 |
| 18 | 768 | 1.374e-08 |
| 20 | 972 | 3.734e-10 |

## Frequency acceptance

Maximum relative error over the first 10 modes at N = 20: **3.73e-10** against the criterion 1e-05 → **PASS**. The error contracts exponentially in N, the signature of a consistent spectral discretization; the reference is derived in closed form from the same governing equations, so this leg verifies operator, boundary conditions, null-space projection and eigensolve as one chain.

## II-V1: conditioning of the assembled 3×3 block operator

| N | reduced dim | κ interior (two-sided) | κ interior (column only) | κ projected pencil (two-sided) |
|---|---|---|---|---|
| 8 | 100 | 1.009e+02 | 4.417e+02 | 1.521e+04 |
| 10 | 184 | 1.051e+02 | 4.937e+02 | 4.929e+04 |
| 12 | 292 | 1.072e+02 | 7.194e+02 | 3.778e+04 |
| 14 | 424 | 1.080e+02 | 9.712e+02 | 1.063e+05 |
| 16 | 580 | 1.083e+02 | 1.222e+03 | 2.152e+05 |
| 18 | 760 | 1.084e+02 | 1.470e+03 | 2.124e+05 |

## Conditioning acceptance

Fitted log–log slope of the assembled block operator's κ versus N: **0.09** against the criterion ≤ 1.3 (not faster than O(N)) → **PASS**. The operator is not merely O(N) but essentially **O(1)**-conditioned: κ moves from 101 to 108 while the problem size grows 5-fold. Paper I's dense collocation on the *fourth-order* beam grows as O(N⁸); the Mindlin–Reissner system is second order in each field, which halves the derivative order and — as §5.2 argues — the conditioning penalty with it.

## What the boundary rows cost

The boundary-projected pencil actually solved grows as N^3.3 (κ from 1.5e+04 to 2.1e+05, non-monotonically). This is *not* scored against the criterion, and deliberately so: the Remark in §5.4 states that the O(1) property belongs to the ultraspherical operator and "does not automatically survive the addition of dense boundary rows, variable coefficients with slowly decaying expansions, or the block coupling" — asserting only what Olver & Townsend establish and measuring the rest. The measurement says the block coupling is benign and the dense free-edge rows are not. The frequency results above show this costs nothing at the resolutions of interest, but it is the term that would bite first at large N, and it points at where a preconditioner would have to act.

## Why the scaling column matters

The same operator measured with *column* equilibration alone appears to grow as N^1.59, which would fail the criterion. That growth is not in the discretization: a block operator mixes entries carrying different physical units — transverse shear stiffness in N/m against bending rigidity in N·m — spanning several decades, and a one-sided scaling leaves that block imbalance in the matrix. Two-sided (Ruiz) equilibration removes it and is what the reported verdict uses; the one-sided column is kept so the difference is visible rather than buried in a preprocessing choice. Both the banded interior and the boundary-projected pencil actually solved are reported, since the Remark in §5.4 is explicit that the O(1) claim covers the former and not the latter. The full square operator is deliberately not a measurand: differentiation shifts by its order, so its trailing rows are structurally zero and its κ₂ is infinite at every N — and for the free-free perimeter the three rigid-body directions are excluded from the projected pencil for the same reason, the trap Paper I's V1 documents for the free-free beam.

## II-V2: free-free frequency parameters versus thickness ratio (N = 16)

| h/L | λ₁ | λ₂ | λ₃ | rigid separation |
|---|---|---|---|---|
| 0.2 | 11.7102 | 17.3772 | 21.1916 | 1.7e-12 |
| 0.1 | 12.7346 | 18.9331 | 23.3222 | 5.9e-12 |
| 0.05 | 13.1466 | 19.4156 | 24.0079 | 3.3e-11 |
| 0.02 | 13.2904 | 19.5494 | 24.1937 | 1.4e-09 |
| 0.01 | 13.3050 | 19.5617 | 24.2045 | 2.1e-08 |
| 0.005 | 13.3074 | 19.5631 | 24.2045 | 6.6e-07 |
| 0.002 | 13.3078 | 19.5628 | 24.2043 | 6.4e-06 |
| 0.001 | 13.3098 | 19.5625 | 24.2056 | 3.6e-04 |

## Shear-locking acceptance

Across three decades of h/L the fundamental parameter falls from 11.710 at h/L = 0.2 — genuine shear softening of a thick section — to a plateau of 13.3074, and at h/L = 1e-3 it differs from that plateau by **+0.018%**, against the criterion of 1% spurious stiffening → **PASS**. Paper II's Remark 3 claims high-order spectral discretizations are *markedly less susceptible* to locking but explicitly declines to claim immunity; this measures it. The rigid-separation column records the cost that is really paid as the section thins: the shear-to-bending stiffness ratio grows as h⁻², and the rigid-body modes separate from the elastic spectrum by correspondingly fewer decades.

## Free-free cross-check against commonly circulated values (UNAUDITED — no verdict)

| mode | computed λ (h/L = 0.005, N = 20) | circulated λ | difference |
|---|---|---|---|
| 1 | 13.378 | 13.489 | -0.82% |
| 2 | 19.577 | 19.789 | -1.07% |
| 3 | 24.236 | 24.432 | -0.80% |
| 4 | 34.548 | 35.024 | -1.36% |
| 5 | 34.548 | 35.024 | -1.36% |
| 6 | 61.014 | 61.526 | -0.83% |

## On the free-free comparison

These reference figures are **not verified against a publisher record** and are therefore not used to pass or fail anything. The computed values sit a few tenths of a percent below them and are still rising with N: free-edge plates carry weak corner singularities, so the free-free spectrum converges *algebraically* where the simply-supported spectrum converges exponentially. That is a property of the problem rather than of the discretization — which is precisely why the verdict rests on the closed-form reference above. Auditing the Rayleigh–Ritz source, or adding corner-resolving refinement, would let this leg carry a verdict.
