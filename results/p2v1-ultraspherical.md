# II-V1: Ultraspherical operator — conditioning vs N (univariate leg)

- **Failure criterion (stated in advance, Paper II §8):** κ growing faster than O(N); Mindlin–Reissner block-operator leg pending roadmap item 9
- **Verdict:** **PASS**
- **Generated:** 2026-07-31 21:03 UTC · numpy 2.5.1 · scipy 1.18.0 · CPython 3.12.13 (x86_64)

## Conditioning of the fourth-order variable-EI operator (ultraspherical)

| N | uniform interior raw | uniform precond. | uniform bordered | stepped interior raw | stepped precond. | stepped bordered |
|---|---|---|---|---|---|---|
| 32 | 7.75e+00 | 1.00 | 5.01e+00 | 6.13e+01 | 13.13 | 3.21e+05 |
| 64 | 1.58e+01 | 1.00 | 5.92e+00 | 1.24e+02 | 12.49 | 3.08e+05 |
| 128 | 3.18e+01 | 1.00 | 1.06e+01 | 2.53e+02 | 12.49 | 3.08e+05 |
| 256 | 6.38e+01 | 1.00 | 4.04e+01 | 5.13e+02 | 12.49 | 3.08e+05 |
| 512 | 1.28e+02 | 1.00 | 2.11e+02 | 1.03e+03 | 12.49 | 3.08e+05 |

## Acceptance (operator conditioning)

Fitted log–log slope of the raw interior κ versus N: **1.01** (uniform), **1.02** (stepped) against the criterion ≤ 1.3 (not faster than O(N)) → **PASS**. Under the leading-diagonal right preconditioner the interior κ is **≤ 13.1 at every N** — the O(1) statement of Olver & Townsend, reproduced. The bordered square system (clamped–clamped) grows as N^1.4: the dense boundary rows cost conditioning that the banded interior does not, which is exactly the caveat the Remark in Paper II §5.4 raises and defers to measurement. Free-free conditions are *not* used for this measurement: they leave the rigid-body null space, so the bordered free-free system is singular by construction and the configuration is a generalized eigenproblem (cross-checked below), not a BVP.

## Dense collocation (Paper I V1, elastic κ) vs ultraspherical interior

| N | collocation κ (O(N⁸)) | ultraspherical κ (O(N)) | ratio |
|---|---|---|---|
| 32 | 5.21e+06 | 7.75e+00 | 6.7e+05 |
| 64 | 1.22e+09 | 1.58e+01 | 7.8e+07 |

## Accuracy cross-check

Free-free beam eigenvalues from the ultraspherical pencil at N = 32 match the analytic solution to **1.5e-08** relative (**PASS**) — the same physical problem Paper I's V1 verified in collocation form, now reproduced by the second, independent discretization.

## Block-operator leg — PENDING

The stated II-V1 target is the assembled Mindlin–Reissner 3×3 block operator, which requires the plate kernel of roadmap item 9. This run establishes the univariate machinery and its conditioning behavior; the task is not counted complete until the block measurement runs.
