# II-V9-V10: Lambert targeting and post-boost bus dispensing

- **Failure criterion (stated in advance, Paper II §8):** V9: relative arrival error > 1e-7 on any physically flyable transfer, or endpoint energy/angular-momentum mismatch > 1e-9. V10: any released vehicle missing its aimpoint by > 1 m, or the ordering search returning a cost above the exhaustive optimum
- **Verdict:** **PASS**
- **Generated:** 2026-08-04 21:31 UTC · numpy 2.5.1 · scipy 1.18.0 · CPython 3.14.6 (x86_64)

## V9 — Lambert transfer envelope

| quantity | measured | criterion | verdict |
|---|---|---|---|
| worst relative arrival error | 2.422e-10 | < 1e-07 | PASS |
| worst endpoint energy mismatch | 2.744e-14 | < 1e-09 | PASS |
| worst endpoint angular-momentum mismatch | 1.871e-13 | < 1e-09 | PASS |
| transfers solved | 720 | — | — |
| of those, physically flyable | 385 | — | — |
| Householder iterations (median / max) | 3 / 8 | — | — |

The arrival check propagates Lambert's velocity through the coast integrator, which shares no code with the solver. The invariance check needs no propagation and so is applied to every transfer, including the 335 whose periapsis lies inside the Earth and which the integrator cannot fly.

## V9 — correction targeting under J2, and the inverse-time-to-go law

| burn t/T | t_go (s) | ΔV (m/s) | ΔV·t_go (km) | residual miss (m) |
|---|---|---|---|---|
| 0.05 | 1710 | 2.736 | 4.68 | 6.63e-01 |
| 0.25 | 1350 | 3.386 | 4.57 | 4.20e-01 |
| 0.50 | 900 | 4.991 | 4.49 | 2.05e-01 |
| 0.75 | 450 | 9.768 | 4.40 | 4.11e-03 |
| 0.90 | 180 | 24.081 | 4.33 | 1.52e-02 |

Uncorrected miss is 4.32 km. The product ΔV·t_go is constant to 7.9% across the arc and equals that miss, which is the |δr|/t_go scaling appearing as a measurement rather than as an assertion. Residual miss stays below 1 m everywhere, including at t/T = 0.9 where the vehicle is nearly collinear with its own aimpoint and the two-body seed alone is useless.

## V10 — dispensing four vehicles, all 24 orderings enumerated

| quantity | measured | criterion | verdict |
|---|---|---|---|
| worst achieved miss, any vehicle | 2.751e-01 m | < 1 m | PASS |
| search cost vs exhaustive optimum | 405.49 vs 405.49 m/s | not above optimum | PASS |
| search method reported | exhaustive | — | — |
| cheapest ordering | 405.49 m/s (1, 3, 0, 2) | — | — |
| dearest ordering | 748.43 m/s | — | — |
| spread across orderings | 85% | — | — |
| natural order (0,1,2,3) | 748.43 m/s — the worst available | — | — |

Ordering is the dominant cost lever, not the individual maneuvers. The natural index order is not merely suboptimal here; it is the worst of the twenty-four.

## V10 — accumulated error and terminal dispersion along the sequence

| release | aimpoint | ΔV (m/s) | 1σ dispersion (m) |
|---|---|---|---|
| 0 | 0 | 56.09 | 2210 |
| 1 | 1 | 125.41 | 4230 |
| 2 | 2 | 195.62 | 5070 |

The bus covariance grows monotonically — each maneuver contributes a positive-semidefinite block and none is removed. The terminal dispersions rise monotonically, so in this configuration the last vehicle released is also the least accurate. That ordering is not guaranteed either way: accumulated error pushes it up along the sequence while the shrinking remaining flight time pushes it down, and which dominates depends on the release schedule and the aimpoint spread. The unit tests exercise a configuration where the two cross and the dispersions are non-monotone. Which vehicle needs the accuracy budget is therefore a result of the schedule and cannot be read off the release order.

## What these tasks establish, and what they do not

Neither task compares against a published number. The reference is an *independent numerical path through the same physics*: Lambert solves a boundary-value problem in closed form while the coast propagator integrates the equations of motion, and the two share no code. Agreement rules out a large class of implementation errors in either.

It does **not** rule out a shared modelling assumption, and this is a weaker claim than validation against measurement. Both tasks are stated that way deliberately rather than being dressed as validation. What would strengthen them is a published transfer case with tabulated terminal state, and a published dispensing budget for a stated aimpoint geometry; neither is currently in this repository.
