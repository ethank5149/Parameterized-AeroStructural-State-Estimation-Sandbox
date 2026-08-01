# II-V5: Regime transition — coast conservation, step size and wall clock

- **Failure criterion (stated in advance, Paper II §8):** secular energy drift exceeding 1e-8 per orbit
- **Verdict:** **PASS**
- **Generated:** 2026-08-01 01:24 UTC · numpy 2.5.1 · scipy 1.18.0 · CPython 3.12.13 (x86_64)

## Configuration

Circular orbit at 400 km altitude, 51.6° inclination; period 5553.6 s. Gravity is the J₂ model of Eq. (7.1)–(7.2) with the paper's constants (μ = 3.986004418e+14 m³/s², R⊕ = 6.37814e+06 m, J₂ = 0.00108263).

## Invariant drift versus arc length

| revolutions | relative energy drift | drift per orbit | h_z drift |
|---|---|---|---|
| 0.25 | 7.594e-16 | 3.038e-15 | 5.908e-16 |
| 0.5 | 1.304e-14 | 2.607e-14 | 6.144e-15 |
| 1 | 1.658e-14 | 1.658e-14 | 8.625e-15 |
| 2 | 3.595e-14 | 1.797e-14 | 1.855e-14 |

## Conservation acceptance

Worst secular energy drift: **2.61e-14 per orbit** against the criterion 1e-08 → **PASS**, a margin of roughly 6 decades. The polar angular momentum is reported alongside because the J₂ field is axisymmetric as well as conservative: a scheme can leak one invariant while holding the other, so both are checked.

## Secular J₂ signature over one orbit

| quantity | measured | analytic first-order |
|---|---|---|
| nodal regression (deg/day) | -5.0135 | -5.0023 |
| J₂ vs spherical position difference at half an orbit | 38.1 km | order kilometres (§7.1) |

## Reading the secular terms

Nodal regression agrees with the classical first-order rate to 0.2% (**PASS**); the residual is the short-period oscillation the secular average deliberately omits. The 38 km separation from a spherical model over half an orbit is what §7.1 means by 'large compared with any meaningful terminal accuracy requirement' — the reason J₂ is not optional here.

## Aerodynamic-to-gravitational acceleration ratio versus altitude

| altitude (km) | ratio |
|---|---|
| 0 | 3.83e+02 |
| 50 | 1.08e+00 |
| 86 | 1.59e-02 |
| 100 | 3.07e-03 |
| 150 | 8.69e-06 |
| 200 | 2.46e-08 |
| 300 | 1.97e-13 |
| 500 | 1.26e-23 |

## Regime transition

The ratio decays monotonically through 25 decades with a log-slope varying by 4.2e-07 relative between samples — that is, smoothly, with no discontinuity anywhere (**PASS**). This is the whole mechanism of §7.2: no Kármán-line switch is taken, the aerodynamic term simply stops mattering, so one integration spans boost, coast and re-entry without a handoff to reinterpolate across. The single-scale-height exponential used here over-predicts density above about 86 km, which makes these crossing altitudes conservative.

## Coast strategies over 300 s with a 100 rad/s structural mode

| strategy | RHS evaluations | mean step (s) | wall (s) | energy drift | final modal energy ratio |
|---|---|---|---|---|---|
| single-integration | 93,290 | 0.0502 | 1.694 | 3.29e-15 | 1.46e-17 |
| frozen-structure | 6,220 | 0.8021 | 0.114 | 8.37e-13 | 9.90e-07 |

## Strategy acceptance

Freezing the structural block once its modal energy has fallen to 9.9e-07 of its initial value costs **15× fewer right-hand-side evaluations** and 15× less wall clock, with both strategies holding energy to better than 1e-08 (**PASS**). The measurement settles Remark 5 in favour of freezing *for a coast of this length*: the structural stability bound, not the orbital dynamics, sets the step while the block is live, and the block is demonstrably quiescent long before the coast ends. The switch is defensible precisely because the freeze condition is checked rather than assumed — and note the saving grows with coast duration, since the ring-down time is fixed while the coast is not.
