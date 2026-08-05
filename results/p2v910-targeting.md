# II-V9-V14: Lambert targeting, bus dispensing, glide guidance, fractional orbital profiles

- **Failure criterion (stated in advance, Paper II §8):** V9: relative arrival error > 1e-7 on any physically flyable transfer, or endpoint energy/angular-momentum mismatch > 1e-9. V10: any released vehicle missing its aimpoint by > 1 m, or the ordering search returning a cost above the exhaustive optimum. V11: range integral differing from its closed form by > 1e-9, flown range not monotone in commanded drag, or reversals failing to reduce crossrange by 10x. V12: the Kepler deorbit solve differing from the integrated trajectory beyond tolerance, or the three-leg range accounting failing to close. V13: Breguet range not linear in L/D, mass-ratio doublings not adding equal range, or the cruise-climb differing between vehicles. V14: containment radii disagreeing with their closed-form limits, a failed radius/probability round-trip, or any architecture without a ledger, verdict and CEP/R95 pair
- **Verdict:** **PASS**
- **Generated:** 2026-08-05 10:10 UTC · numpy 2.5.1 · scipy 1.18.0 · CPython 3.14.6 (x86_64)

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

## V11 — range-energy relation and flown range

| reference bank (deg) | predicted range (km) | flown (km) | prediction error | reversals |
|---|---|---|---|---|
| 30 | 8890 | 7844 | +13% | 1 |
| 45 | 7259 | 7729 | -6% | 1 |
| 60 | 5133 | 7469 | -31% | 1 |
| 70 | 3511 | 7044 | -50% | 1 |

Each reference is the equilibrium-glide drag profile at the stated nominal bank, so all four are flyable; a larger bank asks the vehicle to fly deeper and shorter, which is how range is traded. The prediction is the shallow-glide integral of the range-energy relation and the flown value comes from the closed-loop 3-DOF trajectory, so the gap is the cos-gamma term the prediction drops plus residual tracking error, and it widens with bank as the command approaches saturation. Best agreement is 6% at moderate bank. Flown range strictly decreasing in reference bank: satisfied. Against its own closed form at constant drag the integral itself is exact to 3.33e-16 relative — that check is pure quadrature and is independent of whether any vehicle could fly the profile.

## V11 — terminal crossrange, with and without bank reversals

| configuration | reversals | crossrange (km) | downrange (km) |
|---|---|---|---|
| single bank sign held | 0 | +1199 | 7673 |
| scheduled-deadband reversals | 3 | -31 | 7832 |

Reversals reduce terminal crossrange by a factor of 39, against a criterion of 10. The uncorrected case is the honest baseline: it is not a failure mode but the natural behaviour of a lifting vehicle holding one bank sign, and it is what the lateral logic exists to remove.

The target here is placed at the range the longitudinal profile actually delivers. That is load-bearing rather than tidy: the lateral logic steers on bearing to the target, so a profile that overflies inverts the bearing part-way through and the deadband stops meaning what it should. Placing the target 1000 km short of the delivered range degrades the benefit from 39x to 4x — which is how this criterion first failed. Range matching is a precondition for the lateral channel, not an independent concern.

## V12 — deorbit design curve from a 200 km circular parking orbit

| perigee (km) | ΔV (m/s) | of orbital speed | transfer arc (deg) | arc (km) | entry γ (deg) |
|---|---|---|---|---|---|
| +80 | 35.9 | 0.46% | 131.4 | 14629 | -0.393 |
| +50 | 45.0 | 0.58% | 108.8 | 12117 | -0.623 |
| +0 | 60.3 | 0.77% | 89.1 | 9920 | -0.884 |
| -100 | 91.4 | 1.17% | 69.3 | 7711 | -1.261 |
| -400 | 188.3 | 2.42% | 46.2 | 5141 | -2.042 |
| -1000 | 401.0 | 5.15% | 30.5 | 3395 | -3.192 |

A negative perigee is virtual — the vehicle never reaches it — and is simply how a steep entry is specified. The burn is cheap in every case: what it buys is timing, not energy. Perigee depth is the dominant choice, trading an order of magnitude in ΔV for roughly a factor of four in transfer arc. Monotonicity of ΔV, arc and entry angle across the sweep: satisfied.

## V12 — closed-form solve against the independent integrator

| quantity | worst discrepancy | criterion | verdict |
|---|---|---|---|
| entry radius | 4.731e-07 m | < 1e-3 m | PASS |
| swept angle | 1.403e-13 rad | < 1e-10 rad | PASS |
| entry speed | 6.312e-10 m/s | < 1e-6 m/s | PASS |
| flight-path angle | 4.775e-14 rad | < 1e-10 rad | PASS |
| ΔV vs vis-viva | 0.000e+00 | < 1e-12 rel | PASS |
| three-leg range closure | 1.110e-16 | < 1e-12 rel | PASS |

The deorbit solve is closed-form Kepler; the reference is the coast integrator advancing the equations of motion from the post-burn state. The two share no code, so agreement at this level is a real check on both. Ground-track walk for this orbit is 22.2 deg per revolution, which is what allows the entry interface to be repositioned by waiting rather than by manoeuvring.

## V13 — Breguet range against its analytic scalings

| property | measured | criterion | verdict |
|---|---|---|---|
| linearity in L/D | 0.000e+00 rel | < 1e-12 | PASS |
| range added per doubling of mass ratio, spread over 4 decades | 6.661e-16 | < 1e-9 | PASS |
| cruise-climb across dissimilar vehicles | 0.000e+00 rel | < 1e-12 | PASS |
| cruise-climb at 30% fuel | 3.03 km | — | — |

Each doubling of mass ratio adds the same absolute range regardless of where it starts, which is the precise content of 'logarithmic in mass ratio'. The cruise-climb is H ln(m_i/m_f) with wing loading, lift coefficient and L/D all cancelling, so two vehicles sharing only a fuel fraction climb identically.

## V13 — where the usual gloss on that scaling goes wrong

| change | range multiplier |
|---|---|
| L/D doubled, 4 -> 8 | 2.000 |
| fuel fraction doubled, 0.30 -> 0.60 | 2.569 |

Fuel is commonly said to show diminishing returns while L/D does not. Over this range the opposite holds, because ln(1/(1-f)) is *convex* in fuel fraction: its derivative 1/(1-f) grows, so doubling f more than doubles the logarithm. The diminishing return is in mass ratio, not in fuel fraction — the row above measures that one directly — and the two are statements about different variables.

## V14 — containment statistics against their closed-form limits

| property | measured | criterion | verdict |
|---|---|---|---|
| elliptical integral at unit aspect ratio vs Rayleigh | 2.220e-16 rel | < 1e-4 | PASS |
| degenerate axis vs the normal quantile | 3.703e-07 rel | < 1e-4 | PASS |
| radius/probability round-trip | 6.467e-11 | < 1e-8 | PASS |
| R95/CEP ratio monotone in elongation | 2.079 -> 2.905 | non-decreasing | PASS |

The radial part of the containment integral is analytic, leaving a one-dimensional quadrature, so the elliptical answers are exact rather than a fitted correction to the circular ones. The ratio rises with elongation towards the one-dimensional value 1.96/0.6745 = 2.906, which means scaling a CEP by the circular 2.079 under-states the 95% radius for every real dispersion.

## V14 — end-to-end budget over every named architecture

| architecture | closes | range (km) | ΔV (m/s) | CEP (m) | R95 (m) | R95/CEP |
|---|---|---|---|---|---|---|
| ballistic-single | no | 2300 | 6030 | 621 | 1290 | 2.079 |
| ballistic-multiple | no | 2300 | 6180 | 950 | 1976 | 2.079 |
| boost-glide | no | 5000 | 6000 | 1051 | 2808 | 2.671 |
| boost-glide-multiple | no | 5000 | 6180 | 1261 | 2792 | 2.214 |
| fractional-orbital-single | yes | 10571 | 6188 | 1088 | 2495 | 2.294 |
| fractional-orbital-glide | yes | 10571 | 6188 | 1323 | 3248 | 2.456 |
| fractional-orbital-multiple | yes | 10571 | 6338 | 1320 | 2881 | 2.182 |
| fractional-orbital-multiple-glide | yes | 10571 | 6338 | 1533 | 3536 | 2.307 |
| ballistic-mixed | no | 5000 | 6180 | 1312 | 2880 | 2.195 |
| fractional-orbital-mixed | yes | 10571 | 6338 | 1577 | 3603 | 2.284 |
| powered-cruise | no | 6040 | 6000 | 847 | 1998 | 2.359 |

Range, propellant and accuracy for one launch site and two aimpoints. Which leg absorbs the range remainder differs by family — parking arc for fractional-orbital profiles, which costs time and no propellant, and boost for suborbital ones, which costs both — so a 'does not close' verdict means different things in the two cases and the budget names which. Every ratio exceeds the circular 2.079, Ratios at exactly 2.079 are isotropic dispersions, which arise where a midcourse correction resets to an isotropic floor and nothing anisotropic follows; every other ratio exceeds it, so the circular scaling never over-states the 95% radius and usually under-states it. Deorbit, dispensing, glide and terminal contributions are now derived from the phase models; only boost injection remains a stated specification.

## What these tasks establish, and what they do not

Neither task compares against a published number. The reference is an *independent numerical path through the same physics*: Lambert solves a boundary-value problem in closed form while the coast propagator integrates the equations of motion, and the two share no code. Agreement rules out a large class of implementation errors in either.

It does **not** rule out a shared modelling assumption, and this is a weaker claim than validation against measurement. Both tasks are stated that way deliberately rather than being dressed as validation. What would strengthen them is a published transfer case with tabulated terminal state, and a published dispensing budget for a stated aimpoint geometry; neither is currently in this repository.
