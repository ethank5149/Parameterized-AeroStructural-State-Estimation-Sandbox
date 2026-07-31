# INT: Coupled single-trajectory integration (roadmap item 13)

- **Failure criterion (stated in advance, this runner — see the scope note):** self-stated (not a tabulated V&V task): a state dimension that changes in flight; a branch on flight regime; an aerothermal loop that does not close within one right-hand side; or an implicit cost that grows with retained structural modes
- **Verdict:** **PASS**
- **Generated:** 2026-07-31 21:04 UTC · numpy 2.5.1 · scipy 1.18.0 · CPython 3.12.13 (x86_64)

## Configuration

Global state of 71 components: 3 position, 3 velocity, 4 quaternion, 3 body rate, 1 mass, 12 structural modal, 44 thermal (temperature and three component densities on the Landau grid) and 1 recession. Entry at 120 km and 6500 m/s, flight path -8°, propagated 110 s by a single BDF call.

## Fixed state dimension

The trajectory is a 71 × 111 array: the dimension is a property of the configuration, fixed at construction, and nothing in the right-hand side can add or remove a degree of freedom → **PASS**. This is the property the rank-3 batching argument of Paper I §5.2 rests on; a moving-mesh formulation cannot offer it because each replicate's mesh diverges after its first remesh.

## Single integration across regimes

The right-hand side contains no branch on altitude or on the Kármán line (source inspected: clean), and the acceleration is continuous across it — the largest step-to-step change over 90–110 km is 2.74× the median, i.e. no jump → **PASS**. The atmospheric terms decay with the density model and simply stop mattering, which is the mechanism Paper II §7.2 relies on to avoid a phase handoff.

## Trajectory summary

| quantity | start | peak | end |
|---|---|---|---|
| altitude (km) | 120.0 | — | 12.3 |
| dynamic pressure (kPa) | 0.019 | 1059.9 | 803.4 |
| stagnation heat flux (MW/m²) | 0.0125 | 1.718 | 0.337 |
| wall temperature (K) | 300 | — | 1712 |
| recession (mm) | 0.000 | — | 1.3802 |
| effective nose radius (m) | 0.3000 | — | 0.3014 |

## Aerothermal loop closure

The recession rate is non-negative at every sampled state (minimum 0.000e+00 m/s), so recession is irreversible as the oxidative model requires; the differenced dense output dips by at most -3.2e-11 m, which is interpolation noise at the solver tolerance rather than physics. Recession grows R_eff from 0.3000 to 0.3014 m, which reduces the convective heating this run would see at fixed flight conditions by 0.23% → **PASS**. Nose blunting is self-limiting, and the loop closes inside one right-hand side rather than across a coupling iteration — which is what Paper II §4.1 requires for the feedback to act within the same time step.

## Implicit cost versus retained structural modes (60 s arc, BDF)

| modes | ω_max (rad/s) | RHS evaluations | wall (s) |
|---|---|---|---|
| 2 | 730.5 | 727 | 0.19 |
| 3 | 1432.0 | 726 | 0.19 |
| 4 | 2366.9 | 715 | 0.19 |
| 5 | 3534.7 | 713 | 0.19 |
| 6 | 4972.9 | 721 | 0.20 |

## Stiffness acceptance

Right-hand-side evaluations vary by only 1.02× as the highest retained mode goes from 730 to 4973 rad/s → **PASS**. A fully implicit method removes the Prop. 2 constraint on the *coupled* system, confirming on the assembly what V3 measured on the structural block alone. Worth recording separately: LSODA — nominally a stiffness-switching method — fails to switch here and needs on the order of 10⁵–10⁶ evaluations for the same arc, so 'adaptive' is not a substitute for 'implicit' in this system.

## I-V8 residual: theoretical occupancy of the batched stage kernel (SM 8.6, 82 SMs)

| threads/block | registers/thread | blocks/SM | warps/SM | occupancy | limiter |
|---|---|---|---|---|---|
| 64 | 27 | 16 | 32 | 0.667 | shared_memory |
| 128 | 27 | 12 | 48 | 1.000 | warps |
| 256 | 27 | 6 | 48 | 1.000 | warps |
| 512 | 27 | 3 | 48 | 1.000 | warps |
| 1024 | 27 | 1 | 32 | 0.667 | warps |

## On occupancy

I-V8 asks for *achieved* occupancy, which is a hardware counter and needs Nsight Compute — not available here. What is computed above is **theoretical** occupancy: the standard CUDA occupancy model evaluated from the compiled kernel's register and shared-memory footprint against the device's per-SM limits. That is exact arithmetic, not an estimate, and it bounds achieved occupancy from above. Together with the throughput-saturation curve already in I-V8 — the externally observable consequence of occupancy — it closes most of the instrumentation gap. The residual is the difference between the bound and the counter, which remains **pending** a profiler.
