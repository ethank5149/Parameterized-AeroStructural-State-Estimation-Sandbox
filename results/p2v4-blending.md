# II-V4: Aerodynamic blending — integrated loads and trim sensitivity

- **Failure criterion (stated in advance, Paper II §8):** trim incidence shifting by more than 0.1° across the blend-width sweep
- **Verdict:** **PASS**
- **Generated:** 2026-08-01 01:24 UTC · numpy 2.5.1 · scipy 1.18.0 · CPython 3.12.13 (x86_64)

## Configuration

Generic cambered lifting body, 3840 panels, wetted area 37.20 m²; M∞ = 8, γ = 1.4, q∞ = 57.6 kPa. Moment reference at mid-length. C_p,max from the Rayleigh–Pitot relation is 1.8274.

## Integrated loads at α = 6° and trim incidence versus blend width

| δ_blend (rad) | δ_blend (deg) | normal force (kN) | pitching moment (kN·m) | trim α (deg) |
|---|---|---|---|---|
| 0 (unblended) | 0.00 | 33.6679 | 6.4966 | 5.20785 |
| 0.0025 | 0.14 | 33.6675 | 6.4960 | 5.20791 |
| 0.0050 | 0.29 | 33.6668 | 6.4949 | 5.20816 |
| 0.0100 | 0.57 | 33.6647 | 6.4885 | 5.20926 |
| 0.0200 | 1.15 | 33.6559 | 6.4645 | 5.21428 |
| 0.0400 | 2.29 | 33.6208 | 6.3451 | 5.23439 |

## Acceptance

Trim incidence moves by **0.0265°** across the full sweep — from the unblended closure to a half-width of 2.3° — against the criterion of 0.1° → **PASS**. The blend is load-neutral at the widths that matter: it perturbs only panels within the seam band, whose pressure contributions are small and whose moment arms about the reference point largely cancel.

## Convergence to the unblended limit

Trim offset from the unblended closure grows monotonically with the blend width (0.00006°, 0.00031°, 0.00141°, 0.00643°, 0.02654°) and scales smoothly to zero as δ_blend → 0 (monotone). That is the behaviour a numerical expedient should show: the blended closure is a controlled perturbation of the physical one, not an independent model.

## The seam at δ_c = 0 (Paper II, Remark 2)

| quantity | unblended | blended (δ_blend = 0.05 rad) |
|---|---|---|
| C_p jump across the seam | 2.52e-08 | — |
| dC_p/dδ, leeward side | +0.2520 | +0.1276 |
| dC_p/dδ, windward side | +0.0000 | +0.1244 |
| relative slope mismatch | 1.00 | 2.58e-02 |

## Reading the seam

The unblended closure is continuous (2.5e-08) but its slope jumps: the Newtonian branch has zero slope at δ_c = 0 while the expansion branch does not — the C⁰-but-not-C¹ defect Remark 2 identifies. The C² smoothstep removes it, closing the slope mismatch to 2.6e-02 (**PASS**). A cubic smoothstep would restore C¹ only; the quintic used here also matches second derivatives at the band edges, which is what keeps the closure inside the framework's smoothness claim.
