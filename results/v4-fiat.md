# I-V4-FIAT: Ablation — independent FIAT-formulation solver (cross-verification leg)

- **Failure criterion (stated in advance, Paper I §8):** any conservation, convergence, interface-exactness or Jacobian property failing its closed form; the stated 5% recession criterion against a published FIAT reference case is NOT evaluated here — see scope
- **Verdict:** **PASS**
- **Generated:** 2026-08-01 22:23 UTC · numpy 2.5.1 · scipy 1.18.0 · CPython 3.12.13 (x86_64)

## Closed-form and conservation checks

| check | measured | expected | pass |
|---|---|---|---|
| energy stored / energy supplied, sealed inert stack | 1 | 1 | yes |
| pyrolysis gas released / solid mass lost | 1.00487 | 1 | yes |
| steady wall temperature / series-resistance value (k ratio 10:1) | 1 | 1 | yes |
| max relative Jacobian error vs central differences | 9.72067e-07 | 0 | yes |
| ln(1+2λB')/(2λB') vs 2λB'_1/(exp(2λB'_1)−1) | 0 | 0 | yes |
| gray-kernel flux in radiative equilibrium (expect 0) | 1.59255e-16 | 0 | yes |

Every entry is a property the discretisation must have exactly, or to a stated tolerance, independent of any external data. The series-resistance check is the one that earns the harmonic conductivity mean at ply interfaces: an arithmetic mean fails it by a margin that grows with the conductivity ratio, and the error lands on the bondline temperature, which is the number a sizing run exists to produce.

## Convergence of terminal recession under refinement

| refinement | coarse | medium | fine | observed order | pass |
|---|---|---|---|---|---|
| cells (20 → 40 → 80) | 0.441424 mm | 0.439907 mm | 0.439706 mm | 2.92 | yes |
| steps (60 → 120 → 240) | 0.439330 mm | 0.439907 mm | 0.440204 mm | 0.96 | yes |

Recession is a *derived* quantity — it integrates a surface thermochemistry lookup driven by a wall temperature that is itself the solution of the energy balance — so it is the strictest single scalar to converge, and the one the stated criterion is written against. Spatial refinement carries the geometric cell distribution with it, so the observed order is not the formal order of a uniform grid. Time refinement is backward Euler, first order by construction; a higher observed order here would be a sign of an under-resolved run, not a better scheme.

## Relationship to the stated V4 criterion — still PENDING

The failure criterion in Paper I §8 is *recession within 5% of a FIAT reference case*. This leg does not evaluate it and must not be read as doing so.

What now exists is an **independent implementation of FIAT's published formulation** — Chen & Milos 1999 Eqs. (1)–(11), Milos, Chen & Squire 2006 — written from the open literature, since FIAT itself is US-government-controlled and cannot be run here. The project therefore has two structurally different solvers for the same physics: Chebyshev collocation with method-of-lines integration on a Landau grid, and conservative finite volume with backward Euler and an analytic Newton Jacobian. Agreement between them is a genuine result about the discretisations, and it is not a validation against FIAT.

Closing the criterion as written still needs a published reference case carrying its own wall boundary condition and material property set; `docs/FIAT-reference-data.md` specifies what that requires and where to look. I-V4 remains **partially complete**.

## PICA conductivity against pressure — MEDLI2 paper, Table 3

| model | virgin 1 atm | char 1 atm | virgin 0.001 atm | char 0.001 atm | virgin pressure ratio |
|---|---|---|---|---|---|
| Heritage | 0.174 | 0.224 | 0.520 | 0.202 | ×2.99 |
| MEDLI2 | 0.169 | 0.169 | 0.127 | 0.143 | ×0.75 |

All eight values are published and are reproduced here exactly. The **two published models disagree about the sign of the effect**: the Heritage model has virgin conductivity rising by a factor of three as pore-gas pressure falls to 0.001 atm, the MEDLI2 re-measurement has it falling by a quarter, and at 0.001 atm they differ by a factor of **4.1** — in the regime that governs entry. A solver with pressure-independent conductivity can represent neither, which is why the property model now interpolates in log-pressure between the published anchors and clamps rather than extrapolating beyond them. Neither model is presented as correct.

## Decomposition kinetics — targets and round-trip identifiability

| check | measured | target | pass |
|---|---|---|---|
| TGA char yield vs published bulk densities | 0.828892 | 0.828467 | yes |
| 2% mass-loss onset at 20 K/min (K) | 557.052 | 557 | yes |
| peak mass-loss-rate temperature at 20 K/min (K) | 799.2 | 799 | yes |
| worst relative error recovering A from a 3x-perturbed guess | 6.75704e-12 | 0 | yes |
| worst relative error recovering E from a 3x-perturbed guess | 3.9313e-13 | 0 | yes |

**No published Arrhenius triplets for PICA exist in this repository's reference set.** The MEDLI2 material-response paper characterises conductivity, specific heat and density and is silent on decomposition rates; the MSL reconstruction paper notes that 'no kinetic rate-limited recession model for PICA exists that is sufficiently validated for use in TPS design'. Rather than assert three numbers, the triplets are pinned to the stated targets above and those targets are checked here, so the assumption is visible and falsifiable. The char yield is not a free parameter: it follows from the published virgin and char bulk densities.

The last two rows are the useful part. Forward-modelling a scan and then fitting it recovers the generating pre-exponentials and activation energies to better than a part in a thousand, so a real thermogravimetric curve — one curve — closes the largest remaining gap in the material model the moment one is available.

## Published PICA pyrolysis kinetics — and the limit of FIAT Eq. (8)

| model form | peak at 10 K/min | peak at 366 K/min | shift |
|---|---|---|---|
| parallel — Torres-Herrador 2019 Table 2, *in* Eq. (8)'s form | 576 K | 681 K | **+105 K** |
| competitive — Torres-Herrador 2020 Table 1, *outside* it | 829 K | 700 K | **-129 K** |

The two heating rates are the ones the 2020 model was calibrated against: Wong et al. at 10 K/min and Bessire & Minton at 366 K/min. Carbon/phenolic is measured to shift its pyrolysis peak **down** in temperature as the heating rate rises — Stokes reported it above 300 K/min — and Torres-Herrador et al. state that parallel mechanisms are 'not able to reproduce this effect due to their mathematical formulation'.

That is reproduced here: **PASS**. FIAT Eq. (8) is a sum of independent parallel reactions, and such a sum can only shift its peak upward, because every term does. Recovering the measured direction needs two reactions competing for the same reactant. **This is a model-form limitation of Eq. (8), not a calibration error, and no refitting removes it.** It matters in flight rather than in the laboratory: heating rates across the MSL heat shield run from 60 to 60000 K/min, while legacy TGA calibration data rarely exceeds tens of K/min.

Two cross-checks between unrelated sources, both passing: the competitive model's slow-branch char yield is 0.836 against 0.828 from the published bulk densities, and the 2019 set's density-loss fractions (0.544 of the resin) scaled by PICA's 94/274 resin fraction give a 18.3% composite mass loss against the 17.2% those same densities imply.

## Material used, by provenance

| quantity | value | source |
|---|---|---|
| virgin bulk density | 274.0 kg/m³ | published (Heritage PICA, MEDLI2 paper): 274 |
| RT conductivity, both pressures | 8 values | published (Table 3) |
| char bulk density | 227.0 kg/m³ | reconstructed from composition |
| PICA kinetics (parallel) | 6 reactions | published (Torres-Herrador 2019, Table 2) |
| PICA kinetics (competitive) | 10 parameters | published (Torres-Herrador 2020, Tables 1-2) |
| kinetics used by the solver | — | **Eq. (8) parallel form; see the model-form table** |
| conductivity/c_p slopes | — | **representative, not published** |
| B' table | — | **synthetic logistic, not thermochemistry** |

The solver is what is being verified here, not the material. The bulk density and room-temperature conductivities are taken from the MEDLI2 material-response paper's Heritage PICA row and are reproduced exactly; everything else is of the right magnitude and no more. Recession numbers in this report describe the discretisation and must not be read as PICA predictions.
