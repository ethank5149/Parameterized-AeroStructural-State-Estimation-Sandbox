# Fidelity upgrade backlog

Upgrades that the references in [`reference/`](reference/) can support but that
the implementation does not yet carry. Each item states **what is assumed
now**, **which source closes it**, and **what measurably changes** — so that
nothing here is a wish, and anything picked up can be finished against a
citation rather than against judgement.

Ordering within each section is by leverage, not by effort. Items marked
**[blocks V-task]** are on the critical path for one of the two verification
tasks still partial (I-V4, II-V8). Completed items are struck through and
kept, with what they actually found — including where the original framing
turned out to be wrong, since an item that was mis-scoped is worth more as a
record than as a deletion.

Two standing rules, inherited from how the rest of this repository was built:

- An upgrade is not done when it runs. It is done when it is *checked against
  the source that motivated it* — a number, a stated scaling, or an
  independent numerical path.
- Where a source cannot close an assumption, say so in the code and leave the
  assumption labelled. Several items below exist only because a previous
  version of that label was too quiet.

---

## 1. Navigation and injection

The injection chain is derived end to end except for two inputs. Both have a
source sitting unread in `reference/`.

- [x] ~~**Initial platform alignment from Titterton & Weston.**~~ **Done —
      and alignment turned out not to be an independent input at all.** A
      stationary gyrocompass levels against gravity and finds north by
      nulling east Earth rate, so its accuracy follows from the *same*
      biases Groves already tabulates. Titterton & Weston §10.3.2 gives
      both closed forms and two numbers to check them against; we
      reproduce **1.0000 mrad** tilt for a 1 milli-g bias and **0.9402
      mrad** azimuth for 0.01 °/hr at 45° against his rounded 1 mrad. The
      guesses were wrong in both directions — tilt 3× pessimistic, azimuth
      optimistic by 10× (marine) to **300× (tactical)**. Two results fell
      out: a **tactical-grade IMU cannot usefully gyrocompass** (5.4°
      azimuth error, since 1 °/hr is 7% of Earth rate) and needs an
      external reference; and **accelerometer bias enters twice and
      exactly equally**, since tilt is `B/g` so the alignment term
      `½g(B/g)t²` *is* the accelerometer term. Injection error at aviation
      grade fell 44.8 m / 0.299 m/s → **18.8 m / 0.127 m/s**.

- [ ] **Full Schuler-loop INS error propagation** (Gelb Ch. 4; Titterton
      Ch. 12). `injection_error` is polynomial and refuses beyond a quarter
      Schuler period — correct for boost, useless for coast. A full error-state
      model would cover INS drift *through* the coast, which is what actually
      sets the navigation quality feeding the midcourse reset floor
      (`DISPERSION_RESETS[MIDCOURSE] = 512 m`, currently a stated nav
      covariance rather than a propagated one).
- [ ] **Gravity anomaly and vertical deflection as error sources.** Gelb
      Fig. 8.2-1 lists gravity uncertainty alongside accelerometer and gyro
      error; we carry neither. For a long ballistic arc this is a real
      contributor to the injection-to-entry mapping and is currently absent
      rather than assumed-small.

## 2. In-depth ablation physics

- [x] ~~**Pore pressure, and the inconsistency it currently papers over.**~~
      **Investigated and closed — the inconsistency is real and the error is
      under 10 %.** [`solver.py`](src/passes/thermal/fiat/solver.py) does feed
      *boundary-layer edge* pressure to a conductivity that is a function of
      *pore gas* pressure. A Darcy solve
      ([`pore_pressure.py`](src/passes/thermal/fiat/pore_pressure.py),
      verified to machine precision against the analytic uniform-source case)
      run on real solver pyrolysis profiles gives pore/surface pressure
      ratios up to **31**, but conductivity errors of only **0.1–8.4 %**,
      because the MEDLI2 model interpolates in *log* pressure and clamps at
      its 1 atm anchor. Left as a diagnostic and deliberately **not** wired
      in: doing so would inject an unmeasured permeability into the main
      solve path to buy a change well inside the 27 % experimental scatter.
      *Two corrections to what this item originally claimed:* **Park &
      Lawrence does not close it** — that paper measures MX4926 carbon
      *cloth* phenolic, a dense ~1.45 g/cm³ nozzle liner at 10⁻¹⁷–10⁻²¹ m²,
      not a ~90 % porous PICA preform. And it does **not** block I-V4.

      **Update — Marschall & Milos has since been obtained, and settles it
      against the stated condition.** The condition written above was
      "worth obtaining only if PICA turns out to sit below 10⁻¹² m²." It
      does not. They measure **FiberForm**, the carbon preform PICA is made
      from, at **7.9×10⁻¹¹ to 5.5×10⁻¹⁰ m²** — the *lowest* specimen is 79×
      above the threshold. The old placeholder bracket had roughly the right
      top end (within 5.5×) and was 79× too tight at the bottom, which is
      exactly why the swept worst case (8.4 %) overstated the measured one.
      Re-running the sweep at the measured permeability gives a peak
      pore/surface ratio of **1.02** at 27.3 kPa and **1.79** at 2.3 kPa,
      for conductivity errors of **0.07 % and 2.43 %**. Including
      Klinkenberg slip cuts the worst ratio further, 1.79 → 1.39, because
      pyrolysis gas is light and hot and therefore deep in the slip regime
      (b ≈ 11 kPa at 2000 K). Implemented in
      [`permeability.py`](src/passes/thermal/fiat/permeability.py) and
      verified against both of the authors' own experiments: the helium
      prediction (**21,484 Pa** against their stated 21,490, and 1.45 % from
      their measurement) and the 293–1200 K furnace series. The pore-pressure
      module stays a diagnostic — the measurement removed the objection that
      kept it out of the solver, and strengthened the conclusion that
      objection was protecting.
- [ ] **Moisture and water phase change.** Omidy et al., *Effects of water
      phase change on the material response of low-density carbon-phenolic
      ablators* (in `reference/`, uncited). We model no moisture at all — and
      the pyrolysis-composition work already found the sub-350 °C water peak
      material enough to distort an elemental balance. Omidy quantifies the
      thermal response effect directly.
- [ ] **Non-constant pyrolysis gas composition through the pyrolysis zone.**
      Rabinovitch's central criticism is that every code assumes constant
      elemental composition while measurement shows it varying strongly with
      temperature (his Table 3, Sykes at 50 °C intervals; Table 4, Trick over
      three ranges). Both tables are transcribed in this repo's findings but
      not implemented. Lachaud & Mansour's PATO paper (in `reference/`) is the
      only published work Rabinovitch credits with doing it, and gives the
      formulation.
- [ ] **Coronene / PAH species in the gas model.** Equilibrium below ~1700 °C
      predicts up to **70 % coronene by mass** (Rabinovitch Fig. 4a), and no
      PICA thermal-response code — FIAT included — carries any large PAH
      species. This is a model-form gap on the gas side comparable to the
      Eq. (8) gap already documented on the solid side. Our Mutation++ species
      set stops at C₆H₆.

## 3. Ablation validation data still unmined

Three arcjet/flight datasets sit in `reference/` uncited. Each would widen the
I-V4 comparison beyond the seven Milos & Chen analysis cases.

- [ ] **Covington et al., *Performance of a Low Density Ablative Heat Shield
      Material*** — additional PICA arcjet performance data. **[blocks I-V4]**
- [ ] **McDougall et al., *Early Response of Ablative Materials to Arcjet
      Testing*** — the transient first seconds, which our quasi-steady surface
      energy balance handles worst and which nothing currently tests.
- [ ] **Balter-Peterson et al., arc jet facility description** — partially
      mined for model geometry (20.3 cm stagnation model), but the facility
      calibration chain is what would let the Zoby-derived effective radius be
      *confirmed* rather than inferred. Currently we report 9.0 cm recovered
      against a 10.15 cm physical radius and stop short of claiming agreement.

## 4. Aerothermal fidelity

- [x] ~~**Wall catalycity.**~~ **Done — and the framework could not express
      the case at all before.** Fay & Riddell publish *three* correlations;
      Anderson Eqs. (17.89)–(17.91) sets them side by side. We carried the
      two that differ only in Lewis exponent. The third — frozen boundary
      layer over a **noncatalytic** wall — has bracket `1 − h_D/h_0e`, which
      no exponent reaches, since matching it needs `Le^β = 0`. Catalycity is
      now a `WallCatalycity` enum rather than an exponent. The effect is
      large: **2.85× at h_D/h_0e = 0.6**, reproducing the "more than a factor
      of two" Anderson reports from Fig. 17.5, and PICA's char is not fully
      catalytic. Wired into II-V8 as three new checks.

- [ ] **Surface catalytic efficiency for PICA specifically.** The
      noncatalytic case now exists, but which of the three applies to a
      charred PICA surface is not established here — real surfaces are
      *partially* catalytic, between the two bounds, and the framework
      offers no way to sit between them. A finite-rate recombination
      coefficient would; Anderson's Fig. 17.5 abscissa (the recombination
      rate parameter) is the axis that interpolation would run along.

- [ ] **A real Navier-Stokes solution to check the correlations against.**
      Wright et al., *Data-Parallel Line Relaxation Method for the
      Navier-Stokes Equations* (in `reference/`, uncited) is the DPLR method
      paper. Our heating is Fay-Riddell plus Tauber-Sutton plus a Lees
      distribution — three correlations composed, with no CFD anywhere in the
      chain. **[blocks II-V8]**
- [ ] **Distributed heating over a real vehicle.** Hollis et al., *MSL
      Heatshield Aerothermodynamics: Design and Reconstruction*, and Bose
      et al., *Reconstruction of Aerothermal Environment and Heat Shield
      Response of MSL* (both uncited). The Lees distribution we use is a
      similarity solution; these give measured and CFD-reconstructed
      distributions over an actual flown heatshield.
- [ ] **Thermal cost of bank reversals.** The 0.43 km glide configuration
      needs **53 reversals at unlimited roll rate, 41 at 30 °/s** — roughly
      one every 20 s — and the vehicle rolls through high-heating attitudes
      each time. `passes.aerothermal` could price that; nothing currently
      does, so the accuracy/actuator trade is reported without its thermal
      side, and the tightest-accuracy configuration is the one whose thermal
      cost is least known.

## 5. Estimation against flight data

Our filter work (V5, V6) is verified against synthetic truth. Three MEDLI-family
papers in `reference/` are uncited and would replace that with flight data.

- [ ] **Karlgaard et al., MEDLI and MEDLI2 trajectory reconstruction**
      (two papers) — reconstruction algorithms *and* flight results, i.e. both
      a method to compare against and an answer to compare to.
- [ ] **Dutta et al., *Comparison of Statistical Estimation Techniques for
      MEDLI-like Data Sources*** — EKF against UKF against batch, on this
      exact problem class. Our adaptive filter is EKF-only, and nothing
      establishes that choice was right for this plant.

## 6. Guidance fidelity

- [ ] **Adjoint miss-distance analysis, not just direct integration.**
      `homing_miss` integrates Zarchan's Ch. 6 loop directly, which is exact
      for the deterministic linear model and verified against three of his
      stated results. It cannot do the *stochastic* sources — glint, receiver
      noise, radome refraction — which is precisely what the adjoint method
      exists for: **Ch. 3** (*Method of Adjoints and the Homing Loop*,
      including "Adjoints for Deterministic Systems") and **Ch. 4** (*Noise
      Analysis*, including "Adjoints for Noise-Driven Systems" and "Example
      of a Stochastic Adjoint"). **Ch. 5** (*Covariance Analysis and the
      Homing Loop*) gives the alternative route to the same answer. Those
      terms set a real seeker's noise floor, and we carry none of them.
- [ ] **Radome refraction and the parasitic loop.** A real hypersonic seeker
      behind a radome closes a parasitic feedback path that can destabilise
      the guidance loop entirely — Zarchan **Ch. 22** shows low airframe
      damping *increasing* sensitivity to radome slope (Fig. 22.6). Our loop
      is a clean single lag with no parasitic path.
- [ ] **Variable time of arrival targeting.** Both
      [`midcourse.py`](src/passes/guidance/midcourse.py) and
      [`bus.py`](src/passes/guidance/bus.py) implement fixed time of arrival
      only and refuse VTA explicitly. VTA is materially cheaper: the footprint
      work measured the cost of a 50 km downrange displacement falling **54 %
      (35.1 → 16.1 m/s) for 10 s of arrival slip**, while crossrange and
      radial gained *nothing* — their optima sit at zero slip. That is a
      quantified saving currently left on the table, available in one
      direction only.
- [ ] **Multi-revolution Lambert.** [`lambert.py`](src/passes/orbital/lambert.py)
      solves the zero-revolution case and refuses higher ones because each
      revolution admits two branches needing an explicit choice. Multi-rev is
      required for any parking arc beyond one revolution — which the budget
      already flags as its own failure mode. Zarchan **Ch. 13** (*Lambert
      Guidance*) covers the guidance application and would also give an
      independent check on the Izzo solver beyond the self-consistency and
      closed-form cases used now.

## 7. Plant fidelity

- [ ] **Earth rotation in the glide plant.** [`entry.py`](src/passes/guidance/entry.py)
      integrates over a *non-rotating* sphere, deliberately, to keep the V11
      verification attributable to the guidance law. Rotation is a few percent
      of crossrange over a long glide — small, but no longer negligible now
      that the crossrange dispersion is down to 0.43 km.
- [ ] **A boost phase that is simulated rather than stated.** `evaluate`
      charges boost a stated `boost_range` and `boost_delta_v`. Nothing
      integrates an ascent trajectory, so boost is the one leg whose range
      and ΔV are asserted rather than computed — and for suborbital
      architectures it is also the *slack leg* that decides closure, which
      means the closure verdict rests on the least-derived number in the
      budget. Zarchan **Ch. 12** (*Boosters*) gives staging and the gravity
      turn *and carries a worked numerical example* to verify against; the
      Minotaur and Falcon user guides in `reference/` give real staging
      masses and performance curves as a second check.
- [x] ~~**The ballistic entry leg is a hard-coded 300 km.**~~ **Done — and
      the "roughly geometry-fixed" defence was wrong by up to 6.5×.** The
      leg is now computed by
      [`ballistic_entry.py`](src/passes/flight/ballistic_entry.py) from the
      Allen–Eggers closed form (NACA TN 4047, via Gallais), giving
      `R = h_E/tan γ_E`. The old constant turns out to encode one specific
      entry angle — **21.8° from a 120 km interface** — not a geometric
      invariant: the range runs 448 km at 15° to 69 km at 60°. The default
      reproduces 300.02 km so no existing result moved silently.

      Three findings, two of them corrections to this implementation:

      * **A factor-of-two error in the peak-deceleration altitude**, caught
        by numerical cross-check rather than inspection. The critical
        density is `β sin γ/H`, not `2β sin γ/H`, which placed every peak
        exactly `H ln 2` = 4.85 km too low — a *constant* offset, and so
        invisible in any single case. It is also invisible in `a_max`, which
        is stationary there by construction, so the headline result was
        right while the altitude was not.
      * **The geometric range is an upper bound, not an estimate.** The real
        trajectory steepens under gravity, so it lands short. Measured
        against numerical integration: 1.8–3 % for a heavy vehicle at 45–60°,
        but 11–41 % at 20° and 22–73 % for a low-β capsule at any angle. The
        error table is published in the docstring and the floor was set from
        it at 15° rather than guessed.
      * **`a_max = V_E² sin γ/(2eH)` contains no vehicle property at all** —
        verified to 0.1 % across a 100× span in ballistic coefficient. A
        ballistic entry's peak load cannot be improved by changing the
        vehicle, only by arriving slower or shallower. Ballistic coefficient
        sets *where* the peak happens, not how hard.

      One limitation is now explicit rather than silent: the closed form
      neglects gravity, so for a low-β vehicle it decelerates toward zero
      where a real one settles at terminal velocity.
      `allen_eggers_applicable_at_impact` flags that, because the failure
      mode is returning a plausible small number rather than a NaN.

      Zarchan **Ch. 11** (*Strategic Considerations*) remains the route to
      a *flight-time* model for this leg, which is still `nan`.
- [ ] **Aerodynamics from the panel model instead of a fixed L/D.** The glide
      and cruise vehicles carry a constant lift-to-drag ratio.
      `passes.aerodynamics` already has the blended Newtonian/Prandtl-Meyer
      closure, and Paper II §3.2 notes that substituting it makes the coupling
      to deformed-shape incidence live — which is the entire premise of the
      aeroelastic half of the framework, currently unexercised by the guidance
      half.

## 8. Verification tasks still partial

- [ ] **I-V4** — recession is now compared against *measurement* (seven Milos
      & Chen conditions, 4 % median error against 27 % experimental scatter),
      but the criterion as written names a **FIAT reference case**, and the
      5 % it states is not meaningful against data scattering by 27 %. Either
      obtain FIAT reference output, or amend the criterion in Paper I §8 and
      say why. Leaving it silently unmet is the one option that is not
      acceptable.
- [ ] **II-V8** — the radiative half is verified against Tauber & Sutton's own
      table. The convective half needs published Fay-Riddell reference
      conditions with tabulated equilibrium-air properties (ρₑμₑ, ρ_wμ_w, h_D);
      the source presents its results as figures, not tables. Either transcribe
      the figures, or substitute a DPLR solution (§4 above) as the reference.

---

## 9. Newly available sources, not yet mined

A large reference drop landed after this list was written. These are the
items it makes reachable that are *not* already covered above, ordered by
what they would change.

- [ ] **Arcjet facility calibration from the primary guide.** Terrazas-Salinas
      et al., *Test Planning Guide for NASA Ames Research Center Arc Jet
      Complex and Range Complex*. This is the facility document the
      Balter-Peterson item above wanted: it should carry the calibration
      chain that would let the Zoby-derived **9.0 cm effective radius** be
      confirmed against the 10.15 cm physical radius rather than left as a
      consistent-but-uncorroborated inference. Supersedes the second half of
      §3's Balter-Peterson item.
- [ ] **Launch-vehicle dispersion as a Monte Carlo, not a tolerance.**
      Hanson & Beard, *Applying Monte Carlo Simulation to Launch Vehicle
      Design and Requirements Analysis*, and Pinier, *A New Aerodynamic Data
      Dispersion Method for Launch Vehicle Design*. The budget's boost
      dispersion is derived from IMU grade alone; these give the *method* for
      a full ascent dispersion, and Pinier specifically addresses aerodynamic
      database uncertainty, which we carry as nothing at all. Also the
      honest answer to why our Falcon 9 cross-check is order-of-magnitude
      only: user guides publish worst-case tolerances, not 1σ budgets.
- [ ] **Waypoint and no-fly-zone constrained glide.** Jorris, *Common Aero
      Vehicle Autonomous Reentry Trajectory Optimization Satisfying Waypoint
      and No-Fly Zone Constraints*. Our glide guidance flies to a target with
      a crossrange deadband; it cannot express an intermediate waypoint or a
      keep-out region, both of which are first-order constraints on a real
      HGV trajectory and would interact directly with the bank-reversal
      schedule the accuracy work is built on.
- [ ] **Boost-phase guidance error analysis.** Siouris, *Missile Guidance
      and Control Systems*, complementing the Zarchan Ch. 12 item in §7.
- [ ] **Stage separation.** Pamadi et al. and Couchman. Boost is currently
      one leg with a stated ΔV; staging events are where a real dispersion
      budget picks up contributions we model as zero.
- [ ] **The reentry-dynamics texts as cross-checks rather than sources.**
      Regan, Gallais, Mooij, Loh, Hankey, Tewari, Teofilatto. Gallais is
      already cited for Allen-Eggers. These overlap heavily with each other
      and with what is implemented; the value is in finding where they
      *disagree* with our closed forms, not in adding citations.

---

## Deliberately not on this list

Recorded so they are not re-proposed as oversights:

- **Multi-revolution parking arcs beyond one revolution** — the coast model is
  built for a fractional orbit; J₂ secular drift over many revolutions needs
  higher zonal harmonics and luni-solar terms that Paper II §7.1 explicitly
  scopes out.
- **Plane-change deorbit** — a plane change at orbital speed costs km/s against
  tens of m/s for the entire deorbit burn. Azimuth freedom is bought at
  insertion or not at all; this is a conclusion, not a gap.
- **An optimiser over architecture parameters** — `evaluate` is deliberately an
  objective function. Sweeping glide range, perigee depth or fuel fraction is
  the caller's job, and burying the trade inside an optimiser would hide the
  thing the budget exists to expose.
