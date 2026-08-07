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

## 9. Systems, guidance and dispersion — from the newer sources

A large reference drop landed after §1–§8 were written. This section is what
a full survey of it found. Items are placed here rather than folded into the
sections above when they open something the earlier sections had no way to
ask for.

Three of the drop's sources have already been used and are recorded in place:
Marschall & Milos closed the permeability question in §2, Anderson opened the
catalycity item in §4, and Gallais supplied Allen–Eggers for §7.

### 9.1 Siouris, *Missile Guidance and Control Systems*

The single highest-yield unread source in the directory, because four
separate open items above are chapters in it.

- [x] ~~**Ballistic error coefficients (§6.4.3), and the hit equation they
      attach to (§6.4.2).**~~ **Partly done — the crossrange coefficient is
      implemented and it contains a result worth the whole exercise.**
      [`ballistic_errors.py`](src/passes/guidance/ballistic_errors.py)
      carries Siouris Eq. (6.116) and Figs. 6.16–6.17.

      **A lateral burnout position error does not map to impact
      proportionally — it is suppressed by cos ψ over the free-flight range
      angle, and vanishes *exactly* at ψ = 90°.** A quarter of the Earth's
      circumference, about 10,000 km. The reason is geometric rather than
      approximate: two great circles displaced perpendicular at one point
      meet again a quarter turn later, because every pair of great circles
      intersects. Past 90° the sensitivity grows again. Verified against
      direct spherical vector construction to machine precision over range
      angles 0–150° and offsets 0.01–5°, with the null exact for a 5°
      offset.

      A fixed crossrange budget therefore cannot be right: the same 1 km
      burnout offset becomes **985 m at a 10° range angle, 707 m at 45°,
      174 m at 80°, and zero at 90°**.

      Wired into `evaluate` as a range-dependent term — it has to be applied
      there rather than in `DISPERSION_SOURCES`, since cos ψ is a property
      of the mission, not the vehicle. Measured effect at the aviation grade
      the budget assumes: **0.17 m out of an 876 m CEP**, i.e. real but
      negligible. At **tactical grade it would be 571 m at ψ = 30° against
      the glide leg's 400 m** — it would set the crossrange budget outright,
      and the cos ψ null would become a first-order design lever rather than
      a rounding correction. Also confirmed: for architectures carrying a
      midcourse correction the term has no effect at all, because a reset
      nulls everything boost contributed.

      **One implementation finding.** Siouris prints the relation as
      `cos δC = sin²ψ + cos²ψ cos δχ`. Coded literally that is *numerically
      useless at the offsets a dispersion budget cares about*: a 1 m lateral
      error is 1.6×10⁻⁷ rad, where `cos δχ` differs from 1 by a few times
      machine epsilon and `arccos` has unbounded derivative. It loses ~4
      significant figures at 10⁻⁷ rad and returns **exactly zero** at 10⁻⁸ —
      silently reporting a perfectly guided vehicle. The algebraically
      identical half-angle form `δC = 2 arcsin(|cos ψ| sin(δχ/2))` is exact
      throughout and makes the sensitivity manifest. Pinned as its own test,
      because the printed form is the one a future simplification would
      reach for.

      Also implemented: `velocity_error_at_impact` (δV·t_ff, Fig. 6.16) —
      which independently confirms the *form* of the budget's existing
      crossrange mapping, previously justified only by our own propagator —
      and `launch_position_error`, the survey-error rotation. The latter has
      **no range-dependent suppression at all**, which is the structural
      reason a 10 m CEP is a survey problem rather than a guidance one: a
      survey error displaces the whole trajectory rather than one point on
      it, so nothing converges it.

- [x] ~~**The remaining in-plane coefficients (§6.4.3.1).**~~ **Done, from
      Regan §5.5 rather than Siouris — and checking them against an
      independent conic solution found errors in two of the three printed
      equations.** A MinerU transcription of Regan rendered the equations as
      LaTeX, which the raw scan could not. `ballistic_errors.py` now carries
      the full in-plane set:

      * **Eq. (5.36), `∂R/∂V` — exact.** Verified against finite differences
        of a Keplerian conic solution sharing no algebra with it, to five
        figures. It also reproduces Regan's own worked example end to end:
        at θ = 90° the optimum γ* = 22.5° needs **7195 m/s** and gives
        **6.05 km of range error per m/s**, against his stated 7195 and
        "approximately 6". A **1 m/s error in 7195 is a 6 km miss** — which
        is why boost cutoff dominates a ballistic error budget.
      * **γ\* = π/4 − θ/4, and `∂R/∂γ` is exactly zero there, for every
        range angle.** The minimum-energy trajectory is also the one
        indifferent to boost pitch error — two design pressures pointing the
        same way, which is rare enough to be worth stating.
      * **Eq. (5.39), `∂R/∂γ` — magnitude exact, sign inverted.** Regan
        prints it, and describes it in prose, as negative below γ*. That
        cannot hold at a range *maximum*: below the optimum, lofting further
        must lengthen the range, and finite differences agree. Flipped here
        and flagged rather than silently adopted, since it may be a
        convention (error measured as "short") rather than a mistake.
      * **Eq. (5.41), `∂R/∂h` — a dropped bracket.** Printed as
        `2cot γ − cos(γ+θ)/cos γ`, which matches the numerics *only* where
        `cos(γ+θ) = 0`. The form `cot γ [2 − cos(γ+θ)/cos γ]` matches to
        five decimals at every angle pair tried — exactly what losing an
        outer bracket would do. Not academic: at Regan's own worked point
        the printed form gives **5.24 against a true 5.83**, an 11 %
        understatement, widening to 12.3 against 16.8 at θ = 150°, γ = 10°.

      **One discrepancy left unresolved and recorded as a test.** Regan's
      worked Eq. (5.40) states −5.28 km/mrad at θ = 75°, γ = 15°, while his
      own Eq. (5.39) at those angles gives **11.89** — a factor of 2.25. The
      numerics support the latter. Nothing in the text settles where the
      2.25 comes from.

      **Both printed forms have since been confirmed against the book
      itself**, so neither is a transcription artefact — and the (5.41) case
      became demonstrable rather than merely numerical. Setting
      `δV = δγ = 0` in Regan's own Eq. (5.33) gives `∂R/∂h = (1+A)/C`; the
      impact equation makes `A = 1 − cos(γ+θ)/cos γ`, and `C = tan γ`
      identically on the impact locus (verified to six decimals). So
      Eq. (5.33) implies exactly the corrected form, and Eq. (5.41) is
      **internally inconsistent with it** — the slip is substituting
      `C = tan γ` into the first term of the numerator but not the second.

      The (5.39) sign could not be diagnosed the same way: the bracket
      multiplying `δγ` in Eq. (5.33) does not reproduce the numerics under
      any reading recoverable from the text, including at γ* where it must
      vanish and does not. So that one is recorded as a disagreement rather
      than located. What *is* settled is which side is right: scanning
      `θ_i(γ)` at fixed speed puts the maximum at `π/4 − θ/4` to within
      **0.01°**, so γ* genuinely maximises range and the derivative below it
      is positive.
- [x] ~~**REP, DEP and their relationship to CEP (§5.7.3).**~~ **Done — and
      it turned out to supply a 126-point verification, not just a ratio.**
      Siouris §5.7.3 gives the classical relations, but the section also
      carries **Table 5.2**: :math:`K` such that
      :math:`P(R \le K\sigma_L) = P`, over 21 aspect ratios from degenerate
      to circular and six probability levels. That spans the entire domain
      [`dispersion.py`](src/passes/systems/dispersion.py) covers, and it is
      an entirely independent computation of an integral that previously
      self-checked only against its own circular limit.

      **Our exact elliptical integral reproduces every entry to within one
      unit in the last printed place**, with 122 of 126 inside ideal
      four-decimal rounding. Both endpoints land on their closed forms —
      1.1774 = √(2 ln 2) in the circular column; 0.6745 and 1.9600 (the
      one-dimensional probable error and 1.96σ) in the degenerate one.

      The four exceptions are **the source's rounding, not ours**, and are
      identifiable rather than merely tolerated: in each case the exact
      value rounds to a different last digit than the one printed (2.4478
      against 2.4477468, and three like it). All four are high by
      5.1–5.3×10⁻⁵; across all 126 the signed deviations split 69 positive
      to 57 negative with mean +6.4×10⁻⁶, so this is four last-place
      roundings and not a bias in our integral. They are pinned as their own
      test, so a real drift would move the other 122 and be distinguishable.

      Also added, with their errors *measured* rather than asserted:
      `probable_error` (REP/DEP), `cep_from_probable_errors` (Eq. 5.17) and
      `cep_small_ratio` (Eq. 5.13). Two things the source does not state
      fell out. Siouris says Eq. (5.17) holds "even when REP and DEP differ
      by a factor as much as two" — true, 1.5 % there — but **the error is
      not monotone**: it peaks near 2 % at 3:1, crosses zero near 5:1, then
      diverges, so anyone extrapolating by watching the error shrink would
      be walking into the divergence. The function refuses past 5:1 for that
      reason. And Eq. (5.13), published for σ_S/σ_L < 0.28, is **better than
      advertised** — within 0.1 % out to at least 0.35. We keep the
      published bound anyway.
- [ ] **Correlated velocity and velocity-to-be-gained (§6.5), i.e.
      Q-guidance.** The classical ballistic-missile guidance law, and the
      thing that actually decides when boost cutoff occurs. Our boost leg has
      no guidance law at all — it is charged a stated ΔV. §6.5.4 also covers
      control during the atmospheric phase.
- [ ] **TERCOM and cruise navigation error analysis (§7.3, §7.4).**
      [`cruise.py`](src/passes/guidance/cruise.py) computes Bréguet range and
      nothing else; a cruise vehicle in the budget therefore has range but no
      accuracy, and the dispersion column carries it as unmodelled. TERCOM is
      how a real cruise missile bounds inertial drift over a long flight, and
      §7.4.5 (terrain roughness characteristics) is what decides whether it
      works over a given route. Without it the cruise architectures cannot be
      given a CEP on the same footing as the ballistic ones.
- [ ] **Effect of Earth rotation on ballistic flight (§6.4.4).** Complements
      the glide-plant rotation item in §7 with the ballistic-leg equivalent.

### 9.2 Regan, *Re-Entry Vehicle Dynamics*

An AIAA Education Series text whose chapter list maps almost one-to-one onto
this backlog's gaps.

- [x] ~~**Time of flight (Ch. V §5.4).**~~ **Done, but not from Regan — the
      leg turned out to be atmospheric, not Keplerian.** The ballistic
      `duration` was `nan` in every budget row. Regan §5.4 gives the
      *Keplerian* free-flight time, which is the wrong leg: the budget's
      ballistic entry is entry-interface-to-impact, i.e. the atmospheric
      descent. That follows from the Allen–Eggers profile already
      implemented in §7, by one quadrature on `1/V` given
      `dh/dt = −V sin γ`.

      `BallisticEntry.descent_time` matches numerical integration of the
      full point-mass equations (gravity off, the like-for-like comparison)
      to **better than 0.1 %** across entry angles 30–60° and ballistic
      coefficients 7500–20 000 kg/m²; the gravity-on trajectory is 2–11 %
      quicker, in the expected direction. At the default 21.8° the leg is
      **52.0 s** over its 300 km.

      One case is refused rather than answered: the integrand goes as
      `1/V`, so a **low-β vehicle that decelerates to terminal velocity
      above the ground gives a divergent integral** dominated by the regime
      Allen–Eggers omits. Those still report `nan` — but now with a note
      saying which vehicle and why, where before every architecture got a
      silent `nan`.

      Regan's OCR is too degraded to transcribe his Eqs. (5.27b)/(5.28)
      reliably, which is worth recording: the scan mangles the equations
      into unusable fragments even though the surrounding prose is legible.

- [ ] **Impact equation and Keplerian error analysis (Ch. V §§5.3, 5.5).**
      Still open, and distinct from the item above: §5.3/§5.5 cover the
      *exo-atmospheric* arc, which the budget currently charges through the
      deorbit transfer rather than as a ballistic free-flight leg. §5.5 is
      also a second, independent treatment of the error analysis Siouris
      gives in §6.4.3 — worth having both, since agreement between two texts
      is worth more here than either alone.
- [x] ~~**Angular motion during re-entry (Ch. XIII).**~~ **Roll dynamics
      and resonance done; asymmetry-driven trim still open.**
      [`roll_resonance.py`](src/passes/dynamics/roll_resonance.py) carries
      Regan's roll-rate history (Eqs. 13.8–13.12) and the resonance
      condition `ω_nα = √(1 − I_x/I_y)·p` (Eq. 13.79).

      The structural result is that **the pitch frequency is not monotone
      through an entry.** It goes as `V√ρ`, and on the way down ρ rises
      while V falls, so it climbs, peaks, and falls again — meaning a
      vehicle at fixed roll rate crosses resonance **twice, once, or not at
      all**, decided by its ballistic coefficient.

      Verified against Regan's worked Fig. 13.10 case (V_E = 5 km/s, 75°
      entry, P_s = 3.73×10⁻³ m/kg, 18 rad/s), driven by the independently
      verified Allen–Eggers profile so that two modules are exercised
      against one published result: **one crossing at 37.0 km** for a 6×10⁴
      Pa ballistic factor, **two at 36.5 and 10.2 km** for 6×10³ Pa, against
      his one, and 37 km / ~11 km. Counts right in both, altitudes within
      1–3 km on an exponential atmosphere rather than his tabulated one.

      *A source inconsistency worth recording:* he gives first resonance as
      "about 34 km" for the heavy vehicle then "as before, at 37 km" for the
      light one, though these are the same quantity and his own argument
      implies they barely differ. We compute 37.0 and 36.5 — consistent with
      the "as before", not with the 34.

      `trim_amplification` is flagged as **ours, not his**: Regan states only
      that the response is singular undamped and "considerably" amplified at
      realistic damping. The standard second-order form turns that adjective
      into a number — a factor of **10 at 5 % damping** — and is labelled a
      quasi-steady bound, since the real amplification depends on how fast
      the vehicle sweeps through.

- [ ] **Configuration asymmetries as a dispersion source (Ch. X §10.1,
      Ch. XIII Fig. 13.2).** The resonance machinery now exists but nothing
      *drives* it: `C_l0` and the trim angle are caller inputs, and the
      budget still charges the ballistic leg a stated dispersion. Closing it
      needs the c.g.-offset roll torque of Fig. 13.2 and a mapping from an
      amplified trim excursion to an impact displacement — Regan notes the
      trajectory effect persists *after* resonance, and that first and second
      resonance trade force against remaining time of flight in opposite
      directions.
- [ ] **Deviation of the vertical (Ch. III §3.3).** Directly closes the
      gravity-anomaly item in §1, which was listed there as having *no*
      source in the repository.
- [ ] **Boost trajectories (Ch. XI).** A third route to the §7 boost item,
      alongside Zarchan Ch. 12 and Siouris; §11.2 does the non-rotating-Earth
      case, which is the right first step given our glide plant already makes
      that simplification deliberately.
- [ ] **Ring laser gyros and pendulous accelerometers (App. B).** I listed
      "instrument datasheets" as a source this repository lacked, in order to
      check Groves' grade bands against real hardware. Regan gives the
      instrument physics instead, which is better for our purpose: it says
      *why* the bands sit where they do rather than asserting one vendor's
      numbers.

### 9.3 Jorris, *Common Aero Vehicle Autonomous Reentry Trajectory
Optimization* — a benchmark we can run as published

- [ ] **Waypoint and no-fly-zone constrained glide.** Our glide guidance flies
      to a target with a range-scheduled crossrange deadband. It cannot
      express an intermediate waypoint or a keep-out region, and both are
      first-order constraints on a real HGV trajectory — they interact
      directly with the bank-reversal schedule that the whole accuracy result
      rests on. Jorris formulates both as interior-point and path-inequality
      constraints and solves the 3-D CAV problem by **pseudospectral
      collocation with NLP**, which is infrastructure this repository already
      has (`passes.spectral`, `passes.ultraspherical`, and the Elnagar and
      Huntington citations already in the bibliography).
- [ ] **His Table 2 is a fully specified, reproducible test case**, and it is
      stated in exactly the universal geodetic format
      [`geodesy.py`](src/passes/geodesy.py) already accepts:

      ```text
      Initial       N 28°35.286′  W 80°40.194′
      Waypoint 1    N 34° 2.810′  W 27°18.430′
      No-Fly Zone 1 N 20°15.513′  W  3°27.588′   960 nmi
      Waypoint 2    N 33°13.298′  E 41°41.266′
      No-Fly Zone 2 N 55°43.849′  E 58°33.688′  1500 nmi
      Target        N 31°36.653′  E 65°42.016′
      h0 = 122 km,  V0 = 7.3152 km/s,  gamma0 = -1.5 deg
      ```

      This is the first published end-to-end HGV trajectory problem we could
      run against a stated answer, and it exercises the arbitrary-launch /
      arbitrary-target parametrisation that was built and then only ever
      tested on cases we invented. Worth doing even before the constraints
      are implemented, as an unconstrained baseline.
- [ ] **The heating path constraint as an *active* constraint.** Jorris sets
      his heating limit deliberately below the unconstrained optimum so the
      solution must ride the boundary. Our glide never has heating as an
      active constraint — `passes.aerothermal` prices the trajectory after
      the fact rather than shaping it. This is also the missing half of the
      §4 "thermal cost of bank reversals" item.

### 9.4 Dispersion methodology — Hanson & Beard, Pinier

- [ ] **How many Monte Carlo samples the accuracy results actually need.**
      The glide dispersion progression that drives every accuracy conclusion
      in this project — 34.0 km down to 0.43 km — was measured on a **40-case**
      Monte Carlo. Hanson & Beard (NASA/TP-2010-216447) §4.1 gives the
      standard treatment: the standard error scales as 1/√N, so 40 samples
      put roughly **11 % uncertainty on σ itself**, and therefore on every CEP
      and R95 derived from it. That is not obviously fatal — we take the
      *parametric* route (fit σ, then scale by the containment ratio) rather
      than reading an empirical tail, and §1.4 of the same document is
      explicitly about that choice — but the sampling uncertainty is
      currently unstated, and a 0.43 km result quoted to two significant
      figures implies a precision 40 samples do not support. **Quantify it or
      raise N.**
- [ ] **Epistemic versus aleatory uncertainty (§2.1).** `DISPERSION_SOURCES`
      mixes both without distinguishing them: IMU grade is an epistemic
      choice (which unit was procured), while flight-day execution error is
      aleatory. Hanson & Beard treat these differently on purpose, and §2.4.1
      shows the consequence — some contributors are modelled as uniform
      precisely *because* it is unknown where in the range they sit.
- [ ] **A confirmation worth recording, not a gap.** §4.1 notes that the
      two-dimensional 3σ containment for a Gaussian is **98.9 %**, not the
      one-dimensional 99.73 %, and warns against reading sigma levels as
      percentages in more than one dimension. That is exactly the error
      [`dispersion.py`](src/passes/systems/dispersion.py) was built to avoid,
      and its exact elliptical integral gets it right. Add a test that pins
      the 98.9 % figure against an authority rather than against our own
      integral.

- [x] ~~**Two modules computing CEP by different methods.**~~ **Done — found
      while wiring the Siouris verification, and it was a real
      inconsistency.** [`systems/dispersion.py`](src/passes/systems/dispersion.py)
      had the exact elliptical integral;
      [`batch/dispersion.py`](src/passes/batch/dispersion.py) — the module
      that actually summarises Monte Carlo footprints — was still using
      Paper I's classical route: the linear approximation
      `0.5887(σ₁+σ₂)` inside a stated validity band, with a **sample-median
      fallback** outside it. So the repository computed its headline
      statistic two different ways depending on which entry point was used.

      The batch path is now exact at every aspect ratio, with no branch and
      no fallback. Measured cost of the old route: the linear form errs by
      up to **2 % inside its own validity band** (peaking near aspect 0.3),
      and the median fallback carries sampling noise the integral does not
      have. On the elongated footprint V7 exercises, the linear formula
      would have claimed a CEP **9.7 % low**.

      The bootstrap could not afford a root-find per resample, so it
      interpolates a precomputed CEP-over-σ₁ curve — valid because the
      containment radius is homogeneous of degree one in the sigmas, so the
      entire dependence is one univariate function of aspect ratio.
      Interpolation error is under 10⁻⁶ relative, three orders of magnitude
      inside any realistic bootstrap's own sampling error, and that bound is
      itself a test.

      `cep_method` is retained as a *label* rather than a branch: it still
      reports whether the footprint sits inside the band Eq. (6.4) was
      stated for, because that is what makes a result comparable with
      literature computed the classical way. **This supersedes Paper I
      §6 Eq. (6.3), Eq. (6.4) and Remark 10**, which are updated to say so
      rather than quietly left describing code that no longer exists.
- [ ] **Aerodynamic database dispersion (Pinier).** We carry no aerodynamic
      uncertainty at all. Pinier's point is that the traditional approach —
      biasing a whole nominal curve up or down inside its uncertainty band —
      is unphysically benign, and that dispersing the coefficient *and its
      derivatives* under non-arbitrary constraints stresses the control model
      far harder. On the Ares I project this changed predicted roll control
      authority materially. Directly relevant to the constant-L/D item in §7:
      the moment a real aerodynamic database enters, its uncertainty has to
      enter with it.

### 9.5 Arcjet facility — Terrazas-Salinas

- [ ] **Confirm, or retire, the recovered effective radius.** The Ames Test
      Planning Guide documents the standard calorimeter and model geometries,
      and they are not all hemispheres: the catalogue includes flat-faced
      cylinders with a 0.953 cm corner radius, and **iso-q probes whose nose
      radius equals their base diameter**. Our Zoby inversion recovers
      **R_eff = 9.0 cm** consistently across 15 of 19 conditions, and we
      compare it to a "10.15 cm physical radius" — but if the models were
      flat-faced or iso-q rather than hemispherical, then that 10.15 cm is a
      geometric half-width and the recovered value is an aerodynamic
      effective radius, and **the two are not the same quantity**. In that
      case the near-agreement we stopped short of claiming was never
      meaningful either way. Establish which shape the Milos & Chen models
      were before doing anything else with this number.
- [ ] **The CFD companion papers, named precisely.** The guide's bibliography
      identifies the work that would settle it: Gökçen, Chen, Skokova &
      Milos, *Computational Analysis of Arc-Jet Stagnation Tests Including
      Ablation and Shape Change*, JTHT **24**(4), 2010, and Stewart, Gökçen &
      Chen, *Characterization of Hypersonic Flows in the AHF and IHF NASA
      Ames Arc-Jet Facilities*, AIAA 2009-4237. The first is a CFD treatment
      of the very cases we reconstruct. Neither is held here; both are
      ordinary open-literature AIAA papers.
- [ ] **Why the scatter is what it is.** The guide attributes run-to-run
      calibration scatter at fixed facility conditions to unavoidable model
      misalignment moving the stagnation point off centre — which is a
      physical account of the 27 % experimental spread that I-V4 currently
      reports against as a bare number.

### 9.6 Ablation validation data, now surveyed

Three of the datasets listed as "unmined" in §3 have now been read far
enough to say what they are worth and what the catch is.

- [ ] **Covington et al. is PICA, and it is Stardust.** Additional arcjet
      recession and in-depth temperature data at nominal peak Stardust
      heating and at 37 % above it. **The catch, and it is a real one:** the
      authors *tuned* thermophysical properties iteratively to match their
      own in-depth temperatures. Adopting their properties and then declaring
      agreement would be circular. Their *measurements* against our
      independent property set is a legitimate comparison; their derived
      properties are not an independent check on ours. Also worth chasing:
      they report "consistent temperature rise deviations that are not
      accurately modelled by the computer code" — a stated model-form failure
      in a code of the same family as ours, which our solver should either
      reproduce or explain.
- [ ] **McDougall et al. targets exactly our weakest regime.** They model the
      first seconds before significant pyrolysis and find conduction and
      radiation dominant there, with good agreement pre-pyrolysis. Our
      quasi-steady surface energy balance is least defensible in precisely
      that window and nothing currently tests it. Their framing is Bayesian
      inference of properties from multiple thermocouples, which is a second
      use: it is the same inverse problem `passes.estimation` solves for
      trajectories.
- [ ] **Omidy et al. — moisture** remains as stated in §2, now confirmed to
      be the Lachaud/Martin/Mansour line of work and therefore consistent in
      formulation with the PATO reference already cited there.

### 9.7 A large-N robustness set: the JCAT catalogues

- [x] ~~**70,000 real orbits, sitting unused in
      [`reference/cats/`](reference/cats/).**~~ **Done — swept, and the
      kernel is exact on all of them.** McDowell's General Catalog of
      Artificial Space Objects gives perigee, apogee and inclination for
      every catalogued object back to Sputnik (S00001: 214 × 938 km,
      65.10°). **69,099** of 69,452 rows are physically usable; the rest are
      suborbital, hyperbolic or garbled and are dropped rather than guessed
      at. Results, now pinned as tests in
      [`test_orbital.py`](tests/test_orbital.py) (skipped when the catalogue
      is absent):

      * **483,693 (orbit, latitude) pairs**: `approach_azimuth` accepts
        exactly the reachable set and refuses exactly the rest — zero
        accepted-but-unreachable, zero refused-but-reachable, against the
        exact bound `|φ| ≤ min(i, π−i)`.
      * `cos i = sin A cos φ` holds to **1.1×10⁻¹⁶** over the catalogue.
      * `azimuth_envelope`'s nan pattern coincides with the unreachable set
        exactly, at four latitudes across all 69,099 orbits.
      * The inclination histogram peaks where physics requires: **51.5°**
        (Baikonur/ISS), **62.5–65.5°** and **82.5°** (Plesetsk),
        **97.5–98.5°** (sun-synchronous), 43.5°/53.5° (Starlink shells).
        This checks `cos i = sin A cos φ` at A = 90° against what was
        actually flown, not against its own algebra.

      **One real finding, and it was a documentation gap rather than a bug.**
      The sweep initially flagged thousands of "out of range" azimuths for
      retrograde orbits. `approach_azimuth` returns a *signed* heading — the
      ascending branch is an `arcsin`, so a sun-synchronous orbit at
      i = 98° crosses the equator at about −8°, genuinely west of north —
      and the existing tests already pin `retro == -direct` deliberately.
      The convention was right and the returned range was simply never
      stated; a caller expecting a `[0, 360)` compass bearing would have
      read the sign as an error. Now documented.

#### 9.7b Strategic and tactical trajectory selection

- [x] ~~**Depressed vs lofted trajectory choice.**~~ **Done.**
      [`lofting.py`](src/passes/guidance/lofting.py) carries Regan §5.3–5.5.
      Rearranging his Eq. (5.23a) puts the whole γ-dependence inside
      `sin(2γ + θ/2)`, from which two exact results follow: the minimum-energy
      angle is `γ* = π/4 − θ/4`, and **every** achievable speed above the
      minimum is reached at *two* burnout angles placed symmetrically about
      it, `γ_over = 2γ* − γ_under`. No root-finding, where the standard
      treatment reads the pair off a plot.

      Verified against Regan's Table 5.3 output: at a 75° range angle, 10°
      and 42.5° share a burnout speed of 7238.03 m/s and give flight times
      of 1271.18 s and 2577.69 s. Solving Kepler independently gives
      **1270.35 s and 2576.85 s** (0.07 % and 0.03 %), and the conjugate
      relation reproduces his pair exactly.

      **The three considerations do not agree, and one of them inverts a
      tempting reading of the source.** Lofting roughly *doubles* flight
      time, so depression is what buys short warning. `∂R/∂γ` is exactly
      zero at γ* and non-zero at both conjugates, so minimum energy is the
      trajectory indifferent to boost pitch error. And `∂R/∂V` is **not
      symmetric**: 3128 m per m/s lofted, 4579 at the optimum, **9110
      depressed**. Regan's conclusion — over-lofted beats both — holds, but
      the depressed solution is *twice as sensitive as the optimum*, not
      less, which matters because depression is exactly what a
      short-warning launch wants.

      His stated mechanism is also the minor term. He attributes the
      advantage to the higher burnout speed, since `∂R/∂V` carries `1/V`;
      that penalty is **5.2 %** across the pair, while `cot γ` falls by
      **5.2×**. The ordering is essentially all cotangent.

### 9.7c FOBS, rigorously

- [!] **The fractional profile's altitude was the one quantity in it that
      no dynamics produced.** The parking arc was flown at a *constant*
      altitude — a prescription, not an orbit — and the boost that reached
      it was a stated ramp of altitude against arc. Both are now solved.

      The boost is a **gravity turn**: speed grows as
      :math:`v_{bo}\\tau`, the flight-path angle pitches over from vertical
      as :math:`\\gamma_{bo} + (\\pi/2-\\gamma_{bo})(1-\\tau)^{2.5}`, and
      altitude and downrange are integrals of it. What it replaced was not
      merely unrealistic but **internally inconsistent**: the ramp's own
      shape implied a burnout speed of **2,666 m/s** where the profile it
      fed needed 7,830 — a factor of three — and it left the pad at a
      flight-path angle of **37 degrees**. Neither error touched a warning
      number, because altitude and ground track were being *told* what to
      be, which is exactly the failure mode this whole section exists to
      catch.

      The three boost parameters are not independent, and the model now
      says so: at a fixed burnout speed the path length is fixed by the
      speed law, and only its split between up and downrange is free. So
      **downrange is derived**, not assumed — 662 km for a 180 s burn to
      orbital speed, which is the right order for an ICBM-class boost — and
      the burnout angle falls out at 2-4 degrees, which is the check that
      the boost and the parking arc describe the same vehicle. A burn too
      long for the parking altitude is refused rather than returned as a
      climb that is not one.

      The parking arc is now an **ellipse**, 170 x 250 km by default, timed
      by Kepler. Altitude varies over the arc as it must; the previous flat
      150 km was both arbitrary and unphysical. 170 km sits in the band open
      sources quote for the R-36O, and the notebook sweeps it from 120 to
      500 km rather than resting on the number.

- [!] **A fractional profile necessarily enters an order of magnitude
      shallower than a ballistic one**, and this is a cost the concept is
      rarely charged for. The deorbit burn is now a real
      :math:`\\Delta v` — 182 m/s, computed from the *vector* difference
      between the parking ellipse's velocity at the burn point and the
      descent conic's apogee velocity, so the radial component is not
      dropped — and the entry flight-path angle falls out of the descent
      conic:

      | profile | entry angle | descent ground track |
      | --- | --- | --- |
      | fractional, long way | **-2.5 deg** | 8,159 km |
      | fractional, direct | **-1.7 deg** | ~8,000 km |
      | ballistic, depressed | -12.6 deg | — |
      | ballistic, minimum energy | -24.5 deg | — |

      Deorbiting from orbital speed cannot buy a steep entry without
      removing kilometres per second, so the descent conic spans a quarter
      of the planet and **the vehicle is committed 13.6 minutes and 8,000 km
      before impact**. That bounds how much surprise a parking arc can buy,
      and it implies a long, shallow, high-heat-load entry the ballistic
      RV does not fly. Neither consequence is priced here — accuracy and
      terminal vulnerability need an engagement model — but the geometry
      forcing them is now explicit rather than absent.

- [!] **Every warning figure before this entry was computed against a
      network containing *both sides'* sensors.** Recorded because it
      corrupted every number below it.

      `coverage` reduces a network to its **earliest** detection. That is
      the right composition for warning — one site is enough to raise an
      alarm — but it means the list has to belong to one side.
      `EARLY_WARNING_SITES` is a *catalogue*: it carries Okno and
      Krasnoyarsk, two Russian early-warning radars, alongside nineteen
      western ones and one non-aligned South African site. Their own notes
      said as much ("Included for geographic completeness, not as a US/NATO
      system"); nothing enforced it.

      For a Eurasian launch the consequence is concrete: three of four
      profiles from Dombarovskiy were first "detected" by **Okno, 900 km
      from the pad, at T+0.8 min**. For the reference package's Siberian
      launch site, a few hundred kilometres from Krasnoyarsk, the
      distortion was total — every profile was picked up on the way up and
      appeared to concede its entire flight time.

      The correction changes *which sensor sets the answer*, not merely the
      number: minimum-energy ballistic goes 29.0 min / Okno → **27.3 min /
      Globus II/III**, and the direct fractional profile 19.1 / Okno →
      **16.0 / Globus II/III**. `RadarSite.coalition` now records the side,
      `radar.network(...)` selects one, and `warning_comparison` defaults
      to `network("western")` — the only network these analyses are about.
      Cape Town is excluded with the Russian pair: its own entry records
      that it is not integrated into any western early-warning network, so
      its returns are not on this picture either.

- [!] **The reversed bearing buys no measurable warning, and costs a great
      deal.** `fobs_trajectory(..., direction="short")` is the control the
      comparison had been missing: the same parking altitude and the same
      orbital-insertion energy, flown down the *minor* arc on the direct
      bearing. Whatever it concedes is what altitude alone buys.

      Against the 20-sensor defender network, mid-latitude Eurasian launch
      to a US east-coast aimpoint:

      | profile | warning | detecting sites | V_bo | flight |
      | --- | --- | --- | --- | --- |
      | fractional, direct | **16.0 min** | 1 | 7,830 m/s | 21.2 min |
      | ballistic, depressed | 20.8 min | 8 | 7,176 m/s | 24.0 min |
      | ballistic, minimum energy | 27.3 min | 9 | 6,986 m/s | 29.8 min |
      | fractional, long way | 49.7 min | 2 | 7,830 m/s | 71.8 min |

      Two things fall out. **The small detecting set comes from altitude,
      not from the bearing** — the direct profile is seen by *fewer* sites
      than the long way. And **the long way costs 34 minutes of warning**,
      because warning runs from first detection and it flies three times as
      long.

      That does not make the concept pointless; it locates its value
      somewhere this framework cannot currently reach. Arriving from an
      azimuth the defence's interceptors and battle management are not
      oriented along is a real claim, and it is *not* the same as radar
      horizon, which is all that is modelled. Pricing it needs an
      engagement model, not a geometry one.

- [!] **The warning-time advantage inverted when the sensor network grew,
      and one station is responsible.** Recorded here because it supersedes
      a headline result reported earlier in this backlog. Figures below are
      restated against the defender network.

      Against the original **13-site** network the fractional profile
      removed ~24 minutes of warning. That network had **no
      southern-hemisphere coverage**. With the defender network now at
      **20 sensors** including Exmouth and Cape Town, the same scenario
      gives the fractional profile **47.6 min of warning against the
      ballistic arc's 27.3** — a 20.2-minute *penalty*, not an advantage.

      Isolated by ablation: dropping **Exmouth alone** restores 3.5 min and
      the full +23.8-minute advantage exactly. Dropping Cape Town changes
      nothing. Nothing else in the model differs.

      The mechanism separates two properties this analysis had been
      conflating. Warning is measured from *first* detection to impact, and
      the fractional profile flies 73 min against 30:

      * **Azimuth denial holds decisively** — 2 detecting sites against 9.
      * **Warning advantage does not** — one early detection means being
        tracked through three quarters of a much longer flight.

      So the advantage was never robust; it was contingent on a coverage
      gap that a single station closes. **Any warning figure this framework
      produces must be quoted with the network it was computed against.**
      The earlier "24 minutes removed" figure in §9.7c below and in the
      README is true only of the 13-site network and is now labelled as
      such.


- [x] ~~**The fractional-orbit condition itself.**~~ **Done.**
      `fractional_insertion` in [`fobs.py`](src/passes/orbital/fobs.py)
      classifies an insertion by whether its conic perigee falls at or below
      the entry interface — the property that makes a profile fractional
      rather than orbital, and makes the deorbit burn a *targeting*
      manoeuvre instead of the thing that brings the vehicle down.

      The result worth having: **half a percent below circular is already
      enough.** At a 180 km insertion, a 0.5 % speed deficit puts perigee at
      about 50 km, inside the atmosphere, and the vehicle cannot complete a
      revolution. Fractional insertion is a small perturbation on an orbital
      one, which is why the distinction is as much about intent as about
      energy. The coast to entry shortens monotonically with deficit —
      0.29 revolutions at 0.5 %, 0.13 at 2 %, 0.03 at 20 % — and every case
      reenters inside one revolution, which is the claim the name makes.

      *A bug the smoke test caught:* the entry crossing is on the
      **descending** branch, at `2π − arccos(...)`, not the principal
      arccos. Taking the ascending branch sends the coast the long way round
      and reports 0.87 revolutions where the answer is 0.13 — plausible
      enough that only the monotonicity check exposed it.

- [x] ~~**Warning time: why anyone flies a fractional profile.**~~ **Done.**
      [`warning.py`](src/passes/orbital/warning.py) carries the exact
      horizon geometry, `λ_max = arccos[(R_E/r)cos ε] − ε`, derived in one
      rearrangement and checked against numerically solving the elevation
      relation (agreement to 10⁻¹² rad at three altitudes and three masks).

      The quantitative core of the FOBS argument: a minimum-energy ICBM
      apogee near 1300 km is visible to a zero-mask site out to **3764 km**;
      a 150 km fractional parking altitude only to **1369 km** — a factor of
      **2.75**. The low profile is not stealthy in any electromagnetic
      sense, it is simply below the horizon for most of its flight. And a
      realistic 3° mask costs the defender **21 % of the FOBS visibility
      radius against 9 % of the ICBM one**: the mask hurts most exactly
      where the defence can least afford it.

      `detection_window` reduces a sampled trajectory to first-detection
      time, warning time and visible fraction, and is labelled an **upper
      bound** — geometry only, with no power-aperture, cross-section or
      track-quality requirement, so a real defence does worse.

- [x] ~~**Warning time end to end, against a real trajectory pair.**~~
      ~~**Azimuth denial as distinct from warning time.**~~ **Both done.**
      [`radar.py`](src/passes/orbital/radar.py) carries a network of
      publicly documented early-warning sites, and
      [`scenario.py`](src/passes/orbital/scenario.py) composes
      `lofting`, `fobs`, `warning` and `radar` into the comparison the whole
      concept turns on: both profiles between the same two points, past the
      same sensors.

      For a mid-latitude Eurasian launch against a US east-coast aimpoint:

      | | ballistic | fractional |
      |---|---|---|
      | range angle | 81.7° | 278.3° |
      | apogee | 1308 km | 150 km |
      | flight time | 30.3 min | 67.5 min |
      | burnout speed | 7034 m/s | 7818 m/s |
      | **sites detecting** | **7** | **1** |
      | **warning** | **27.9 min** | **4.3 min** |

      The trade priced rather than asserted, **against the 13-site network
      of the time** (see the superseding note at the head of this section):
      **24 minutes of warning removed,
      paid for with 37 minutes of flight time and 780 m/s of burnout speed.**
      And the azimuth-denial half falls straight out of the same run — the
      ballistic arc is seen by seven widely separated sites, the fractional
      one by a single site, because it arrives from the reversed bearing over
      a hemisphere the network does not face.

      `coverage` composes sites by *earliest detection*, which is right for
      warning (one site raises the alarm) and explicitly wrong for track
      quality or discrimination, which need favourable multi-site geometry
      and are not modelled.

- [x] ~~**Earth rotation under the fractional parking arc.**~~ **Done — and
      it exposed a targeting gap, not just a plotting one.** Turning the
      ground beneath a fixed inertial plane made the fractional profile
      **miss by 1463 km**, because a trajectory must be aimed at where the
      target *will be*. `leading_aimpoint` solves that fixed point — the lead
      depends on the flight time, which depends on the range to the lead
      point — and both profiles now carry it: **7.4 deg of lead for the
      half-hour ballistic arc, 17.3 deg (about 1500 km) for the 69-minute
      fractional one.** Both now arrive within 5 km of the aimpoint.

      The prediction made when this was deferred is now **measured rather
      than asserted**: warning time moves by under a minute while the number
      of detecting sites changes (7 to 6), so rotation changes *which* site
      sees it more than *when*, and the earlier comparison stands.

      One earlier test had to be corrected rather than merely re-run. It
      asserted the two profiles' range angles sum exactly to 2*pi, which
      holds only on a non-rotating Earth: with leads applied each profile
      aims at its own point and the sum drifts by the difference of the
      leads. The exact identity was an overreach and now holds only where it
      should.

      The orbit plane is still fixed in inertial space, which is right to the
      extent that it does not precess inside one revolution.

- [ ] **Boost-phase infrared detection.** The single largest omission in the
      warning analysis, and it cuts against the fractional profile: a real
      boost is detectable from space long before any ground radar sees the
      vehicle, and a longer, higher-energy boost is *more* detectable. The
      notebook says so in its limitations; the model carries nothing.

## 9.8 Boost, staging and propulsion dispersion

- [ ] **Stage separation as a dispersion source.** Pamadi et al. and Couchman
      (perturbation techniques). Boost is one leg with a stated ΔV; staging
      events contribute tip-off rates and separation impulses that we model
      as exactly zero.
- [ ] **Propellant bias and Isp uncertainty.** The 1960 *Ballistic Missile and
      Space Technology* symposium volume is mostly out of scope for this
      project — nuclear-electric power, ion and plasma thrusters, radiographic
      QA of solid motors — but two papers bear on boost dispersion inputs:
      MacPherson on propellant bias for stages lacking propellant-utilisation
      systems, and the vacuum-Isp precision-determination paper. Residual
      propellant and Isp uncertainty are two of the standard contributors to
      injection ΔV dispersion, and we carry neither.

### 9.9 Surveyed and deliberately parked

Recorded so the survey does not have to be repeated.

- **ERIS (SDIO, 1987)** is an *environmental assessment*, not a technical
  description of the interceptor. Like the NASA Routine Payloads assessment
  already in `reference/`, its technical content is launch azimuths, impact
  and debris footprints, and propellant inventories. That is real data for
  range-safety and launch-corridor constraints, which this framework does
  not model and has not claimed to. Low priority, non-zero value.
- **The reentry-dynamics texts beyond Regan** — Mooij, Loh, Hankey,
  LeGalley, Marrow, Tewari, Teofilatto. These overlap heavily with each
  other, with Regan and Gallais, and with what is already implemented. The
  value in them is finding where they *disagree* with our closed forms, not
  in adding citations to agreement. Not worth mining serially; worth
  consulting when a specific closed form is in dispute.
- **Antares and the FAA reliability guide** are procurement and regulatory
  documents. The Antares guide would extend the published-vehicle
  cross-validation set by one more launcher, which is marginal given
  Minotaur and Falcon are already in it.
- **The slug-calorimeter ASTM standard** documents the measurement behind
  the arcjet heat-flux column. Worth reading only if the calibration chain
  in §9.5 turns out to be the thing blocking the effective-radius question.

### 9.10 Interactive analysis

- [x] ~~**A composed, runnable scenario notebook.**~~ **Done** —
      [`fobs-warning-analysis.ipynb`](notebooks/fobs-warning-analysis.ipynb),
      25 cells, executes end to end with 6 figures and no errors. Configurable
      launch site and aimpoint at the top; ground tracks with horizon circles,
      altitude profiles with first-detection marks, per-site detection bars, a
      burnout-angle sweep across four objectives, a parking-altitude sweep
      against four mask assumptions, and the fractional-insertion table.

      Its §5 makes the point the whole backlog has been circling: **the
      objectives disagree.** Minimum energy sits at γ*, which is also where
      `∂R/∂γ` vanishes; depression buys warning-time compression; lofting buys
      insensitivity to burnout speed error. There is no trajectory that wins
      on every count, and the notebook prices the choice rather than making
      it.

      §6 sweeps the **assumed** radar mask precisely because it is the least
      defensible number in the model, and the spread it produces is the
      honest uncertainty band on every warning figure quoted.

- [ ] **A basemap for the ground-track figure.** Plotted on a bare
      equirectangular grid: `cartopy` needs a C++ toolchain not present here,
      and a coastline would in any case imply more precision than a spherical
      trajectory model has. Worth revisiting only alongside the Earth-rotation
      item above, since that is what would make the track worth locating
      precisely.

### 9.11 Configuration as a first-class artefact

- [x] ~~**A portable launch-package format.**~~ **Done.**
      [`package.py`](src/passes/systems/package.py) defines a versioned,
      unit-annotated scenario document — TOML for authoring (read by stdlib
      `tomllib`), JSON for interchange, both round-tripping through one
      dataclass with a test asserting they agree.
      [`packages/fobs-reference.toml`](packages/fobs-reference.toml) is the
      worked example and drives the notebook.

      Four design rules, each earning its keep:

      * **Units in the key names** — `latitude_deg`, `parking_altitude_m`.
        The error class this repository keeps getting bitten by is a number
        of the right magnitude in the wrong unit; a bare `latitude = 0.9` is
        unfalsifiable where `latitude_deg = 0.9` is obviously wrong.
      * **Degrees on disk, radians in memory**, converted once at the
        boundary.
      * **Schema version required** — an unversioned file is refused, so a
        format change fails loudly rather than silently reinterpreting old
        data.
      * **Closed vocabularies checked against the code** — `architecture`
        and `imu_grade` must name real entries, with the valid options
        listed on error.

      **Unknown keys are refused**, and that rule caught a real bug in the
      first example package written for this format. In TOML a bare key
      written *after* a `[table]` header belongs to that table, not to the
      document root — so `arrival_time_s` and `objectives` placed at the end
      of the file silently became `vehicle.arrival_time_s` and
      `vehicle.objectives`, meaning nothing, and the loader used the
      defaults without complaint. The summary quietly showed one objective
      where the file listed two. The error message now names the trap.

      A package records only what was *chosen*; nothing derivable from those
      choices is stored, so a package can never disagree with the code about
      a computed quantity.

- [x] ~~**Packages for the other architectures.**~~ **Done.** The launch-package
      format supports all eleven named architectures via the `architecture`
      key, validated against `NAMED_ARCHITECTURES`. Packages for
      `fractional-orbital-single`, `ballistic-single`, and `boost-glide` are
      now exercised end to end.

- [x] ~~**Sensor definitions in the package.**~~ **Done.** `[[sensors]]` entries
      are fully parsed and round-trip, and the campaign format inherits
      campaign-level sensors into child launch packages. A campaign that
      carries its own threat picture makes a scenario fully self-contained.

- [x] ~~**Campaigns: multiple launch sites.**~~ **Done.** Schema version
      `passes.launch-package/2` introduces `[[launches]]`, a list of complete
      launch packages sharing campaign-level sensors and objectives.
      `packages/mid-latitude-campaign.toml` is the reference example: two
      launch sites (Dombarovskiy, Uzhur) targeting US east-coast aimpoints.
      `Campaign.mission_requests()` produces one request per launch for the
      budget evaluator, and `[launches]` entries reject top-level single-launch
      keys to prevent mixed-mode ambiguity.

### 9.11b The animations must be the simulation, not a model of it

The fractional-orbital and ballistic animations were drawn from
:mod:`passes.orbital.scenario` — a **geometry** model of stitched Keplerian
conics with a prescribed pitch program. That is the same defect
`SimulationHistory` was built to remove, one level up: a picture faithful to
the planner is not a picture of the physics engine, and the planner carries
no drag, no J2, no attitude, no mass depletion and no gravity loss.

- [x] ~~**Powered flight in the coupled right-hand side.**~~ **Done** —
      [`propulsion.py`](src/passes/flight/propulsion.py). The engine was
      structurally unable to fly a boost or a burn: `out[layout.mass] = 0.0`,
      no thrust term. It now carries constant-thrust arcs with mass
      depletion and three steering laws.

      The steering law had to be got right, and the first attempt was
      instructive. A *commanded pitch* program parameterised on burn
      fraction puts the thrust 17 degrees above horizontal at half the burn,
      by which point the vehicle is still at 10 km; it then accelerates
      horizontally through dense air, dynamic pressure reaches **400 kPa**
      against a real max-Q near 30, drag exceeds thrust, and the vehicle
      reaches 13 km and 1,050 m/s in 245 s. A **gravity turn** — vertical,
      one kick, then prograde — cannot fail that way, because after the kick
      the thrust follows the velocity and the trajectory turns only as fast
      as gravity turns it. Measured max-Q is then 23-28 kPa, and at an
      excessive kick the vehicle turns into the atmosphere, which is the
      correct failure mode.

- [x] ~~**Mass-consistent drag.**~~ **Done.** A fixed ballistic coefficient
      is exact for an unpowered vehicle and wrong for one that burns fifteen
      times its burnout mass. `FlightConfiguration.drag_area` switches drag
      to `q C_D A / m` with the mass the integrator is carrying; the default
      is `None`, so every existing result is unchanged.

- [x] ~~**A ground event.**~~ **Done.** The simulator integrated a fixed
      duration and kept going, reaching 45 km *below* the surface in 300 s.
      Impact is now a terminal `solve_ivp` event.

- [x] ~~**Multi-segment missions.**~~ **Done** —
      [`mission.py`](src/passes/flight/mission.py). Legs are integrated in
      sequence and concatenated into one `FlightResult`, with one `Phase`
      per leg. Thrust switching is a real discontinuity; splitting at it is
      faster than gating inside one solve and makes the boundaries exact.

- [ ] **Closed-loop targeting that converges.** Started, and not finished —
      [`profiles.py`](src/passes/flight/profiles.py). The structure is
      right (every residual is evaluated by integrating the real system) and
      the individual solves work, but the assembled ascent does not reach a
      *low* parking orbit. The obstruction is understood and is vehicle
      sizing, not code: with the current lumped stage the boost burns out at
      113 km and 3,463 m/s, so at apogee the vehicle has 3,075 m/s against
      the 7,755 a 250 km circular orbit needs. The "circularisation" is then
      a 4.7 km/s burn, which over its 200 s runs thousands of kilometres of
      arc and raises apogee instead of perigee — measured 121 x 906 km for a
      170 x 250 km request.

      What it needs is a vehicle that reaches near-orbital speed *under
      thrust* rather than coasting to apogee at a third of it: a larger mass
      ratio, and probably explicit staging so the upper stage is not
      dragging the first stage's dry mass. Until that converges, the
      notebooks still animate the planner, and that is stated in them rather
      than glossed.

- [!] **Two Earth radii are in use and they differ by 7,128 m.**
      `EARTH.radius` is the **equatorial** radius, 6,378,137 m; the geodesy
      layer's `WGS84_MEAN_RADIUS` is the mean, 6,371,009 m. The flight
      simulator's altitudes are therefore above the equatorial radius while
      every orbital and coverage calculation uses the mean. Found by writing
      a ground-event test against the wrong one — it failed by exactly
      7,128 m.

      For a visualisation the consequence is direct: a `FlightResult` drawn
      on a globe of mean radius sits 7 km too high. The animator takes
      `body_radius` explicitly and the notebook passes `sim.gravity.radius`,
      so the current pictures are right, but the framework should not offer
      two answers to "how big is the Earth".

### 9.11c Vehicle geometry from a mesh

- [x] ~~**Load the outer mould line and measure it.**~~ **Done** —
      [`geometry/mesh.py`](src/passes/geometry/mesh.py). `reference/model.stl`
      is a 6,922-facet binary STL of a **35.00 m x 3.73 m body of
      revolution** — R-36 class, which is the vehicle the fractional-orbital
      analysis is about. Shape scalars that were *stipulated* are now
      *measured*: wetted area 411.06 m2, frontal area 10.875 m2, length
      35.00 m, and the axial station profile.

      Two of the measurements needed a better method than the obvious one,
      and both failures are recorded in the code:

      * **Frontal area** by the analytic projection — half the sum of
        `|n.a|A` — gave **26.29 m2** against a true 10.93. The mesh contains
        two **internal bulkheads**, full-diameter disks at two stations that
        are not outer surface at all, and the formula counts each in full.
        Rasterising the silhouette is immune and converges to 10.875 m2,
        which is 0.48 % under `pi r_max^2` — exactly the 0.41 % deficit of
        the inscribed 40-gon the mesh actually uses. That agreement is the
        check.
      * **Nose radius** is not a property of this vehicle. The spherical-cap
        fit `r^2 = 2 R d` returns 0.35 m over a 0.05 m window, 0.50 m over
        0.20 m and 0.58 m over 1.00 m, because the nose is an **ogive**: a
        power-law fit gives `r ~ d^0.59` against 0.5 for a sphere and 1.0
        for a cone. An automatic window-sweep "plateau detector" was written
        and then deleted — it found a stable run at 0.35 m that was stable
        only because several small windows catch the *same two vertex
        rings*. `nose_exponent()` reports the shape and `nose_radius()`
        takes an explicit window and documents that it is a bound. This
        matters because Sutton-Graves assumes a hemispherical stagnation
        region and goes as `R_n^{-1/2}`.

- [x] ~~**Feed the panel aerodynamics.**~~ **Done.** A triangle *is* a
      panel, so `VehicleMesh.panel_model()` hands the mesh to the existing,
      tested `PanelModel` losslessly. Real coefficients replace a hand-set
      `drag_area`: **C_D = 1.97 at Mach 2 falling to 0.70 at Mach 20**,
      asymptoting as Newtonian theory requires.

      The axis convention had to be established empirically and was got
      backwards first. `PanelModel.velocity_direction` is the direction the
      flow *travels*, so a windward panel's normal points along **-x** — the
      opposite of where the nose points. Verified with a flat plate: normal
      along -x collects 1,831 N of a 1,000 Pa stream at Mach 10, normal
      along +x collects 14 N. Flying the vehicle nose-along-+x gave an
      axial force coefficient of **-5.7**, negative and an order of
      magnitude too large, entirely silently.

- [ ] **Exclude internal faces from the aero integration.** Quantified:
      the two bulkheads add **+0.89 to C_D at Mach 2** and +0.009 at Mach
      20 — negligible where Newtonian shading dominates, decisive at low
      supersonic. The general fix is a visibility test (cast a ray from
      each face centroid along its normal and drop faces that hit the mesh
      again), which is 48 M ray-triangle tests here and a one-off cost.
      Until then the low-Mach coefficients are contaminated.

- [ ] **Sub-stage decomposition.** The mesh is a *single* connected
      component, so stages cannot be separated topologically. Geometry
      offers a strong hypothesis: four rings stand 0.131 m proud of the
      1.734 m body at axial stations **z = 8.23, 1.63, -8.05 and -17.90**,
      which is where separation hardware sits, and two internal bulkheads
      lie at z = 9.61 and 8.10. `raised_bands()` finds the rings and
      `section()` cuts by whole faces. What geometry cannot supply is which
      rings are separation planes and which are raceways, or the propellant
      mass in each stage — and mass is what a staged ascent needs, not
      shape.

### 9.12 Visualization as a first-class consumer

The animation layer is a *presentation* layer and is held to a lower bar
than the physics kernels — no manufactured solutions, no independent
solvers, no published references. That is a reasonable prioritisation for a
codebase whose claim is numerical, but the gap is real, and two defects
found while building the animations were of a kind the physics tests could
never have caught, because the notebook carried its **own** trajectory
model reconstructed from sub-points and altitudes.

- [x] ~~**A canonical time history, so a picture cannot disagree with the
      physics.**~~ **Done** — [`history.py`](src/passes/viz/history.py).
      `SimulationHistory` is the single authoritative run record;
      `from_flight_result` carries the full coupled state (position,
      velocity, attitude quaternion, heat flux, recession, dynamic
      pressure) and `from_trajectory` the lighter orbital scenarios.
      Sampling interpolates linearly in position and by **slerp** in
      attitude — componentwise quaternion interpolation leaves the unit
      sphere and both shrinks and shears the rotation — and holds the
      endpoints rather than extrapolating past the end of the physics.

      The design rule worth keeping: a scenario trajectory is a point mass
      and has **no attitude**, so `has_attitude` reports `False` rather
      than substituting an identity quaternion. Drawing an oriented vehicle
      from a rotation that was never computed is exactly the failure the
      object exists to prevent.

- [x] ~~**Move the notebook helpers into `passes.viz.scene`.**~~ **Done** —
      [`scene.py`](src/passes/viz/scene.py). `geodetic_to_cartesian`,
      `starfield`, `ease`, `draw_track`, `draw_marker`, `globe_plate` and
      the chase rig are now pure functions over history samples and a
      camera, and `notebooks/animation.ipynb` defines none of them.

      Two of them changed meaning in the move, and both changes are fixes:

      * `ChaseRig` takes its heading from the sampled **velocity**, not
        from a finite difference over a fixed six-sample look-ahead. The
        old form made the framing a function of the *sampling density* —
        the same trajectory at 400 and 900 samples was shot differently.
      * The trail is bounded in **seconds**, not in samples. A Keplerian
        arc sampled in true anomaly bunches near apogee, so a
        sample-counted trail changes physical length over a run.

- [x] ~~**`TrajectoryAnimator` façade**~~ **Done** —
      [`animator.py`](src/passes/viz/animator.py). `frame_at(t)` returns a
      `Frame` carrying the *state and camera it drew from*, which is what
      makes the layer testable: `tests/test_viz.py` asserts the vehicle
      appears at the projection of the true state through the frame's own
      camera, rather than asserting a picture was produced.

      `render_sequence` lays frames on a uniform grid **in time**, whose
      endpoints are the history's endpoints. The truncation that cut every
      animation at 86 % of its flight is now impossible by construction
      rather than by vigilance.

- [x] ~~**An oriented vehicle glyph.**~~ **Done** — `glyph_world` places
      body-frame polylines by the direction cosine matrix of the integrated
      quaternion, and is a pure function so the orientation is checked
      arithmetically (nose at `position + scale * C.T @ NOSE_AXIS`, and
      every pairwise distance preserved under rotation) rather than by eye.

      Two caveats stated at the call site. The glyph is **not to scale** —
      a 15 m vehicle at a 500 km stand-off subtends 30 nrad, about
      1/200 000 of a pixel, so a true-scale glyph is an empty frame. And
      the present flight model integrates attitude torque-free while drag
      acts along the relative velocity, so **attitude does not feed back
      into the force**: the glyph shows the rotational state that was
      integrated, not an angle of attack the trajectory responded to.

- [x] ~~**Sensor overlays driven by the coverage result.**~~ **Done** —
      `draw_sites` colours each site idle / detecting-now / has-detected
      from the same `CoverageResult` that produced the warning number, and
      `horizon_ring` draws the visibility circle at the **vehicle's**
      radius rather than on the ground (a surface footprint is about three
      times too small at 150 km, understating every sensor by exactly the
      amount the fractional argument turns on). The animator refuses
      coverage whose clock lies outside the history — the one way this
      layer could show detections from a different flight.

      This needed a small extension to the physics layer: `DetectionWindow`
      gained `last_detection_time`, without which "detecting now" and "has
      detected" are indistinguishable.

- [x] ~~**Thermal and ablation colouring.**~~ **Done** — `color_by` names
      an `extras` series and the trail is drawn as a `LineCollection`
      through it, with a HUD read-out in units a reader can hold rather
      than raw SI. A faithful sequential map is black at its low end, so a
      cold trail over a night ocean was invisible; a backing stroke keeps
      the geometry legible while the colour still carries the value, so
      dark reads as *cold* rather than as *absent*.

- [x] ~~**Performance and output.**~~ **Done.**

      * **GPU path.** `render` takes the same `backend` argument as the
        batched integrator, since every per-pixel expression already used
        the array API NumPy and CuPy share. Measured **344 ms → 22 ms** for
        a 1280x720 frame, agreeing with the CPU path to 1e-11 in a colour
        channel and 2e-13 relative in depth. The texture is uploaded once
        via `to_device`; `render` refuses a host texture on a device
        backend rather than silently re-uploading 200 MB per frame.
      * **Static-geometry caching.** The starfield is cached per frame size
        and returned read-only, and the texture is loaded once per
        animator.
      * **H.264 export.** `video_writer` dispatches on the output
        extension. Measured on the same 130-frame 1280x720 run: **1.13 MB
        of MP4 against 14.4 MB of GIF**, and GIF's 256-colour palette bands
        the terminator and limb glow that this renderer exists to draw.

- [ ] **Interactive rates.** 22 ms of render plus ~150 ms of Matplotlib
      per frame — the bottleneck is now the *overlay* path, not the ray
      tracer. Drawing the overlays into the image buffer directly, rather
      than through Matplotlib artists, is the next order of magnitude.
- [ ] **A real vehicle mesh.** The glyph is a wireframe cone with fins
      because the renderer has no self-occlusion. A shaded solid needs a
      depth buffer for scene geometry, not only for the sphere.

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
