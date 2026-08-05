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
      The correct source is Marschall & Milos, *Gas Permeability of Rigid
      Fibrous Refractory Insulations* (cited by Lachaud & Mansour), which
      this repository does not hold — worth obtaining only if PICA turns out
      to sit below 10⁻¹² m².
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
- [ ] **The ballistic entry leg is a hard-coded 300 km.**
      `_BALLISTIC_ENTRY_RANGE` in [`budget.py`](src/passes/systems/budget.py)
      is a stated constant standing in for entry-interface-to-impact range,
      justified as "roughly geometry-fixed for a steep entry" — which is true
      enough to be plausible and not true enough to be derived. Zarchan
      **Ch. 11** (*Strategic Considerations*) gives closed-form ballistic
      solutions, a hit equation and flight time, which would replace the
      constant with a function of entry angle and speed. It is currently the
      only leg in the budget whose range is neither computed nor sourced.
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
