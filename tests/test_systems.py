"""System-level composition: which phase sequences form admissible systems."""

from pathlib import Path

import numpy as np
import pytest

from passes.geodesy import GeodeticPosition, great_circle_range
from passes.guidance import CruiseVehicle, EntryVehicle
from passes.guidance.ballistic_errors import crossrange_offset_sensitivity
from passes.guidance.inertial import IMU_GRADES, injection_error
from passes.systems import (
    CEP_OVER_SIGMA,
    DISPERSION_SOURCES,
    NAMED_ARCHITECTURES,
    R95_OVER_SIGMA,
    SIOURIS_TABLE_5_2,
    SIOURIS_TABLE_5_2_PROBABILITIES,
    SIOURIS_TABLE_5_2_RATIOS,
    Architecture,
    MissionRequest,
    Payload,
    Phase,
    PhaseRegime,
    accuracy_statistics,
    cep_from_probable_errors,
    cep_small_ratio,
    containment_probability,
    containment_radius,
    containment_ratio,
    describe,
    enumerate_architectures,
    evaluate,
    probable_error,
    validate,
)
from passes.systems.package import (
    Campaign,
    LaunchPackage,
    PackageError,
    load_campaign,
    load_package,
)


class TestPhaseTaxonomy:
    def test_every_phase_has_a_regime(self):
        for phase in Phase:
            assert isinstance(phase.regime, PhaseRegime)

    def test_boost_is_the_only_ascent_phase(self):
        ascent = [p for p in Phase if p.regime is PhaseRegime.ASCENT]
        assert ascent == [Phase.BOOST]

    def test_payload_capabilities_are_consistent(self):
        assert Payload.SINGLE_RV.is_multiple is False
        assert Payload.MULTIPLE_RV.is_multiple is True
        assert Payload.MULTIPLE_GLIDER.is_multiple is True
        assert Payload.MIXED.is_multiple is True
        assert Payload.MIXED.is_mixed is True
        assert Payload.MULTIPLE_RV.is_mixed is False
        assert Payload.SINGLE_RV.is_lifting is False
        assert Payload.GLIDER.is_lifting is True
        assert Payload.CRUISER.is_propelled is True
        assert Payload.GLIDER.is_propelled is False
        # A propelled vehicle is necessarily a lifting one here.
        for payload in Payload:
            if payload.is_propelled:
                assert payload.is_lifting


class TestNamedArchitectures:
    def test_all_named_architectures_are_admissible(self):
        """They are validated at import, so this mostly guards against a
        future edit weakening a rule without noticing what it admits."""
        for name, architecture in NAMED_ARCHITECTURES.items():
            validate(architecture.phases, architecture.payload)
            assert architecture.name == name

    def test_the_expected_reference_systems_are_covered(self):
        """The permutations the framework claims to span: ballistic and
        fractional-orbital, single and multiple bodies, ballistic entry and
        glide, plus powered cruise."""
        expected = {
            "ballistic-single",
            "ballistic-multiple",
            "boost-glide",
            "boost-glide-multiple",
            "fractional-orbital-single",
            "fractional-orbital-glide",
            "fractional-orbital-multiple",
            "fractional-orbital-multiple-glide",
            "powered-cruise",
            "ballistic-mixed",
            "fractional-orbital-mixed",
        }
        assert set(NAMED_ARCHITECTURES) == expected

    def test_orbital_flag_and_terminal_regime_read_correctly(self):
        fobs = NAMED_ARCHITECTURES["fractional-orbital-glide"]
        assert fobs.is_orbital
        assert fobs.terminal_regime is Phase.GLIDE
        ballistic = NAMED_ARCHITECTURES["ballistic-single"]
        assert not ballistic.is_orbital
        assert ballistic.terminal_regime is Phase.BALLISTIC

    def test_description_is_informative(self):
        text = describe(NAMED_ARCHITECTURES["fractional-orbital-multiple"])
        assert "fractional-orbital" in text
        assert "multiple bodies" in text

    def test_architecture_without_a_terminal_regime_reports_it(self):
        stub = Architecture(phases=(Phase.BOOST,), payload=Payload.SINGLE_RV)
        with pytest.raises(ValueError, match="no terminal regime"):
            _ = stub.terminal_regime


class TestCompositionRules:
    """Each rule is about physics or information, so each gets its own test
    and its own reason rather than a shared 'invalid' assertion."""

    def test_deorbit_presupposes_an_orbit(self):
        with pytest.raises(ValueError, match="presupposes an orbit"):
            validate((Phase.BOOST, Phase.DEORBIT, Phase.BALLISTIC), Payload.SINGLE_RV)

    def test_a_parking_orbit_must_be_left(self):
        with pytest.raises(ValueError, match="must be left by a deorbit"):
            validate((Phase.BOOST, Phase.PARKING, Phase.BALLISTIC), Payload.SINGLE_RV)

    def test_deorbit_cannot_precede_its_own_parking_orbit(self):
        with pytest.raises(ValueError, match="cannot precede"):
            validate(
                (Phase.BOOST, Phase.DEORBIT, Phase.PARKING, Phase.BALLISTIC),
                Payload.SINGLE_RV,
            )

    def test_a_uniform_payload_has_exactly_one_terminal_regime(self):
        """Glide, cruise and ballistic entry are alternative descriptions of
        the whole atmospheric arc, not successive stages — for a payload
        whose bodies all fly the same arc. A mixed payload is the exception
        and is tested separately."""
        with pytest.raises(ValueError, match="has one terminal regime"):
            validate((Phase.BOOST, Phase.GLIDE, Phase.BALLISTIC), Payload.GLIDER)
        with pytest.raises(ValueError, match="needs a terminal regime"):
            validate((Phase.BOOST, Phase.TERMINAL), Payload.SINGLE_RV)

    def test_exoatmospheric_phases_cannot_follow_entry(self):
        with pytest.raises(ValueError, match="cannot follow"):
            validate(
                (Phase.BOOST, Phase.BALLISTIC, Phase.MIDCOURSE),
                Payload.SINGLE_RV,
            )
        with pytest.raises(ValueError, match="cannot follow"):
            validate(
                (Phase.BOOST, Phase.GLIDE, Phase.DISPENSE),
                Payload.MULTIPLE_GLIDER,
            )

    def test_terminal_homing_is_last(self):
        with pytest.raises(ValueError, match="last phase"):
            validate(
                (Phase.BOOST, Phase.TERMINAL, Phase.BALLISTIC),
                Payload.SINGLE_RV,
            )

    def test_a_single_body_has_nothing_to_dispense(self):
        with pytest.raises(ValueError, match="nothing to dispense"):
            validate(
                (Phase.BOOST, Phase.DISPENSE, Phase.BALLISTIC),
                Payload.SINGLE_RV,
            )

    def test_multiple_bodies_must_actually_separate(self):
        """Without a dispensing phase the bodies never separate, so the
        payload is effectively single and calling it multiple is a fiction."""
        with pytest.raises(ValueError, match="must dispense"):
            validate((Phase.BOOST, Phase.BALLISTIC), Payload.MULTIPLE_RV)

    def test_a_non_lifting_body_cannot_glide(self):
        with pytest.raises(ValueError, match="no useful lift"):
            validate((Phase.BOOST, Phase.GLIDE), Payload.SINGLE_RV)

    def test_an_unpropelled_body_cannot_cruise(self):
        with pytest.raises(ValueError, match="no propulsion"):
            validate((Phase.BOOST, Phase.CRUISE), Payload.GLIDER)

    def test_a_cruiser_cannot_be_staged_through_orbit(self):
        """An airbreather needs atmosphere, and orbit is where there is
        none. This is the rule that most looks like doctrine and is in fact
        physics."""
        with pytest.raises(ValueError, match="air-breathing cruiser"):
            validate(
                (Phase.BOOST, Phase.PARKING, Phase.DEORBIT, Phase.CRUISE),
                Payload.CRUISER,
            )

    def test_sequences_must_begin_with_boost_and_not_repeat(self):
        with pytest.raises(ValueError, match="begins with boost"):
            validate((Phase.BALLISTIC,), Payload.SINGLE_RV)
        with pytest.raises(ValueError, match="must not repeat"):
            validate(
                (Phase.BOOST, Phase.MIDCOURSE, Phase.MIDCOURSE, Phase.BALLISTIC),
                Payload.SINGLE_RV,
            )
        with pytest.raises(ValueError, match="at least one phase"):
            validate((), Payload.SINGLE_RV)


class TestEnumeration:
    def test_enumeration_is_closed_under_the_rules(self):
        """Anything the enumerator produces must independently validate.
        This is the property that makes the enumeration meaningful: it is
        derived from the rules rather than curated alongside them."""
        for architecture in enumerate_architectures():
            validate(architecture.phases, architecture.payload)

    def test_every_named_architecture_is_enumerated(self):
        """The named list is a selection from the admissible set, not a
        parallel one. If a named case were inadmissible this would catch
        it, and if the enumerator were over-constrained it would too."""
        produced = {(a.phases, a.payload) for a in enumerate_architectures()}
        for name, architecture in NAMED_ARCHITECTURES.items():
            assert (architecture.phases, architecture.payload) in produced, name

    def test_enumeration_can_be_restricted_to_one_payload(self):
        gliders = enumerate_architectures(Payload.GLIDER)
        assert gliders
        assert all(a.payload is Payload.GLIDER for a in gliders)
        assert all(Phase.DISPENSE not in a.phases for a in gliders)
        assert all(a.terminal_regime is Phase.GLIDE for a in gliders)

    def test_a_single_reentry_vehicle_admits_exactly_twelve(self):
        """A closed count, worth pinning because it is derivable by hand:
        ballistic entry is forced, then midcourse, parking+deorbit and
        terminal homing are independently optional, with the three-phase
        ordering of an optional midcourse against parking+deorbit giving
        the extra factor."""
        assert len(enumerate_architectures(Payload.SINGLE_RV)) == 12

    def test_mixed_payloads_fly_both_arcs_concurrently(self):
        """A bus can dispense glide vehicles and ballistic reentry vehicles
        on the same pass, so terminal regime stops being a property of the
        architecture and becomes a property of each body."""
        mixed = NAMED_ARCHITECTURES["fractional-orbital-mixed"]
        assert mixed.payload is Payload.MIXED
        assert set(mixed.terminal_regimes) == {Phase.GLIDE, Phase.BALLISTIC}
        # Asking for *the* regime is a category error and must say so
        # rather than silently returning the first.
        with pytest.raises(ValueError, match="2 terminal regimes"):
            _ = mixed.terminal_regime
        assert "mixed bodies" in describe(mixed)

    def test_a_mixed_payload_must_be_exactly_glide_and_ballistic(self):
        with pytest.raises(ValueError, match="exactly those two regimes"):
            validate((Phase.BOOST, Phase.DISPENSE, Phase.GLIDE), Payload.MIXED)
        with pytest.raises(ValueError, match="exactly those two regimes"):
            validate(
                (Phase.BOOST, Phase.DISPENSE, Phase.CRUISE, Phase.BALLISTIC),
                Payload.MIXED,
            )

    def test_concurrent_arcs_have_a_canonical_order(self):
        """The two arcs are flown at once by different bodies, so their
        order in the phase list carries no meaning. Admitting both orders
        would enumerate every mixed architecture twice."""
        with pytest.raises(ValueError, match="list glide first"):
            validate(
                (Phase.BOOST, Phase.DISPENSE, Phase.BALLISTIC, Phase.GLIDE),
                Payload.MIXED,
            )

    def test_every_mixed_architecture_dispenses_and_carries_both_arcs(self):
        produced = enumerate_architectures(Payload.MIXED)
        assert produced
        for architecture in produced:
            assert Phase.DISPENSE in architecture.phases
            assert set(architecture.terminal_regimes) == {
                Phase.GLIDE,
                Phase.BALLISTIC,
            }
            assert architecture.phases.index(Phase.GLIDE) < architecture.phases.index(
                Phase.BALLISTIC
            )

    def test_a_cruiser_never_reaches_orbit_in_any_admissible_sequence(self):
        for architecture in enumerate_architectures(Payload.CRUISER):
            assert not architecture.is_orbital
            assert architecture.terminal_regime is Phase.CRUISE

    def test_multiple_body_architectures_all_dispense(self):
        for payload in (Payload.MULTIPLE_RV, Payload.MULTIPLE_GLIDER):
            produced = enumerate_architectures(payload)
            assert produced
            assert all(Phase.DISPENSE in a.phases for a in produced)

    def test_every_payload_admits_at_least_one_architecture(self):
        """A payload with no admissible sequence would mean the rules had
        excluded a capability the taxonomy claims to describe."""
        for payload in Payload:
            assert enumerate_architectures(payload), payload

    def test_enumeration_is_sorted_by_length(self):
        lengths = [len(a.phases) for a in enumerate_architectures()]
        assert lengths == sorted(lengths)


class TestMissionBudget:
    """End-to-end accounting: what an architecture costs and whether it
    closes for a stated launch site and aimpoints."""

    def _request(self, *targets, arrival=3000.0):
        site = GeodeticPosition.from_degrees(45.0, 60.0, label="launch")
        points = targets or (GeodeticPosition.from_degrees(38.0, -100.0),)
        return MissionRequest(launch_site=site, aimpoints=tuple(points), arrival_time=arrival)

    def _vehicles(self):
        return EntryVehicle(ballistic_coefficient=200.0, lift_to_drag=2.0), (
            CruiseVehicle(400.0, 4.0, 0.1, 0.30)
        )

    def test_request_takes_the_farthest_aimpoint_and_the_widest_spread(self):
        """Every body is dispensed from one trajectory, so the farthest
        target sets the range the architecture must close."""
        near = GeodeticPosition.from_degrees(50.0, 55.0)
        far = GeodeticPosition.from_degrees(38.0, -100.0)
        request = self._request(near, far)
        assert request.required_range == pytest.approx(great_circle_range(request.launch_site, far))
        assert request.aimpoint_spread == pytest.approx(great_circle_range(near, far))
        assert self._request(far).aimpoint_spread == 0.0

    def test_orbital_profiles_absorb_the_remainder_in_the_parking_arc(self):
        """The structural reason a fractional-orbital profile is flexible
        about range: the slack leg costs time, not propellant."""
        entry, cruise = self._vehicles()
        request = self._request()
        budget = evaluate(
            NAMED_ARCHITECTURES["fractional-orbital-single"],
            request,
            entry_vehicle=entry,
            cruise_vehicle=cruise,
        )
        assert budget.closes
        assert budget.slack_phase is Phase.PARKING
        assert budget.shortfall == pytest.approx(0.0)
        assert budget.total_range == pytest.approx(request.required_range, rel=1e-9)
        parking = next(leg for leg in budget.legs if leg.phase is Phase.PARKING)
        assert parking.is_slack
        assert parking.delta_v == 0.0

    def test_suborbital_profiles_charge_the_remainder_to_boost(self):
        """No free slack leg, so the remainder is bought with propellant and
        an architecture that cannot reach is infeasible for that booster
        rather than merely expensive."""
        entry, cruise = self._vehicles()
        budget = evaluate(
            NAMED_ARCHITECTURES["ballistic-single"],
            self._request(),
            entry_vehicle=entry,
            cruise_vehicle=cruise,
        )
        assert not budget.closes
        assert budget.slack_phase is Phase.BOOST
        assert budget.shortfall > 5.0e6
        assert "infeasible for the stated booster" in budget.reason

    def test_a_long_glide_can_overshoot_and_the_diagnosis_says_so(self):
        """The two ways to fail are not opposite ends of one scale. Here the
        fixed legs already cover more than the required range, and the
        remedy is a shorter glide rather than a bigger booster."""
        entry, _ = self._vehicles()
        request = self._request()
        overshoot = evaluate(
            NAMED_ARCHITECTURES["fractional-orbital-glide"],
            request,
            entry_vehicle=entry,
            glide_range=6.0e6,
        )
        assert not overshoot.closes
        assert "overshoot" in overshoot.reason
        assert overshoot.total_range > request.required_range
        # And the remedy the diagnosis names actually works.
        shortened = evaluate(
            NAMED_ARCHITECTURES["fractional-orbital-glide"],
            request,
            entry_vehicle=entry,
            glide_range=3.0e6,
        )
        assert shortened.closes
        assert shortened.reason == ""

    def test_mixed_payload_charges_only_the_longer_concurrent_arc(self):
        """Glide and ballistic bodies separate and fly in parallel, so
        adding both would double-count a distance covered once."""
        entry, _ = self._vehicles()
        budget = evaluate(
            NAMED_ARCHITECTURES["fractional-orbital-mixed"],
            self._request(),
            entry_vehicle=entry,
            glide_range=3.0e6,
        )
        glide = next(leg for leg in budget.legs if leg.phase is Phase.GLIDE)
        ballistic = next(leg for leg in budget.legs if leg.phase is Phase.BALLISTIC)
        assert glide.ground_range == pytest.approx(3.0e6)
        assert ballistic.ground_range == 0.0
        assert "not charged as transport" in ballistic.note

    def test_dispensing_cost_scales_with_the_number_of_bodies(self):
        entry, _ = self._vehicles()
        two = self._request(
            GeodeticPosition.from_degrees(38.0, -100.0),
            GeodeticPosition.from_degrees(40.0, -95.0),
        )
        four = self._request(
            GeodeticPosition.from_degrees(38.0, -100.0),
            GeodeticPosition.from_degrees(40.0, -95.0),
            GeodeticPosition.from_degrees(42.0, -90.0),
            GeodeticPosition.from_degrees(36.0, -105.0),
        )
        costs = []
        for request in (two, four):
            budget = evaluate(
                NAMED_ARCHITECTURES["fractional-orbital-multiple"],
                request,
                entry_vehicle=entry,
            )
            costs.append(next(leg for leg in budget.legs if leg.phase is Phase.DISPENSE).delta_v)
        assert costs[1] == pytest.approx(3.0 * costs[0] / 1.0, rel=1e-9)

    def test_deorbit_leg_carries_its_real_transfer_arc(self):
        """Charged from the actual Kepler solve, not a constant."""
        entry, _ = self._vehicles()
        budget = evaluate(
            NAMED_ARCHITECTURES["fractional-orbital-single"],
            self._request(),
            entry_vehicle=entry,
        )
        deorbit = next(leg for leg in budget.legs if leg.phase is Phase.DEORBIT)
        assert 3.0e6 < deorbit.ground_range < 8.0e6
        assert 100.0 < deorbit.delta_v < 300.0
        assert deorbit.duration > 0.0
        assert "entry gamma" in deorbit.note

    def test_a_glide_architecture_demands_a_vehicle_to_charge_it(self):
        """Refusing beats silently charging a default vehicle the caller
        never specified."""
        with pytest.raises(ValueError, match="entry_vehicle is required"):
            evaluate(NAMED_ARCHITECTURES["boost-glide"], self._request())
        with pytest.raises(ValueError, match="cruise_vehicle is required"):
            evaluate(NAMED_ARCHITECTURES["powered-cruise"], self._request())

    def test_request_validates_its_own_inputs(self):
        site = GeodeticPosition.from_degrees(45.0, 60.0)
        with pytest.raises(ValueError, match="at least one aimpoint"):
            MissionRequest(launch_site=site, aimpoints=(), arrival_time=100.0)
        with pytest.raises(ValueError, match="arrival_time"):
            MissionRequest(
                launch_site=site,
                aimpoints=(GeodeticPosition.from_degrees(0.0, 0.0),),
                arrival_time=0.0,
            )

    def test_every_named_architecture_can_be_costed(self):
        """The accounting must span the whole taxonomy, whether or not any
        given architecture closes for this particular geometry."""
        entry, cruise = self._vehicles()
        request = self._request()
        for name, architecture in NAMED_ARCHITECTURES.items():
            budget = evaluate(
                architecture,
                request,
                entry_vehicle=entry,
                cruise_vehicle=cruise,
            )
            assert budget.total_delta_v > 0.0, name
            assert budget.summary(), name
            assert (budget.reason == "") is budget.closes, name


class TestAccuracyStatistics:
    """CEP and the 95% radius, and why the usual conversion factor is wrong."""

    def test_circular_constants_are_exact(self):
        assert pytest.approx(np.sqrt(2.0 * np.log(2.0)), rel=1e-15) == CEP_OVER_SIGMA
        assert pytest.approx(np.sqrt(2.0 * np.log(20.0)), rel=1e-15) == R95_OVER_SIGMA
        assert (
            pytest.approx(np.sqrt(np.log(20.0) / np.log(2.0)), rel=1e-15)
            == R95_OVER_SIGMA / CEP_OVER_SIGMA
        )
        # And the decimal values usually quoted, to the precision quoted.
        assert pytest.approx(1.1774, abs=5e-5) == CEP_OVER_SIGMA
        assert pytest.approx(2.4477, abs=5e-5) == R95_OVER_SIGMA

    def test_general_path_reproduces_the_closed_form_when_circular(self):
        """The elliptical quadrature must agree with the Rayleigh closed
        form at unit aspect ratio, or the two disagree everywhere."""
        stats = accuracy_statistics(100.0, 100.0)
        assert stats.cep == pytest.approx(100.0 * CEP_OVER_SIGMA, rel=1e-9)
        assert stats.r95 == pytest.approx(100.0 * R95_OVER_SIGMA, rel=1e-9)
        assert stats.is_circular

    def test_containment_probability_is_consistent_with_its_own_radius(self):
        """Round-trip: the radius containing p must contain p."""
        for major, minor in ((100.0, 100.0), (300.0, 100.0), (500.0, 50.0)):
            for p in (0.3, 0.5, 0.95, 0.99):
                r = containment_radius(p, major, minor)
                assert containment_probability(r, major, minor) == pytest.approx(p, abs=1e-8)

    def test_the_ratio_rises_as_the_ellipse_elongates(self):
        """The measured behaviour, and the reason a computed ratio beats an
        assumed one. Scaling a CEP by the circular 2.079 *under-states* the
        95% radius for any real dispersion."""
        ratios = [containment_ratio(100.0, 100.0 * f) for f in (1.0, 0.5, 0.3, 0.1, 0.02)]
        assert ratios == sorted(ratios)
        assert ratios[0] == pytest.approx(2.0789, abs=1e-3)
        # The one-dimensional limit is the normal-distribution equivalent,
        # 1.96 / 0.6745.
        assert ratios[-1] == pytest.approx(1.959964 / 0.674490, rel=2e-3)

    def test_a_degenerate_axis_reduces_to_the_normal_distribution(self):
        """With one sigma zero the disc becomes an interval, and the
        containment radius must be the ordinary normal quantile."""
        assert containment_radius(0.95, 100.0, 0.0) == pytest.approx(195.9964, rel=1e-4)
        assert containment_radius(0.5, 100.0, 0.0) == pytest.approx(67.449, rel=1e-4)

    def test_sigmas_are_ordered_and_validated(self):
        """Passing them the other way round must give the same answer."""
        a = accuracy_statistics(100.0, 400.0)
        b = accuracy_statistics(400.0, 100.0)
        assert a.cep == pytest.approx(b.cep)
        assert a.sigma_major == pytest.approx(400.0)
        with pytest.raises(ValueError, match="at least one sigma"):
            accuracy_statistics(0.0, 0.0)
        with pytest.raises(ValueError, match="must be finite"):
            accuracy_statistics(-1.0, 100.0)
        with pytest.raises(ValueError, match="probability must lie"):
            containment_radius(1.0, 100.0)


class TestBudgetDispersion:
    def _setup(self):
        site = GeodeticPosition.from_degrees(45.0, 60.0)
        request = MissionRequest(
            launch_site=site,
            aimpoints=(
                GeodeticPosition.from_degrees(38.0, -100.0),
                GeodeticPosition.from_degrees(40.0, -95.0),
            ),
            arrival_time=3000.0,
        )
        return request, EntryVehicle(200.0, 2.0), CruiseVehicle(400.0, 4.0, 0.1, 0.3)

    def test_every_architecture_reports_cep_and_r95(self):
        """The accuracy column must span the whole taxonomy, mixed payloads
        included."""
        request, entry, cruise = self._setup()
        for name, architecture in NAMED_ARCHITECTURES.items():
            budget = evaluate(
                architecture,
                request,
                entry_vehicle=entry,
                cruise_vehicle=cruise,
                glide_range=3.0e6,
            )
            assert budget.accuracy is not None, name
            assert budget.accuracy.cep > 0.0, name
            assert budget.accuracy.r95 > budget.accuracy.cep, name
            assert "CEP" in budget.summary(), name

    def test_dispersions_are_elliptical_so_the_naive_factor_understates(self):
        """Every architecture's dispersion is elongated, so scaling CEP by
        the circular 2.079 gives a 95% radius that is too small."""
        request, entry, cruise = self._setup()
        for architecture in NAMED_ARCHITECTURES.values():
            stats = evaluate(
                architecture,
                request,
                entry_vehicle=entry,
                cruise_vehicle=cruise,
                glide_range=3.0e6,
            ).accuracy
            assert stats.ratio > 2.078
            assert stats.cep * 2.078922 < stats.r95

    def test_guidance_phases_reduce_the_accumulated_error(self):
        """A correction multiplies what is already accumulated, so removing
        terminal homing must make the answer worse."""
        request, entry, _ = self._setup()
        with_homing = NAMED_ARCHITECTURES["fractional-orbital-single"]
        without = Architecture(
            phases=tuple(p for p in with_homing.phases if p is not Phase.TERMINAL),
            payload=with_homing.payload,
        )
        guided = evaluate(with_homing, request, entry_vehicle=entry)
        unguided = evaluate(without, request, entry_vehicle=entry)
        assert unguided.accuracy.cep > guided.accuracy.cep

    def test_dispensing_degrades_accuracy_relative_to_a_single_body(self):
        request, entry, _ = self._setup()
        single = evaluate(
            NAMED_ARCHITECTURES["fractional-orbital-single"],
            request,
            entry_vehicle=entry,
        )
        multiple = evaluate(
            NAMED_ARCHITECTURES["fractional-orbital-multiple"],
            request,
            entry_vehicle=entry,
        )
        assert multiple.accuracy.cep > single.accuracy.cep

    def test_a_correction_resets_the_error_rather_than_scaling_it(self):
        """The defining property, and the one this model first got wrong.

        A midcourse correction nulls the *trajectory* error outright; what
        survives is the error in the estimate it was computed from plus the
        error in executing it, and neither depends on how large the
        incoming error was. So the post-correction error must be
        independent of everything before it. Modelling the phase as a
        multiplier instead makes a good correction look worse after a bad
        boost and better after a good one, which is exactly backwards."""
        request, entry, _ = self._setup()
        architecture = NAMED_ARCHITECTURES["ballistic-single"]
        floors = []
        for boost_error in ((300.0, 200.0), (1200.0, 900.0), (5000.0, 4000.0)):
            original = DISPERSION_SOURCES[Phase.BOOST]
            DISPERSION_SOURCES[Phase.BOOST] = boost_error
            try:
                budget = evaluate(architecture, request, entry_vehicle=entry)
                floors.append(budget.accuracy.cep)
            finally:
                DISPERSION_SOURCES[Phase.BOOST] = original
        # A twenty-fold spread in boost error leaves the answer unchanged.
        assert max(floors) == pytest.approx(min(floors), rel=1e-12)

    def test_architectures_without_a_correction_do_inherit_boost_error(self):
        """The complement: remove the correction and the boost error must
        propagate through, or the reset is being applied unconditionally."""
        request, entry, _ = self._setup()
        with_correction = NAMED_ARCHITECTURES["ballistic-single"]
        without = Architecture(
            phases=tuple(p for p in with_correction.phases if p is not Phase.MIDCOURSE),
            payload=with_correction.payload,
        )
        original = DISPERSION_SOURCES[Phase.BOOST]
        try:
            DISPERSION_SOURCES[Phase.BOOST] = (300.0, 200.0)
            small = evaluate(without, request, entry_vehicle=entry).accuracy.cep
            DISPERSION_SOURCES[Phase.BOOST] = (5000.0, 4000.0)
            large = evaluate(without, request, entry_vehicle=entry).accuracy.cep
        finally:
            DISPERSION_SOURCES[Phase.BOOST] = original
        assert large > 5.0 * small


class TestAgainstSiourisTable52:
    """The containment integral against 126 published numbers.

    Before this, `containment_radius` was checked only against its own
    circular closed form and its own consistency with
    `containment_probability`. Siouris Table 5.2 spans the entire domain —
    21 aspect ratios from degenerate to circular, six probability levels —
    and is an entirely independent computation.
    """

    def test_reproduces_every_non_degenerate_entry(self):
        """All 120 two-dimensional entries, to one unit in the table's last
        printed place.

        The bound is 1e-4 rather than the 5e-5 of ideal rounding because
        three of these 120 entries are one unit high in the source — see
        `test_the_four_entries_the_source_rounds_up`. Every other entry is
        inside 5e-5, i.e. agreement to every digit published.
        """
        worst = 0.0
        for i, ratio in enumerate(SIOURIS_TABLE_5_2_RATIOS):
            if ratio == 0.0:
                continue  # degenerate row, handled separately
            for j, probability in enumerate(SIOURIS_TABLE_5_2_PROBABILITIES):
                ours = containment_radius(probability, 1.0, ratio)
                worst = max(worst, abs(ours - SIOURIS_TABLE_5_2[i][j]))
        assert worst < 1.0e-4, f"worst deviation from Table 5.2 is {worst:.2e}"

    def test_the_four_entries_the_source_rounds_up(self):
        """Four of the 126 entries sit just over half a unit in the last
        place from the exact value, and in every case the exact value
        rounds to a *different* last digit — so the discrepancy is the
        table's rounding, not ours.

        Pinned rather than absorbed into a loose tolerance: if our integral
        ever drifted, these four would move and the rest would not, which
        distinguishes a real regression from this known artefact.
        """
        from scipy.stats import norm

        known = {
            (0.00, 0.75): 1.1504,
            (0.05, 0.50): 0.6764,
            (0.75, 0.90): 1.9034,
            (1.00, 0.95): 2.4478,
        }
        for (ratio, probability), published in known.items():
            if ratio == 0.0:
                exact = float(norm.ppf(0.5 + 0.5 * probability))
            else:
                exact = containment_radius(probability, 1.0, ratio)
            deviation = published - exact
            assert 5.0e-5 < deviation < 6.0e-5
            # The tell: the exact value rounds to a different last digit.
            assert round(exact, 4) != published

        # And they are the *only* four. Everything else is inside ideal
        # rounding, so this is not a systematic bias in our integral.
        outliers = 0
        for i, ratio in enumerate(SIOURIS_TABLE_5_2_RATIOS):
            for j, probability in enumerate(SIOURIS_TABLE_5_2_PROBABILITIES):
                if ratio == 0.0:
                    exact = float(norm.ppf(0.5 + 0.5 * probability))
                else:
                    exact = containment_radius(probability, 1.0, ratio)
                if abs(SIOURIS_TABLE_5_2[i][j] - exact) > 5.0e-5:
                    outliers += 1
        assert outliers == 4

    def test_degenerate_row_is_the_one_dimensional_normal(self):
        """The sigma_S = 0 row is not a two-dimensional result at all: with
        no spread on one axis the miss distance is a folded normal on the
        other, so the entries must be normal quantiles. That the table and
        the closed form agree here is what makes the rest of it credible."""
        from scipy.stats import norm

        for j, probability in enumerate(SIOURIS_TABLE_5_2_PROBABILITIES):
            quantile = float(norm.ppf(0.5 + 0.5 * probability))
            assert quantile == pytest.approx(SIOURIS_TABLE_5_2[0][j], abs=1e-4)
        # The two entries everyone recognises.
        assert SIOURIS_TABLE_5_2[0][1] == pytest.approx(0.6745, abs=5e-5)
        assert SIOURIS_TABLE_5_2[0][4] == pytest.approx(1.9600, abs=5e-5)

    def test_circular_column_is_the_rayleigh_closed_form(self):
        """The other endpoint. The published 1.1774 and 2.4478 must be
        sqrt(2 ln 2) and sqrt(-2 ln 0.05) exactly."""
        circular = SIOURIS_TABLE_5_2[-1]
        assert circular[1] == pytest.approx(CEP_OVER_SIGMA, abs=5e-5)
        # 2.4478 is one of the four entries the source rounds up.
        assert circular[4] == pytest.approx(R95_OVER_SIGMA, abs=1e-4)
        for j, probability in enumerate(SIOURIS_TABLE_5_2_PROBABILITIES):
            closed_form = float(np.sqrt(-2.0 * np.log(1.0 - probability)))
            assert closed_form == pytest.approx(circular[j], abs=1e-4)

    def test_table_is_monotone_in_both_directions(self):
        """Guards the transcription against a transposed or shuffled row.
        K must rise with probability across a row, and rise with aspect
        ratio down a column — a rounder distribution needs a larger radius
        to contain the same fraction."""
        for row in SIOURIS_TABLE_5_2:
            assert list(row) == sorted(row)
        for j in range(len(SIOURIS_TABLE_5_2_PROBABILITIES)):
            column = [row[j] for row in SIOURIS_TABLE_5_2]
            assert column == sorted(column)

    def test_probable_error_is_the_one_dimensional_half(self):
        """REP and DEP are per-axis 50% points; CEP is the radial one and is
        always larger. Conflating them is the classic error this pair of
        names exists to prevent."""
        assert probable_error(1.0) == pytest.approx(0.6745, abs=5e-5)
        assert probable_error(0.0) == 0.0
        assert probable_error(250.0) == pytest.approx(0.6745 * 250.0, rel=1e-4)
        assert accuracy_statistics(1.0, 1.0).cep > probable_error(1.0)

    def test_classical_cep_approximation_holds_where_it_is_claimed_to(self):
        """Siouris states Eq. (5.17) is usable out to a REP/DEP ratio of 2.
        Checked against our exact integral rather than taken on trust: at
        exactly that ratio the error is 1.5%."""
        for ratio in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
            exact = accuracy_statistics(1.0, ratio).cep
            approximate = cep_from_probable_errors(
                probable_error(1.0), probable_error(ratio)
            )
            assert abs(approximate - exact) / exact < 0.016

    def test_classical_approximation_refuses_where_it_diverges(self):
        """The error is non-monotone: it shrinks again near a 5:1 ratio
        before diverging. Refusing past 5:1 stops a caller reading that
        accidental near-zero as validity."""
        with pytest.raises(ValueError, match="beyond where"):
            cep_from_probable_errors(1.0, 0.1)
        with pytest.raises(ValueError, match="degenerate"):
            cep_from_probable_errors(1.0, 0.0)
        # ...and the non-monotonicity itself, which is why the guard exists.
        error_at = {}
        for ratio in (0.5, 0.35, 0.2, 0.11):
            exact = accuracy_statistics(1.0, ratio).cep
            error_at[ratio] = (0.873 * 0.6745 * (1.0 + ratio) - exact) / exact
        assert error_at[0.35] > error_at[0.5] > 0.0
        assert error_at[0.2] < error_at[0.35]
        assert error_at[0.11] < 0.0

    def test_small_ratio_formula_is_better_than_its_stated_range(self):
        """Eq. (5.13) is published for sigma_S/sigma_L < 0.28 and holds to
        0.1% throughout. We keep the published bound rather than widening
        it on our own spot checks, but the accuracy is worth pinning."""
        for ratio in (0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.27):
            exact = accuracy_statistics(1.0, ratio).cep
            assert abs(cep_small_ratio(1.0, ratio) - exact) / exact < 0.0011
        with pytest.raises(ValueError, match="outside the range"):
            cep_small_ratio(1.0, 0.28)

    def test_the_two_approximations_bracket_the_exact_answer_together(self):
        """Neither formula covers the whole range, but between them they do:
        Eq. (5.13) below 0.28 and Eq. (5.17) above it. This checks the
        handover is seamless rather than leaving a gap where a caller has
        to use a formula outside its range."""
        below = accuracy_statistics(1.0, 0.27).cep
        above = accuracy_statistics(1.0, 0.29).cep
        assert abs(cep_small_ratio(1.0, 0.27) - below) / below < 0.0011
        approximate_above = cep_from_probable_errors(
            probable_error(1.0), probable_error(0.29)
        )
        assert abs(approximate_above - above) / above < 0.025


class TestBallisticLegDuration:
    """The one budget entry that used to be `nan` for every architecture."""

    @staticmethod
    def _request():
        return MissionRequest(
            GeodeticPosition(np.deg2rad(45.0), np.deg2rad(-100.0), 0.0),
            (GeodeticPosition(np.deg2rad(50.0), np.deg2rad(40.0), 0.0),),
            3000.0,
        )

    def _ballistic_leg(self, **kwargs):
        budget = evaluate(NAMED_ARCHITECTURES["ballistic-single"], self._request(), **kwargs)
        return next(leg for leg in budget.legs if leg.phase is Phase.BALLISTIC)

    def test_duration_is_now_finite(self):
        leg = self._ballistic_leg()
        assert np.isfinite(leg.duration)
        assert 30.0 < leg.duration < 90.0

    def test_range_and_duration_both_fall_with_a_steeper_entry(self):
        """They come from the same Allen-Eggers profile, so they must move
        together; a steeper entry is both shorter and quicker."""
        legs = [self._ballistic_leg(ballistic_entry_angle=np.deg2rad(d))
                for d in (20.0, 30.0, 45.0, 60.0)]
        ranges = [leg.ground_range for leg in legs]
        durations = [leg.duration for leg in legs]
        assert ranges == sorted(ranges, reverse=True)
        assert durations == sorted(durations, reverse=True)

    def test_low_ballistic_coefficient_still_reports_nan_and_says_why(self):
        """A light body decelerates to terminal velocity above the ground,
        where the closed form's descent-time integral diverges. Reporting
        `nan` is right; reporting it *silently* was the old behaviour and is
        what the note now fixes."""
        leg = self._ballistic_leg(ballistic_coefficient=2000.0)
        assert np.isnan(leg.duration)
        assert "duration unavailable" in leg.note
        assert "terminal" in leg.note

    def test_the_default_still_reproduces_the_old_constant_range(self):
        """The 300 km constant this replaced corresponded to 21.8 degrees.
        The default must not have moved any existing result."""
        assert self._ballistic_leg().ground_range == pytest.approx(300.0e3, rel=1e-4)


class TestBoostLateralErrorIsRangeDependent:
    """Siouris Eq. (6.116) wired into the budget: a lateral burnout position
    error reaches the impact point suppressed by cos(psi), so the boost
    crossrange contribution depends on how far the mission flies."""

    @staticmethod
    def _budget(target_longitude: float):
        launch = GeodeticPosition(np.deg2rad(45.0), np.deg2rad(-100.0), 0.0)
        target = GeodeticPosition(np.deg2rad(45.0), np.deg2rad(target_longitude), 0.0)
        request = MissionRequest(launch, (target,), 3000.0)
        return evaluate(
            NAMED_ARCHITECTURES["boost-glide"],
            request,
            entry_vehicle=EntryVehicle(ballistic_coefficient=5000.0, lift_to_drag=2.0),
        )

    def test_crossrange_contribution_falls_with_range(self):
        """Monotone over the span where cos(psi) is falling. The effect is
        small at the aviation grade the budget assumes -- 0.17 m out of a
        876 m CEP -- but it is derived rather than stated, and its size is
        now a measurement rather than an assumption."""
        ceps = [self._budget(lon).accuracy.cep for lon in (-70.0, -40.0, 0.0, 40.0, 80.0)]
        assert ceps == sorted(ceps, reverse=True)
        assert ceps[0] - ceps[-1] < 1.0

    def test_a_midcourse_correction_makes_the_term_irrelevant(self):
        """`ballistic-single` carries a midcourse reset, which nulls
        everything boost contributed. The term must therefore have no
        effect there at all -- a reset is not a multiplier."""
        launch = GeodeticPosition(np.deg2rad(45.0), np.deg2rad(-100.0), 0.0)
        ceps = set()
        for lon in (-70.0, 0.0, 80.0):
            target = GeodeticPosition(np.deg2rad(45.0), np.deg2rad(lon), 0.0)
            request = MissionRequest(launch, (target,), 3000.0)
            budget = evaluate(NAMED_ARCHITECTURES["ballistic-single"], request)
            ceps.add(round(budget.accuracy.cep, 9))
        assert len(ceps) == 1

    def test_the_term_would_dominate_at_a_worse_imu_grade(self):
        """Why it is worth carrying despite being small at aviation grade.
        The lateral contribution scales with the IMU's position error, and
        at tactical grade it is 571 m at a 30 degree range angle against the
        glide leg's 400 m -- so it would set the crossrange budget, and the
        cos(psi) suppression would become a first-order design lever rather
        than a rounding correction."""
        sensitivity = crossrange_offset_sensitivity(np.deg2rad(30.0))
        contributions = {
            grade: injection_error(IMU_GRADES[grade], 300.0).position * sensitivity
            for grade in ("marine", "aviation", "intermediate", "tactical")
        }
        assert contributions["aviation"] < 20.0
        assert contributions["tactical"] > 400.0
        # ...and at a quarter circumference even the tactical case vanishes.
        assert (
            injection_error(IMU_GRADES["tactical"], 300.0).position
            * crossrange_offset_sensitivity(0.5 * np.pi)
        ) == pytest.approx(0.0, abs=1e-9)


class TestLaunchPackage:
    """The portable configuration format."""

    _MINIMAL = """
schema = "passes.launch-package/1"

[launch]
latitude_deg = 45.0
longitude_deg = -100.0

[[aimpoints]]
latitude_deg = 50.0
longitude_deg = 40.0
"""

    def test_a_minimal_package_loads_with_documented_defaults(self):
        package = LaunchPackage.from_toml(self._MINIMAL)
        assert package.profile.architecture == "ballistic-single"
        assert package.vehicle.imu_grade == "aviation"
        assert package.profile.burnout_flight_path_angle is None
        assert np.rad2deg(package.launch.latitude) == pytest.approx(45.0)
        assert np.rad2deg(package.aimpoints[0].longitude) == pytest.approx(40.0)

    def test_round_trips_through_both_formats_identically(self):
        package = LaunchPackage.from_toml(self._MINIMAL)
        assert LaunchPackage.from_json(package.to_json()) == package
        assert LaunchPackage.from_toml(package.to_toml()) == package
        assert LaunchPackage.from_json(package.to_json()) == LaunchPackage.from_toml(
            package.to_toml()
        )

    def test_the_shipped_reference_package_loads_and_round_trips(self):
        path = Path("packages/fobs-reference.toml")
        if not path.exists():
            pytest.skip("reference package not present")
        package = load_package(path)
        assert package.profile.architecture == "fractional-orbital-single"
        assert package.arrival_time == pytest.approx(4200.0)
        assert package.objectives == ("warning_time", "burnout_speed")
        assert LaunchPackage.from_json(package.to_json()) == package

    def test_a_missing_or_wrong_schema_is_refused(self):
        """Guessing at an unversioned file is how a format change silently
        reinterprets old data."""
        with pytest.raises(PackageError, match="schema"):
            LaunchPackage.from_toml(self._MINIMAL.replace('"passes.launch-package/1"', '"other/9"'))
        with pytest.raises(PackageError, match="schema"):
            LaunchPackage.from_toml(
                self._MINIMAL.replace('schema = "passes.launch-package/1"', "")
            )

    def test_unknown_keys_are_refused_because_toml_scoping_is_a_trap(self):
        """A bare key written after a [table] header belongs to that table,
        not to the document root. Without this check a package that looks
        like it sets `arrival_time_s` at top level would silently set
        `vehicle.arrival_time_s`, which means nothing, and the loader would
        use the default and say so nowhere.

        This is not hypothetical: it happened to the first example package
        written for this format, and the summary quietly showed one
        objective where the file listed two.
        """
        trapped = self._MINIMAL + """
[vehicle]
imu_grade = "marine"
arrival_time_s = 4200.0
"""
        with pytest.raises(PackageError, match="unknown key"):
            LaunchPackage.from_toml(trapped)
        # ...and the message says why, not just that.
        try:
            LaunchPackage.from_toml(trapped)
        except PackageError as error:
            assert "belongs to that table" in str(error)

    def test_closed_vocabularies_are_checked_against_the_code(self):
        """A typo must be an error with the options listed, never a silent
        fallback."""
        with pytest.raises(PackageError, match="architecture"):
            LaunchPackage.from_toml(
                self._MINIMAL + '\n[profile]\narchitecture = "ballistic-signle"\n'
            )
        with pytest.raises(PackageError, match="imu_grade"):
            LaunchPackage.from_toml(
                self._MINIMAL + '\n[vehicle]\nimu_grade = "military"\n'
            )
        with pytest.raises(PackageError, match="objective"):
            LaunchPackage.from_dict(
                {**LaunchPackage.from_toml(self._MINIMAL).to_dict(),
                 "objectives": ["minimise_regret"]}
            )

    def test_units_are_named_so_a_wrong_unit_cannot_be_plausible(self):
        """Degrees on disk, radians in memory, converted once. A file that
        supplied radians would be out of range and rejected rather than
        quietly flying a different scenario."""
        package = LaunchPackage.from_toml(self._MINIMAL)
        assert package.launch.latitude == pytest.approx(np.deg2rad(45.0))
        with pytest.raises(PackageError, match="latitude_deg"):
            LaunchPackage.from_toml(self._MINIMAL.replace("45.0", "145.0"))

    def test_geometry_is_validated_not_merely_parsed(self):
        with pytest.raises(PackageError, match="entry_interface_altitude_m"):
            LaunchPackage.from_toml(
                self._MINIMAL
                + "\n[profile]\nparking_altitude_m = 100e3\n"
                "entry_interface_altitude_m = 150e3\n"
            )
        with pytest.raises(PackageError, match="arrival_time_s"):
            LaunchPackage.from_toml("arrival_time_s = -1.0\n" + self._MINIMAL)

    def test_it_produces_the_objects_the_framework_consumes(self):
        package = LaunchPackage.from_toml(self._MINIMAL)
        request = package.mission_request()
        assert request.launch_site == package.launch
        assert request.aimpoints == package.aimpoints
        budget = evaluate(package.architecture(), request)
        assert budget is not None

    def test_with_profile_copies_rather_than_mutates(self):
        """Sweeps must not contaminate the baseline they started from."""
        package = LaunchPackage.from_toml(self._MINIMAL)
        swept = package.with_profile(parking_altitude=300.0e3)
        assert swept.profile.parking_altitude == 300.0e3
        assert package.profile.parking_altitude == 150.0e3
        assert swept.launch == package.launch

    def test_load_package_refuses_to_sniff_a_format(self, tmp_path):
        path = tmp_path / "scenario.cfg"
        path.write_text(self._MINIMAL, encoding="utf-8")
        with pytest.raises(PackageError, match="unrecognised package suffix"):
            load_package(path)


class TestCampaign:
    """The campaign format — multiple launch packages sharing sensors."""

    _MINIMAL = """
schema = "passes.launch-package/2"

[[launches]]
schema = "passes.launch-package/1"

[launches.launch]
latitude_deg = 45.0
longitude_deg = -100.0

[[launches.aimpoints]]
latitude_deg = 50.0
longitude_deg = 40.0

[[launches]]
schema = "passes.launch-package/1"

[launches.launch]
latitude_deg = 55.0
longitude_deg = 90.0

[[launches.aimpoints]]
latitude_deg = 40.0
longitude_deg = -77.0
"""

    def test_a_minimal_campaign_loads_with_documented_defaults(self):
        campaign = Campaign.from_toml(self._MINIMAL)
        assert len(campaign.launches) == 2
        assert campaign.launches[0].profile.architecture == "ballistic-single"
        assert campaign.launches[1].vehicle.imu_grade == "aviation"
        assert campaign.objectives == ("warning_time",)

    def test_round_trips_through_both_formats_identically(self):
        campaign = Campaign.from_toml(self._MINIMAL)
        assert Campaign.from_json(campaign.to_json()) == campaign
        assert Campaign.from_toml(campaign.to_toml()) == campaign

    def test_a_missing_or_wrong_schema_is_refused(self):
        with pytest.raises(PackageError, match="schema"):
            Campaign.from_toml(
                self._MINIMAL.replace(
                    '"passes.launch-package/2"', '"passes.launch-package/1"'
                )
            )
        with pytest.raises(PackageError, match="schema"):
            Campaign.from_toml(
                self._MINIMAL.replace('schema = "passes.launch-package/2"', "")
            )

    def test_single_launch_keys_rejected_when_launches_present(self):
        """A campaign that mixes [[launches]] with top-level launch keys
        is rejected, because the intent is ambiguous."""
        bad = self._MINIMAL + '[launch]\nlatitude_deg = 45.0\n'
        with pytest.raises(PackageError, match="must not also carry"):
            Campaign.from_toml(bad)

    def test_campaign_without_launches_is_refused(self):
        text = 'schema = "passes.launch-package/2"\n'
        with pytest.raises(PackageError, match="launches"):
            Campaign.from_toml(text)

    def test_empty_launches_list_is_refused(self):
        text = (
            'schema = "passes.launch-package/2"\n'
            'launches = []\n'
        )
        with pytest.raises(PackageError, match="non-empty"):
            Campaign.from_toml(text)

    def test_child_launch_schema_must_match_single_package(self):
        bad = self._MINIMAL.replace(
            '"passes.launch-package/1"', '"passes.launch-package/3"'
        )
        with pytest.raises(PackageError, match="launch schema"):
            Campaign.from_toml(bad)

    def test_campaign_inherits_sensors_from_top_level(self):
        """When a launch does not declare its own sensors, it inherits the
        campaign-level sensors."""
        text = self._MINIMAL + """
[[sensors]]
name = "Pituffik"
latitude_deg = 76.6
longitude_deg = -68.3
mask_elevation_deg = 5.0
note = "BMEWS, Greenland"
"""
        campaign = Campaign.from_toml(text)
        assert len(campaign.sensors) == 1
        assert campaign.sensors[0].name == "Pituffik"

    def test_campaign_produces_mission_requests_per_launch(self):
        campaign = Campaign.from_toml(self._MINIMAL)
        requests = campaign.mission_requests()
        assert len(requests) == 2
        assert requests[0].launch_site == campaign.launches[0].launch
        assert requests[1].aimpoints == campaign.launches[1].aimpoints

    def test_load_campaign_refuses_to_sniff_a_format(self, tmp_path):
        path = tmp_path / "campaign.cfg"
        path.write_text(self._MINIMAL, encoding="utf-8")
        with pytest.raises(PackageError, match="unrecognised package suffix"):
            load_campaign(path)

    def test_the_shipped_reference_campaign_loads_and_round_trips(self):
        path = Path("packages/mid-latitude-campaign.toml")
        if not path.exists():
            pytest.skip("reference campaign not present")
        campaign = load_campaign(path)
        assert len(campaign.launches) == 2
        assert campaign.launches[0].launch.label == "Dombarovskiy (Silo 1A)"
        assert campaign.launches[1].launch.label == "Uzhur (Silo 1A)"
        assert Campaign.from_json(campaign.to_json()) == campaign
