"""System-level composition: which phase sequences form admissible systems."""

import numpy as np
import pytest

from passes.geodesy import GeodeticPosition, great_circle_range
from passes.guidance import CruiseVehicle, EntryVehicle
from passes.systems import (
    CEP_OVER_SIGMA,
    DISPERSION_SOURCES,
    NAMED_ARCHITECTURES,
    R95_OVER_SIGMA,
    Architecture,
    MissionRequest,
    Payload,
    Phase,
    PhaseRegime,
    accuracy_statistics,
    containment_probability,
    containment_radius,
    containment_ratio,
    describe,
    enumerate_architectures,
    evaluate,
    validate,
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
