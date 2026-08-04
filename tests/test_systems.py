"""System-level composition: which phase sequences form admissible systems."""

import pytest

from passes.geodesy import GeodeticPosition, great_circle_range
from passes.guidance import CruiseVehicle, EntryVehicle
from passes.systems import (
    NAMED_ARCHITECTURES,
    Architecture,
    MissionRequest,
    Payload,
    Phase,
    PhaseRegime,
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
