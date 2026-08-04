"""System-level composition: which phase sequences form admissible systems."""

import pytest

from passes.systems import (
    NAMED_ARCHITECTURES,
    Architecture,
    Payload,
    Phase,
    PhaseRegime,
    describe,
    enumerate_architectures,
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

    def test_exactly_one_terminal_regime(self):
        """Glide, cruise and ballistic entry are alternative descriptions of
        the whole atmospheric arc, not successive stages."""
        with pytest.raises(ValueError, match="exactly one of glide"):
            validate((Phase.BOOST, Phase.GLIDE, Phase.BALLISTIC), Payload.GLIDER)
        with pytest.raises(ValueError, match="exactly one of glide"):
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
