"""Vehicle mass distribution, and the geometry that has to hold it."""

import numpy as np
import pytest

from passes.systems.mass import (
    STORABLE_BULK_DENSITY,
    Stage,
    VehicleMassModel,
    sarmat_mass_model,
)


def _cylinder(length=10.0, radius=1.0, n=200):
    """Station/radius profile of a plain cylinder, nose at station 0."""
    stations = np.linspace(0.0, length, n)
    return stations, np.full(n, radius)


class TestStage:
    def test_gross_is_dry_plus_propellant(self):
        stage = Stage("s", 0.0, 5.0, dry_mass=1000.0, propellant_mass=9000.0)
        assert stage.gross_mass == pytest.approx(10000.0)
        assert stage.length == pytest.approx(5.0)

    def test_a_stage_must_extend_aft(self):
        with pytest.raises(ValueError, match="must extend aft"):
            Stage("s", 5.0, 1.0, dry_mass=1.0)

    def test_a_massless_stage_is_refused(self):
        with pytest.raises(ValueError, match="no dry mass"):
            Stage("s", 0.0, 1.0, dry_mass=0.0)

    def test_the_tank_defaults_to_the_whole_stage(self):
        assert Stage("s", 2.0, 7.0, dry_mass=1.0).tank == (2.0, 7.0)
        assert Stage("s", 2.0, 7.0, dry_mass=1.0, tank_forward=3.0).tank == (3.0, 7.0)


class TestGeometryIntegration:
    def test_a_cylinder_volume_matches_the_closed_form(self):
        stations, radii = _cylinder(length=10.0, radius=2.0)
        model = VehicleMassModel(
            [Stage("s", 0.0, 10.0, dry_mass=100.0)], stations, radii
        )
        assert model.volume(0.0, 10.0) == pytest.approx(np.pi * 4.0 * 10.0, rel=1e-6)

    def test_a_uniform_shell_has_its_centroid_at_mid_length(self):
        stations, radii = _cylinder(length=10.0, radius=1.0)
        model = VehicleMassModel(
            [Stage("s", 0.0, 10.0, dry_mass=1000.0)], stations, radii
        )
        assert model.state().centre_of_mass == pytest.approx(5.0, abs=1e-6)

    def test_shell_inertia_matches_the_closed_form(self):
        """A thin cylindrical shell of radius a: I_roll = m a^2, and
        I_pitch = m(a^2/2 + L^2/12) about its centre. Checked against
        arithmetic, not against the code that produced it."""
        length, radius, mass = 10.0, 1.5, 1000.0
        stations, radii = _cylinder(length, radius)
        model = VehicleMassModel(
            [Stage("s", 0.0, length, dry_mass=mass)], stations, radii
        )
        state = model.state()
        assert state.roll_inertia == pytest.approx(mass * radius**2, rel=1e-6)
        assert state.pitch_inertia == pytest.approx(
            mass * (radius**2 / 2.0 + length**2 / 12.0), rel=1e-4
        )

    def test_the_inertia_tensor_is_diagonal_by_symmetry(self):
        stations, radii = _cylinder()
        model = VehicleMassModel([Stage("s", 0.0, 10.0, dry_mass=1.0)], stations, radii)
        tensor = model.state().inertia_tensor()
        assert np.allclose(tensor - np.diag(np.diag(tensor)), 0.0)
        assert tensor[1, 1] == pytest.approx(tensor[2, 2])


class TestPropellantDraining:
    @staticmethod
    def _model():
        stations, radii = _cylinder(length=10.0, radius=1.0)
        volume = np.pi * 1.0 * 10.0
        return VehicleMassModel(
            [Stage("s", 0.0, 10.0, dry_mass=1000.0,
                   propellant_mass=volume * STORABLE_BULK_DENSITY)],
            stations, radii,
        )

    def test_a_full_tank_fills_the_bay(self):
        model = self._model()
        assert model.audit()[0]["fill_fraction"] == pytest.approx(1.0, rel=1e-6)

    def test_the_surface_moves_aft_as_it_drains(self):
        """Liquid settles aft under thrust, so the tank empties from the
        front. Modelling it as a uniformly thinning column instead would
        pin the propellant centroid and lose the centre-of-mass travel
        entirely — which on the reference vehicle is metres."""
        model = self._model()
        stage = model.stages[0]
        fronts = [
            model.propellant_front(stage, f * stage.propellant_mass, STORABLE_BULK_DENSITY)
            for f in (1.0, 0.75, 0.5, 0.25, 0.0)
        ]
        assert fronts == sorted(fronts), "the surface must move aft monotonically"
        assert fronts[0] == pytest.approx(0.0, abs=1e-3)
        assert fronts[-1] == pytest.approx(10.0, abs=1e-3)

    def test_half_a_cylindrical_tank_leaves_the_aft_half(self):
        model = self._model()
        stage = model.stages[0]
        front = model.propellant_front(stage, 0.5 * stage.propellant_mass, STORABLE_BULK_DENSITY)
        assert front == pytest.approx(5.0, abs=1e-3)

    def test_the_centre_of_mass_travels_and_comes_back(self):
        """Not monotone, and the non-monotonicity is physical. A full
        cylindrical tank has the same centroid as its shell, so the vehicle
        starts balanced; draining from the front moves the propellant
        centroid aft and drags the vehicle's with it; and once the tank is
        empty only the shell is left, back at mid-length. Asserting a
        monotone drift would encode a model that is wrong at both ends."""
        model = self._model()
        stage = model.stages[0]
        centres = [
            model.state(burned={"s": f * stage.propellant_mass}).centre_of_mass
            for f in (0.0, 0.5, 1.0)
        ]
        assert centres[0] == pytest.approx(5.0, abs=1e-3)
        assert centres[2] == pytest.approx(5.0, abs=1e-3)
        assert centres[1] > centres[0] + 0.5, "it must move aft while draining"

    def test_mass_falls_by_exactly_what_was_burned(self):
        model = self._model()
        stage = model.stages[0]
        start = model.state().mass
        end = model.state(burned={"s": 0.3 * stage.propellant_mass}).mass
        assert start - end == pytest.approx(0.3 * stage.propellant_mass, rel=1e-9)

    def test_burning_more_than_is_loaded_is_clamped(self):
        model = self._model()
        state = model.state(burned={"s": 1e12})
        assert state.propellant_remaining["s"] == pytest.approx(0.0)
        assert state.mass == pytest.approx(model.stages[0].dry_mass)


class TestStaging:
    @staticmethod
    def _two_stage():
        stations, radii = _cylinder(length=20.0, radius=1.0)
        return VehicleMassModel(
            [
                Stage("upper", 0.0, 8.0, dry_mass=1000.0, propellant_mass=5000.0),
                Stage("lower", 8.0, 20.0, dry_mass=3000.0, propellant_mass=30000.0),
            ],
            stations, radii,
        )

    def test_jettison_removes_the_stage_entirely(self):
        model = self._two_stage()
        before = model.state()
        after = model.state(jettisoned=("lower",))
        assert after.stages_present == ("upper",)
        assert after.mass == pytest.approx(6000.0)
        assert after.mass < before.mass

    def test_staging_moves_the_centre_of_mass_forward(self):
        model = self._two_stage()
        lower = model.stages[1]
        burnt = model.state(burned={"lower": lower.propellant_mass})
        staged = model.state(burned={"lower": lower.propellant_mass}, jettisoned=("lower",))
        assert staged.centre_of_mass < burnt.centre_of_mass

    def test_an_unknown_stage_is_refused(self):
        with pytest.raises(ValueError, match="unknown stage"):
            self._two_stage().state(burned={"booster": 1.0})

    def test_jettisoning_everything_is_refused(self):
        with pytest.raises(ValueError, match="no vehicle left"):
            self._two_stage().state(jettisoned=("upper", "lower"))

    def test_duplicate_stage_names_are_refused(self):
        stations, radii = _cylinder()
        with pytest.raises(ValueError, match="unique"):
            VehicleMassModel(
                [Stage("s", 0.0, 5.0, dry_mass=1.0), Stage("s", 5.0, 10.0, dry_mass=1.0)],
                stations, radii,
            )


class TestSarmatModel:
    @staticmethod
    def _profile():
        # The scaled reference body: 35.4 m, 3.0 m diameter, near-constant
        # radius with a nose cone over the first 3.3 m.
        stations = np.linspace(0.0, 35.4, 400)
        radii = np.where(stations < 3.3, 1.5 * (stations / 3.3) ** 0.59, 1.5)
        return stations, radii

    def test_the_totals_match_open_reporting(self):
        model = sarmat_mass_model(*self._profile())
        assert model.gross_mass == pytest.approx(208.1e3, rel=1e-9)
        propellant = sum(s.propellant_mass for s in model.stages)
        assert propellant == pytest.approx(178.0e3 + 1.8e3, rel=1e-9)

    def test_the_booster_dry_mass_lands_where_the_predecessor_did(self):
        """208.1 gross - 178 propellant - 10 throw leaves 20.1 t of booster
        structure, and the R-36M2's two boosters are 13-15 t and 3-5 t.
        That the residual falls in the documented band is the one
        consistency check available on numbers nobody publishes."""
        model = sarmat_mass_model(*self._profile())
        stages = {s.name: s for s in model.stages}
        assert 13.0e3 < stages["stage1"].dry_mass < 17.0e3
        assert 3.0e3 < stages["stage2"].dry_mass < 6.0e3

    def test_every_inferred_number_says_that_it_is_inferred(self):
        model = sarmat_mass_model(*self._profile())
        for stage in model.stages:
            if stage.name in ("stage1", "stage2", "bus"):
                assert "inferred" in stage.provenance["dry_mass"]

    def test_the_as_mapped_rings_cannot_hold_the_propellant(self):
        """The finding. Reading the rings as payload/bus, bus/stage-2,
        stage-2/stage-1 puts 142 t of propellant into a 72 m3 bay."""
        model = sarmat_mass_model(*self._profile(), separations=(6.887, 13.565, 23.354))
        rows = {row["stage"]: row for row in model.audit()}
        assert not rows["stage1"]["feasible"]
        assert float(rows["stage1"]["fill_fraction"]) > 1.2
        with pytest.raises(ValueError, match="does not fit the geometry"):
            model.check()

    def test_shifting_the_reading_one_ring_forward_is_consistent(self):
        """Treating the ring 23.4 m aft as a stage-1 *intertank* frame —
        which is what a tandem oxidiser/fuel stage carries — gives ordinary
        booster fill fractions."""
        model = sarmat_mass_model(*self._profile(), separations=(3.8, 6.887, 13.565))
        model.check()
        rows = {row["stage"]: row for row in model.audit()}
        assert 0.70 < float(rows["stage1"]["fill_fraction"]) < 0.98
        assert 0.55 < float(rows["stage2"]["fill_fraction"]) < 0.95

    def test_an_impossible_mass_budget_is_refused(self):
        with pytest.raises(ValueError, match="negative structure"):
            sarmat_mass_model(*self._profile(), propellant_mass=205.0e3)
        with pytest.raises(ValueError, match="throw weight"):
            sarmat_mass_model(*self._profile(), throw_weight=-1.0)

    def test_the_centre_of_mass_travels_metres_during_the_first_burn(self):
        """Why a scalar mass is not enough: the flight simulator carried
        one, and this vehicle's centre of mass moves several metres before
        staging."""
        model = sarmat_mass_model(*self._profile(), separations=(3.8, 6.887, 13.565))
        stage1 = {s.name: s for s in model.stages}["stage1"]
        start = model.state().centre_of_mass
        end = model.state(burned={"stage1": stage1.propellant_mass}).centre_of_mass
        assert abs(end - start) > 2.0
