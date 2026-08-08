"""Staging: what is attached, and where the spent stages went.

The two claims worth testing separately are that the glyph shrinks by the
length the mass model says it lost, and that a jettisoned stage's
trajectory is *integrated* rather than drawn. The second is the one that
would be easy to fake and is not.
"""

import numpy as np
import pytest

from passes.geodesy import WGS84_MEAN_RADIUS, GeodeticPosition
from passes.orbital.gravity import EARTH
from passes.orbital.scenario import Event, fobs_trajectory
from passes.systems.mass import Stage, VehicleMassModel, sarmat_mass_model
from passes.viz.history import SimulationHistory
from passes.viz.staging import (
    Separation,
    StagingPlan,
    jettison_histories,
    propagate_jettison,
    stack_polylines,
)

LAUNCH = GeodeticPosition.from_degrees(51.09, 59.84, 0.0)
TARGET = GeodeticPosition.from_degrees(38.90, -77.04, 0.0)


def _profile() -> tuple[np.ndarray, np.ndarray]:
    """The scaled reference body: 35.4 m, 3.0 m diameter, nose over 3.3 m."""
    stations = np.linspace(0.0, 35.4, 400)
    radii = np.where(stations < 3.3, 1.5 * (stations / 3.3) ** 0.59, 1.5)
    return stations, radii


def _model() -> VehicleMassModel:
    return sarmat_mass_model(*_profile())


def _plan(model: VehicleMassModel | None = None) -> StagingPlan:
    built = model if model is not None else _model()
    return StagingPlan(
        model=built,
        separations=(
            Separation("stage1", 120.0),
            Separation("stage2", 300.0),
            Separation("bus", 900.0),
        ),
    )


def _history(samples: int = 300) -> SimulationHistory:
    return SimulationHistory.from_trajectory(
        fobs_trajectory(
            LAUNCH, TARGET, parking_altitude=170e3, parking_apogee=250e3,
            samples=samples,
        ),
        WGS84_MEAN_RADIUS,
    )


class TestStagingPlan:
    def test_stages_leave_in_the_order_given(self):
        plan = _plan()
        assert plan.present_at(0.0) == ("payload", "bus", "stage2", "stage1")
        assert plan.present_at(119.9) == ("payload", "bus", "stage2", "stage1")
        # A separation at exactly t counts as done, matching the mass
        # model's own convention that an event's state is the state after.
        assert plan.present_at(120.0) == ("payload", "bus", "stage2")
        assert plan.present_at(400.0) == ("payload", "bus")
        assert plan.present_at(1000.0) == ("payload",)

    def test_a_stage_the_model_does_not_have_is_refused(self):
        with pytest.raises(ValueError, match="does not have"):
            StagingPlan(_model(), (Separation("stage 3", 10.0),))

    def test_a_stage_may_not_separate_twice(self):
        with pytest.raises(ValueError, match="only separate once"):
            StagingPlan(
                _model(), (Separation("stage1", 10.0), Separation("stage1", 20.0))
            )

    def test_separations_must_be_in_time_order(self):
        """Out of order describes a vehicle that cannot fly.

        Dropping stage 2 before stage 1 would leave the first stage
        attached to nothing, and the only symptom would be a strange
        picture.
        """
        with pytest.raises(ValueError, match="time order"):
            StagingPlan(
                _model(), (Separation("stage1", 300.0), Separation("stage2", 120.0))
            )

    def test_from_events_names_the_event_it_could_not_find(self):
        events = (Event("lift-off", 0.0, ""), Event("boost cutoff", 118.0, ""))
        with pytest.raises(KeyError, match="insertion cutoff"):
            StagingPlan.from_events(
                _model(), events,
                {"stage1": "boost cutoff", "stage2": "insertion cutoff"},
            )

    def test_from_events_takes_the_times_the_mission_reported(self):
        events = (
            Event("lift-off", 0.0, ""),
            Event("boost cutoff", 118.0, ""),
            Event("insertion cutoff", 305.0, ""),
        )
        plan = StagingPlan.from_events(
            _model(), events,
            {"stage1": "boost cutoff", "stage2": "insertion cutoff"},
        )
        assert [s.time for s in plan.separations] == [118.0, 305.0]

    def test_phases_name_what_is_flying(self):
        phases = _plan().phases()
        assert phases[0].name == "payload + bus + stage2 + stage1"
        assert phases[-1].name == "payload"
        assert [p.start_time for p in phases] == [0.0, 120.0, 300.0, 900.0]


class TestStackGeometry:
    def test_the_stack_shortens_by_the_length_the_model_lost(self):
        """The claim the glyph makes, checked arithmetically.

        A fixed-shape glyph cannot say this and a normalised one says the
        opposite — a shorter stack drawn to the same screen size appears to
        *grow* at separation.
        """
        plan = _plan()
        stations, radii = _profile()
        for time, expected in (
            (0.0, plan.stage("stage1").aft),
            (150.0, plan.stage("stage2").aft),
            (400.0, plan.stage("bus").aft),
            (1000.0, plan.stage("payload").aft),
        ):
            lines = stack_polylines(plan, time, stations, radii)
            points = np.concatenate(lines)
            # Stations run aft from the tip; the glyph negates them, so the
            # stack's aft end is the most negative x.
            assert float(-points[:, 0].min()) == pytest.approx(expected, abs=1e-9)
            assert float(points[:, 0].max()) == pytest.approx(0.0, abs=1e-9)

    def test_the_glyph_follows_the_mould_line(self):
        """Radii come from the profile, not from a drawn cone."""
        plan = _plan()
        stations, radii = _profile()
        points = np.concatenate(stack_polylines(plan, 0.0, stations, radii))
        radial = np.hypot(points[:, 1], points[:, 2])
        expected = np.interp(-points[:, 0], stations, radii)
        assert np.allclose(radial, expected, atol=1e-9)
        # The nose tip is a point, the barrel is at the full radius.
        assert radial.max() == pytest.approx(1.5, abs=1e-9)

    def test_mismatched_mould_line_arrays_are_refused(self):
        with pytest.raises(ValueError, match="matching 1-D arrays"):
            stack_polylines(_plan(), 0.0, np.linspace(0, 1, 10), np.zeros(9))


class TestJettison:
    """A spent stage flies; it is not drawn falling."""

    @staticmethod
    def _spent() -> Stage:
        return Stage(name="stage1", forward=13.412, aft=35.4, dry_mass=8.0e3)

    def test_the_spent_stage_leaves_from_the_state_it_separated_at(self):
        history = _history()
        separation = Separation("stage1", 400.0, relative_speed=1.5)
        spent = propagate_jettison(history, separation, self._spent(), 7.07)
        at_separation = history.sample(400.0)
        assert spent.times[0] == pytest.approx(400.0)
        assert np.allclose(spent.positions[0], at_separation["position"])
        # Same speed less the push-off, along the flight direction.
        assert float(np.linalg.norm(spent.velocities[0])) == pytest.approx(
            float(np.linalg.norm(at_separation["velocity"])) - 1.5, abs=1e-6
        )

    def test_it_diverges_from_the_vehicle_because_it_has_to(self):
        """The content of the picture is the divergence, and it is physical.

        An empty stage is light and blunt where the stack was heavy and
        slender, so it decelerates harder. If the two stayed together, the
        drag model would not be doing anything.
        """
        history = _history()
        separation = Separation("stage1", 300.0)
        spent = propagate_jettison(history, separation, self._spent(), 7.07)
        later = min(float(spent.times[-1]), 900.0)
        gap = float(
            np.linalg.norm(
                spent.sample(later)["position"] - history.sample(later)["position"]
            )
        )
        # Ten minutes on, the two are kilometres apart, and not because of
        # the 1.5 m/s push-off, which would give 540 m at most.
        assert gap > 5.0e3

    def test_the_push_off_is_not_what_separates_them(self):
        """Stated in the module note, so it is measured rather than asserted.

        Zeroing the separation impulse must leave the same trajectory to
        within a small fraction: the divergence is drag and gravity, not
        the hardware.
        """
        history = _history()
        stage = self._spent()
        pushed = propagate_jettison(
            history, Separation("stage1", 300.0, relative_speed=1.5), stage, 7.07
        )
        free = propagate_jettison(
            history, Separation("stage1", 300.0, relative_speed=0.0), stage, 7.07
        )
        later = min(float(pushed.times[-1]), float(free.times[-1]))
        gap = float(
            np.linalg.norm(
                pushed.sample(later)["position"] - free.sample(later)["position"]
            )
        )
        reference = float(
            np.linalg.norm(
                pushed.sample(later)["position"] - history.sample(later)["position"]
            )
        )
        assert gap < 0.25 * reference

    def test_a_stage_dropped_low_comes_down_and_stops_there(self):
        """Terminated at the surface, not carried on underground.

        The same defect ``fly_mission``'s ground event exists for: the
        first version of that integrated a fixed duration and reached 45 km
        *below* the surface.
        """
        history = _history()
        spent = propagate_jettison(
            history, Separation("stage1", 120.0), self._spent(), 7.07
        )
        radii = np.linalg.norm(spent.positions, axis=1)
        assert radii.min() >= EARTH.radius - 1.0
        if spent.times[-1] < history.times[-1] - 1.0:
            # It came down: the last sample is the impact, on the surface.
            assert float(radii[-1]) == pytest.approx(EARTH.radius, abs=1.0)

    def test_a_heavier_stage_falls_more_slowly(self):
        """Ballistic coefficient is read from the stage, not assumed.

        Same geometry, ten times the dry mass: ten times the ballistic
        coefficient, so less deceleration and a higher altitude later.
        """
        history = _history()
        light = propagate_jettison(
            history, Separation("stage1", 200.0), self._spent(), 7.07, duration=600.0
        )
        heavy = propagate_jettison(
            history, Separation("stage1", 200.0),
            Stage(name="stage1", forward=13.412, aft=35.4, dry_mass=80.0e3),
            7.07, duration=600.0,
        )
        assert float(np.linalg.norm(heavy.positions[-1])) > float(
            np.linalg.norm(light.positions[-1])
        )

    def test_a_separation_with_no_flight_left_says_so(self):
        history = _history()
        with pytest.raises(ValueError, match="no flight left"):
            propagate_jettison(
                history, Separation("stage1", float(history.times[-1])),
                self._spent(), 7.07,
            )

    def test_a_missing_frontal_area_names_the_stage(self):
        with pytest.raises(KeyError, match="stage2"):
            jettison_histories(
                _history(), _plan(), {"stage1": 7.07}, duration=60.0,
            )

    def test_every_separation_gets_a_history(self):
        spent = jettison_histories(
            _history(), _plan(),
            {"stage1": 7.07, "stage2": 7.07, "bus": 3.0},
            duration=120.0,
        )
        assert set(spent) == {"stage1", "stage2", "bus"}
        for name, run in spent.items():
            assert name in run.label
            assert run.velocities is not None
            assert run.times[0] == pytest.approx(
                next(s.time for s in _plan().separations if s.stage == name)
            )
