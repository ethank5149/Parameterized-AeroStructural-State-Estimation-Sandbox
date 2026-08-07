"""Powered flight: thrust in the coupled right-hand side, and missions.

The engine could not fly a boost or a burn — ``out[layout.mass] = 0.0``,
no thrust term, no terminal event — which is why the fractional-orbital
animations were drawn from a *geometry* model instead. These tests cover
the pieces that close that gap.
"""

import numpy as np
import pytest

from passes.flight.mission import MissionSegment, fly_mission
from passes.flight.propulsion import STANDARD_GRAVITY, Burn, thrust_direction
from passes.flight.simulator import FlightConfiguration, FlightSimulator
from passes.geodesy import WGS84_MEAN_RADIUS

_R = WGS84_MEAN_RADIUS
_UP = np.array([1.0, 0.0, 0.0])
_PLANE = np.array([0.0, 0.0, 1.0])


@pytest.fixture(scope="module")
def sim():
    return FlightSimulator(FlightConfiguration(n_modes=4, drag_area=10.0, nose_radius=1.2))


def _pad(sim, mass=400.0e3):
    return sim.state_at(_R * _UP, 1.0e-3 * _UP, mass=mass)


class TestBurn:
    def test_mass_flow_is_thrust_over_exhaust_velocity(self):
        burn = Burn(100.0, 3.0e5, 3000.0)
        assert burn.mass_flow == pytest.approx(100.0)

    def test_ideal_delta_v_is_tsiolkovsky(self):
        burn = Burn(100.0, 3.0e5, 3000.0)
        # 10 t of propellant from a 50 t vehicle.
        assert burn.ideal_delta_v(50.0e3) == pytest.approx(3000.0 * np.log(50.0 / 40.0))

    def test_a_burn_that_would_empty_the_vehicle_is_refused(self):
        with pytest.raises(ValueError, match="mass would go non-positive"):
            Burn(1000.0, 3.0e5, 3000.0).ideal_delta_v(50.0e3)

    def test_standard_gravity_is_the_isp_conversion(self):
        assert pytest.approx(9.80665) == STANDARD_GRAVITY

    def test_rejects_nonsense(self):
        for kwargs in ({"duration": 0.0}, {"thrust": -1.0}, {"exhaust_velocity": 0.0}):
            with pytest.raises(ValueError):
                Burn(**{"duration": 10.0, "thrust": 1.0, "exhaust_velocity": 1.0, **kwargs})
        with pytest.raises(ValueError, match="steering must be"):
            Burn(10.0, 1.0, 1.0, steering="sideways")
        with pytest.raises(ValueError, match="needs plane_normal"):
            Burn(10.0, 1.0, 1.0, steering="gravity_turn")


class TestSteering:
    def test_a_gravity_turn_leaves_the_pad_vertical(self):
        """Not a detail: the whole reason the pitch program was replaced."""
        burn = Burn(200.0, 1.0, 1.0, "gravity_turn", _PLANE, vertical_time=12.0)
        direction = thrust_direction(burn, 0.0, _R * _UP, np.zeros(3))
        assert np.allclose(direction, _UP)
        assert np.allclose(thrust_direction(burn, 11.0, _R * _UP, np.zeros(3)), _UP)

    def test_the_kick_pitches_over_by_the_commanded_angle(self):
        burn = Burn(
            200.0, 1.0, 1.0, "gravity_turn", _PLANE,
            vertical_time=10.0, kick_time=20.0, kick_angle=0.10,
        )
        direction = thrust_direction(burn, 29.9, _R * _UP, np.array([0.0, 100.0, 0.0]))
        # At the end of the kick the thrust is kick_angle off vertical.
        assert float(np.arccos(np.clip(direction @ _UP, -1, 1))) == pytest.approx(
            0.10, abs=1e-3
        )

    def test_after_the_kick_it_follows_the_velocity(self):
        """This is the mechanism. Gravity turns the trajectory; the steering
        does not, which is why a gravity turn cannot be commanded into the
        atmosphere the way a fixed pitch program can."""
        burn = Burn(
            200.0, 1.0, 1.0, "gravity_turn", _PLANE,
            vertical_time=10.0, kick_time=20.0, kick_angle=0.10,
        )
        velocity = np.array([300.0, 900.0, 0.0])
        direction = thrust_direction(burn, 120.0, _R * _UP, velocity)
        assert np.allclose(direction, velocity / np.linalg.norm(velocity))

    def test_prograde_and_retrograde_oppose(self):
        velocity = np.array([10.0, -200.0, 30.0])
        forward = thrust_direction(Burn(10.0, 1.0, 1.0, "prograde"), 0.0, _R * _UP, velocity)
        back = thrust_direction(Burn(10.0, 1.0, 1.0, "retrograde"), 0.0, _R * _UP, velocity)
        assert np.allclose(forward, -back)
        assert np.allclose(forward, velocity / np.linalg.norm(velocity))

    def test_velocity_steering_stays_finite_at_zero_speed(self):
        """An implicit integrator evaluates the right-hand side at *trial*
        states inside its Newton iteration, and those can pass through zero
        velocity. Raising there aborts a solve that would have converged;
        this is the softened normalisation that fixed it."""
        direction = thrust_direction(
            Burn(10.0, 1.0, 1.0, "prograde"), 0.0, _R * _UP, np.zeros(3)
        )
        assert np.all(np.isfinite(direction))

    def test_the_thrust_direction_is_always_a_unit_vector(self):
        burn = Burn(200.0, 1.0, 1.0, "gravity_turn", _PLANE)
        for elapsed in (0.0, 5.0, 15.0, 25.0, 60.0, 200.0):
            direction = thrust_direction(
                burn, elapsed, _R * _UP, np.array([100.0, 4000.0, 0.0])
            )
            assert float(np.linalg.norm(direction)) == pytest.approx(1.0, abs=1e-12)


class TestPoweredFlight:
    def test_thrust_accelerates_and_burns_mass(self, sim):
        pad = _pad(sim)
        burn = Burn(60.0, 4.0e6, 3400.0, "gravity_turn", _PLANE)
        flown = fly_mission(sim, pad, [MissionSegment(60.0, burn, "boost")],
                            samples_per_segment=20)
        mass = flown.result.states[sim.layout.mass, :]
        speed = np.linalg.norm(flown.result.states[sim.layout.velocity, :], axis=0)
        assert mass[-1] < mass[0]
        assert mass[0] - mass[-1] == pytest.approx(burn.mass_flow * 60.0, rel=1e-6)
        assert speed[-1] > 400.0
        assert np.all(np.diff(mass) < 0.0), "mass must fall monotonically under thrust"

    def test_a_coast_conserves_mass(self, sim):
        state = sim.state_at((_R + 400e3) * _UP, np.array([0.0, 7600.0, 0.0]), mass=5.0e3)
        flown = fly_mission(sim, state, [MissionSegment(300.0, None, "coast")],
                            samples_per_segment=20)
        mass = flown.result.states[sim.layout.mass, :]
        assert np.ptp(mass) == pytest.approx(0.0, abs=1e-9)

    def test_the_flown_delta_v_falls_short_of_tsiolkovsky(self, sim):
        """Gravity and drag take their share, and the gap is the number a
        closed-form two-body planner cannot produce."""
        pad = _pad(sim)
        burn = Burn(180.0, 4.0e6, 3400.0, "gravity_turn", _PLANE, kick_angle=0.06)
        flown = fly_mission(sim, pad, [MissionSegment(180.0, burn, "boost")],
                            samples_per_segment=20)
        achieved = float(
            np.linalg.norm(flown.result.states[sim.layout.velocity, -1])
        )
        ideal = burn.ideal_delta_v(400.0e3)
        assert achieved < ideal
        assert 500.0 < ideal - achieved < 3000.0

    def test_max_dynamic_pressure_is_physical(self, sim):
        """A real launcher passes through 25-35 kPa. A commanded-pitch
        program that turned horizontal at 10 km reached 400 kPa and drag
        exceeded thrust; that is how the steering law was found to be
        wrong."""
        pad = _pad(sim)
        burn = Burn(200.0, 5.4e6, 3400.0, "gravity_turn", _PLANE, kick_angle=0.06)
        flown = fly_mission(sim, pad, [MissionSegment(200.0, burn, "boost")],
                            samples_per_segment=60)
        assert 10.0e3 < flown.result.dynamic_pressure.max() < 60.0e3

    def test_mass_consistent_drag_differs_from_a_fixed_beta(self):
        """A launcher burns most of its mass, so holding the ballistic
        coefficient fixed through the ascent understates the drag on the
        light end by the same factor."""
        fixed = FlightSimulator(FlightConfiguration(n_modes=2, nose_radius=1.2))
        scaled = FlightSimulator(
            FlightConfiguration(n_modes=2, nose_radius=1.2, drag_area=10.0)
        )
        burn = Burn(120.0, 4.0e6, 3400.0, "gravity_turn", _PLANE)
        speeds = []
        for engine in (fixed, scaled):
            flown = fly_mission(
                engine, _pad(engine), [MissionSegment(120.0, burn, "boost")],
                samples_per_segment=10,
            )
            speeds.append(float(np.linalg.norm(flown.result.states[engine.layout.velocity, -1])))
        assert speeds[0] != pytest.approx(speeds[1], rel=1e-6)


class TestMissionStructure:
    def test_segments_are_concatenated_with_a_monotone_clock(self, sim):
        pad = _pad(sim)
        burn = Burn(40.0, 4.0e6, 3400.0, "gravity_turn", _PLANE)
        flown = fly_mission(
            sim, pad,
            [MissionSegment(40.0, burn, "boost"), MissionSegment(60.0, None, "coast")],
            samples_per_segment=15,
        )
        assert np.all(np.diff(flown.result.times) > 0.0)
        assert flown.result.times[0] == pytest.approx(0.0)
        assert flown.result.times[-1] == pytest.approx(100.0)

    def test_each_leg_becomes_a_named_phase_that_tiles_the_flight(self, sim):
        pad = _pad(sim)
        burn = Burn(40.0, 4.0e6, 3400.0, "gravity_turn", _PLANE)
        flown = fly_mission(
            sim, pad,
            [MissionSegment(40.0, burn, "boost"), MissionSegment(60.0, None, "coast")],
            samples_per_segment=15,
        )
        assert [p.name for p in flown.phases] == ["boost", "coast"]
        assert flown.phases[0].end_time == pytest.approx(flown.phases[1].start_time)
        assert flown.phases[-1].end_time == pytest.approx(flown.flight_time)

    def test_the_ground_event_stops_the_flight_at_the_surface(self, sim):
        """The simulator had no terminal condition and integrated 45 km
        *below* the surface in 300 s.

        Measured against ``sim.gravity.radius``, not against
        ``WGS84_MEAN_RADIUS``. Writing this test the other way caught a
        real inconsistency: ``EARTH.radius`` is the **equatorial** radius
        and the geodesy layer's is the **mean**, 7,128 m apart, so the two
        disagree by exactly that everywhere they meet.
        """
        surface = sim.gravity.radius
        state = sim.state_at(
            (surface + 90e3) * _UP,
            np.array([-2000.0, 3000.0, 0.0]),
            mass=1500.0,
        )
        flown = fly_mission(
            sim, state,
            [MissionSegment(600.0, None, "entry", stop_at_ground=True)],
            samples_per_segment=60,
        )
        altitude = (
            np.linalg.norm(flown.result.states[sim.layout.position, :], axis=0) - surface
        )
        assert flown.impacted
        assert altitude[-1] == pytest.approx(0.0, abs=1.0)
        assert np.all(altitude > -1.0), "the vehicle must not burrow"
        assert flown.flight_time < 600.0

    def test_a_burn_whose_program_disagrees_with_its_leg_is_refused(self):
        with pytest.raises(ValueError, match="programmed over"):
            MissionSegment(50.0, Burn(40.0, 1.0, 1.0, "prograde"), "boost")

    def test_velocity_steering_from_rest_is_refused_before_integrating(self, sim):
        at_rest = sim.state_at(_R * _UP, np.zeros(3), mass=400.0e3)
        with pytest.raises(ValueError, match="at rest"):
            fly_mission(
                sim, at_rest,
                [MissionSegment(30.0, Burn(30.0, 4.0e6, 3400.0, "prograde"), "boost")],
                samples_per_segment=5,
            )

    def test_an_empty_mission_is_refused(self, sim):
        with pytest.raises(ValueError, match="at least one segment"):
            fly_mission(sim, _pad(sim), [])
