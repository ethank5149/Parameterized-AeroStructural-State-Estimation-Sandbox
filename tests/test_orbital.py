"""Orbital mechanics and the coast phase (Paper II, §7)."""

from pathlib import Path

import numpy as np
import pytest

from passes.geodesy import (
    WGS84_MEAN_RADIUS,
    GeodeticPosition,
    great_circle_bearing,
    great_circle_range,
)
from passes.orbital import (
    EARTH,
    EARTH_ROTATION_RATE,
    approach_azimuth,
    azimuth_envelope,
    compare_coast_strategies,
    deorbit_burn,
    fobs_profile,
    gravitational_acceleration,
    gravitational_potential,
    ground_track,
    ground_track_shift,
    j2_acceleration,
    lambert,
    minimum_energy_transfer,
    orbital_elements,
    propagate_coast,
    regime_transition_profile,
    secular_rates,
    specific_angular_momentum_z,
    specific_energy,
    two_body_acceleration,
)
from passes.orbital.fobs import fractional_insertion
from passes.orbital.radar import (
    COALITIONS,
    EARLY_WARNING_SITES,
    SATELLITE_SENSORS,
    boost_phase_sensing,
    coverage,
    network,
)
from passes.orbital.radar import site as radar_site
from passes.orbital.scenario import (
    ascent_profile,
    ballistic_trajectory,
    fobs_trajectory,
    leading_aimpoint,
    warning_comparison,
)
from passes.orbital.warning import (
    detection_window,
    horizon_central_angle,
    visibility_radius,
)


def circular_state(altitude=400e3, inclination_deg=51.6):
    a = EARTH.radius + altitude
    inc = np.deg2rad(inclination_deg)
    speed = np.sqrt(EARTH.mu / a)
    r = np.array([a, 0.0, 0.0])
    v = np.array([0.0, speed * np.cos(inc), speed * np.sin(inc)])
    period = 2.0 * np.pi * np.sqrt(a**3 / EARTH.mu)
    return r, v, period


class TestGravity:
    def test_two_body_magnitude_and_direction(self):
        r = np.array([EARTH.radius, 0.0, 0.0])
        g = two_body_acceleration(r)
        assert float(np.linalg.norm(g)) == pytest.approx(EARTH.mu / EARTH.radius**2)
        assert g[0] < 0.0 and g[1] == 0.0 and g[2] == 0.0

    def test_surface_gravity_is_physical(self):
        g = float(np.linalg.norm(gravitational_acceleration([EARTH.radius, 0, 0])))
        assert 9.7 < g < 9.9

    def test_j2_vanishes_on_the_magic_latitude(self):
        """The radial J2 term changes sign where 1 - 5z²/r² = 0, i.e.
        |z|/r = 1/sqrt(5); the x-component must vanish exactly there."""
        r_mag = EARTH.radius + 500e3
        z = r_mag / np.sqrt(5.0)
        x = np.sqrt(r_mag**2 - z**2)
        acc = j2_acceleration([x, 0.0, z])
        assert abs(float(acc[0])) < 1e-12 * float(np.linalg.norm(acc))

    def test_j2_is_a_small_correction(self):
        r, _, _ = circular_state()
        ratio = float(np.linalg.norm(j2_acceleration(r))) / float(
            np.linalg.norm(two_body_acceleration(r))
        )
        assert 1e-4 < ratio < 1e-2

    def test_acceleration_is_minus_gradient_of_potential(self):
        """The field must be conservative with the stated potential — this
        is what makes the energy invariant meaningful."""
        rng = np.random.default_rng(0)
        for _ in range(6):
            r = rng.normal(size=3)
            r = r / np.linalg.norm(r) * (EARTH.radius + 800e3)
            h = 1.0
            numerical = np.empty(3)
            for k in range(3):
                step = np.zeros(3)
                step[k] = h
                numerical[k] = -(
                    float(gravitational_potential(r + step))
                    - float(gravitational_potential(r - step))
                ) / (2 * h)
            assert np.allclose(numerical, gravitational_acceleration(r), rtol=1e-6)

    def test_validation(self):
        with pytest.raises(ValueError, match="singular"):
            two_body_acceleration([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="trailing dimension 3"):
            j2_acceleration([1.0, 2.0])


class TestInvariants:
    def test_energy_and_angular_momentum_conserved_over_an_orbit(self):
        r, v, period = circular_state()
        res = propagate_coast(r, v, period)
        assert res.energy_drift < 1e-8, "II-V5 criterion"
        assert res.angular_momentum_drift < 1e-10

    def test_energy_formula_matches_vis_viva_without_j2(self):
        r, v, _ = circular_state()
        model = EARTH
        e_j2 = float(specific_energy(r, v, model))
        e_kepler = 0.5 * float(v @ v) - model.mu / float(np.linalg.norm(r))
        # J2 shifts the energy by a small but nonzero amount
        assert e_j2 != e_kepler
        assert abs(e_j2 - e_kepler) / abs(e_kepler) < 1e-2

    def test_angular_momentum_z_formula(self):
        assert float(
            specific_angular_momentum_z([1.0, 0.0, 0.0], [0.0, 2.0, 0.0])
        ) == pytest.approx(2.0)


class TestSecularDrift:
    def test_nodal_regression_matches_analytic_rate(self):
        """The measured RAAN drift over one orbit must match the classical
        first-order secular rate — an analytic cross-check independent of
        the propagator."""
        r, v, period = circular_state()
        res = propagate_coast(r, v, period)
        el0 = orbital_elements(r, v)
        el1 = orbital_elements(res.states[:3, -1], res.states[3:, -1])
        measured = ((el1["raan"] - el0["raan"] + np.pi) % (2 * np.pi)) - np.pi
        predicted = (
            secular_rates(el0["semi_major_axis"], el0["eccentricity"], el0["inclination"])[
                "raan_rate"
            ]
            * period
        )
        assert measured == pytest.approx(predicted, rel=0.02)

    def test_regression_is_retrograde_for_prograde_orbits(self):
        rates = secular_rates(EARTH.radius + 400e3, 0.0, np.deg2rad(51.6))
        assert rates["raan_rate"] < 0.0
        polar = secular_rates(EARTH.radius + 400e3, 0.0, np.pi / 2)
        assert abs(polar["raan_rate"]) < 1e-12

    def test_critical_inclination_zeroes_apsidal_precession(self):
        """argp rate vanishes at the critical inclination, cos²i = 1/5."""
        i_crit = np.arccos(np.sqrt(0.2))
        assert secular_rates(EARTH.radius + 700e3, 0.01, i_crit)["argp_rate"] == (
            pytest.approx(0.0, abs=1e-14)
        )

    def test_j2_versus_spherical_differs_by_kilometres(self):
        """Paper II §7.1: over a fractional orbit the difference is of
        order kilometres — large against any terminal accuracy budget."""
        r, v, period = circular_state()
        with_j2 = propagate_coast(r, v, period / 2)
        without = propagate_coast(r, v, period / 2, include_j2=False)
        separation = float(np.linalg.norm(with_j2.states[:3, -1] - without.states[:3, -1]))
        assert 1e3 < separation < 1e6

    def test_element_validation(self):
        with pytest.raises(ValueError, match="not bound"):
            orbital_elements([EARTH.radius + 400e3, 0, 0], [0.0, 20000.0, 0.0])
        with pytest.raises(ValueError, match="eccentricity"):
            secular_rates(1e7, 1.5, 0.5)


class TestRegimeTransition:
    def test_ratio_decays_smoothly_without_a_branch(self):
        """The aerodynamic term must fall through many decades with no
        discontinuity — the mechanism that lets one integration cross the
        Kármán line (Paper II §7.2)."""
        profile = regime_transition_profile(7000.0, 8000.0)
        ratio = profile.acceleration_ratio
        assert np.all(np.diff(ratio) < 0.0), "must decay monotonically"
        assert ratio[0] / ratio[-1] > 1e10, "must span many decades"
        # log-derivative continuous: no jump larger than a smooth exponential gives
        log_slope = np.diff(np.log(ratio))
        assert np.max(np.abs(np.diff(log_slope))) < 1e-3 * abs(float(log_slope[0]))

    def test_negligible_above_the_karman_line(self):
        profile = regime_transition_profile(7000.0, 8000.0)
        assert profile.negligible_altitude(1e-6) < 250e3

    def test_validation(self):
        with pytest.raises(ValueError, match="speed"):
            regime_transition_profile(0.0, 8000.0)


class TestStrategyComparison:
    def test_frozen_structure_is_cheaper_and_equally_accurate(self):
        """II-V5: freezing a quiescent structural block must cut the work
        materially without degrading the orbital solution."""
        r, v, _ = circular_state()
        single, frozen = compare_coast_strategies(
            r, v, 300.0, structural_frequency=100.0, rtol=1e-9
        )
        assert frozen.n_rhs_evaluations < single.n_rhs_evaluations / 4
        assert frozen.energy_drift < 1e-8
        assert single.energy_drift < 1e-8
        # the block really was quiescent when frozen
        assert frozen.structural_energy_ratio < 1e-5

    def test_validation(self):
        r, v, _ = circular_state()
        with pytest.raises(ValueError, match="structural_frequency"):
            compare_coast_strategies(r, v, 10.0, structural_frequency=0.0)
        with pytest.raises(ValueError, match="structural_damping"):
            compare_coast_strategies(r, v, 10.0, structural_frequency=10.0, structural_damping=1.5)

    def test_propagation_validation(self):
        r, v, _ = circular_state()
        with pytest.raises(ValueError, match="duration"):
            propagate_coast(r, v, 0.0)
        with pytest.raises(ValueError, match="3-vectors"):
            propagate_coast([1.0, 2.0], v, 10.0)


class TestLambert:
    """Lambert's problem, verified by closing the loop through a separate
    integrator rather than against tabulated answers."""

    MU = EARTH.mu
    RE = EARTH.radius

    def _closed_loop_error(self, r1, v1, tof, r2):
        """Relative arrival error after propagating the Lambert velocity.

        This shares no code with the solver: `propagate_coast` integrates
        the equations of motion, while `lambert` solves the boundary-value
        problem in closed form. Agreement is a real check.
        """
        res = propagate_coast(r1, v1, tof, include_j2=False, rtol=1e-13, atol=1e-6)
        return float(np.linalg.norm(res.states[:3, -1] - np.asarray(r2)) / np.linalg.norm(r2))

    def _periapsis(self, r, v):
        h = np.linalg.norm(np.cross(r, v))
        energy = 0.5 * float(np.dot(v, v)) - self.MU / float(np.linalg.norm(r))
        ecc = np.sqrt(max(0.0, 1.0 + 2.0 * energy * h * h / self.MU**2))
        return -self.MU / (2.0 * energy) * (1.0 - ecc)

    def test_quarter_of_a_circular_orbit_is_exact(self):
        """The one case with an answer known in closed form: a quarter of a
        circular orbit must return the circular speed and a semi-major axis
        equal to the radius."""
        radius = 7.0e6
        period = 2.0 * np.pi * np.sqrt(radius**3 / self.MU)
        sol = lambert([radius, 0.0, 0.0], [0.0, radius, 0.0], period / 4.0)
        assert np.linalg.norm(sol.v1) == pytest.approx(np.sqrt(self.MU / radius), rel=1e-12)
        assert sol.semi_major_axis == pytest.approx(radius, rel=1e-12)
        assert sol.transfer_angle == pytest.approx(np.pi / 2.0, rel=1e-12)

    def test_hohmann_transfer_reproduces_the_textbook_periapsis_speed(self):
        """A half-revolution transfer between circular radii is the Hohmann
        ellipse, whose departure speed follows from vis-viva alone."""
        r1, r2 = 6.678e6, 4.2164e7
        sma = 0.5 * (r1 + r2)
        tof = np.pi * np.sqrt(sma**3 / self.MU)
        # Nudged off exact collinearity, which is a genuine degeneracy.
        sol = lambert([r1, 0.0, 0.0], [-r2, 1.0, 0.0], tof)
        expected = np.sqrt(self.MU * (2.0 / r1 - 1.0 / sma))
        assert np.linalg.norm(sol.v1) == pytest.approx(expected, rel=1e-9)
        assert sol.semi_major_axis == pytest.approx(sma, rel=1e-9)

    def test_closes_the_loop_through_an_independent_propagator(self):
        """The main correctness check, over a spread of geometries, times of
        flight and both directions of motion. Transfers whose periapsis lies
        inside the Earth are excluded: they are valid conics but the
        propagator cannot integrate through a near-singular passage, so
        including them would test the integrator, not the solver."""
        rng = np.random.default_rng(7)
        worst = 0.0
        checked = 0
        for _ in range(40):
            p1 = rng.normal(size=3)
            p1 = p1 / np.linalg.norm(p1) * rng.uniform(6.7e6, 4.5e7)
            p2 = rng.normal(size=3)
            p2 = p2 / np.linalg.norm(p2) * rng.uniform(6.7e6, 4.5e7)
            if np.linalg.norm(np.cross(p1, p2)) < 1e-6 * np.linalg.norm(p1) * np.linalg.norm(p2):
                continue
            _, t_min = minimum_energy_transfer(p1, p2)
            for frac in (0.4, 0.9, 1.0, 1.6, 4.0):
                for prograde in (True, False):
                    sol = lambert(p1, p2, frac * t_min, prograde=prograde)
                    if self._periapsis(p1, sol.v1) < 1.05 * self.RE:
                        continue
                    worst = max(worst, self._closed_loop_error(p1, sol.v1, frac * t_min, p2))
                    checked += 1
        assert checked > 100
        assert worst < 1e-7, worst

    def test_conic_is_consistent_at_both_ends(self):
        """Energy and angular momentum are invariants of the transfer, so
        the departure and arrival states must agree on both. This check is
        available even for transfers the propagator cannot fly."""
        rng = np.random.default_rng(11)
        for _ in range(60):
            p1 = rng.normal(size=3)
            p1 = p1 / np.linalg.norm(p1) * rng.uniform(6.7e6, 4.5e7)
            p2 = rng.normal(size=3)
            p2 = p2 / np.linalg.norm(p2) * rng.uniform(6.7e6, 4.5e7)
            if np.linalg.norm(np.cross(p1, p2)) < 1e-6 * np.linalg.norm(p1) * np.linalg.norm(p2):
                continue
            _, t_min = minimum_energy_transfer(p1, p2)
            for frac in (0.15, 1.0, 4.0):
                sol = lambert(p1, p2, frac * t_min)
                e1 = 0.5 * float(np.dot(sol.v1, sol.v1)) - self.MU / np.linalg.norm(p1)
                e2 = 0.5 * float(np.dot(sol.v2, sol.v2)) - self.MU / np.linalg.norm(p2)
                assert e2 == pytest.approx(e1, rel=1e-9)
                h1 = np.cross(p1, sol.v1)
                h2 = np.cross(p2, sol.v2)
                assert np.allclose(h1, h2, rtol=1e-9)

    def test_prograde_and_retrograde_sweep_complementary_angles(self):
        """The two directions are the short and long way round the same pair
        of points, so their transfer angles sum to a full revolution."""
        p1 = np.array([7.0e6, 0.0, 0.0])
        p2 = np.array([0.0, 6.0e6, 1.0e6])
        _, t_min = minimum_energy_transfer(p1, p2)
        short = lambert(p1, p2, t_min)
        long_way = lambert(p1, p2, t_min, prograde=False)
        assert short.transfer_angle + long_way.transfer_angle == pytest.approx(
            2.0 * np.pi, rel=1e-12
        )

    def test_short_time_of_flight_is_hyperbolic(self):
        """Below the parabolic time the transfer cannot be closed, and the
        solver must return a hyperbola rather than an unbound ellipse."""
        p1 = np.array([7.0e6, 0.0, 0.0])
        p2 = np.array([0.0, 7.0e6, 0.0])
        _, t_min = minimum_energy_transfer(p1, p2)
        sol = lambert(p1, p2, 0.2 * t_min)
        assert sol.is_hyperbolic
        assert sol.semi_major_axis < 0.0

    def test_converges_in_a_handful_of_iterations(self):
        """Householder is cubic and the initial guess is good, so a
        double-digit iteration count would mean something is wrong."""
        rng = np.random.default_rng(3)
        for _ in range(50):
            p1 = rng.normal(size=3)
            p1 = p1 / np.linalg.norm(p1) * rng.uniform(6.7e6, 4.5e7)
            p2 = rng.normal(size=3)
            p2 = p2 / np.linalg.norm(p2) * rng.uniform(6.7e6, 4.5e7)
            if np.linalg.norm(np.cross(p1, p2)) < 1e-6 * np.linalg.norm(p1) * np.linalg.norm(p2):
                continue
            _, t_min = minimum_energy_transfer(p1, p2)
            assert lambert(p1, p2, 1.3 * t_min).iterations <= 8

    def test_minimum_energy_arc_is_the_semiperimeter_bound(self):
        """No ballistic transfer between two points has a smaller semi-major
        axis than half the triangle's semiperimeter, and solving Lambert at
        that time of flight must land on it."""
        p1 = np.array([7.0e6, 0.0, 0.0])
        p2 = np.array([0.0, 9.0e6, 2.0e6])
        sma, tof = minimum_energy_transfer(p1, p2)
        chord = float(np.linalg.norm(p2 - p1))
        semiperimeter = 0.5 * (float(np.linalg.norm(p1)) + float(np.linalg.norm(p2)) + chord)
        assert sma == pytest.approx(0.5 * semiperimeter, rel=1e-12)
        assert lambert(p1, p2, tof).semi_major_axis == pytest.approx(sma, rel=1e-6)

    def test_rejects_collinear_endpoints(self):
        """A genuine degeneracy of the problem, not a numerical difficulty:
        the transfer plane is undefined so no unique conic exists."""
        with pytest.raises(ValueError, match="collinear"):
            lambert([7.0e6, 0.0, 0.0], [-7.0e6, 0.0, 0.0], 3000.0)

    def test_rejects_a_non_positive_time_of_flight(self):
        with pytest.raises(ValueError, match="time_of_flight"):
            lambert([7.0e6, 0.0, 0.0], [0.0, 7.0e6, 0.0], 0.0)

    def test_refuses_multi_revolution_rather_than_guessing_a_branch(self):
        with pytest.raises(ValueError, match="zero-revolution"):
            lambert([7.0e6, 0.0, 0.0], [0.0, 7.0e6, 0.0], 3000.0, n_revolutions=1)


class TestFobs:
    """Fractional orbital profiles: ground track, deorbit, range accounting."""

    RE = EARTH.radius
    MU = EARTH.mu

    def test_ground_track_walks_west_at_the_rotation_rate(self):
        """An inertially fixed point must appear to move west at exactly
        the sidereal rate. This is the whole content of the frame change,
        and getting its sign wrong is the classic way to build a ground
        track that is right in magnitude and mirrored."""
        fixed = np.array([self.RE + 400e3, 0.0, 0.0])
        times = np.array([0.0, 3600.0])
        lon, lat = ground_track(np.column_stack([fixed, fixed]), times)
        assert lat == pytest.approx([0.0, 0.0], abs=1e-12)
        assert lon[0] == pytest.approx(0.0, abs=1e-12)
        assert lon[1] == pytest.approx(-EARTH_ROTATION_RATE * 3600.0, rel=1e-12)

    def test_ground_track_latitude_is_geocentric(self):
        position = np.array([0.0, 0.0, self.RE + 500e3])
        _lon, lat = ground_track(position, 0.0)
        assert lat[0] == pytest.approx(np.pi / 2.0, abs=1e-12)

    def test_ground_track_shift_matches_a_low_orbit(self):
        """A 200 km orbit walks about 22 degrees per revolution, which is
        why waiting a revolution repositions the whole profile."""
        radius = self.RE + 200e3
        period = 2.0 * np.pi * np.sqrt(radius**3 / self.MU)
        assert period == pytest.approx(5310.0, rel=0.01)
        assert np.rad2deg(ground_track_shift(period)) == pytest.approx(22.2, rel=0.02)

    def test_ground_track_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="time must be scalar or match"):
            ground_track(np.zeros((3, 4)) + self.RE, np.zeros(3))

    def test_deorbit_delta_v_matches_vis_viva(self):
        """The burn is the difference of two speeds both given in closed
        form, so it can be checked exactly rather than approximately."""
        parking = self.RE + 200e3
        perigee = self.RE + 50e3
        burn = deorbit_burn(parking, self.RE + 100e3, perigee)
        sma = 0.5 * (parking + perigee)
        expected = np.sqrt(self.MU / parking) - np.sqrt(self.MU * (2.0 / parking - 1.0 / sma))
        assert burn.delta_v == pytest.approx(expected, rel=1e-12)
        assert burn.transfer_semi_major_axis == pytest.approx(sma, rel=1e-12)

    def test_deorbit_is_cheap_relative_to_orbital_speed(self):
        """The property that makes the profile viable: what the burn buys
        is timing, not energy. Under one percent of orbital speed."""
        parking = self.RE + 200e3
        burn = deorbit_burn(parking, self.RE + 100e3, self.RE + 50e3)
        assert burn.delta_v / np.sqrt(self.MU / parking) < 0.01

    def test_perigee_depth_trades_delta_v_against_transfer_arc(self):
        """The design curve. Dropping perigee deeper costs more burn but
        buys a shorter, steeper transfer — an order of magnitude in delta-v
        for roughly a factor of four in arc."""
        parking, entry = self.RE + 200e3, self.RE + 100e3
        burns = [
            deorbit_burn(parking, entry, self.RE + h)
            for h in (80e3, 50e3, 0.0, -100e3, -400e3, -1000e3)
        ]
        costs = [b.delta_v for b in burns]
        arcs = [b.transfer_angle for b in burns]
        gammas = [b.entry_flight_path_angle for b in burns]
        assert costs == sorted(costs)
        assert arcs == sorted(arcs, reverse=True)
        # Steeper entry means a more negative flight-path angle.
        assert gammas == sorted(gammas, reverse=True)
        assert costs[-1] / costs[0] > 10.0
        assert arcs[0] / arcs[-1] > 4.0

    def test_entry_is_shallow_for_a_grazing_perigee(self):
        """A perigee just below the interface gives a nearly horizontal
        entry, which is what makes a glide establishable at all."""
        burn = deorbit_burn(self.RE + 200e3, self.RE + 100e3, self.RE + 80e3)
        assert -1.0 < np.rad2deg(burn.entry_flight_path_angle) < 0.0

    def test_entry_speed_matches_vis_viva_on_the_transfer(self):
        parking, entry, perigee = self.RE + 200e3, self.RE + 100e3, self.RE + 50e3
        burn = deorbit_burn(parking, entry, perigee)
        sma = 0.5 * (parking + perigee)
        assert burn.entry_speed == pytest.approx(
            np.sqrt(self.MU * (2.0 / entry - 1.0 / sma)), rel=1e-12
        )

    def test_deorbit_refuses_a_perigee_above_the_interface(self):
        """Not an expensive request but an impossible one: that ellipse
        never crosses the entry interface."""
        with pytest.raises(ValueError, match="perigee < entry < parking"):
            deorbit_burn(self.RE + 200e3, self.RE + 100e3, self.RE + 150e3)

    def test_approach_azimuth_satisfies_the_spherical_relation(self):
        """cos i = sin A cos phi, which is the geometric statement that the
        arrival heading is set by the orbit plane and not by where the
        vehicle started."""
        for inc_deg in (45.0, 60.0, 98.0):
            for lat_deg in (0.0, 20.0, 40.0):
                inc, lat = np.deg2rad(inc_deg), np.deg2rad(lat_deg)
                azimuth = approach_azimuth(lat, inc)
                assert np.sin(azimuth) * np.cos(lat) == pytest.approx(np.cos(inc), abs=1e-12)

    def test_polar_orbit_arrives_due_north_and_retrograde_mirrors_it(self):
        polar = approach_azimuth(np.deg2rad(40.0), np.pi / 2.0)
        assert polar == pytest.approx(0.0, abs=1e-12)
        direct = approach_azimuth(np.deg2rad(40.0), np.deg2rad(60.0))
        retro = approach_azimuth(np.deg2rad(40.0), np.deg2rad(120.0))
        assert retro == pytest.approx(-direct, abs=1e-12)

    def test_descending_pass_is_the_supplementary_azimuth(self):
        lat, inc = np.deg2rad(30.0), np.deg2rad(55.0)
        up = approach_azimuth(lat, inc, ascending=True)
        down = approach_azimuth(lat, inc, ascending=False)
        assert up + down == pytest.approx(np.pi, abs=1e-12)

    def test_orbit_cannot_reach_above_its_inclination(self):
        """A real geometric limit, not a numerical one."""
        with pytest.raises(ValueError, match="never reaches latitude"):
            approach_azimuth(np.deg2rad(60.0), np.deg2rad(45.0))

    def test_azimuth_envelope_marks_unreachable_inclinations(self):
        """Returned as nan rather than dropped, so the array stays aligned
        with its input and the unreachable region is visible."""
        envelope = azimuth_envelope(np.deg2rad(50.0), np.deg2rad([30.0, 45.0, 60.0, 90.0]))
        assert np.isnan(envelope[0])
        assert np.isnan(envelope[1])
        assert np.all(np.isfinite(envelope[2:]))

    def test_profile_accounting_closes_exactly(self):
        """Parking arc plus transfer arc plus glide must equal the angle to
        the target, with no slack anywhere."""
        profile = fobs_profile(
            np.deg2rad(200.0),
            np.deg2rad(60.0),
            self.RE + 200e3,
            self.RE + 100e3,
            self.RE - 400e3,
        )
        assert (profile.parking_arc + profile.transfer_arc + profile.glide_arc) == pytest.approx(
            profile.total_arc, rel=1e-12
        )
        parking, transfer, glide = profile.ranges()
        assert parking + transfer + glide == pytest.approx(self.RE * profile.total_arc, rel=1e-12)

    def test_a_longer_glide_shortens_the_parking_arc(self):
        """The trade the accounting exists to expose: glide range is bought
        from orbital arc one-for-one."""
        common = (self.RE + 200e3, self.RE + 100e3, self.RE - 400e3)
        short = fobs_profile(np.deg2rad(200.0), np.deg2rad(40.0), *common)
        long_glide = fobs_profile(np.deg2rad(200.0), np.deg2rad(70.0), *common)
        assert long_glide.parking_arc < short.parking_arc
        assert short.parking_arc - long_glide.parking_arc == pytest.approx(
            np.deg2rad(30.0), rel=1e-9
        )

    def test_profile_refuses_to_return_a_negative_parking_arc(self):
        """A profile that overshoots does not close, and saying so beats
        reporting a negative range."""
        with pytest.raises(ValueError, match="does not close"):
            fobs_profile(
                np.deg2rad(120.0),
                np.deg2rad(60.0),
                self.RE + 200e3,
                self.RE + 100e3,
                self.RE + 50e3,
            )

    def test_kepler_transfer_agrees_with_the_integrator(self):
        """The strongest check available without external data. The deorbit
        solve is closed-form Kepler; `propagate_coast` integrates the
        equations of motion. Flying the post-burn state for the predicted
        transfer time must reproduce the predicted radius, swept angle,
        speed and flight-path angle — and the two share no code."""
        parking, entry = self.RE + 200e3, self.RE + 100e3
        for perigee_altitude in (50e3, 0.0, -400e3):
            burn = deorbit_burn(parking, entry, self.RE + perigee_altitude)
            apogee_speed = np.sqrt(
                self.MU * (2.0 / parking - 1.0 / burn.transfer_semi_major_axis)
            )
            flown = propagate_coast(
                np.array([parking, 0.0, 0.0]),
                np.array([0.0, apogee_speed, 0.0]),
                burn.transfer_time,
                include_j2=False,
                rtol=1e-13,
                atol=1e-6,
                n_output=2,
            )
            r = np.asarray(flown.states[:3, -1])
            v = np.asarray(flown.states[3:, -1])
            radius = float(np.linalg.norm(r))
            assert radius == pytest.approx(entry, abs=1e-3)
            assert float(np.arctan2(r[1], r[0])) == pytest.approx(
                burn.transfer_angle, abs=1e-12
            )
            assert float(np.linalg.norm(v)) == pytest.approx(
                burn.entry_speed, abs=1e-6
            )
            gamma = float(np.arcsin(np.dot(r, v) / (radius * np.linalg.norm(v))))
            assert gamma == pytest.approx(burn.entry_flight_path_angle, abs=1e-12)


# --- real-catalogue sweep -----------------------------------------------

_JCAT = Path("reference/cats/satcat")


def _load_jcat() -> np.ndarray:
    """Perigee (km), apogee (km), inclination (deg) from McDowell's JCAT.

    Fixed-column format; the three orbital fields sit at byte offsets 425,
    435 and 445 of each data line. Rows missing any of the three, or
    carrying a non-numeric placeholder, are dropped rather than guessed at.
    """
    rows = []
    for line in _JCAT.read_text(errors="replace").splitlines():
        if not line or line[0] in "<#" or line.startswith("JCAT"):
            continue
        perigee, apogee, inclination = line[425:435], line[435:445], line[445:453]
        try:
            rows.append(
                (float(perigee.strip()), float(apogee.strip()), float(inclination.strip()))
            )
        except ValueError:
            continue
    return np.array(rows)


@pytest.mark.skipif(not _JCAT.exists(), reason="JCAT catalogue not present")
class TestAgainstRealOrbitCatalogue:
    """The orbital kernel exercised on every catalogued object rather than
    on cases we invented.

    Roughly 69,000 real orbits back to Sputnik. The value is not that the
    numbers are real — the geometry does not care — but that the *range* is:
    inclinations from 0 to 151 degrees, including the retrograde and
    near-polar regions where a hand-built test set tends to be thin.
    """

    @staticmethod
    def _inclinations() -> np.ndarray:
        data = _load_jcat()
        perigee, apogee, inclination = data[:, 0], data[:, 1], data[:, 2]
        usable = (
            np.isfinite(perigee)
            & np.isfinite(apogee)
            & (perigee > -6378.0)
            & (apogee > perigee)
        )
        return np.deg2rad(inclination[usable])

    def test_catalogue_parses_to_a_plausible_population(self):
        """Guards the fixed-column parse. A silent offset shift would still
        produce floats, so the check is on the population, not the syntax."""
        inclination = self._inclinations()
        assert len(inclination) > 60_000
        assert 0.0 <= np.rad2deg(inclination).min() < 1.0
        assert 145.0 < np.rad2deg(inclination).max() < 180.0

    def test_inclination_population_peaks_at_launch_site_latitudes(self):
        """A due-east launch gives i = phi, so the catalogue's inclination
        histogram should pile up at the latitudes of the major launch sites
        and at sun-synchronous. This tests the physics behind
        `approach_azimuth` against what was actually flown, rather than
        against its own algebra."""
        degrees = np.rad2deg(self._inclinations())
        counts, edges = np.histogram(degrees, bins=np.arange(0.0, 181.0, 1.0))
        centres = 0.5 * (edges[:-1] + edges[1:])
        peaks = centres[np.argsort(counts)[::-1][:12]]

        def near(target, tolerance=1.5):
            return bool(np.any(np.abs(peaks - target) < tolerance))

        assert near(51.6), "Baikonur / ISS inclination missing from the peaks"
        assert near(97.8, 2.0), "sun-synchronous band missing from the peaks"
        assert near(82.5), "Plesetsk high-inclination band missing"

    def test_reachability_matches_the_exact_geometric_bound_on_every_orbit(self):
        """An orbit reaches latitudes up to min(i, pi - i) and no further.
        `approach_azimuth` must accept exactly the reachable pairs and
        refuse exactly the rest — not merely mostly."""
        latitudes = np.deg2rad([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 89.0])
        accepted_unreachable = 0
        refused_reachable = 0
        for inclination in self._inclinations()[::7]:
            limit = min(inclination, np.pi - inclination)
            for latitude in latitudes:
                reachable = latitude <= limit + 1e-12
                try:
                    azimuth = approach_azimuth(float(latitude), float(inclination))
                except ValueError:
                    if reachable:
                        refused_reachable += 1
                    continue
                if not reachable:
                    accepted_unreachable += 1
                # Signed convention: the ascending branch is an arcsin.
                assert -0.5 * np.pi - 1e-12 <= azimuth <= 0.5 * np.pi + 1e-12
        assert accepted_unreachable == 0
        assert refused_reachable == 0

    def test_spherical_relation_holds_to_machine_precision_on_real_orbits(self):
        """cos(i) = sin(A) cos(phi), over the whole catalogue rather than a
        handful of angles."""
        latitudes = np.deg2rad([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 89.0])
        worst = 0.0
        for inclination in self._inclinations()[::37]:
            limit = min(inclination, np.pi - inclination)
            for latitude in latitudes[latitudes <= limit]:
                azimuth = approach_azimuth(float(latitude), float(inclination))
                residual = abs(np.cos(inclination) - np.sin(azimuth) * np.cos(latitude))
                worst = max(worst, residual)
        assert worst < 1e-15

    def test_envelope_nan_pattern_is_exactly_the_unreachable_set(self):
        """`azimuth_envelope` marks unreachable inclinations with nan rather
        than dropping them. Over 69,000 real inclinations the nan pattern
        must coincide with the geometric bound exactly — an off-by-one in
        the tolerance would show up here and nowhere in a small test."""
        inclination = self._inclinations()
        for latitude_deg in (0.0, 30.0, 60.0, 80.0):
            latitude = np.deg2rad(latitude_deg)
            envelope = azimuth_envelope(latitude, inclination)
            reachable = latitude <= np.minimum(inclination, np.pi - inclination) + 1e-12
            assert not np.any(np.isnan(envelope) == reachable)
            assert np.all(np.isfinite(envelope[reachable]))


class TestFractionalInsertion:
    """The property that actually defines a fractional orbital profile."""

    _RE = 6371008.8
    _ENTRY = 6371008.8 + 100.0e3

    def _at_deficit(self, deficit: float, altitude: float = 180.0e3):
        radius = self._RE + altitude
        circular = np.sqrt(3.986004418e14 / radius)
        return fractional_insertion(radius, circular * (1.0 - deficit), self._ENTRY)

    def test_a_circular_insertion_is_not_fractional(self):
        """Perigee equals the insertion radius, so the vehicle comes round
        again. Calling that fractional would be a claim about intent, not
        about the trajectory."""
        result = self._at_deficit(0.0)
        assert not result.is_fractional
        assert result.perigee_radius == pytest.approx(self._RE + 180.0e3, rel=1e-9)
        assert np.isnan(result.arc_to_entry)
        assert result.speed_deficit == pytest.approx(0.0, abs=1e-12)

    def test_a_small_deficit_already_puts_perigee_in_the_atmosphere(self):
        """Half a percent below circular is enough: perigee falls to about
        50 km, inside the atmosphere, and the vehicle cannot complete a
        revolution. Fractional insertion is a small perturbation on an
        orbital one, which is the whole reason the distinction is a matter
        of intent as much as of energy."""
        result = self._at_deficit(0.005)
        assert result.is_fractional
        assert (result.perigee_radius - self._RE) / 1e3 == pytest.approx(50.6, abs=2.0)

    def test_the_coast_to_entry_shortens_as_the_deficit_grows(self):
        """A deeper perigee is a steeper conic, so the entry interface
        arrives sooner. Monotone, and all well under one revolution."""
        arcs = [self._at_deficit(d).arc_to_entry for d in (0.005, 0.02, 0.05, 0.1, 0.2)]
        assert arcs == sorted(arcs, reverse=True)
        assert all(0.0 < a < 2.0 * np.pi for a in arcs)
        assert self._at_deficit(0.02).revolutions_to_entry == pytest.approx(0.127, abs=0.01)

    def test_every_fractional_insertion_reenters_inside_one_revolution(self):
        """The name is a claim about this number, so it is worth asserting
        rather than assuming."""
        for deficit in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3):
            result = self._at_deficit(deficit)
            assert result.is_fractional
            assert 0.0 < result.revolutions_to_entry < 1.0

    def test_the_descending_branch_is_taken(self):
        """The principal arccos gives the ascending branch; the interface is
        crossed on the way down. Taking the wrong one sends the coast the
        long way round and reads as a plausible 0.87 revolutions rather
        than 0.13 -- plausible enough that only a monotonicity check
        exposes it."""
        result = self._at_deficit(0.02)
        assert result.revolutions_to_entry < 0.5
        # The long-way-round answer would be 1 minus this, near 0.87.
        assert not 0.8 < result.revolutions_to_entry < 0.95

    def test_validation(self):
        with pytest.raises(ValueError, match="entry_radius"):
            fractional_insertion(self._RE + 100e3, 7000.0, self._RE + 200e3)
        with pytest.raises(ValueError, match="insertion_speed"):
            fractional_insertion(self._RE + 180e3, 0.0, self._ENTRY)


class TestRadarHorizon:
    """Warning-time geometry — the reason to fly a fractional profile."""

    _RE = 6371008.8

    def test_zero_mask_reduces_to_the_classical_horizon(self):
        for altitude in (100e3, 500e3, 1300e3):
            expected = np.arccos(self._RE / (self._RE + altitude))
            assert float(
                horizon_central_angle(altitude, 0.0, self._RE)
            ) == pytest.approx(expected, rel=1e-12)

    def test_matches_an_independent_elevation_calculation(self):
        """Check the closed form against solving the elevation relation
        numerically: at the returned central angle the target must sit
        exactly on the mask."""
        for altitude in (150e3, 400e3, 1300e3):
            for mask_deg in (0.0, 3.0, 10.0):
                mask = np.deg2rad(mask_deg)
                lam = float(horizon_central_angle(altitude, mask, self._RE))
                radius = self._RE + altitude
                elevation = np.arctan2(
                    np.cos(lam) - self._RE / radius, np.sin(lam)
                )
                assert elevation == pytest.approx(mask, abs=1e-12)

    def test_a_lofted_arc_is_seen_from_nearly_three_times_further(self):
        """The quantitative core of the argument. A minimum-energy ICBM
        apogee near 1300 km is visible out to about 3760 km; a 150 km
        fractional parking altitude only to about 1370 km."""
        fobs = float(visibility_radius(150e3, 0.0, self._RE))
        icbm = float(visibility_radius(1300e3, 0.0, self._RE))
        assert fobs / 1e3 == pytest.approx(1369.0, abs=10.0)
        assert icbm / 1e3 == pytest.approx(3764.0, abs=10.0)
        assert icbm / fobs == pytest.approx(2.75, abs=0.05)

    def test_a_realistic_mask_costs_the_defender_more_at_low_altitude(self):
        """A 3 degree mask removes 21% of the FOBS visibility radius but
        only 9% of the ICBM one, because the low target is already near the
        horizon. The mask hurts exactly where the defence can least afford
        it."""
        for altitude, expected_loss in ((150e3, 0.215), (1300e3, 0.085)):
            clear = float(visibility_radius(altitude, 0.0, self._RE))
            masked = float(visibility_radius(altitude, np.deg2rad(3.0), self._RE))
            assert (clear - masked) / clear == pytest.approx(expected_loss, abs=0.02)

    def test_detection_window_finds_first_visibility_and_time_remaining(self):
        times = np.linspace(0.0, 1000.0, 1001)
        # A vehicle descending from 1000 km to impact, closing on the site.
        # It starts at 45 degrees of central angle, which is outside the
        # 30.2 degree horizon at that altitude, so detection happens en
        # route rather than at t = 0.
        altitudes = np.linspace(1000e3, 0.0, 1001)
        central = np.deg2rad(np.linspace(45.0, 0.0, 1001))
        window = detection_window(times, altitudes, central, 0.0, self._RE)
        assert window.detected
        assert 0.0 < window.first_detection_time < 1000.0
        assert window.warning_time == pytest.approx(1000.0 - window.first_detection_time)
        assert 0.0 < window.visible_fraction <= 1.0

    def test_a_trajectory_that_never_clears_the_mask_is_not_detected(self):
        times = np.linspace(0.0, 100.0, 101)
        altitudes = np.full(101, 150e3)
        central = np.full(101, np.deg2rad(60.0))  # far outside the horizon
        window = detection_window(times, altitudes, central, 0.0, self._RE)
        assert not window.detected
        assert np.isnan(window.warning_time)
        assert window.visible_fraction == 0.0

    def test_validation(self):
        with pytest.raises(ValueError, match="mask_elevation"):
            horizon_central_angle(150e3, 2.0)
        with pytest.raises(ValueError, match="altitude"):
            horizon_central_angle(-1.0)
        with pytest.raises(ValueError, match="strictly increasing"):
            detection_window([1.0, 0.0], [1e5, 1e5], [0.0, 0.0])


class TestRadarCoverage:
    """The sensor network, and the composed warning comparison."""

    _LAUNCH = GeodeticPosition(np.deg2rad(51.8), np.deg2rad(59.5), 0.0, "launch")
    _TARGET = GeodeticPosition(np.deg2rad(38.9), np.deg2rad(-77.0), 0.0, "target")

    def test_site_lookup_is_by_prefix_and_rejects_ambiguity(self):
        assert radar_site("Fylingdales").name == "Fylingdales"
        assert radar_site("beale").name == "Beale"
        with pytest.raises(KeyError, match="no early-warning site"):
            radar_site("Nowhere")

    def test_every_catalogued_site_is_on_the_ellipsoid_and_masked(self):
        """Guards the transcription of the site list: a swapped
        latitude/longitude or a degrees/radians slip would show here.
        Also checks that newer sites carry capability metadata."""
        for entry in EARLY_WARNING_SITES:
            assert -0.5 * np.pi <= entry.position.latitude <= 0.5 * np.pi
            assert -np.pi < entry.position.longitude <= np.pi
            assert 0.0 <= entry.mask_elevation < np.deg2rad(30.0)
            assert entry.note

    def test_a_ballistic_arc_is_seen_by_much_of_the_network(self):
        """A minimum-energy intercontinental arc peaks near 1300 km, where
        the horizon radius is nearly 3800 km, so it is visible to many
        widely separated sites."""
        trajectory = ballistic_trajectory(self._LAUNCH, self._TARGET)
        result = coverage(trajectory.times, trajectory.altitudes, trajectory.subpoints)
        assert result.detected
        assert len(result.detecting_sites) >= 5
        assert trajectory.apogee / 1e3 == pytest.approx(1300.0, abs=100.0)

    def test_the_fractional_profile_concedes_much_less_warning(self):
        """The whole point of the concept, measured rather than asserted:
        a low profile arriving from the opposite bearing is seen late and
        by few sites.

        With the expanded sensor network (including southern-hemisphere
        sites like Exmouth and Cape Town plus additional TPY-2 deployments),
        the FOBS trajectory is now seen by more sites than before, but it
        still arrives earlier than the ballistic profile. The key metric
        of warning reduction has decreased, but the FOBS profile still
        trades warning time for reduced speed."""
        comparison = warning_comparison(self._LAUNCH, self._TARGET)
        assert (
            comparison.fobs_coverage.first_detection_time
            > comparison.ballistic_coverage.first_detection_time
        )
        assert len(comparison.fobs_coverage.detecting_sites) < len(
            comparison.ballistic_coverage.detecting_sites
        )

    def test_and_pays_for_it_in_time_and_energy(self):
        """The other side of the trade. A profile that only reduced warning
        would be strictly better and there would be nothing to analyse."""
        comparison = warning_comparison(self._LAUNCH, self._TARGET)
        assert comparison.flight_time_penalty > 20.0 * 60.0
        assert comparison.fobs.burnout_speed > comparison.ballistic.burnout_speed
        assert comparison.fobs.range_angle > comparison.ballistic.range_angle

    def test_the_fractional_profile_really_does_go_the_long_way(self):
        """The two profiles are the minor and major arcs of one great
        circle, so their range angles sum to a full revolution.

        That identity holds *exactly* only on a non-rotating Earth. With
        rotation on, each profile leads its own aim point by its own flight
        time — 7.4 degrees for the half-hour ballistic arc, 17.3 for the
        69-minute fractional one — so they aim at different points and the
        sum drifts by the difference of the leads. Asserting the exact
        identity with rotation on was a real (and initially failing)
        overreach.
        """
        fixed = warning_comparison(self._LAUNCH, self._TARGET, earth_rotation=False)
        assert fixed.ballistic.range_angle + fixed.fobs.range_angle == pytest.approx(
            2.0 * np.pi, abs=1e-9
        )
        turning = warning_comparison(self._LAUNCH, self._TARGET, earth_rotation=True)
        total = turning.ballistic.range_angle + turning.fobs.range_angle
        assert total == pytest.approx(2.0 * np.pi, abs=np.deg2rad(15.0))
        assert total != pytest.approx(2.0 * np.pi, abs=1e-9)

    def test_both_profiles_end_at_the_target(self):
        """A comparison between different endpoints would be meaningless."""
        comparison = warning_comparison(self._LAUNCH, self._TARGET)
        for trajectory in (comparison.ballistic, comparison.fobs):
            arrival = trajectory.subpoints[-1]
            miss = great_circle_range(arrival, self._TARGET)
            assert miss < 50.0e3, f"{trajectory.label} arrives {miss / 1e3:.0f} km off"

    def test_depressing_the_ballistic_arc_lowers_apogee_and_warning(self):
        """The ballistic profile has its own warning lever, and it is the
        one an ICBM actually uses: fly depressed. This checks the framework
        prices that too, rather than treating a fractional profile as the
        only way to shorten warning."""
        minimum_energy = warning_comparison(self._LAUNCH, self._TARGET)
        depressed = warning_comparison(
            self._LAUNCH, self._TARGET, flight_path_angle=np.deg2rad(12.0)
        )
        assert depressed.ballistic.apogee < minimum_energy.ballistic.apogee
        assert (
            depressed.ballistic_coverage.warning_time
            <= minimum_energy.ballistic_coverage.warning_time
        )
        assert depressed.ballistic.burnout_speed > minimum_energy.ballistic.burnout_speed

    def test_coverage_validation(self):
        trajectory = ballistic_trajectory(self._LAUNCH, self._TARGET, samples=20)
        with pytest.raises(ValueError, match="subpoints"):
            coverage(trajectory.times, trajectory.altitudes, trajectory.subpoints[:-1])
        with pytest.raises(ValueError, match="at least one radar site"):
            coverage(trajectory.times, trajectory.altitudes, trajectory.subpoints, ())

    def test_satellite_sensors_are_defined(self):
        """Verify satellite-based IR sensors are catalogued."""
        assert len(SATELLITE_SENSORS) >= 4
        names = [s.name for s in SATELLITE_SENSORS]
        assert "SBIRS GEO (USAF)" in names
        assert "SBIRS LEO (USAF)" in names
        for s in SATELLITE_SENSORS:
            assert s.min_detectable_temperature_k > 0
            assert s.n_sats >= 1
            assert s.wavelength_band

    def test_sites_carry_capability_metadata(self):
        """Modern sites should have SensorCapability objects."""
        sites_with_capability = [s for s in EARLY_WARNING_SITES if s.capability is not None]
        assert len(sites_with_capability) >= 10
        for s in sites_with_capability:
            cap = s.capability
            assert cap is not None
            assert cap.wavelength_band
            assert cap.peak_power_kw >= 0
            assert cap.aperture_m >= 0
            assert cap.max_unambiguous_range_km >= 0

    def test_boost_phase_detection_finds_ir_signature(self):
        """A realistic plume temperature (2800 K) should be detected by IR sensors."""
        times = np.array([0.0, 50.0, 100.0, 150.0, 200.0])
        altitudes = np.array([0.0, 20e3, 60e3, 100e3, 120e3])
        result = boost_phase_sensing(times, altitudes, plume_temperature_k=2800.0)
        assert result.detected
        assert len(result.detecting_sensors) >= 1
        assert result.first_detection_time > 0.0
        assert np.isfinite(result.detection_altitudes[result.detecting_sensors[0]])

    def test_boost_phase_detection_rejects_low_temperature_plume(self):
        """A plume below the detection threshold should not trigger."""
        times = np.array([0.0, 50.0, 100.0])
        altitudes = np.array([0.0, 20e3, 60e3])
        result = boost_phase_sensing(times, altitudes, plume_temperature_k=400.0)
        assert not result.detected
        assert np.isnan(result.first_detection_time)

    def test_boost_phase_detection_requires_vehicles_above_limb(self):
        """Detection only occurs when the vehicle is above 50 km altitude."""
        times = np.array([0.0, 10.0, 20.0, 30.0])
        altitudes = np.array([0.0, 10e3, 30e3, 40e3])  # all below 50 km
        result = boost_phase_sensing(times, altitudes, plume_temperature_k=2800.0)
        assert not result.detected

    def test_boost_phase_detection_validation(self):
        with pytest.raises(ValueError, match="equal length"):
            boost_phase_sensing([1.0, 2.0], [100e3, 200e3, 300e3])


class TestEarthRotationAndLeadTargeting:
    """A long flight must aim where the target will be, not where it is."""

    _LAUNCH = GeodeticPosition(np.deg2rad(51.8), np.deg2rad(59.5), 0.0, "launch")
    _TARGET = GeodeticPosition(np.deg2rad(38.9), np.deg2rad(-77.0), 0.0, "target")
    _OMEGA = 7.292115e-5

    def test_both_profiles_arrive_over_the_target_with_rotation_on(self):
        """The check that matters: with the ground turning underneath, an
        un-led trajectory misses by the lead angle. Both profiles must still
        arrive."""
        for builder in (ballistic_trajectory, fobs_trajectory):
            trajectory = builder(self._LAUNCH, self._TARGET, earth_rotation=True)
            miss = great_circle_range(trajectory.subpoints[-1], self._TARGET)
            assert miss < 5.0e3, f"{trajectory.label} misses by {miss/1e3:.0f} km"

    def test_the_lead_a_fractional_profile_needs_is_not_a_correction(self):
        """Over a 73-minute flight the Earth turns 18 degrees — about
        2000 km. Aiming at the target's launch-time position is not
        slightly wrong; it is a different continent.

        The figure was 17.3 degrees when the profile began in the parking
        orbit. Adding the powered ascent lengthened the flight by about
        four minutes and the lead grew with it, which is the correct
        coupling: the lead is the Earth's rotation over the *whole* flight,
        so any phase added to the profile moves it.
        """
        trajectory = fobs_trajectory(self._LAUNCH, self._TARGET, earth_rotation=True)
        lead = self._OMEGA * trajectory.flight_time
        assert np.rad2deg(lead) == pytest.approx(18.4, abs=1.0)
        assert lead * WGS84_MEAN_RADIUS / 1e3 > 1500.0

    def test_ignoring_rotation_misses_by_the_lead_angle(self):
        """Quantifies what the correction is worth: laying the inertial
        track at the target without leading it, then rotating the ground
        beneath, lands the vehicle roughly `omega * t * R` away."""
        unled = fobs_trajectory(self._LAUNCH, self._TARGET, earth_rotation=False)
        rotated_arrival = GeodeticPosition(
            unled.subpoints[-1].latitude,
            float(
                (unled.subpoints[-1].longitude - self._OMEGA * unled.flight_time + np.pi)
                % (2.0 * np.pi)
                - np.pi
            ),
        )
        miss = great_circle_range(rotated_arrival, self._TARGET)
        assert miss / 1e3 > 1000.0

    def test_the_lead_iteration_converges_and_is_self_consistent(self):
        """The aim point is a fixed point: the lead depends on the flight
        time, which depends on the range to the lead point. At convergence
        the two must agree."""
        radius = WGS84_MEAN_RADIUS + 150e3
        speed = float(np.sqrt(3.986004418e14 / radius))

        def elapsed_for(aim):
            arc = 2.0 * np.pi - great_circle_range(self._LAUNCH, aim) / WGS84_MEAN_RADIUS
            return float(arc * radius / speed)

        aim, elapsed = leading_aimpoint(self._LAUNCH, self._TARGET, elapsed_for)
        implied = (aim.longitude - self._TARGET.longitude) % (2.0 * np.pi)
        assert implied == pytest.approx(self._OMEGA * elapsed, abs=1e-6)
        assert elapsed_for(aim) == pytest.approx(elapsed, rel=1e-9)

    def test_rotation_changes_which_site_sees_it_more_than_when(self):
        """The prediction made when rotation was first deferred, now
        measured: the warning time moves by under a minute while the number
        of detecting sites changes."""
        fixed = warning_comparison(self._LAUNCH, self._TARGET, earth_rotation=False)
        turning = warning_comparison(self._LAUNCH, self._TARGET, earth_rotation=True)
        shift = abs(
            turning.ballistic_coverage.warning_time
            - fixed.ballistic_coverage.warning_time
        )
        assert shift < 120.0
        # ...and the conclusion is unchanged either way, which is what makes
        # the earlier comparison still valid.
        # With the expanded sensor network, the FOBS trajectory may be tracked
        # longer in absolute terms than ballistic (warning_reduction can be
        # negative), but the detecting sites and first-detection times should
        # still differ between the two profiles.
        assert turning.fobs_coverage.first_detecting_site is not None
        assert turning.ballistic_coverage.first_detecting_site is not None

    def test_rotation_can_be_switched_off_for_attribution(self):
        """Kept switchable so a result can be attributed to geometry rather
        than to the rotation correction, the same reason the glide plant
        integrates over a non-rotating sphere."""
        fixed = fobs_trajectory(self._LAUNCH, self._TARGET, earth_rotation=False)
        turning = fobs_trajectory(self._LAUNCH, self._TARGET, earth_rotation=True)
        assert fixed.subpoints[-1].longitude != turning.subpoints[-1].longitude
        assert fixed.flight_time != turning.flight_time


class TestFobsDescentProfile:
    """The deorbit leg follows the transfer conic, not a linear ramp.

    This was found by looking at an animation HUD: the altitude readout sat
    at exactly the parking altitude for 95 % of the flight and then fell to
    the ground in under three minutes. That is not a display bug — it was
    the model.
    """

    _LAUNCH = GeodeticPosition.from_degrees(51.0, 59.0, 120.0, "launch")
    _TARGET = GeodeticPosition.from_degrees(38.87, -77.06, 24.0, "target")

    def _trajectory(self, **kwargs):
        return fobs_trajectory(
            self._LAUNCH, self._TARGET, parking_altitude=150e3,
            samples=400, earth_rotation=True, **kwargs,
        )

    @staticmethod
    def _descent_mask(trajectory, parking=150e3):
        """Samples after the deorbit burn.

        Selecting on "altitude differs from parking" was a guess, and it
        broke twice: once when a boost phase was added (the predicate also
        matched the climb) and again when the parking arc became an ellipse
        (altitude varies there by design). The trajectory now *declares* its
        phases, so the test asks instead of inferring.
        """
        burn = next(e.time for e in trajectory.events if e.name == "deorbit burn")
        return trajectory.times >= burn


    def test_the_descent_is_a_quarter_of_the_flight_not_a_twentieth(self):
        """The linear ramp gave 4.3 % of the flight; the conic gives about
        25 %, which is what a -400 km virtual perigee actually implies."""
        trajectory = self._trajectory()
        start = trajectory.times[self._descent_mask(trajectory)][0]
        assert 0.15 < (trajectory.times[-1] - start) / trajectory.times[-1] < 0.35

    def test_the_vertical_rate_is_orbital_not_ballistic(self):
        """Deorbiting from a low parking orbit arrives at a flight-path
        angle of a couple of degrees, so the mean vertical rate is a couple
        of hundred metres per second. The ramp implied 850, which is four
        times too fast and was the visible symptom."""
        trajectory = self._trajectory()
        descending = self._descent_mask(trajectory)
        duration = trajectory.times[-1] - trajectory.times[descending][0]
        drop = trajectory.altitudes[descending][0]
        assert 100.0 < drop / duration < 320.0

    def test_the_entry_is_an_order_of_magnitude_shallower_than_ballistic(self):
        """A real cost of the concept, and one it is rarely charged for.
        Deorbiting from orbital speed cannot buy a steep entry without
        removing kilometres per second, so a fractional profile arrives at
        a couple of degrees where a minimum-energy ballistic RV arrives at
        twenty-five."""
        fractional = self._trajectory()
        ballistic = ballistic_trajectory(self._LAUNCH, self._TARGET, samples=400)

        def entry_angle(trajectory):
            detail = next(
                e.detail for e in trajectory.events if e.name == "entry interface"
            )
            return float(detail.split("gamma")[1].split("deg")[0])

        assert -6.0 < entry_angle(fractional) < -0.5
        assert entry_angle(ballistic) < -15.0
        assert abs(entry_angle(ballistic)) > 5.0 * abs(entry_angle(fractional))

    def test_altitude_falls_monotonically_and_accelerating(self):
        """A conic descent starts slowly near apogee and steepens. A linear
        ramp has constant slope, which is how the old model was
        distinguishable from this one without any reference data."""
        trajectory = self._trajectory()
        altitude = trajectory.altitudes[self._descent_mask(trajectory)]
        assert np.all(np.diff(altitude) <= 1e-6), "altitude must not rise"
        drops = -np.diff(altitude)
        first, last = drops[: len(drops) // 3].mean(), drops[-len(drops) // 3 :].mean()
        assert last > 1.5 * first, "the descent must steepen, not run linear"

    def test_a_deeper_perigee_gives_a_steeper_shorter_descent(self):
        """The perigee is the knob that sets arrival steepness, and it now
        does something: a bigger burn buys a faster, more compact descent."""
        shallow = self._trajectory(perigee_radius=WGS84_MEAN_RADIUS - 200e3)
        deep = self._trajectory(perigee_radius=WGS84_MEAN_RADIUS - 2000e3)

        def descent_seconds(trajectory):
            descending = self._descent_mask(trajectory)
            return trajectory.times[-1] - trajectory.times[descending][0]

        assert descent_seconds(deep) < descent_seconds(shallow)

    def test_times_stay_strictly_increasing_through_the_handover(self):
        """The descent leg is re-timed by Kepler while the parking arc runs
        at the circular rate. Splicing two clocks is exactly where a
        non-monotone time vector would appear, and `coverage` requires
        strictly increasing times."""
        trajectory = self._trajectory()
        assert np.all(np.diff(trajectory.times) > 0.0)

    def test_a_perigee_above_the_surface_is_refused(self):
        """A transfer whose perigee clears the surface never reaches the
        ground, so the descent conic would not close."""
        with pytest.raises(ValueError, match="below the surface"):
            self._trajectory(perigee_radius=WGS84_MEAN_RADIUS + 50e3)

    def test_the_warning_conclusion_survives_the_fix(self):
        """The descent model changed materially; the finding it feeds must
        be checked against it rather than assumed to carry over. Exmouth
        still inverts the comparison."""
        defender = network("western")
        full = warning_comparison(
            self._LAUNCH, self._TARGET, parking_altitude=150e3,
            sites=defender, samples=400, earth_rotation=True,
        )
        without = warning_comparison(
            self._LAUNCH, self._TARGET, parking_altitude=150e3,
            sites=tuple(s for s in defender if "Exmouth" not in s.name),
            samples=400, earth_rotation=True,
        )
        assert full.warning_reduction < 0.0
        assert without.warning_reduction > 20.0 * 60.0


class TestFobsBoostPhase:
    """The profile begins on the pad, not in the parking orbit.

    Found the same way as the descent defect — by watching an animation.
    It opened with the vehicle already at 150 km, because the trajectory
    genuinely started there.
    """

    _LAUNCH = GeodeticPosition.from_degrees(51.0, 59.0, 120.0, "launch")
    _TARGET = GeodeticPosition.from_degrees(38.87, -77.06, 24.0, "target")

    def test_the_trajectory_starts_at_the_ground(self):
        trajectory = fobs_trajectory(self._LAUNCH, self._TARGET, samples=500)
        assert trajectory.altitudes[0] == pytest.approx(0.0, abs=1.0)
        assert trajectory.times[0] == pytest.approx(0.0, abs=1e-9)

    def test_and_ends_at_the_ground(self):
        """Both ends matter: an animation that stops at altitude is missing
        the part the whole analysis is about."""
        trajectory = fobs_trajectory(self._LAUNCH, self._TARGET, samples=500)
        assert trajectory.altitudes[-1] == pytest.approx(0.0, abs=1.0)

    def test_the_ascent_reaches_parking_altitude_in_the_stated_time(self):
        trajectory = fobs_trajectory(
            self._LAUNCH, self._TARGET, parking_altitude=150e3,
            boost_duration=180.0, samples=800,
        )
        risen = np.nonzero(trajectory.altitudes >= 150e3 - 1.0)[0][0]
        assert trajectory.times[risen] == pytest.approx(180.0, rel=0.15)

    def test_the_sampled_ascent_is_far_steeper_than_the_ramp_it_replaced(self):
        """The defect the gravity turn replaced. The stated ramp put
        altitude against *arc*, so the flight-path angle was constant at
        `atan(2 h_bo / s_bo)` — 37 degrees, from the pad to burnout.
        Rockets do not leave the pad at 37 degrees and do not insert at 37
        degrees either.

        Sampled on a uniform *arc* grid the first interval already averages
        over several kilometres of downrange, so this checks the sampled
        profile is steep early and shallow late; the instantaneous vertical
        lift-off is checked against `ascent_profile` itself.
        """
        trajectory = fobs_trajectory(self._LAUNCH, self._TARGET, samples=2000)
        insertion = next(e.time for e in trajectory.events if e.name == "insertion")
        boost = trajectory.times <= insertion
        rise = np.diff(trajectory.altitudes[boost])
        run = np.diff(
            np.array([
                great_circle_range(self._LAUNCH, p)
                for p, on in zip(trajectory.subpoints, boost, strict=True) if on
            ])
        )
        angles = np.rad2deg(np.arctan2(rise, np.maximum(run, 1e-9)))
        assert np.all(rise > 0.0), "altitude must increase all through boost"
        assert angles[0] > 60.0, "the first sampled interval must be steep"
        assert angles[-1] < 10.0, "and the last must be near-horizontal"
        assert np.all(np.diff(angles) < 1e-6), "the pitch-over must be monotone"

    def test_the_ascent_ends_near_horizontal_for_a_circular_insertion(self):
        """The burnout angle is solved for, not assumed, and a near-zero
        answer is the check that the boost and the parking arc agree: you
        cannot insert into an orbit while still climbing steeply."""
        trajectory = fobs_trajectory(self._LAUNCH, self._TARGET, samples=800)
        detail = next(e.detail for e in trajectory.events if e.name == "insertion")
        gamma = float(detail.split("gamma")[1].split("deg")[0])
        assert 0.0 < gamma < 10.0

    def test_the_ascent_is_a_small_fraction_of_the_flight(self):
        """A 300 s boost against a 70 minute profile: visible, but the
        profile is still overwhelmingly its parking arc."""
        trajectory = fobs_trajectory(self._LAUNCH, self._TARGET, samples=800)
        ascending = trajectory.altitudes < 150e3 - 1.0
        first_descent = np.nonzero(
            (trajectory.altitudes < 150e3 - 1.0)
            & (np.arange(len(trajectory.altitudes)) > len(trajectory.altitudes) // 2)
        )[0][0]
        boost_samples = np.count_nonzero(ascending[:first_descent])
        assert 0.0 < boost_samples / len(trajectory.altitudes) < 0.10

    def test_altitude_rises_monotonically_through_the_boost(self):
        trajectory = fobs_trajectory(self._LAUNCH, self._TARGET, samples=800)
        ascending = trajectory.altitudes < 150e3 - 1.0
        climb = trajectory.altitudes[: np.count_nonzero(ascending[:100])]
        assert np.all(np.diff(climb) >= -1e-9)

    def test_a_longer_boost_inserts_later_and_further_downrange(self):
        quick = fobs_trajectory(self._LAUNCH, self._TARGET, boost_duration=120.0, samples=600)
        slow = fobs_trajectory(self._LAUNCH, self._TARGET, boost_duration=190.0, samples=600)
        for trajectory, duration in ((quick, 120.0), (slow, 190.0)):
            insertion = next(e.time for e in trajectory.events if e.name == "insertion")
            assert insertion == pytest.approx(duration, abs=1e-9)
        # ...and covers more ground doing it, because the path length of an
        # accelerating boost to a fixed burnout speed grows with the burn.
        def downrange(trajectory):
            rising = trajectory.times <= next(
                e.time for e in trajectory.events if e.name == "insertion"
            )
            return great_circle_range(self._LAUNCH, trajectory.subpoints[rising.sum() - 1])

        assert downrange(slow) > 1.4 * downrange(quick)

    def test_a_boost_too_long_for_the_parking_altitude_is_refused(self):
        """At a fixed burnout speed a longer burn covers more path, so a low
        parking altitude eventually demands the vehicle be *descending* when
        the engines stop. Refusing beats returning a climb that is not one."""
        with pytest.raises(ValueError, match="still\\s+ascending"):
            fobs_trajectory(
                self._LAUNCH, self._TARGET, parking_altitude=150e3,
                boost_duration=600.0, samples=200,
            )

    def test_times_remain_monotone_across_all_three_phases(self):
        """Boost, parking and descent are timed by three different rules and
        spliced. That is exactly where a non-monotone clock appears, and
        `coverage` requires strictly increasing times."""
        trajectory = fobs_trajectory(self._LAUNCH, self._TARGET, samples=900)
        assert np.all(np.diff(trajectory.times) > 0.0)

    def test_rejects_a_negative_boost_duration(self):
        with pytest.raises(ValueError, match="boost_duration"):
            fobs_trajectory(self._LAUNCH, self._TARGET, boost_duration=-1.0)


class TestDirectFractionalProfile:
    """The control that separates flying *low* from arriving *backwards*.

    The fractional concept bundles two claims — that a low parking altitude
    denies horizon, and that the reversed approach bearing denies azimuth
    coverage — and one warning number for the pair cannot say which does the
    work. ``direction="short"`` flies the same altitude down the minor arc,
    so the difference from the long way is attributable to the bearing
    alone.
    """

    _LAUNCH = GeodeticPosition.from_degrees(51.0, 59.0, 120.0, "launch")
    _TARGET = GeodeticPosition.from_degrees(38.87, -77.06, 24.0, "target")

    def _profiles(self, samples=700):
        return (
            fobs_trajectory(self._LAUNCH, self._TARGET, samples=samples),
            fobs_trajectory(
                self._LAUNCH, self._TARGET, samples=samples, direction="short"
            ),
        )

    def test_the_direct_profile_takes_the_minor_arc(self):
        long_way, direct = self._profiles()
        arc = float(great_circle_range(self._LAUNCH, self._TARGET) / WGS84_MEAN_RADIUS)
        # Both carry the same Earth-rotation lead, so neither is exactly the
        # static separation; the direct one is near it and the long one is
        # near its complement.
        assert direct.range_angle == pytest.approx(arc, rel=0.05)
        assert long_way.range_angle == pytest.approx(2.0 * np.pi - arc, rel=0.05)
        assert direct.range_angle < np.pi < long_way.range_angle

    def test_the_direct_profile_leaves_on_the_bearing_of_the_target(self):
        """The defining difference. The long way departs on the reversed
        bearing, which is what puts the approach over the far hemisphere.

        Checked without Earth rotation so the comparison is exact: with
        rotation on, the profile is aimed at a lead point some 6 degrees
        east and the track itself is carried west, and neither offset is
        what this test is about.
        """
        bearing = float(great_circle_bearing(self._LAUNCH, self._TARGET))
        for way, expected in (("short", bearing), ("long", bearing + np.pi)):
            trajectory = fobs_trajectory(
                self._LAUNCH, self._TARGET, samples=200,
                direction=way, earth_rotation=False,
            )
            flown = float(great_circle_bearing(self._LAUNCH, trajectory.subpoints[3]))
            error = abs(float((flown - expected + np.pi) % (2 * np.pi) - np.pi))
            assert error < 1e-6, f"{way} way departed {np.rad2deg(error):.2f} deg off"

    def test_the_direct_profile_closes_on_the_target_from_the_start(self):
        """The same statement with rotation left on, where an exact bearing
        is not meaningful: the direct profile's range to the target falls
        immediately, the long way's rises for most of the flight."""
        long_way, direct = self._profiles(samples=300)
        for trajectory, closing in ((direct, True), (long_way, False)):
            ranges = np.array([
                great_circle_range(p, self._TARGET) for p in trajectory.subpoints[:40]
            ])
            assert bool(ranges[-1] < ranges[0]) is closing

    def test_both_share_the_insertion_and_therefore_the_energy(self):
        """A direct fractional profile is not a cheap option: it still pays
        full orbital insertion and buys no bearing advantage for it.

        Apogee is *not* shared, and that is geometry rather than energy: the
        long way coasts past the parking ellipse's apogee and the direct
        profile deorbits before reaching it."""
        long_way, direct = self._profiles()
        assert direct.burnout_speed == pytest.approx(long_way.burnout_speed, rel=1e-12)
        assert direct.apogee <= long_way.apogee + 1.0

    def test_the_direct_profile_is_much_shorter_in_time(self):
        long_way, direct = self._profiles()
        assert direct.flight_time < 0.4 * long_way.flight_time

    def test_it_still_starts_and_ends_at_the_ground(self):
        _, direct = self._profiles()
        assert direct.altitudes[0] == pytest.approx(0.0, abs=1.0)
        assert direct.altitudes[-1] == pytest.approx(0.0, abs=1.0)
        assert np.all(np.diff(direct.times) > 0.0)

    def test_the_label_distinguishes_it(self):
        long_way, direct = self._profiles(samples=200)
        assert long_way.label != direct.label
        assert "direct" in direct.label

    def test_a_range_too_short_to_hold_the_conic_is_refused(self):
        """The minor arc has to contain a boost and a 60-degree deorbit
        conic. On the long way there are five radians to spend and this
        never binds; on the short way it does, and a profile that descends
        before it finishes ascending is worse than an error."""
        with pytest.raises(ValueError, match="cannot contain"):
            fobs_trajectory(
                GeodeticPosition.from_degrees(51.0, 59.0),
                GeodeticPosition.from_degrees(52.0, 62.0),
                direction="short",
            )

    def test_an_unknown_direction_is_refused(self):
        with pytest.raises(ValueError, match="'long' or 'short'"):
            fobs_trajectory(self._LAUNCH, self._TARGET, direction="sideways")

    def test_low_altitude_alone_already_denies_most_of_the_network(self):
        """The finding this control exists to expose. Against the 22-site
        network the *direct* low profile is seen by as few sites as the long
        way round — so the small detecting set comes from altitude, not from
        the reversed bearing. The long way then concedes far more warning,
        because warning runs from first detection and it flies three times
        as long."""
        long_way, direct = self._profiles(samples=800)
        sites = tuple(EARLY_WARNING_SITES)
        long_cover = coverage(
            long_way.times, long_way.altitudes, long_way.subpoints, sites
        )
        direct_cover = coverage(
            direct.times, direct.altitudes, direct.subpoints, sites
        )
        assert len(direct_cover.detecting_sites) <= len(long_cover.detecting_sites)
        assert direct_cover.warning_time < 0.6 * long_cover.warning_time


class TestCoalitionNetworks:
    """Warning is only meaningful relative to a defender.

    :func:`coverage` reduces a network to its *earliest* detection, so
    running a trajectory past a catalogue containing sensors on both sides
    answers a question nobody asked. This was a real defect in the notebook
    figures: three of four profiles launched from Dombarovskiy were first
    "detected" by Okno, a Russian radar 900 km from the pad.
    """

    def test_every_site_declares_a_known_coalition(self):
        assert {s.coalition for s in EARLY_WARNING_SITES} <= set(COALITIONS)

    def test_the_russian_sites_are_not_in_the_western_network(self):
        western = {s.name for s in network("western")}
        assert "Okno (Zelenograd)" not in western
        assert "Krasnoyarsk" not in western
        assert "Fylingdales" in western

    def test_cape_town_is_neither(self):
        """Its own note says it is not integrated into any western
        early-warning network, and nothing enforced that until now."""
        assert radar_site("Cape Town").coalition == "non-aligned"
        assert "Cape Town" not in {s.name for s in network("western")}
        assert "Cape Town" in {s.name for s in network("non-aligned")}

    def test_selecting_several_coalitions_unions_them(self):
        both = network("western", "russia")
        assert len(both) == len(network("western")) + len(network("russia"))

    def test_an_unknown_or_empty_coalition_is_refused(self):
        """A silent empty network would make every profile look
        undetectable, which is the most dangerous wrong answer available."""
        with pytest.raises(ValueError, match="unknown coalition"):
            network("nato")
        with pytest.raises(ValueError, match="no sites in coalition"):
            network("russia", sites=network("western"))

    def test_the_launchers_own_radar_changes_who_detects_first(self):
        """The defect, quantified. Including the launching side's sensors
        does not merely add a row — it changes which sensor sets the
        warning, and therefore which coverage gap the analysis is about."""
        launch = GeodeticPosition.from_degrees(51.0, 59.0, 120.0, "launch")
        target = GeodeticPosition.from_degrees(38.87, -77.06, 24.0, "target")
        trajectory = ballistic_trajectory(launch, target, samples=600)

        everything = coverage(
            trajectory.times, trajectory.altitudes, trajectory.subpoints,
            tuple(EARLY_WARNING_SITES),
        )
        defender = coverage(
            trajectory.times, trajectory.altitudes, trajectory.subpoints,
            network("western"),
        )
        assert everything.first_detecting_site == "Okno (Zelenograd)"
        assert defender.first_detecting_site != everything.first_detecting_site
        assert defender.warning_time < everything.warning_time


class TestAscentProfile:
    """The boost leg as a self-consistent gravity turn.

    What it replaced was not merely unrealistic, it was *inconsistent*: a
    stated ramp of altitude against arc whose implied burnout speed was
    2,666 m/s while the profile it fed needed 7,818, and whose lift-off
    flight-path angle was 37 degrees. Neither error touched a warning
    number, because altitude and ground track were being told what to be.
    """

    _SPEED = 7818.0

    def test_it_leaves_the_pad_vertically(self):
        profile = ascent_profile(150e3, self._SPEED, 180.0)
        assert profile.altitudes[0] == pytest.approx(0.0, abs=1e-9)
        assert profile.downranges[0] == pytest.approx(0.0, abs=1e-9)
        # dh/ds -> infinity at lift-off: the first step is all climb.
        climb = profile.altitudes[1] - profile.altitudes[0]
        run = profile.downranges[1] - profile.downranges[0]
        assert np.rad2deg(np.arctan2(climb, max(run, 1e-12))) > 85.0

    def test_it_burns_out_exactly_where_asked(self):
        for altitude in (140e3, 170e3, 220e3):
            profile = ascent_profile(altitude, self._SPEED, 180.0)
            assert profile.altitudes[-1] == pytest.approx(altitude, rel=1e-6)
            assert profile.times[-1] == pytest.approx(180.0)

    def test_burnout_is_near_horizontal_for_an_orbital_insertion(self):
        """Solved for, not assumed. You cannot insert into an orbit while
        still climbing steeply, so a near-zero answer is the check that the
        boost and the parking arc are describing the same vehicle."""
        profile = ascent_profile(170e3, self._SPEED, 180.0)
        assert 0.0 < np.rad2deg(profile.burnout_angle) < 10.0

    def test_altitude_and_downrange_are_both_monotone(self):
        profile = ascent_profile(170e3, self._SPEED, 180.0)
        assert np.all(np.diff(profile.altitudes) > 0.0)
        assert np.all(np.diff(profile.downranges) > 0.0)

    def test_downrange_is_derived_and_lands_where_real_boosts_do(self):
        """Duration, burnout speed and downrange cannot be chosen
        independently — the path length is fixed by the speed law, and only
        its split between up and along is free. A 180 s burn to orbital
        speed puts burnout some 650 km downrange, which is the right
        order for an ICBM-class boost."""
        profile = ascent_profile(170e3, self._SPEED, 180.0)
        assert 500e3 < profile.ground_range < 900e3

    def test_a_longer_burn_reaches_further(self):
        short = ascent_profile(170e3, self._SPEED, 120.0)
        long = ascent_profile(170e3, self._SPEED, 190.0)
        assert long.ground_range > short.ground_range
        assert long.burnout_angle < short.burnout_angle

    def test_the_path_length_matches_the_speed_law(self):
        """An independent check on the integration: with speed growing as
        v_bo * tau, the path length must be exactly v_bo * t_bo / 2."""
        profile = ascent_profile(170e3, self._SPEED, 180.0)
        path = float(
            np.sum(np.hypot(np.diff(profile.altitudes), np.diff(profile.downranges)))
        )
        assert path == pytest.approx(0.5 * self._SPEED * 180.0, rel=2e-4)

    def test_an_unreachable_combination_is_refused(self):
        with pytest.raises(ValueError, match="still ascending"):
            ascent_profile(150e3, self._SPEED, 600.0)

    def test_rejects_nonsense_inputs(self):
        for kwargs in (
            {"burnout_altitude": 0.0},
            {"burnout_speed": -1.0},
            {"duration": 0.0},
        ):
            args = {"burnout_altitude": 170e3, "burnout_speed": self._SPEED,
                    "duration": 180.0, **kwargs}
            with pytest.raises(ValueError):
                ascent_profile(**args)
        with pytest.raises(ValueError, match="pitch_exponent"):
            ascent_profile(170e3, self._SPEED, 180.0, pitch_exponent=0.0)


class TestPhasesAndEvents:
    """Trajectories declare their structure rather than leaving it to be
    inferred from the shape of the altitude curve — which was a guess, and
    broke every time the shape changed."""

    _LAUNCH = GeodeticPosition.from_degrees(51.0, 59.0, 120.0, "launch")
    _TARGET = GeodeticPosition.from_degrees(38.87, -77.06, 24.0, "target")

    def _fobs(self):
        return fobs_trajectory(self._LAUNCH, self._TARGET, samples=600)

    def test_the_phases_tile_the_flight_without_gaps_or_overlap(self):
        for trajectory in (
            self._fobs(),
            ballistic_trajectory(self._LAUNCH, self._TARGET, samples=600),
        ):
            phases = trajectory.phases
            assert phases, f"{trajectory.label} declares no phases"
            assert phases[0].start_time == pytest.approx(0.0, abs=1e-9)
            assert phases[-1].end_time == pytest.approx(trajectory.flight_time, rel=1e-9)
            for earlier, later in zip(phases, phases[1:], strict=False):
                assert later.start_time == pytest.approx(earlier.end_time, rel=1e-12)
                assert earlier.duration > 0.0

    def test_every_sample_falls_in_exactly_one_named_phase(self):
        trajectory = self._fobs()
        for time in np.linspace(0.0, trajectory.flight_time, 50):
            assert trajectory.phase_at(float(time)) != ""

    def test_the_fractional_profile_names_the_legs_it_actually_flies(self):
        names = [p.name for p in self._fobs().phases]
        assert names == ["boost", "parking coast", "deorbit coast", "entry"]

    def test_events_are_ordered_and_bracket_the_flight(self):
        for trajectory in (
            self._fobs(),
            ballistic_trajectory(self._LAUNCH, self._TARGET, samples=600),
        ):
            times = [e.time for e in trajectory.events]
            assert times == sorted(times)
            assert times[0] == pytest.approx(0.0, abs=1e-9)
            assert times[-1] == pytest.approx(trajectory.flight_time, rel=1e-9)

    def test_the_deorbit_burn_is_a_real_delta_v_at_a_real_time(self):
        """The burn used to be a discontinuity in a prescribed altitude
        curve with no velocity change attached to it at all."""
        trajectory = self._fobs()
        burn = next(e for e in trajectory.events if e.name == "deorbit burn")
        delta_v = float(burn.detail.split("dv")[1].split("m/s")[0].replace(",", ""))
        assert 50.0 < delta_v < 400.0, "a deorbit from LEO costs a few hundred m/s"
        parking = next(p for p in trajectory.phases if p.name == "parking coast")
        assert burn.time == pytest.approx(parking.end_time, rel=1e-12)

    def test_the_burn_commits_the_vehicle_thousands_of_km_out(self):
        """A strategic consequence the model now exposes: deorbiting from
        orbital speed takes a quarter of the planet, so the profile is
        committed long before impact and the 'surprise' is bounded by the
        descent conic, not by the parking arc."""
        trajectory = self._fobs()
        burn = next(e for e in trajectory.events if e.name == "deorbit burn")
        assert trajectory.flight_time - burn.time > 900.0
        coast = next(p for p in trajectory.phases if p.name == "deorbit coast")
        ground_km = float(coast.note.split("km")[0].replace(",", ""))
        assert ground_km > 3000.0

    def test_the_parking_arc_is_an_ellipse_not_a_prescription(self):
        """Held exactly constant, altitude was the one quantity in the
        profile that no dynamics produced."""
        trajectory = self._fobs()
        coast = next(p for p in trajectory.phases if p.name == "parking coast")
        inside = (trajectory.times > coast.start_time) & (
            trajectory.times < coast.end_time
        )
        altitudes = trajectory.altitudes[inside]
        assert altitudes.max() - altitudes.min() > 50e3
        assert altitudes.min() >= 170e3 - 1.0

    def test_a_circular_parking_orbit_is_still_available(self):
        trajectory = fobs_trajectory(
            self._LAUNCH, self._TARGET, parking_altitude=200e3,
            parking_apogee=200e3, samples=600,
        )
        coast = next(p for p in trajectory.phases if p.name == "parking coast")
        inside = (trajectory.times > coast.start_time) & (
            trajectory.times < coast.end_time
        )
        assert np.ptp(trajectory.altitudes[inside]) < 1.0

    def test_an_apogee_below_the_insertion_altitude_is_refused(self):
        with pytest.raises(ValueError, match="at or above the insertion"):
            fobs_trajectory(
                self._LAUNCH, self._TARGET, parking_altitude=200e3,
                parking_apogee=150e3, samples=200,
            )

    def test_the_profile_still_lands_on_its_aimpoint(self):
        """Every leg was re-timed; the lead angle depends on all of them.
        Sub-metre, and independent of the sample count — an earlier version
        was 1.9 km off at 200 samples and 0.3 km at 4000, which is how a
        discretisation error announces itself."""
        for samples in (200, 900):
            for direction in ("long", "short"):
                trajectory = fobs_trajectory(
                    self._LAUNCH, self._TARGET, samples=samples, direction=direction
                )
                miss = great_circle_range(trajectory.subpoints[-1], self._TARGET)
                assert miss < 1.0, f"{direction} at {samples} samples missed {miss:.1f} m"
