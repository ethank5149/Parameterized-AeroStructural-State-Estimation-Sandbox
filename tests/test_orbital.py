"""Orbital mechanics and the coast phase (Paper II, §7)."""

import numpy as np
import pytest

from passes.orbital import (
    EARTH,
    compare_coast_strategies,
    gravitational_acceleration,
    gravitational_potential,
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
