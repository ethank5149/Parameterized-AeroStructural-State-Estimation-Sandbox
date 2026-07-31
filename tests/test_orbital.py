"""Orbital mechanics and the coast phase (Paper II, §7)."""

import numpy as np
import pytest

from passes.orbital import (
    EARTH,
    compare_coast_strategies,
    gravitational_acceleration,
    gravitational_potential,
    j2_acceleration,
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
        separation = float(
            np.linalg.norm(with_j2.states[:3, -1] - without.states[:3, -1])
        )
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
            compare_coast_strategies(r, v, 10.0, structural_frequency=10.0,
                                     structural_damping=1.5)

    def test_propagation_validation(self):
        r, v, _ = circular_state()
        with pytest.raises(ValueError, match="duration"):
            propagate_coast(r, v, 0.0)
        with pytest.raises(ValueError, match="3-vectors"):
            propagate_coast([1.0, 2.0], v, 10.0)
