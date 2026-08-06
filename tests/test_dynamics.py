"""6-DOF state: attitude kinematics and deformed-surface incidence (Paper II, §3)."""

import numpy as np
import pytest
import scipy.integrate

from passes.dynamics import (
    dcm_from_quaternion,
    deformed_normal,
    local_incidence,
    quaternion_derivative,
    quaternion_norm_error,
)
from passes.dynamics.roll_resonance import (
    pitch_frequency,
    resonance_condition_ratio,
    resonance_crossings,
    roll_rate,
    steady_state_roll_rate,
    trim_amplification,
)
from passes.flight.ballistic_entry import (
    EXPONENTIAL_ATMOSPHERE_EARTH,
    BallisticEntry,
)


class TestQuaternionKinematics:
    def test_norm_is_invariant_of_the_exact_flow(self):
        """Without stabilization ||q|| is conserved by the exact flow, so
        d/dt(||q||^2) must vanish identically."""
        rng = np.random.default_rng(0)
        q = rng.standard_normal(4)
        q /= np.linalg.norm(q)
        w = rng.standard_normal(3)
        dq = quaternion_derivative(q, w, baumgarte_gain=0.0)
        assert abs(2.0 * q @ dq) < 1e-15

    def test_constant_rate_gives_exact_rotation(self):
        """Rotating at a constant body rate about z for time t must equal
        the analytic half-angle quaternion."""
        omega = np.array([0.0, 0.0, 1.7])
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        t_end = 1.1
        sol = scipy.integrate.solve_ivp(
            lambda _t, q: quaternion_derivative(q, omega, 0.0),
            (0.0, t_end),
            q0,
            rtol=1e-12,
            atol=1e-14,
        )
        angle = 1.7 * t_end
        expected = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
        assert np.allclose(sol.y[:, -1], expected, atol=1e-10)

    def test_baumgarte_pulls_norm_back_to_unity(self):
        """The unit sphere must be attracting, not merely invariant."""
        q0 = 1.10 * np.array([1.0, 0.0, 0.0, 0.0])
        k_q = 5.0
        sol = scipy.integrate.solve_ivp(
            lambda _t, q: quaternion_derivative(q, np.array([0.3, -0.2, 0.5]), k_q),
            (0.0, 6.0 / k_q),
            q0,
            rtol=1e-11,
            atol=1e-13,
        )
        assert abs(quaternion_norm_error(q0)) > 0.09
        assert abs(float(quaternion_norm_error(sol.y[:, -1]))) < 1e-4

    def test_baumgarte_leaves_unit_quaternion_untouched(self):
        q = np.array([0.6, 0.8, 0.0, 0.0])
        w = np.array([0.1, 0.2, 0.3])
        assert np.allclose(
            quaternion_derivative(q, w, 0.0), quaternion_derivative(q, w, 10.0)
        )

    def test_batched(self):
        rng = np.random.default_rng(3)
        q = rng.standard_normal((5, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        w = rng.standard_normal((5, 3))
        batch = quaternion_derivative(q, w, 2.0)
        for i in range(5):
            assert np.allclose(batch[i], quaternion_derivative(q[i], w[i], 2.0))

    def test_validation(self):
        with pytest.raises(ValueError, match="trailing dimension 4"):
            quaternion_derivative(np.zeros(3), np.zeros(3))
        with pytest.raises(ValueError, match="trailing dimension 3"):
            quaternion_derivative(np.array([1.0, 0, 0, 0]), np.zeros(4))
        with pytest.raises(ValueError, match="baumgarte_gain"):
            quaternion_derivative(np.array([1.0, 0, 0, 0]), np.zeros(3), -1.0)


class TestDirectionCosineMatrix:
    def test_identity_quaternion(self):
        assert np.allclose(dcm_from_quaternion([1.0, 0, 0, 0]), np.eye(3))

    def test_orthonormal_and_proper(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            q = rng.standard_normal(4)
            c = dcm_from_quaternion(q)
            assert np.allclose(c @ c.T, np.eye(3), atol=1e-13)
            assert np.linalg.det(c) == pytest.approx(1.0, abs=1e-13)

    def test_hamilton_convention_not_jpl(self):
        """A +90 deg rotation about z maps the ECI x-axis to the body
        -y axis under the Hamilton ECI->body convention. The JPL
        convention gives the transpose, which is also a valid rotation —
        which is exactly why this is asserted rather than assumed."""
        q = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
        c = dcm_from_quaternion(q)
        assert np.allclose(c @ np.array([1.0, 0.0, 0.0]), [0.0, -1.0, 0.0], atol=1e-14)
        assert np.allclose(c @ np.array([0.0, 1.0, 0.0]), [1.0, 0.0, 0.0], atol=1e-14)

    def test_composition_matches_quaternion_product(self):
        """C(q) for a 2*theta rotation equals C(q_theta) applied twice."""
        theta = 0.37
        q1 = np.array([np.cos(theta / 2), 0, np.sin(theta / 2), 0])
        q2 = np.array([np.cos(theta), 0, np.sin(theta), 0])
        c1 = dcm_from_quaternion(q1)
        assert np.allclose(c1 @ c1, dcm_from_quaternion(q2), atol=1e-13)

    def test_normalizes_input(self):
        q = np.array([2.0, 0.0, 0.0, 0.0])
        assert np.allclose(dcm_from_quaternion(q), np.eye(3))
        with pytest.raises(ValueError, match="non-zero"):
            dcm_from_quaternion(np.zeros(4))


class TestDeformedNormal:
    def test_flat_undeformed_surface(self):
        n = deformed_normal(0.0, 0.0)
        assert np.allclose(n, [0.0, 0.0, 1.0])

    def test_midsurface_normal_formula(self):
        """offset = 0 must give n ∝ (-w_x, -w_y, 1)."""
        w_x, w_y = 0.3, -0.2
        n = deformed_normal(w_x, w_y)
        expected = np.array([-w_x, -w_y, 1.0])
        expected /= np.linalg.norm(expected)
        assert np.allclose(n, expected)

    def test_offset_uses_rotation_fields(self):
        """A non-zero half-thickness makes the normal depend on the
        rotations, not just the midsurface slope."""
        common = {"w_x": 0.1, "w_y": 0.05}
        n_mid = deformed_normal(**common)
        n_off = deformed_normal(
            **common,
            phi_x=0.2, phi_y=-0.1,
            phi_x_x=0.4, phi_x_y=0.1, phi_y_x=-0.2, phi_y_y=0.3,
            offset=0.05,
        )
        assert not np.allclose(n_mid, n_off)
        assert np.linalg.norm(n_off) == pytest.approx(1.0)

    def test_offset_requires_derivatives(self):
        with pytest.raises(ValueError, match="rotation derivatives"):
            deformed_normal(0.1, 0.1, offset=0.05)

    def test_batched_shape(self):
        w_x = np.linspace(-0.3, 0.3, 7)
        w_y = np.zeros(7)
        n = deformed_normal(w_x, w_y)
        assert n.shape == (7, 3)
        assert np.allclose(np.linalg.norm(n, axis=1), 1.0)


class TestLocalIncidence:
    def test_sign_convention_windward_positive(self):
        """Paper II, Remark 1: with n the OUTWARD normal, a panel facing
        into the flow must give delta_c > 0. Dropping the negation
        inverts the entire pressure and heating distribution."""
        normal = np.array([0.0, 0.0, 1.0])  # upward-facing panel
        v_from_above = np.array([0.0, 0.0, -1.0])  # flow arriving downward
        assert float(local_incidence(normal, v_from_above)) == pytest.approx(np.pi / 2)
        v_from_below = np.array([0.0, 0.0, 1.0])
        assert float(local_incidence(normal, v_from_below)) == pytest.approx(-np.pi / 2)

    def test_edge_on_flow_is_zero(self):
        assert float(
            local_incidence([0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
        ) == pytest.approx(0.0, abs=1e-15)

    def test_known_angle(self):
        # a panel whose outward normal is tilted so that sin(delta) = sin(theta)
        theta = np.deg2rad(30.0)
        normal = np.array([0.0, -np.cos(theta), np.sin(theta)])
        v = np.array([0.0, 0.0, -1.0])
        assert float(local_incidence(normal, v)) == pytest.approx(theta, abs=1e-12)

    def test_velocity_need_not_be_normalized(self):
        n = np.array([0.0, 0.0, 1.0])
        assert float(local_incidence(n, [0.0, 0.0, -7.3])) == pytest.approx(np.pi / 2)

    def test_validation(self):
        with pytest.raises(ValueError, match="non-zero"):
            local_incidence([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="trailing dimension 3"):
            local_incidence([0.0, 1.0], [0.0, 0.0, 1.0])


class TestRollResonance:
    """Regan §13.2 and §13.5: roll dynamics, and the resonance a reentry
    vehicle can cross twice."""

    _PS = 3.73e-3          # Regan's stability factor, m/kg
    _ROLL = 18.0           # rad/s
    _VE = 5000.0
    _GAMMA = np.deg2rad(75.0)

    def _crossings(self, ballistic_pascals: float):
        entry = BallisticEntry(
            self._VE,
            self._GAMMA,
            entry_altitude=120.0e3,
            ballistic_coefficient=ballistic_pascals / 9.80665,
        )
        altitudes = np.linspace(0.0, 80.0e3, 8001)
        return resonance_crossings(
            altitudes,
            entry.velocity(altitudes),
            EXPONENTIAL_ATMOSPHERE_EARTH.density(altitudes),
            self._ROLL,
            self._PS,
        )

    def test_reproduces_regans_worked_resonance_case(self):
        """His Fig. 13.10: one resonance for a 6e4 Pa ballistic factor, two
        for 6e3 Pa at about 37 km and 11 km. Driven by the independently
        verified Allen-Eggers profile, so this exercises two modules against
        one published result."""
        heavy = self._crossings(6.0e4)
        light = self._crossings(6.0e3)
        assert len(heavy) == 1
        assert len(light) == 2
        assert heavy[0].altitude / 1e3 == pytest.approx(37.0, abs=1.0)
        assert light[0].altitude / 1e3 == pytest.approx(37.0, abs=1.5)
        assert light[1].altitude / 1e3 == pytest.approx(11.0, abs=1.5)

    def test_a_lighter_vehicle_is_the_one_that_resonates_twice(self):
        """The structural claim: lowering the ballistic coefficient makes a
        second crossing more likely, because the vehicle decelerates higher
        and its pitch frequency peaks further above the roll rate."""
        assert len(self._crossings(6.0e4)) == 1
        assert len(self._crossings(6.0e3)) == 2

    def test_first_and_second_resonance_are_distinguished(self):
        """The two crossings are not interchangeable: one happens with the
        pitch frequency still rising, the other on the way back down. That
        distinction is what separates a thin-air excursion with a long time
        to act from a forceful one with almost none."""
        light = self._crossings(6.0e3)
        assert light[0].ascending != light[1].ascending
        assert light[0].altitude > light[1].altitude

    def test_a_fast_enough_roll_never_resonates(self):
        """Zero crossings is a physical outcome, not a failure. Spinning
        far above the peak pitch frequency means the condition is never
        met."""
        entry = BallisticEntry(self._VE, self._GAMMA, ballistic_coefficient=612.0)
        altitudes = np.linspace(0.0, 80.0e3, 4001)
        assert (
            resonance_crossings(
                altitudes,
                entry.velocity(altitudes),
                EXPONENTIAL_ATMOSPHERE_EARTH.density(altitudes),
                500.0,
                self._PS,
            )
            == ()
        )

    def test_pitch_frequency_is_non_monotone_through_an_entry(self):
        """The whole reason for two crossings: omega ~ V sqrt(rho), and on
        the way down rho rises while V falls, so it peaks in between."""
        entry = BallisticEntry(self._VE, self._GAMMA, ballistic_coefficient=612.0)
        altitudes = np.linspace(0.0, 80.0e3, 2001)
        omega = pitch_frequency(
            entry.velocity(altitudes), EXPONENTIAL_ATMOSPHERE_EARTH.density(altitudes), self._PS
        )
        peak = int(np.argmax(omega))
        assert 0 < peak < omega.size - 1, "the peak must be interior"
        assert omega[peak] > omega[0] and omega[peak] > omega[-1]

    def test_resonance_condition_is_within_five_percent_of_equal_rates(self):
        """sqrt(1 - Ix/Iy) = 0.949 at the representative 0.1, which is why
        the condition is usually quoted as 'pitch frequency equals roll
        rate'. Kept visible rather than absorbed."""
        assert resonance_condition_ratio(0.1) == pytest.approx(0.9487, abs=1e-4)
        assert resonance_condition_ratio(0.0) == 1.0
        assert resonance_condition_ratio(0.3) < resonance_condition_ratio(0.1)

    def test_roll_rate_decays_to_its_steady_state(self):
        """Eq. (13.11): exponential approach, independent of the sign of the
        initial offset."""
        steady = 12.0
        times = np.linspace(0.0, 400.0, 401)
        for initial in (0.0, 30.0):
            history = roll_rate(times, initial, steady, 25.0)
            assert history[0] == pytest.approx(initial)
            assert history[-1] == pytest.approx(steady, abs=1e-3)
            assert np.all(np.abs(history - steady) <= abs(initial - steady) + 1e-12)

    def test_steady_state_roll_rate_is_independent_of_inertia(self):
        """p_ss = -C_l0 V/(C_lp d) contains no inertia and no dynamic
        pressure: those set how fast it is reached, not what it is."""
        value = steady_state_roll_rate(0.001, -0.02, 4000.0, 0.5)
        assert value == pytest.approx(0.001 * 4000.0 / (0.02 * 0.5))
        # Doubling speed doubles it; the driving moment sets the sign.
        assert steady_state_roll_rate(0.001, -0.02, 8000.0, 0.5) == pytest.approx(2 * value)
        assert steady_state_roll_rate(-0.001, -0.02, 4000.0, 0.5) == pytest.approx(-value)

    def test_trim_amplification_peaks_at_resonance_and_scales_as_one_over_zeta(self):
        """A factor of 10 at 5% damping. The point of computing it is that
        'considerable amplification' becomes a number."""
        roll = 18.0
        at_resonance = resonance_condition_ratio(0.1) * roll
        peak = float(trim_amplification(at_resonance, roll, 0.05))
        assert peak == pytest.approx(10.0, rel=0.02)
        assert float(trim_amplification(at_resonance, roll, 0.25)) == pytest.approx(2.0, rel=0.02)
        # Far from resonance the trim is essentially unamplified.
        assert float(trim_amplification(10.0 * at_resonance, roll, 0.05)) < 1.05

    def test_validation(self):
        with pytest.raises(ValueError, match="stability_factor"):
            pitch_frequency(4000.0, 0.1, 0.0)
        with pytest.raises(ValueError, match="inertia_ratio"):
            resonance_condition_ratio(1.0)
        with pytest.raises(ValueError, match="roll_damping_coefficient"):
            steady_state_roll_rate(0.001, 0.02, 4000.0, 0.5)
        with pytest.raises(ValueError, match="time_constant"):
            roll_rate(np.array([0.0]), 1.0, 0.0, -1.0)
        with pytest.raises(ValueError, match="damping_ratio"):
            trim_amplification(10.0, 18.0, 0.0)
        with pytest.raises(ValueError, match="equal length"):
            resonance_crossings([1.0, 2.0], [1.0], [1.0], 18.0, 3.73e-3)
