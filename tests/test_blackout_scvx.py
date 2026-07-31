"""Blackout gating and successive convexification (Paper II, §6)."""

import numpy as np
import pytest

from passes.estimation.blackout import (
    GNSS_L1_ANGULAR_FREQUENCY,
    BlackoutGate,
    InertialErrorBudget,
    plasma_frequency,
    propagate_unaided_covariance,
    saha_electron_density,
    unaided_position_variance,
)
from passes.guidance import (
    SCvxConfig,
    linearize_trajectory,
    solve_scvx,
    solve_subproblem,
    solve_subproblem_l2,
)

BUDGET = InertialErrorBudget(
    accel_psd=1e-4, accel_bias_variance=1e-6, gyro_bias_variance=1e-10
)
_GRAVITY = np.array([0.0, -3.71])
_DRAG = 0.02


def _dynamics(x, u):
    v = x[2:4]
    speed = float(np.linalg.norm(v))
    return np.concatenate([v, u + _GRAVITY - _DRAG * speed * v])


def _jacobians(x, u):
    v = x[2:4]
    speed = float(np.linalg.norm(v))
    a = np.zeros((4, 4))
    a[0, 2] = a[1, 3] = 1.0
    if speed > 0.0:
        a[2:4, 2:4] = -_DRAG * (speed * np.eye(2) + np.outer(v, v) / speed)
    b = np.zeros((4, 2))
    b[2, 0] = b[3, 1] = 1.0
    return a, b


class TestSaha:
    def test_ionization_is_monotone_in_temperature(self):
        t = np.linspace(2000.0, 15000.0, 40)
        n_e = saha_electron_density(t, 1e23)
        assert np.all(np.diff(n_e) > 0.0)

    def test_cold_gas_is_essentially_neutral(self):
        assert float(saha_electron_density(1000.0, 1e23)) / 1e23 < 1e-12

    def test_hot_gas_approaches_full_ionization(self):
        assert float(saha_electron_density(60000.0, 1e20)) / 1e20 > 0.9

    def test_ionization_fraction_bounded(self):
        t = np.geomspace(500.0, 1e6, 60)
        alpha = saha_electron_density(t, 1e22) / 1e22
        assert np.all((alpha >= 0.0) & (alpha <= 1.0))

    def test_denser_gas_recombines(self):
        """Saha: at fixed temperature a denser gas is less ionized."""
        sparse = float(saha_electron_density(8000.0, 1e21)) / 1e21
        dense = float(saha_electron_density(8000.0, 1e25)) / 1e25
        assert dense < sparse

    def test_validation(self):
        with pytest.raises(ValueError, match="temperature"):
            saha_electron_density(-1.0, 1e23)
        with pytest.raises(ValueError, match="number_density"):
            saha_electron_density(5000.0, 0.0)


class TestPlasmaFrequency:
    def test_formula_and_scaling(self):
        n_e = 1e18
        omega = float(plasma_frequency(n_e))
        assert float(plasma_frequency(4 * n_e)) == pytest.approx(2 * omega)

    def test_blackout_threshold_is_reached_at_entry_conditions(self):
        """A shock layer at entry temperatures must exceed L1."""
        n_e = saha_electron_density(8000.0, 1e23)
        assert float(plasma_frequency(n_e)) > GNSS_L1_ANGULAR_FREQUENCY

    def test_validation(self):
        with pytest.raises(ValueError, match="electron_density"):
            plasma_frequency(-1.0)


class TestBlackoutGate:
    def test_bare_threshold_chatters_but_hysteresis_does_not(self):
        """The II-V6 chattering criterion, measured directly."""
        rng = np.random.default_rng(4)
        omega = GNSS_L1_ANGULAR_FREQUENCY * (1.0 + 0.02 * rng.standard_normal(400))
        bare = BlackoutGate(hysteresis=0.0)
        damped = BlackoutGate(hysteresis=0.1)
        bare_states = [bare.update(float(w)) for w in omega]
        damped_states = [damped.update(float(w)) for w in omega]
        bare_transitions = int(np.sum(np.diff(np.asarray(bare_states, int)) != 0))
        damped_transitions = int(np.sum(np.diff(np.asarray(damped_states, int)) != 0))
        assert bare_transitions > 50, "premise: the bare threshold must chatter"
        # one transition is the legitimate initial latch into blackout;
        # chattering means repeated crossings after it
        assert damped_transitions <= 1

    def test_acquires_and_releases_at_the_right_thresholds(self):
        gate = BlackoutGate(hysteresis=0.25)
        assert not gate.update(0.5 * GNSS_L1_ANGULAR_FREQUENCY)
        assert gate.update(1.01 * GNSS_L1_ANGULAR_FREQUENCY)
        # still blacked out just below the carrier: release needs the margin
        assert gate.update(0.95 * GNSS_L1_ANGULAR_FREQUENCY)
        assert not gate.update(0.70 * GNSS_L1_ANGULAR_FREQUENCY)

    def test_observation_mask_zeroes_gnss_rows(self):
        gate = BlackoutGate()
        assert np.all(gate.observation_mask(1e5, 3) == 1.0)
        assert np.all(gate.observation_mask(1e12, 3) == 0.0)

    def test_reset_and_validation(self):
        gate = BlackoutGate()
        gate.reset(blacked_out=True)
        assert gate.blacked_out
        with pytest.raises(ValueError, match="hysteresis"):
            BlackoutGate(hysteresis=-0.1)
        with pytest.raises(ValueError, match="plasma frequency"):
            gate.update(-1.0)


class TestCovarianceGrowth:
    @pytest.mark.parametrize(
        ("channel", "exponent"),
        [("velocity_random_walk", 3), ("accel_bias", 4), ("gyro_bias", 6)],
    )
    def test_exponents_match_proposition_three(self, channel, exponent):
        t = np.geomspace(1.0, 100.0, 40)
        variance = unaided_position_variance(t, BUDGET, channel)
        slope = float(np.polyfit(np.log(t), np.log(variance), 1)[0])
        assert slope == pytest.approx(exponent, abs=1e-9)

    @pytest.mark.parametrize(
        "channel", ["velocity_random_walk", "accel_bias", "gyro_bias", "all"]
    )
    def test_independent_propagation_reproduces_closed_form(self, channel):
        """The Lyapunov propagation shares no code with the closed form,
        so agreement is a real check of Prop. 3."""
        t = np.linspace(1.0, 80.0, 30)
        closed = unaided_position_variance(t, BUDGET, channel)
        propagated = propagate_unaided_covariance(t, BUDGET, channel)
        assert np.max(np.abs(propagated - closed) / closed) < 1e-9

    def test_gyro_channel_dominates_at_long_blackout(self):
        """Paper II's Remark: the t^6 term dominates past a few tens of
        seconds, which is why a quadratic model under-predicts badly."""
        short = 5.0
        long = 120.0
        gyro_short = float(unaided_position_variance(short, BUDGET, "gyro_bias"))
        other_short = float(
            unaided_position_variance(short, BUDGET, "velocity_random_walk")
        ) + float(unaided_position_variance(short, BUDGET, "accel_bias"))
        gyro_long = float(unaided_position_variance(long, BUDGET, "gyro_bias"))
        other_long = float(
            unaided_position_variance(long, BUDGET, "velocity_random_walk")
        ) + float(unaided_position_variance(long, BUDGET, "accel_bias"))
        assert gyro_short < other_short
        assert gyro_long > other_long

    def test_quadratic_model_underpredicts(self):
        """A guidance layer sizing a trigger on t^2 under-predicts the
        true growth at the durations that matter."""
        t = np.array([10.0, 60.0])
        actual = unaided_position_variance(t, BUDGET, "all")
        quadratic = actual[0] * (t / t[0]) ** 2
        assert quadratic[1] < actual[1]

    def test_validation(self):
        with pytest.raises(ValueError, match="channels"):
            unaided_position_variance(1.0, BUDGET, "nonsense")
        with pytest.raises(ValueError, match="duration"):
            unaided_position_variance(-1.0, BUDGET)
        with pytest.raises(ValueError, match="accel_psd"):
            InertialErrorBudget(-1.0, 1.0, 1.0)


class TestSCvx:
    def test_linearization_is_exact_at_the_reference_point(self):
        states = np.array([[0.0, 400.0, 50.0, -40.0], [25.0, 380.0, 48.0, -41.0]])
        controls = np.array([[1.0, 2.0]])
        dt = 0.5
        a, b, z = linearize_trajectory(_dynamics, _jacobians, states, controls, dt)
        predicted = a[0] @ states[0] + b[0] @ controls[0] + z[0]
        exact = states[0] + dt * _dynamics(states[0], controls[0])
        assert np.allclose(predicted, exact, atol=1e-12)

    def test_virtual_controls_are_free_in_sign(self):
        """Remark 1: linearization error has no preferred sign, so the
        recovered virtual control must be able to go negative."""
        n, nx, nu = 6, 4, 2
        rng = np.random.default_rng(1)
        a = np.tile(np.eye(nx), (n, 1, 1))
        b = np.tile(np.vstack([np.zeros((2, nu)), np.eye(nu)]), (n, 1, 1))
        z = rng.normal(scale=5.0, size=(n, nx))
        ref = np.zeros((n + 1, nx))
        sol = solve_subproblem(
            a, b, z, np.zeros(nx), np.zeros(nx), ref,
            control_limit=1e-3, penalty_weight=1.0, trust_radius=1e-3,
        )
        assert sol.success
        assert np.any(sol.virtual_controls < -1e-9), "must admit negative components"
        assert np.any(sol.virtual_controls > 1e-9)

    def test_l1_penalty_reaches_exactly_zero_at_finite_weight(self):
        """The II-V7 criterion: exact, not asymptotic."""
        x0 = np.array([0.0, 400.0, 50.0, -40.0])
        target = np.array([300.0, 0.0, 0.0, 0.0])
        result = solve_scvx(
            _dynamics, _jacobians, x0, target, n_steps=40, dt=0.5,
            control_limit=30.0, n_controls=2,
            config=SCvxConfig(penalty_weight=1e3, trust_radius=100.0),
        )
        assert result.virtual_norm == 0.0
        assert result.converged

    def test_below_threshold_the_penalty_is_not_exact(self):
        """Exactness holds above a finite threshold, not at every weight —
        stating the threshold exists means showing it bites below."""
        x0 = np.array([0.0, 400.0, 50.0, -40.0])
        target = np.array([300.0, 0.0, 0.0, 0.0])
        weak = solve_scvx(
            _dynamics, _jacobians, x0, target, n_steps=40, dt=0.5,
            control_limit=30.0, n_controls=2,
            config=SCvxConfig(penalty_weight=1.0, trust_radius=100.0, max_iterations=25),
        )
        assert weak.virtual_norm > 1.0

    def test_quadratic_penalty_only_decays_as_one_over_weight(self):
        """The contrast the paper draws: L2 approaches zero, L1 attains it."""
        n, nx, nu = 5, 4, 2
        rng = np.random.default_rng(7)
        a = np.tile(np.eye(nx), (n, 1, 1))
        b = np.tile(np.vstack([np.zeros((2, nu)), np.eye(nu)]), (n, 1, 1))
        z = rng.normal(scale=0.5, size=(n, nx))
        ref = np.zeros((n + 1, nx))
        norms = []
        for weight in (1e1, 1e3):
            sol = solve_subproblem_l2(
                a, b, z, np.zeros(nx), np.zeros(nx), ref,
                control_limit=0.5, penalty_weight=weight, trust_radius=0.5,
            )
            norms.append(sol.virtual_norm)
        assert norms[0] > 0.0 and norms[1] > 0.0, "L2 never reaches exactly zero"
        assert norms[1] < norms[0]

    def test_trust_region_logic_and_config_validation(self):
        with pytest.raises(ValueError, match="penalty_weight"):
            SCvxConfig(penalty_weight=0.0)
        with pytest.raises(ValueError, match="trust"):
            SCvxConfig(trust_radius=1e-9, trust_min=1.0)
        with pytest.raises(ValueError, match="rho"):
            SCvxConfig(rho_reject=1.0, rho_contract=0.5)
        with pytest.raises(ValueError, match="contract_factor"):
            SCvxConfig(contract_factor=2.0)

    def test_solver_validation(self):
        x0 = np.zeros(4)
        with pytest.raises(ValueError, match="n_steps"):
            solve_scvx(_dynamics, _jacobians, x0, x0, 0, 0.5, 1.0, 2)
        with pytest.raises(ValueError, match="n_controls"):
            solve_scvx(_dynamics, _jacobians, x0, x0, 5, 0.5, 1.0, 0)
        with pytest.raises(ValueError, match="same dimension"):
            solve_scvx(_dynamics, _jacobians, x0, np.zeros(3), 5, 0.5, 1.0, 2)
