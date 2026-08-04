"""Guidance numerics: t_go branch analysis, precision behavior, AC-APN."""

import numpy as np
import pytest

from passes.guidance import (
    ExecutionErrorModel,
    TgoStatus,
    apn_acceleration,
    correction_maneuver,
    los_rate,
    miss_sensitivity,
    schedule_corrections,
    time_to_go,
    time_to_go_naive,
)
from passes.orbital import EARTH, lambert, propagate_coast


def quadratic_residual(r, v, a, t):
    """R(t) = R - V t - 0.5 A t^2 evaluated in float64."""
    return r - v * t - 0.5 * a * t * t


class TestTimeToGoBranches:
    def test_agrees_with_naive_in_benign_regime(self):
        r, v, a = 1.0e4, 1.2e3, -40.0
        stable = time_to_go(r, v, a)
        naive = time_to_go_naive(r, v, a)
        t, status = stable.item()
        assert status is TgoStatus.OK
        assert t == pytest.approx(float(naive), rel=1e-12)

    def test_continuity_at_zero_acceleration(self):
        r, v = 5.0e3, 8.0e2
        exact_linear = r / v
        t0, s0 = time_to_go(r, v, 0.0).item()
        t_eps, s_eps = time_to_go(r, v, 1e-300).item()
        assert s0 is TgoStatus.OK and s_eps is TgoStatus.OK
        assert t0 == pytest.approx(exact_linear, rel=1e-15)
        assert t_eps == pytest.approx(exact_linear, rel=1e-12)

    @pytest.mark.parametrize(
        ("r", "v", "a"),
        [
            (1.0e4, 1.0e3, 0.0),  # constant closure
            (1.0e4, 1.0e3, -30.0),  # decelerating, still intercepts
            (1.0e4, 1.0e3, 50.0),  # accelerating closure
            (2.0e3, -50.0, 20.0),  # opening now, accelerating closure
            (5.0e2, 0.0, 10.0),  # zero rate, positive acceleration
            (1.0e4, 3.0e3, -1.0e-9),  # near-zero deceleration
        ],
    )
    def test_root_property_on_feasible_branches(self, r, v, a):
        """Wherever status is OK, t_go must satisfy the quadratic to rounding
        and be the smallest positive root."""
        t, status = time_to_go(r, v, a).item()
        assert status is TgoStatus.OK
        assert t > 0
        scale = abs(r) + abs(v) * t + 0.5 * abs(a) * t * t
        assert abs(quadratic_residual(r, v, a, t)) <= 1e-12 * scale
        # no earlier positive root: R(t') > 0 on a sample of (0, t)
        ts = np.linspace(1e-12, t * (1 - 1e-9), 100)
        assert np.all(quadratic_residual(r, v, a, ts) > -1e-9 * scale)

    def test_linear_fallback_guard(self):
        """D < 0 with positive closure: Remark 6 clamp, no NaN."""
        r, v, a = 1.0e4, 100.0, -10.0  # D = 1e4 - 2e5 < 0
        t, status = time_to_go(r, v, a).item()
        assert status is TgoStatus.LINEAR_FALLBACK
        assert t == pytest.approx(r / v, rel=1e-15)
        assert np.isfinite(t)

    @pytest.mark.parametrize(
        ("r", "v", "a"),
        [
            (1.0e4, -100.0, 0.0),  # opening, no acceleration
            (1.0e4, -100.0, -5.0),  # opening, decelerating further
            (1.0e4, 0.0, 0.0),  # static
            (1.0e4, 0.0, -1.0),  # static, opening acceleration
        ],
    )
    def test_no_closure_is_inf_not_nan(self, r, v, a):
        t, status = time_to_go(r, v, a).item()
        assert status is TgoStatus.NO_CLOSURE
        assert np.isposinf(t)

    def test_intercept_now(self):
        t, status = time_to_go(0.0, -5.0, 1.0).item()
        assert status is TgoStatus.INTERCEPT_NOW
        assert t == 0.0

    def test_invalid_inputs_flagged_not_raised(self):
        res = time_to_go(np.array([1e4, np.nan]), 100.0, 0.0)
        assert TgoStatus(res.status[1]) is TgoStatus.INVALID_INPUT
        assert np.isnan(res.t_go[1])
        assert TgoStatus(res.status[0]) is TgoStatus.OK

    def test_negative_range_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            time_to_go(-1.0, 100.0, 0.0)

    def test_broadcasting_and_flags_shape(self):
        r = np.array([[1e4], [0.0]])
        v = np.array([100.0, -100.0, 100.0])
        res = time_to_go(r, v, 0.0)
        assert res.t_go.shape == (2, 3)
        assert res.status.shape == (2, 3)
        assert TgoStatus(res.status[0, 0]) is TgoStatus.OK
        assert TgoStatus(res.status[0, 1]) is TgoStatus.NO_CLOSURE
        assert TgoStatus(res.status[1, 2]) is TgoStatus.INTERCEPT_NOW
        assert res.feasible[0, 0] and not res.feasible[0, 1]

    def test_random_property_sweep(self):
        """5000 random states: every OK lane satisfies its quadratic; no lane
        is NaN unless flagged INVALID_INPUT."""
        rng = np.random.default_rng(42)
        n = 5000
        r = rng.uniform(0.0, 1e5, n)
        v = rng.uniform(-2e3, 4e3, n)
        a = rng.uniform(-200.0, 200.0, n)
        res = time_to_go(r, v, a)
        ok = res.status == TgoStatus.OK
        resid = quadratic_residual(r[ok], v[ok], a[ok], res.t_go[ok])
        scale = r[ok] + np.abs(v[ok]) * res.t_go[ok] + 1.0
        assert np.max(np.abs(resid) / scale) < 1e-10
        assert np.all(res.t_go[ok] > 0)
        not_invalid = res.status != TgoStatus.INVALID_INPUT
        assert not np.any(np.isnan(res.t_go[not_invalid]))


class TestTimeToGoPrecision:
    """The heart of V6: the conjugate form must hold accuracy as A_c -> 0."""

    def test_stable_beats_naive_in_double(self):
        r, v = 1.0e4, 1.0e3
        t_ref = r / v  # limit value; correction is O(a) and negligible below
        for a in (1e-8, -1e-8, 1e-11, -1e-11, 1e-13, -1e-13):
            t_stable, status = time_to_go(r, v, a).item()
            assert status is TgoStatus.OK
            # correction term: t ≈ (R/V)(1 - a R / (2 V^2) ...)
            expected = 2 * r / (v + np.sqrt(v * v + 2 * a * r))
            assert abs(t_stable - expected) / expected < 1e-15
            assert abs(t_stable - t_ref) / t_ref < 1e-6  # small perturbation
        # the naive form at a = 1e-13 has already lost most digits
        t_naive = float(time_to_go_naive(r, v, 1e-13))
        assert abs(t_naive - t_ref) / t_ref > 1e-10, (
            "if the naive form is suddenly accurate the V6 premise is wrong"
        )

    def test_float32_pipeline_stays_float32(self):
        r = np.float32(1.0e4)
        v = np.float32(1.0e3)
        a = np.float32(1.0e-4)
        res = time_to_go(r, v, a)
        assert res.t_go.dtype == np.float32
        assert res.discriminant.dtype == np.float32
        eps32 = np.finfo(np.float32).eps
        assert abs(float(res.t_go) - 10.0) / 10.0 < 10 * eps32

    def test_float32_stable_accuracy_where_naive_collapses(self):
        r, v, a = np.float32(1.0e4), np.float32(1.0e3), np.float32(1.0e-3)
        t64 = float(time_to_go(np.float64(r), np.float64(v), np.float64(a)).t_go)
        t32 = float(time_to_go(r, v, a).t_go)
        n32 = float(time_to_go_naive(r, v, a))
        eps32 = np.finfo(np.float32).eps
        assert abs(t32 - t64) / t64 < 8 * eps32
        assert abs(n32 - t64) / t64 > 100 * eps32


class TestApn:
    def test_pn_term_normal_to_relative_velocity(self):
        rng = np.random.default_rng(1)
        r = rng.standard_normal(3) * 1e4
        v = rng.standard_normal(3) * 1e3
        cmd = apn_acceleration(r, v, np.zeros(3), np.zeros(3), nav_gain=4.0)
        assert abs(cmd @ v) <= 1e-6 * np.linalg.norm(cmd) * np.linalg.norm(v)

    def test_collision_course_pure_feedforward(self):
        """r parallel to v: LOS rate is zero, command reduces to the
        augmentation and gravity terms."""
        r = np.array([1.0e4, 0.0, 0.0])
        v = np.array([-2.0e3, 0.0, 0.0])
        a_t = np.array([0.0, 3.0, 0.0])
        g = np.array([0.0, 0.0, -9.81])
        cmd = apn_acceleration(-r * -1.0, v, a_t, g, nav_gain=3.0)
        # g is normal to v here, so g_perp = g
        assert np.allclose(cmd, 1.5 * a_t - g, atol=1e-12)

    def test_gravity_component_along_velocity_removed(self):
        v = np.array([100.0, 0.0, -100.0])
        g = np.array([0.0, 0.0, -9.81])
        r = np.array([5.0e3, 1.0, 0.0])
        cmd_with_g = apn_acceleration(r, v, np.zeros(3), g)
        cmd_no_g = apn_acceleration(r, v, np.zeros(3), np.zeros(3))
        g_term = cmd_no_g - cmd_with_g  # = g_perp
        assert abs(g_term @ v) <= 1e-10 * np.linalg.norm(g) * np.linalg.norm(v)
        g_parallel = g - g_term
        assert np.allclose(np.cross(g_parallel, v), 0.0, atol=1e-8)

    def test_los_rate_formula(self):
        r = np.array([2.0, 0.0, 0.0])
        v = np.array([0.0, 3.0, 0.0])
        lam = los_rate(r, v)
        assert np.allclose(lam, [0.0, 0.0, 1.5])

    def test_batched_leading_axes(self):
        rng = np.random.default_rng(9)
        r = rng.standard_normal((5, 3)) * 1e4
        v = rng.standard_normal((5, 3)) * 1e3
        a_t = rng.standard_normal((5, 3))
        g = np.tile([0.0, 0.0, -9.81], (5, 1))
        batch = apn_acceleration(r, v, a_t, g)
        for i in range(5):
            single = apn_acceleration(r[i], v[i], a_t[i], g[i])
            assert np.allclose(batch[i], single, rtol=1e-14)

    def test_degenerate_inputs_raise(self):
        with pytest.raises(ValueError, match="zero range"):
            los_rate(np.zeros(3), np.ones(3))
        with pytest.raises(ValueError, match="zero relative velocity"):
            apn_acceleration(np.ones(3), np.zeros(3), np.zeros(3), np.zeros(3))
        with pytest.raises(ValueError, match="trailing dimension 3"):
            apn_acceleration(np.ones(4), np.ones(3), np.zeros(3), np.zeros(3))
        with pytest.raises(ValueError, match="nav_gain"):
            apn_acceleration(np.ones(3), np.ones(3), np.zeros(3), np.zeros(3), nav_gain=0.0)


class TestMidcourseCorrection:
    """Midcourse trajectory correction on the exoatmospheric arc."""

    T_ARC = 1800.0

    def _nominal(self):
        """A ballistic arc and the aimpoint it actually reaches.

        Building the target by *flying* the nominal rather than by naming a
        point guarantees the aimpoint is reachable, so a failure to hit it
        is a failure of the correction and never of the geometry.
        """
        r0 = np.array([6.7e6, 0.0, 0.0])
        v0 = lambert(r0, np.array([2.0e6, 6.4e6, 0.5e6]), self.T_ARC).v1
        target = np.asarray(
            propagate_coast(r0, v0, self.T_ARC, rtol=1e-12, atol=1e-6, n_output=2).states[:3, -1]
        )
        return r0, v0, target

    def _state_at(self, r0, v0, t):
        arc = propagate_coast(r0, v0, t, rtol=1e-12, atol=1e-6, n_output=2)
        return np.asarray(arc.states[:3, -1]), np.asarray(arc.states[3:, -1])

    def _miss_after(self, r, v, tof, target):
        arc = propagate_coast(r, v, tof, rtol=1e-12, atol=1e-6, n_output=2)
        return float(np.linalg.norm(np.asarray(arc.states[:3, -1]) - target))

    def test_correction_nulls_the_miss_across_the_whole_arc(self):
        """The defining property, checked by flying the corrected state.
        This must hold at both ends: early, where Lambert's geometry is
        well conditioned, and late, where the vehicle is nearly collinear
        with its own aimpoint and Lambert alone is useless."""
        r0, v0, target = self._nominal()
        perturbed = v0 + np.array([2.0, -1.5, 0.8])
        for fraction in (0.02, 0.1, 0.5, 0.9, 0.98):
            burn = fraction * self.T_ARC
            r, v = self._state_at(r0, perturbed, burn)
            correction = correction_maneuver(r, v, target, self.T_ARC - burn)
            assert self._miss_after(r, correction.v_required, self.T_ARC - burn, target) < 1.0

    def test_cost_follows_the_inverse_time_to_go_law(self):
        """The trade the module exists to expose. Correcting a fixed
        position error costs |dv| ~ |dr| / t_go, so the *product* of
        maneuver magnitude and time-to-go should be roughly invariant along
        the arc, and should equal the uncorrected miss distance. Both are
        non-trivial predictions and both hold to about 10%."""
        r0, v0, target = self._nominal()
        perturbed = v0 + np.array([2.0, -1.5, 0.8])
        uncorrected = self._miss_after(r0, perturbed, self.T_ARC, target)
        products = []
        for fraction in (0.05, 0.25, 0.5, 0.75, 0.9):
            burn = fraction * self.T_ARC
            r, v = self._state_at(r0, perturbed, burn)
            correction = correction_maneuver(r, v, target, self.T_ARC - burn)
            products.append(correction.cost * (self.T_ARC - burn))
        for product in products:
            assert product == pytest.approx(uncorrected, rel=0.12)

    def test_delaying_the_burn_costs_more(self):
        """Fuel argues for correcting early, monotonically."""
        r0, v0, target = self._nominal()
        perturbed = v0 + np.array([2.0, -1.5, 0.8])
        costs = []
        for fraction in (0.05, 0.25, 0.5, 0.75, 0.9):
            burn = fraction * self.T_ARC
            r, v = self._state_at(r0, perturbed, burn)
            costs.append(correction_maneuver(r, v, target, self.T_ARC - burn).cost)
        assert costs == sorted(costs)

    def test_refinement_beats_the_raw_lambert_seed(self):
        """Lambert is a two-body solve and the arc carries J2, so the
        unrefined answer misses by kilometres. This is the measurement that
        justifies running a Newton iteration at all."""
        r0, v0, target = self._nominal()
        r, v = self._state_at(r0, v0 + np.array([2.0, -1.5, 0.8]), 0.25 * self.T_ARC)
        tof = 0.75 * self.T_ARC
        raw = correction_maneuver(r, v, target, tof, refine=False)
        refined = correction_maneuver(r, v, target, tof)
        raw_miss = self._miss_after(r, raw.v_required, tof, target)
        assert raw_miss > 1.0e3
        assert self._miss_after(r, refined.v_required, tof, target) < 1.0
        assert refined.refinements <= 8

    def test_sensitivity_tends_to_the_free_particle_limit(self):
        """Over a short arc gravity has not had time to act, so a velocity
        change moves the terminal position by that change times the elapsed
        time: dr_f/dv_0 -> t * I. Anything else means the finite-difference
        steps are mis-scaled."""
        r0, v0, _ = self._nominal()
        tof = 10.0
        sensitivity = miss_sensitivity(r0, v0, tof)
        assert np.allclose(sensitivity.velocity, tof * np.eye(3), atol=1e-3)
        # The position block departs from the identity by the gravity
        # gradient acting over the arc, and it does so with the *structure*
        # of the tidal tensor (mu/r^3)(3 rr^T - I): stretching along the
        # radial direction and compressing across it, in a 2:-1 ratio. That
        # is a sharper check than any blanket tolerance, and it would catch
        # a sign error or a transposed Jacobian that a norm test would not.
        radial = np.asarray(r0) / float(np.linalg.norm(r0))
        tidal = EARTH.mu / float(np.linalg.norm(r0)) ** 3 * tof**2
        expected = np.eye(3) + 0.5 * tidal * (3.0 * np.outer(radial, radial) - np.eye(3))
        assert np.allclose(sensitivity.position, expected, atol=1e-6)

    def test_sensitivity_agrees_with_a_direct_perturbation(self):
        """The finite-difference Jacobian must predict the effect of a
        finite velocity change, tested against an actual propagation rather
        than against a second finite difference."""
        r0, v0, _ = self._nominal()
        tof = 600.0
        sensitivity = miss_sensitivity(r0, v0, tof)
        kick = np.array([0.5, -0.3, 0.2])
        base = propagate_coast(r0, v0, tof, rtol=1e-12, atol=1e-6, n_output=2)
        bumped = propagate_coast(r0, v0 + kick, tof, rtol=1e-12, atol=1e-6, n_output=2)
        actual = np.asarray(bumped.states[:3, -1]) - np.asarray(base.states[:3, -1])
        predicted = sensitivity.velocity @ kick
        assert np.allclose(predicted, actual, rtol=2e-3)

    def test_execution_covariance_is_anisotropic_about_the_burn(self):
        """A burn is far better known along its own axis than across it,
        and collapsing that to an isotropic sigma would hide the dominant
        error direction of a large maneuver."""
        model = ExecutionErrorModel(magnitude_fraction=0.01, pointing_sigma=0.05)
        dv = np.array([100.0, 0.0, 0.0])
        cov = model.covariance(dv)
        assert cov[0, 0] == pytest.approx(1.0**2)
        assert cov[1, 1] == pytest.approx(5.0**2)
        assert cov[1, 1] > cov[0, 0]
        assert np.allclose(cov, cov.T)
        assert np.all(np.linalg.eigvalsh(cov) >= 0.0)

    def test_execution_error_scales_with_the_burn_but_the_fixed_part_does_not(self):
        model = ExecutionErrorModel(magnitude_fraction=0.01, pointing_sigma=0.0, fixed_sigma=0.02)
        small = model.covariance([1.0, 0.0, 0.0])[0, 0]
        large = model.covariance([100.0, 0.0, 0.0])[0, 0]
        assert small == pytest.approx(0.01**2 + 0.02**2)
        assert large == pytest.approx(1.0**2 + 0.02**2)
        # A zero burn still carries the fixed term and nothing else.
        assert model.covariance(np.zeros(3))[0, 0] == pytest.approx(0.02**2)

    def test_residual_miss_grows_with_navigation_uncertainty(self):
        """The knowledge side of the trade: a burn computed from a worse
        estimate leaves a larger expected miss, even executed perfectly."""
        r0, v0, target = self._nominal()
        r, v = self._state_at(r0, v0, 0.25 * self.T_ARC)
        tof = 0.75 * self.T_ARC
        misses = []
        for scale in (1.0, 4.0):
            cov = np.diag([100.0, 100.0, 100.0, 0.01, 0.01, 0.01]) * scale**2
            misses.append(
                correction_maneuver(r, v, target, tof, state_covariance=cov).residual_miss
            )
        assert misses[1] > 3.0 * misses[0]

    def test_execution_error_adds_to_the_expected_miss(self):
        r0, v0, target = self._nominal()
        r, v = self._state_at(r0, v0 + np.array([5.0, -3.0, 2.0]), 0.1 * self.T_ARC)
        tof = 0.9 * self.T_ARC
        cov = np.diag([100.0, 100.0, 100.0, 0.01, 0.01, 0.01])
        perfect = correction_maneuver(r, v, target, tof, state_covariance=cov)
        sloppy = correction_maneuver(
            r,
            v,
            target,
            tof,
            state_covariance=cov,
            execution=ExecutionErrorModel(magnitude_fraction=0.05, pointing_sigma=0.02),
        )
        assert sloppy.residual_miss > perfect.residual_miss

    def test_miss_is_not_reported_without_a_covariance(self):
        """Refusing to invent a number is the point: the maneuver is still
        computed, but the miss it would leave is unknowable without a
        navigation covariance."""
        r0, v0, target = self._nominal()
        r, v = self._state_at(r0, v0, 0.5 * self.T_ARC)
        result = correction_maneuver(r, v, target, 0.5 * self.T_ARC)
        assert np.isnan(result.residual_miss)
        assert np.isfinite(result.cost)

    def test_schedule_flies_the_plan_and_lands_on_the_target(self):
        """Each burn is computed from the state actually reached, so a
        multi-burn plan must still arrive."""
        r0, v0, target = self._nominal()
        perturbed = v0 + np.array([2.0, -1.5, 0.8])
        plan = schedule_corrections(
            r0, perturbed, target, self.T_ARC, [0.2 * self.T_ARC, 0.7 * self.T_ARC]
        )
        assert len(plan.corrections) == 2
        assert plan.total_cost == pytest.approx(sum(c.cost for c in plan.corrections))
        # The second burn cleans up almost nothing, because the first one
        # already targeted the true dynamics.
        assert plan.corrections[1].cost < 0.05 * plan.corrections[0].cost

    def test_a_single_late_burn_costs_more_than_a_single_early_one(self):
        r0, v0, target = self._nominal()
        perturbed = v0 + np.array([2.0, -1.5, 0.8])
        early = schedule_corrections(r0, perturbed, target, self.T_ARC, [0.1 * self.T_ARC])
        late = schedule_corrections(r0, perturbed, target, self.T_ARC, [0.8 * self.T_ARC])
        assert late.total_cost > 3.0 * early.total_cost

    def test_schedule_rejects_out_of_order_or_out_of_range_burns(self):
        r0, v0, target = self._nominal()
        with pytest.raises(ValueError, match="strictly increasing"):
            schedule_corrections(r0, v0, target, self.T_ARC, [900.0, 400.0])
        with pytest.raises(ValueError, match="strictly inside"):
            schedule_corrections(r0, v0, target, self.T_ARC, [0.0, 900.0])
        with pytest.raises(ValueError, match="strictly inside"):
            schedule_corrections(r0, v0, target, self.T_ARC, [900.0, self.T_ARC])

    def test_rejects_degenerate_inputs(self):
        r0, v0, target = self._nominal()
        with pytest.raises(ValueError, match="time_of_flight"):
            correction_maneuver(r0, v0, target, 0.0)
        with pytest.raises(ValueError, match="3-vector"):
            correction_maneuver([1.0, 2.0], v0, target, 100.0)
        with pytest.raises(ValueError, match="state_covariance"):
            correction_maneuver(r0, v0, target, 100.0, state_covariance=np.eye(3))
        with pytest.raises(ValueError, match="must be finite"):
            ExecutionErrorModel(magnitude_fraction=float("nan"))
        with pytest.raises(ValueError, match="must be finite"):
            ExecutionErrorModel(pointing_sigma=-1.0)

    def test_execution_induced_miss_is_independent_of_burn_time(self):
        """The counter-intuitive result, and the reason burning early is
        not penalised for accuracy. Execution error scales with the
        maneuver, the maneuver scales as 1/t_go, and terminal sensitivity
        to a velocity change scales as t_go — so the two cancel. With the
        navigation covariance zeroed to isolate the effect, the expected
        miss is flat across a nineteen-fold range of time-to-go even though
        the maneuver itself grows seventeen-fold."""
        r0, v0, target = self._nominal()
        perturbed = v0 + np.array([2.0, -1.5, 0.8])
        no_knowledge_error = np.zeros((6, 6))
        execution = ExecutionErrorModel(
            magnitude_fraction=0.02, pointing_sigma=5.0e-3, fixed_sigma=0.0
        )
        misses, costs = [], []
        for fraction in (0.05, 0.4, 0.8, 0.95):
            burn = fraction * self.T_ARC
            r, v = self._state_at(r0, perturbed, burn)
            result = correction_maneuver(
                r,
                v,
                target,
                self.T_ARC - burn,
                state_covariance=no_knowledge_error,
                execution=execution,
            )
            misses.append(result.residual_miss)
            costs.append(result.cost)
        assert max(costs) / min(costs) > 15.0
        assert max(misses) / min(misses) < 1.1

    def test_accuracy_alone_always_favours_the_latest_burn(self):
        """Given a covariance that improves along the arc, expected miss
        falls monotonically, so there is no interior accuracy optimum — the
        trade against fuel is genuinely two-objective."""
        r0, v0, target = self._nominal()
        perturbed = v0 + np.array([2.0, -1.5, 0.8])

        def covariance(t):
            remaining = max(1e-3, 1.0 - t / self.T_ARC)
            sigma_r = 50.0 + 5000.0 * remaining**1.5
            sigma_v = 0.05 + 5.0 * remaining**1.5
            return np.diag([sigma_r**2] * 3 + [sigma_v**2] * 3)

        execution = ExecutionErrorModel(
            magnitude_fraction=0.02, pointing_sigma=5.0e-3, fixed_sigma=0.01
        )
        misses = []
        for fraction in (0.05, 0.2, 0.4, 0.6, 0.8, 0.95):
            burn = fraction * self.T_ARC
            r, v = self._state_at(r0, perturbed, burn)
            misses.append(
                correction_maneuver(
                    r,
                    v,
                    target,
                    self.T_ARC - burn,
                    state_covariance=covariance(burn),
                    execution=execution,
                ).residual_miss
            )
        assert misses == sorted(misses, reverse=True)
