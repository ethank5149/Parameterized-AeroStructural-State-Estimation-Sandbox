"""Guidance numerics: t_go branch analysis, precision behavior, AC-APN."""

import numpy as np
import pytest

from passes.guidance import (
    TgoStatus,
    apn_acceleration,
    los_rate,
    time_to_go,
    time_to_go_naive,
)


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
            (1.0e4, 1.0e3, 0.0),        # constant closure
            (1.0e4, 1.0e3, -30.0),      # decelerating, still intercepts
            (1.0e4, 1.0e3, 50.0),       # accelerating closure
            (2.0e3, -50.0, 20.0),       # opening now, accelerating closure
            (5.0e2, 0.0, 10.0),         # zero rate, positive acceleration
            (1.0e4, 3.0e3, -1.0e-9),    # near-zero deceleration
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
            (1.0e4, -100.0, 0.0),    # opening, no acceleration
            (1.0e4, -100.0, -5.0),   # opening, decelerating further
            (1.0e4, 0.0, 0.0),       # static
            (1.0e4, 0.0, -1.0),      # static, opening acceleration
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
