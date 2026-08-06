"""Batch layer (Paper I, §5–§6): sampling, propagation, dispersion stats."""

import numpy as np
import pytest
import scipy.stats

from passes.batch import (
    DispersionSpec,
    EntryDispersionModel,
    cuda_available,
    henze_zirkler,
    rk4_batch,
    sample_dispersions,
    summarize_dispersion,
)
from passes.systems.dispersion import containment_radius


class TestSampling:
    def test_deterministic_given_seed(self):
        specs = [DispersionSpec("a", 1.0, 0.1), DispersionSpec("b", 5.0, 0.5)]
        s1 = sample_dispersions(specs, 100, seed=42)
        s2 = sample_dispersions(specs, 100, seed=42)
        s3 = sample_dispersions(specs, 100, seed=43)
        assert np.array_equal(s1["a"], s2["a"]) and np.array_equal(s1["b"], s2["b"])
        assert not np.array_equal(s1["a"], s3["a"])

    def test_appending_spec_preserves_earlier_draws(self):
        base = [DispersionSpec("a", 1.0, 0.1)]
        extended = [*base, DispersionSpec("c", 0.0, 1.0)]
        assert np.array_equal(
            sample_dispersions(base, 50, 7)["a"], sample_dispersions(extended, 50, 7)["a"]
        )

    def test_zero_sigma_pins_nominal(self):
        out = sample_dispersions([DispersionSpec("a", 3.5, 0.0)], 20, 0)
        assert np.all(out["a"] == 3.5)

    def test_truncation_by_rejection(self):
        spec = DispersionSpec("a", 0.0, 1.0, lower=-1.0, upper=1.0)
        draws = sample_dispersions([spec], 5000, 1)["a"]
        assert np.all((draws >= -1.0) & (draws <= 1.0))
        # rejection, not clipping: no mass piled on the bounds
        assert np.sum(np.isclose(draws, 1.0, atol=1e-6)) < 5

    def test_validation(self):
        with pytest.raises(ValueError, match="sigma"):
            DispersionSpec("a", 0.0, -1.0)
        with pytest.raises(ValueError, match="lower < upper"):
            DispersionSpec("a", 0.0, 1.0, lower=2.0, upper=1.0)
        with pytest.raises(ValueError, match="duplicate"):
            sample_dispersions(
                [DispersionSpec("a", 0.0, 1.0), DispersionSpec("a", 1.0, 1.0)], 10, 0
            )


class TestRk4Batch:
    def test_fourth_order_convergence(self):
        """dy/dt = -lambda y with per-replicate lambda: global error O(dt^4)."""
        lam = np.array([0.5, 1.0, 2.0])
        y0 = np.ones((3, 1))

        def rhs(_t, y, _xp):
            return -lam[:, None] * y

        errs = []
        for n_steps in (20, 40, 80):
            y = rk4_batch(rhs, y0, 0.0, 1.0, n_steps)
            errs.append(np.max(np.abs(y[:, 0] - np.exp(-lam))))
        rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
        assert np.all(rates > 3.7), f"rates {rates}, expected ~4"

    def test_batch_matches_per_replicate(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0.5, 2.0, 5)

        def rhs(t, y, _xp):
            return a[:, None] * np.cos(t) * y

        y0 = rng.uniform(0.5, 1.5, (5, 1))
        batch = rk4_batch(rhs, y0, 0.0, 2.0, 100)
        for i in range(5):
            def rhs_i(t, y, xp, i=i):
                return a[i] * np.cos(t) * y

            single = rk4_batch(rhs_i, y0[i : i + 1], 0.0, 2.0, 100)
            assert np.allclose(batch[i], single[0], rtol=1e-14)

    @pytest.mark.skipif(not cuda_available(), reason="no CUDA device")
    def test_cupy_backend_matches_numpy(self):
        lam = np.array([0.5, 1.0, 2.0, 4.0])

        def rhs(_t, y, xp):
            return -xp.asarray(lam)[:, None] * y

        y0 = np.ones((4, 2))
        y_cpu = rk4_batch(rhs, y0, 0.0, 1.0, 50, backend="numpy")
        y_gpu = rk4_batch(rhs, y0, 0.0, 1.0, 50, backend="cupy")
        from passes.batch import to_numpy

        assert np.allclose(y_cpu, to_numpy(y_gpu), rtol=1e-13)

    def test_validation(self):
        def rhs(_t, y, _xp):
            return y

        with pytest.raises(ValueError, match="n_steps"):
            rk4_batch(rhs, np.ones((2, 1)), 0.0, 1.0, 0)
        with pytest.raises(ValueError, match="t_end"):
            rk4_batch(rhs, np.ones((2, 1)), 1.0, 0.0, 10)
        with pytest.raises(ValueError, match="shape"):
            rk4_batch(rhs, np.ones(3), 0.0, 1.0, 10)


class TestEntryModel:
    def test_impacts_finite_and_downrange(self):
        model = EntryDispersionModel()
        pts = model.fly(200, seed=1)
        assert pts.shape == (200, 2)
        assert np.all(np.isfinite(pts))
        assert np.all(pts[:, 0] > 0.0), "impacts must be downrange of release"

    def test_zero_dispersion_collapses_footprint(self):
        model = EntryDispersionModel(
            beta_rel_sigma=0.0,
            speed_rel_sigma=0.0,
            flight_path_sigma_deg=0.0,
            azimuth_sigma_deg=0.0,
            density_bias_rel_sigma=0.0,
            wind_sigma=0.0,
        )
        pts = model.fly(20, seed=5)
        assert np.max(np.abs(pts - pts[0])) < 1e-9

    def test_reproducible(self):
        model = EntryDispersionModel()
        assert np.array_equal(model.fly(50, seed=9), model.fly(50, seed=9))

    def test_heavier_ballistic_coefficient_flies_farther(self):
        light = EntryDispersionModel(beta_nominal=3000.0)
        heavy = EntryDispersionModel(beta_nominal=20000.0)
        x_light = np.mean(light.fly(100, seed=2)[:, 0])
        x_heavy = np.mean(heavy.fly(100, seed=2)[:, 0])
        assert x_heavy > x_light

    @pytest.mark.skipif(not cuda_available(), reason="no CUDA device")
    def test_gpu_matches_cpu(self):
        model = EntryDispersionModel()
        cpu = model.fly(64, seed=3, backend="numpy")
        gpu = model.fly(64, seed=3, backend="cupy")
        assert np.allclose(cpu, gpu, rtol=1e-10, atol=1e-6)


class TestHenzeZirkler:
    def test_accepts_normal_sample(self):
        rng = np.random.default_rng(12)
        x = rng.multivariate_normal([0, 0], [[2.0, 0.5], [0.5, 1.0]], size=2000)
        _, p = henze_zirkler(x)
        assert p > 0.05, f"normal sample rejected with p = {p}"

    def test_rejects_uniform_square(self):
        rng = np.random.default_rng(12)
        x = rng.uniform(-1, 1, size=(2000, 2))
        _, p = henze_zirkler(x)
        assert p < 0.001, f"uniform square accepted with p = {p}"

    def test_rejects_ring(self):
        rng = np.random.default_rng(12)
        theta = rng.uniform(0, 2 * np.pi, 1500)
        r = 5.0 + rng.normal(0, 0.2, 1500)
        x = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        _, p = henze_zirkler(x)
        assert p < 1e-6

    def test_p_value_calibration_under_null(self):
        """Across many normal samples, rejections at level 0.05 must occur
        at roughly 5%."""
        rng = np.random.default_rng(99)
        rejections = sum(
            henze_zirkler(rng.standard_normal((150, 2)))[1] < 0.05 for _ in range(200)
        )
        assert 2 <= rejections <= 22, f"{rejections}/200 rejections at nominal 5% level"

    def test_validation(self):
        with pytest.raises(ValueError, match="singular"):
            henze_zirkler(np.tile([1.0, 2.0], (30, 1)))
        with pytest.raises(ValueError):
            henze_zirkler(np.ones((2, 2)))


class TestDispersionSummary:
    def test_isotropic_gaussian_recovers_theory(self):
        rng = np.random.default_rng(31)
        sigma = 40.0
        pts = sigma * rng.standard_normal((20000, 2))
        rep = summarize_dispersion(pts, seed=1)
        # CEP = 1.1774 sigma for the isotropic case (Eq. 6.2). The exact
        # elliptical integral and the linear approximation agree here --
        # 2*0.5887 = 1.1774 -- which is precisely why the circular case
        # cannot distinguish them and why the anisotropic test below is the
        # one that matters.
        assert rep.cep == pytest.approx(np.sqrt(2 * np.log(2)) * sigma, rel=0.03)
        assert rep.cep_method == "exact-elliptical"
        assert rep.r95 == pytest.approx(2.4477 * sigma, rel=0.03)
        assert rep.relative_standard_error == pytest.approx(1 / np.sqrt(2 * 20000))
        # empirical containment of the R95 ellipse
        centered = pts - rep.mean
        proj = centered @ rep.axes
        inside = np.sum(
            (proj[:, 0] / rep.r95_semi_axes[0]) ** 2
            + (proj[:, 1] / rep.r95_semi_axes[1]) ** 2
            <= 1.0
        )
        assert inside / 20000 == pytest.approx(0.95, abs=0.01)
        assert rep.hz_p_value > 0.01

    def test_r95_multiplier_matches_paper_constant(self):
        assert np.sqrt(scipy.stats.chi2.ppf(0.95, 2)) == pytest.approx(2.4477, abs=2e-4)

    def test_cep_is_exact_outside_the_classical_validity_band(self):
        """Below an aspect ratio of 0.25 the classical route fell back to a
        sample median. The exact elliptical integral needs no fallback, so
        the estimate must now match `containment_radius` rather than the
        median -- and the label must still record that the footprint is
        outside the band Eq. (6.4) was stated for, since that is what makes
        results comparable with literature computed the old way.
        """
        rng = np.random.default_rng(8)
        pts = np.column_stack(
            [100.0 * rng.standard_normal(5000), 5.0 * rng.standard_normal(5000)]
        )
        rep = summarize_dispersion(pts, seed=2)
        assert rep.aspect_ratio < 0.25
        assert rep.cep_method == "exact-elliptical (outside Eq. 6.4 band)"
        exact = float(containment_radius(0.5, float(rep.sigma[0]), float(rep.sigma[1])))
        assert rep.cep == pytest.approx(exact, rel=1e-12)
        # The median fallback it replaced is a different number, and noisier.
        median_radius = np.median(np.linalg.norm(pts - pts.mean(axis=0), axis=1))
        assert rep.cep != pytest.approx(median_radius, rel=1e-6)

    def test_bootstrap_interpolation_tracks_the_exact_integral(self):
        """The bootstrap cannot afford a root-find per resample, so it
        interpolates a precomputed CEP-over-sigma curve. That shortcut must
        not introduce error the confidence interval would inherit."""
        from passes.batch.dispersion import _CEP_ASPECT_GRID, _CEP_OVER_SIGMA_MAJOR

        for aspect in (0.02, 0.17, 0.33, 0.5, 0.78, 0.99):
            interpolated = float(
                np.interp(aspect, _CEP_ASPECT_GRID, _CEP_OVER_SIGMA_MAJOR)
            )
            exact = float(containment_radius(0.5, 1.0, aspect))
            assert abs(interpolated - exact) / exact < 1e-6

    def test_bootstrap_ci_brackets_estimate_and_scales(self):
        rng = np.random.default_rng(17)
        pts = 10.0 * rng.standard_normal((2000, 2))
        rep = summarize_dispersion(pts, bootstrap_samples=500, seed=3)
        assert rep.cep_ci[0] < rep.cep < rep.cep_ci[1]
        assert rep.r95_ci[0] < rep.r95 < rep.r95_ci[1]
        # CI half-width should be commensurate with the 1/sqrt(2N) bound
        half = 0.5 * (rep.cep_ci[1] - rep.cep_ci[0])
        assert half == pytest.approx(1.96 * rep.relative_standard_error * rep.cep, rel=0.5)

    def test_bootstrap_reproducible(self):
        rng = np.random.default_rng(23)
        pts = rng.standard_normal((500, 2))
        r1 = summarize_dispersion(pts, seed=11)
        r2 = summarize_dispersion(pts, seed=11)
        assert r1.cep_ci == r2.cep_ci and r1.r95_ci == r2.r95_ci

    def test_validation(self):
        with pytest.raises(ValueError, match=r"\(n, 2\)"):
            summarize_dispersion(np.ones((50, 3)))
        with pytest.raises(ValueError, match="at least 10"):
            summarize_dispersion(np.ones((5, 2)))
        with pytest.raises(ValueError, match="degenerate"):
            summarize_dispersion(np.tile([1.0, 2.0], (100, 1)))
