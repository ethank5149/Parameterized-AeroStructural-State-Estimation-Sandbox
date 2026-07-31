"""Aerothermal correlations (Paper II, §4)."""

import numpy as np
import pytest

from passes.aerothermal import (
    TauberSuttonRadiation,
    fay_riddell,
    lees_distribution,
    newtonian_velocity_gradient,
    stefan_recession_rate,
    sutton_graves,
)
from passes.aerothermal.stagnation import (
    LEWIS_EXPONENT_EQUILIBRIUM,
    LEWIS_EXPONENT_FROZEN_CATALYTIC,
    SUTTON_GRAVES_EARTH,
)
from passes.thermal.surface import STEFAN_BOLTZMANN

# representative (generic) stagnation inputs used across tests
FR_ARGS = {
    "edge_density": 0.05,
    "edge_viscosity": 6.0e-5,
    "wall_density": 0.3,
    "wall_viscosity": 4.0e-5,
    "velocity_gradient": 5.0e3,
    "total_enthalpy_edge": 2.0e7,
    "wall_enthalpy": 1.5e6,
    "dissociation_enthalpy": 5.0e6,
}


def synthetic_ts():
    """Synthetic velocity-function surrogate, clearly labeled as such."""
    v = np.linspace(9000.0, 16000.0, 15)
    f = (v / 9000.0) ** 8.5
    return TauberSuttonRadiation(
        v, f, coefficient=100.0,
        provenance="synthetic V^8.5 surrogate for unit tests; NOT Tauber–Sutton table data",
    )


class TestVelocityGradient:
    def test_formula(self):
        dudx = newtonian_velocity_gradient(0.5, 5.0e4, 100.0, 0.05)
        assert float(dudx) == pytest.approx(np.sqrt(2 * (5.0e4 - 100.0) / 0.05) / 0.5)

    def test_radius_scaling(self):
        d1 = newtonian_velocity_gradient(0.2, 5.0e4, 100.0, 0.05)
        d2 = newtonian_velocity_gradient(0.4, 5.0e4, 100.0, 0.05)
        assert float(d1 / d2) == pytest.approx(2.0)

    def test_rejects_nonphysical(self):
        with pytest.raises(ValueError, match="exceed"):
            newtonian_velocity_gradient(0.5, 100.0, 200.0, 0.05)
        with pytest.raises(ValueError):
            newtonian_velocity_gradient(-0.5, 5e4, 100.0, 0.05)


class TestFayRiddell:
    def test_heating_scales_as_sqrt_velocity_gradient(self):
        """q ∝ (du/dx)^{1/2} — hence R_eff^{-1/2} through Eq. 4.2."""
        args = dict(FR_ARGS)
        q1 = fay_riddell(**args)
        args["velocity_gradient"] = 4.0 * FR_ARGS["velocity_gradient"]
        q2 = fay_riddell(**args)
        assert float(q2 / q1) == pytest.approx(2.0, rel=1e-12)

    def test_driving_potential_linear(self):
        args = dict(FR_ARGS)
        q1 = fay_riddell(**args)
        args["wall_enthalpy"] = 0.5 * (
            FR_ARGS["total_enthalpy_edge"] + FR_ARGS["wall_enthalpy"]
        )
        q2 = fay_riddell(**args)
        ratio_expected = (FR_ARGS["total_enthalpy_edge"] - args["wall_enthalpy"]) / (
            FR_ARGS["total_enthalpy_edge"] - FR_ARGS["wall_enthalpy"]
        )
        assert float(q2 / q1) == pytest.approx(ratio_expected, rel=1e-12)

    def test_lewis_exponent_choice_matters_by_percent(self):
        """Paper II: the bracketed factor differs by several percent
        between equilibrium and frozen/catalytic."""
        q_eq = fay_riddell(**FR_ARGS, lewis_exponent=LEWIS_EXPONENT_EQUILIBRIUM)
        q_fr = fay_riddell(**FR_ARGS, lewis_exponent=LEWIS_EXPONENT_FROZEN_CATALYTIC)
        diff = float(q_fr / q_eq) - 1.0
        assert 0.002 < diff < 0.1, f"exponent sensitivity {diff:.4f}"

    def test_bracket_reduces_to_unity_without_dissociation(self):
        args = dict(FR_ARGS)
        args["dissociation_enthalpy"] = 0.0
        q_eq = fay_riddell(**args, lewis_exponent=0.52)
        q_fr = fay_riddell(**args, lewis_exponent=0.63)
        assert float(q_eq) == pytest.approx(float(q_fr), rel=1e-14)

    def test_magnitude_is_reentry_scale(self):
        """Generic peak-heating inputs must land in the MW/m² decade —
        a sanity check on units, not a validation claim."""
        q = float(fay_riddell(**FR_ARGS))
        assert 1.0e5 < q < 1.0e8

    def test_validation(self):
        bad = dict(FR_ARGS)
        bad["wall_enthalpy"] = 3.0e7
        with pytest.raises(ValueError, match="below edge total"):
            fay_riddell(**bad)
        bad = dict(FR_ARGS)
        bad["dissociation_enthalpy"] = 3.0e7
        with pytest.raises(ValueError, match="h_D"):
            fay_riddell(**bad)
        with pytest.raises(ValueError, match="lewis_exponent"):
            fay_riddell(**FR_ARGS, lewis_exponent=1.5)


class TestSuttonGraves:
    def test_formula_and_constant(self):
        q = sutton_graves(1.0e-3, 0.5, 5000.0)
        assert float(q) == pytest.approx(
            SUTTON_GRAVES_EARTH * np.sqrt(1.0e-3 / 0.5) * 5000.0**3
        )

    def test_velocity_cubed(self):
        assert float(
            sutton_graves(1e-3, 0.5, 6000.0) / sutton_graves(1e-3, 0.5, 3000.0)
        ) == pytest.approx(8.0)

    def test_same_radius_trade_direction_as_fay_riddell(self):
        """Both convective models must decrease with blunting."""
        assert float(sutton_graves(1e-3, 0.8, 5000.0)) < float(
            sutton_graves(1e-3, 0.2, 5000.0)
        )


class TestTauberSutton:
    def test_exponents(self):
        ts = synthetic_ts()
        base = ts.heat_flux(0.5, 1e-3, 12000.0)
        assert float(ts.heat_flux(1.0, 1e-3, 12000.0) / base) == pytest.approx(2.0)
        assert float(ts.heat_flux(0.5, 2e-3, 12000.0) / base) == pytest.approx(
            2.0**1.22, rel=1e-12
        )

    def test_interpolates_table_nodes_exactly(self):
        v = np.linspace(9000.0, 16000.0, 15)
        f = (v / 9000.0) ** 8.5
        ts = synthetic_ts()
        for vi, fi in zip(v, f, strict=True):
            assert float(ts.heat_flux(1.0, 1.0, vi)) == pytest.approx(
                100.0 * fi, rel=1e-12
            )

    def test_opposite_radius_trade_produces_interior_optimum(self):
        """Paper II §4.2: q_conv falls and q_rad rises with R_eff, so the
        total has an interior minimum — the blunting trade. The surrogate
        coefficient is scaled so the two components cross near R = 1 m."""
        rho, v = 3.0e-4, 12000.0
        vv = np.linspace(9000.0, 16000.0, 15)
        ff = (vv / 9000.0) ** 8.5
        q_conv_at_1 = float(sutton_graves(rho, 1.0, v))
        f_at_v = np.interp(v, vv, ff)
        coeff = q_conv_at_1 / (rho**1.22 * f_at_v)  # q_rad(R=1) = q_conv(R=1)
        ts = TauberSuttonRadiation(
            vv, ff, coefficient=coeff,
            provenance="synthetic surrogate scaled for the blunting-trade test",
        )
        radii = np.linspace(0.05, 3.0, 120)
        total = sutton_graves(rho, radii, v) + ts.heat_flux(
            radii, rho, np.full_like(radii, v)
        )
        i_min = int(np.argmin(total))
        assert 0 < i_min < radii.size - 1, "optimum radius must be interior"

    def test_refuses_extrapolation_and_bad_tables(self):
        ts = synthetic_ts()
        with pytest.raises(ValueError, match="outside tabulated"):
            ts.heat_flux(0.5, 1e-3, 8000.0)
        v = np.linspace(9000.0, 16000.0, 15)
        with pytest.raises(ValueError, match="provenance"):
            TauberSuttonRadiation(v, (v / 9000.0) ** 8, coefficient=1.0)
        with pytest.raises(ValueError, match="non-decreasing"):
            TauberSuttonRadiation(
                v, np.linspace(10, 1, 15), coefficient=1.0, provenance="x"
            )


class TestLeesDistribution:
    def test_continuous_at_stagnation_region_boundary(self):
        r_eff = 0.4
        eps = 1e-9
        inner = lees_distribution(1e6, 0.8, r_eff - eps, r_eff)
        outer = lees_distribution(1e6, 0.8, r_eff + eps, r_eff)
        assert float(inner) == pytest.approx(float(outer), rel=1e-6)

    def test_stagnation_region_plateau_and_decay(self):
        x = np.array([0.0, 0.1, 0.4, 1.6, 6.4])
        q = lees_distribution(1e6, np.pi / 2, x, 0.4)
        assert q[0] == q[1] == q[2] == pytest.approx(1e6)
        assert float(q[3]) == pytest.approx(1e6 * 0.5)  # sqrt(0.4/1.6)
        assert float(q[4]) == pytest.approx(1e6 * 0.25)

    def test_leeward_zero(self):
        q = lees_distribution(1e6, np.array([-0.3, 0.0, 0.3]), np.full(3, 1.0), 0.4)
        assert q[0] == 0.0 and q[1] == 0.0 and q[2] > 0.0

    def test_incidence_factor(self):
        q30 = lees_distribution(1e6, np.pi / 6, 2.0, 0.4)
        q90 = lees_distribution(1e6, np.pi / 2, 2.0, 0.4)
        assert float(q30 / q90) == pytest.approx(0.5, rel=1e-12)

    def test_validation(self):
        with pytest.raises(ValueError, match="stagnation_flux"):
            lees_distribution(-1.0, 0.5, 1.0, 0.4)
        with pytest.raises(ValueError, match="running_length"):
            lees_distribution(1e6, 0.5, -1.0, 0.4)


class TestStefanRecession:
    def test_energy_balance(self):
        # T_w chosen so the net flux is positive (re-radiation ~7.7e5 W/m2)
        q, t_w, eps, q_cond = 2.0e6, 2000.0, 0.85, 3.0e5
        rho, dh = 1900.0, 2.0e7
        sdot = stefan_recession_rate(q, t_w, eps, q_cond, rho, dh)
        net = q - eps * STEFAN_BOLTZMANN * t_w**4 - q_cond
        assert float(sdot) == pytest.approx(net / (rho * dh), rel=1e-14)

    def test_irreversibility_clamp(self):
        """Net cooling must give zero recession, not negative."""
        sdot = stefan_recession_rate(1.0e4, 3000.0, 0.9, 0.0, 1900.0, 2.0e7)
        assert float(sdot) == 0.0

    def test_balanced_wall_no_recession(self):
        t_w = 2000.0
        eps = 0.85
        q = eps * STEFAN_BOLTZMANN * t_w**4  # exactly re-radiated
        assert float(stefan_recession_rate(q, t_w, eps, 0.0, 1900.0, 2e7)) == 0.0

    def test_validation(self):
        with pytest.raises(ValueError, match="emissivity"):
            stefan_recession_rate(1e6, 2500.0, 1.5, 0.0, 1900.0, 2e7)
        with pytest.raises(ValueError, match="wall_temperature"):
            stefan_recession_rate(1e6, -1.0, 0.9, 0.0, 1900.0, 2e7)
        with pytest.raises(ValueError, match="material_density"):
            stefan_recession_rate(1e6, 2500.0, 0.9, 0.0, -1.0, 2e7)
