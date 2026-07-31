"""Stagnation-point convective heating (Paper II, §4.1).

Fay–Riddell (Eq. 4.1) with the terms Paper II insists on specifying
because the equation "is frequently quoted without them": the Lewis
exponent is an explicit argument with the two physically distinct values
named (0.52 equilibrium boundary layer, 0.63 frozen with fully catalytic
wall), the dissociation enthalpy enters through the bracket, and the
stagnation velocity gradient comes from the modified Newtonian estimate
(Eq. 4.2) — which is where recession feeds back: growing
:math:`R_{\\mathrm{eff}}` reduces convective heating as
:math:`R_{\\mathrm{eff}}^{-1/2}`.

The Sutton–Graves correlation is provided as the screening fallback the
Remark in §4.1 describes, as a *separate function*: it is a correlation,
not a theory, and results obtained with it must not be reported as
Fay–Riddell results.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "LEWIS_EXPONENT_EQUILIBRIUM",
    "LEWIS_EXPONENT_FROZEN_CATALYTIC",
    "SUTTON_GRAVES_EARTH",
    "fay_riddell",
    "newtonian_velocity_gradient",
    "sutton_graves",
]

_FloatArray = NDArray[np.float64]

#: Lewis exponent for an equilibrium boundary layer (Paper II, §4.1).
LEWIS_EXPONENT_EQUILIBRIUM = 0.52
#: Lewis exponent for a frozen boundary layer with a fully catalytic wall.
LEWIS_EXPONENT_FROZEN_CATALYTIC = 0.63
#: Sutton–Graves constant for Earth air, SI units (Paper II, §4.1 Remark).
SUTTON_GRAVES_EARTH = 1.7415e-4


def _positive(value: ArrayLike, name: str) -> _FloatArray:
    arr = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"{name} must be finite and > 0")
    return arr


def newtonian_velocity_gradient(
    effective_radius: ArrayLike,
    stagnation_pressure: ArrayLike,
    freestream_pressure: ArrayLike,
    stagnation_density: ArrayLike,
) -> _FloatArray:
    """Stagnation velocity gradient :math:`(du_e/dx)_s` (Paper II, Eq. 4.2).

    .. math::

        \\left(\\frac{du_e}{dx}\\right)_s = \\frac{1}{R_{\\mathrm{eff}}}
        \\sqrt{\\frac{2(p_s - p_\\infty)}{\\rho_s}} .

    Units: 1/s. This is the ablation coupling point — recession updates
    :math:`R_{\\mathrm{eff}}` here within the same step.
    """
    r_eff = _positive(effective_radius, "effective_radius")
    p_s = _positive(stagnation_pressure, "stagnation_pressure")
    p_inf = np.asarray(freestream_pressure, dtype=np.float64)
    rho_s = _positive(stagnation_density, "stagnation_density")
    if np.any(p_inf < 0.0) or not np.all(np.isfinite(p_inf)):
        raise ValueError("freestream_pressure must be finite and >= 0")
    if np.any(p_s <= p_inf):
        raise ValueError(
            "stagnation pressure must exceed freestream pressure; the modified "
            "Newtonian estimate has no meaning otherwise"
        )
    return np.asarray(np.sqrt(2.0 * (p_s - p_inf) / rho_s) / r_eff)


def fay_riddell(
    edge_density: ArrayLike,
    edge_viscosity: ArrayLike,
    wall_density: ArrayLike,
    wall_viscosity: ArrayLike,
    velocity_gradient: ArrayLike,
    total_enthalpy_edge: ArrayLike,
    wall_enthalpy: ArrayLike,
    dissociation_enthalpy: ArrayLike,
    prandtl: float = 0.71,
    lewis: float = 1.4,
    lewis_exponent: float = LEWIS_EXPONENT_EQUILIBRIUM,
) -> _FloatArray:
    """Fay–Riddell stagnation convective heat flux (Paper II, Eq. 4.1), W/m².

    .. math::

        \\dot q_{s} = 0.763\\,\\mathrm{Pr}^{-0.6}
        (\\rho_e \\mu_e)^{0.4} (\\rho_w \\mu_w)^{0.1}
        \\sqrt{(du_e/dx)_s}\\,(h_{0e} - h_w)
        \\Big[1 + \\big(Le^{\\beta_{Le}} - 1\\big) \\tfrac{h_D}{h_{0e}}\\Big].

    Parameters
    ----------
    edge_density, edge_viscosity:
        Boundary-layer edge :math:`\\rho_e` (kg/m³), :math:`\\mu_e`
        (Pa·s).
    wall_density, wall_viscosity:
        Gas properties at wall temperature.
    velocity_gradient:
        :math:`(du_e/dx)_s` (1/s), from
        :func:`newtonian_velocity_gradient`.
    total_enthalpy_edge, wall_enthalpy:
        :math:`h_{0e}`, :math:`h_w` (J/kg), with
        :math:`h_{0e} > h_w` (heating, not cooling).
    dissociation_enthalpy:
        Free-stream mixture dissociation enthalpy :math:`h_D` (J/kg) at
        edge conditions, :math:`0 \\le h_D \\le h_{0e}`.
    prandtl, lewis:
        :math:`\\Pr \\approx 0.71`, :math:`Le \\approx 1.4` for air.
    lewis_exponent:
        :math:`\\beta_{Le}`: 0.52 equilibrium, 0.63 frozen/catalytic —
        an explicit choice, not a default hidden in a constant, because
        the bracket differs by several percent between them.
    """
    if not (np.isfinite(prandtl) and prandtl > 0.0):
        raise ValueError(f"prandtl must be finite and > 0, got {prandtl}")
    if not (np.isfinite(lewis) and lewis > 0.0):
        raise ValueError(f"lewis must be finite and > 0, got {lewis}")
    if not (np.isfinite(lewis_exponent) and 0.0 < lewis_exponent < 1.0):
        raise ValueError(f"lewis_exponent must be in (0, 1), got {lewis_exponent}")
    rho_mu_e = _positive(edge_density, "edge_density") * _positive(
        edge_viscosity, "edge_viscosity"
    )
    rho_mu_w = _positive(wall_density, "wall_density") * _positive(
        wall_viscosity, "wall_viscosity"
    )
    dudx = _positive(velocity_gradient, "velocity_gradient")
    h0e = _positive(total_enthalpy_edge, "total_enthalpy_edge")
    h_w = np.asarray(wall_enthalpy, dtype=np.float64)
    h_d = np.asarray(dissociation_enthalpy, dtype=np.float64)
    if np.any(h_w >= h0e):
        raise ValueError("wall enthalpy must be below edge total enthalpy (heating case)")
    if np.any(h_d < 0.0) or np.any(h_d > h0e):
        raise ValueError("dissociation enthalpy must satisfy 0 <= h_D <= h_0e")

    bracket = 1.0 + (lewis**lewis_exponent - 1.0) * h_d / h0e
    return np.asarray(
        0.763
        * prandtl**-0.6
        * rho_mu_e**0.4
        * rho_mu_w**0.1
        * np.sqrt(dudx)
        * (h0e - h_w)
        * bracket
    )


def sutton_graves(
    freestream_density: ArrayLike,
    effective_radius: ArrayLike,
    freestream_velocity: ArrayLike,
    constant: float = SUTTON_GRAVES_EARTH,
) -> _FloatArray:
    """Sutton–Graves screening correlation
    :math:`\\dot q_s = k_{SG}\\sqrt{\\rho_\\infty/R_{\\mathrm{eff}}}\\,V_\\infty^3`
    (W/m²).

    This is the fallback of the Remark in Paper II §4.1 for use before a
    boundary-layer edge solution exists. It is a correlation, not a
    theory; results computed here must not be reported as Fay–Riddell
    results.
    """
    if not (np.isfinite(constant) and constant > 0.0):
        raise ValueError(f"constant must be finite and > 0, got {constant}")
    rho = _positive(freestream_density, "freestream_density")
    r_eff = _positive(effective_radius, "effective_radius")
    v = _positive(freestream_velocity, "freestream_velocity")
    return np.asarray(constant * np.sqrt(rho / r_eff) * v**3)
