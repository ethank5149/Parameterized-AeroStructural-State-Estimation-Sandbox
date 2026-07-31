"""Blended windward/leeward pressure closure (Paper II, §3.3).

Modified Newtonian pressure alone is inadequate for slender waveriders
at moderate incidence: it assigns zero pressure coefficient to every
leeward surface, discarding leeward suction. The closure is therefore
blended — modified Newtonian on windward panels (Eq. 3.5),

.. math::

    C_p = C_{p,\\max}\\sin^2\\delta_c, \\qquad
    C_{p,\\max} = \\frac{2}{\\gamma M_\\infty^2}
    \\left(\\frac{p_{02}}{p_\\infty} - 1\\right)

with the pressure ratio from the Rayleigh–Pitot relation, and
Prandtl–Meyer expansion on leeward panels down to the vacuum limit
:math:`C_p = -2/(\\gamma M_\\infty^2)`.

**The seam.** The two branches are :math:`C^0` but not :math:`C^1` at
:math:`\\delta_c = 0`: the windward branch has zero slope there while
the expansion branch does not. That is a genuine violation of the
smoothness the framework otherwise maintains, and it happens exactly at
the shoulder line where panels change branch. The branches are blended
over :math:`|\\delta_c| < \\delta_{\\mathrm{blend}}` by a :math:`C^2`
smoothstep. This is a numerical expedient, not physics — its effect on
integrated loads is verification task II-V4, which is why the blend
width is an explicit parameter everywhere rather than a hidden constant.

Both branches are evaluated by *smooth extensions* through the seam —
the Newtonian form is analytic in :math:`\\delta_c`, and the
Prandtl–Meyer branch continues into isentropic compression for
:math:`\\delta_c > 0` — so the blend inherits :math:`C^2` continuity
from the smoothstep alone. Clamping one branch at the seam instead would
leave a second-derivative kink and defeat the purpose.
"""

from __future__ import annotations

import numpy as np
import scipy.optimize
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "blended_pressure_coefficient",
    "newtonian_pressure_coefficient",
    "prandtl_meyer_angle",
    "prandtl_meyer_pressure_coefficient",
    "rayleigh_pitot_cp_max",
    "smoothstep",
    "vacuum_pressure_coefficient",
]

_FloatArray = NDArray[np.float64]


def _check_mach(mach: float) -> float:
    m = float(mach)
    if not (np.isfinite(m) and m > 1.0):
        raise ValueError(f"freestream Mach must be finite and supersonic, got {mach}")
    return m


def _check_gamma(gamma: float) -> float:
    g = float(gamma)
    if not (np.isfinite(g) and g > 1.0):
        raise ValueError(f"gamma must be finite and > 1, got {gamma}")
    return g


def rayleigh_pitot_cp_max(mach: float, gamma: float = 1.4) -> float:
    """:math:`C_{p,\\max}` behind a normal shock (Paper II, Eq. 3.5).

    Uses the Rayleigh–Pitot relation

    .. math::

        \\frac{p_{02}}{p_\\infty} =
        \\left[\\frac{(\\gamma+1)^2M^2}{4\\gamma M^2 - 2(\\gamma-1)}
        \\right]^{\\frac{\\gamma}{\\gamma-1}}
        \\frac{1 - \\gamma + 2\\gamma M^2}{\\gamma+1}.

    Approaches the classical hypersonic limit (1.839 for
    :math:`\\gamma = 1.4`) as :math:`M \\to \\infty`.
    """
    m = _check_mach(mach)
    g = _check_gamma(gamma)
    m2 = m * m
    ratio = ((g + 1.0) ** 2 * m2 / (4.0 * g * m2 - 2.0 * (g - 1.0))) ** (g / (g - 1.0))
    ratio *= (1.0 - g + 2.0 * g * m2) / (g + 1.0)
    return float(2.0 / (g * m2) * (ratio - 1.0))


def vacuum_pressure_coefficient(mach: float, gamma: float = 1.4) -> float:
    """The vacuum limit :math:`C_p = -2/(\\gamma M_\\infty^2)`."""
    m = _check_mach(mach)
    g = _check_gamma(gamma)
    return float(-2.0 / (g * m * m))


def newtonian_pressure_coefficient(
    incidence: ArrayLike, cp_max: float
) -> _FloatArray:
    """Modified Newtonian :math:`C_p = C_{p,\\max}\\sin^2\\delta_c`.

    Evaluated as the analytic function of :math:`\\delta_c`, without
    clamping at the seam — the smooth extension the blend requires.
    """
    delta = np.asarray(incidence, dtype=np.float64)
    return np.asarray(float(cp_max) * np.sin(delta) ** 2)


def prandtl_meyer_angle(mach: ArrayLike, gamma: float = 1.4) -> _FloatArray:
    """Prandtl–Meyer function :math:`\\nu(M)` in radians."""
    g = _check_gamma(gamma)
    m = np.asarray(mach, dtype=np.float64)
    if np.any(m < 1.0):
        raise ValueError("Prandtl–Meyer function requires M >= 1")
    root = np.sqrt(np.maximum(m * m - 1.0, 0.0))
    factor = np.sqrt((g + 1.0) / (g - 1.0))
    return np.asarray(factor * np.arctan(root / factor) - np.arctan(root))


def _mach_from_prandtl_meyer(nu: float, gamma: float) -> float:
    """Invert :math:`\\nu(M)` by Brent on the monotone branch."""
    nu_max = 0.5 * np.pi * (np.sqrt((gamma + 1.0) / (gamma - 1.0)) - 1.0)
    if nu <= 0.0:
        return 1.0
    if nu >= nu_max:
        return np.inf
    hi = 2.0
    while float(prandtl_meyer_angle(hi, gamma)) < nu:
        hi *= 2.0
        if hi > 1.0e6:  # pragma: no cover - nu < nu_max guarantees a bracket
            break
    return float(
        scipy.optimize.brentq(
            lambda m: float(prandtl_meyer_angle(m, gamma)) - nu, 1.0, hi, xtol=1e-13
        )
    )


def prandtl_meyer_pressure_coefficient(
    incidence: ArrayLike, mach: float, gamma: float = 1.4
) -> _FloatArray:
    """Leeward branch: isentropic turn through :math:`-\\delta_c`.

    Negative ``incidence`` is an expansion (the physical leeward case);
    positive ``incidence`` continues the same isentropic relation into
    compression, which is the smooth extension the blend evaluates
    inside the seam band. The result is floored at the vacuum limit,
    beyond which the surface is treated as being at vacuum pressure.
    """
    m1 = _check_mach(mach)
    g = _check_gamma(gamma)
    delta = np.atleast_1d(np.asarray(incidence, dtype=np.float64))
    nu1 = float(prandtl_meyer_angle(m1, g))
    cp_vac = vacuum_pressure_coefficient(m1, g)
    stagnation_factor = 1.0 + 0.5 * (g - 1.0) * m1 * m1

    out = np.empty_like(delta)
    for idx, d in np.ndenumerate(delta):
        nu2 = nu1 - float(d)
        # nu2 <= 0 means the turn has driven the flow back to sonic; the
        # isentropic branch ends there
        m2 = 1.0 if nu2 <= 0.0 else _mach_from_prandtl_meyer(nu2, g)
        if not np.isfinite(m2):
            out[idx] = cp_vac
            continue
        ratio = (stagnation_factor / (1.0 + 0.5 * (g - 1.0) * m2 * m2)) ** (
            g / (g - 1.0)
        )
        out[idx] = 2.0 / (g * m1 * m1) * (ratio - 1.0)
    out = np.maximum(out, cp_vac)
    return np.asarray(out.reshape(np.shape(incidence)))


def smoothstep(t: ArrayLike) -> _FloatArray:
    """:math:`C^2` smoothstep :math:`6t^5 - 15t^4 + 10t^3` on :math:`[0,1]`.

    Its first *and* second derivatives vanish at both ends, which is
    what makes the blended closure :math:`C^2` at the band edges — a
    cubic smoothstep would only give :math:`C^1`.
    """
    x = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    return np.asarray(x * x * x * (x * (6.0 * x - 15.0) + 10.0))


def blended_pressure_coefficient(
    incidence: ArrayLike,
    mach: float,
    gamma: float = 1.4,
    blend_width: float = 0.02,
    cp_max: float | None = None,
) -> _FloatArray:
    """Blended :math:`C_p` over the whole incidence range.

    Parameters
    ----------
    incidence:
        Local incidence :math:`\\delta_c` (rad); positive windward.
    mach, gamma:
        Freestream conditions.
    blend_width:
        :math:`\\delta_{\\mathrm{blend}}` (rad), the half-width of the
        seam band. Must be small relative to the incidence variation
        across a collocation cell; zero selects the unblended
        :math:`C^0` closure, which II-V4 uses as its baseline.
    cp_max:
        Override for :math:`C_{p,\\max}`; defaults to the Rayleigh–Pitot
        value at ``mach``.
    """
    m = _check_mach(mach)
    g = _check_gamma(gamma)
    width = float(blend_width)
    if not (np.isfinite(width) and width >= 0.0):
        raise ValueError(f"blend_width must be finite and >= 0, got {blend_width}")
    if width >= 0.5 * np.pi:
        raise ValueError(
            f"blend_width {width} spans the whole incidence range; it must be "
            f"small relative to the incidence variation across a cell"
        )
    cpm = rayleigh_pitot_cp_max(m, g) if cp_max is None else float(cp_max)

    delta = np.asarray(incidence, dtype=np.float64)
    windward = newtonian_pressure_coefficient(delta, cpm)
    leeward = prandtl_meyer_pressure_coefficient(delta, m, g)

    if width == 0.0:
        return np.asarray(np.where(delta > 0.0, windward, leeward))
    weight = smoothstep((delta + width) / (2.0 * width))
    return np.asarray((1.0 - weight) * leeward + weight * windward)
