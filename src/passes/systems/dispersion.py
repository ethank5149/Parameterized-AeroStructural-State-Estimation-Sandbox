"""Terminal accuracy statistics: CEP and the 95% radius.

A dispersion is reported as a one-sigma number by the modules that produce
it, and consumed as a containment radius by anyone asking how accurate a
system is. This module is the conversion, and it is less trivial than it
looks because the usual shortcut is only valid in a case that rarely holds.

The circular case, and why the shortcut fails
---------------------------------------------

For an isotropic bivariate normal with per-axis sigma :math:`\\sigma`, the
miss distance is Rayleigh distributed and the radius containing a fraction
:math:`p` is available in closed form:

.. math::

    R_p = \\sigma \\sqrt{-2 \\ln(1 - p)},

so :math:`\\mathrm{CEP} = \\sigma\\sqrt{2\\ln 2} \\approx 1.1774\\sigma` and
:math:`R_{95} = \\sigma\\sqrt{2\\ln 20} \\approx 2.4477\\sigma`. Their ratio
is a constant, :math:`R_{95}/\\mathrm{CEP} \\approx 2.079`, and it is
extremely common to convert between the two by that factor.

**That factor is only correct for a circular distribution**, and terminal
dispersions are usually not circular: a glide vehicle's downrange error and
its crossrange error come from different mechanisms and differ by a factor
of several. As the ellipse elongates the ratio **rises**: measured here it
runs 2.079 at unit aspect ratio, 2.339 at 2:1, 2.646 at 3.3:1, and
approaches 2.906 in the degenerate one-dimensional limit — which is exactly
:math:`1.96/0.6745`, the normal-distribution equivalent. Scaling a CEP up
to a 95% radius by the circular 2.079 therefore **under-states** the 95%
radius for any real, elongated dispersion, by up to 40% at the limit.
:func:`containment_ratio` computes it instead of assuming it.

The elliptical case
-------------------

For principal-axis sigmas :math:`\\sigma_1, \\sigma_2` the probability of
falling inside radius :math:`r` has no elementary closed form, but it
reduces to a one-dimensional integral that is exact and cheap. Working in
polar coordinates the radial part integrates analytically, leaving

.. math::

    P(R \\le r) = \\frac{1}{2\\pi\\sigma_1\\sigma_2}
        \\int_0^{2\\pi} \\frac{1 - e^{-a(\\theta) r^2}}{2a(\\theta)}
        \\,\\mathrm{d}\\theta,
    \\qquad
    a(\\theta) = \\frac{\\cos^2\\theta}{2\\sigma_1^2}
              + \\frac{\\sin^2\\theta}{2\\sigma_2^2},

which is quadrature in one variable and then a root-find in :math:`r`. No
approximation formula is used, so the elliptical answers are as exact as
the circular ones rather than being a fitted correction to them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.integrate
import scipy.optimize
import scipy.special

__all__ = [
    "CEP_OVER_SIGMA",
    "R95_OVER_SIGMA",
    "AccuracyStatistics",
    "accuracy_statistics",
    "containment_probability",
    "containment_radius",
    "containment_ratio",
]

#: Circular-case constants, exact.
CEP_OVER_SIGMA = float(np.sqrt(2.0 * np.log(2.0)))
R95_OVER_SIGMA = float(np.sqrt(-2.0 * np.log(0.05)))


def _check_sigmas(sigma_major: float, sigma_minor: float) -> tuple[float, float]:
    a, b = float(sigma_major), float(sigma_minor)
    for name, value in (("sigma_major", a), ("sigma_minor", b)):
        if not (np.isfinite(value) and value >= 0.0):
            raise ValueError(f"{name} must be finite and >= 0, got {value}")
    if a < b:
        a, b = b, a
    if a == 0.0:
        raise ValueError(
            "at least one sigma must be positive; a distribution with no "
            "spread has no containment radius to report"
        )
    return a, b


def containment_probability(radius: float, sigma_major: float, sigma_minor: float) -> float:
    """Probability of falling within ``radius`` of the aimpoint.

    Exact for any aspect ratio: the radial integral is analytic and only
    the angular one is quadrature.
    """
    a, b = _check_sigmas(sigma_major, sigma_minor)
    r = float(radius)
    if not (np.isfinite(r) and r >= 0.0):
        raise ValueError(f"radius must be finite and >= 0, got {r}")
    if r == 0.0:
        return 0.0
    if b == 0.0:
        # Degenerate to one dimension: the "disc" is an interval.
        return float(scipy.special.erf(r / (a * np.sqrt(2.0))))

    def integrand(theta: float) -> float:
        coefficient = np.cos(theta) ** 2 / (2.0 * a**2) + np.sin(theta) ** 2 / (2.0 * b**2)
        return float((1.0 - np.exp(-coefficient * r**2)) / (2.0 * coefficient))

    # Fourfold symmetry: integrate a quadrant and scale.
    value, _ = scipy.integrate.quad(integrand, 0.0, 0.5 * np.pi, limit=200)
    return float(4.0 * value / (2.0 * np.pi * a * b))


def containment_radius(
    probability: float, sigma_major: float, sigma_minor: float | None = None
) -> float:
    """Radius (same units as sigma) containing ``probability`` of the miss.

    ``sigma_minor`` defaults to ``sigma_major``, i.e. the circular case,
    which is then evaluated from the closed form rather than by root-find.
    """
    p = float(probability)
    if not (0.0 < p < 1.0):
        raise ValueError(f"probability must lie in (0, 1), got {p}")
    minor = sigma_major if sigma_minor is None else sigma_minor
    a, b = _check_sigmas(sigma_major, minor)
    if a == b:
        return float(a * np.sqrt(-2.0 * np.log(1.0 - p)))
    # Bracket generously: the elliptical radius lies between the circular
    # answers for the two sigmas.
    low = b * np.sqrt(-2.0 * np.log(1.0 - p))
    high = a * np.sqrt(-2.0 * np.log(1.0 - p))
    return float(
        scipy.optimize.brentq(
            lambda r: containment_probability(r, a, b) - p,
            0.5 * low,
            2.0 * high,
            xtol=1e-9 * a,
        )
    )


def containment_ratio(sigma_major: float, sigma_minor: float) -> float:
    """:math:`R_{95}/\\mathrm{CEP}` for this ellipse.

    Equal to 2.079 for a circular distribution and *larger* for an
    elongated one, approaching 2.906 in the one-dimensional limit. Scaling
    a CEP by the circular constant therefore under-states the 95% radius
    for any real dispersion, and this is the function that says by how
    much.
    """
    a, b = _check_sigmas(sigma_major, sigma_minor)
    return containment_radius(0.95, a, b) / containment_radius(0.5, a, b)


@dataclass(frozen=True)
class AccuracyStatistics:
    """Terminal accuracy of one body.

    Attributes
    ----------
    sigma_major, sigma_minor:
        Principal-axis one-sigma dispersions (m).
    cep:
        Circular error probable (m) — the radius containing half the
        impacts.
    r95:
        Radius (m) containing 95%.
    ratio:
        ``r95 / cep``, computed rather than assumed. Departure from 2.079
        measures how far from circular the dispersion is.
    """

    sigma_major: float
    sigma_minor: float
    cep: float
    r95: float
    ratio: float

    @property
    def is_circular(self) -> bool:
        return abs(self.ratio / (R95_OVER_SIGMA / CEP_OVER_SIGMA) - 1.0) < 1e-6


def accuracy_statistics(sigma_major: float, sigma_minor: float | None = None) -> AccuracyStatistics:
    """CEP and 95% radius from principal-axis sigmas."""
    minor = sigma_major if sigma_minor is None else sigma_minor
    a, b = _check_sigmas(sigma_major, minor)
    cep = containment_radius(0.5, a, b)
    r95 = containment_radius(0.95, a, b)
    return AccuracyStatistics(
        sigma_major=a,
        sigma_minor=b,
        cep=cep,
        r95=r95,
        ratio=r95 / cep,
    )
