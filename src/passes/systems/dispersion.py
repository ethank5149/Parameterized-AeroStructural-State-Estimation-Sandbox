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

Verified against a published table, not only against itself
-----------------------------------------------------------

That integral was for a long time checked only by its own limits. Siouris
Table 5.2 tabulates :math:`K` such that :math:`P(R \\le K\\sigma_L) = P`,
over 21 aspect ratios from degenerate to circular and six probability
levels — 126 published numbers spanning the whole domain this module
covers. :data:`SIOURIS_TABLE_5_2` carries it, and
:func:`containment_radius` reproduces **every entry to within one unit in
the last printed place**, with 122 of the 126 inside ideal four-decimal
rounding. Both endpoints land on their closed forms:
:math:`1.1774 = \\sqrt{2\\ln 2}` in the circular column, and
:math:`0.6745`, :math:`1.9600` — the one-dimensional probable error and the
familiar :math:`1.96\\sigma` — in the degenerate one.

The four exceptions are the *source's* rounding, not ours, and they are
identifiable rather than merely tolerated: in each case the exact value
rounds to a different last digit than the one printed.

.. code-block:: text

    sigma_S/sigma_L   P      published   exact        rounds to
        0.00         0.75     1.1504     1.1503494     1.1503
        0.05         0.50     0.6764     0.6763479     0.6763
        0.75         0.90     1.9034     1.9033494     1.9033
        1.00         0.95     2.4478     2.4477468     2.4477

All four are high by 5.1-5.3e-5, i.e. the table rounds up where the value
rounds down. Across all 126 entries the signed deviations split 69 positive
to 57 negative with a mean of +6.4e-6, so this is four last-place roundings
rather than a bias in either direction.

The classical approximations, and where they stop working
----------------------------------------------------------

Before the integral was cheap, this conversion was done with fitted
formulae, and they remain the vocabulary of the weapon-delivery
literature: a dispersion is reported as REP and DEP (range and deflection
*probable errors*, :math:`0.6745\\sigma` each) rather than as sigmas.
:func:`probable_error` and :func:`cep_from_probable_errors` provide them so
that results here can be read against that literature.

They are provided with their errors measured rather than with their
validity asserted. Against the exact integral:

.. code-block:: text

    sigma_S/sigma_L   Eq. (5.17) error   Eq. (5.13) error
        1.00               +0.02 %              --
        0.70               +0.48 %              --
        0.50               +1.48 %           +2.49 %
        0.35               +2.07 %           -0.02 %
        0.28                  --             -0.05 %
        0.20               +0.11 %           +0.10 %
        0.10               -5.02 %           +0.01 %

Two things fall out that the source does not state. Siouris says
Eq. (5.17) holds "even when REP and DEP differ by a factor as much as
two"; at exactly that limit the error is **1.5 %**, so the claim is sound,
but the error is **not monotone** — it peaks near 2 % at a ratio of about
3:1, crosses zero near 5:1, and then diverges. Anyone extrapolating past
the stated bound by watching the error shrink would be misled. And
Eq. (5.13), stated for :math:`\\sigma_S/\\sigma_L < 0.28`, is **better than
advertised**: it holds to 0.1 % out to at least 0.35.

None of this is used internally. :func:`containment_radius` is exact and
costs a quadrature; the approximations exist for reading and reporting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.integrate
import scipy.optimize
import scipy.special

__all__ = [
    "CEP_OVER_SIGMA",
    "PROBABLE_ERROR_OVER_SIGMA",
    "R95_OVER_SIGMA",
    "SIOURIS_TABLE_5_2",
    "SIOURIS_TABLE_5_2_PROBABILITIES",
    "SIOURIS_TABLE_5_2_RATIOS",
    "AccuracyStatistics",
    "accuracy_statistics",
    "cep_from_probable_errors",
    "cep_small_ratio",
    "containment_probability",
    "containment_radius",
    "containment_ratio",
    "probable_error",
]

#: Circular-case constants, exact.
CEP_OVER_SIGMA = float(np.sqrt(2.0 * np.log(2.0)))
R95_OVER_SIGMA = float(np.sqrt(-2.0 * np.log(0.05)))

#: One-dimensional probable error over sigma: the 50 % point of a normal,
#: :math:`\Phi^{-1}(0.75) = 0.674490`. Siouris prints 0.6745.
PROBABLE_ERROR_OVER_SIGMA = 0.6744897501960817

#: Aspect ratios :math:`\sigma_S/\sigma_L` indexing Siouris Table 5.2.
SIOURIS_TABLE_5_2_RATIOS: tuple[float, ...] = (
    0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
)

#: Containment probabilities columning Siouris Table 5.2.
SIOURIS_TABLE_5_2_PROBABILITIES: tuple[float, ...] = (0.30, 0.50, 0.75, 0.90, 0.95, 0.99)

#: Siouris Table 5.2, transcribed as published: :math:`K` such that
#: :math:`P(R \le K\sigma_L) = P`, rows indexed by
#: :data:`SIOURIS_TABLE_5_2_RATIOS` and columns by
#: :data:`SIOURIS_TABLE_5_2_PROBABILITIES`.
#:
#: Kept as published rather than regenerated from
#: :func:`containment_radius`, which would make the comparison circular and
#: destroy the only independent check this module has. The first row is the
#: degenerate :math:`\sigma_S = 0` case, where the answer is the
#: one-dimensional normal quantile rather than a two-dimensional one.
SIOURIS_TABLE_5_2: tuple[tuple[float, ...], ...] = (
    (0.3853, 0.6745, 1.1504, 1.6449, 1.9600, 2.5758),
    (0.3886, 0.6764, 1.1514, 1.6456, 1.9606, 2.5763),
    (0.3987, 0.6820, 1.1547, 1.6479, 1.9625, 2.5778),
    (0.4169, 0.6916, 1.1603, 1.6518, 1.9658, 2.5803),
    (0.4421, 0.7059, 1.1683, 1.6573, 1.9704, 2.5838),
    (0.4705, 0.7254, 1.1788, 1.6646, 1.9765, 2.5884),
    (0.4997, 0.7499, 1.1925, 1.6738, 1.9842, 2.5942),
    (0.5285, 0.7779, 1.2097, 1.6852, 1.9937, 2.6013),
    (0.5568, 0.8079, 1.2310, 1.6992, 2.0051, 2.6100),
    (0.5842, 0.8389, 1.2564, 1.7163, 2.0190, 2.6203),
    (0.6109, 0.8704, 1.2853, 1.7371, 2.0359, 2.6326),
    (0.6369, 0.9021, 1.3172, 1.7621, 2.0564, 2.6474),
    (0.6621, 0.9337, 1.3514, 1.7915, 2.0813, 2.6653),
    (0.6867, 0.9651, 1.3874, 1.8251, 2.1111, 2.6875),
    (0.7107, 0.9962, 1.4247, 1.8625, 2.1460, 2.7151),
    (0.7342, 1.0271, 1.4631, 1.9034, 2.1858, 2.7492),
    (0.7571, 1.0577, 1.5023, 1.9472, 2.2303, 2.7907),
    (0.7796, 1.0880, 1.5422, 1.9936, 2.2791, 2.8401),
    (0.8017, 1.1181, 1.5827, 2.0424, 2.3318, 2.8974),
    (0.8233, 1.1479, 1.6237, 2.0932, 2.3881, 2.9625),
    (0.8446, 1.1774, 1.6651, 2.1460, 2.4478, 3.0349),
)


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


def probable_error(sigma: float) -> float:
    """One-dimensional probable error, :math:`0.6745\\sigma`.

    The 50 % point of a single normal axis: half of all misses fall within
    this distance of the aimpoint *in that axis alone*. Applied to the
    downrange axis this is REP (range probable error), and to the
    crossrange axis DEP (deflection probable error) — the pair in which
    weapon-delivery accuracy is conventionally quoted.

    Not to be confused with CEP, which is a *radial* 50 % point in two
    dimensions and is always the larger number.

    Parameters
    ----------
    sigma:
        One-sigma dispersion along one axis, any length unit.

    Returns
    -------
    float
        Probable error, same unit.
    """
    value = float(sigma)
    if not (np.isfinite(value) and value >= 0.0):
        raise ValueError(f"sigma must be finite and >= 0, got {value}")
    return PROBABLE_ERROR_OVER_SIGMA * value


def cep_from_probable_errors(range_probable: float, deflection_probable: float) -> float:
    """Siouris Eq. (5.17), :math:`\\mathrm{CEP} = 0.873(\\mathrm{REP} + \\mathrm{DEP})`.

    The classical approximation, provided for reading results against the
    weapon-delivery literature. **Prefer :func:`accuracy_statistics`**,
    which is exact for any aspect ratio at the cost of one quadrature.

    Accuracy, measured against that exact result rather than asserted: the
    source states the relation holds "even when REP and DEP differ by a
    factor as much as two", and at exactly that ratio the error is 1.5 %.
    The error is **not monotone** in the ratio — it peaks near 2 % around
    3:1, crosses zero near 5:1, then diverges (-5 % at 10:1). A caller who
    extrapolated past the stated bound by watching the error shrink would
    be walking into the divergence, so this function refuses beyond a ratio
    of 5 rather than returning the accidental near-zero.

    Parameters
    ----------
    range_probable, deflection_probable:
        REP and DEP, same length unit. Order does not matter.

    Returns
    -------
    float
        Approximate CEP, same unit.
    """
    rep, dep = float(range_probable), float(deflection_probable)
    for name, value in (("range_probable", rep), ("deflection_probable", dep)):
        if not (np.isfinite(value) and value >= 0.0):
            raise ValueError(f"{name} must be finite and >= 0, got {value}")
    large, small = max(rep, dep), min(rep, dep)
    if large == 0.0:
        raise ValueError("at least one probable error must be positive")
    if small > 0.0 and large / small > 5.0:
        raise ValueError(
            f"REP/DEP ratio {large / small:.2f} is beyond where Eq. (5.17) is "
            "usable; its error is non-monotone and diverges past about 5:1. "
            "Use accuracy_statistics, which is exact at any ratio."
        )
    if small == 0.0:
        raise ValueError(
            "a zero probable error is the degenerate one-dimensional case, "
            "which Eq. (5.17) does not cover; use accuracy_statistics."
        )
    return 0.873 * (rep + dep)


def cep_small_ratio(sigma_major: float, sigma_minor: float) -> float:
    """Siouris Eq. (5.13), the strongly elongated case.

    .. math::

        \\mathrm{CEP} = 0.9263\\,(\\sigma_S/\\sigma_L)^{2.09}\\,\\sigma_L
                      + 0.6745\\,\\sigma_L

    Stated for :math:`\\sigma_S/\\sigma_L < 0.28`, and measurement against
    the exact integral shows it is **better than advertised**: the error
    stays within 0.1 % throughout that range and remains within 0.1 % out
    to at least 0.35, only reaching 2.5 % at 0.5. The bound below is
    therefore the source's, not ours; it is kept because extending a fitted
    formula past its published range on the strength of our own spot checks
    is how a citation stops meaning anything.

    As with :func:`cep_from_probable_errors`, prefer
    :func:`accuracy_statistics` unless the point is to reproduce the
    classical route.

    Parameters
    ----------
    sigma_major, sigma_minor:
        Principal-axis sigmas, any length unit. Order does not matter.

    Returns
    -------
    float
        Approximate CEP, same unit.
    """
    large, small = _check_sigmas(sigma_major, sigma_minor)
    ratio = small / large
    if ratio >= 0.28:
        raise ValueError(
            f"sigma ratio {ratio:.3f} is outside the range Eq. (5.13) is "
            "published for (< 0.28). Use accuracy_statistics, or "
            "cep_from_probable_errors for the near-circular case."
        )
    return float((0.9263 * ratio**2.09 + PROBABLE_ERROR_OVER_SIGMA) * large)
