"""Propagation of burnout errors into impact errors.

Implements the ballistic error coefficients of

* G. M. Siouris, *Missile Guidance and Control Systems* (Springer, 2004),
  §6.4.3, Eq. (6.116) and Figs. 6.14-6.17.

This module exists to replace stated numbers with derived ones.
:data:`passes.systems.budget.DISPERSION_SOURCES` says of itself that its
entries "are parametric inputs, not derived results" — order-of-magnitude
figures chosen to exercise the accounting. Ballistic error coefficients are
the derivation: given a perturbation at thrust termination, they say what
it becomes at impact, in closed form.

The assumptions are Siouris's, and they are the free-flight ones: inverse
square gravity, no atmosphere, and a fixed time of flight. Everything here
is therefore about the *exo-atmospheric* mapping from burnout to entry
interface. The atmospheric leg has its own treatment in
:mod:`passes.flight.ballistic_entry`.

The result worth knowing before any of the algebra
--------------------------------------------------

A lateral displacement of the burnout point does **not** produce a
proportional crossrange miss. Siouris Eq. (6.116a) gives the exact spherical
relation

.. math:: \\cos \\delta C = \\sin^2\\psi + \\cos^2\\psi \\, \\cos \\delta\\chi

with :math:`\\psi` the free-flight range angle, reducing for small angles to

.. math:: \\delta C \\approx \\delta\\chi \\, |\\cos \\psi|.

So the sensitivity is :math:`\\cos\\psi`, and it **vanishes at a range angle
of 90 degrees** — a quarter of the Earth's circumference, about 10,000 km.
At that range a lateral burnout offset produces *no crossrange miss at all*,
exactly, for any offset size. The reason is geometric rather than
approximate: two great circles displaced perpendicular to each other at one
point converge again a quarter turn later, because every pair of great
circles intersects. Past 90 degrees the sensitivity grows again with the
opposite sign.

This is the sort of thing a stated dispersion number cannot express. A
crossrange budget quoted as a fixed metre count is implicitly assuming a
range, and is wrong by an unbounded factor at the wrong one.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "crossrange_from_lateral_offset",
    "crossrange_offset_sensitivity",
    "launch_position_error",
    "velocity_error_at_impact",
]

_FloatArray = NDArray[np.float64]


def crossrange_from_lateral_offset(
    offset_angle: ArrayLike,
    range_angle: float,
) -> _FloatArray:
    """Crossrange miss (rad) from a lateral burnout displacement.

    Siouris Eq. (6.116a), exact rather than the small-angle form:

    .. math:: \\cos \\delta C = \\sin^2\\psi + \\cos^2\\psi \\, \\cos \\delta\\chi

    Evaluated through an equivalent half-angle form rather than as printed:

    .. math:: \\delta C = 2\\arcsin\\!\\big(|\\cos\\psi|\\,\\sin(\\delta\\chi/2)\\big)

    The two are algebraically identical — subtract each side of the first
    from one and apply :math:`1-\\cos x = 2\\sin^2(x/2)` — but the printed
    form is **numerically useless for small offsets**, which is the case
    that matters. A crossrange budget cares about metre-scale offsets, i.e.
    :math:`\\delta\\chi \\sim 10^{-7}` rad, where :math:`\\cos\\delta\\chi`
    differs from 1 by :math:`5\\times10^{-15}` — a few times machine epsilon
    — and :math:`\\arccos` near its endpoint has unbounded derivative. Coded
    literally it loses about four significant figures there. The half-angle
    form never forms a quantity near 1 and is exact throughout; it also
    makes the :math:`|\\cos\\psi|` sensitivity manifest rather than
    emergent.

    Verified against direct spherical construction — displace the burnout
    point perpendicular to the trajectory plane, fly the same range angle on
    the same initial heading, and measure the great-circle separation at
    impact — to machine precision across range angles 0 to 150 degrees and
    offsets from 0.01 to 5 degrees.

    Parameters
    ----------
    offset_angle:
        Lateral displacement of the burnout point (rad, as an angle
        subtended at the Earth's centre). Scalar or array. Multiply a linear
        offset by ``1/R_earth`` to get it.
    range_angle:
        Free-flight range angle :math:`\\psi` (rad), burnout to impact.

    Returns
    -------
    numpy.ndarray
        Crossrange miss (rad), non-negative. Multiply by the Earth radius
        for a distance.

    Notes
    -----
    Returned unsigned because the exact relation is even in
    :math:`\\delta\\chi` — the miss is the same magnitude whichever side the
    burnout point is displaced to. The sign is carried by the caller if it
    matters.
    """
    chi = np.atleast_1d(np.asarray(offset_angle, dtype=np.float64))
    if not np.all(np.isfinite(chi)):
        msg = "offset_angle must be finite"
        raise ValueError(msg)
    psi = float(range_angle)
    if not (np.isfinite(psi) and 0.0 <= psi < np.pi):
        msg = f"range_angle must lie in [0, pi), got {psi}"
        raise ValueError(msg)
    half = abs(np.cos(psi)) * np.sin(0.5 * chi)
    return np.asarray(2.0 * np.arcsin(np.clip(half, -1.0, 1.0)))


def crossrange_offset_sensitivity(range_angle: float) -> float:
    """:math:`|\\cos\\psi|`, the small-angle crossrange sensitivity.

    The derivative of :func:`crossrange_from_lateral_offset` at zero offset.
    Reported separately because it is the number a dispersion budget wants —
    "a metre of lateral burnout error becomes this many metres at impact" —
    and because its **zero at 90 degrees** is the single most surprising
    entry in the whole error budget.

    Parameters
    ----------
    range_angle:
        Free-flight range angle (rad).

    Returns
    -------
    float
        Dimensionless sensitivity in :math:`[0, 1]`.
    """
    psi = float(range_angle)
    if not (np.isfinite(psi) and 0.0 <= psi < np.pi):
        msg = f"range_angle must lie in [0, pi), got {psi}"
        raise ValueError(msg)
    return float(abs(np.cos(psi)))


def velocity_error_at_impact(
    velocity_error: float,
    time_of_flight: float,
) -> float:
    """Impact displacement (m) from a burnout velocity error, ``dV * t_ff``.

    Siouris Fig. 6.16: a lateral velocity error at thrust termination simply
    integrates over the free-flight time, because nothing acts to correct it
    and gravity is (to first order) parallel over the displacement.

    This is the same structure the mission budget already used for its
    crossrange entry, where the velocity error was multiplied by a transfer
    sensitivity of 850 s obtained from our own propagator. That agreement is
    worth stating: an independent text gives the same first-order law, so
    the budget's crossrange mapping was right in form even while its inputs
    were stated rather than derived.

    Parameters
    ----------
    velocity_error:
        One-sigma velocity error at burnout (m/s).
    time_of_flight:
        Free-flight time (s).

    Returns
    -------
    float
        Impact displacement (m).
    """
    dv, tof = float(velocity_error), float(time_of_flight)
    for name, value in (("velocity_error", dv), ("time_of_flight", tof)):
        if not (np.isfinite(value) and value >= 0.0):
            msg = f"{name} must be finite and >= 0, got {value}"
            raise ValueError(msg)
    return dv * tof


def launch_position_error(
    latitude_error: float,
    longitude_error: float,
    latitude: float,
    bearing: float,
    earth_radius: float = 6378137.0,
) -> tuple[float, float]:
    """Resolve a launch-site survey error into downrange and crossrange (m).

    Siouris Fig. 6.17. A latitude error displaces the launch point north by
    :math:`R_e\\,\\delta L`; a longitude error displaces it east by
    :math:`R_e\\cos L\\,\\delta\\lambda`. Both then project onto the
    trajectory's bearing:

    .. math::

        \\delta \\mathrm{DR} &= N \\cos B + E \\sin B \\\\
        \\delta \\mathrm{CR} &= -N \\sin B + E \\cos B

    This is the term that makes a 10 m CEP a *survey* problem rather than a
    guidance one: the error enters the impact point undiminished, with no
    range-dependent suppression of the kind :func:`crossrange_offset_sensitivity`
    provides for a lateral burnout offset, because it is a displacement of
    the whole trajectory rather than of one point on it.

    Parameters
    ----------
    latitude_error, longitude_error:
        Survey errors (rad).
    latitude:
        Launch-site geodetic latitude (rad), which sets the longitude
        error's linear size through :math:`\\cos L`. At high latitude a
        given longitude error matters less; at the pole, not at all.
    bearing:
        Launch azimuth (rad from north, positive east).
    earth_radius:
        Sphere radius (m).

    Returns
    -------
    tuple[float, float]
        ``(downrange, crossrange)`` displacement (m), signed. Crossrange is
        positive to the right of the bearing.

    Notes
    -----
    The latitude terms reproduce Siouris Fig. 6.17 exactly. His longitude
    *crossrange* term carries the opposite sign to the rotation above; the
    archived scan does not settle whether that is a different crossrange
    convention (positive-left) or a transcription artefact, and rather than
    guess, this function implements the rotation, which is derivable and
    self-consistent. A caller comparing against the figure should check the
    sign convention rather than assume it.
    """
    for name, value in (
        ("latitude_error", latitude_error),
        ("longitude_error", longitude_error),
        ("latitude", latitude),
        ("bearing", bearing),
    ):
        if not np.isfinite(value):
            msg = f"{name} must be finite, got {value}"
            raise ValueError(msg)
    if not (np.isfinite(earth_radius) and earth_radius > 0.0):
        msg = f"earth_radius must be finite and > 0, got {earth_radius}"
        raise ValueError(msg)
    if abs(latitude) > 0.5 * np.pi:
        msg = f"latitude must lie in [-pi/2, pi/2], got {latitude}"
        raise ValueError(msg)

    north = earth_radius * float(latitude_error)
    east = earth_radius * np.cos(float(latitude)) * float(longitude_error)
    downrange = north * np.cos(bearing) + east * np.sin(bearing)
    crossrange = -north * np.sin(bearing) + east * np.cos(bearing)
    return float(downrange), float(crossrange)
