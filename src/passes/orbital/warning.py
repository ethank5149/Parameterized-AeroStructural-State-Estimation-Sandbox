"""Radar horizon geometry and warning time — what FOBS is actually for.

:mod:`passes.orbital.fobs` already carries the mechanics of a fractional
orbital profile: the ground track, the deorbit transfer, and the approach
azimuth that a full orbital insertion makes free. What it did not carry is
the *reason* to fly one. A fractional orbital profile is not chosen because
it is cheap — it costs more energy than a minimum-energy ballistic arc to
the same target, and takes longer. It is chosen because of what a ground
radar can see, and when.

This module supplies that half: the exact spherical geometry of when a
vehicle rises above a site's horizon mask, and therefore how much warning a
given trajectory concedes.

The geometry, which is exact
----------------------------

A site on a sphere of radius :math:`R_E` sees a vehicle at radius :math:`r`
at elevation :math:`\\varepsilon` above its local horizontal when the
central angle :math:`\\lambda` between them satisfies

.. math:: \\tan\\varepsilon = \\frac{\\cos\\lambda - R_E/r}{\\sin\\lambda}.

Setting :math:`\\varepsilon` to the site's mask and solving gives the
visibility radius in closed form,

.. math:: \\lambda_{\\max} = \\arccos\\!\\big[(R_E/r)\\cos\\varepsilon\\big] - \\varepsilon,

which reduces to the familiar :math:`\\arccos(R_E/r)` at zero mask. No
iteration is needed, and the derivation is a one-line rearrangement:
:math:`\\cos\\lambda - \\tan\\varepsilon\\sin\\lambda
= \\cos(\\lambda+\\varepsilon)/\\cos\\varepsilon`.

Why altitude dominates everything else
--------------------------------------

The visibility radius depends on the vehicle's radius and nothing else — not
on its speed, its size, or the direction it came from. And it grows fast:
:math:`\\lambda_{\\max}` goes as roughly :math:`\\sqrt{h}` for small
:math:`h`, so an apogee ten times higher is seen from about three times as
far away.

That is the whole argument. A minimum-energy intercontinental ballistic arc
has an apogee near 1300 km and is visible to a zero-mask site out to about
34 degrees of central angle — some 3800 km. A fractional orbital profile at
a 150 km parking altitude is visible only within about 12 degrees, roughly
1400 km. The low profile is not stealthy in any electromagnetic sense; it is
simply below the horizon for most of its flight.

The second half of the argument is direction, and this module does not
supply it: a fractional orbital insertion can arrive from any azimuth,
including over a pole, which is the province of
:func:`passes.orbital.fobs.approach_azimuth`. Warning time is short *and*
comes from a bearing the defence may not be looking along. This module
prices only the first of those.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from passes.geodesy import WGS84_MEAN_RADIUS

__all__ = [
    "DetectionWindow",
    "detection_window",
    "horizon_central_angle",
    "visibility_radius",
]

_FloatArray = NDArray[np.float64]


def horizon_central_angle(
    altitude: ArrayLike,
    mask_elevation: float = 0.0,
    body_radius: float = WGS84_MEAN_RADIUS,
) -> _FloatArray:
    """Greatest central angle (rad) at which a vehicle is still visible.

    .. math::

        \\lambda_{\\max} = \\arccos\\!\\big[(R_E/r)\\cos\\varepsilon\\big]
                          - \\varepsilon

    Parameters
    ----------
    altitude:
        Vehicle altitude above the sphere (m), scalar or array.
    mask_elevation:
        Minimum elevation the site can work at (rad). Real radars are
        masked by terrain, by their own mounting, and by refraction and
        clutter near the horizon; 0 is the geometric limit and optimistic
        for the defender. Values of 3-5 degrees are typical.
    body_radius:
        Sphere radius (m).

    Returns
    -------
    numpy.ndarray
        Central angle (rad), zero where the vehicle is below the mask even
        directly overhead — which cannot happen for positive altitude, but
        is clamped rather than returned negative for a mask above 90
        degrees' worth of geometry.

    Notes
    -----
    Spherical, not WGS-84. The oblateness correction to a horizon radius is
    of order the flattening, 1/298, which is far inside the uncertainty in
    any real mask angle; using the mean radius and saying so is more honest
    than an ellipsoidal calculation with an invented mask.
    """
    epsilon = float(mask_elevation)
    if not (np.isfinite(epsilon) and -0.5 * np.pi < epsilon < 0.5 * np.pi):
        msg = f"mask_elevation must lie in (-pi/2, pi/2), got {epsilon}"
        raise ValueError(msg)
    if not (np.isfinite(body_radius) and body_radius > 0.0):
        msg = f"body_radius must be finite and > 0, got {body_radius}"
        raise ValueError(msg)
    h = np.asarray(altitude, dtype=np.float64)
    if np.any(h < 0.0):
        msg = "altitude must be non-negative"
        raise ValueError(msg)
    ratio = body_radius / (body_radius + h) * np.cos(epsilon)
    return np.asarray(np.maximum(np.arccos(np.clip(ratio, -1.0, 1.0)) - epsilon, 0.0))


def visibility_radius(
    altitude: ArrayLike,
    mask_elevation: float = 0.0,
    body_radius: float = WGS84_MEAN_RADIUS,
) -> _FloatArray:
    """Ground-range radius (m) of the region that can see the vehicle.

    :func:`horizon_central_angle` times the body radius — the surface
    distance from the sub-vehicle point out to the horizon circle.
    """
    return np.asarray(
        body_radius * horizon_central_angle(altitude, mask_elevation, body_radius)
    )


@dataclass(frozen=True)
class DetectionWindow:
    """When a site first sees a trajectory, and how long is then left.

    Attributes
    ----------
    detected:
        Whether the trajectory ever rises above the mask for this site.
    first_detection_time:
        Time of first visibility (s, on the trajectory's own clock);
        ``nan`` when never detected.
    warning_time:
        Seconds from first detection to the last sample — the time the
        defence has, if the last sample is impact. ``nan`` when never
        detected.
    first_detection_altitude:
        Vehicle altitude at first detection (m); ``nan`` when never
        detected.
    visible_fraction:
        Fraction of the sampled trajectory spent above the mask. Useful
        separately from the warning time: a profile can be detected early
        and then lost behind the horizon again, which a single warning
        number hides.
    last_detection_time:
        Time of the **last** sample above the mask (s); ``nan`` when never
        detected. With ``first_detection_time`` this bounds the interval
        the site contributes anything over, which ``visible_fraction``
        summarises but does not locate. Note the pair spans any gaps: a
        trajectory that sets and rises again is bracketed, not split, and
        ``visible_fraction`` is what reveals the difference.
    """

    detected: bool
    first_detection_time: float
    warning_time: float
    first_detection_altitude: float
    visible_fraction: float
    last_detection_time: float = float("nan")


def detection_window(
    times: ArrayLike,
    altitudes: ArrayLike,
    central_angles: ArrayLike,
    mask_elevation: float = 0.0,
    body_radius: float = WGS84_MEAN_RADIUS,
) -> DetectionWindow:
    """Reduce a sampled trajectory to what one radar site learns from it.

    Parameters
    ----------
    times:
        Sample times (s), increasing.
    altitudes:
        Vehicle altitude at each sample (m).
    central_angles:
        Central angle from the site to the sub-vehicle point at each sample
        (rad). Compute with :func:`passes.geodesy.great_circle_range`
        divided by the body radius, or from a ground track.
    mask_elevation, body_radius:
        As for :func:`horizon_central_angle`.

    Returns
    -------
    DetectionWindow

    Notes
    -----
    Detection is treated as purely geometric: above the mask is detected,
    below it is not. A real radar also needs enough power-aperture at the
    slant range and enough radar cross-section to close the link, and a
    real engagement needs track quality rather than first return. This is
    therefore an **upper bound on warning time** — the defence cannot do
    better than geometry, and will usually do worse. Stated because a
    warning-time figure invites being read as a prediction.
    """
    t = np.asarray(times, dtype=np.float64)
    h = np.asarray(altitudes, dtype=np.float64)
    lam = np.asarray(central_angles, dtype=np.float64)
    if not (t.shape == h.shape == lam.shape) or t.ndim != 1 or t.size < 2:
        msg = "times, altitudes and central_angles must be 1-D arrays of equal length >= 2"
        raise ValueError(msg)
    if np.any(np.diff(t) <= 0.0):
        msg = "times must be strictly increasing"
        raise ValueError(msg)

    visible = lam <= horizon_central_angle(h, mask_elevation, body_radius)
    fraction = float(np.count_nonzero(visible) / visible.size)
    if not np.any(visible):
        return DetectionWindow(False, float("nan"), float("nan"), float("nan"), 0.0)
    seen = np.flatnonzero(visible)
    first, last = int(seen[0]), int(seen[-1])
    return DetectionWindow(
        detected=True,
        first_detection_time=float(t[first]),
        warning_time=float(t[-1] - t[first]),
        first_detection_altitude=float(h[first]),
        visible_fraction=fraction,
        last_detection_time=float(t[last]),
    )
