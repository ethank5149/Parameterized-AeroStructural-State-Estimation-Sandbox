"""Fractional orbital trajectories: ground track, deorbit, entry interface.

A fractional orbital profile inserts into a low parking orbit, coasts
through part of one revolution, and deorbits onto an entry interface from
which the vehicle glides. Paper II's title claims this regime and §7 builds
the coast; what this module adds is the three things that only appear once
the trajectory is *fractional* rather than a single ballistic arc.

**The Earth turns underneath.** Every other module in this package works in
an inertial frame, where that does not matter. A ground track does: over
one low-Earth revolution of about 88 minutes the planet rotates roughly
22 degrees, so the point on the ground beneath the vehicle is not the point
an inertial calculation puts it. :func:`ground_track` does the rotation
explicitly, and :func:`ground_track_shift` gives the per-revolution
westward walk that follows from it.

**The approach azimuth is free.** This is the property that distinguishes
the profile, and it is a statement about geometry rather than about
performance. A direct ballistic arc between two points lies in the plane
containing them and the centre of the Earth, so its approach azimuth at
the target is fixed by the endpoints. A fractional orbit chooses its own
plane — any inclination at or above the target latitude will do — and
therefore arrives on an azimuth set by *where in the orbit the deorbit
happens*, not by where it started. :func:`approach_azimuth` computes it,
and :func:`azimuth_envelope` sweeps the deorbit point to show the range of
azimuths a given orbit admits.

**Range is bought in three currencies.** Total ground range is the arc
flown in the parking orbit, plus the arc of the deorbit transfer, plus the
glide. They trade against each other: deorbiting earlier spends less
orbital arc and demands more glide, and the glide is the expensive one
because it is the only leg that costs energy the vehicle has to carry.
:func:`fobs_profile` closes that accounting.

Scope
-----

Coplanar deorbit from a circular parking orbit, which is the only sensible
choice: a plane change at orbital speed costs kilometres per second, while
the entire deorbit burn costs tens of metres per second. The transfer to
entry interface is Keplerian, since it lasts minutes and spends almost all
of that above the sensible atmosphere. Earth is spherical here, matching
:mod:`passes.guidance.entry`; the oblateness that
:mod:`passes.orbital.gravity` carries matters for the parking orbit's
secular drift over many revolutions and not for a fractional one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from passes.orbital.gravity import EARTH, GravityModel

__all__ = [
    "EARTH_ROTATION_RATE",
    "DeorbitBurn",
    "FobsProfile",
    "approach_azimuth",
    "azimuth_envelope",
    "deorbit_burn",
    "fobs_profile",
    "ground_track",
    "ground_track_shift",
]

_FloatArray = NDArray[np.float64]

#: Earth's sidereal rotation rate (rad/s).
EARTH_ROTATION_RATE = 7.292115e-5


def ground_track(
    position: ArrayLike,
    time: ArrayLike,
    gmst_epoch: float = 0.0,
    rotation_rate: float = EARTH_ROTATION_RATE,
) -> tuple[_FloatArray, _FloatArray]:
    """Earth-fixed longitude and geocentric latitude (rad) beneath a state.

    Parameters
    ----------
    position:
        Inertial position, shape ``(3,)`` or ``(3, n)`` (m).
    time:
        Seconds since ``gmst_epoch``. Scalar or length ``n``.
    gmst_epoch:
        Greenwich sidereal angle (rad) at :math:`t = 0`.

    Notes
    -----
    Latitude is **geocentric**, not geodetic. On a spherical Earth the two
    coincide; on the real one they differ by up to about 0.19 degrees at
    mid-latitudes, which is 21 km on the ground. That is small against a
    glide but not against a terminal error budget, and it is a conversion
    to apply at the point where an ellipsoid is introduced rather than a
    correction to smuggle in here.
    """
    r = np.atleast_2d(np.asarray(position, dtype=np.float64).T).T
    if r.shape[0] != 3:
        raise ValueError(f"position must have leading dimension 3, got {r.shape}")
    t = np.atleast_1d(np.asarray(time, dtype=np.float64))
    if t.size != 1 and t.size != r.shape[1]:
        raise ValueError(
            f"time must be scalar or match the number of states: got {t.size} "
            f"times for {r.shape[1]} states"
        )
    radius = np.linalg.norm(r, axis=0)
    if np.any(radius == 0.0):
        raise ValueError("position must be non-zero to define a ground track")
    theta = gmst_epoch + rotation_rate * t
    longitude = np.arctan2(r[1], r[0]) - theta
    longitude = (longitude + np.pi) % (2.0 * np.pi) - np.pi
    latitude = np.arcsin(np.clip(r[2] / radius, -1.0, 1.0))
    return np.asarray(longitude), np.asarray(latitude)


def ground_track_shift(period: float, rotation_rate: float = EARTH_ROTATION_RATE) -> float:
    """Westward longitude walk (rad) per revolution.

    The signature property of a low orbit's ground track, and the reason a
    fractional profile can be timed to place its entry interface almost
    anywhere: the track repeats only after many revolutions, so waiting
    changes where the orbit passes.
    """
    if not (np.isfinite(period) and period > 0.0):
        raise ValueError(f"period must be finite and > 0, got {period}")
    return float(rotation_rate * period)


@dataclass(frozen=True)
class DeorbitBurn:
    """A coplanar retrograde burn from a circular parking orbit.

    Attributes
    ----------
    delta_v:
        Burn magnitude (m/s). Retrograde, applied at the parking radius,
        which becomes the transfer apogee.
    transfer_semi_major_axis:
        Semi-major axis (m) of the descent ellipse.
    perigee_radius:
        Transfer perigee (m). Chosen below the entry interface so the
        trajectory actually crosses it rather than grazing.
    transfer_angle:
        True anomaly swept from the burn to the entry interface (rad).
    transfer_time:
        Elapsed time (s) over that arc.
    entry_speed:
        Inertial speed (m/s) at the entry interface.
    entry_flight_path_angle:
        Flight-path angle (rad) there, negative descending. This is the
        number that decides whether the entry is survivable and whether the
        glide can be established: too steep and the vehicle cannot pull out
        before the peak load, too shallow and it skips.
    """

    delta_v: float
    transfer_semi_major_axis: float
    perigee_radius: float
    transfer_angle: float
    transfer_time: float
    entry_speed: float
    entry_flight_path_angle: float


def deorbit_burn(
    parking_radius: float,
    entry_radius: float,
    perigee_radius: float,
    model: GravityModel = EARTH,
) -> DeorbitBurn:
    """Solve the coplanar deorbit from a circular parking orbit.

    The burn is applied retrograde at ``parking_radius``, which therefore
    becomes the apogee of a descent ellipse with the requested perigee. The
    vehicle then coasts to ``entry_radius``.

    Parameters
    ----------
    parking_radius, entry_radius, perigee_radius:
        Geocentric radii (m), ordered ``perigee < entry < parking``. The
        perigee must sit *below* the entry interface: an ellipse whose
        perigee lies above it never crosses, and the request is then not
        an expensive deorbit but an impossible one.

    Notes
    -----
    Deorbit is cheap and that is the point of the profile. From a 200 km
    circular orbit, dropping perigee to 50 km costs about 70 m/s against an
    orbital speed near 7.8 km/s — under one percent. What the burn buys is
    not energy but *timing*: it converts a choice of when to fire into a
    choice of where the entry interface falls.
    """
    for name, value in (
        ("parking_radius", parking_radius),
        ("entry_radius", entry_radius),
        ("perigee_radius", perigee_radius),
    ):
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be finite and > 0, got {value}")
    if not perigee_radius < entry_radius < parking_radius:
        raise ValueError(
            f"require perigee < entry < parking radius, got "
            f"{perigee_radius:.6g} < {entry_radius:.6g} < {parking_radius:.6g}; "
            f"an ellipse whose perigee lies above the entry interface never "
            f"crosses it"
        )

    mu = model.mu
    sma = 0.5 * (parking_radius + perigee_radius)
    eccentricity = (parking_radius - perigee_radius) / (parking_radius + perigee_radius)
    speed_circular = np.sqrt(mu / parking_radius)
    speed_apogee = np.sqrt(mu * (2.0 / parking_radius - 1.0 / sma))
    delta_v = float(speed_circular - speed_apogee)

    # True anomaly at the entry interface, measured from perigee. The burn
    # happens at apogee (nu = pi), so the swept angle is the difference.
    semi_latus = sma * (1.0 - eccentricity**2)
    cos_nu = (semi_latus / entry_radius - 1.0) / eccentricity
    nu_entry = float(np.arccos(np.clip(cos_nu, -1.0, 1.0)))
    swept = float(np.pi - nu_entry)

    # Kepler's equation on the descending half.
    def mean_anomaly(nu: float) -> float:
        eccentric = 2.0 * np.arctan2(
            np.sqrt(1.0 - eccentricity) * np.sin(0.5 * nu),
            np.sqrt(1.0 + eccentricity) * np.cos(0.5 * nu),
        )
        return float(eccentric - eccentricity * np.sin(eccentric))

    period_factor = np.sqrt(sma**3 / mu)
    transfer_time = float(period_factor * (mean_anomaly(np.pi) - mean_anomaly(nu_entry)))

    speed_entry = float(np.sqrt(mu * (2.0 / entry_radius - 1.0 / sma)))
    momentum = np.sqrt(mu * semi_latus)
    # Descending, so the flight-path angle is negative.
    cos_gamma = np.clip(momentum / (entry_radius * speed_entry), -1.0, 1.0)
    gamma = -float(np.arccos(cos_gamma))

    return DeorbitBurn(
        delta_v=delta_v,
        transfer_semi_major_axis=float(sma),
        perigee_radius=float(perigee_radius),
        transfer_angle=swept,
        transfer_time=transfer_time,
        entry_speed=speed_entry,
        entry_flight_path_angle=gamma,
    )


def approach_azimuth(latitude: float, inclination: float, ascending: bool = True) -> float:
    """Heading (rad from north, positive east) of an orbit at a latitude.

    From the spherical-triangle relation :math:`\\cos i = \\sin A \\cos
    \\phi`, so the azimuth an orbit crosses a given latitude on is fixed by
    the inclination alone — not by where the vehicle started. That is the
    geometric content of "the approach azimuth is free": choosing the
    orbit plane chooses the arrival heading, and the descending pass gives
    the supplementary azimuth to the ascending one.

    Returns
    -------
    float
        A **signed** heading, not a compass bearing in :math:`[0, 2\\pi)`.
        The ascending branch is an :func:`~numpy.arcsin` and therefore lies
        in :math:`[-\\pi/2, \\pi/2]`; the descending branch is
        :math:`\\pi - A` and lies in :math:`[\\pi/2, 3\\pi/2]`. A negative
        value means west of north, which is what a retrograde orbit
        genuinely does — a sun-synchronous orbit at :math:`i = 98°` crosses
        the equator ascending at about :math:`-8°`, i.e. very slightly west
        of due north. Add :math:`2\\pi` if a bearing is wanted; this is
        stated because a caller expecting :math:`[0, 360)` would otherwise
        read the sign as an error rather than as information.

    Raises
    ------
    ValueError
        If ``|latitude|`` exceeds the inclination reachable, which is a
        real geometric limit rather than a numerical one: an orbit never
        reaches latitudes above its inclination.
    """
    lat = float(latitude)
    inc = float(inclination)
    if not (np.isfinite(lat) and np.isfinite(inc)):
        raise ValueError("latitude and inclination must be finite")
    reachable = min(inc, np.pi - inc)
    if abs(lat) > reachable + 1e-12:
        raise ValueError(
            f"an orbit of inclination {np.rad2deg(inc):.3f} deg never reaches "
            f"latitude {np.rad2deg(lat):.3f} deg"
        )
    sin_azimuth = np.clip(np.cos(inc) / max(np.cos(lat), 1e-15), -1.0, 1.0)
    azimuth = float(np.arcsin(sin_azimuth))
    return azimuth if ascending else float(np.pi - azimuth)


def azimuth_envelope(latitude: float, inclinations: ArrayLike) -> _FloatArray:
    """Approach azimuths (rad) available at a latitude, over inclinations.

    Returns ``nan`` for inclinations that cannot reach the latitude, rather
    than dropping them, so the array stays aligned with its input and the
    unreachable region is visible instead of silently absent.
    """
    lat = float(latitude)
    out = []
    for inc in np.atleast_1d(np.asarray(inclinations, dtype=np.float64)):
        try:
            out.append(approach_azimuth(lat, float(inc)))
        except ValueError:
            out.append(np.nan)
    return np.asarray(out)


@dataclass(frozen=True)
class FobsProfile:
    """Range accounting across the three legs of a fractional profile.

    Attributes
    ----------
    burn:
        The deorbit solution.
    parking_arc:
        Central angle (rad) flown in the parking orbit before the burn.
    transfer_arc:
        Central angle (rad) from burn to entry interface.
    glide_arc:
        Central angle (rad) the glide must cover to reach the target.
    total_arc:
        Their sum, which must equal the angle from insertion to target.
    """

    burn: DeorbitBurn
    parking_arc: float
    transfer_arc: float
    glide_arc: float
    total_arc: float

    def ranges(self, radius: float = EARTH.radius) -> tuple[float, float, float]:
        """Surface ranges (m) of the three legs, at the given Earth radius."""
        return (
            radius * self.parking_arc,
            radius * self.transfer_arc,
            radius * self.glide_arc,
        )


def fobs_profile(
    total_arc: float,
    glide_arc: float,
    parking_radius: float,
    entry_radius: float,
    perigee_radius: float,
    model: GravityModel = EARTH,
) -> FobsProfile:
    """Close the range accounting for a fractional orbital profile.

    Given how far the target is from insertion and how much of that the
    glide will cover, the parking arc is whatever is left over once the
    deorbit transfer has taken its share. A negative remainder means the
    profile does not close — the glide and transfer already overshoot — and
    is rejected rather than returned as a negative range.
    """
    burn = deorbit_burn(parking_radius, entry_radius, perigee_radius, model=model)
    total = float(total_arc)
    glide = float(glide_arc)
    if not (np.isfinite(total) and total > 0.0):
        raise ValueError(f"total_arc must be finite and > 0, got {total}")
    if not (np.isfinite(glide) and glide >= 0.0):
        raise ValueError(f"glide_arc must be finite and >= 0, got {glide}")
    parking = total - glide - burn.transfer_angle
    if parking < 0.0:
        raise ValueError(
            f"the profile does not close: the glide ({np.rad2deg(glide):.2f} "
            f"deg) and deorbit transfer "
            f"({np.rad2deg(burn.transfer_angle):.2f} deg) already exceed the "
            f"{np.rad2deg(total):.2f} deg to the target, leaving no parking "
            f"arc. Deorbit later, or command a shorter glide"
        )
    return FobsProfile(
        burn=burn,
        parking_arc=float(parking),
        transfer_arc=burn.transfer_angle,
        glide_arc=glide,
        total_arc=total,
    )
