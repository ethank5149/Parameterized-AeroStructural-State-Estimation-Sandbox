"""Composed warning-time scenarios: fractional-orbital against ballistic.

The pieces to answer the question a fractional orbital profile exists to
raise have been in place for a while and were never joined up:
:mod:`passes.guidance.lofting` prices the ballistic alternatives,
:mod:`passes.orbital.fobs` builds the fractional profile,
:mod:`passes.orbital.warning` gives the horizon, and
:mod:`passes.orbital.radar` gives the sensors. This module composes them
into the single comparison the concept turns on: **how much warning does
each profile concede, to a real sensor network, between the same two
points?**

The two profiles, and why the comparison is fair
------------------------------------------------

Both trajectories start at the same launch site and end at the same
aimpoint. What differs is the path between:

* The **ballistic** arc takes the short great circle, range angle
  :math:`\\theta`, on a Keplerian conic whose apogee is set by the burnout
  flight-path angle. Minimum-energy, depressed and lofted variants are all
  available through :mod:`passes.guidance.lofting`, and they differ enormously
  in both apogee and flight time.
* The **fractional orbital** profile takes the *long* way, range angle
  :math:`2\\pi - \\theta`, at a low constant parking altitude, then deorbits
  onto the target. Going the long way is not a trick to make the numbers
  look better — it *is* the concept. It is what produces an approach from
  the opposite bearing, over the pole or the southern hemisphere, and it is
  why the profile costs more energy and takes longer while conceding less
  warning.

The trade is therefore not "FOBS is better". It is that a fractional profile
buys **approach geometry and horizon exposure** with **energy and time**,
and this module measures all four so the trade can be read rather than
asserted.

What is simplified, and how much it matters
--------------------------------------------

The parking arc is flown at constant altitude on a great circle, and the
Earth is not rotated underneath it. Both are wrong in ways that matter for a
real profile — :func:`passes.orbital.fobs.ground_track` exists precisely
because the planet turns about 22 degrees under one low revolution — and
both are deliberate here, because the warning comparison is dominated by
altitude and by which side of the planet the approach comes from, neither of
which rotation changes qualitatively. A ground track that included rotation
would move the sub-vehicle path east-west by up to a couple of thousand
kilometres, which would change *which* site detects first without much
changing *when*. That is worth doing and is not done here; it is recorded in
the backlog rather than hidden.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from passes.geodesy import (
    WGS84_MEAN_RADIUS,
    GeodeticPosition,
    great_circle_bearing,
    great_circle_range,
)
from passes.guidance.lofting import (
    burnout_speed_for_range,
    free_flight_time,
    optimum_burnout_angle,
)
from passes.orbital.fobs import EARTH_ROTATION_RATE
from passes.orbital.radar import EARLY_WARNING_SITES, CoverageResult, RadarSite, coverage

__all__ = [
    "Trajectory",
    "WarningComparison",
    "ballistic_trajectory",
    "fobs_trajectory",
    "great_circle_point",
    "leading_aimpoint",
    "warning_comparison",
]

_FloatArray = NDArray[np.float64]
_MU = 3.986004418e14


def great_circle_point(
    origin: GeodeticPosition,
    bearing: float,
    angular_distance: float,
) -> GeodeticPosition:
    """Walk ``angular_distance`` (rad) from ``origin`` along ``bearing``.

    The direct problem on a sphere. Used to lay a sub-vehicle track down a
    great circle; geodetic and geocentric latitude are treated as the same
    thing here, which is the spherical approximation this whole comparison
    rests on.
    """
    lat, lon = float(origin.latitude), float(origin.longitude)
    latitude = np.arcsin(
        np.sin(lat) * np.cos(angular_distance)
        + np.cos(lat) * np.sin(angular_distance) * np.cos(bearing)
    )
    longitude = lon + np.arctan2(
        np.sin(bearing) * np.sin(angular_distance) * np.cos(lat),
        np.cos(angular_distance) - np.sin(lat) * np.sin(latitude),
    )
    return GeodeticPosition(
        float(latitude), float((longitude + np.pi) % (2.0 * np.pi) - np.pi)
    )


def leading_aimpoint(
    launch: GeodeticPosition,
    target: GeodeticPosition,
    flight_time: Callable[[GeodeticPosition], float],
    tolerance: float = 1.0e-9,
    max_iterations: int = 40,
) -> tuple[GeodeticPosition, float]:
    """Where to aim so the target arrives under the impact point.

    A trajectory is flown in an inertial plane; the target rotates with the
    Earth. Over a flight of length :math:`t` the target moves east by
    :math:`\\omega t` — **16.9 degrees, about 1500 km, for a 67-minute
    fractional-orbital profile**, and still 7.5 degrees for a half-hour
    ballistic arc. Aiming at where the target *is* therefore misses by
    roughly that much, which is not a refinement but the difference between
    hitting a target and hitting a different continent.

    So the aim point is the target's position at *arrival*, and finding it
    is a fixed point: the lead depends on the flight time, which depends on
    the range to the lead point. This iterates it, which converges in a
    handful of steps because the map is a strong contraction —
    :math:`\\omega` times the sensitivity of flight time to range is well
    under one for any Earth-bound trajectory.

    Parameters
    ----------
    launch, target:
        Endpoints, with ``target`` the position at launch.
    flight_time:
        Given a candidate aim point, the seconds of flight to reach it.
    tolerance:
        Convergence tolerance on the lead angle (rad).
    max_iterations:
        Cap. Exceeding it raises rather than returning a half-converged
        aim point.

    Returns
    -------
    tuple[GeodeticPosition, float]
        The aim point in the inertial frame, and the flight time to it.
    """
    lead = 0.0
    for _ in range(max_iterations):
        aim = GeodeticPosition(
            target.latitude,
            float((target.longitude + lead + np.pi) % (2.0 * np.pi) - np.pi),
            target.altitude,
            target.label,
        )
        elapsed = float(flight_time(aim))
        updated = EARTH_ROTATION_RATE * elapsed
        if abs(updated - lead) < tolerance:
            return aim, elapsed
        lead = updated
    msg = (
        f"lead-angle iteration did not converge in {max_iterations} steps; "
        "the flight time is not a contraction of the aim point, which "
        "should not happen for an Earth-bound trajectory"
    )
    raise ValueError(msg)


@dataclass(frozen=True)
class Trajectory:
    """A sampled flight, in the form the coverage model consumes.

    Attributes
    ----------
    label:
        What this profile is.
    times:
        Seconds from launch, strictly increasing, ending at impact.
    altitudes:
        Altitude above the sphere (m).
    subpoints:
        Sub-vehicle ground points.
    range_angle:
        Total central angle flown (rad).
    burnout_speed:
        Speed required at the end of boost (m/s) — the energy price.
    """

    label: str
    times: _FloatArray
    altitudes: _FloatArray
    subpoints: list[GeodeticPosition]
    range_angle: float
    burnout_speed: float

    @property
    def flight_time(self) -> float:
        return float(self.times[-1])

    @property
    def apogee(self) -> float:
        return float(np.max(self.altitudes))


def ballistic_trajectory(
    launch: GeodeticPosition,
    target: GeodeticPosition,
    flight_path_angle: float | None = None,
    samples: int = 400,
    body_radius: float = WGS84_MEAN_RADIUS,
    earth_rotation: bool = True,
) -> Trajectory:
    """A Keplerian ballistic arc on the short great circle.

    Parameters
    ----------
    launch, target:
        Endpoints.
    flight_path_angle:
        Burnout angle above local horizontal (rad). Defaults to the
        minimum-energy :math:`\\gamma^*`. Pass a smaller value for a
        depressed profile or a larger one for a lofted one; the conjugate
        of any choice is available from
        :func:`passes.guidance.lofting.conjugate_flight_path_angle`.
    samples:
        Trajectory samples.
    """
    if earth_rotation:
        def elapsed_for(aim: GeodeticPosition) -> float:
            arc = great_circle_range(launch, aim) / body_radius
            angle = optimum_burnout_angle(arc) if flight_path_angle is None else float(
                flight_path_angle
            )
            return free_flight_time(arc, angle, _MU, body_radius)

        aimpoint, _ = leading_aimpoint(launch, target, elapsed_for)
    else:
        aimpoint = target
    theta = great_circle_range(launch, aimpoint) / body_radius
    if not 0.0 < theta < np.pi:
        msg = (
            "launch and target must be separated by less than 180 deg, got "
            f"{np.rad2deg(theta):.1f}"
        )
        raise ValueError(msg)
    gamma = optimum_burnout_angle(theta) if flight_path_angle is None else float(flight_path_angle)
    speed = burnout_speed_for_range(theta, gamma, _MU, body_radius)

    # Conic in the trajectory plane: sample true anomaly from burnout round
    # to impact, symmetric about apogee.
    momentum = body_radius * speed * np.cos(gamma)
    energy = 0.5 * speed**2 - _MU / body_radius
    semi_major = -_MU / (2.0 * energy)
    parameter = momentum**2 / _MU
    eccentricity = float(np.sqrt(max(1.0 - parameter / semi_major, 0.0)))
    nu0 = float(np.arccos(np.clip((parameter / body_radius - 1.0) / eccentricity, -1.0, 1.0)))
    nu = np.linspace(nu0, 2.0 * np.pi - nu0, samples)
    radius = parameter / (1.0 + eccentricity * np.cos(nu))

    # Time via Kepler, so the sampling is in true anomaly but the clock is right.
    eccentric = 2.0 * np.arctan2(
        np.sqrt(1.0 - eccentricity) * np.sin(0.5 * nu),
        np.sqrt(1.0 + eccentricity) * np.cos(0.5 * nu),
    )
    mean = np.unwrap(eccentric - eccentricity * np.sin(eccentric))
    times = (mean - mean[0]) / np.sqrt(_MU / semi_major**3)

    bearing = great_circle_bearing(launch, aimpoint)
    swept = np.linspace(0.0, theta, samples)
    inertial = [great_circle_point(launch, bearing, float(s)) for s in swept]
    if earth_rotation:
        subpoints = [
            GeodeticPosition(
                point.latitude,
                float(
                    (point.longitude - EARTH_ROTATION_RATE * float(t) + np.pi)
                    % (2.0 * np.pi)
                    - np.pi
                ),
                point.altitude,
                point.label,
            )
            for point, t in zip(inertial, times, strict=True)
        ]
    else:
        subpoints = inertial
    return Trajectory(
        label="ballistic",
        times=np.asarray(times, dtype=np.float64),
        # Clamped at zero: the conic returns to exactly the launch radius
        # at both ends, and round-off there shows up as a few microns of
        # negative altitude that the horizon model rightly refuses.
        altitudes=np.maximum(np.asarray(radius - body_radius, dtype=np.float64), 0.0),
        subpoints=subpoints,
        range_angle=float(theta),
        burnout_speed=float(speed),
    )


def fobs_trajectory(
    launch: GeodeticPosition,
    target: GeodeticPosition,
    parking_altitude: float = 150.0e3,
    entry_altitude: float = 100.0e3,
    samples: int = 400,
    body_radius: float = WGS84_MEAN_RADIUS,
    earth_rotation: bool = True,
) -> Trajectory:
    """A fractional orbital profile taking the long way round.

    The vehicle inserts into a low parking arc on the great circle through
    launch and target, flies the **major** arc :math:`2\\pi-\\theta` — which
    is what produces the reversed approach bearing — and then descends onto
    the target over the final portion.

    Parameters
    ----------
    parking_altitude:
        Constant altitude of the parking arc (m).
    entry_altitude:
        Altitude at which the descent is taken to reach the atmosphere; the
        last part of the profile drops from parking to this and then to the
        surface.
    earth_rotation:
        Whether to carry the sub-vehicle track in the **rotating** frame.
        A fractional profile spends the better part of an hour aloft, over
        which the planet turns roughly 15 degrees per hour — some 1600 km at
        the equator. The orbit plane is inertial; the ground beneath it is
        not, so the track walks west relative to a non-rotating calculation.

        This matters here for a specific reason: the whole warning
        comparison is about *which* sensors the track passes near, and a
        1600 km westward walk is comparable to a site's entire horizon
        radius at parking altitude. Leaving it out was the largest
        remaining approximation in the fractional profile.

        The rotation is applied to the ground track only; the trajectory
        itself is still flown on a fixed great circle in inertial space,
        which is correct to the extent that the plane does not precess
        appreciably inside one revolution.
    """
    if not (np.isfinite(parking_altitude) and parking_altitude > entry_altitude > 0.0):
        msg = (
            "need parking_altitude > entry_altitude > 0, got "
            f"{parking_altitude} and {entry_altitude}"
        )
        raise ValueError(msg)
    radius = body_radius + parking_altitude
    speed = float(np.sqrt(_MU / radius))

    def elapsed_for(aim: GeodeticPosition) -> float:
        arc = 2.0 * np.pi - great_circle_range(launch, aim) / body_radius
        return float(arc * radius / speed)

    if earth_rotation:
        aimpoint, _ = leading_aimpoint(launch, target, elapsed_for)
    else:
        aimpoint = target
    theta = 2.0 * np.pi - great_circle_range(launch, aimpoint) / body_radius

    # Away from the aim point: reversed bearing, so the approach arrives
    # from the opposite side. This is the defining geometry of the profile.
    bearing = float((great_circle_bearing(launch, aimpoint) + np.pi) % (2.0 * np.pi))
    swept = np.linspace(0.0, theta, samples)
    inertial = [great_circle_point(launch, bearing, float(s)) for s in swept]

    # Times first, because the rotation correction needs them.
    elapsed = swept * radius / speed

    if earth_rotation:
        # The plane is inertial; the ground turns eastward beneath it, so
        # the sub-vehicle longitude walks west by omega * t.
        subpoints = [
            GeodeticPosition(
                point.latitude,
                float(
                    (point.longitude - EARTH_ROTATION_RATE * float(t) + np.pi)
                    % (2.0 * np.pi)
                    - np.pi
                ),
                point.altitude,
                point.label,
            )
            for point, t in zip(inertial, elapsed, strict=True)
        ]
    else:
        subpoints = inertial

    # Altitude: constant on the parking arc, then a descent over the final
    # stretch. The descent arc is taken from the deorbit transfer geometry
    # of a Hohmann-like drop to the entry interface, which for these
    # altitudes is a few degrees of arc.
    descent_arc = float(np.arccos(np.clip(body_radius / radius, -1.0, 1.0)))
    altitudes = np.full(samples, parking_altitude, dtype=np.float64)
    descending = swept > (theta - descent_arc)
    if np.any(descending):
        fraction = (swept[descending] - (theta - descent_arc)) / descent_arc
        altitudes[descending] = np.maximum(parking_altitude * (1.0 - fraction), 0.0)

    return Trajectory(
        label="fractional-orbital",
        times=np.asarray(elapsed, dtype=np.float64),
        altitudes=altitudes,
        subpoints=subpoints,
        range_angle=float(theta),
        burnout_speed=speed,
    )


@dataclass(frozen=True)
class WarningComparison:
    """Both profiles, priced against the same sensor network."""

    ballistic: Trajectory
    fobs: Trajectory
    ballistic_coverage: CoverageResult
    fobs_coverage: CoverageResult

    @property
    def warning_reduction(self) -> float:
        """Seconds of warning the fractional profile removes.

        Positive when the fractional profile concedes less warning. Can be
        negative: a long way round is a long flight, and if the network
        picks it up early in the parking arc it may be tracked for longer
        in absolute terms than a short ballistic arc — which is exactly the
        kind of result the comparison exists to expose rather than assume.
        """
        return float(self.ballistic_coverage.warning_time - self.fobs_coverage.warning_time)

    @property
    def flight_time_penalty(self) -> float:
        """Extra seconds the fractional profile spends in flight."""
        return float(self.fobs.flight_time - self.ballistic.flight_time)


def warning_comparison(
    launch: GeodeticPosition,
    target: GeodeticPosition,
    flight_path_angle: float | None = None,
    parking_altitude: float = 150.0e3,
    sites: tuple[RadarSite, ...] = EARLY_WARNING_SITES,
    samples: int = 400,
    body_radius: float = WGS84_MEAN_RADIUS,
    earth_rotation: bool = True,
) -> WarningComparison:
    """Fly both profiles between the same points, past the same sensors."""
    ballistic = ballistic_trajectory(
        launch, target, flight_path_angle, samples, body_radius, earth_rotation
    )
    fobs = fobs_trajectory(
        launch,
        target,
        parking_altitude,
        samples=samples,
        body_radius=body_radius,
        earth_rotation=earth_rotation,
    )
    return WarningComparison(
        ballistic=ballistic,
        fobs=fobs,
        ballistic_coverage=coverage(
            ballistic.times, ballistic.altitudes, ballistic.subpoints, sites, body_radius
        ),
        fobs_coverage=coverage(
            fobs.times, fobs.altitudes, fobs.subpoints, sites, body_radius
        ),
    )
