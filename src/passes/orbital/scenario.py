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
from typing import Literal

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
from passes.orbital.radar import CoverageResult, RadarSite, coverage, network

__all__ = [
    "AscentProfile",
    "Event",
    "Phase",
    "Trajectory",
    "WarningComparison",
    "ascent_profile",
    "ballistic_trajectory",
    "fobs_trajectory",
    "great_circle_point",
    "leading_aimpoint",
    "max_boost_duration",
    "warning_comparison",
]

_FloatArray = NDArray[np.float64]
_MU = 3.986004418e14

#: Conventional entry interface (m). Not a physical boundary — the
#: atmosphere has no edge — but the altitude at which aerodynamic
#: deceleration starts to matter, and the one every entry text uses.
_ENTRY_INTERFACE = 100.0e3


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
class Phase:
    """A named stretch of a flight, with the clock it occupies.

    Carried on the trajectory rather than reconstructed downstream. A
    renderer that inferred "this must be the parking arc because the
    altitude stopped changing" would be guessing at something the producer
    knows exactly, and would guess wrong for a lofted ballistic arc whose
    altitude also flattens near apogee.
    """

    name: str
    start_time: float
    end_time: float
    note: str = ""

    @property
    def duration(self) -> float:
        return float(self.end_time - self.start_time)

    def contains(self, time: float) -> bool:
        return bool(self.start_time <= time <= self.end_time)


@dataclass(frozen=True)
class Event:
    """An instant worth marking: burnout, a burn, an interface crossing."""

    name: str
    time: float
    detail: str = ""


@dataclass(frozen=True)
class AscentProfile:
    """The boost leg, as a self-consistent gravity turn.

    Parameterised by the two quantities that are actually known — how long
    the vehicle burns, and how fast it has to be going at the end — with
    **downrange derived** rather than assumed. The three cannot be chosen
    independently: a vehicle accelerating to :math:`v_{bo}` over
    :math:`t_{bo}` covers a path length fixed by the speed law, and only
    the split of that path between up and downrange is free.

    That is what the previous stated ramp got wrong, and badly. It
    prescribed altitude against *arc* and time against the square root of
    arc, with downrange a free parameter; the shape it produced implied a
    burnout speed of **2,666 m/s** where the profile it fed needed 7,818 —
    a factor of three — and it left the pad at a flight-path angle of
    **37 degrees** instead of going up. Neither error was visible in any
    warning number, because warning depends on altitude and ground track
    and both of those were being *told* what to be.

    The model here is kinematic, not propulsive: speed grows as
    :math:`v_{bo}\\tau` and the flight-path angle pitches over as
    :math:`\\gamma_{bo} + (\\pi/2 - \\gamma_{bo})(1-\\tau)^{n}` from
    vertical at lift-off. Altitude and downrange are then integrals of it,
    so the profile is consistent with itself by construction. It carries no
    gravity loss, no drag loss and no staging, which is why the implied
    mean acceleration comes out around 4.4 g rather than the ~2 g of a real
    first stage — the model reproduces the *trajectory*, not the propulsion
    needed to fly it.

    Attributes
    ----------
    times, altitudes, downranges:
        Sampled boost leg (s, m, m), starting at zero.
    burnout_angle:
        Flight-path angle at burnout (rad). Falls out of the solve rather
        than being assumed, and near-zero is the correct answer for a
        circular insertion.
    burnout_speed:
        Speed at burnout (m/s) — the circular speed the parking arc needs.
    """

    times: _FloatArray
    altitudes: _FloatArray
    downranges: _FloatArray
    burnout_angle: float
    burnout_speed: float

    @property
    def ground_range(self) -> float:
        return float(self.downranges[-1])

    @property
    def duration(self) -> float:
        return float(self.times[-1])


def max_boost_duration(
    burnout_altitude: float,
    burnout_speed: float,
    pitch_exponent: float = 2.5,
    samples: int = 512,
) -> float:
    """Longest burn that still ends *ascending* at the requested altitude.

    A real coupling rather than a numerical nuisance: at a fixed burnout
    speed the path length grows with the burn, so a low parking orbit and a
    long burn are not compatible — the vehicle would have to be descending
    when the engines stop. Insert at 120 km and the ceiling is about 165 s;
    at 250 km it is well past any sensible burn.

    Exposed because a parking-altitude sweep needs it. The alternative is
    catching the exception :func:`ascent_profile` raises, which works but
    hides the trade the sweep exists to show.
    """
    tau = np.linspace(0.0, 1.0, int(samples))
    gamma = 0.5 * np.pi * (1.0 - tau) ** pitch_exponent
    integrand = tau * np.sin(gamma)
    climb = float(np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(tau)))
    return float(burnout_altitude / (burnout_speed * climb))


def ascent_profile(
    burnout_altitude: float,
    burnout_speed: float,
    duration: float,
    pitch_exponent: float = 2.5,
    samples: int = 512,
) -> AscentProfile:
    """Solve a gravity-turn boost that ends where the parking arc starts.

    Parameters
    ----------
    burnout_altitude:
        Altitude at the end of boost (m).
    burnout_speed:
        Speed at the end of boost (m/s).
    duration:
        Burn time (s).
    pitch_exponent:
        Shape of the pitch-over. Larger holds the vehicle vertical longer.
    samples:
        Resolution of the integration.

    Raises
    ------
    ValueError
        If no positive burnout angle reaches the requested altitude in the
        requested time. That is a real constraint rather than a numerical
        failure: at a fixed burnout speed a longer burn covers more path,
        so a *low* burnout altitude eventually demands the vehicle be
        descending before the engines stop. At 7.8 km/s and 150 km it binds
        at about 200 s.
    """
    for name, value in (
        ("burnout_altitude", burnout_altitude),
        ("burnout_speed", burnout_speed),
        ("duration", duration),
    ):
        if not (np.isfinite(value) and value > 0.0):
            msg = f"{name} must be finite and > 0, got {value}"
            raise ValueError(msg)
    if pitch_exponent <= 0.0:
        msg = f"pitch_exponent must be > 0, got {pitch_exponent}"
        raise ValueError(msg)

    tau = np.linspace(0.0, 1.0, int(samples))
    scale = float(burnout_speed) * float(duration)

    def climb(angle: float) -> _FloatArray:
        gamma = angle + (0.5 * np.pi - angle) * (1.0 - tau) ** pitch_exponent
        integrand = tau * np.sin(gamma)
        return np.asarray(
            scale * np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1])
                                                     * np.diff(tau))])
        )

    # Monotone in the burnout angle: a steeper finish climbs further. The
    # bracket starts at zero because a non-positive burnout angle means the
    # vehicle is level or descending at the end of boost, which is not an
    # ascent.
    reachable = float(climb(0.0)[-1])
    if reachable > burnout_altitude:
        msg = (
            f"a {duration:.0f} s burn to {burnout_speed:,.0f} m/s already climbs "
            f"{reachable / 1e3:,.0f} km before its flight-path angle reaches zero, "
            f"so it cannot burn out at {burnout_altitude / 1e3:,.0f} km still "
            "ascending; shorten the burn or raise the parking altitude"
        )
        raise ValueError(msg)

    low, high = 0.0, 0.5 * np.pi - 1e-9
    for _ in range(200):
        mid = 0.5 * (low + high)
        if climb(mid)[-1] < burnout_altitude:
            low = mid
        else:
            high = mid
    angle = 0.5 * (low + high)

    gamma = angle + (0.5 * np.pi - angle) * (1.0 - tau) ** pitch_exponent
    downrange = tau * np.cos(gamma)
    ground = scale * np.concatenate(
        [[0.0], np.cumsum(0.5 * (downrange[1:] + downrange[:-1]) * np.diff(tau))]
    )
    return AscentProfile(
        times=tau * float(duration),
        altitudes=climb(angle),
        downranges=np.asarray(ground),
        burnout_angle=float(angle),
        burnout_speed=float(burnout_speed),
    )


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
    phases:
        Named stretches of the flight, contiguous and covering it. Empty
        when the producer does not distinguish any.
    events:
        Instants worth marking, in time order.
    """

    label: str
    times: _FloatArray
    altitudes: _FloatArray
    subpoints: list[GeodeticPosition]
    range_angle: float
    burnout_speed: float
    phases: tuple[Phase, ...] = ()
    events: tuple[Event, ...] = ()

    @property
    def flight_time(self) -> float:
        return float(self.times[-1])

    @property
    def apogee(self) -> float:
        return float(np.max(self.altitudes))

    def phase_at(self, time: float) -> str:
        """Name of the phase covering ``time``, or ``""`` if none is declared."""
        for phase in self.phases:
            if phase.contains(time):
                return phase.name
        return ""


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
    # Clamped at zero: the conic returns to exactly the launch radius at
    # both ends, and round-off there shows up as a few microns of negative
    # altitude that the horizon model rightly refuses.
    altitudes = np.maximum(np.asarray(radius - body_radius, dtype=np.float64), 0.0)

    # Phases, so a renderer names them rather than inferring them. A lofted
    # arc's altitude also flattens near apogee, so "the altitude stopped
    # changing" is not a usable test for anything.
    apogee_index = int(np.argmax(altitudes))
    apogee_time = float(times[apogee_index])
    entry_radius = body_radius + _ENTRY_INTERFACE
    entry_anomaly = 2.0 * np.pi - float(
        np.arccos(np.clip((parameter / entry_radius - 1.0) / eccentricity, -1.0, 1.0))
    )
    entry_angle = float(
        np.arctan2(
            (_MU / momentum) * eccentricity * np.sin(entry_anomaly),
            momentum / entry_radius,
        )
    )
    entry_mask = np.zeros(times.size, dtype=bool)
    entry_mask[apogee_index:] = altitudes[apogee_index:] <= _ENTRY_INTERFACE
    entry_hits = np.nonzero(entry_mask)[0]
    entry_time = float(times[entry_hits[0]]) if entry_hits.size else float(times[-1])
    impact_time = float(times[-1])
    phases = (
        Phase("ascent", 0.0, apogee_time, "burnout to apogee"),
        Phase("descent", apogee_time, entry_time, "apogee to the entry interface"),
        Phase("entry", entry_time, impact_time,
              f"below the {_ENTRY_INTERFACE / 1e3:.0f} km interface"),
    )
    events = (
        Event("burnout", 0.0,
              f"{speed:,.0f} m/s, gamma {np.rad2deg(gamma):.1f} deg"),
        Event("apogee", apogee_time, f"{altitudes[apogee_index] / 1e3:,.0f} km"),
        Event("entry interface", entry_time,
              f"{_ENTRY_INTERFACE / 1e3:.0f} km, gamma {np.rad2deg(entry_angle):.1f} deg"),
        Event("impact", impact_time, f"{aimpoint.label or 'aimpoint'}"),
    )
    return Trajectory(
        label="ballistic",
        times=np.asarray(times, dtype=np.float64),
        altitudes=altitudes,
        subpoints=subpoints,
        range_angle=float(theta),
        burnout_speed=float(speed),
        phases=phases,
        events=events,
    )


def fobs_trajectory(
    launch: GeodeticPosition,
    target: GeodeticPosition,
    parking_altitude: float = 170.0e3,
    parking_apogee: float | None = None,
    entry_altitude: float = 100.0e3,
    samples: int = 400,
    body_radius: float = WGS84_MEAN_RADIUS,
    earth_rotation: bool = True,
    perigee_radius: float | None = None,
    boost_duration: float | None = 180.0,
    direction: Literal["long", "short"] = "long",
) -> Trajectory:
    """A fractional orbital profile taking the long way round.

    The vehicle boosts on a gravity turn into a low elliptical parking
    orbit on the great circle through launch and target, coasts the
    **major** arc :math:`2\\pi-\\theta` — which is what produces the
    reversed approach bearing — and then fires retrograde onto a descent
    conic that reaches the target.

    Every leg is a Keplerian coast or an integrated ascent, and the times
    come from Kepler's equation throughout. Nothing is prescribed except
    the shape of the boost.

    Parameters
    ----------
    parking_altitude:
        **Insertion (perigee) altitude** of the parking orbit (m).

        The default of 170 km sits in the 150-180 km band open sources
        quote for the R-36O, and it is a *modelling choice* rather than a
        measurement — which is why `notebooks/fobs-warning-analysis.ipynb`
        sweeps it from 120 to 500 km against four mask assumptions instead
        of resting on one number. Lower is better for horizon denial and
        worse for drag; the sweep is the honest form of the answer.
    parking_apogee:
        Apogee altitude of the parking orbit (m). Defaults to
        ``parking_altitude + 80 km``.

        A real insertion is never exactly circular, and the previous
        version of this function pretended otherwise: it flew a *constant*
        altitude, which is not an orbit but a prescription, and it made the
        vehicle's altitude the one quantity in the profile that no dynamics
        produced. On an ellipse the altitude varies over the arc — 170 km
        at insertion, 250 km half a revolution later — and the ground rate
        varies with it, which is the behaviour that makes a parking coast
        read as a coast.

        Pass ``parking_altitude`` here to recover a circular orbit.
    direction:
        Which way round the great circle to fly.

        ``"long"`` is the fractional orbital concept: the major arc
        :math:`2\\pi-\\theta`, arriving on the reversed bearing.

        ``"short"`` flies the same low parking altitude down the **minor**
        arc :math:`\\theta`, on the direct bearing, and exists as a control.
        The fractional concept bundles two separate claims — that flying
        *low* denies horizon, and that arriving from the *opposite* bearing
        denies azimuth coverage — and quoting one warning number for the
        pair makes it impossible to say which is doing the work. A direct
        low profile isolates the first: same altitude, same sensor network,
        same aimpoint, ordinary approach geometry. Whatever warning it
        concedes is what altitude alone buys, and the difference from the
        long way is what the reversed approach buys.

        It is also the right comparison against a depressed ballistic arc,
        which is the other way of trading energy for a low, fast flight down
        the same minor arc. Note it is **not** a cheap option: a direct
        profile still pays full orbital insertion, so it costs the same
        burnout speed as the long way while giving up the reversed bearing.
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
    apogee_altitude = (
        parking_altitude + 80.0e3 if parking_apogee is None else float(parking_apogee)
    )
    if apogee_altitude < parking_altitude - 1e-9:
        msg = (
            f"parking_apogee must be at or above the insertion altitude, got "
            f"{apogee_altitude} below {parking_altitude}; insertion is at perigee"
        )
        raise ValueError(msg)

    # --- the parking conic. Insertion is at perigee (nu = 0), so the true
    # anomaly along the arc equals the central angle swept since insertion.
    park_perigee = body_radius + parking_altitude
    park_apogee = body_radius + apogee_altitude
    park_sma = 0.5 * (park_perigee + park_apogee)
    park_ecc = (park_apogee - park_perigee) / (park_apogee + park_perigee)
    park_parameter = park_sma * (1.0 - park_ecc**2)
    park_momentum = float(np.sqrt(_MU * park_parameter))
    speed = float(np.sqrt(_MU * (2.0 / park_perigee - 1.0 / park_sma)))
    park_period = 2.0 * np.pi * float(np.sqrt(park_sma**3 / _MU))

    perigee = body_radius - 400.0e3 if perigee_radius is None else float(perigee_radius)
    if not perigee < body_radius:
        msg = (
            f"perigee_radius must lie below the surface for the conic to reach "
            f"it, got {perigee} against a body radius of {body_radius}"
        )
        raise ValueError(msg)

    if boost_duration is None:
        # A stated policy, not a silent repair: take the longest burn the
        # requested parking altitude can accept, with 10 % of margin, and
        # cap it at the nominal 180 s. Used by sweeps over parking
        # altitude, where a fixed burn is infeasible at the low end.
        boost_duration = min(
            180.0, 0.9 * max_boost_duration(parking_altitude, speed)
        )
    if not (np.isfinite(boost_duration) and boost_duration > 0.0):
        msg = f"boost_duration must be finite and > 0, got {boost_duration}"
        raise ValueError(msg)
    ascent = ascent_profile(parking_altitude, speed, float(boost_duration))
    boost_arc = ascent.ground_range / body_radius

    if direction not in ("long", "short"):
        msg = f"direction must be 'long' or 'short', got {direction!r}"
        raise ValueError(msg)
    long_way = direction == "long"

    def _arc(aim: GeodeticPosition) -> float:
        minor = great_circle_range(launch, aim) / body_radius
        return float(2.0 * np.pi - minor if long_way else minor)

    def _park_time(anomaly: _FloatArray | float) -> _FloatArray:
        """Seconds from insertion (perigee) to a parking true anomaly."""
        nu = np.asarray(anomaly, dtype=np.float64)
        eccentric = 2.0 * np.arctan2(
            np.sqrt(1.0 - park_ecc) * np.sin(0.5 * nu),
            np.sqrt(1.0 + park_ecc) * np.cos(0.5 * nu),
        )
        mean = np.unwrap(np.atleast_1d(eccentric - park_ecc * np.sin(eccentric)))
        # Unwrapping keeps the clock monotone past nu = pi, where the
        # eccentric anomaly's principal branch folds back.
        return np.asarray(np.where(nu < 0.0, 0.0, mean * park_period / (2.0 * np.pi)))

    def _descent_geometry(burn_radius: float) -> tuple[float, float, float, float, float]:
        """Descent conic from a burn at ``burn_radius``: e, a, p, arc, time."""
        ecc = (burn_radius - perigee) / (burn_radius + perigee)
        sma = 0.5 * (burn_radius + perigee)
        param = sma * (1.0 - ecc**2)
        cos_impact = np.clip((param / body_radius - 1.0) / ecc, -1.0, 1.0)
        impact_nu = 2.0 * np.pi - float(np.arccos(cos_impact))
        arc = impact_nu - np.pi  # the burn point is the conic's apogee
        eccentric = 2.0 * np.arctan2(
            np.sqrt(1.0 - ecc) * np.sin(0.5 * impact_nu),
            np.sqrt(1.0 + ecc) * np.cos(0.5 * impact_nu),
        )
        mean = float(eccentric - ecc * np.sin(eccentric))
        # Apogee sits at M = pi; the impact branch is past it, and the
        # half-angle form returns the folded branch, so unfold by 2*pi.
        if mean < np.pi:
            mean += 2.0 * np.pi
        span = (mean - np.pi) / float(np.sqrt(_MU / sma**3))
        return ecc, sma, param, arc, span

    def _deorbit(arc: float) -> tuple[float, float, float, float, float, float]:
        """Burn true anomaly and descent conic for a total arc.

        A fixed point, because the burn radius depends on where on the
        parking ellipse the burn falls and that depends on how much arc the
        descent needs. The coupling is weak — the ellipse's radius varies by
        the apogee-perigee difference over an arc the descent spans in tens
        of degrees — so this converges to machine precision in a few steps.
        """
        anomaly = arc - boost_arc
        ecc = sma = param = descent = span = 0.0
        for _ in range(60):
            radius = float(park_parameter / (1.0 + park_ecc * np.cos(anomaly)))
            ecc, sma, param, descent, span = _descent_geometry(radius)
            updated = arc - descent - boost_arc
            if abs(updated - anomaly) < 1e-13:
                anomaly = updated
                break
            anomaly = updated
        return anomaly, ecc, sma, param, descent, span

    def elapsed_for(aim: GeodeticPosition) -> float:
        """Total flight time to a candidate aim point: all three legs.

        Pricing only the parking ellipse — as if the vehicle coasted the
        whole arc at orbital altitude — leaves the lead angle wrong by the
        difference between the parking and descent clocks over the descent
        arc. That is small, and it is not nothing: 1.3 km of miss on the
        direct profile, from a routine that claims to converge to 1e-9 rad.
        """
        arc = _arc(aim)
        anomaly, _, _, _, _, descent_span = _deorbit(arc)
        coast = float(_park_time(np.array([anomaly]))[0])
        return float(boost_duration + coast + descent_span)

    if earth_rotation:
        aimpoint, _ = leading_aimpoint(launch, target, elapsed_for)
    else:
        aimpoint = target
    theta = _arc(aimpoint)

    # Away from the aim point on the long way: reversed bearing, so the
    # approach arrives from the opposite side. That is the defining geometry
    # of the fractional profile, and taking the minor arc instead is exactly
    # what the "short" control gives up.
    heading = float(great_circle_bearing(launch, aimpoint))
    bearing = float((heading + np.pi) % (2.0 * np.pi)) if long_way else heading
    swept = np.linspace(0.0, theta, samples)
    inertial = [great_circle_point(launch, bearing, float(s)) for s in swept]

    # --- the deorbit conic, and where the burn has to happen.
    #
    # Coupled: the burn radius comes from where on the parking ellipse the
    # burn falls, and where it falls depends on how much arc the descent
    # conic needs, which depends on the burn radius. The coupling is weak
    # (the parking ellipse's radius varies by 80 km over an arc the descent
    # spans in tens of degrees), so a few fixed-point steps converge to
    # microradians. Iterating is still better than assuming, because the
    # assumption would silently mis-place the impact point.
    burn_anomaly, eccentricity, semi_major, parameter, descent_arc, _ = _deorbit(theta)
    burn_radius = float(park_parameter / (1.0 + park_ecc * np.cos(burn_anomaly)))

    # On the long way there are 4-5 radians to spend and this never binds.
    # On the minor arc it can: the deorbit conic needs some 60 degrees, the
    # boost a few more, and a short-range shot simply has nowhere to put a
    # parking phase. Refusing is right — the alternative is a profile that
    # is descending before it has finished ascending.
    if theta <= boost_arc + descent_arc:
        msg = (
            f"a {direction}-way fractional profile over {np.rad2deg(theta):.1f} deg "
            f"of arc cannot contain a {np.rad2deg(boost_arc):.1f} deg boost and a "
            f"{np.rad2deg(descent_arc):.1f} deg deorbit conic; raise perigee_radius, "
            "shorten the boost, or use a ballistic profile at this range"
        )
        raise ValueError(msg)

    # Three timing rules spliced: the integrated ascent, Kepler on the
    # parking ellipse, Kepler on the descent ellipse.
    ascending = swept < boost_arc
    descending = swept > (theta - descent_arc)
    coasting = ~ascending & ~descending

    elapsed = np.zeros(samples, dtype=np.float64)
    if np.any(ascending):
        # Invert the ascent's monotone downrange to get time at each sampled
        # arc, so the boost leg is sampled on the same grid as everything
        # else without re-solving it.
        elapsed[ascending] = np.interp(
            swept[ascending] * body_radius, ascent.downranges, ascent.times
        )
    elapsed[coasting] = boost_duration + _park_time(swept[coasting] - boost_arc)
    burn_time = float(boost_duration + _park_time(np.array([burn_anomaly]))[0])

    # --- altitudes, one rule per leg
    altitudes = np.empty(samples, dtype=np.float64)
    if np.any(ascending):
        altitudes[ascending] = np.interp(
            swept[ascending] * body_radius, ascent.downranges, ascent.altitudes
        )
    # The parking leg is a conic, not a prescription: altitude follows
    # r(nu) = p / (1 + e cos nu) from insertion at perigee, so it climbs to
    # apogee half a revolution later and comes back down. Holding it exactly
    # constant made the vehicle's altitude the one quantity in the profile
    # that no dynamics produced.
    altitudes[coasting] = (
        park_parameter / (1.0 + park_ecc * np.cos(swept[coasting] - boost_arc))
        - body_radius
    )
    if np.any(descending):
        anomaly = np.pi + (swept[descending] - (theta - descent_arc))
        altitudes[descending] = np.maximum(
            parameter / (1.0 + eccentricity * np.cos(anomaly)) - body_radius, 0.0
        )

        # Re-time the descent leg by Kepler: the vehicle speeds up as it
        # falls, so carrying the parking rate would stretch the descent.
        def _mean_anomaly(nu: _FloatArray) -> _FloatArray:
            eccentric = 2.0 * np.arctan2(
                np.sqrt(1.0 - eccentricity) * np.sin(0.5 * nu),
                np.sqrt(1.0 + eccentricity) * np.cos(0.5 * nu),
            )
            return np.asarray(np.unwrap(eccentric - eccentricity * np.sin(eccentric)))

        # Anchored at the burn itself (nu = pi), not at the first sample
        # after it. Re-zeroing on `offsets[0]` discards up to one sample
        # interval of descent, and because the ground track is then rotated
        # by that clock the whole profile lands short: 1.9 km of miss at 200
        # samples, 0.9 km at 900, 0.3 km at 4000 — a discretisation error
        # masquerading as a modelling one, visible only because it scaled
        # with the sample count.
        mean_motion = np.sqrt(_MU / semi_major**3)
        offsets = _mean_anomaly(anomaly) - float(_mean_anomaly(np.array([np.pi]))[0])
        elapsed[descending] = burn_time + offsets / mean_motion

    # The rotation correction comes *after* every leg has been timed. Doing
    # it earlier is not a style point: the descent samples then carry a
    # clock of zero, get no westward walk at all, and the profile lands
    # 1,550 km east of its aimpoint while every non-rotating test still
    # passes. Found exactly that way.
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

    # --- the deorbit burn, as an actual delta-v rather than a discontinuity
    #
    # At the burn point the parking ellipse has both a radial and a
    # transverse velocity component; the descent conic has apogee there, so
    # its radial component is zero. The burn is therefore not purely
    # retrograde, and taking only the transverse difference would understate
    # it. Both components are differenced.
    park_radial = (_MU / park_momentum) * park_ecc * float(np.sin(burn_anomaly))
    park_transverse = park_momentum / burn_radius
    descent_transverse = float(np.sqrt(_MU * parameter)) / burn_radius
    delta_v = float(np.hypot(park_radial, park_transverse - descent_transverse))

    entry_index = np.nonzero(descending & (altitudes <= entry_altitude))[0]
    entry_time = float(elapsed[entry_index[0]]) if entry_index.size else float(elapsed[-1])

    # Entry flight-path angle, which is where the concept pays a price it is
    # rarely charged for. Deorbiting from orbital speed cannot produce a
    # steep entry without removing kilometres per second, so a fractional
    # profile arrives an order of magnitude shallower than a ballistic RV —
    # and the descent conic therefore spans thousands of kilometres of
    # ground track, committing the vehicle long before impact.
    entry_radius = body_radius + entry_altitude
    entry_anomaly = 2.0 * np.pi - float(
        np.arccos(np.clip((parameter / entry_radius - 1.0) / eccentricity, -1.0, 1.0))
    )
    entry_momentum = float(np.sqrt(_MU * parameter))
    entry_angle = float(
        np.arctan2(
            (_MU / entry_momentum) * eccentricity * np.sin(entry_anomaly),
            entry_momentum / entry_radius,
        )
    )
    entry_speed = float(np.sqrt(_MU * (2.0 / entry_radius - 1.0 / semi_major)))
    insertion_time = float(boost_duration)
    impact_time = float(elapsed[-1])

    descent_ground = descent_arc * body_radius
    phases = (
        Phase("boost", 0.0, insertion_time,
              f"gravity turn to {parking_altitude / 1e3:.0f} km, "
              f"burnout gamma {np.rad2deg(ascent.burnout_angle):.1f} deg"),
        Phase("parking coast", insertion_time, burn_time,
              f"{parking_altitude / 1e3:.0f} x {apogee_altitude / 1e3:.0f} km, "
              f"e = {park_ecc:.4f}"),
        Phase("deorbit coast", burn_time, entry_time,
              f"{descent_ground / 1e3:,.0f} km of ground track to the interface"),
        Phase("entry", entry_time, impact_time,
              f"below the {entry_altitude / 1e3:.0f} km interface"),
    )
    events = (
        Event("lift-off", 0.0, f"{launch.label or 'launch site'}"),
        Event("insertion", insertion_time,
              f"{parking_altitude / 1e3:.0f} km, {speed:,.0f} m/s, "
              f"gamma {np.rad2deg(ascent.burnout_angle):.1f} deg"),
        Event("deorbit burn", burn_time,
              f"retrograde dv {delta_v:,.0f} m/s at {(burn_radius - body_radius) / 1e3:.0f} km"),
        Event("entry interface", entry_time,
              f"{entry_altitude / 1e3:.0f} km, {entry_speed / 1e3:.2f} km/s, "
              f"gamma {np.rad2deg(entry_angle):.1f} deg"),
        Event("impact", impact_time, f"{aimpoint.label or 'aimpoint'}"),
    )

    return Trajectory(
        label="fractional-orbital" if long_way else "fractional-orbital (direct)",
        times=np.asarray(elapsed, dtype=np.float64),
        altitudes=altitudes,
        subpoints=subpoints,
        range_angle=float(theta),
        burnout_speed=speed,
        phases=phases,
        events=events,
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
    sites: tuple[RadarSite, ...] | None = None,
    samples: int = 400,
    body_radius: float = WGS84_MEAN_RADIUS,
    earth_rotation: bool = True,
) -> WarningComparison:
    """Fly both profiles between the same points, past the same sensors.

    Parameters
    ----------
    sites:
        Sensor network. Defaults to ``network("western")`` — the US/NATO
        and allied sensors — rather than the whole of
        :data:`~passes.orbital.radar.EARLY_WARNING_SITES`, and the
        distinction is not cosmetic. The catalogue also holds Russian
        early-warning radars and one non-aligned site, and
        :func:`~passes.orbital.radar.coverage` reduces a network to its
        *earliest* detection: passing the full catalogue for a Eurasian
        launch reports a Russian radar seeing its own missile a minute
        after lift-off, which is warning to nobody. A non-aligned sensor's
        returns are not on the western picture either.

        Pass an explicit tuple to model a different defender.
    """
    network_sites = network("western") if sites is None else tuple(sites)
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
            ballistic.times, ballistic.altitudes, ballistic.subpoints,
            network_sites, body_radius,
        ),
        fobs_coverage=coverage(
            fobs.times, fobs.altitudes, fobs.subpoints, network_sites, body_radius
        ),
    )
