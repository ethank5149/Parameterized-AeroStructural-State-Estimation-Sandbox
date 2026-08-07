"""Complete missions, targeted with the integrator rather than planned.

:mod:`passes.orbital.scenario` answers "what must the vehicle do" in closed
form, cheaply, on a two-body sphere with no drag, no J2, no gravity loss and
no mass depletion. That is a **planner**, and it is what the animations used
to be drawn from. This module flies the thing.

The distinction is not academic. Between the planner and the integrator sit
every loss the planner does not model, and they are large: a launch that the
planner says needs 7.8 km/s of burnout speed needs some 9.3 km/s of delivered
:math:`\\Delta v` to achieve it. Drawing a picture from the planner and calling
it a simulation would be drawing the answer to a different question.

How the targeting works
-----------------------

Every solve below is a **shooting** solve: the residual is evaluated by
integrating the real coupled system, so the answer satisfies the real
dynamics rather than the planner's.

* **Boost.** Two knobs — the gravity-turn kick angle and the burn duration —
  against two targets, the osculating perigee and apogee at burnout. Bounded
  least squares, because the feasible region has a hard edge: pitch over too
  far and the vehicle turns into the atmosphere, dynamic pressure passes
  400 kPa and drag exceeds thrust.
* **Deorbit.** One knob, the retrograde burn duration, against one target,
  the osculating perigee. Monotone, so a bracketed root.
* **Where to fire it.** One knob, the parking coast duration, against the
  along-track miss at impact. This is what makes the flown profile arrive at
  the aim point instead of near it.
* **Which plane to fly in.** The aim point moves under the orbit while the
  vehicle is in flight, so the plane that contains launch and target *at
  arrival* depends on the arrival time, which depends on the plane. A fixed
  point, iterated a few times around whole flown missions.

What is still approximate, stated plainly
-----------------------------------------

The vehicle is a single lumped stage with constant thrust and constant
exhaust velocity — no staging, no throttling, no fairing jettison. Its
attitude is integrated but torque-free and does not feed the force model, so
the thrust direction is a *steering command*, not the pointing of a body
whose control authority was checked. The atmosphere is the simulator's
exponential one. None of that is hidden by the animation, because the
animation shows the states the integrator produced under exactly these
assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.optimize
from numpy.typing import NDArray

from passes.flight.mission import MissionResult, MissionSegment, fly_mission
from passes.flight.propulsion import Burn
from passes.flight.simulator import FlightSimulator
from passes.geodesy import GeodeticPosition, great_circle_range
from passes.orbital.fobs import EARTH_ROTATION_RATE

__all__ = [
    "FlownProfile",
    "LaunchVehicle",
    "fly_fractional_orbital",
    "osculating_apsides",
]

_FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LaunchVehicle:
    r"""A lumped launcher: one stage, constant thrust, constant exhaust speed.

    Defaults describe an ICBM-class vehicle: 400 t off the pad at a
    thrust-to-weight of 1.38, exhausting at 3,400 m/s. That gives a max-Q
    near 25 kPa against a real 30, and a mass ratio with enough ideal
    :math:`\Delta v` to reach orbit *after* the roughly 1,900 m/s of gravity
    and drag loss the integrator charges — which a closed-form planner does
    not know about and cannot size against.

    Three thrust levels, because a launch is not one burn. The main engine
    lifts off and pitches over; a lower-thrust upper stage circularises at
    apogee; a small motor deorbits. Sizing them the same would put 5.4 MN
    on an 18 t vehicle, which is 30 g.
    """

    liftoff_mass: float = 400.0e3
    thrust: float = 5.4e6
    exhaust_velocity: float = 3400.0
    insertion_thrust: float = 1.2e6
    insertion_exhaust_velocity: float = 3400.0
    deorbit_thrust: float = 60.0e3
    deorbit_exhaust_velocity: float = 2900.0
    drag_area: float = 10.0
    nose_radius: float = 1.2

    @property
    def thrust_to_weight(self) -> float:
        return float(self.thrust / (self.liftoff_mass * 9.80665))


@dataclass(frozen=True)
class FlownProfile:
    """A mission that was integrated, with the residuals it converged to."""

    mission: MissionResult
    label: str
    aimpoint: GeodeticPosition
    miss_distance: float
    """Great-circle distance from impact to the aim point (m)."""
    ideal_delta_v: float
    achieved_delta_v: float
    solver_iterations: int
    notes: dict[str, float] = field(default_factory=dict)

    @property
    def gravity_and_drag_loss(self) -> float:
        """Delivered minus achieved :math:`\\Delta v` on the boost (m/s).

        The number the planner cannot produce. Everything a closed-form
        two-body launch model omits shows up here.
        """
        return float(self.ideal_delta_v - self.achieved_delta_v)


def osculating_apsides(
    position: _FloatArray, velocity: _FloatArray, mu: float, body_radius: float
) -> tuple[float, float]:
    """Perigee and apogee **altitude** of the instantaneous two-body conic.

    The targeting quantity. Burnout altitude and speed alone do not say what
    orbit the vehicle is in; the apsides do, and they are what a parking
    orbit is specified by.
    """
    r = np.asarray(position, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    radius = float(np.linalg.norm(r))
    energy = 0.5 * float(v @ v) - mu / radius
    if energy >= 0.0:
        return float("inf"), float("inf")
    semi_major = -mu / (2.0 * energy)
    momentum = np.cross(r, v)
    parameter = float(momentum @ momentum) / mu
    eccentricity = float(np.sqrt(max(1.0 - parameter / semi_major, 0.0)))
    return (
        float(semi_major * (1.0 - eccentricity) - body_radius),
        float(semi_major * (1.0 + eccentricity) - body_radius),
    )


def _cartesian(position: GeodeticPosition, radius: float) -> _FloatArray:
    lat, lon = float(position.latitude), float(position.longitude)
    return radius * np.array(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )


def _subpoint(state: _FloatArray, elapsed: float, layout: object) -> GeodeticPosition:
    """Sub-vehicle ground point in the rotating frame."""
    r = state[layout.position]  # type: ignore[attr-defined]
    radius = float(np.linalg.norm(r))
    latitude = float(np.arcsin(np.clip(r[2] / radius, -1.0, 1.0)))
    longitude = float(np.arctan2(r[1], r[0]) - EARTH_ROTATION_RATE * elapsed)
    return GeodeticPosition(latitude, float((longitude + np.pi) % (2.0 * np.pi) - np.pi))


def _plane_normal(launch: GeodeticPosition, aim: GeodeticPosition, long_way: bool) -> _FloatArray:
    """Unit normal of the plane through both points, oriented by direction."""
    a = _cartesian(launch, 1.0)
    b = _cartesian(aim, 1.0)
    normal = np.cross(a, b)
    span = float(np.linalg.norm(normal))
    if span < 1e-9:
        msg = "launch and aim point are coincident or antipodal; no plane is defined"
        raise ValueError(msg)
    normal = normal / span
    # cross(normal, r_hat) is the downrange direction; flipping the normal
    # flips it, which is exactly the long-way/short-way choice.
    return np.asarray(-normal if long_way else normal)


def _apsides_of(simulator: FlightSimulator, state: _FloatArray) -> tuple[float, float]:
    return osculating_apsides(
        state[simulator.layout.position], state[simulator.layout.velocity],
        simulator.gravity.mu, simulator.gravity.radius,
    )


def _run(
    simulator: FlightSimulator,
    state: _FloatArray,
    duration: float,
    burn: Burn | None,
    label: str,
    samples: int = 6,
) -> _FloatArray:
    """Fly one leg and return the final state. The solvers' inner loop."""
    flown = fly_mission(
        simulator, state, [MissionSegment(duration, burn, label)],
        samples_per_segment=samples,
    )
    return np.asarray(flown.result.states[:, -1])


def _bisect(
    f: object, low: float, high: float, target: float, tolerance: float, steps: int = 40
) -> float:
    """Bisection on a monotone shooting residual.

    Bisection rather than a secant or Brent: every evaluation is a full
    integration, some parameter values make the vehicle turn into the
    atmosphere and fail outright, and bisection is the only bracketed method
    that never leaves the bracket. Slower per digit and it always finishes.
    """
    f_low = f(low)  # type: ignore[operator]
    f_high = f(high)  # type: ignore[operator]
    if (f_low - target) * (f_high - target) > 0.0:
        msg = (
            f"no solution between {low:g} and {high:g}: the residual runs from "
            f"{f_low:g} to {f_high:g} and never crosses {target:g}"
        )
        raise ValueError(msg)
    for _ in range(steps):
        mid = 0.5 * (low + high)
        value = f(mid)  # type: ignore[operator]
        if abs(value - target) < tolerance:
            return float(mid)
        if (f_low - target) * (value - target) <= 0.0:
            high = mid
        else:
            low, f_low = mid, value
    return float(0.5 * (low + high))


def _solve_boost(
    simulator: FlightSimulator,
    pad_state: _FloatArray,
    normal: _FloatArray,
    vehicle: LaunchVehicle,
    duration: float,
    target_apogee: float,
) -> tuple[Burn, _FloatArray]:
    """Solve the gravity-turn kick that puts apogee at the parking apogee.

    One knob against one target, and monotone: a shallower kick keeps the
    vehicle vertical longer and throws apogee higher. The burn duration is
    *not* also solved here — a launcher's first stage burns to depletion,
    and the circularisation burn is what trims the orbit.
    """
    def apogee_for(kick: float) -> float:
        burn = Burn(
            duration=duration, thrust=vehicle.thrust,
            exhaust_velocity=vehicle.exhaust_velocity, steering="gravity_turn",
            plane_normal=normal, kick_angle=kick, label="boost",
        )
        try:
            final = _run(simulator, pad_state, duration, burn, "boost")
        except Exception:
            # Too much kick turns the vehicle into the atmosphere. Report an
            # apogee below any target so the bracket keeps its sign.
            return -1.0e9
        _, apogee = _apsides_of(simulator, final)
        return float(apogee) if np.isfinite(apogee) else 1.0e9

    kick = _bisect(apogee_for, 0.10, 0.02, target_apogee, 2.0e3)
    burn = Burn(
        duration=duration, thrust=vehicle.thrust,
        exhaust_velocity=vehicle.exhaust_velocity, steering="gravity_turn",
        plane_normal=normal, kick_angle=kick, label="boost",
    )
    return burn, _run(simulator, pad_state, duration, burn, "boost")


def _time_to_apogee(simulator: FlightSimulator, state: _FloatArray) -> float:
    """Coast time until the flight-path angle crosses zero, by integration."""
    mu = simulator.gravity.mu
    r = state[simulator.layout.position]
    v = state[simulator.layout.velocity]
    radius = float(np.linalg.norm(r))
    energy = 0.5 * float(v @ v) - mu / radius
    if energy >= 0.0:
        msg = "the boost left an unbound orbit; there is no apogee to coast to"
        raise ValueError(msg)
    semi_major = -mu / (2.0 * energy)
    period = 2.0 * np.pi * float(np.sqrt(semi_major**3 / mu))

    def climb_rate(duration: float) -> float:
        final = _run(simulator, state, max(duration, 1.0), None, "coast")
        position = final[simulator.layout.position]
        velocity = final[simulator.layout.velocity]
        return float(position @ velocity) / float(np.linalg.norm(position))

    # Apogee is inside half a period by construction; the radial rate falls
    # through zero there, so it brackets cleanly.
    return _bisect(climb_rate, 1.0, 0.5 * period, 0.0, 1.0e-2)


def _solve_insertion(
    simulator: FlightSimulator,
    state: _FloatArray,
    vehicle: LaunchVehicle,
    target_perigee: float,
    max_duration: float = 400.0,
) -> Burn:
    """Prograde burn at apogee that raises perigee to the parking perigee."""
    def perigee_after(duration: float) -> float:
        burn = Burn(
            duration=max(duration, 1.0), thrust=vehicle.insertion_thrust,
            exhaust_velocity=vehicle.insertion_exhaust_velocity,
            steering="prograde", label="insertion",
        )
        try:
            final = _run(simulator, state, max(duration, 1.0), burn, "insertion")
        except Exception:
            return 1.0e9
        perigee, _ = _apsides_of(simulator, final)
        return float(perigee) if np.isfinite(perigee) else 1.0e9

    duration = _bisect(perigee_after, 1.0, max_duration, target_perigee, 1.0e3)
    return Burn(
        duration=duration, thrust=vehicle.insertion_thrust,
        exhaust_velocity=vehicle.insertion_exhaust_velocity,
        steering="prograde", label="insertion",
    )


def _solve_deorbit(
    simulator: FlightSimulator,
    state: _FloatArray,
    vehicle: LaunchVehicle,
    target_perigee: float,
    max_duration: float = 400.0,
) -> Burn:
    """Retrograde burn duration that drops perigee to ``target_perigee``."""
    def perigee_after(duration: float) -> float:
        burn = Burn(
            duration=max(duration, 1.0), thrust=vehicle.deorbit_thrust,
            exhaust_velocity=vehicle.deorbit_exhaust_velocity,
            steering="retrograde", label="deorbit",
        )
        try:
            final = _run(simulator, state, max(duration, 1.0), burn, "deorbit")
        except Exception:
            return -1.0e9
        perigee, _ = _apsides_of(simulator, final)
        return float(perigee) if np.isfinite(perigee) else -1.0e9

    duration = _bisect(perigee_after, 1.0, max_duration, target_perigee, 5.0e3)
    return Burn(
        duration=duration, thrust=vehicle.deorbit_thrust,
        exhaust_velocity=vehicle.deorbit_exhaust_velocity,
        steering="retrograde", label="deorbit",
    )


def _segments(
    boost: Burn,
    to_apogee: float,
    insertion: Burn,
    parking: float,
    deorbit: Burn,
    entry_limit: float,
) -> list[MissionSegment]:
    """The six legs of a fractional orbital profile, in order."""
    return [
        MissionSegment(boost.duration, boost, "boost",
                       "gravity turn: vertical, kick, then prograde"),
        MissionSegment(max(to_apogee, 1.0), None, "ascent coast",
                       "unpowered, up the transfer ellipse to apogee"),
        MissionSegment(insertion.duration, insertion, "insertion burn",
                       "prograde at apogee, raising perigee into the parking orbit"),
        MissionSegment(max(parking, 1.0), None, "parking coast",
                       "unpowered, on the orbit the insertion achieved"),
        MissionSegment(deorbit.duration, deorbit, "deorbit burn",
                       "retrograde, integrated as a finite arc"),
        MissionSegment(max(entry_limit, 1.0), None, "deorbit coast and entry",
                       "to impact", stop_at_ground=True),
    ]


def _solve_ascent(
    simulator: FlightSimulator,
    pad_state: _FloatArray,
    normal: _FloatArray,
    vehicle: LaunchVehicle,
    kick_angle: float,
    target_perigee: float,
    target_apogee: float,
) -> tuple[Burn, float, Burn, _FloatArray, int]:
    """Solve the ascent as nested monotone shots.

    Three quantities have to come out right — the burn time, the coast to
    apogee, and the insertion burn — and solving them jointly is both
    slower and less robust than nesting them, because each is monotone on
    its own:

    * **innermost**, the insertion burn duration against perigee. Longer
      burn, higher perigee.
    * **middle**, the boost duration against the *burnout* apogee. Longer
      burn, higher apogee. The coast to apogee is not a free parameter at
      all — it is where the radial rate crosses zero, found by integration.
    * **outermost**, the burnout apogee *target* against the apogee the
      vehicle actually ends up in. These differ because the insertion burn
      is not an impulse: it runs some 700 km of arc, so its second half
      fires past apogee and raises the far side. Aiming the boost at the
      apogee you want gives you a much higher one — measured at 906 km for
      a 250 km request — and the outer loop is what removes that bias.

    Nesting costs integrations and buys the property that no loop can leave
    its bracket, which matters because part of the parameter space makes the
    vehicle turn into the atmosphere and fail outright.
    """
    calls = 0

    def boost_for(duration: float) -> Burn:
        return Burn(
            duration=duration, thrust=vehicle.thrust,
            exhaust_velocity=vehicle.exhaust_velocity, steering="gravity_turn",
            plane_normal=normal, kick_angle=kick_angle, label="boost",
        )

    def burnout_apogee(duration: float) -> float:
        nonlocal calls
        calls += 1
        try:
            final = _run(simulator, pad_state, duration, boost_for(duration), "boost")
        except Exception:
            return -1.0e9
        _, apogee = _apsides_of(simulator, final)
        return float(apogee) if np.isfinite(apogee) else 1.0e9

    def longest_feasible(ceiling: float = 300.0, floor: float = 150.0) -> float:
        """Longest boost this kick angle can actually fly.

        Part of the duration range is not merely wrong but *infeasible*:
        past a point the vehicle is still under thrust when it turns into
        the dense atmosphere, and the integration fails outright. Bisection
        needs a bracket whose ends are both real, so the top is walked down
        until it is.
        """
        probe = ceiling
        while probe > floor:
            if burnout_apogee(probe) > -1.0e8:
                return probe
            probe -= 10.0
        return floor

    def ascend(boost_apogee: float) -> tuple[Burn, float, Burn, _FloatArray, float]:
        nonlocal calls
        duration = _bisect(
            burnout_apogee, 140.0, longest_feasible(), boost_apogee, 3.0e3, steps=14
        )
        boost = boost_for(duration)
        burnout = _run(simulator, pad_state, duration, boost, "boost")
        coast = _time_to_apogee(simulator, burnout)
        at_apogee = _run(simulator, burnout, coast, None, "ascent coast")
        calls += 16

        def perigee_after(seconds: float) -> float:
            nonlocal calls
            calls += 1
            burn = Burn(
                duration=max(seconds, 1.0), thrust=vehicle.insertion_thrust,
                exhaust_velocity=vehicle.insertion_exhaust_velocity,
                steering="prograde", label="insertion",
            )
            try:
                final = _run(simulator, at_apogee, max(seconds, 1.0), burn, "insertion")
            except Exception:
                return 1.0e9
            perigee, _ = _apsides_of(simulator, final)
            return float(perigee) if np.isfinite(perigee) else 1.0e9

        seconds = _bisect(perigee_after, 1.0, 260.0, target_perigee, 3.0e3, steps=14)
        insertion = Burn(
            duration=seconds, thrust=vehicle.insertion_thrust,
            exhaust_velocity=vehicle.insertion_exhaust_velocity,
            steering="prograde", label="insertion",
        )
        in_orbit = _run(simulator, at_apogee, seconds, insertion, "insertion")
        calls += 1
        _, achieved = _apsides_of(simulator, in_orbit)
        return boost, coast, insertion, in_orbit, float(achieved)

    def achieved_apogee(boost_apogee: float) -> float:
        return ascend(boost_apogee)[4]

    # The outer correction is a damped fixed point rather than a bracketed
    # root. The bias is roughly multiplicative — aim the boost at A and the
    # vehicle ends up at kA for a k that varies slowly — so scaling the aim
    # by the ratio converges in three or four passes. A bracket would have
    # to be guessed, and both of its ends can fall outside the reachable
    # envelope, where the boost solve has no solution at all.
    aim = float(target_apogee)
    boost = insertion = None
    coast = 0.0
    in_orbit = pad_state
    for _ in range(5):
        boost, coast, insertion, in_orbit, achieved = ascend(aim)
        if abs(achieved - target_apogee) < 15.0e3:
            break
        ratio = float(np.clip(target_apogee / max(achieved, 1.0e3), 0.25, 4.0))
        aim = float(np.clip(aim * ratio, 40.0e3, 4.0 * target_apogee))
    assert boost is not None and insertion is not None
    return boost, coast, insertion, in_orbit, calls


def fly_fractional_orbital(
    simulator: FlightSimulator,
    launch: GeodeticPosition,
    target: GeodeticPosition,
    vehicle: LaunchVehicle | None = None,
    parking_perigee: float = 170.0e3,
    parking_apogee: float = 250.0e3,
    entry_perigee: float = -400.0e3,
    kick_angle: float = 0.07,
    long_way: bool = True,
    samples_per_segment: int = 150,
    outer_iterations: int = 2,
) -> FlownProfile:
    """Fly a fractional orbital profile end to end through the integrator.

    Every state in the returned mission came out of
    :meth:`~passes.flight.simulator.FlightSimulator.rhs` — J2 gravity, drag
    against the mass the integrator is carrying, mass depletion under
    thrust, the structural block and the charring thermal block — and every
    burn was solved by shooting against that same right-hand side.

    The profile is six legs and three burns, which is what an orbital
    insertion actually is: boost on a gravity turn to a transfer ellipse,
    coast to apogee, circularise into the parking orbit, coast the long (or
    short) way, fire retrograde, and fall.

    Parameters
    ----------
    parking_perigee, parking_apogee:
        Target parking orbit, as altitudes (m).
    entry_perigee:
        Altitude the deorbit burn drops perigee to (m). Negative, because
        the conic has to reach the ground.
    kick_angle:
        Gravity-turn kick (rad). Held fixed and the burn *time* solved
        against it: the two trade off against apogee, and fixing the
        cheaper one leaves a monotone scalar problem.
    long_way:
        Fly the major arc — the fractional-orbital concept — or the minor
        one, which is the control separating flying low from arriving
        backwards.
    outer_iterations:
        Fixed-point passes over whole flown missions. The aim point moves
        under the orbit during the flight, so the plane containing launch
        and target *at arrival* depends on the arrival time, which depends
        on the plane.
    """
    vehicle = vehicle or LaunchVehicle()
    body_radius = simulator.gravity.radius
    pad = simulator.state_at(
        _cartesian(launch, body_radius + max(float(launch.altitude), 0.0)),
        np.zeros(3), mass=vehicle.liftoff_mass,
    )
    # A vanishing upward velocity. The gravity turn holds vertical through
    # the first seconds regardless, but a hard zero leaves the flight-path
    # angle undefined for the diagnostics.
    pad[simulator.layout.velocity] = 1.0e-3 * _cartesian(launch, 1.0)

    lead = 0.0
    calls = 0
    mission: MissionResult | None = None
    aim = target
    parking_guess = 3200.0 if long_way else 300.0
    boost = insertion = deorbit = None
    to_apogee = 0.0

    for _ in range(int(outer_iterations)):
        aim = GeodeticPosition(
            target.latitude,
            float((target.longitude + lead + np.pi) % (2.0 * np.pi) - np.pi),
            target.altitude, target.label,
        )
        normal = _plane_normal(launch, aim, long_way)

        boost, to_apogee, insertion, in_orbit, ascent_calls = _solve_ascent(
            simulator, pad, normal, vehicle, kick_angle,
            parking_perigee, parking_apogee,
        )
        calls += ascent_calls

        # The deorbit burn barely depends on *where* on a near-circular
        # parking orbit it is fired, so it is solved once here and reused
        # while searching the coast. The delivered mission re-solves it at
        # the chosen coast and the reported miss comes from that flight, so
        # the shortcut speeds the search without entering the answer.
        probe = _run(simulator, in_orbit, parking_guess, None, "parking coast")
        nominal = _solve_deorbit(simulator, probe, vehicle, entry_perigee)
        calls += 40

        def miss(
            parking: float,
            _boost: Burn = boost,
            _ins: Burn = insertion,
            _deorbit: Burn = nominal,
            _aim: GeodeticPosition = aim,
            _to_apogee: float = to_apogee,
        ) -> float:
            flown = fly_mission(
                simulator, pad,
                _segments(_boost, _to_apogee, _ins, parking, _deorbit, 5000.0),
                samples_per_segment=6,
            )
            impact = _subpoint(
                flown.result.states[:, -1], flown.flight_time, simulator.layout
            )
            return float(great_circle_range(impact, _aim))

        # Coarse scan, then a bounded refinement. The miss is smooth and
        # V-shaped in the parking coast but its minimum is nowhere near the
        # middle of any interval guessable in advance, and Brent needs a
        # valid three-point bracket to start; a bounded search does not.
        scan = np.linspace(0.35 * parking_guess, 1.9 * parking_guess, 16)
        values = [miss(float(c)) for c in scan]
        calls += scan.size
        best = int(np.argmin(values))
        found = scipy.optimize.minimize_scalar(
            miss,
            bounds=(float(scan[max(best - 1, 0)]), float(scan[min(best + 1, 15)])),
            method="bounded", options={"xatol": 0.5, "maxiter": 30},
        )
        calls += int(found.nfev)
        parking_guess = float(max(found.x, 1.0))

        settled = _run(simulator, in_orbit, parking_guess, None, "parking coast")
        deorbit = _solve_deorbit(simulator, settled, vehicle, entry_perigee)
        mission = fly_mission(
            simulator, pad,
            _segments(boost, to_apogee, insertion, parking_guess, deorbit, 5000.0),
            samples_per_segment=samples_per_segment,
        )
        calls += 40
        lead = EARTH_ROTATION_RATE * mission.flight_time

    assert mission is not None and boost is not None and deorbit is not None
    impact = _subpoint(
        mission.result.states[:, -1], mission.flight_time, simulator.layout
    )
    burnout_index = int(np.searchsorted(mission.result.times, boost.duration))
    achieved = float(
        np.linalg.norm(mission.result.states[simulator.layout.velocity, burnout_index])
    )
    orbit_index = int(np.searchsorted(mission.result.times, mission.phases[3].start_time))
    perigee, apogee = _apsides_of(simulator, mission.result.states[:, orbit_index])
    return FlownProfile(
        mission=mission,
        label="fractional-orbital" if long_way else "fractional-orbital (direct)",
        aimpoint=aim,
        miss_distance=float(great_circle_range(impact, aim)),
        ideal_delta_v=boost.ideal_delta_v(vehicle.liftoff_mass),
        achieved_delta_v=achieved,
        solver_iterations=calls,
        notes={
            "boost_duration": boost.duration,
            "kick_angle_deg": float(np.rad2deg(boost.kick_angle)),
            "coast_to_apogee": to_apogee,
            "insertion_duration": float(insertion.duration) if insertion else 0.0,
            "parking_coast": parking_guess,
            "deorbit_duration": deorbit.duration,
            "achieved_perigee": perigee,
            "achieved_apogee": apogee,
        },
    )
