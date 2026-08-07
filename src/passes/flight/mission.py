"""Multi-segment flight through the coupled simulator.

A fractional orbital profile is boost, coast, deorbit burn, coast, entry.
Each leg is the same right-hand side with a different thrust state, so the
mission is a *sequence of solves* handed off state to state, concatenated
into one :class:`~passes.flight.simulator.FlightResult`.

Why not one solve with a gated thrust
-------------------------------------

Thrust switching is a real discontinuity in the derivative. A stiff
implicit integrator asked to step across one either rejects steps until it
crawls, or takes a step straddling the event and smears it — and the second
failure is silent. Splitting at the discontinuity makes each solve smooth,
makes the burn boundaries exact rather than resolved, and is faster. The
cost is that the simulator's "one integrator call" property applies per leg
rather than per mission, which is a fair description of what a staged
vehicle does.

The ground event
----------------

The simulator had no terminal condition: it integrated a fixed duration and
kept going, reaching 45 km *below* the surface in 300 s. Impact is now a
terminal ``solve_ivp`` event, so a flight ends where it ends and the last
sample is the impact state rather than the nearest output grid point.

What this buys the animations
-----------------------------

Everything drawn now comes from the integrator. The geometry model in
:mod:`passes.orbital.scenario` remains the *planner* — it answers "what
must the vehicle do", cheaply, in closed form — and the simulator is the
*truth*. They disagree, and the disagreement is the point: the planner is
two-body with no drag and no J2 and no gravity loss, and
:func:`fly_mission` reports what those cost.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import scipy.integrate
from numpy.typing import NDArray

from passes.aerothermal import sutton_graves
from passes.flight.propulsion import Burn
from passes.flight.simulator import FlightResult, FlightSimulator

if TYPE_CHECKING:  # pragma: no cover - typing only
    from passes.orbital.scenario import Event, Phase

__all__ = ["MissionSegment", "fly_mission"]

_FloatArray = NDArray[np.float64]
_RHO0 = 1.225
_H_SCALE = 8500.0


@dataclass(frozen=True)
class MissionSegment:
    """One leg: a coast or a burn, with a name that reaches the picture.

    Attributes
    ----------
    duration:
        Length of the leg (s).
    burn:
        Thrust arc, or ``None`` for a coast. A burn's own ``duration`` is
        what its steering program is parameterised against; this
        ``duration`` is how long the leg is integrated, and the two are
        required to agree so a pitch program cannot be truncated silently.
    label:
        Phase name, carried to :class:`~passes.orbital.scenario.Phase`.
    note:
        One line of detail for the phase.
    stop_at_ground:
        Whether impact ends the mission during this leg. Only the last leg
        normally wants this; an early one would end the flight in the
        middle of a boost.
    """

    duration: float
    burn: Burn | None = None
    label: str = "coast"
    note: str = ""
    stop_at_ground: bool = False

    def __post_init__(self) -> None:
        if not (np.isfinite(self.duration) and self.duration > 0.0):
            msg = f"segment duration must be finite and > 0, got {self.duration}"
            raise ValueError(msg)
        if self.burn is not None and not np.isclose(
            self.burn.duration, self.duration, rtol=1e-9
        ):
            msg = (
                f"segment {self.label!r} lasts {self.duration} s but its burn is "
                f"programmed over {self.burn.duration} s; a pitch program run for "
                "the wrong time does not reach its commanded final angle"
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class MissionResult:
    """A flown mission: the concatenated result plus its structure."""

    result: FlightResult
    phases: tuple[Phase, ...]
    events: tuple[Event, ...]
    impacted: bool
    segments: tuple[MissionSegment, ...] = field(repr=False, default=())

    @property
    def flight_time(self) -> float:
        return float(self.result.times[-1])


def fly_mission(
    simulator: FlightSimulator,
    initial_state: _FloatArray,
    segments: list[MissionSegment] | tuple[MissionSegment, ...],
    samples_per_segment: int = 120,
    rtol: float = 1e-9,
    method: str = "BDF",
) -> MissionResult:
    """Fly ``segments`` in order through the coupled simulator.

    Parameters
    ----------
    simulator:
        The engine. Its right-hand side carries J2 gravity, drag, attitude,
        the structural block and the charring thermal block, and now
        thrust.
    initial_state:
        Packed state at ignition of the first segment.
    segments:
        Legs, in order.
    samples_per_segment:
        Output density per leg. Uniform per *leg*, not per mission, so a
        ten-second deorbit burn is sampled as finely as a fifty-minute
        coast — which is what makes the burn visible in an animation
        rather than a corner cut between two samples.
    rtol, method:
        Integrator settings. ``LSODA`` rather than the simulator's default
        ``BDF``: a powered launch is not stiff while it is in the dense
        atmosphere accelerating, and LSODA switches when the structural
        block starts to bite. The choice is checked in the tests by
        integrating the same mission both ways.

    Returns
    -------
    MissionResult
        The concatenated flight, with one
        :class:`~passes.orbital.scenario.Phase` per leg and an
        :class:`~passes.orbital.scenario.Event` at every boundary.
    """
    from passes.orbital.scenario import Event, Phase

    if not segments:
        msg = "a mission needs at least one segment"
        raise ValueError(msg)

    layout = simulator.layout
    body_radius = simulator.gravity.radius
    state = np.asarray(initial_state, dtype=np.float64).copy()
    if state.shape != (layout.size,):
        msg = f"initial_state must have shape ({layout.size},), got {state.shape}"
        raise ValueError(msg)

    def ground(_t: float, y: _FloatArray, *_args: object) -> float:
        """Altitude, as a terminal event. Zero at the surface."""
        return float(np.linalg.norm(y[layout.position]) - body_radius)

    ground.terminal = True  # type: ignore[attr-defined]
    ground.direction = -1.0  # type: ignore[attr-defined]

    times: list[_FloatArray] = []
    blocks: list[_FloatArray] = []
    phases: list[Phase] = []
    events: list[Event] = []
    clock = 0.0
    evaluations = 0
    impacted = False
    started = time.perf_counter()

    for index, segment in enumerate(segments):
        burn = segment.burn
        if burn is None or burn.steering not in ("prograde", "retrograde"):
            continue
        speed = float(np.linalg.norm(state[layout.velocity])) if index == 0 else 1.0
        if speed < 1.0e-6:
            msg = (
                f"segment {index} ({segment.label!r}) uses {burn.steering!r} steering "
                "but the vehicle is at rest; velocity-relative steering has no "
                "direction there, so a launch needs 'gravity_turn'"
            )
            raise ValueError(msg)

    events.append(Event("lift-off", 0.0, segments[0].label))
    for index, segment in enumerate(segments):
        span = (0.0, float(segment.duration))
        grid = np.linspace(*span, int(samples_per_segment))
        burn = segment.burn

        solution = scipy.integrate.solve_ivp(
            lambda t, y, _b=burn: simulator.rhs(t, y, burn=_b),
            span,
            state,
            method=method,
            t_eval=grid,
            rtol=rtol,
            atol=simulator.absolute_tolerances(),
            events=ground if segment.stop_at_ground else None,
            dense_output=segment.stop_at_ground,
        )
        if not solution.success:
            msg = f"segment {index} ({segment.label!r}) failed: {solution.message}"
            raise RuntimeError(msg)
        evaluations += int(solution.nfev)

        leg_times = np.asarray(solution.t)
        leg_states = np.asarray(solution.y)
        if segment.stop_at_ground and solution.t_events is not None:
            hits = np.asarray(solution.t_events[0])
            if hits.size:
                impacted = True
                impact_time = float(hits[0])
                keep = leg_times < impact_time
                leg_times = np.concatenate([leg_times[keep], [impact_time]])
                leg_states = np.concatenate(
                    [leg_states[:, keep], solution.sol(impact_time)[:, None]], axis=1
                )

        # Drop the duplicated hand-over sample so the concatenated clock is
        # strictly increasing, which `SimulationHistory` requires.
        offset = 1 if times else 0
        times.append(clock + leg_times[offset:])
        blocks.append(leg_states[:, offset:])
        phases.append(
            Phase(segment.label, clock, clock + float(leg_times[-1]), segment.note)
        )
        clock += float(leg_times[-1])
        state = leg_states[:, -1].copy()

        if index + 1 < len(segments) and not impacted:
            events.append(Event(_boundary_name(segment, segments[index + 1]), clock,
                                _boundary_detail(state, layout, body_radius)))
        if impacted:
            break

    events.append(
        Event("impact" if impacted else "end of mission", clock,
              _boundary_detail(state, layout, body_radius))
    )

    stacked = np.concatenate(blocks, axis=1)
    stamps = np.concatenate(times)
    return MissionResult(
        result=_assemble(simulator, stamps, stacked, evaluations,
                         time.perf_counter() - started),
        phases=tuple(phases[: len(phases)]),
        events=tuple(events),
        impacted=impacted,
        segments=tuple(segments),
    )


def _boundary_name(current: MissionSegment, following: MissionSegment) -> str:
    """What to call the instant between two legs."""
    if current.burn is not None and following.burn is None:
        return f"{current.burn.label} cutoff"
    if current.burn is None and following.burn is not None:
        return f"{following.burn.label} ignition"
    return f"{following.label} start"


def _boundary_detail(
    state: _FloatArray, layout: object, body_radius: float
) -> str:
    position = state[layout.position]  # type: ignore[attr-defined]
    velocity = state[layout.velocity]  # type: ignore[attr-defined]
    radius = float(np.linalg.norm(position))
    speed = float(np.linalg.norm(velocity))
    gamma = 0.0
    if radius > 0.0 and speed > 0.0:
        gamma = float(
            np.rad2deg(np.arcsin(np.clip(float(position @ velocity) / (radius * speed),
                                         -1.0, 1.0)))
        )
    return (
        f"{(radius - body_radius) / 1e3:,.0f} km, {speed:,.0f} m/s, "
        f"gamma {gamma:+.1f} deg"
    )


def _assemble(
    simulator: FlightSimulator,
    times: _FloatArray,
    states: _FloatArray,
    evaluations: int,
    wall: float,
) -> FlightResult:
    """Rebuild the derived diagnostics over the concatenated states.

    Recomputed here rather than concatenated from the per-leg results,
    because they are pure functions of the state and recomputing cannot
    drift out of step with it.
    """
    layout = simulator.layout
    recession = states[layout.recession, :]
    r_eff = np.asarray([simulator.effective_radius(float(s)) for s in recession])
    radius = np.linalg.norm(states[layout.position, :], axis=0)
    altitude = radius - simulator.gravity.radius
    density = _RHO0 * np.exp(-np.maximum(altitude, 0.0) / _H_SCALE)
    speed = np.linalg.norm(states[layout.velocity, :], axis=0)
    heat = np.where(
        (density > 0.0) & (speed > 0.0),
        simulator.config.heat_load_fraction
        * np.asarray(
            sutton_graves(np.maximum(density, 1e-300), r_eff, np.maximum(speed, 1e-300))
        ),
        0.0,
    )
    return FlightResult(
        times=times,
        states=states,
        layout=layout,
        n_rhs_evaluations=evaluations,
        wall_time=wall,
        effective_radius=r_eff,
        stagnation_heat_flux=np.asarray(heat),
        dynamic_pressure=np.asarray(0.5 * density * speed**2),
    )
