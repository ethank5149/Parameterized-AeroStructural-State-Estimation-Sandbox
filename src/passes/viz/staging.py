"""Staging: which stages are still attached, and where the spent ones went.

The animation drew one vehicle glyph for the whole flight. A staged
launcher is not one vehicle — an RS-28 class missile arrives at its parking
orbit having shed roughly **93 % of its lift-off mass**, and the picture
that shows the same body from the pad to impact is not a rendering
shortcut, it is a different vehicle from the one the mass model integrated.

Two things live here, and they are separate on purpose.

**What is attached.** :class:`StagingPlan` reads the
:class:`~passes.systems.mass.VehicleMassModel` and a list of separations and
answers, for any instant, which stages the vehicle still has. The glyph is
then built from those stages' *own stations* — :attr:`Stage.forward` and
:attr:`Stage.aft`, metres aft of the nose tip, the same numbers the centre
of mass and the inertia come from — so a stack that has dropped its first
stage is drawn shorter by exactly the length the mass model says it lost.

**Where the spent stages went.** This is the part that could have been
faked and is not. Each jettisoned stage is **integrated**, from the vehicle
state at the instant of separation, under the same J2 gravity the flight
used and its own drag, with its own dry mass and frontal area taken from
the mass model and the mould line. What comes back is a
:class:`~passes.viz.history.SimulationHistory` like any other, which the
renderer draws with the same track and marker code as the vehicle, and
which a test can interrogate.

The one invented number, declared
---------------------------------

Real separation is not free: retro-rockets, springs or a pneumatic pusher
impart a small relative velocity so the spent stage does not re-contact the
stack. :attr:`Separation.relative_speed` is that number, it defaults to
1.5 m/s along the negative flight direction, and it is a **parameter with
no source in this framework** — the mass model has no separation hardware
in it. Everything else about the discarded trajectory is integrated.

It also barely matters, and that is worth stating rather than hiding: over
the following minute a 1.5 m/s offset separates the bodies by 90 m, while
the difference in *drag* between a 200-tonne stack and a 9-tonne empty
stage separates them by kilometres. The parameter sets the first second of
the picture; the physics sets the rest.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from passes.orbital.gravity import EARTH, GravityModel, gravitational_acceleration
from passes.orbital.scenario import Event, Phase
from passes.systems.mass import Stage, VehicleMassModel
from passes.viz.history import SimulationHistory

__all__ = [
    "Separation",
    "StagingPlan",
    "propagate_jettison",
    "stack_polylines",
]

_FloatArray = NDArray[np.float64]

#: Drag coefficient used for a tumbling spent stage. A body that has lost
#: its attitude control tumbles, and a tumbling cylinder's orientation-
#: averaged drag coefficient is near unity — well above the 0.2-0.3 of the
#: same body flying nose-first. It is a round number and it is labelled as
#: one; the spent stage's trajectory is a picture, not a debris footprint.
TUMBLING_DRAG_COEFFICIENT = 1.0


@dataclass(frozen=True)
class Separation:
    """One stage leaving the stack.

    Attributes
    ----------
    stage:
        Name of the :class:`~passes.systems.mass.Stage` being jettisoned.
    time:
        Mission time of separation (s).
    relative_speed:
        Speed imparted by the separation hardware (m/s), along the negative
        flight direction. See the module note: this is the one number here
        with no source, and it sets only the first seconds of the picture.
    """

    stage: str
    time: float
    relative_speed: float = 1.5

    def __post_init__(self) -> None:
        if not np.isfinite(self.time):
            msg = f"separation time must be finite, got {self.time}"
            raise ValueError(msg)
        if not (np.isfinite(self.relative_speed) and self.relative_speed >= 0.0):
            msg = (
                f"relative_speed must be finite and >= 0, got {self.relative_speed}"
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class StagingPlan:
    """A mass model plus when each of its stages is discarded.

    Attributes
    ----------
    model:
        The vehicle. Its :attr:`~passes.systems.mass.VehicleMassModel.stages`
        are the only names a separation may refer to.
    separations:
        In time order. Checked on construction, because a plan that drops
        stage 2 before stage 1 describes a vehicle that cannot fly and would
        otherwise only show up as a strange picture.
    """

    model: VehicleMassModel
    separations: tuple[Separation, ...] = ()

    def __post_init__(self) -> None:
        known = {stage.name for stage in self.model.stages}
        for separation in self.separations:
            if separation.stage not in known:
                msg = (
                    f"separation names stage {separation.stage!r}, which the mass "
                    f"model does not have; it has {sorted(known)}"
                )
                raise ValueError(msg)
        names = [s.stage for s in self.separations]
        if len(set(names)) != len(names):
            msg = f"a stage may only separate once, got {names}"
            raise ValueError(msg)
        times = [s.time for s in self.separations]
        if list(times) != sorted(times):
            msg = f"separations must be in time order, got {times}"
            raise ValueError(msg)

    @classmethod
    def from_events(
        cls,
        model: VehicleMassModel,
        events: Sequence[Event],
        at: Mapping[str, str],
        relative_speed: float = 1.5,
    ) -> StagingPlan:
        """Build a plan by naming, for each stage, the event it leaves at.

        Explicit rather than inferred. Event names come from
        :func:`~passes.flight.mission.fly_mission`, which builds them from
        *burn* labels — "boost cutoff", "insertion cutoff" — and a burn
        label is not a stage name. Guessing the correspondence from a
        substring match would be right for the bundled profile and silently
        wrong for a two-stage one that calls both burns "boost".

        Parameters
        ----------
        at:
            ``{stage name: event name}``. Every event named must exist.

        Raises
        ------
        KeyError
            Naming the missing event and listing what the mission did emit,
            which is the whole diagnostic a caller needs.
        """
        by_name = {event.name: event for event in events}
        found: list[Separation] = []
        for stage, event_name in at.items():
            event = by_name.get(event_name)
            if event is None:
                msg = (
                    f"stage {stage!r} is set to separate at event {event_name!r}, "
                    f"which this mission does not have; it emitted "
                    f"{sorted(by_name)}"
                )
                raise KeyError(msg)
            found.append(
                Separation(stage=stage, time=float(event.time),
                           relative_speed=relative_speed)
            )
        return cls(model=model, separations=tuple(sorted(found, key=lambda s: s.time)))

    def stage(self, name: str) -> Stage:
        """The named stage, or a ``KeyError`` naming what there is."""
        for stage in self.model.stages:
            if stage.name == name:
                return stage
        msg = f"no stage {name!r}; the model has {[s.name for s in self.model.stages]}"
        raise KeyError(msg)

    def present_at(self, time: float) -> tuple[str, ...]:
        """Stages still attached at ``time`` (s), forward to aft.

        A separation at exactly ``time`` counts as **done**, matching
        :class:`~passes.systems.mass.VehicleMassModel`'s own convention that
        an event's state is the state after it.
        """
        gone = {s.stage for s in self.separations if s.time <= float(time)}
        return tuple(s.name for s in self.model.stages if s.name not in gone)

    def jettisoned_by(self, time: float) -> tuple[Separation, ...]:
        """Separations that have already happened at ``time`` (s)."""
        return tuple(s for s in self.separations if s.time <= float(time))

    def phases(self) -> tuple[Phase, ...]:
        """One phase per stack configuration, for a timeline or a caption.

        Named by what is *flying*, not by what was dropped — "stage 1 +
        stage 2 + bus + payload" and then "stage 2 + bus + payload" — so a
        viewer reads the vehicle rather than the event log.
        """
        edges = [0.0, *[s.time for s in self.separations]]
        out: list[Phase] = []
        for index, start in enumerate(edges):
            present = self.present_at(start)
            if not present:
                break
            end = edges[index + 1] if index + 1 < len(edges) else float("inf")
            out.append(Phase(" + ".join(present), start, end, "stack configuration"))
        return tuple(out)


def stack_polylines(
    plan: StagingPlan,
    time: float,
    stations: _FloatArray,
    radii: _FloatArray,
    generators: int = 4,
) -> list[_FloatArray]:
    """Body-frame polylines for the stack as it stands at ``time``.

    Drawn from the mass model's **own stations** against the mould line, so
    the glyph is the vehicle the mass properties were computed for: a stack
    that has dropped its first stage is shorter by exactly the length the
    model says it lost, and the separation rings fall where
    :func:`~passes.systems.mass.sarmat_mass_model` put them.

    Parameters
    ----------
    plan:
        The staging plan.
    time:
        Mission time (s).
    stations, radii:
        The outer mould line, metres aft of the tip. The same arrays the
        mass model was built from.
    generators:
        Meridian lines drawn along the body, so it reads as a solid of
        revolution rather than as an outline.

    Returns
    -------
    list[numpy.ndarray]
        Polylines of shape ``(n, 3)`` in the body frame, **in metres**, with
        the nose tip at the origin and ``+x`` forward. Aft stations become
        negative ``x``, which is the sign convention
        :data:`~passes.viz.scene.NOSE_AXIS` and the glyph share; the mass
        model's stations run the other way and are negated here rather than
        anywhere a reader would have to notice.
    """
    line_x = np.asarray(stations, dtype=np.float64)
    line_r = np.asarray(radii, dtype=np.float64)
    if line_x.shape != line_r.shape or line_x.ndim != 1 or line_x.size < 2:
        msg = (
            f"stations and radii must be matching 1-D arrays of at least two "
            f"points, got {line_x.shape} and {line_r.shape}"
        )
        raise ValueError(msg)
    if generators < 1:
        msg = f"generators must be >= 1, got {generators}"
        raise ValueError(msg)

    present = plan.present_at(time)
    lines: list[_FloatArray] = []
    angles = np.linspace(0.0, 2.0 * np.pi, generators + 1)[:-1]
    ring_angles = np.linspace(0.0, 2.0 * np.pi, 25)

    for name in present:
        stage = plan.stage(name)
        # Sample the mould line across this stage only, keeping the source
        # points inside it so a change of slope is not cut off.
        inside = (line_x > stage.forward) & (line_x < stage.aft)
        cuts = np.concatenate(
            [[stage.forward], line_x[inside], [stage.aft]]
        )
        radius = np.interp(cuts, line_x, line_r)
        for phi in angles:
            lines.append(
                np.stack(
                    [-cuts, radius * np.cos(phi), radius * np.sin(phi)], axis=1
                )
            )
        # A ring at each end: these are the separation planes, and they are
        # what makes a staged stack read as a stack.
        for station in (stage.forward, stage.aft):
            ring_radius = float(np.interp(station, line_x, line_r))
            lines.append(
                np.stack(
                    [
                        np.full_like(ring_angles, -station),
                        ring_radius * np.cos(ring_angles),
                        ring_radius * np.sin(ring_angles),
                    ],
                    axis=1,
                )
            )
    return lines


def propagate_jettison(
    history: SimulationHistory,
    separation: Separation,
    stage: Stage,
    frontal_area: float,
    duration: float | None = None,
    samples: int = 240,
    gravity: GravityModel = EARTH,
    density: object | None = None,
    drag_coefficient: float = TUMBLING_DRAG_COEFFICIENT,
    rtol: float = 1.0e-8,
) -> SimulationHistory:
    """Fly a jettisoned stage from separation, as its own trajectory.

    **Integrated, not drawn.** The spent stage leaves with the stack's state
    at the separation instant, plus the small relative velocity the
    separation hardware imparts, and is then propagated under the same J2
    gravity the flight used and its own drag. It diverges from the vehicle
    because its ballistic coefficient is different — an empty stage is light
    and blunt where the stack was heavy and slender — and that divergence is
    the whole content of the picture.

    Parameters
    ----------
    history:
        The vehicle's run. Sampled at ``separation.time`` for the initial
        state; nothing is re-derived.
    separation:
        Which stage, when, and with what push-off.
    stage:
        The stage itself. Its
        :attr:`~passes.systems.mass.Stage.dry_mass` sets the ballistic
        coefficient — the propellant is gone, which is why it was dropped.
    frontal_area:
        Reference area for drag (m^2), from the mould line at the stage.
    duration:
        Seconds to fly. ``None`` runs to the end of ``history``, so the
        spent stage is on screen for as long as the flight it left.
    samples:
        Output density.
    density:
        Anything with a ``density(altitude)`` method — a
        :class:`~passes.atmosphere.model.TabulatedAtmosphere` is the
        intended one. ``None`` uses the same isothermal law
        :class:`~passes.flight.simulator.FlightSimulator` reports its
        diagnostics with, so the two agree.
    drag_coefficient:
        See :data:`TUMBLING_DRAG_COEFFICIENT`.

    Returns
    -------
    SimulationHistory
        Labelled with the stage name, carrying positions and velocities and
        an ``altitude`` extras series. Terminates at the surface, so a first
        stage that comes down does not carry on underground — which is the
        defect :func:`~passes.flight.mission.fly_mission` was given its
        ground event for.
    """
    import scipy.integrate

    if not (np.isfinite(frontal_area) and frontal_area > 0.0):
        msg = f"frontal_area must be finite and > 0, got {frontal_area}"
        raise ValueError(msg)

    start = float(np.clip(separation.time, history.times[0], history.times[-1]))
    state = history.sample(start)
    position = np.asarray(state["position"], dtype=np.float64)
    velocity = np.asarray(state["velocity"], dtype=np.float64)

    speed = float(np.linalg.norm(velocity))
    if speed > 1.0e-6:
        velocity = velocity - separation.relative_speed * velocity / speed

    span = (
        float(history.times[-1] - start) if duration is None else float(duration)
    )
    if span <= 0.0:
        msg = (
            f"the jettison at t={start:.1f} s has no flight left in a history "
            f"ending at t={history.times[-1]:.1f} s; pass an explicit duration"
        )
        raise ValueError(msg)

    ballistic = float(stage.dry_mass) / (drag_coefficient * float(frontal_area))
    radius_floor = gravity.radius

    def air_density(altitude: float) -> float:
        if density is None:
            return float(1.225 * np.exp(-max(altitude, 0.0) / 8500.0))
        return float(density.density(altitude))  # type: ignore[attr-defined]

    def rhs(_t: float, y: _FloatArray) -> _FloatArray:
        r, v = y[:3], y[3:]
        accel = gravitational_acceleration(r, gravity)
        altitude = float(np.linalg.norm(r)) - radius_floor
        rho = air_density(altitude)
        speed_now = float(np.linalg.norm(v))
        if speed_now > 0.0 and rho > 0.0:
            accel = accel - (0.5 * rho * speed_now / ballistic) * v
        return np.concatenate([v, accel])

    def ground(_t: float, y: _FloatArray) -> float:
        return float(np.linalg.norm(y[:3]) - radius_floor)

    ground.terminal = True  # type: ignore[attr-defined]
    ground.direction = -1.0  # type: ignore[attr-defined]

    grid = np.linspace(0.0, span, max(int(samples), 2))
    solution = scipy.integrate.solve_ivp(
        rhs,
        (0.0, span),
        np.concatenate([position, velocity]),
        method="LSODA",
        t_eval=grid,
        rtol=rtol,
        atol=1.0e-6,
        events=ground,
        dense_output=True,
    )
    if not solution.success:  # pragma: no cover - a failure here is a bug
        msg = f"jettisoned {separation.stage!r} failed to integrate: {solution.message}"
        raise RuntimeError(msg)

    times = np.asarray(solution.t)
    states = np.asarray(solution.y)
    hits = np.asarray(solution.t_events[0]) if solution.t_events else np.array([])
    if hits.size:
        impact = float(hits[0])
        keep = times < impact
        times = np.concatenate([times[keep], [impact]])
        states = np.concatenate([states[:, keep], solution.sol(impact)[:, None]], axis=1)

    positions = states[:3, :].T
    return SimulationHistory(
        label=f"{separation.stage} (jettisoned)",
        times=start + times,
        positions=positions,
        velocities=states[3:, :].T,
        extras={
            "altitude": np.asarray(
                np.linalg.norm(positions, axis=1) - radius_floor
            )
        },
        events=(Event("separation", start, f"from {history.label}"),),
    )


def jettison_histories(
    history: SimulationHistory,
    plan: StagingPlan,
    frontal_areas: Mapping[str, float] | Iterable[tuple[str, float]],
    **kwargs: object,
) -> dict[str, SimulationHistory]:
    """Propagate every separation in a plan, keyed by stage name."""
    areas = dict(frontal_areas)
    out: dict[str, SimulationHistory] = {}
    for separation in plan.separations:
        area = areas.get(separation.stage)
        if area is None:
            msg = (
                f"no frontal area for stage {separation.stage!r}; supplied "
                f"{sorted(areas)}"
            )
            raise KeyError(msg)
        out[separation.stage] = propagate_jettison(
            history, separation, plan.stage(separation.stage), area,
            **kwargs,  # type: ignore[arg-type]
        )
    return out
