"""A canonical time history, so a picture cannot disagree with the physics.

The animation layer had a structural weakness worth naming plainly: it
rebuilt its own notion of where the vehicle was, from sub-satellite points
and altitudes, inside a notebook. That is a *second* trajectory model. It
was fed by the real one and so mostly agreed with it, but nothing enforced
that, and two of the defects found while building the animations were
exactly this class of thing — a frame index that stopped at 86 % of the
flight, and a camera placed 96 km underground. Neither was visible to any
physics test, because the physics did not know the animation existed.

:class:`SimulationHistory` removes the second model. It is the one
authoritative record of a run, and the renderer samples it rather than
reconstructing anything. Two producers feed it:

* :meth:`SimulationHistory.from_flight_result` — the coupled multi-physics
  simulator, which carries full state: position, velocity, attitude
  quaternion, angular rate, modal coordinates, temperatures, recession.
* :meth:`SimulationHistory.from_trajectory` — the lighter orbital scenario
  objects, which carry position and time and nothing else.

The second is deliberately *lossy in a declared way*: a scenario
:class:`~passes.orbital.scenario.Trajectory` genuinely has no attitude, so
the history reports ``has_attitude == False`` rather than inventing one.
Consumers ask, and draw a bare marker instead of an oriented body. Silently
substituting an identity quaternion would be the failure this module exists
to prevent.

What is *not* here
------------------

No rendering, no cameras, no interpolation policy beyond the linear and
spherical-linear minimum. This is the data structure and its conversions;
:mod:`passes.viz.scene` consumes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from passes.dynamics.attitude import dcm_from_quaternion

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters for typing
    from passes.flight.simulator import FlightResult
    from passes.orbital.scenario import Trajectory

__all__ = ["SimulationHistory"]

_FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SimulationHistory:
    """One run, sampled, in the form every renderer consumes.

    Attributes
    ----------
    label:
        What this run is.
    times:
        Seconds from the start, strictly increasing.
    positions:
        Inertial position, shape ``(n, 3)`` (m).
    velocities:
        Inertial velocity, shape ``(n, 3)`` (m/s), or ``None`` when the
        producer does not carry one. A chase camera falls back to finite
        differences of ``positions``, which is why this is optional rather
        than required.
    quaternions:
        Attitude, shape ``(n, 4)`` scalar-first, or ``None``. See the
        module docstring on why absence is reported rather than faked.
    extras:
        Any additional per-sample series the producer computed — heat flux,
        recession, dynamic pressure. Carried through untouched so an
        overlay can colour by them without a second physics call.
    """

    label: str
    times: _FloatArray
    positions: _FloatArray
    velocities: _FloatArray | None = None
    quaternions: _FloatArray | None = None
    extras: dict[str, _FloatArray] | None = None

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=np.float64)
        positions = np.asarray(self.positions, dtype=np.float64)
        if times.ndim != 1 or times.size < 2:
            msg = f"times must be 1-D with at least two samples, got shape {times.shape}"
            raise ValueError(msg)
        if np.any(np.diff(times) <= 0.0):
            msg = "times must be strictly increasing"
            raise ValueError(msg)
        if positions.shape != (times.size, 3):
            msg = (
                f"positions must have shape ({times.size}, 3) to match times, "
                f"got {positions.shape}"
            )
            raise ValueError(msg)
        for name, value, width in (
            ("velocities", self.velocities, 3),
            ("quaternions", self.quaternions, 4),
        ):
            if value is None:
                continue
            array = np.asarray(value, dtype=np.float64)
            if array.shape != (times.size, width):
                msg = (
                    f"{name} must have shape ({times.size}, {width}), got {array.shape}"
                )
                raise ValueError(msg)

    # -- properties ------------------------------------------------------

    @property
    def has_attitude(self) -> bool:
        """Whether this producer carried a real attitude.

        Consumers must branch on this rather than assume. A scenario
        trajectory has none, and drawing an oriented vehicle from an
        invented identity quaternion would be a picture of something that
        was never computed.
        """
        return self.quaternions is not None

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0])

    def radii(self) -> _FloatArray:
        return np.asarray(np.linalg.norm(self.positions, axis=1))

    def altitudes(self, body_radius: float) -> _FloatArray:
        return np.asarray(self.radii() - body_radius)

    # -- sampling --------------------------------------------------------

    def sample(self, time: float) -> dict[str, Any]:
        """State at an arbitrary time, by interpolation.

        Position and velocity interpolate linearly; attitude uses spherical
        linear interpolation, because componentwise interpolation of a
        quaternion leaves the unit sphere and the resulting rotation both
        shrinks and shears. Outside the recorded span the endpoints are
        held rather than extrapolated — a camera asking for a time the run
        does not cover should see the last real state, not a linear guess
        past the end of the physics.
        """
        t = float(np.clip(time, self.times[0], self.times[-1]))
        upper = int(np.searchsorted(self.times, t, side="left"))
        if upper <= 0:
            index, blend = 0, 0.0
        elif upper >= self.times.size:
            index, blend = self.times.size - 2, 1.0
        else:
            index = upper - 1
            span = self.times[index + 1] - self.times[index]
            blend = float((t - self.times[index]) / span) if span > 0.0 else 0.0

        state: dict[str, Any] = {
            "time": t,
            "position": (1.0 - blend) * self.positions[index]
            + blend * self.positions[index + 1],
        }
        if self.velocities is not None:
            state["velocity"] = (1.0 - blend) * self.velocities[index] + (
                blend * self.velocities[index + 1]
            )
        else:
            step = self.positions[index + 1] - self.positions[index]
            span = self.times[index + 1] - self.times[index]
            state["velocity"] = step / span if span > 0.0 else np.zeros(3)
        if self.quaternions is not None:
            state["quaternion"] = _slerp(
                self.quaternions[index], self.quaternions[index + 1], blend
            )
            state["dcm"] = dcm_from_quaternion(state["quaternion"])
        if self.extras:
            for name, series in self.extras.items():
                state[name] = float(
                    (1.0 - blend) * series[index] + blend * series[index + 1]
                )
        return state

    # -- producers -------------------------------------------------------

    @classmethod
    def from_flight_result(
        cls, result: FlightResult, label: str = "coupled flight"
    ) -> SimulationHistory:
        """Build from the coupled simulator, keeping everything it computed."""
        layout = result.layout
        states = np.asarray(result.states, dtype=np.float64)
        extras: dict[str, _FloatArray] = {}
        for name in ("effective_radius", "stagnation_heat_flux", "dynamic_pressure"):
            series = getattr(result, name, None)
            if series is not None:
                extras[name] = np.asarray(series, dtype=np.float64)
        extras["recession"] = np.asarray(result.recession, dtype=np.float64)
        return cls(
            label=label,
            times=np.asarray(result.times, dtype=np.float64),
            positions=states[layout.position, :].T,
            velocities=states[layout.velocity, :].T,
            quaternions=states[layout.quaternion, :].T,
            extras=extras,
        )

    @classmethod
    def from_trajectory(
        cls, trajectory: Trajectory, body_radius: float
    ) -> SimulationHistory:
        """Build from an orbital scenario trajectory.

        Attitude is **not** synthesised: a scenario trajectory is a point
        mass on a great circle and has none, so ``quaternions`` stays
        ``None`` and :attr:`has_attitude` reports it.
        """
        radii = body_radius + np.asarray(trajectory.altitudes, dtype=np.float64)
        latitudes = np.array([p.latitude for p in trajectory.subpoints])
        longitudes = np.array([p.longitude for p in trajectory.subpoints])
        positions = np.stack(
            [
                radii * np.cos(latitudes) * np.cos(longitudes),
                radii * np.cos(latitudes) * np.sin(longitudes),
                radii * np.sin(latitudes),
            ],
            axis=1,
        )
        return cls(
            label=trajectory.label,
            times=np.asarray(trajectory.times, dtype=np.float64),
            positions=positions,
            extras={"altitude": np.asarray(trajectory.altitudes, dtype=np.float64)},
        )


def _slerp(start: _FloatArray, end: _FloatArray, blend: float) -> _FloatArray:
    """Spherical linear interpolation between two unit quaternions."""
    a = np.asarray(start, dtype=np.float64)
    b = np.asarray(end, dtype=np.float64)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        # q and -q are the same rotation; take the short way round.
        b, dot = -b, -dot
    if dot > 0.9995:
        result = a + blend * (b - a)
    else:
        angle = float(np.arccos(np.clip(dot, -1.0, 1.0)))
        sine = np.sin(angle)
        result = (np.sin((1.0 - blend) * angle) * a + np.sin(blend * angle) * b) / sine
    norm = float(np.linalg.norm(result))
    return np.asarray(result / norm if norm > 0.0 else np.array([1.0, 0.0, 0.0, 0.0]))
