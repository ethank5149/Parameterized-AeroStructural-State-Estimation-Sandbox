"""Hypersonic glide guidance: drag tracking and bank-angle modulation.

A lifting entry vehicle has exactly one continuous control that matters over
the glide: the **bank angle**. Its magnitude sets how much of the lift
vector is held vertical, which governs how fast the vehicle sheds energy
and therefore how far it flies. Its *sign* sets which way the horizontal
component points, which governs crossrange. The two are almost independent,
and the whole architecture below follows from that split.

Why energy, not time
--------------------

Terminal accuracy is a question of where the vehicle is when it has slowed
down, not of where it is at a particular clock reading. Using specific
energy as the independent variable makes that explicit and removes the
need to predict a time of flight. The key relation is exact rather than a
modelling choice: with :math:`e = V^2/2 - \\mu/r` and drag deceleration
:math:`D`,

.. math::

    \\frac{\\mathrm{d}e}{\\mathrm{d}t} = -D V, \\qquad
    \\frac{\\mathrm{d}R}{\\mathrm{d}t} = V \\cos\\gamma
    \\;\\Longrightarrow\\;
    \\frac{\\mathrm{d}R}{\\mathrm{d}e} = -\\frac{\\cos\\gamma}{D}.

So range-to-go is :math:`\\int \\cos\\gamma \\, \\mathrm{d}e / D` over the
energy still to be depleted, and on a shallow glide where
:math:`\\cos\\gamma \\approx 1` it is simply the integral of :math:`1/D`.
**Commanding a drag profile is therefore commanding a range**, which is why
entry guidance in the Shuttle lineage tracks drag rather than altitude or
flight-path angle. :func:`range_to_go` evaluates that integral and
:class:`DragTracker` closes a loop on it.

Bank reversals, and why the deadband is scheduled
-------------------------------------------------

Holding a constant bank sign accumulates crossrange indefinitely, so the
sign is flipped whenever heading error leaves a deadband. A *constant*
deadband is the obvious choice and is wrong: early in the glide the vehicle
is fast and a given heading error converts into enormous crossrange, while
late in the glide there is not enough energy left to correct anything, so
reversing costs more in drag disturbance than it buys. The deadband is
therefore widened at high velocity and narrowed at low, which produces
few reversals early and tight control late. :func:`bank_reversal_needed`
implements that with an explicit schedule rather than a tuned constant.

Scope
-----

Three-degree-of-freedom point mass over a **non-rotating** spherical Earth
with an exponential atmosphere. Rotation matters for a real trajectory —
it is a few percent of crossrange over a long glide — and is omitted here
deliberately: adding it changes the plant but not the guidance structure,
and leaving it out keeps the verification properties in this module
attributable to the guidance law. Aerodynamics are a fixed lift-to-drag
ratio, which is the right level of fidelity for a guidance study and is
where :mod:`passes.aerodynamics` would be substituted for a real vehicle.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.integrate
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "DragTracker",
    "EntryVehicle",
    "GlideResult",
    "GlideState",
    "atmospheric_density",
    "bank_reversal_needed",
    "crossrange_deadband",
    "range_to_go",
    "simulate_glide",
]

_FloatArray = NDArray[np.float64]

#: Exponential atmosphere, matching :mod:`passes.orbital.coast`.
_RHO0 = 1.225
_H_SCALE = 8500.0
_MU = 3.986004418e14
_R_EARTH = 6378137.0


def atmospheric_density(altitude: ArrayLike) -> _FloatArray:
    """Exponential-atmosphere density (kg/m³) at geometric altitude (m).

    Clamped at zero altitude rather than extrapolated: below the surface
    the model has no meaning, and letting it grow keeps a diverging
    trajectory numerically alive long past the point where it stopped
    describing anything.
    """
    h = np.asarray(altitude, dtype=np.float64)
    return np.asarray(_RHO0 * np.exp(-np.maximum(h, 0.0) / _H_SCALE))


@dataclass(frozen=True)
class EntryVehicle:
    """Point-mass glide vehicle.

    Attributes
    ----------
    ballistic_coefficient:
        :math:`m/(C_D S)` (kg/m²). Sets how deep the vehicle must fly to
        generate a given drag, and so where in the atmosphere the glide
        settles.
    lift_to_drag:
        :math:`L/D`, held constant. This is the single number that decides
        how much of the glide is controllable: crossrange capability scales
        roughly with its square.
    max_bank:
        Largest bank magnitude the vehicle will command (rad). Bounding it
        matters because the drag tracker's authority is
        :math:`\\cos\\sigma`, which goes to zero at 90 degrees — an
        unbounded command can ask for a bank that leaves no vertical lift
        at all and drops the vehicle out of the glide.
    """

    ballistic_coefficient: float
    lift_to_drag: float
    max_bank: float = np.deg2rad(80.0)

    def __post_init__(self) -> None:
        if not (np.isfinite(self.ballistic_coefficient) and self.ballistic_coefficient > 0.0):
            raise ValueError(
                f"ballistic_coefficient must be finite and > 0, got {self.ballistic_coefficient}"
            )
        if not (np.isfinite(self.lift_to_drag) and self.lift_to_drag > 0.0):
            raise ValueError(f"lift_to_drag must be finite and > 0, got {self.lift_to_drag}")
        if not (0.0 < self.max_bank < 0.5 * np.pi):
            raise ValueError(
                f"max_bank must lie in (0, pi/2); at pi/2 the vertical lift "
                f"component vanishes and the drag tracker loses all "
                f"authority. Got {self.max_bank}"
            )

    def drag_acceleration(self, altitude: float, speed: float) -> float:
        """Drag deceleration (m/s²)."""
        rho = float(atmospheric_density(altitude))
        return 0.5 * rho * speed**2 / self.ballistic_coefficient

    def lift_acceleration(self, altitude: float, speed: float) -> float:
        """Lift acceleration magnitude (m/s²), before banking."""
        return self.lift_to_drag * self.drag_acceleration(altitude, speed)


@dataclass(frozen=True)
class GlideState:
    """Point-mass state over a spherical Earth.

    Angles in radians, distances in metres, speed in m/s. ``heading`` is
    measured from local north, positive toward east.
    """

    radius: float
    longitude: float
    latitude: float
    speed: float
    flight_path_angle: float
    heading: float

    @property
    def altitude(self) -> float:
        return self.radius - _R_EARTH

    @property
    def specific_energy(self) -> float:
        """:math:`V^2/2 - \\mu/r` (J/kg), the guidance independent variable."""
        return 0.5 * self.speed**2 - _MU / self.radius

    def as_array(self) -> _FloatArray:
        return np.array(
            [
                self.radius,
                self.longitude,
                self.latitude,
                self.speed,
                self.flight_path_angle,
                self.heading,
            ]
        )


def range_to_go(
    drag_profile: Callable[[float], float],
    energy: float,
    energy_final: float,
    n_nodes: int = 201,
) -> float:
    """Downrange still available (m) from the reference drag profile.

    Evaluates :math:`\\int_{e_f}^{e} \\mathrm{d}e / D(e)` by Simpson
    quadrature. The shallow-glide approximation :math:`\\cos\\gamma \\approx
    1` is used, which is worth stating: on a glide holding
    :math:`|\\gamma| < 5^\\circ` it costs under 0.4%, and it is the whole
    reason a drag profile can be treated as a range command in the first
    place.

    Parameters
    ----------
    drag_profile:
        Callable mapping specific energy (J/kg) to commanded drag
        deceleration (m/s²). Must be strictly positive over the interval;
        a zero would make the integral diverge, which is the correct
        answer to "how far can I fly with no drag" and a useless one to
        act on.
    energy, energy_final:
        Current and terminal specific energy (J/kg). ``energy`` must
        exceed ``energy_final`` — energy is depleted along a glide.
    """
    e0 = float(energy)
    ef = float(energy_final)
    if not (np.isfinite(e0) and np.isfinite(ef)):
        raise ValueError("energy and energy_final must both be finite")
    if e0 <= ef:
        raise ValueError(
            f"energy must exceed energy_final: a glide depletes energy, so "
            f"there is no range to go from e={e0:.6g} down to ef={ef:.6g}"
        )
    if n_nodes < 3 or n_nodes % 2 == 0:
        raise ValueError(f"n_nodes must be odd and >= 3, got {n_nodes}")

    nodes = np.linspace(ef, e0, n_nodes)
    drag = np.array([float(drag_profile(e)) for e in nodes])
    if not np.all(np.isfinite(drag)) or np.any(drag <= 0.0):
        raise ValueError(
            "drag_profile must be finite and strictly positive over the "
            "energy interval; a non-positive drag makes the range integral "
            "diverge"
        )
    return float(scipy.integrate.simpson(1.0 / drag, x=nodes))


def crossrange_deadband(speed: float, high: float, low: float) -> float:
    """Heading-error deadband (rad), scheduled on speed.

    Linear in speed between a wide value at entry interface and a tight
    value at handover. Wide early is what keeps the reversal count low
    while there is still plenty of energy to correct with; tight late is
    what actually delivers the terminal crossrange.

    Parameters
    ----------
    speed:
        Current speed (m/s).
    high, low:
        Deadband (rad) at 7000 m/s and at 1000 m/s respectively. ``high``
        must exceed ``low``, since scheduling the other way round would
        reverse constantly at high speed and then stop correcting exactly
        when correction is cheapest to act on.
    """
    if not (np.isfinite(high) and np.isfinite(low)):
        raise ValueError("high and low deadbands must be finite")
    if not (0.0 < low < high):
        raise ValueError(
            f"require 0 < low < high; a deadband that tightens with speed "
            f"reverses continually at entry and stops correcting at "
            f"handover. Got high={high}, low={low}"
        )
    fraction = np.clip((float(speed) - 1000.0) / 6000.0, 0.0, 1.0)
    return float(low + (high - low) * fraction)


def bank_reversal_needed(heading_error: float, speed: float, high: float, low: float) -> bool:
    """Whether the bank sign should flip now.

    ``heading_error`` is the signed angle (rad) from the current heading to
    the bearing of the target, wrapped to :math:`(-\\pi, \\pi]`.
    """
    if not np.isfinite(heading_error):
        raise ValueError("heading_error must be finite")
    return abs(float(heading_error)) > crossrange_deadband(speed, high, low)


@dataclass(frozen=True)
class DragTracker:
    """Proportional-derivative tracker on a reference drag profile.

    The command is the *cosine* of bank angle, because that is what enters
    the vertical force balance linearly:

    .. math::

        \\cos\\sigma_{\\mathrm{cmd}} = \\cos\\sigma_{\\mathrm{ref}}
          + k_p (D - D_{\\mathrm{ref}}) + k_d (\\dot D - \\dot D_{\\mathrm{ref}}).

    Flying *above* the reference drag means too much deceleration, so the
    correction must reduce vertical lift — hence the **positive** sign on
    :math:`k_p`, which reads backwards until one notices that increasing
    :math:`\\cos\\sigma` raises the vehicle and thins the air it is in.

    Attributes
    ----------
    gain_proportional, gain_derivative:
        On drag error (m/s²) and drag-rate error (m/s³).
    reference_cosine:
        Nominal :math:`\\cos\\sigma` about which the loop closes.
    """

    gain_proportional: float = 0.02
    gain_derivative: float = 0.5
    reference_cosine: float = 0.5

    def __post_init__(self) -> None:
        for name in ("gain_proportional", "gain_derivative", "reference_cosine"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 < self.reference_cosine <= 1.0:
            raise ValueError(f"reference_cosine must lie in (0, 1], got {self.reference_cosine}")

    def command(
        self,
        drag: float,
        drag_reference: float,
        drag_rate: float = 0.0,
        drag_rate_reference: float = 0.0,
        max_bank: float = np.deg2rad(80.0),
    ) -> float:
        """Commanded bank magnitude (rad), saturated to ``max_bank``.

        Saturation is on the *bank angle*, not on the cosine, so that a
        saturated command is still a physically meaningful attitude rather
        than an out-of-range cosine silently clipped to one.
        """
        correction = self.gain_proportional * (drag - drag_reference)
        correction += self.gain_derivative * (drag_rate - drag_rate_reference)
        cosine = np.clip(self.reference_cosine + correction, -1.0, 1.0)
        bank = float(np.arccos(cosine))
        return float(np.clip(bank, 0.0, max_bank))


@dataclass(frozen=True)
class GlideResult:
    """Outcome of a closed-loop glide."""

    times: _FloatArray
    states: _FloatArray
    """Shape ``(6, n)``: radius, longitude, latitude, speed, gamma, heading."""
    bank: _FloatArray
    reversals: int
    terminal_speed: float
    downrange: float
    """Great-circle arc flown (m)."""
    crossrange: float
    """Signed great-circle offset from the initial heading's great circle (m)."""

    @property
    def altitudes(self) -> _FloatArray:
        return np.asarray(self.states[0] - _R_EARTH)


def _great_circle(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Central angle (rad) between two points, by the haversine form."""
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def simulate_glide(
    vehicle: EntryVehicle,
    initial: GlideState,
    drag_reference: Callable[[float], float],
    tracker: DragTracker | None = None,
    target: tuple[float, float] | None = None,
    terminal_speed: float = 1000.0,
    max_time: float = 3000.0,
    deadband_high: float = np.deg2rad(12.0),
    deadband_low: float = np.deg2rad(2.0),
    n_output: int = 601,
) -> GlideResult:
    """Fly a glide with the drag tracker and bank-reversal logic closed.

    Integration terminates on ``terminal_speed``, on reaching the surface,
    or on ``max_time``, whichever comes first. Which one fired is visible
    from the returned state rather than being reported separately.

    Parameters
    ----------
    drag_reference:
        Callable mapping specific energy (J/kg) to commanded drag (m/s²).
    target:
        ``(longitude, latitude)`` in radians. Supplying it enables bank
        reversals; without a target the bank sign is held and the result
        shows the uncorrected crossrange drift, which is the useful
        baseline to compare against.

    Notes
    -----
    Bank *sign* is a discrete state, so the right-hand side is not smooth
    across a reversal. Rather than let an adaptive integrator chatter on
    that discontinuity, the sign is held fixed within each integration
    segment and reversals are applied between segments on a fixed guidance
    cycle. That is also what a real flight computer does.
    """
    if tracker is None:
        tracker = DragTracker()
    if not (np.isfinite(terminal_speed) and terminal_speed > 0.0):
        raise ValueError(f"terminal_speed must be finite and > 0, got {terminal_speed}")
    if not (np.isfinite(max_time) and max_time > 0.0):
        raise ValueError(f"max_time must be finite and > 0, got {max_time}")
    if initial.speed <= terminal_speed:
        raise ValueError(
            f"initial speed {initial.speed:.6g} m/s is already at or below the "
            f"terminal speed {terminal_speed:.6g} m/s; there is no glide to fly"
        )

    guidance_period = 1.0
    sign = 1.0
    reversals = 0
    state = initial.as_array()
    clock = 0.0
    times = [0.0]
    history = [state.copy()]
    banks: list[float] = []

    def derivatives(_t: float, y: _FloatArray, bank: float) -> _FloatArray:
        radius, _lon, lat, speed, gamma, heading = y
        altitude = radius - _R_EARTH
        drag = vehicle.drag_acceleration(altitude, speed)
        lift = vehicle.lift_to_drag * drag
        gravity = _MU / radius**2
        return np.array(
            [
                speed * np.sin(gamma),
                speed * np.cos(gamma) * np.sin(heading) / (radius * np.cos(lat)),
                speed * np.cos(gamma) * np.cos(heading) / radius,
                -drag - gravity * np.sin(gamma),
                (lift * np.cos(bank) + (speed**2 / radius - gravity) * np.cos(gamma)) / speed,
                (
                    lift * np.sin(bank) / np.cos(gamma)
                    + speed**2 * np.cos(gamma) * np.sin(heading) * np.tan(lat) / radius
                )
                / speed,
            ]
        )

    while clock < max_time and state[3] > terminal_speed and state[0] > _R_EARTH:
        radius, lon, lat, speed, _gamma, heading = state
        energy = 0.5 * speed**2 - _MU / radius
        drag = vehicle.drag_acceleration(radius - _R_EARTH, speed)
        magnitude = tracker.command(drag, float(drag_reference(energy)), max_bank=vehicle.max_bank)
        if target is not None:
            bearing = _bearing(lon, lat, target[0], target[1])
            error = _wrap(bearing - heading)
            if bank_reversal_needed(error, speed, deadband_high, deadband_low):
                desired = float(np.sign(error)) or 1.0
                if desired != sign:
                    sign = desired
                    reversals += 1
        bank = sign * magnitude
        banks.append(bank)

        span = (clock, min(clock + guidance_period, max_time))
        solution = scipy.integrate.solve_ivp(
            derivatives,
            span,
            state,
            args=(bank,),
            rtol=1e-10,
            atol=1e-9,
            dense_output=False,
        )
        if not solution.success:  # pragma: no cover - integrator failure
            raise RuntimeError(f"glide integration failed: {solution.message}")
        state = np.asarray(solution.y[:, -1])
        clock = float(solution.t[-1])
        times.append(clock)
        history.append(state.copy())

    trajectory = np.asarray(history).T
    stacked = np.asarray(times)
    if n_output >= 3 and stacked.size > 2:
        sampled = np.linspace(stacked[0], stacked[-1], n_output)
        trajectory = np.array([np.interp(sampled, stacked, row) for row in trajectory])
        stacked = sampled

    downrange = _R_EARTH * _great_circle(
        initial.longitude, initial.latitude, float(trajectory[1, -1]), float(trajectory[2, -1])
    )
    crossrange = _crossrange(initial, float(trajectory[1, -1]), float(trajectory[2, -1]))
    return GlideResult(
        times=stacked,
        states=trajectory,
        bank=np.asarray(banks),
        reversals=reversals,
        terminal_speed=float(trajectory[3, -1]),
        downrange=downrange,
        crossrange=crossrange,
    )


def _wrap(angle: float) -> float:
    """Wrap to ``(-pi, pi]``."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _bearing(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Initial great-circle bearing (rad) from north, positive toward east."""
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return float(np.arctan2(y, x))


def _crossrange(initial: GlideState, longitude: float, latitude: float) -> float:
    """Signed offset (m) from the great circle of the initial heading.

    Positive to the right of the initial heading. This is the quantity the
    bank-reversal logic exists to bound, so it is reported directly rather
    than left to be inferred from the terminal position.
    """
    arc = _great_circle(initial.longitude, initial.latitude, longitude, latitude)
    if arc == 0.0:
        return 0.0
    bearing = _bearing(initial.longitude, initial.latitude, longitude, latitude)
    return float(_R_EARTH * np.arcsin(np.sin(arc) * np.sin(bearing - initial.heading)))
