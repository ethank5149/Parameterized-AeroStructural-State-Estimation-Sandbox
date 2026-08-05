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

Tracking drag is not controlling range
--------------------------------------

The relation above makes a drag profile *equivalent* to a range, but only
if the vehicle flies the profile it was given from the state it was
expected to be in. It will not be. Measured over forty closed-loop glides
with a 900 m entry-interface position error and 3% ballistic-coefficient
scatter, tracking the reference drag profile alone gives a terminal
dispersion of **34 km downrange and 35 km crossrange** — a fortyfold
amplification of the entry error, because nothing in the loop ever notices
the range error accumulating. The tracker flies its profile faithfully and
lands wherever that profile happens to reach.

Closing an outer loop on range fixes most of it, and is what entry
guidance in the Shuttle lineage does: predict the range the current
profile will fly with :func:`range_to_go`, compare against the distance
actually remaining, and scale the reference drag to null the difference.
``range_gain`` on :func:`simulate_glide` enables it. At gain 20 the same
Monte Carlo gives **11.3 km downrange and 6.7 km crossrange**, a factor of
four better, with crossrange improving fivefold. The residual is dominated
by downrange and does not shrink much further, because the glide is
terminated by a *speed* gate rather than by arrival: even perfect range
control cannot help once the vehicle has run out of energy at the wrong
place. Handing over to terminal guidance on a range-and-energy condition
rather than on speed alone is the next structural fix, and it is not made
here.

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

Two things learned by building this, both of which cost a failed
verification run first
----------------------------------------------------------------------

**A reference drag profile has to be flyable, and constant drag is not.**
The equilibrium glide supports only about 2 m/s² at entry interface,
because near orbital speed centrifugal relief cancels most of the
vehicle's weight; it rises to roughly 10 m/s² by handover. Commanding a
constant 20 m/s² is therefore not a demanding profile but an impossible
one over the first half of the glide, and the tracker responds by sitting
against its bank stop. Built from :func:`equilibrium_glide_profile`
instead, predicted and flown range agree to 6% at moderate bank.

**The two channels are independent in mechanism but not in use.** Bank
magnitude sets range and bank sign sets crossrange, and neither enters the
other's law. But the lateral logic steers on *bearing to the target*, so a
longitudinal profile that overflies inverts that bearing part-way through
the glide and the deadband stops meaning what it should. With the target
at the delivered range, reversals cut terminal crossrange by 39x; with it
1000 km short, by 4x. Range matching is a precondition for the lateral
channel rather than a separate concern.
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
    "crossrange_deadband_from_range",
    "equilibrium_glide_drag",
    "equilibrium_glide_profile",
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


def equilibrium_glide_drag(
    vehicle: EntryVehicle, speed: float, radius: float, bank: float
) -> float:
    """Drag deceleration (m/s²) of the equilibrium glide at this state.

    Setting :math:`\\gamma \\approx 0` and :math:`\\dot\\gamma \\approx 0`
    in the vertical force balance leaves the lift supporting the residual
    of weight over centrifugal relief,

    .. math::

        \\frac{L}{D} D \\cos\\sigma = g - \\frac{V^2}{r},

    so the flyable drag is fixed by speed, altitude and bank alone.

    **This is the constraint that makes a reference profile feasible, and
    ignoring it is a good way to build a tracker that cannot possibly
    work.** At entry interface the term in parentheses is small — the
    vehicle is near orbital speed and centrifugal relief cancels most of
    its weight — so the equilibrium drag is only about 2 m/s². It rises
    towards roughly 10 m/s² by the time the vehicle has slowed to
    handover. A constant-drag reference of, say, 20 m/s² is therefore not
    a demanding command but an *impossible* one over the first half of the
    glide, and a tracker asked to hold it will simply sit against its bank
    stop. :func:`equilibrium_glide_profile` builds a reference that
    respects this.
    """
    gravity = _MU / radius**2
    relief = speed**2 / radius
    authority = vehicle.lift_to_drag * float(np.cos(bank))
    # Guarding on a strict sign is not enough: cos(pi/2) evaluates to 6e-17
    # rather than to zero, which sails past `authority <= 0` and returns a
    # drag of 5e16 m/s^2 instead of refusing. The threshold is relative to
    # L/D so it means "a thousandth of the vehicle's lift", not an absolute
    # number that would be wrong for a different vehicle.
    if authority <= 1e-3 * vehicle.lift_to_drag:
        raise ValueError(
            f"bank of {np.rad2deg(bank):.3f} deg leaves no usable vertical "
            f"lift, so no equilibrium glide exists there"
        )
    return float(max(gravity - relief, 0.0) / authority)


def equilibrium_glide_profile(
    vehicle: EntryVehicle, radius: float, bank: float
) -> Callable[[float], float]:
    """A flyable reference drag profile in specific energy.

    Returns :math:`D(e)` obtained by inverting :math:`e = V^2/2 - \\mu/r`
    for speed at fixed reference radius and evaluating
    :func:`equilibrium_glide_drag`. Holding the radius fixed is an
    approximation — a real glide descends — but it is the right one here:
    it makes the profile a pure function of energy, which is what
    :func:`range_to_go` integrates over, and the residual is absorbed by
    the tracker.

    ``bank`` is the nominal bank the profile is built for. Commanding a
    profile built at a *larger* bank asks the vehicle to fly deeper and
    shorter, which is how range is traded.
    """

    def profile(energy: float) -> float:
        speed_squared = 2.0 * (float(energy) + _MU / radius)
        speed = float(np.sqrt(max(speed_squared, 0.0)))
        return equilibrium_glide_drag(vehicle, speed, radius, bank)

    return profile


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


def crossrange_deadband_from_range(
    range_to_go: float, allowed_miss: float, high: float, low: float
) -> float:
    """Heading-error deadband (rad) that bounds the *crossrange miss*.

    Scheduling on speed is the traditional choice and it has a defect that
    only shows once the glide hands over on range rather than on speed: the
    schedule spans 7000 to 1000 m/s, the vehicle hands over around 1700,
    and the deadband is therefore still 3.2 degrees when the flight ends.
    At 150 km of range-to-go that is 8.4 km of crossrange — which is, to
    within a few percent, exactly the crossrange dispersion measured.

    A deadband on heading error is the wrong invariant. What matters is the
    lateral distance that error becomes, and that shrinks with range:

    .. math::

        \\delta(R) = \\arctan\\frac{\\Delta_{\\text{allowed}}}{R},

    clamped to ``[low, high]``. Note which way this runs: it is **tight far
    out and permissive close in**, the opposite of the speed schedule and
    the opposite of intuition. That is correct, because a given heading
    error becomes a larger miss the further away you are. Holding heading
    tightly through the long middle of the glide is what leaves crossrange
    already small at handover; relaxing near the end costs little, because
    little lateral distance can accumulate in the range that remains.
    """
    remaining = float(range_to_go)
    allowed = float(allowed_miss)
    if not (np.isfinite(remaining) and remaining > 0.0):
        raise ValueError(f"range_to_go must be finite and > 0, got {remaining}")
    if not (np.isfinite(allowed) and allowed > 0.0):
        raise ValueError(f"allowed_miss must be finite and > 0, got {allowed}")
    if not (0.0 < low < high):
        raise ValueError(f"require 0 < low < high, got high={high}, low={low}")
    return float(np.clip(np.arctan(allowed / remaining), low, high))


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

    **Integral action is not optional here.** Proportional control alone
    leaves a steady-state error against the persistent disturbance that
    gravity and the thinning atmosphere present: measured without it, a
    commanded 12 m/s² was flown at 7.5 and a commanded 40 at 16.2. The
    ordering survived — more commanded drag still meant less range — but
    the *magnitudes* did not, which would make the range prediction of
    :func:`range_to_go` useless as anything but a trend. The integral term
    removes that offset wherever the bank command is not saturated.

    Attributes
    ----------
    gain_proportional, gain_derivative:
        On drag error (m/s²) and drag-rate error (m/s³).
    gain_integral:
        On accumulated drag error (m/s² · s). The loop is stateless, so the
        accumulator is owned and anti-windup limited by the caller —
        :func:`simulate_glide` does this — which keeps the controller pure
        and puts the reset logic where the saturation is known.
    reference_cosine:
        Nominal :math:`\\cos\\sigma` about which the loop closes.
    """

    gain_proportional: float = 0.02
    gain_derivative: float = 0.5
    gain_integral: float = 0.004
    reference_cosine: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "gain_proportional",
            "gain_derivative",
            "gain_integral",
            "reference_cosine",
        ):
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
        integral: float = 0.0,
    ) -> float:
        """Commanded bank magnitude (rad), saturated to ``max_bank``.

        Saturation is on the *bank angle*, not on the cosine, so that a
        saturated command is still a physically meaningful attitude rather
        than an out-of-range cosine silently clipped to one.
        """
        correction = self.gain_proportional * (drag - drag_reference)
        correction += self.gain_derivative * (drag_rate - drag_rate_reference)
        correction += self.gain_integral * integral
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
    crossrange_tolerance: float | None = None,
    roll_rate_limit: float | None = None,
    range_gain: float = 0.0,
    terminal_energy: float | None = None,
    handover_range: float | None = None,
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
    if not (np.isfinite(range_gain) and range_gain >= 0.0):
        raise ValueError(f"range_gain must be finite and >= 0, got {range_gain}")
    if range_gain > 0.0 and target is None:
        raise ValueError("range closure needs a target to measure range-to-go against")
    if not (np.isfinite(terminal_speed) and terminal_speed > 0.0):
        raise ValueError(f"terminal_speed must be finite and > 0, got {terminal_speed}")
    if not (np.isfinite(max_time) and max_time > 0.0):
        raise ValueError(f"max_time must be finite and > 0, got {max_time}")
    if initial.speed <= terminal_speed:
        raise ValueError(
            f"initial speed {initial.speed:.6g} m/s is already at or below the "
            f"terminal speed {terminal_speed:.6g} m/s; there is no glide to fly"
        )

    if roll_rate_limit is not None and not (np.isfinite(roll_rate_limit) and roll_rate_limit > 0.0):
        raise ValueError(
            f"roll_rate_limit must be finite and > 0 when given, got {roll_rate_limit}"
        )
    guidance_period = 1.0
    sign = 1.0
    # Actual bank, as distinct from commanded. With an unlimited roll rate
    # the two are identical and a reversal is free. They are not: a
    # reversal sweeps the bank angle through zero, where the vertical lift
    # component is *maximum*, so the vehicle lofts every time it reverses.
    # The faster the roll, the briefer that excursion.
    actual_bank = 0.0
    reversals = 0
    integral = 0.0
    # Anti-windup bound. Chosen so the integral term alone can swing the
    # commanded cosine over its full range and no further: without it the
    # accumulator keeps growing while the bank is saturated and then takes
    # a large part of the glide to unwind, which shows up as a slow
    # oscillation rather than as the offset it was meant to remove.
    integral_limit = 2.0 / max(tracker.gain_integral, 1e-12)
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

    def _still_flying() -> bool:
        """Whether the glide continues.

        Terminating on speed alone ends the flight wherever the vehicle
        happens to have got to, which is why it leaves a large downrange
        dispersion however well the range loop is tuned: the stopping
        condition is defined on the *vehicle's* state and not on its
        relationship to the target. Handing over when range-to-go reaches
        the terminal acquisition range instead defines the condition
        relative to the target, so the arrival point is the handover point
        by construction. The speed gate is retained underneath it as a
        floor, because a vehicle out of energy must stop regardless.
        """
        if state[3] <= terminal_speed or state[0] <= _R_EARTH:
            return False
        if handover_range is not None and target is not None:
            remaining = _R_EARTH * _great_circle(
                float(state[1]), float(state[2]), target[0], target[1]
            )
            if remaining <= handover_range:
                return False
        return True

    if handover_range is not None:
        if target is None:
            raise ValueError("a handover range needs a target to measure range-to-go against")
        if not (np.isfinite(handover_range) and handover_range > 0.0):
            raise ValueError(f"handover_range must be finite and > 0, got {handover_range}")

    while clock < max_time and _still_flying():
        radius, lon, lat, speed, _gamma, heading = state
        energy = 0.5 * speed**2 - _MU / radius
        drag = vehicle.drag_acceleration(radius - _R_EARTH, speed)
        commanded = float(drag_reference(energy))
        # Outer range loop. Tracking a drag profile alone controls *drag*,
        # not range: the vehicle flies the commanded profile faithfully and
        # lands wherever that profile happens to reach. Measured, a 900 m
        # entry-interface error then becomes a 49 km terminal scatter,
        # because nothing in the loop notices the range error accumulating.
        #
        # Closing the loop the way entry guidance in the Shuttle lineage
        # does: predict the range the current profile will fly with
        # `range_to_go`, compare it with the great-circle distance actually
        # remaining, and scale the reference drag to null the difference.
        # Commanding more drag shortens the flight, hence the sign.
        if range_gain > 0.0 and target is not None:
            final_energy = (
                0.5 * terminal_speed**2 - _MU / radius
                if terminal_energy is None
                else terminal_energy
            )
            if energy > final_energy:
                predicted = range_to_go(drag_reference, energy, final_energy)
                remaining = _R_EARTH * _great_circle(lon, lat, target[0], target[1])
                if remaining > 0.0:
                    error = (predicted - remaining) / remaining
                    commanded *= float(np.clip(1.0 + range_gain * error, 0.25, 4.0))
        magnitude = tracker.command(drag, commanded, max_bank=vehicle.max_bank, integral=integral)
        # Anti-windup: stop accumulating while the command is against a
        # stop. Without this the accumulator keeps growing through the
        # saturated stretch and then takes a large part of the glide to
        # unwind, which appears as a slow oscillation rather than as the
        # offset the term was added to remove.
        if 0.0 < magnitude < vehicle.max_bank:
            integral = float(
                np.clip(
                    integral + (drag - commanded) * guidance_period,
                    -integral_limit,
                    integral_limit,
                )
            )
        if target is not None:
            bearing = _bearing(lon, lat, target[0], target[1])
            error = _wrap(bearing - heading)
            if crossrange_tolerance is None:
                deadband = crossrange_deadband(speed, deadband_high, deadband_low)
            else:
                deadband = crossrange_deadband_from_range(
                    max(
                        _R_EARTH * _great_circle(lon, lat, target[0], target[1]),
                        1.0,
                    ),
                    crossrange_tolerance,
                    deadband_high,
                    deadband_low,
                )
            if abs(error) > deadband:
                desired = float(np.sign(error)) or 1.0
                if desired != sign:
                    sign = desired
                    reversals += 1
        commanded_bank = sign * magnitude
        if roll_rate_limit is None:
            actual_bank = commanded_bank
        else:
            step_limit = roll_rate_limit * guidance_period
            delta = float(np.clip(commanded_bank - actual_bank, -step_limit, step_limit))
            actual_bank = actual_bank + delta
        bank = actual_bank
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
