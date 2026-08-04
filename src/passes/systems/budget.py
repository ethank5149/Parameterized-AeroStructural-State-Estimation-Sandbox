"""End-to-end budgets: what an architecture costs, and whether it closes.

:mod:`passes.systems.architecture` says which phase sequences are
admissible. That is a statement about composability and says nothing about
whether a given sequence can actually reach a given target from a given
launch site. This module answers that, by charging each phase its range,
its :math:`\\Delta v` and its time, and then checking whether the pieces
add up to the geometry the mission asks for.

The closure question, and why one leg is special
------------------------------------------------

Most legs contribute a range fixed by the vehicle and the phase: a deorbit
transfer sweeps an arc set by how deep perigee goes, a glide covers a
distance set by its drag profile, a cruise by its fuel. Those are
*computed*. What makes an architecture close is the one remaining leg,
which absorbs whatever is left over:

* **Fractional-orbital profiles** absorb it in the **parking arc**. The
  vehicle simply stays in orbit longer, which costs time and no propellant
  at all. This is the structural reason the profile is flexible about
  range — and it is why the closure test for these architectures is
  whether the remainder is non-negative and inside one revolution, not
  whether it is affordable.

* **Suborbital profiles** have no such leg, so the remainder falls on
  **boost**, and boost range is bought with propellant. Here the closure
  test is against a stated boost capability, and an architecture that
  overshoots does not merely cost more — it is infeasible for that booster.

Treating those two cases identically is the most common way to get a range
budget wrong, so :class:`MissionBudget` names which leg took up the slack
and how much.

What this does not do
---------------------

It does not optimise. Every leg is evaluated at the parameters it is
given, and sweeping those parameters to find a good design is the caller's
job — :func:`evaluate` is the objective function, not the optimiser.
Dispersion is propagated where a model exists for it (bus dispensing) and
reported as ``nan`` where one does not, rather than being assumed zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from passes.geodesy import (
    WGS84_MEAN_RADIUS,
    GeodeticPosition,
    great_circle_range,
)
from passes.guidance.cruise import CruiseVehicle, cruise_range
from passes.guidance.entry import EntryVehicle
from passes.orbital.fobs import deorbit_burn
from passes.orbital.gravity import EARTH
from passes.systems.architecture import Architecture, Phase
from passes.systems.dispersion import AccuracyStatistics, accuracy_statistics

__all__ = [
    "DISPERSION_SOURCES",
    "LegBudget",
    "MissionBudget",
    "MissionRequest",
    "evaluate",
]

_FloatArray = NDArray[np.float64]

#: Ground range a ballistic reentry covers between the entry interface and
#: impact. Short compared with any other leg, and roughly geometry-fixed
#: for a steep entry, so it is a constant here rather than a model.
_BALLISTIC_ENTRY_RANGE = 300.0e3
#: Terminal homing is a correction, not a transport leg.
_TERMINAL_RANGE = 0.0

#: Representative one-sigma error contributions (m), downrange and
#: crossrange, keyed by the phase that produces them.
#:
#: **These are parametric inputs, not derived results**, and the defaults
#: are order-of-magnitude figures chosen to exercise the accounting. The
#: one exception is dispensing, which is scaled from the bus execution
#: model. Anyone quoting a CEP from this module is quoting these numbers
#: plus arithmetic, and the honest thing is to say so rather than let a
#: computed-looking figure imply a computed provenance.
DISPERSION_SOURCES: dict[Phase, tuple[float, float]] = {
    Phase.BOOST: (1200.0, 900.0),
    Phase.MIDCOURSE: (-0.90, -0.90),
    Phase.DISPENSE: (400.0, 250.0),
    Phase.DEORBIT: (600.0, 300.0),
    Phase.GLIDE: (800.0, 250.0),
    Phase.CRUISE: (500.0, 500.0),
    Phase.BALLISTIC: (350.0, 350.0),
    Phase.TERMINAL: (-0.85, -0.85),
}


@dataclass(frozen=True)
class MissionRequest:
    """Where the mission starts, where it must reach, and when.

    Both ends are given in the universal format of :mod:`passes.geodesy`,
    so a request is stated the way a problem actually arrives rather than
    as inertial vectors.

    Attributes
    ----------
    launch_site:
        Geodetic launch location.
    aimpoints:
        One or more geodetic targets. The *farthest* sets the range the
        architecture must close, since every body is dispensed from a
        common trajectory.
    arrival_time:
        Seconds from launch at which the aimpoints must be reached.
    """

    launch_site: GeodeticPosition
    aimpoints: tuple[GeodeticPosition, ...]
    arrival_time: float

    def __post_init__(self) -> None:
        if not self.aimpoints:
            raise ValueError("a mission request needs at least one aimpoint")
        if not (np.isfinite(self.arrival_time) and self.arrival_time > 0.0):
            raise ValueError(f"arrival_time must be finite and > 0, got {self.arrival_time}")

    @property
    def required_range(self) -> float:
        """Great-circle range (m) to the farthest aimpoint."""
        return max(great_circle_range(self.launch_site, target) for target in self.aimpoints)

    @property
    def aimpoint_spread(self) -> float:
        """Largest separation (m) between any two aimpoints.

        What the dispensing leg has to cover, and therefore what sizes the
        bus manoeuvres.
        """
        if len(self.aimpoints) < 2:
            return 0.0
        return max(
            great_circle_range(a, b)
            for i, a in enumerate(self.aimpoints)
            for b in self.aimpoints[i + 1 :]
        )


@dataclass(frozen=True)
class LegBudget:
    """One phase's contribution to the mission."""

    phase: Phase
    ground_range: float
    """Downrange contributed (m). Zero for legs that do not transport."""
    delta_v: float
    """Propellant cost (m/s)."""
    duration: float
    """Elapsed time (s), ``nan`` where the phase has no independent clock."""
    note: str = ""
    is_slack: bool = False
    """Whether this leg absorbed the remaining range rather than setting it."""


@dataclass(frozen=True)
class MissionBudget:
    """The full accounting, and whether it closes."""

    architecture: Architecture
    request: MissionRequest
    legs: tuple[LegBudget, ...] = field(repr=False)
    slack_phase: Phase
    """Which leg absorbed the remainder."""
    closes: bool
    shortfall: float
    """Magnitude (m) by which the accounting fails to close, zero if it does.

    Deliberately unsigned, because the two ways to fail are not opposite
    ends of one scale and reading them off a sign is how they get confused.
    :attr:`reason` says which one happened.
    """
    reason: str = ""
    """Why it does not close, empty when it does.

    An architecture can fail by not reaching far enough *or* by overshooting
    — a long glide added to a long deorbit transfer can carry the vehicle
    past its own target, and the remedy for that is a shorter glide or a
    steeper deorbit, not a bigger booster. Naming the failure is the
    difference between a number and a diagnosis.
    """
    accuracy: AccuracyStatistics | None = None
    """Terminal accuracy: principal sigmas, CEP and 95% radius.

    ``None`` only if the architecture contributes no dispersion at all,
    which cannot happen for any admissible sequence since boost always
    contributes.
    """

    @property
    def total_delta_v(self) -> float:
        return float(sum(leg.delta_v for leg in self.legs))

    @property
    def total_range(self) -> float:
        return float(sum(leg.ground_range for leg in self.legs))

    def summary(self) -> str:
        verdict = "closes" if self.closes else "DOES NOT CLOSE"
        return (
            f"{self.architecture.payload.value}: {verdict}, "
            f"{self.total_range / 1e3:.0f} km against "
            f"{self.request.required_range / 1e3:.0f} km required, "
            f"{self.total_delta_v:.0f} m/s, slack in {self.slack_phase.value}"
            + (
                f", CEP {self.accuracy.cep:.0f} m / R95 {self.accuracy.r95:.0f} m"
                if self.accuracy is not None
                else ""
            )
            + (f" — {self.reason}" if self.reason else "")
        )


def evaluate(
    architecture: Architecture,
    request: MissionRequest,
    entry_vehicle: EntryVehicle | None = None,
    cruise_vehicle: CruiseVehicle | None = None,
    glide_range: float = 6.0e6,
    parking_radius: float = EARTH.radius + 200e3,
    entry_radius: float = EARTH.radius + 100e3,
    perigee_radius: float = EARTH.radius - 400e3,
    boost_range: float = 2.0e6,
    boost_delta_v: float = 6.0e3,
    midcourse_delta_v: float = 30.0,
    dispense_delta_v_per_body: float = 150.0,
    cruise_speed: float = 8.0 * 295.0,
    cruise_specific_impulse: float = 1200.0,
) -> MissionBudget:
    """Charge each phase and test whether the architecture closes.

    Parameters
    ----------
    glide_range:
        Range (m) the glide leg is commanded to cover. This is an input
        rather than something derived, because it is set by the drag
        profile the guidance is given — see
        :func:`passes.guidance.entry.equilibrium_glide_profile` — and
        choosing it is a design decision, not a consequence.
    boost_range, boost_delta_v:
        The booster's capability. For a suborbital architecture these are
        what closure is tested against, since boost is the slack leg.

    Notes
    -----
    A mixed payload flies both a glide and a ballistic arc concurrently.
    Only the **longer** of the two is charged as transport, because the
    bodies separate and fly in parallel — adding them would double-count
    a distance the vehicle covers once.
    """
    legs: list[LegBudget] = []

    legs.append(
        LegBudget(
            phase=Phase.BOOST,
            ground_range=boost_range,
            delta_v=boost_delta_v,
            duration=float("nan"),
            note="stated booster capability",
        )
    )

    if Phase.MIDCOURSE in architecture.phases:
        legs.append(
            LegBudget(
                phase=Phase.MIDCOURSE,
                ground_range=0.0,
                delta_v=midcourse_delta_v,
                duration=float("nan"),
                note="correction, not transport",
            )
        )

    if Phase.DISPENSE in architecture.phases:
        bodies = max(len(request.aimpoints), 2)
        legs.append(
            LegBudget(
                phase=Phase.DISPENSE,
                ground_range=0.0,
                delta_v=dispense_delta_v_per_body * (bodies - 1),
                duration=float("nan"),
                note=f"{bodies} bodies, {bodies - 1} retargeting manoeuvres",
            )
        )

    if Phase.DEORBIT in architecture.phases:
        burn = deorbit_burn(parking_radius, entry_radius, perigee_radius)
        legs.append(
            LegBudget(
                phase=Phase.DEORBIT,
                ground_range=WGS84_MEAN_RADIUS * burn.transfer_angle,
                delta_v=burn.delta_v,
                duration=burn.transfer_time,
                note=(f"entry gamma {np.rad2deg(burn.entry_flight_path_angle):.2f} deg"),
            )
        )

    # Terminal regimes. A mixed payload flies two concurrently and only the
    # longer counts as transport.
    regimes = architecture.terminal_regimes
    atmospheric: list[LegBudget] = []
    if Phase.GLIDE in regimes:
        if entry_vehicle is None:
            raise ValueError(
                "this architecture glides, so an entry_vehicle is required to charge the glide leg"
            )
        atmospheric.append(
            LegBudget(
                phase=Phase.GLIDE,
                ground_range=float(glide_range),
                delta_v=0.0,
                duration=float("nan"),
                note=f"L/D {entry_vehicle.lift_to_drag:.1f}, unpowered",
            )
        )
    if Phase.CRUISE in regimes:
        if cruise_vehicle is None:
            raise ValueError(
                "this architecture cruises, so a cruise_vehicle is required "
                "to charge the cruise leg"
            )
        atmospheric.append(
            LegBudget(
                phase=Phase.CRUISE,
                ground_range=cruise_range(cruise_vehicle, cruise_speed, cruise_specific_impulse),
                delta_v=0.0,
                duration=float("nan"),
                note="Breguet, excludes acceleration and descent",
            )
        )
    if Phase.BALLISTIC in regimes:
        atmospheric.append(
            LegBudget(
                phase=Phase.BALLISTIC,
                ground_range=_BALLISTIC_ENTRY_RANGE,
                delta_v=0.0,
                duration=float("nan"),
                note="entry interface to impact",
            )
        )
    if len(atmospheric) > 1:
        longest = max(atmospheric, key=lambda leg: leg.ground_range)
        for leg in atmospheric:
            legs.append(
                leg
                if leg is longest
                else LegBudget(
                    phase=leg.phase,
                    ground_range=0.0,
                    delta_v=leg.delta_v,
                    duration=leg.duration,
                    note=(f"{leg.note}; flown concurrently, not charged as transport"),
                )
            )
    else:
        legs.extend(atmospheric)

    if Phase.TERMINAL in architecture.phases:
        legs.append(
            LegBudget(
                phase=Phase.TERMINAL,
                ground_range=_TERMINAL_RANGE,
                delta_v=0.0,
                duration=float("nan"),
                note="homing correction",
            )
        )

    # Closure. The slack leg differs by architecture and this is the whole
    # point of the accounting.
    fixed = sum(leg.ground_range for leg in legs)
    remainder = request.required_range - fixed

    if architecture.is_orbital:
        slack_phase = Phase.PARKING
        # Parking arc costs time, not propellant. It must be non-negative
        # and must fit inside one revolution — beyond that the profile is a
        # multi-revolution one, which the coast model is not built for.
        one_revolution = 2.0 * np.pi * WGS84_MEAN_RADIUS
        closes = 0.0 <= remainder <= one_revolution
        if remainder < 0.0:
            shortfall = -remainder
            reason = (
                f"overshoot: the fixed legs already cover "
                f"{fixed / 1e3:.0f} km against {request.required_range / 1e3:.0f} "
                f"km required, so no parking arc closes it. Shorten the glide "
                f"or deorbit more steeply"
            )
        elif remainder > one_revolution:
            shortfall = remainder - one_revolution
            reason = (
                f"the remainder needs {remainder / one_revolution:.2f} "
                f"revolutions of parking arc; this is a multi-revolution "
                f"profile, which the coast model is not built for"
            )
        else:
            shortfall = 0.0
            reason = ""
        legs.insert(
            1,
            LegBudget(
                phase=Phase.PARKING,
                ground_range=max(remainder, 0.0),
                delta_v=0.0,
                duration=float("nan"),
                note="absorbs the remainder; costs time, not propellant",
                is_slack=True,
            ),
        )
    else:
        slack_phase = Phase.BOOST
        # Boost range is bought with propellant, so the remainder is tested
        # against the booster's stated capability rather than merely being
        # required non-negative.
        closes = remainder <= 0.0
        shortfall = max(0.0, remainder)
        reason = (
            ""
            if closes
            else (
                f"boost is the only slack leg and it is short by "
                f"{remainder / 1e3:.0f} km; suborbital profiles buy range "
                f"with propellant, so this is infeasible for the stated "
                f"booster rather than merely expensive"
            )
        )
        legs[0] = LegBudget(
            phase=Phase.BOOST,
            ground_range=boost_range,
            delta_v=boost_delta_v,
            duration=float("nan"),
            note=(
                f"stated capability; {remainder / 1e3:+.0f} km of range "
                f"{'still needed' if remainder > 0 else 'to spare'}"
            ),
            is_slack=True,
        )

    # Error budget. Reducing phases (guidance) multiply what is already
    # accumulated; contributing phases add in quadrature. Order matters,
    # so the ledger is walked in phase order rather than summed blindly:
    # a correction cannot remove an error injected after it.
    down = 0.0
    cross = 0.0
    for leg in sorted(legs, key=lambda item: architecture.phases.index(item.phase)):
        source = DISPERSION_SOURCES.get(leg.phase)
        if source is None:
            continue
        d, c = source
        if d < 0.0:
            down *= -d
            cross *= -c
        else:
            down = float(np.hypot(down, d))
            cross = float(np.hypot(cross, c))
    accuracy = accuracy_statistics(down, cross) if max(down, cross) > 0.0 else None

    return MissionBudget(
        architecture=architecture,
        request=request,
        legs=tuple(legs),
        slack_phase=slack_phase,
        closes=closes,
        shortfall=float(shortfall),
        reason=reason,
        accuracy=accuracy,
    )
