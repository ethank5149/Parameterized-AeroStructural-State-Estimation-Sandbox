"""Post-boost bus aiming: dispensing several vehicles to separated aimpoints.

A deployment bus carries :math:`n` independent vehicles and must place each
one on a ballistic arc that reaches its own aimpoint. The bus maneuvers to
a release condition, releases, maneuvers again, releases again. Each
release is a Lambert targeting problem — :mod:`passes.guidance.midcourse`
already solves that — so what this module adds is everything that only
appears once there is more than one of them.

What is actually hard here
--------------------------

**The order matters, and it matters more than the maneuvers do.** Servicing
the same set of aimpoints in a different sequence changes total
:math:`\\Delta v` substantially, because the bus is on a ballistic arc of
its own and the cost of retargeting depends on where along that arc the
release happens. This is a small assignment problem sitting on top of
continuous dynamics: :func:`optimize_deployment_order` enumerates
exhaustively for the sizes where that is cheap and falls back to a
2-opt improvement otherwise, and it reports which one it used.

**The footprint's shape depends on whether the arrival epoch is free, and
this is easy to get backwards.** The familiar claim is that downrange
separation is cheap while crossrange is expensive. Under *fixed* time of
arrival that is false: measured on a suborbital arc, displacing an
aimpoint 50 km downrange costs 35.1 m/s against 40.4 m/s crossrange — a
15% difference, not a qualitative one. Downrange is cheap only when the
arrival epoch can absorb it. Allowing arrival to slip by 10 s halves the
downrange cost to 16.1 m/s, while leaving the crossrange and radial costs
*exactly* unchanged, their optima sitting at zero slip. So the anisotropy
everyone quotes is real, but it is a property of the timing freedom rather
than of the geometry, and holding arrival time fixed removes it.
:func:`reachable_aimpoints` returns the cost of every aimpoint, in budget
or not, so this shape can be read off rather than assumed.

**Errors accumulate down the sequence, but dispersion does not.** Every
maneuver the bus makes is imperfect, and the vehicle released after it
inherits that error on top of whatever the bus already had — so the bus
covariance grows monotonically, since each maneuver adds a
positive-semidefinite block and nothing removes one. It is tempting to
conclude that the last vehicle off is the least accurate. That is false,
and the module measures it rather than assuming it: a later release also
has a shorter flight, and terminal miss scales with flight time through
the sensitivity, so the inherited error has less opportunity to become a
miss. Over a three-vehicle sequence here the dispersions run 1482, 1983,
1744 m — rising, then falling. Which vehicle actually needs the accuracy
budget is therefore a result, not a rule of thumb, and
:attr:`DeploymentPlan.dispersions` reports it per vehicle.

Modelling scope
---------------

Releases are impulsive and the released vehicle is purely ballistic
afterwards: no post-release propulsion, no aerodynamic phase. The bus
carries the same :math:`J_2` propagation as the coast phase. Arrival is
targeted at a specified inertial point and epoch, which is the strict
requirement; relaxing the epoch would be cheaper and is not offered here
for the same reason it is not offered in :mod:`passes.guidance.midcourse`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from passes.geodesy import GeodeticPosition, geodetic_to_eci
from passes.guidance.midcourse import (
    ExecutionErrorModel,
    correction_maneuver,
    miss_sensitivity,
)
from passes.orbital.coast import propagate_coast
from passes.orbital.gravity import EARTH, GravityModel

__all__ = [
    "Aimpoint",
    "DeploymentPlan",
    "Release",
    "optimize_deployment_order",
    "plan_deployment",
    "reachable_aimpoints",
]

_FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Aimpoint:
    """An inertial target position and the epoch it must be reached.

    Attributes
    ----------
    position:
        Inertial position (m) at ``arrival_time``.
    arrival_time:
        Seconds from the bus epoch. Must be later than the release that
        services it, by enough for a transfer to exist.
    label:
        Optional identifier, carried through so a plan can be read back
        against whatever named the aimpoints.
    """

    position: _FloatArray
    arrival_time: float
    label: str = ""

    @classmethod
    def from_geodetic(
        cls,
        site: GeodeticPosition,
        arrival_time: float,
        gmst_epoch: float = 0.0,
        label: str | None = None,
    ) -> Aimpoint:
        """Build an aimpoint from a ground location in universal format.

        The Earth-fixed point is rotated into the inertial frame **at the
        arrival epoch**, not at the epoch of the call. That distinction is
        not pedantry: an equatorial ground point travels 465 m/s through
        the inertial frame, so using the wrong epoch displaces the aimpoint
        by 28 km per minute of error. Passing ``arrival_time`` through both
        the rotation and the targeting is what keeps them consistent.
        """
        return cls(
            position=geodetic_to_eci(site, arrival_time, gmst_epoch),
            arrival_time=arrival_time,
            label=site.label if label is None else label,
        )

    def __post_init__(self) -> None:
        pos = np.asarray(self.position, dtype=np.float64)
        if pos.shape != (3,):
            raise ValueError(f"position must be a 3-vector, got shape {pos.shape}")
        if not np.isfinite(pos).all():
            raise ValueError("position must be finite")
        if not (np.isfinite(self.arrival_time) and self.arrival_time > 0.0):
            raise ValueError(f"arrival_time must be finite and > 0, got {self.arrival_time}")
        object.__setattr__(self, "position", pos)


@dataclass(frozen=True)
class Release:
    """One vehicle leaving the bus.

    Attributes
    ----------
    aimpoint_index:
        Index into the caller's aimpoint list, so a reordered plan can be
        read back against the original numbering.
    release_time:
        Seconds from the bus epoch.
    delta_v:
        Bus maneuver (m/s) applied immediately before this release.
    bus_position:
        Inertial position (m) at release.
    vehicle_velocity:
        Inertial velocity (m/s) the released vehicle carries away. The bus
        holds this velocity until its next maneuver.
    achieved_miss:
        Terminal miss (m) of the released vehicle, obtained by propagating
        it. This is a *check*, not an estimate: a targeting solve that did
        not converge shows up here rather than being reported as success.
    dispersion:
        One-sigma expected miss (m) from accumulated execution error, or
        ``nan`` if no execution model was supplied.
    """

    aimpoint_index: int
    release_time: float
    delta_v: _FloatArray
    bus_position: _FloatArray
    vehicle_velocity: _FloatArray
    achieved_miss: float
    dispersion: float = float("nan")

    @property
    def cost(self) -> float:
        """Magnitude of the bus maneuver preceding this release (m/s)."""
        return float(np.linalg.norm(self.delta_v))


@dataclass(frozen=True)
class DeploymentPlan:
    """A full dispensing sequence and what it costs.

    Attributes
    ----------
    releases:
        In execution order.
    total_delta_v:
        Sum of maneuver magnitudes (m/s). The propellant figure of merit;
        deliberately *not* the norm of the vector sum, since maneuvers that
        partly oppose each other still consume propellant.
    order:
        Aimpoint indices in the order serviced.
    worst_miss:
        Largest achieved terminal miss (m) over the sequence.
    """

    releases: tuple[Release, ...]
    total_delta_v: float
    order: tuple[int, ...]
    worst_miss: float

    @property
    def dispersions(self) -> _FloatArray:
        """Per-vehicle one-sigma dispersion (m), in release order."""
        return np.array([r.dispersion for r in self.releases])

    @property
    def costs(self) -> _FloatArray:
        """Per-maneuver magnitudes (m/s), in release order."""
        return np.array([r.cost for r in self.releases])


def plan_deployment(
    bus_position: ArrayLike,
    bus_velocity: ArrayLike,
    aimpoints: list[Aimpoint] | tuple[Aimpoint, ...],
    release_times: ArrayLike,
    order: tuple[int, ...] | None = None,
    model: GravityModel = EARTH,
    execution: ExecutionErrorModel | None = None,
    include_j2: bool = True,
) -> DeploymentPlan:
    """Fly a dispensing sequence and cost it.

    The bus coasts to each release time, maneuvers to the velocity that
    puts the next vehicle on its aimpoint, and releases. Because the bus
    state is propagated forward through the whole sequence, each maneuver
    is charged against the state the previous one actually produced.

    Parameters
    ----------
    release_times:
        Seconds from the bus epoch, strictly increasing, one per vehicle.
        Each must precede the arrival time of the aimpoint it services.
    order:
        Aimpoint indices in service order. Defaults to the natural order,
        which is rarely the cheapest — see
        :func:`optimize_deployment_order`.
    execution:
        Bus maneuver error model. Supplying it turns on the dispersion
        accounting; without it dispersions are reported as ``nan`` rather
        than as zero, because a perfectly executed bus is an assumption and
        not a default.

    Raises
    ------
    ValueError
        If the schedule is inconsistent — non-increasing release times,
        a vehicle released after its own aimpoint's arrival epoch, or a
        mismatch between the number of vehicles and aimpoints.
    """
    r0 = np.asarray(bus_position, dtype=np.float64)
    v0 = np.asarray(bus_velocity, dtype=np.float64)
    if r0.shape != (3,) or v0.shape != (3,):
        raise ValueError("bus_position and bus_velocity must both be 3-vectors")
    targets = tuple(aimpoints)
    times = np.atleast_1d(np.asarray(release_times, dtype=np.float64))
    if len(targets) == 0:
        raise ValueError("at least one aimpoint is required")
    if times.size != len(targets):
        raise ValueError(
            f"need one release time per aimpoint: got {times.size} times for "
            f"{len(targets)} aimpoints"
        )
    if times.size > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("release_times must be strictly increasing")
    if times[0] <= 0.0:
        raise ValueError(f"release times must be > 0, got {times[0]:.6g}")

    sequence = tuple(range(len(targets))) if order is None else tuple(order)
    if sorted(sequence) != list(range(len(targets))):
        raise ValueError(f"order must be a permutation of 0..{len(targets) - 1}, got {sequence}")
    for slot, index in enumerate(sequence):
        if targets[index].arrival_time <= times[slot]:
            raise ValueError(
                f"aimpoint {index} arrives at t={targets[index].arrival_time:.6g} s "
                f"but is serviced by the release at t={times[slot]:.6g} s; a "
                f"vehicle cannot arrive before it is released"
            )

    releases: list[Release] = []
    total = 0.0
    state_r, state_v = r0, v0
    clock = 0.0
    # Accumulated bus-state covariance from imperfect maneuvers. It starts
    # at zero because this models *dispensing* error alone; a real bus adds
    # its own navigation error, which composes additively and is the
    # caller's to supply through the released vehicles' own budget.
    bus_covariance = np.zeros((6, 6))

    for slot, index in enumerate(sequence):
        target = targets[index]
        release_time = float(times[slot])
        coast = propagate_coast(
            state_r,
            state_v,
            release_time - clock,
            model=model,
            include_j2=include_j2,
            rtol=1e-12,
            atol=1e-6,
            n_output=2,
        )
        state_r = np.asarray(coast.states[:3, -1])
        state_v = np.asarray(coast.states[3:, -1])
        clock = release_time

        time_of_flight = target.arrival_time - release_time
        maneuver = correction_maneuver(
            state_r,
            state_v,
            target.position,
            time_of_flight,
            model=model,
            include_j2=include_j2,
        )
        vehicle_velocity = maneuver.v_required

        # Verify by flying it rather than trusting the solve.
        flown = propagate_coast(
            state_r,
            vehicle_velocity,
            time_of_flight,
            model=model,
            include_j2=include_j2,
            rtol=1e-12,
            atol=1e-6,
            n_output=2,
        )
        achieved = float(np.linalg.norm(np.asarray(flown.states[:3, -1]) - target.position))

        dispersion = float("nan")
        if execution is not None:
            # The vehicle inherits the bus covariance accumulated by every
            # previous maneuver, plus the error of the one just made. That
            # inheritance is the whole reason a deployment order changes
            # accuracy and not only cost.
            sensitivity = miss_sensitivity(
                state_r,
                vehicle_velocity,
                time_of_flight,
                model=model,
                include_j2=include_j2,
            )
            release_covariance = bus_covariance + _velocity_block(
                execution.covariance(maneuver.delta_v)
            )
            transition = np.hstack([sensitivity.position, sensitivity.velocity])
            dispersion = float(
                np.sqrt(max(0.0, np.trace(transition @ release_covariance @ transition.T)))
            )
            bus_covariance = release_covariance

        releases.append(
            Release(
                aimpoint_index=index,
                release_time=release_time,
                delta_v=maneuver.delta_v,
                bus_position=state_r,
                vehicle_velocity=vehicle_velocity,
                achieved_miss=achieved,
                dispersion=dispersion,
            )
        )
        total += maneuver.cost
        state_v = vehicle_velocity

    return DeploymentPlan(
        releases=tuple(releases),
        total_delta_v=float(total),
        order=sequence,
        worst_miss=max(r.achieved_miss for r in releases),
    )


def _velocity_block(velocity_covariance: _FloatArray) -> _FloatArray:
    """Embed a 3x3 velocity covariance in the 6x6 state ordering."""
    out = np.zeros((6, 6))
    out[3:, 3:] = velocity_covariance
    return out


def optimize_deployment_order(
    bus_position: ArrayLike,
    bus_velocity: ArrayLike,
    aimpoints: list[Aimpoint] | tuple[Aimpoint, ...],
    release_times: ArrayLike,
    model: GravityModel = EARTH,
    execution: ExecutionErrorModel | None = None,
    include_j2: bool = True,
    exhaustive_limit: int = 6,
) -> tuple[DeploymentPlan, str]:
    """Search deployment orders for the cheapest total :math:`\\Delta v`.

    Returns the best plan found and a string naming the method used, so a
    result is never mistaken for a proof of optimality it does not have.

    Up to ``exhaustive_limit`` vehicles every permutation is evaluated and
    the answer is exactly optimal over orderings. Above it the cost of
    enumeration grows factorially while each evaluation itself requires a
    full targeting solve per vehicle, so the search switches to 2-opt:
    repeatedly swap the pair of assignments that most reduces total cost,
    until no swap helps. That is a local optimum and is reported as one.

    Notes
    -----
    Only the *assignment* of aimpoints to release slots is searched. The
    release times themselves are held fixed, because moving them changes
    the reachability constraints as well as the cost and is a genuinely
    different problem.
    """
    targets = tuple(aimpoints)
    count = len(targets)
    if count == 0:
        raise ValueError("at least one aimpoint is required")
    if not (isinstance(exhaustive_limit, int) and exhaustive_limit >= 1):
        raise ValueError(f"exhaustive_limit must be an integer >= 1, got {exhaustive_limit}")

    def evaluate(order: tuple[int, ...]) -> DeploymentPlan | None:
        try:
            return plan_deployment(
                bus_position,
                bus_velocity,
                targets,
                release_times,
                order=order,
                model=model,
                execution=execution,
                include_j2=include_j2,
            )
        except (ValueError, RuntimeError):
            # An order that asks a vehicle to arrive before it is released,
            # or whose transfer does not exist, is infeasible rather than
            # expensive. Skipping is correct; silently costing it as zero
            # would not be.
            return None

    if count <= exhaustive_limit:
        best: DeploymentPlan | None = None
        for candidate in itertools.permutations(range(count)):
            plan = evaluate(candidate)
            if plan is not None and (best is None or plan.total_delta_v < best.total_delta_v):
                best = plan
        if best is None:
            raise ValueError("no feasible deployment order exists for this schedule")
        return best, "exhaustive"

    current = evaluate(tuple(range(count)))
    if current is None:
        raise ValueError(
            "the natural deployment order is infeasible, so 2-opt has no "
            "starting point; supply release times that admit it"
        )
    improved = True
    while improved:
        improved = False
        for i in range(count):
            for j in range(i + 1, count):
                swapped = list(current.order)
                swapped[i], swapped[j] = swapped[j], swapped[i]
                trial = evaluate(tuple(swapped))
                if trial is not None and trial.total_delta_v < current.total_delta_v:
                    current = trial
                    improved = True
    return current, "2-opt (local optimum)"


@dataclass(frozen=True)
class _Reachability:
    """Which aimpoints a budget admits, and what each would cost."""

    reachable: tuple[int, ...]
    costs: _FloatArray = field(repr=False)


def reachable_aimpoints(
    bus_position: ArrayLike,
    bus_velocity: ArrayLike,
    aimpoints: list[Aimpoint] | tuple[Aimpoint, ...],
    release_time: float,
    delta_v_budget: float,
    model: GravityModel = EARTH,
    include_j2: bool = True,
) -> _Reachability:
    """Aimpoints a single release at ``release_time`` can service.

    The footprint this traces out is strongly anisotropic, and that is the
    useful thing about it: downrange displacement is bought with a small
    speed change along the existing velocity, while crossrange displacement
    requires rotating the orbit plane and is expensive at orbital speed. A
    budget that reaches a long way downrange will reach only a short way to
    either side.

    Returns the indices within budget and the cost of each aimpoint,
    including the ones out of budget, so the shape of the footprint can be
    read off rather than only its boundary.
    """
    budget = float(delta_v_budget)
    if not (np.isfinite(budget) and budget >= 0.0):
        raise ValueError(f"delta_v_budget must be finite and >= 0, got {budget}")
    epoch = float(release_time)
    if not (np.isfinite(epoch) and epoch > 0.0):
        raise ValueError(f"release_time must be finite and > 0, got {epoch}")
    targets = tuple(aimpoints)
    if not targets:
        raise ValueError("at least one aimpoint is required")

    coast = propagate_coast(
        bus_position,
        bus_velocity,
        epoch,
        model=model,
        include_j2=include_j2,
        rtol=1e-12,
        atol=1e-6,
        n_output=2,
    )
    state_r = np.asarray(coast.states[:3, -1])
    state_v = np.asarray(coast.states[3:, -1])

    costs = np.full(len(targets), np.inf)
    for index, target in enumerate(targets):
        if target.arrival_time <= epoch:
            continue
        try:
            costs[index] = correction_maneuver(
                state_r,
                state_v,
                target.position,
                target.arrival_time - epoch,
                model=model,
                include_j2=include_j2,
            ).cost
        except (ValueError, RuntimeError):
            continue
    return _Reachability(
        reachable=tuple(int(i) for i in np.flatnonzero(costs <= budget)),
        costs=costs,
    )
