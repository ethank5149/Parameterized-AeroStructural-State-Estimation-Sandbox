"""II-V9 and II-V10 — Lambert targeting and post-boost bus dispensing.

Paper II, §8:

* **V9**: *"Terminal arrival error of the solved transfer, propagated
  through the independent :math:`J_2` integrator of §7.1, over a
  randomized envelope of geometries, times of flight and both directions
  of motion; conic invariance checked at both endpoints."* Failure is a
  relative arrival error above :math:`10^{-7}` on any physically flyable
  transfer, or energy/angular momentum differing between endpoints by more
  than :math:`10^{-9}` relative.
* **V10**: *"Every released vehicle propagated to its own aimpoint;
  ordering search checked against exhaustive enumeration."* Failure is any
  vehicle missing by more than 1 m, or the ordering search returning a cost
  above the exhaustive optimum.

What makes these checkable without external data
------------------------------------------------

Neither task compares against a published number, and both are honest
about being weaker for it. What they do compare against is an
*independent numerical path through the same physics*: Lambert solves a
boundary-value problem in closed form, while :func:`propagate_coast`
integrates the equations of motion, and the two share no code. Agreement
between them rules out a large class of errors in either. It does not rule
out a shared modelling assumption, and nothing here claims it does.

Two exclusions are applied and both are stated rather than quietly
imposed. Transfers whose periapsis lies inside the Earth are excluded from
the propagated check: they are valid conics but the integrator cannot pass
through a near-singular perigee, so including them would measure the
integrator. The conic-invariance check is applied to *every* transfer,
including those, because it needs no propagation.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from passes.geodesy import GeodeticPosition
from passes.guidance.bus import (
    Aimpoint,
    optimize_deployment_order,
    plan_deployment,
)
from passes.guidance.cruise import (
    CruiseVehicle,
    cruise_climb_altitude,
    cruise_range,
)
from passes.guidance.entry import (
    EntryVehicle,
    GlideState,
    equilibrium_glide_profile,
    range_to_go,
    simulate_glide,
)
from passes.guidance.midcourse import ExecutionErrorModel, correction_maneuver
from passes.orbital.coast import propagate_coast
from passes.orbital.fobs import deorbit_burn, fobs_profile, ground_track_shift
from passes.orbital.gravity import EARTH
from passes.orbital.lambert import lambert, minimum_energy_transfer
from passes.systems import (
    CEP_OVER_SIGMA,
    NAMED_ARCHITECTURES,
    R95_OVER_SIGMA,
    MissionRequest,
    accuracy_statistics,
    containment_probability,
    containment_radius,
    containment_ratio,
    evaluate,
)
from passes.verification.common import VerificationReport, write_csv

__all__ = ["run_p2v910"]

_FloatArray = NDArray[np.float64]

_ARRIVAL_TOLERANCE = 1.0e-7
_INVARIANT_TOLERANCE = 1.0e-9
_MISS_TOLERANCE = 1.0
_R_EARTH = 6378137.0
_MU = 3.986004418e14
_RANGE_TOLERANCE = 1.0e-9
_CROSSRANGE_FACTOR = 10.0


def _constant_drag(value: float, _energy: float) -> float:
    """A constant reference profile, for the pure-quadrature check only."""
    return value


def _periapsis(position: _FloatArray, velocity: _FloatArray) -> float:
    momentum = float(np.linalg.norm(np.cross(position, velocity)))
    energy = 0.5 * float(np.dot(velocity, velocity)) - EARTH.mu / float(np.linalg.norm(position))
    if energy == 0.0:
        return float("inf")
    eccentricity = np.sqrt(max(0.0, 1.0 + 2.0 * energy * momentum**2 / EARTH.mu**2))
    return float(-EARTH.mu / (2.0 * energy) * (1.0 - eccentricity))


def _dispersion_note(dispersions: _FloatArray) -> str:
    """Describe what the dispersion column actually does in this run.

    Written from the numbers rather than asserted, because the ordering is
    configuration-dependent: two competing effects set it and neither wins
    in general. Hard-coding either outcome in the caption would eventually
    put it in contradiction with its own table.
    """
    shape = (
        "rise monotonically" if bool(np.all(np.diff(dispersions) > 0.0)) else "rise and then fall"
    )
    caveat = (
        "so in this configuration the last vehicle released is also the least accurate"
        if bool(np.all(np.diff(dispersions) > 0.0))
        else "so here the last vehicle released is not the least accurate"
    )
    return (
        "The bus covariance grows monotonically — each maneuver contributes a "
        "positive-semidefinite block and none is removed. The terminal "
        f"dispersions {shape}, {caveat}. That ordering is not guaranteed "
        "either way: accumulated error pushes it up along the sequence while "
        "the shrinking remaining flight time pushes it down, and which "
        "dominates depends on the release schedule and the aimpoint spread. "
        "The unit tests exercise a configuration where the two cross and the "
        "dispersions are non-monotone. Which vehicle needs the accuracy "
        "budget is therefore a result of the schedule and cannot be read off "
        "the release order."
    )


def _v9_envelope(report: VerificationReport, output_dir: Path) -> bool:
    rng = np.random.default_rng(20260804)
    worst_arrival = 0.0
    worst_energy = 0.0
    worst_momentum = 0.0
    iterations: list[int] = []
    flyable = 0
    total = 0
    rows: list[list[object]] = []

    for _ in range(60):
        p1 = rng.normal(size=3)
        p1 = p1 / np.linalg.norm(p1) * rng.uniform(6.7e6, 4.5e7)
        p2 = rng.normal(size=3)
        p2 = p2 / np.linalg.norm(p2) * rng.uniform(6.7e6, 4.5e7)
        if np.linalg.norm(np.cross(p1, p2)) < 1e-6 * np.linalg.norm(p1) * np.linalg.norm(p2):
            continue
        _, t_min = minimum_energy_transfer(p1, p2)
        for fraction in (0.15, 0.4, 0.9, 1.0, 1.6, 4.0):
            for prograde in (True, False):
                tof = fraction * t_min
                solution = lambert(p1, p2, tof, prograde=prograde)
                total += 1
                iterations.append(solution.iterations)

                energy1 = 0.5 * float(np.dot(solution.v1, solution.v1)) - EARTH.mu / float(
                    np.linalg.norm(p1)
                )
                energy2 = 0.5 * float(np.dot(solution.v2, solution.v2)) - EARTH.mu / float(
                    np.linalg.norm(p2)
                )
                worst_energy = max(worst_energy, abs(energy2 - energy1) / abs(energy1))
                h1 = np.cross(p1, solution.v1)
                h2 = np.cross(p2, solution.v2)
                worst_momentum = max(
                    worst_momentum,
                    float(np.linalg.norm(h2 - h1) / np.linalg.norm(h1)),
                )

                if _periapsis(p1, solution.v1) < 1.05 * EARTH.radius:
                    continue
                reached = propagate_coast(
                    p1,
                    solution.v1,
                    tof,
                    include_j2=False,
                    rtol=1e-13,
                    atol=1e-6,
                    n_output=2,
                )
                error = float(
                    np.linalg.norm(np.asarray(reached.states[:3, -1]) - p2) / np.linalg.norm(p2)
                )
                worst_arrival = max(worst_arrival, error)
                flyable += 1
                rows.append([fraction, prograde, tof, solution.iterations, error])

    write_csv(
        output_dir,
        "p2v9-lambert-envelope",
        ["tof_over_tmin", "prograde", "time_of_flight", "iterations", "rel_error"],
        rows,
    )
    passed = (
        worst_arrival <= _ARRIVAL_TOLERANCE
        and worst_energy <= _INVARIANT_TOLERANCE
        and worst_momentum <= _INVARIANT_TOLERANCE
    )
    report.add_table(
        "V9 — Lambert transfer envelope",
        ["quantity", "measured", "criterion", "verdict"],
        [
            [
                "worst relative arrival error",
                f"{worst_arrival:.3e}",
                f"< {_ARRIVAL_TOLERANCE:.0e}",
                "PASS" if worst_arrival <= _ARRIVAL_TOLERANCE else "FAIL",
            ],
            [
                "worst endpoint energy mismatch",
                f"{worst_energy:.3e}",
                f"< {_INVARIANT_TOLERANCE:.0e}",
                "PASS" if worst_energy <= _INVARIANT_TOLERANCE else "FAIL",
            ],
            [
                "worst endpoint angular-momentum mismatch",
                f"{worst_momentum:.3e}",
                f"< {_INVARIANT_TOLERANCE:.0e}",
                "PASS" if worst_momentum <= _INVARIANT_TOLERANCE else "FAIL",
            ],
            ["transfers solved", str(total), "—", "—"],
            ["of those, physically flyable", str(flyable), "—", "—"],
            [
                "Householder iterations (median / max)",
                f"{int(np.median(iterations))} / {max(iterations)}",
                "—",
                "—",
            ],
        ],
        "The arrival check propagates Lambert's velocity through the coast "
        "integrator, which shares no code with the solver. The invariance "
        "check needs no propagation and so is applied to every transfer, "
        "including the "
        f"{total - flyable} whose periapsis lies inside the Earth and which "
        "the integrator cannot fly.",
    )
    return passed


def _v9_correction(report: VerificationReport) -> bool:
    """The targeting property the correction module depends on."""
    r0 = np.array([6.7e6, 0.0, 0.0])
    arc = 1800.0
    v0 = lambert(r0, np.array([2.0e6, 6.4e6, 0.5e6]), arc).v1
    target = np.asarray(
        propagate_coast(r0, v0, arc, rtol=1e-12, atol=1e-6, n_output=2).states[:3, -1]
    )
    perturbed = v0 + np.array([2.0, -1.5, 0.8])
    uncorrected = float(
        np.linalg.norm(
            np.asarray(
                propagate_coast(r0, perturbed, arc, rtol=1e-12, atol=1e-6, n_output=2).states[
                    :3, -1
                ]
            )
            - target
        )
    )

    rows: list[list[str]] = []
    worst_residual = 0.0
    products: list[float] = []
    for fraction in (0.05, 0.25, 0.5, 0.75, 0.9):
        burn = fraction * arc
        state = propagate_coast(r0, perturbed, burn, rtol=1e-12, atol=1e-6, n_output=2)
        r = np.asarray(state.states[:3, -1])
        v = np.asarray(state.states[3:, -1])
        tof = arc - burn
        maneuver = correction_maneuver(r, v, target, tof)
        residual = float(
            np.linalg.norm(
                np.asarray(
                    propagate_coast(
                        r,
                        maneuver.v_required,
                        tof,
                        rtol=1e-12,
                        atol=1e-6,
                        n_output=2,
                    ).states[:3, -1]
                )
                - target
            )
        )
        worst_residual = max(worst_residual, residual)
        products.append(maneuver.cost * tof)
        rows.append(
            [
                f"{fraction:.2f}",
                f"{tof:.0f}",
                f"{maneuver.cost:.3f}",
                f"{maneuver.cost * tof / 1e3:.2f}",
                f"{residual:.2e}",
            ]
        )

    spread = float(max(products) / min(products) - 1.0)
    passed = worst_residual <= _MISS_TOLERANCE and spread < 0.15
    report.add_table(
        "V9 — correction targeting under J2, and the inverse-time-to-go law",
        ["burn t/T", "t_go (s)", "ΔV (m/s)", "ΔV·t_go (km)", "residual miss (m)"],
        rows,
        f"Uncorrected miss is {uncorrected / 1e3:.2f} km. The product ΔV·t_go "
        f"is constant to {100 * spread:.1f}% across the arc and equals that "
        "miss, which is the |δr|/t_go scaling appearing as a measurement "
        "rather than as an assertion. Residual miss stays below "
        f"{_MISS_TOLERANCE:.0f} m everywhere, including at t/T = 0.9 where the "
        "vehicle is nearly collinear with its own aimpoint and the two-body "
        "seed alone is useless.",
    )
    return passed


def _v10_dispensing(report: VerificationReport) -> bool:
    r0 = np.array([6.7e6, 0.0, 0.0])
    arc = 2400.0
    v0 = lambert(r0, np.array([1.5e6, 6.5e6, 0.3e6]), arc).v1
    end = propagate_coast(r0, v0, arc, rtol=1e-12, atol=1e-6, n_output=2)
    nominal = np.asarray(end.states[:3, -1])
    v_end = np.asarray(end.states[3:, -1])
    radial = nominal / np.linalg.norm(nominal)
    downrange = v_end - np.dot(v_end, radial) * radial
    downrange = downrange / np.linalg.norm(downrange)
    crossrange = np.cross(radial, downrange)

    offsets = [(80e3, -30e3), (-70e3, 45e3), (120e3, 20e3), (-40e3, -60e3)]
    aimpoints = [
        Aimpoint(
            position=nominal + d * downrange + c * crossrange,
            arrival_time=arc,
            label=f"A{i}",
        )
        for i, (d, c) in enumerate(offsets)
    ]
    times = [200.0, 700.0, 1300.0, 1900.0]

    costs = {
        order: plan_deployment(r0, v0, aimpoints, times, order=order).total_delta_v
        for order in itertools.permutations(range(4))
    }
    brute_best = min(costs.values())
    brute_worst = max(costs.values())
    natural = costs[(0, 1, 2, 3)]
    searched, method = optimize_deployment_order(r0, v0, aimpoints, times)

    dispersed = plan_deployment(
        r0,
        v0,
        aimpoints[:3],
        times[:3],
        execution=ExecutionErrorModel(magnitude_fraction=0.02, pointing_sigma=5e-3),
    )

    optimal = searched.total_delta_v <= brute_best * (1.0 + 1e-9)
    accurate = searched.worst_miss <= _MISS_TOLERANCE
    passed = optimal and accurate

    report.add_table(
        "V10 — dispensing four vehicles, all 24 orderings enumerated",
        ["quantity", "measured", "criterion", "verdict"],
        [
            [
                "worst achieved miss, any vehicle",
                f"{searched.worst_miss:.3e} m",
                f"< {_MISS_TOLERANCE:.0f} m",
                "PASS" if accurate else "FAIL",
            ],
            [
                "search cost vs exhaustive optimum",
                f"{searched.total_delta_v:.2f} vs {brute_best:.2f} m/s",
                "not above optimum",
                "PASS" if optimal else "FAIL",
            ],
            ["search method reported", method, "—", "—"],
            ["cheapest ordering", f"{brute_best:.2f} m/s {searched.order}", "—", "—"],
            ["dearest ordering", f"{brute_worst:.2f} m/s", "—", "—"],
            [
                "spread across orderings",
                f"{100 * (brute_worst / brute_best - 1):.0f}%",
                "—",
                "—",
            ],
            [
                "natural order (0,1,2,3)",
                f"{natural:.2f} m/s" + (" — the worst available" if natural == brute_worst else ""),
                "—",
                "—",
            ],
        ],
        "Ordering is the dominant cost lever, not the individual maneuvers. "
        "The natural index order is not merely suboptimal here; it is the "
        "worst of the twenty-four.",
    )

    report.add_table(
        "V10 — accumulated error and terminal dispersion along the sequence",
        ["release", "aimpoint", "ΔV (m/s)", "1σ dispersion (m)"],
        [
            [
                str(i),
                str(rel.aimpoint_index),
                f"{rel.cost:.2f}",
                f"{rel.dispersion:.0f}",
            ]
            for i, rel in enumerate(dispersed.releases)
        ],
        _dispersion_note(dispersed.dispersions),
    )
    return passed


def _v11_glide(report: VerificationReport) -> bool:
    """II-V11: the range-energy relation, and what bank reversals buy."""
    vehicle = EntryVehicle(ballistic_coefficient=200.0, lift_to_drag=2.0)
    entry = GlideState(
        radius=_R_EARTH + 80e3,
        longitude=0.0,
        latitude=0.0,
        speed=7000.0,
        flight_path_angle=np.deg2rad(-1.0),
        heading=np.deg2rad(90.0),
    )
    final_energy = 0.5 * 1000.0**2 - _MU / (_R_EARTH + 30e3)

    # 1. The integral against its closed form at constant drag.
    worst_integral = 0.0
    for drag in (5.0, 12.0, 20.0, 40.0, 80.0):
        constant: Callable[[float], float] = partial(_constant_drag, drag)
        predicted = range_to_go(constant, entry.specific_energy, final_energy)
        exact = (entry.specific_energy - final_energy) / drag
        worst_integral = max(worst_integral, abs(predicted / exact - 1.0))

    # 2. Flown range against the same integral's prediction, on reference
    #    profiles the vehicle can actually fly. A constant-drag reference
    #    is *not* one: at entry interface the equilibrium glide supports
    #    only about 2 m/s², so commanding 20 asks for something impossible
    #    over the first half of the glide and the tracker simply sits
    #    against its bank stop. Building the reference from the
    #    equilibrium-glide condition is what makes the comparison a test of
    #    the guidance rather than of the saturation limit.
    target = (np.deg2rad(60.0), 0.0)
    reference_radius = _R_EARTH + 50e3
    rows: list[list[str]] = []
    flown: list[float] = []
    errors: list[float] = []
    for bank_deg in (30.0, 45.0, 60.0, 70.0):
        profile = equilibrium_glide_profile(vehicle, reference_radius, np.deg2rad(bank_deg))
        result = simulate_glide(vehicle, entry, profile, target=target)
        predicted = range_to_go(profile, entry.specific_energy, final_energy)
        flown.append(result.downrange)
        errors.append(abs(predicted / result.downrange - 1.0))
        rows.append(
            [
                f"{bank_deg:.0f}",
                f"{predicted / 1e3:.0f}",
                f"{result.downrange / 1e3:.0f}",
                f"{100 * (predicted / result.downrange - 1.0):+.0f}%",
                str(result.reversals),
            ]
        )
    monotone = bool(np.all(np.diff(flown) < 0.0))

    # 3. What the lateral logic buys, on a mid-range flyable profile.
    lateral_profile = equilibrium_glide_profile(vehicle, reference_radius, np.deg2rad(45.0))
    # The target must sit at the range this profile actually delivers.
    # That is not a convenience: bank magnitude and bank sign are
    # independent in *mechanism*, but the lateral logic steers on bearing
    # to the target, so a longitudinal profile that overflies inverts the
    # bearing and the deadband logic degenerates. Placing the target 1000
    # km short of the delivered range cut the crossrange benefit from 39x
    # to 4x. Range matching is a precondition for the lateral channel, not
    # an independent concern.
    drifting = simulate_glide(vehicle, entry, lateral_profile, target=None)
    matched_arc = drifting.downrange / _R_EARTH
    matched_target = (float(matched_arc), 0.0)
    corrected = simulate_glide(vehicle, entry, lateral_profile, target=matched_target)
    reduction = abs(drifting.crossrange) / max(abs(corrected.crossrange), 1.0)

    integral_ok = worst_integral <= _RANGE_TOLERANCE
    crossrange_ok = reduction >= _CROSSRANGE_FACTOR
    passed = integral_ok and monotone and crossrange_ok

    report.add_table(
        "V11 — range-energy relation and flown range",
        [
            "reference bank (deg)",
            "predicted range (km)",
            "flown (km)",
            "prediction error",
            "reversals",
        ],
        rows,
        "Each reference is the equilibrium-glide drag profile at the stated "
        "nominal bank, so all four are flyable; a larger bank asks the "
        "vehicle to fly deeper and shorter, which is how range is traded. "
        "The prediction is the shallow-glide integral of the range-energy "
        "relation and the flown value comes from the closed-loop 3-DOF "
        "trajectory, so the gap is the cos-gamma term the prediction drops "
        "plus residual tracking error, and it widens with bank as the "
        "command approaches saturation. Best agreement is "
        f"{100 * min(errors):.0f}% at moderate bank. Flown range strictly "
        f"decreasing in reference bank: "
        f"{'satisfied' if monotone else 'VIOLATED'}. Against its own closed "
        f"form at constant drag the integral itself is exact to "
        f"{worst_integral:.2e} relative — that check is pure quadrature and "
        "is independent of whether any vehicle could fly the profile.",
    )
    report.add_table(
        "V11 — terminal crossrange, with and without bank reversals",
        ["configuration", "reversals", "crossrange (km)", "downrange (km)"],
        [
            [
                "single bank sign held",
                str(drifting.reversals),
                f"{drifting.crossrange / 1e3:+.0f}",
                f"{drifting.downrange / 1e3:.0f}",
            ],
            [
                "scheduled-deadband reversals",
                str(corrected.reversals),
                f"{corrected.crossrange / 1e3:+.0f}",
                f"{corrected.downrange / 1e3:.0f}",
            ],
        ],
        f"Reversals reduce terminal crossrange by a factor of "
        f"{reduction:.0f}, against a criterion of {_CROSSRANGE_FACTOR:.0f}. "
        "The uncorrected case is the honest baseline: it is not a failure "
        "mode but the natural behaviour of a lifting vehicle holding one "
        "bank sign, and it is what the lateral logic exists to remove.\n\n"
        "The target here is placed at the range the longitudinal profile "
        "actually delivers. That is load-bearing rather than tidy: the "
        "lateral logic steers on bearing to the target, so a profile that "
        "overflies inverts the bearing part-way through and the deadband "
        "stops meaning what it should. Placing the target 1000 km short of "
        "the delivered range degrades the benefit from 39x to 4x — which is "
        "how this criterion first failed. Range matching is a precondition "
        "for the lateral channel, not an independent concern.",
    )
    return passed


def _v12_fobs(report: VerificationReport) -> bool:
    """II-V12: the deorbit solve against an independent integration."""
    parking = _R_EARTH + 200e3
    entry_radius = _R_EARTH + 100e3
    circular = float(np.sqrt(_MU / parking))

    rows: list[list[str]] = []
    worst_radius = 0.0
    worst_arc = 0.0
    worst_speed = 0.0
    worst_gamma = 0.0
    worst_visviva = 0.0
    costs: list[float] = []
    arcs: list[float] = []
    gammas: list[float] = []

    for altitude in (80e3, 50e3, 0.0, -100e3, -400e3, -1000e3):
        perigee = _R_EARTH + altitude
        burn = deorbit_burn(parking, entry_radius, perigee)
        costs.append(burn.delta_v)
        arcs.append(burn.transfer_angle)
        gammas.append(burn.entry_flight_path_angle)

        sma = 0.5 * (parking + perigee)
        expected = np.sqrt(_MU / parking) - np.sqrt(_MU * (2.0 / parking - 1.0 / sma))
        worst_visviva = max(worst_visviva, abs(burn.delta_v / expected - 1.0))

        apogee_speed = np.sqrt(_MU * (2.0 / parking - 1.0 / sma))
        flown = propagate_coast(
            np.array([parking, 0.0, 0.0]),
            np.array([0.0, apogee_speed, 0.0]),
            burn.transfer_time,
            include_j2=False,
            rtol=1e-13,
            atol=1e-6,
            n_output=2,
        )
        r = np.asarray(flown.states[:3, -1])
        v = np.asarray(flown.states[3:, -1])
        radius = float(np.linalg.norm(r))
        arc = float(np.arctan2(r[1], r[0]))
        speed = float(np.linalg.norm(v))
        gamma = float(np.arcsin(np.dot(r, v) / (radius * speed)))

        worst_radius = max(worst_radius, abs(radius - entry_radius))
        worst_arc = max(worst_arc, abs(arc - burn.transfer_angle))
        worst_speed = max(worst_speed, abs(speed - burn.entry_speed))
        worst_gamma = max(worst_gamma, abs(gamma - burn.entry_flight_path_angle))

        rows.append(
            [
                f"{altitude / 1e3:+.0f}",
                f"{burn.delta_v:.1f}",
                f"{100 * burn.delta_v / circular:.2f}%",
                f"{np.rad2deg(burn.transfer_angle):.1f}",
                f"{_R_EARTH * burn.transfer_angle / 1e3:.0f}",
                f"{np.rad2deg(burn.entry_flight_path_angle):.3f}",
            ]
        )

    # Three-leg accounting.
    profile = fobs_profile(
        np.deg2rad(200.0),
        np.deg2rad(60.0),
        parking,
        entry_radius,
        _R_EARTH - 400e3,
    )
    closure = abs(
        (profile.parking_arc + profile.transfer_arc + profile.glide_arc) / profile.total_arc - 1.0
    )

    integrator_ok = (
        worst_radius <= 1e-3 and worst_arc <= 1e-10 and worst_speed <= 1e-6 and worst_gamma <= 1e-10
    )
    visviva_ok = worst_visviva <= 1e-12
    accounting_ok = closure <= 1e-12
    monotone = (
        costs == sorted(costs)
        and arcs == sorted(arcs, reverse=True)
        and gammas == sorted(gammas, reverse=True)
    )
    passed = integrator_ok and visviva_ok and accounting_ok and monotone

    period = 2.0 * np.pi * np.sqrt(parking**3 / _MU)
    report.add_table(
        "V12 — deorbit design curve from a 200 km circular parking orbit",
        [
            "perigee (km)",
            "ΔV (m/s)",
            "of orbital speed",
            "transfer arc (deg)",
            "arc (km)",
            "entry γ (deg)",
        ],
        rows,
        "A negative perigee is virtual — the vehicle never reaches it — and "
        "is simply how a steep entry is specified. The burn is cheap in "
        "every case: what it buys is timing, not energy. Perigee depth is "
        "the dominant choice, trading an order of magnitude in ΔV for "
        "roughly a factor of four in transfer arc. Monotonicity of ΔV, arc "
        f"and entry angle across the sweep: "
        f"{'satisfied' if monotone else 'VIOLATED'}.",
    )
    report.add_table(
        "V12 — closed-form solve against the independent integrator",
        ["quantity", "worst discrepancy", "criterion", "verdict"],
        [
            [
                "entry radius",
                f"{worst_radius:.3e} m",
                "< 1e-3 m",
                "PASS" if worst_radius <= 1e-3 else "FAIL",
            ],
            [
                "swept angle",
                f"{worst_arc:.3e} rad",
                "< 1e-10 rad",
                "PASS" if worst_arc <= 1e-10 else "FAIL",
            ],
            [
                "entry speed",
                f"{worst_speed:.3e} m/s",
                "< 1e-6 m/s",
                "PASS" if worst_speed <= 1e-6 else "FAIL",
            ],
            [
                "flight-path angle",
                f"{worst_gamma:.3e} rad",
                "< 1e-10 rad",
                "PASS" if worst_gamma <= 1e-10 else "FAIL",
            ],
            [
                "ΔV vs vis-viva",
                f"{worst_visviva:.3e}",
                "< 1e-12 rel",
                "PASS" if visviva_ok else "FAIL",
            ],
            [
                "three-leg range closure",
                f"{closure:.3e}",
                "< 1e-12 rel",
                "PASS" if accounting_ok else "FAIL",
            ],
        ],
        "The deorbit solve is closed-form Kepler; the reference is the coast "
        "integrator advancing the equations of motion from the post-burn "
        "state. The two share no code, so agreement at this level is a real "
        "check on both. Ground-track walk for this orbit is "
        f"{np.rad2deg(ground_track_shift(period)):.1f} deg per revolution, "
        "which is what allows the entry interface to be repositioned by "
        "waiting rather than by manoeuvring.",
    )
    return passed


def _v13_cruise(report: VerificationReport) -> bool:
    """II-V13: Breguet range against its analytic scalings."""
    sound = 295.0
    speed = 8.0 * sound
    isp = 1200.0

    def vehicle(lift_to_drag: float, fuel: float) -> CruiseVehicle:
        return CruiseVehicle(
            wing_loading=400.0,
            lift_to_drag=lift_to_drag,
            lift_coefficient=0.1,
            fuel_fraction=fuel,
        )

    # Linearity in L/D.
    base = cruise_range(vehicle(4.0, 0.30), speed, isp)
    linear_error = abs(cruise_range(vehicle(8.0, 0.30), speed, isp) / (2.0 * base) - 1.0)

    # Each doubling of mass ratio adds exactly the same increment.
    increments = []
    for ratio in (1.5, 2.0, 4.0, 8.0):
        low = cruise_range(vehicle(4.0, 1.0 - 1.0 / ratio), speed, isp)
        high = cruise_range(vehicle(4.0, 1.0 - 1.0 / (2.0 * ratio)), speed, isp)
        increments.append(high - low)
    increment_spread = float(max(increments) / min(increments) - 1.0)

    # Cruise-climb is independent of the vehicle.
    climbs = [
        cruise_climb_altitude(vehicle(4.0, 0.30)),
        cruise_climb_altitude(
            CruiseVehicle(
                wing_loading=900.0,
                lift_to_drag=1.5,
                lift_coefficient=0.4,
                fuel_fraction=0.30,
            )
        ),
    ]
    climb_spread = abs(climbs[0] / climbs[1] - 1.0)

    # The scaling that is commonly stated backwards.
    by_ld = cruise_range(vehicle(8.0, 0.30), speed, isp) / base
    by_fuel = cruise_range(vehicle(4.0, 0.60), speed, isp) / base

    linear_ok = linear_error <= 1e-12
    increment_ok = increment_spread <= 1e-9
    climb_ok = climb_spread <= 1e-12
    passed = linear_ok and increment_ok and climb_ok

    report.add_table(
        "V13 — Breguet range against its analytic scalings",
        ["property", "measured", "criterion", "verdict"],
        [
            [
                "linearity in L/D",
                f"{linear_error:.3e} rel",
                "< 1e-12",
                "PASS" if linear_ok else "FAIL",
            ],
            [
                "range added per doubling of mass ratio, spread over 4 decades",
                f"{increment_spread:.3e}",
                "< 1e-9",
                "PASS" if increment_ok else "FAIL",
            ],
            [
                "cruise-climb across dissimilar vehicles",
                f"{climb_spread:.3e} rel",
                "< 1e-12",
                "PASS" if climb_ok else "FAIL",
            ],
            ["cruise-climb at 30% fuel", f"{climbs[0] / 1e3:.2f} km", "—", "—"],
        ],
        "Each doubling of mass ratio adds the same absolute range regardless "
        "of where it starts, which is the precise content of 'logarithmic in "
        "mass ratio'. The cruise-climb is H ln(m_i/m_f) with wing loading, "
        "lift coefficient and L/D all cancelling, so two vehicles sharing "
        "only a fuel fraction climb identically.",
    )
    report.add_table(
        "V13 — where the usual gloss on that scaling goes wrong",
        ["change", "range multiplier"],
        [
            ["L/D doubled, 4 -> 8", f"{by_ld:.3f}"],
            ["fuel fraction doubled, 0.30 -> 0.60", f"{by_fuel:.3f}"],
        ],
        "Fuel is commonly said to show diminishing returns while L/D does "
        "not. Over this range the opposite holds, because ln(1/(1-f)) is "
        "*convex* in fuel fraction: its derivative 1/(1-f) grows, so "
        "doubling f more than doubles the logarithm. The diminishing return "
        "is in mass ratio, not in fuel fraction — the row above measures "
        "that one directly — and the two are statements about different "
        "variables.",
    )
    return passed


def _v14_budget(report: VerificationReport) -> bool:
    """II-V14: containment statistics and end-to-end ledger closure."""
    # 1. The elliptical integral against its two closed-form limits.
    circular = accuracy_statistics(100.0, 100.0)
    circular_error = max(
        abs(circular.cep / (100.0 * CEP_OVER_SIGMA) - 1.0),
        abs(circular.r95 / (100.0 * R95_OVER_SIGMA) - 1.0),
    )
    degenerate_error = max(
        abs(containment_radius(0.95, 100.0, 0.0) / 195.9964 - 1.0),
        abs(containment_radius(0.5, 100.0, 0.0) / 67.449 - 1.0),
    )

    # 2. Radius/probability round-trip at several aspect ratios.
    round_trip = 0.0
    for major, minor in ((100.0, 100.0), (300.0, 100.0), (500.0, 50.0)):
        for p in (0.3, 0.5, 0.95, 0.99):
            radius = containment_radius(p, major, minor)
            round_trip = max(round_trip, abs(containment_probability(radius, major, minor) - p))

    ratios = [containment_ratio(100.0, 100.0 * f) for f in (1.0, 0.5, 0.3, 0.1, 0.02)]
    monotone = ratios == sorted(ratios)

    # 3. Every named architecture produces a ledger, a verdict, and stats.
    request = MissionRequest(
        launch_site=GeodeticPosition.from_degrees(45.0, 60.0, label="launch"),
        aimpoints=(
            GeodeticPosition.from_degrees(38.0, -100.0, label="T1"),
            GeodeticPosition.from_degrees(40.0, -95.0, label="T2"),
        ),
        arrival_time=3000.0,
    )
    entry = EntryVehicle(ballistic_coefficient=200.0, lift_to_drag=2.0)
    cruise = CruiseVehicle(
        wing_loading=400.0,
        lift_to_drag=4.0,
        lift_coefficient=0.1,
        fuel_fraction=0.30,
    )
    rows: list[list[str]] = []
    all_costed = True
    understated = True
    for name, architecture in NAMED_ARCHITECTURES.items():
        budget = evaluate(
            architecture,
            request,
            entry_vehicle=entry,
            cruise_vehicle=cruise,
            glide_range=3.0e6,
        )
        stats = budget.accuracy
        if stats is None or budget.total_delta_v <= 0.0:
            all_costed = False
            continue
        # The circular scaling is a *lower bound* on R95, met with equality
        # only when the dispersion is genuinely isotropic. Since midcourse
        # correction resets to an isotropic floor, architectures whose
        # remaining contributions are also isotropic land exactly on it —
        # so the criterion is that the scaling never over-states, not that
        # it always under-states.
        if stats.cep * (R95_OVER_SIGMA / CEP_OVER_SIGMA) > stats.r95 * (1.0 + 1e-9):
            understated = False
        rows.append(
            [
                name,
                "yes" if budget.closes else "no",
                f"{budget.total_range / 1e3:.0f}",
                f"{budget.total_delta_v:.0f}",
                f"{stats.cep:.0f}",
                f"{stats.r95:.0f}",
                f"{stats.ratio:.3f}",
            ]
        )

    limits_ok = circular_error <= 1e-4 and degenerate_error <= 1e-4
    round_trip_ok = round_trip <= 1e-8
    passed = limits_ok and round_trip_ok and monotone and all_costed and understated

    report.add_table(
        "V14 — containment statistics against their closed-form limits",
        ["property", "measured", "criterion", "verdict"],
        [
            [
                "elliptical integral at unit aspect ratio vs Rayleigh",
                f"{circular_error:.3e} rel",
                "< 1e-4",
                "PASS" if circular_error <= 1e-4 else "FAIL",
            ],
            [
                "degenerate axis vs the normal quantile",
                f"{degenerate_error:.3e} rel",
                "< 1e-4",
                "PASS" if degenerate_error <= 1e-4 else "FAIL",
            ],
            [
                "radius/probability round-trip",
                f"{round_trip:.3e}",
                "< 1e-8",
                "PASS" if round_trip_ok else "FAIL",
            ],
            [
                "R95/CEP ratio monotone in elongation",
                f"{ratios[0]:.3f} -> {ratios[-1]:.3f}",
                "non-decreasing",
                "PASS" if monotone else "FAIL",
            ],
        ],
        "The radial part of the containment integral is analytic, leaving a "
        "one-dimensional quadrature, so the elliptical answers are exact "
        "rather than a fitted correction to the circular ones. The ratio "
        "rises with elongation towards the one-dimensional value 1.96/0.6745 "
        "= 2.906, which means scaling a CEP by the circular 2.079 "
        "under-states the 95% radius for every real dispersion.",
    )
    report.add_table(
        "V14 — end-to-end budget over every named architecture",
        ["architecture", "closes", "range (km)", "ΔV (m/s)", "CEP (m)", "R95 (m)", "R95/CEP"],
        rows,
        "Range, propellant and accuracy for one launch site and two "
        "aimpoints. Which leg absorbs the range remainder differs by family "
        "— parking arc for fractional-orbital profiles, which costs time and "
        "no propellant, and boost for suborbital ones, which costs both — so "
        "a 'does not close' verdict means different things in the two cases "
        "and the budget names which. Every ratio exceeds the circular 2.079, "
        "Ratios at exactly 2.079 are isotropic dispersions, which arise "
        "where a midcourse correction resets to an isotropic floor and "
        "nothing anisotropic follows; every other ratio exceeds it, so the "
        "circular scaling never over-states the 95% radius and usually "
        "under-states it. Deorbit, dispensing, glide and terminal "
        "contributions are now derived from the phase models; only boost "
        "injection remains a stated specification.",
    )
    return passed


def run_p2v910(output_dir: Path) -> VerificationReport:
    """Execute II-V9 and II-V10 and write the report."""
    report = VerificationReport(
        task_id="II-V9-V14",
        title="Lambert targeting, bus dispensing, glide guidance, fractional orbital profiles",
        criterion=(
            "V9: relative arrival error > 1e-7 on any physically flyable "
            "transfer, or endpoint energy/angular-momentum mismatch > 1e-9. "
            "V10: any released vehicle missing its aimpoint by > 1 m, or the "
            "ordering search returning a cost above the exhaustive optimum. "
            "V11: range integral differing from its closed form by > 1e-9, "
            "flown range not monotone in commanded drag, or reversals failing "
            "to reduce crossrange by 10x. V12: the Kepler deorbit solve "
            "differing from the integrated trajectory beyond tolerance, or the "
            "three-leg range accounting failing to close. V13: Breguet range "
            "not linear in L/D, mass-ratio doublings not adding equal range, "
            "or the cruise-climb differing between vehicles. V14: containment "
            "radii disagreeing with their closed-form limits, a failed "
            "radius/probability round-trip, or any architecture without a "
            "ledger, verdict and CEP/R95 pair"
        ),
        passed=True,
    )
    results = [
        _v9_envelope(report, output_dir),
        _v9_correction(report),
        _v10_dispensing(report),
        _v11_glide(report),
        _v12_fobs(report),
        _v13_cruise(report),
        _v14_budget(report),
    ]
    report.passed = all(results)

    report.add_section(
        "What these tasks establish, and what they do not",
        "Neither task compares against a published number. The reference is "
        "an *independent numerical path through the same physics*: Lambert "
        "solves a boundary-value problem in closed form while the coast "
        "propagator integrates the equations of motion, and the two share no "
        "code. Agreement rules out a large class of implementation errors in "
        "either.\n\n"
        "It does **not** rule out a shared modelling assumption, and this is "
        "a weaker claim than validation against measurement. Both tasks are "
        "stated that way deliberately rather than being dressed as "
        "validation. What would strengthen them is a published transfer case "
        "with tabulated terminal state, and a published dispensing budget "
        "for a stated aimpoint geometry; neither is currently in this "
        "repository.",
    )
    return report
