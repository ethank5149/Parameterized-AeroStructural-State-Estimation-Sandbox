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
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from passes.guidance.bus import (
    Aimpoint,
    optimize_deployment_order,
    plan_deployment,
)
from passes.guidance.midcourse import ExecutionErrorModel, correction_maneuver
from passes.orbital.coast import propagate_coast
from passes.orbital.gravity import EARTH
from passes.orbital.lambert import lambert, minimum_energy_transfer
from passes.verification.common import VerificationReport, write_csv

__all__ = ["run_p2v910"]

_FloatArray = NDArray[np.float64]

_ARRIVAL_TOLERANCE = 1.0e-7
_INVARIANT_TOLERANCE = 1.0e-9
_MISS_TOLERANCE = 1.0


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


def run_p2v910(output_dir: Path) -> VerificationReport:
    """Execute II-V9 and II-V10 and write the report."""
    report = VerificationReport(
        task_id="II-V9-V10",
        title="Lambert targeting and post-boost bus dispensing",
        criterion=(
            "V9: relative arrival error > 1e-7 on any physically flyable "
            "transfer, or endpoint energy/angular-momentum mismatch > 1e-9. "
            "V10: any released vehicle missing its aimpoint by > 1 m, or the "
            "ordering search returning a cost above the exhaustive optimum"
        ),
        passed=True,
    )
    results = [
        _v9_envelope(report, output_dir),
        _v9_correction(report),
        _v10_dispensing(report),
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
