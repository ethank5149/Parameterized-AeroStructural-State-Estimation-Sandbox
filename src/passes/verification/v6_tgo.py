"""V6 — guidance: stable vs textbook time-to-go as A_c → 0.

Paper I, §8: *"Eq. (4.16) against Eq. (4.15) in double and single
precision as Â_c → 0; Prop. 4. Failure criterion: Eq. (4.16) losing more
than 1 significant digit where Eq. (4.15) loses many."*

The reference value for every operating point is computed in 50-digit
decimal arithmetic from the *exact binary values* of the rounded inputs
(``Decimal(float)`` is exact), so the comparison isolates algorithmic
error from input representation error. Significant digits retained are
reported as −log₁₀ of the relative error against that reference.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, getcontext
from pathlib import Path

import numpy as np

from passes.guidance import TgoStatus, time_to_go, time_to_go_naive
from passes.verification.common import VerificationReport, write_csv

__all__ = ["run_v6"]

_R_LOS = 1.0e4  # m
_V_C = 1.0e3  # m/s
#: Sweep of closing accelerations, both signs, spanning "dominant" to
#: "16 decades below V_c^2 / (2 R)" — the regime terminal flight lives in.
_A_EXPONENTS = range(2, -15, -1)
_DIGITS_FLOOR_64 = -np.log10(np.finfo(np.float64).eps)  # ~15.95
_DIGITS_FLOOR_32 = -np.log10(np.finfo(np.float32).eps)  # ~6.92
_MAX_DIGITS_LOST = 1.0


def _reference_tgo(r: float, v: float, a: float) -> Decimal:
    """Smallest positive root of R - V t - A t²/2 = 0 in 50-digit decimal."""
    getcontext().prec = 50
    rd, vd, ad = Decimal(r), Decimal(v), Decimal(a)
    disc = vd * vd + 2 * ad * rd
    if disc < 0:
        raise ValueError("reference requested outside the feasible branch")
    return 2 * rd / (vd + disc.sqrt())


def _digits(value: float, reference: Decimal) -> float:
    getcontext().prec = 50
    if reference == 0:
        raise ValueError("reference must be nonzero")
    rel = abs((Decimal(value) - reference) / reference)
    if rel == 0:
        return 50.0
    return float(-rel.log10())


def _sweep(dtype: np.dtype, digits_floor: float) -> tuple[list[list[str]], float, float]:
    rows: list[list[str]] = []
    worst_stable = np.inf
    worst_naive = np.inf
    for sign in (1.0, -1.0):
        for exp in _A_EXPONENTS:
            a = sign * 10.0**exp
            r_c = dtype.type(_R_LOS)
            v_c = dtype.type(_V_C)
            a_c = dtype.type(a)
            if float(v_c) ** 2 + 2.0 * float(a_c) * float(r_c) <= 0.0:
                continue  # guard branch (D < 0), verified separately below
            ref = _reference_tgo(float(r_c), float(v_c), float(a_c))
            res = time_to_go(r_c, v_c, a_c)
            t_stable, status = res.item()
            assert status is TgoStatus.OK, (a, status)
            naive = float(time_to_go_naive(r_c, v_c, a_c))
            d_stable = min(_digits(t_stable, ref), digits_floor)
            d_naive = (
                min(_digits(naive, ref), digits_floor) if np.isfinite(naive) else 0.0
            )
            worst_stable = min(worst_stable, d_stable)
            worst_naive = min(worst_naive, d_naive)
            rows.append([f"{a:.0e}", f"{d_stable:.1f}", f"{d_naive:.1f}"])
    return rows, worst_stable, worst_naive


def run_v6(output_dir: Path) -> VerificationReport:
    report = VerificationReport(
        task_id="V6",
        title="Guidance — stable vs textbook time-to-go as A_c → 0",
        criterion=(
            "the conjugate form (Eq. 4.16) losing more than 1 significant digit "
            "where the textbook form (Eq. 4.15) loses many"
        ),
        passed=True,
    )
    report.add_section(
        "Operating point",
        f"R_LOS = {_R_LOS:g} m, V̂_c = {_V_C:g} m/s, Â_c swept over ±10^{{{max(_A_EXPONENTS)}}} "
        f"… ±10^{{{min(_A_EXPONENTS)}}} m/s². Reference: 50-digit decimal evaluation from "
        "the exact binary inputs.",
    )

    results = {}
    for label, dtype, floor in (
        ("float64", np.dtype(np.float64), _DIGITS_FLOOR_64),
        ("float32", np.dtype(np.float32), _DIGITS_FLOOR_32),
    ):
        rows, worst_stable, worst_naive = _sweep(dtype, floor)
        results[label] = (worst_stable, worst_naive, floor)
        report.add_table(
            f"Significant digits retained ({label}, floor ≈ {floor:.1f})",
            ["Â_c (m/s²)", "stable Eq. 4.16", "naive Eq. 4.15"],
            rows,
        )
        write_csv(
            output_dir,
            f"v6-digits-{label}",
            ["a_c", "digits_stable", "digits_naive"],
            rows,
        )

    checks = []
    for label, (worst_stable, worst_naive, floor) in results.items():
        stable_ok = worst_stable >= floor - _MAX_DIGITS_LOST
        naive_bad = worst_naive <= floor - 5.0
        checks.append(stable_ok)
        naive_note = (
            "the catastrophic loss the criterion presupposes"
            if naive_bad
            else "unexpectedly accurate; the premise of V6 would need review"
        )
        report.add_section(
            f"Acceptance ({label})",
            f"Stable form worst case: **{worst_stable:.1f}** digits (criterion ≥ "
            f"{floor - _MAX_DIGITS_LOST:.1f}, i.e. ≤ 1 digit below the {floor:.1f} "
            f"precision floor) → {'**PASS**' if stable_ok else '**FAIL**'}. "
            f"Textbook form worst case: **{worst_naive:.1f}** digits — {naive_note}.",
        )
    report.passed = bool(all(checks))

    # --- guard and continuity checks (Prop. 4 / Remark 6) -----------------
    t_neg = time_to_go(_R_LOS, _V_C, -100.0)  # D < 0: linear fallback
    t_zero, s_zero = time_to_go(_R_LOS, _V_C, 0.0).item()
    lim = _R_LOS / _V_C
    guard_ok = (
        TgoStatus(int(t_neg.status)) is TgoStatus.LINEAR_FALLBACK
        and np.isfinite(float(t_neg.t_go))
        and s_zero is TgoStatus.OK
        and abs(t_zero - lim) < 1e-14 * lim
    )
    report.passed = bool(report.passed and guard_ok)
    report.add_section(
        "Non-intercept guard and continuity",
        f"D < 0 (Â_c = −100): status LINEAR_FALLBACK, t_go = {float(t_neg.t_go):.6f} s "
        f"(= R/V̂_c), no NaN propagation. At Â_c = 0 exactly: t_go = {t_zero:.15g} s "
        f"against the limit R/V̂_c = {lim:.15g} s. "
        f"{'**PASS**' if guard_ok else '**FAIL**'}.",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run verification task V6")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    report = run_v6(args.output)
    path = report.write(args.output, "v6-tgo")
    print(f"V6 {'PASS' if report.passed else 'FAIL'} -> {path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
