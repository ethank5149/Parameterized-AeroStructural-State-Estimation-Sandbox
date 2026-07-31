"""II-V8 — aerothermal correlations (implementation-verification leg).

Paper II, §8: *"Stagnation heating against Fay–Riddell reference cases;
recession against a FIAT comparison. Failure criterion: heating
disagreement > 5% on reference conditions."*

**Scope of this run.** The executable leg verifies the *implementation*
of the correlations: exact scaling structure of Fay–Riddell through the
modified-Newtonian velocity gradient, the quantified Lewis-exponent
sensitivity, Lees continuity at the stagnation-region boundary, the
opposite-sign radius trade between convective and radiative heating
(with the interior blunting optimum demonstrated), and the leading-edge
recession balance limits. The comparison against *published* Fay–Riddell
reference cases — the stated 5% criterion — requires transcribed
reference data (tabulated conditions with equilibrium-air properties),
which this repository does not yet carry; that leg is **PENDING**, as is
the FIAT recession leg shared with I-V4. No number below is claimed as a
validation against flight or ground-test data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from passes.aerothermal import (
    TauberSuttonRadiation,
    fay_riddell,
    lees_distribution,
    newtonian_velocity_gradient,
    stefan_recession_rate,
    sutton_graves,
)
from passes.aerothermal.stagnation import (
    LEWIS_EXPONENT_EQUILIBRIUM,
    LEWIS_EXPONENT_FROZEN_CATALYTIC,
)
from passes.thermal.surface import STEFAN_BOLTZMANN
from passes.verification.common import VerificationReport, write_csv

__all__ = ["run_p2v8"]

_FR_BASE = {
    "edge_density": 0.05,
    "edge_viscosity": 6.0e-5,
    "wall_density": 0.3,
    "wall_viscosity": 4.0e-5,
    "total_enthalpy_edge": 2.0e7,
    "wall_enthalpy": 1.5e6,
    "dissociation_enthalpy": 5.0e6,
}


def run_p2v8(output_dir: Path) -> VerificationReport:
    report = VerificationReport(
        task_id="II-V8",
        title="Aerothermal correlations — implementation verification leg",
        criterion=(
            "heating disagreement > 5% on published reference conditions "
            "(unevaluated here — see scope); implementation leg: any scaling "
            "or continuity property failing its closed form"
        ),
        passed=True,
    )

    # --- Fay–Riddell structure --------------------------------------------
    checks: list[tuple[str, float, float, bool]] = []

    # R_eff^{-1/2} through the velocity gradient
    p_s, p_inf, rho_s = 5.0e4, 100.0, 0.05
    q_at = {}
    for r_eff in (0.25, 1.0):
        dudx = newtonian_velocity_gradient(r_eff, p_s, p_inf, rho_s)
        q_at[r_eff] = float(fay_riddell(velocity_gradient=float(dudx), **_FR_BASE))
    ratio = q_at[0.25] / q_at[1.0]
    checks.append(("q_conv ratio for R_eff 0.25 vs 1.0 (expect 4^{1/2} = 2)", ratio, 2.0,
                   abs(ratio - 2.0) < 1e-12))

    # driving-potential linearity
    dudx0 = float(newtonian_velocity_gradient(0.5, p_s, p_inf, rho_s))
    args = dict(_FR_BASE)
    q1 = float(fay_riddell(velocity_gradient=dudx0, **args))
    args["wall_enthalpy"] = 0.5 * (_FR_BASE["total_enthalpy_edge"] + _FR_BASE["wall_enthalpy"])
    q2 = float(fay_riddell(velocity_gradient=dudx0, **args))
    expect = (_FR_BASE["total_enthalpy_edge"] - args["wall_enthalpy"]) / (
        _FR_BASE["total_enthalpy_edge"] - _FR_BASE["wall_enthalpy"]
    )
    checks.append(("enthalpy-potential linearity ratio", q2 / q1, expect,
                   abs(q2 / q1 - expect) < 1e-12))

    # Lewis-exponent sensitivity (the paper: "differs by several percent")
    q_eq = float(fay_riddell(velocity_gradient=dudx0, **_FR_BASE,
                             lewis_exponent=LEWIS_EXPONENT_EQUILIBRIUM))
    q_fr = float(fay_riddell(velocity_gradient=dudx0, **_FR_BASE,
                             lewis_exponent=LEWIS_EXPONENT_FROZEN_CATALYTIC))
    lewis_delta = q_fr / q_eq - 1.0
    checks.append(("frozen/catalytic vs equilibrium bracket shift (expect ~1%)",
                   lewis_delta, 0.011, 0.002 < lewis_delta < 0.1))

    # Lees continuity at x = R_eff
    r_eff = 0.4
    inner = float(lees_distribution(1.0e6, 0.8, r_eff * (1 - 1e-9), r_eff))
    outer = float(lees_distribution(1.0e6, 0.8, r_eff * (1 + 1e-9), r_eff))
    checks.append(("Lees jump across x = R_eff (expect 0)",
                   abs(inner - outer) / inner, 0.0, abs(inner - outer) / inner < 1e-6))

    # leading-edge recession: exactly balanced wall must not recede
    t_w, eps = 2000.0, 0.85
    q_bal = eps * STEFAN_BOLTZMANN * t_w**4
    sdot0 = float(stefan_recession_rate(q_bal, t_w, eps, 0.0, 1900.0, 2.0e7))
    checks.append(("recession at radiative-equilibrium wall (expect 0)", sdot0, 0.0,
                   sdot0 == 0.0))

    rows = [[name, f"{got:.6g}", f"{want:.6g}", "yes" if ok else "NO"]
            for name, got, want, ok in checks]
    all_ok = all(ok for _, _, _, ok in checks)
    report.add_table(
        "Closed-form structure checks",
        ["check", "measured", "expected", "pass"],
        rows,
    )
    write_csv(
        output_dir,
        "p2v8-structure-checks",
        ["check", "measured", "expected", "pass"],
        [[name, got, want, ok] for name, got, want, ok in checks],
    )

    # --- the blunting trade -----------------------------------------------
    rho_inf, v_inf = 3.0e-4, 12000.0
    vv = np.linspace(9000.0, 16000.0, 15)
    ff = (vv / 9000.0) ** 8.5
    coeff = float(sutton_graves(rho_inf, 1.0, v_inf)) / (
        rho_inf**1.22 * float(np.interp(v_inf, vv, ff))
    )
    surrogate = TauberSuttonRadiation(
        vv, ff, coefficient=coeff,
        provenance=(
            "synthetic V^8.5 surrogate scaled to cross the convective curve at "
            "R_eff = 1 m; NOT the published Tauber–Sutton table"
        ),
    )
    radii = np.linspace(0.05, 3.0, 300)
    q_conv = np.asarray(sutton_graves(rho_inf, radii, v_inf))
    q_rad = surrogate.heat_flux(radii, rho_inf, np.full_like(radii, v_inf))
    total = q_conv + q_rad
    i_min = int(np.argmin(total))
    trade_ok = 0 < i_min < radii.size - 1
    report.add_section(
        "Opposite-sign radius trade (Paper II §4.2)",
        f"With convective heating falling as R_eff^(-1/2) and radiative rising as "
        f"R_eff^(+1), the total exhibits an interior optimum at "
        f"R_eff ≈ **{radii[i_min]:.2f} m** on the demonstration corridor → "
        f"{'**PASS**' if trade_ok else '**FAIL**'}. A framework modeling only "
        "convection would drive R_eff to the right edge of this sweep — the "
        "over-blunting bias the paper warns of. (Radiative component evaluated "
        "with a clearly-labeled synthetic velocity-function surrogate; the "
        "published Tauber–Sutton table is a required input with provenance, and "
        "the implementation refuses to extrapolate it.)",
    )
    write_csv(
        output_dir,
        "p2v8-blunting-trade",
        ["R_eff", "q_conv", "q_rad_surrogate", "q_total"],
        [[float(r), float(c), float(g), float(t)]
         for r, c, g, t in zip(radii, q_conv, q_rad, total, strict=True)],
    )

    # --- pending legs -------------------------------------------------------
    report.add_section(
        "Published reference cases and FIAT — PENDING",
        "The 5% criterion is stated against published Fay–Riddell reference "
        "conditions, which require transcribed equilibrium-air properties "
        "(ρ_e μ_e, ρ_w μ_w, h_D at the reference states); the FIAT recession "
        "comparison shares the pending status of I-V4. Neither dataset is in "
        "this repository, and no synthetic stand-in is presented as either. "
        "II-V8 is therefore **partially complete**: implementation verified, "
        "reference comparison pending data.",
    )

    report.passed = bool(all_ok and trade_ok)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run verification task II-V8 (implementation leg)"
    )
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    report = run_p2v8(args.output)
    path = report.write(args.output, "p2v8-aerothermal")
    print(f"II-V8 (implementation leg) {'PASS' if report.passed else 'FAIL'} -> {path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
