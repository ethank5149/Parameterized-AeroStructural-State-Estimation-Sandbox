"""Run all implemented verification tasks: ``python -m passes.verification``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from passes.verification.p2v1_ultraspherical import run_p2v1
from passes.verification.p2v8_aerothermal import run_p2v8
from passes.verification.v1_structural import run_v1
from passes.verification.v2_slosh import run_v2
from passes.verification.v3_integrators import run_v3
from passes.verification.v4_thermal import run_v4
from passes.verification.v5_filter import run_v5
from passes.verification.v6_tgo import run_v6
from passes.verification.v7_dispersion import run_v7
from passes.verification.v8_throughput import run_v8


def main() -> int:
    parser = argparse.ArgumentParser(description="Run verification tasks V1–V8")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    all_passed = True
    for runner, stem in (
        (run_v1, "v1-structural"),
        (run_v2, "v2-slosh"),
        (run_v3, "v3-integrators"),
        (run_v4, "v4-thermal"),
        (run_v5, "v5-filter"),
        (run_v6, "v6-tgo"),
        (run_v7, "v7-dispersion"),
        (run_v8, "v8-throughput"),
        (run_p2v1, "p2v1-ultraspherical"),
        (run_p2v8, "p2v8-aerothermal"),
    ):
        report = runner(args.output)
        path = report.write(args.output, stem)
        print(f"{report.task_id} {'PASS' if report.passed else 'FAIL'} -> {path}")
        all_passed = all_passed and report.passed
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
