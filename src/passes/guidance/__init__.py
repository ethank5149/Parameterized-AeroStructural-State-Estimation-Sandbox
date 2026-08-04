"""Guidance numerics: stable time-to-go and the AC-APN command law (Paper I, §4.3)."""

from __future__ import annotations

from passes.guidance.apn import apn_acceleration, los_rate
from passes.guidance.bus import (
    Aimpoint,
    DeploymentPlan,
    Release,
    optimize_deployment_order,
    plan_deployment,
    reachable_aimpoints,
)
from passes.guidance.midcourse import (
    CorrectionPlan,
    CorrectionResult,
    ExecutionErrorModel,
    correction_maneuver,
    miss_sensitivity,
    schedule_corrections,
)
from passes.guidance.scvx import (
    SCvxConfig,
    SCvxResult,
    SubproblemSolution,
    linearize_trajectory,
    solve_scvx,
    solve_subproblem,
    solve_subproblem_l2,
)
from passes.guidance.tgo import TgoResult, TgoStatus, time_to_go, time_to_go_naive

__all__ = [
    "Aimpoint",
    "CorrectionPlan",
    "CorrectionResult",
    "DeploymentPlan",
    "ExecutionErrorModel",
    "Release",
    "SCvxConfig",
    "SCvxResult",
    "SubproblemSolution",
    "TgoResult",
    "TgoStatus",
    "apn_acceleration",
    "correction_maneuver",
    "linearize_trajectory",
    "los_rate",
    "miss_sensitivity",
    "optimize_deployment_order",
    "plan_deployment",
    "reachable_aimpoints",
    "schedule_corrections",
    "solve_scvx",
    "solve_subproblem",
    "solve_subproblem_l2",
    "time_to_go",
    "time_to_go_naive",
]
