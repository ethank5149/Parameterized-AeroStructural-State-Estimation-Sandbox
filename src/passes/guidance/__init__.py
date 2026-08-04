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
from passes.guidance.cruise import (
    CruiseComparison,
    CruiseVehicle,
    crossover_mach,
    cruise_altitude,
    cruise_climb_altitude,
    cruise_dynamic_pressure,
    cruise_range,
    cruise_versus_glide,
    scramjet_specific_impulse,
)
from passes.guidance.entry import (
    DragTracker,
    EntryVehicle,
    GlideResult,
    GlideState,
    atmospheric_density,
    bank_reversal_needed,
    crossrange_deadband,
    range_to_go,
    simulate_glide,
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
    "CruiseComparison",
    "CruiseVehicle",
    "DeploymentPlan",
    "DragTracker",
    "EntryVehicle",
    "ExecutionErrorModel",
    "GlideResult",
    "GlideState",
    "Release",
    "SCvxConfig",
    "SCvxResult",
    "SubproblemSolution",
    "TgoResult",
    "TgoStatus",
    "apn_acceleration",
    "atmospheric_density",
    "bank_reversal_needed",
    "correction_maneuver",
    "crossover_mach",
    "crossrange_deadband",
    "cruise_altitude",
    "cruise_climb_altitude",
    "cruise_dynamic_pressure",
    "cruise_range",
    "cruise_versus_glide",
    "linearize_trajectory",
    "los_rate",
    "miss_sensitivity",
    "optimize_deployment_order",
    "plan_deployment",
    "range_to_go",
    "reachable_aimpoints",
    "schedule_corrections",
    "scramjet_specific_impulse",
    "simulate_glide",
    "solve_scvx",
    "solve_subproblem",
    "solve_subproblem_l2",
    "time_to_go",
    "time_to_go_naive",
]
