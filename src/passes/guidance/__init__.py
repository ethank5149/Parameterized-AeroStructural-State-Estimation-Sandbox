"""Guidance numerics: stable time-to-go and the AC-APN command law (Paper I, §4.3)."""

from __future__ import annotations

from passes.guidance.apn import apn_acceleration, los_rate
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
    "SCvxConfig",
    "SCvxResult",
    "SubproblemSolution",
    "TgoResult",
    "TgoStatus",
    "apn_acceleration",
    "linearize_trajectory",
    "los_rate",
    "solve_scvx",
    "solve_subproblem",
    "solve_subproblem_l2",
    "time_to_go",
    "time_to_go_naive",
]
