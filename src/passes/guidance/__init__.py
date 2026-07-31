"""Guidance numerics: stable time-to-go and the AC-APN command law (Paper I, §4.3)."""

from __future__ import annotations

from passes.guidance.apn import apn_acceleration, los_rate
from passes.guidance.tgo import TgoResult, TgoStatus, time_to_go, time_to_go_naive

__all__ = [
    "TgoResult",
    "TgoStatus",
    "apn_acceleration",
    "los_rate",
    "time_to_go",
    "time_to_go_naive",
]
