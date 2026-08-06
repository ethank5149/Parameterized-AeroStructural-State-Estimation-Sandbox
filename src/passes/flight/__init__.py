"""Coupled single-trajectory flight simulator (Paper I §3.5; Paper II §7.2).

Assembles the structural, thermal, aerothermal, attitude and orbital
kernels into one system of ODEs with fixed dimension, advanced by one
integrator across every flight regime without a phase handoff.
"""

from __future__ import annotations

from passes.flight.ballistic_entry import (
    EXPONENTIAL_ATMOSPHERE_EARTH,
    MINIMUM_BALLISTIC_ENTRY_ANGLE,
    BallisticEntry,
    ExponentialAtmosphere,
    allen_eggers_velocity,
    ballistic_entry_range,
    peak_deceleration,
    peak_deceleration_altitude,
)
from passes.flight.simulator import FlightConfiguration, FlightResult, FlightSimulator
from passes.flight.state import GlobalState, StateLayout

__all__ = [
    "EXPONENTIAL_ATMOSPHERE_EARTH",
    "MINIMUM_BALLISTIC_ENTRY_ANGLE",
    "BallisticEntry",
    "ExponentialAtmosphere",
    "FlightConfiguration",
    "FlightResult",
    "FlightSimulator",
    "GlobalState",
    "StateLayout",
    "allen_eggers_velocity",
    "ballistic_entry_range",
    "peak_deceleration",
    "peak_deceleration_altitude",
]
