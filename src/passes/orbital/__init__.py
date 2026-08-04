"""Orbital mechanics and the coast phase (Paper II, §7)."""

from __future__ import annotations

from passes.orbital.coast import (
    CoastResult,
    RegimeTransition,
    StrategyComparison,
    compare_coast_strategies,
    orbital_elements,
    propagate_coast,
    regime_transition_profile,
    secular_rates,
)
from passes.orbital.fobs import (
    EARTH_ROTATION_RATE,
    DeorbitBurn,
    FobsProfile,
    approach_azimuth,
    azimuth_envelope,
    deorbit_burn,
    fobs_profile,
    ground_track,
    ground_track_shift,
)
from passes.orbital.gravity import (
    EARTH,
    GravityModel,
    gravitational_acceleration,
    gravitational_potential,
    j2_acceleration,
    specific_angular_momentum_z,
    specific_energy,
    two_body_acceleration,
)
from passes.orbital.lambert import (
    LambertSolution,
    lambert,
    minimum_energy_transfer,
)

__all__ = [
    "EARTH",
    "EARTH_ROTATION_RATE",
    "CoastResult",
    "DeorbitBurn",
    "FobsProfile",
    "GravityModel",
    "LambertSolution",
    "RegimeTransition",
    "StrategyComparison",
    "approach_azimuth",
    "azimuth_envelope",
    "compare_coast_strategies",
    "deorbit_burn",
    "fobs_profile",
    "gravitational_acceleration",
    "gravitational_potential",
    "ground_track",
    "ground_track_shift",
    "j2_acceleration",
    "lambert",
    "minimum_energy_transfer",
    "orbital_elements",
    "propagate_coast",
    "regime_transition_profile",
    "secular_rates",
    "specific_angular_momentum_z",
    "specific_energy",
    "two_body_acceleration",
]
