"""Ambient gas state from the ground to the exosphere.

The 1976 standard below 86 km, NRLMSIS above, blended across the seam, plus
the similarity parameters — Mach, Reynolds, Knudsen — that decide which
aerodynamic theory is applicable at a given point of a trajectory.
"""

from __future__ import annotations

from passes.atmosphere.model import (
    Atmosphere,
    Freestream,
    LayeredAtmosphere,
    TabulatedAtmosphere,
    earth_atmosphere,
    tabulate,
)
from passes.atmosphere.wind import (
    NoWind,
    TabulatedWind,
    WindField,
    relative_velocity,
    wind_incidence,
)
from passes.atmosphere.standard import (
    AVOGADRO,
    BOLTZMANN,
    COLLISION_DIAMETER,
    EARTH_RADIUS_EFFECTIVE,
    GAMMA_AIR,
    MOLAR_MASS_SEA_LEVEL,
    SEA_LEVEL_GRAVITY,
    TOP_OF_LOWER_ATMOSPHERE,
    UNIVERSAL_GAS_CONSTANT,
    AtmosphereState,
    USStandard1976,
    geometric_altitude,
    geopotential_altitude,
    gravity,
)
from passes.atmosphere.upper import (
    MODERATE_ACTIVITY,
    SOLAR_MAXIMUM,
    SOLAR_MINIMUM,
    MSISAtmosphere,
    SolarActivity,
)

__all__ = [
    "AVOGADRO",
    "BOLTZMANN",
    "COLLISION_DIAMETER",
    "EARTH_RADIUS_EFFECTIVE",
    "GAMMA_AIR",
    "MODERATE_ACTIVITY",
    "MOLAR_MASS_SEA_LEVEL",
    "MSISAtmosphere",
    "SEA_LEVEL_GRAVITY",
    "SOLAR_MAXIMUM",
    "SOLAR_MINIMUM",
    "TOP_OF_LOWER_ATMOSPHERE",
    "UNIVERSAL_GAS_CONSTANT",
    "Atmosphere",
    "AtmosphereState",
    "Freestream",
    "LayeredAtmosphere",
    "NoWind",
    "SolarActivity",
    "TabulatedAtmosphere",
    "TabulatedWind",
    "WindField",
    "USStandard1976",
    "earth_atmosphere",
    "geometric_altitude",
    "geopotential_altitude",
    "gravity",
    "relative_velocity",
    "tabulate",
    "wind_incidence",
]
