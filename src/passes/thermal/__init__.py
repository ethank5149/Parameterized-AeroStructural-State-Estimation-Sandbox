"""Charring-ablation thermal kernel on a fixed computational domain.

Implements Paper I, §3.4: three-component Arrhenius decomposition
kinetics (Eq. 3.14), the Landau transformation rendering the recession
front stationary in computational coordinates (Eqs. 3.15–3.16), the
in-depth energy equation with pyrolysis-gas convection and grid-velocity
advection (Eq. 3.17), spectral integration of the gas-flux continuity
equation (Eq. 3.18), and the surface energy balance with the
cancellation-safe blowing correction (Eqs. 3.19–3.20).
"""

from __future__ import annotations

from passes.thermal.kinetics import bulk_density, decomposition_rate, degree_of_char
from passes.thermal.landau import LandauFrame
from passes.thermal.material import (
    ArrheniusComponent,
    CharringMaterial,
    LinearBlendProperty,
    demo_material,
)
from passes.thermal.solver import CharringThermalSolver, ThermalState
from passes.thermal.surface import (
    SurfaceEnergyBalance,
    SurfaceEnvironment,
    SurfaceThermochemistry,
    blowing_correction,
)

__all__ = [
    "ArrheniusComponent",
    "CharringMaterial",
    "CharringThermalSolver",
    "LandauFrame",
    "LinearBlendProperty",
    "SurfaceEnergyBalance",
    "SurfaceEnvironment",
    "SurfaceThermochemistry",
    "ThermalState",
    "blowing_correction",
    "bulk_density",
    "decomposition_rate",
    "degree_of_char",
    "demo_material",
]
