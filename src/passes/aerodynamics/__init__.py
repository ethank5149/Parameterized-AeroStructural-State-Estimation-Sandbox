"""Blended hypersonic pressure closure and panel loads (Paper II, §3.3)."""

from __future__ import annotations

from passes.aerodynamics.closure import (
    blended_pressure_coefficient,
    newtonian_pressure_coefficient,
    prandtl_meyer_angle,
    prandtl_meyer_pressure_coefficient,
    rayleigh_pitot_cp_max,
    smoothstep,
    vacuum_pressure_coefficient,
)
from passes.aerodynamics.panels import PanelModel, TrimSolution, curved_lifting_body

__all__ = [
    "PanelModel",
    "TrimSolution",
    "blended_pressure_coefficient",
    "curved_lifting_body",
    "newtonian_pressure_coefficient",
    "prandtl_meyer_angle",
    "prandtl_meyer_pressure_coefficient",
    "rayleigh_pitot_cp_max",
    "smoothstep",
    "vacuum_pressure_coefficient",
]

from passes.aerodynamics.tables import (
    AeroTable,
    Coefficients,
    PanelSolver,
    SweepGrid,
    SweepRun,
    console_progress,
)

__all__ += [
    "AeroTable",
    "Coefficients",
    "PanelSolver",
    "SweepGrid",
    "SweepRun",
    "console_progress",
]

from passes.aerodynamics.composite import (
    PatchedSolver,
    SkinFrictionModel,
    meridian_running_length,
)
from passes.aerodynamics.conical import (
    ConeSolution,
    ObliqueShock,
    mach_angle,
    maximum_cone_angle,
    oblique_shock,
    solve_cone,
    wedge_shock_angle,
)
from passes.aerodynamics.friction import (
    AdiabaticWall,
    BlasiusSolution,
    BoundaryLayer,
    FixedWall,
    RadiativeEquilibriumWall,
    adiabatic_wall_temperature,
    compressible_blasius,
    eckert_reference_temperature,
    laminar_skin_friction,
    recovery_factor,
    reference_temperature,
    turbulent_skin_friction,
)
from passes.aerodynamics.rarefied import (
    FreeMolecularSolver,
    free_molecular_coefficients,
    sine_squared_bridge,
    sphere_free_molecular_drag,
)
from passes.aerodynamics.realgas import (
    EquilibriumAir,
    NormalShock,
    perfect_gas_normal_shock,
)

__all__ += [
    "AdiabaticWall",
    "BlasiusSolution",
    "BoundaryLayer",
    "ConeSolution",
    "EquilibriumAir",
    "FixedWall",
    "FreeMolecularSolver",
    "NormalShock",
    "ObliqueShock",
    "PatchedSolver",
    "RadiativeEquilibriumWall",
    "SkinFrictionModel",
    "adiabatic_wall_temperature",
    "compressible_blasius",
    "eckert_reference_temperature",
    "free_molecular_coefficients",
    "laminar_skin_friction",
    "mach_angle",
    "maximum_cone_angle",
    "meridian_running_length",
    "oblique_shock",
    "perfect_gas_normal_shock",
    "recovery_factor",
    "reference_temperature",
    "sine_squared_bridge",
    "solve_cone",
    "sphere_free_molecular_drag",
    "turbulent_skin_friction",
    "wedge_shock_angle",
]
