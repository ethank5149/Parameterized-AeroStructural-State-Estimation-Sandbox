"""Fully implicit ablation and thermal response of a multilayer TPS stack.

An independent implementation of the formulation published in

* Y.-K. Chen and F. S. Milos, "Ablation and Thermal Response Program for
  Spacecraft Heatshield Analysis," *J. Spacecraft and Rockets* **36**(3),
  1999, pp. 475–483, doi:10.2514/2.3469;
* F. S. Milos, Y.-K. Chen and T. H. Squire, "Updated Ablation and Thermal
  Response Program for Spacecraft Heatshield Analysis," TFAWS06-1008,
  17th Thermal and Fluids Analysis Workshop, 2006.

FIAT itself is US-government-controlled software. It is not used,
invoked, or reproduced here; this package is written from the governing
equations and numerical description in the two open-literature papers
above, which is why it exists — a code whose verification criterion is
stated against FIAT cannot be closed without an independent solver of
the same equations.

Comparisons produced with this package are **code-to-code
cross-verification against an independent implementation of FIAT's
formulation**, and must be reported as such. They are not FIAT results.
"""

from __future__ import annotations

from passes.thermal.fiat.analysis import (
    DepthProbe,
    InterfaceHistory,
    ThicknessResult,
    interface_histories,
    optimize_ply_thickness,
    probe_depths,
    scale_environments,
    sized_stack,
)
from passes.thermal.fiat.bprime import BPrimeTable, TableRangeError
from passes.thermal.fiat.kinetics import (
    TgaTargets,
    calibrated_components,
    fit_arrhenius,
    peak_rate_temperature,
    tga_mass_fraction,
)
from passes.thermal.fiat.materials import (
    HERITAGE_PICA_CONDUCTIVITY,
    MEDLI2_PICA_CONDUCTIVITY,
    PressureConductivity,
    pica_like_material,
    structural_material,
)
from passes.thermal.fiat.radiation import (
    gray_radiative_flux,
    optical_depth,
    rosseland_conductivity,
    rosseland_flux,
)
from passes.thermal.fiat.solver import (
    FiatSolution,
    FiatSolver,
    FiatStep,
    SolverOptions,
)
from passes.thermal.fiat.stack import MaterialStack, Ply, StackGrid
from passes.thermal.fiat.surface import (
    AerothermalEnvironment,
    BackfaceCondition,
    BackfaceKind,
    SurfaceState,
    blowing_reduction,
    solve_surface,
)

__all__ = [
    "HERITAGE_PICA_CONDUCTIVITY",
    "MEDLI2_PICA_CONDUCTIVITY",
    "AerothermalEnvironment",
    "BPrimeTable",
    "BackfaceCondition",
    "BackfaceKind",
    "DepthProbe",
    "FiatSolution",
    "FiatSolver",
    "FiatStep",
    "InterfaceHistory",
    "MaterialStack",
    "Ply",
    "PressureConductivity",
    "SolverOptions",
    "StackGrid",
    "SurfaceState",
    "TableRangeError",
    "TgaTargets",
    "ThicknessResult",
    "blowing_reduction",
    "calibrated_components",
    "fit_arrhenius",
    "gray_radiative_flux",
    "interface_histories",
    "optical_depth",
    "optimize_ply_thickness",
    "peak_rate_temperature",
    "pica_like_material",
    "probe_depths",
    "rosseland_conductivity",
    "rosseland_flux",
    "scale_environments",
    "sized_stack",
    "solve_surface",
    "structural_material",
    "tga_mass_fraction",
]
