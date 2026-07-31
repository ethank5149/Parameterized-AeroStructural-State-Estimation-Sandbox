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
