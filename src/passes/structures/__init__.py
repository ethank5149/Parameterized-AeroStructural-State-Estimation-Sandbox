"""Structural kernel: variable-rigidity beam, null-space BCs, modal solve.

Implements Paper I, §3.1–§3.2 and §3.6: the product-rule Euler–Bernoulli
stiffness operator on a Chebyshev grid, free-free boundary conditions by
null-space projection, the reduced generalized eigenproblem, and the
temporal-integration strategies whose comparison is verification task V3.
"""

from __future__ import annotations

from passes.structures.beam import BeamOperators, assemble_beam
from passes.structures.boundary import (
    FreeFreeProjection,
    free_free_constraints,
    project_free_free,
)
from passes.structures.integrators import (
    ModalPropagator,
    NewmarkIntegrator,
    explicit_dt_limit,
)
from passes.structures.modal import (
    ModalSolution,
    free_free_analytic_frequencies,
    solve_free_free_modes,
)
from passes.structures.profiles import (
    MaterialProfile,
    stepped_profile,
    uniform_profile,
)

__all__ = [
    "BeamOperators",
    "FreeFreeProjection",
    "MaterialProfile",
    "ModalPropagator",
    "ModalSolution",
    "NewmarkIntegrator",
    "assemble_beam",
    "explicit_dt_limit",
    "free_free_analytic_frequencies",
    "free_free_constraints",
    "project_free_free",
    "solve_free_free_modes",
    "stepped_profile",
    "uniform_profile",
]
