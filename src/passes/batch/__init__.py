"""Batched Monte Carlo layer (Paper I, §5–§6).

The computational claim of the paper: because every replicate shares one
fixed-dimension, fixed-sparsity ODE, a Monte Carlo batch is a rank-3
tensor operation (replicate × state × stage) with no per-replicate
remesh. This package provides the array-backend abstraction (NumPy on
CPU, CuPy on CUDA), reproducible per-batch dispersion sampling, the
common-outer-grid batched integrator, and the terminal dispersion
statistics of §6 — :math:`R_{95}`, validity-checked CEP, bootstrap
intervals, and the Henze–Zirkler normality test.
"""

from __future__ import annotations

from passes.batch.backend import cuda_available, get_array_module, to_numpy
from passes.batch.dispersion import (
    DispersionReport,
    henze_zirkler,
    summarize_dispersion,
)
from passes.batch.entry_demo import EntryDispersionModel
from passes.batch.occupancy import (
    AchievedOccupancy,
    OccupancyReport,
    achieved_occupancy,
    device_limits,
    theoretical_occupancy,
)
from passes.batch.propagation import rk4_batch
from passes.batch.sampling import DispersionSpec, sample_dispersions

__all__ = [
    "AchievedOccupancy",
    "DispersionReport",
    "DispersionSpec",
    "EntryDispersionModel",
    "OccupancyReport",
    "achieved_occupancy",
    "cuda_available",
    "device_limits",
    "get_array_module",
    "henze_zirkler",
    "rk4_batch",
    "sample_dispersions",
    "summarize_dispersion",
    "theoretical_occupancy",
    "to_numpy",
]
