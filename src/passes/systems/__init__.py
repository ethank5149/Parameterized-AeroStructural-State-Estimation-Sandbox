"""System-level composition: which phase sequences form admissible systems."""

from passes.systems.architecture import (
    NAMED_ARCHITECTURES,
    Architecture,
    Payload,
    Phase,
    PhaseRegime,
    describe,
    enumerate_architectures,
    validate,
)
from passes.systems.budget import (
    LegBudget,
    MissionBudget,
    MissionRequest,
    evaluate,
)
from passes.systems.dispersion import (
    CEP_OVER_SIGMA,
    R95_OVER_SIGMA,
    AccuracyStatistics,
    accuracy_statistics,
    containment_probability,
    containment_radius,
    containment_ratio,
)

__all__ = [
    "CEP_OVER_SIGMA",
    "NAMED_ARCHITECTURES",
    "R95_OVER_SIGMA",
    "AccuracyStatistics",
    "Architecture",
    "LegBudget",
    "MissionBudget",
    "MissionRequest",
    "Payload",
    "Phase",
    "PhaseRegime",
    "accuracy_statistics",
    "containment_probability",
    "containment_radius",
    "containment_ratio",
    "describe",
    "enumerate_architectures",
    "evaluate",
    "validate",
]
