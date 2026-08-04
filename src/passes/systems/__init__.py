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

__all__ = [
    "NAMED_ARCHITECTURES",
    "Architecture",
    "LegBudget",
    "MissionBudget",
    "MissionRequest",
    "Payload",
    "Phase",
    "PhaseRegime",
    "describe",
    "enumerate_architectures",
    "evaluate",
    "validate",
]
