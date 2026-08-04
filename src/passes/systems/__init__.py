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

__all__ = [
    "NAMED_ARCHITECTURES",
    "Architecture",
    "Payload",
    "Phase",
    "PhaseRegime",
    "describe",
    "enumerate_architectures",
    "validate",
]
