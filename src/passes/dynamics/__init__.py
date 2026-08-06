"""6-DOF state: attitude kinematics and deformed-surface incidence (Paper II, §3)."""

from __future__ import annotations

from passes.dynamics.attitude import (
    dcm_from_quaternion,
    quaternion_derivative,
    quaternion_norm_error,
)
from passes.dynamics.incidence import deformed_normal, local_incidence
from passes.dynamics.roll_resonance import (
    ResonanceCrossing,
    pitch_frequency,
    resonance_condition_ratio,
    resonance_crossings,
    roll_rate,
    steady_state_roll_rate,
    trim_amplification,
)

__all__ = [
    "ResonanceCrossing",
    "dcm_from_quaternion",
    "deformed_normal",
    "local_incidence",
    "pitch_frequency",
    "quaternion_derivative",
    "quaternion_norm_error",
    "resonance_condition_ratio",
    "resonance_crossings",
    "roll_rate",
    "steady_state_roll_rate",
    "trim_amplification",
]
