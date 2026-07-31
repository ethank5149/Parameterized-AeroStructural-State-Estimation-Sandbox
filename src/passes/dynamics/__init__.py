"""6-DOF state: attitude kinematics and deformed-surface incidence (Paper II, §3)."""

from __future__ import annotations

from passes.dynamics.attitude import (
    dcm_from_quaternion,
    quaternion_derivative,
    quaternion_norm_error,
)
from passes.dynamics.incidence import deformed_normal, local_incidence

__all__ = [
    "dcm_from_quaternion",
    "deformed_normal",
    "local_incidence",
    "quaternion_derivative",
    "quaternion_norm_error",
]
