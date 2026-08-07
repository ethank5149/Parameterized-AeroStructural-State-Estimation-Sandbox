"""Presentation-layer rendering for trajectory visualisation.

Kept separate from the physics packages on purpose. Nothing here feeds a
result; it exists so that notebooks stay thin and the rendering itself can
be tested rather than eyeballed.
"""

from __future__ import annotations

from passes.viz.globe import (
    DEFAULT_TEXTURE,
    Camera,
    load_texture,
    look_at,
    project,
    render,
    sun_direction,
)

__all__ = [
    "DEFAULT_TEXTURE",
    "Camera",
    "load_texture",
    "look_at",
    "project",
    "render",
    "sun_direction",
]
