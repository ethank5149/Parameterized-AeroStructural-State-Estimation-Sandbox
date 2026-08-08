"""Presentation-layer rendering for trajectory visualisation.

Kept separate from the physics packages on purpose. Nothing here feeds a
result; it exists so that notebooks stay thin and the rendering itself can
be tested rather than eyeballed.

Layered, so each piece can be tested against something:

* :mod:`~passes.viz.history` — :class:`SimulationHistory`, the one record a
  renderer reads. Removes the second trajectory model a notebook used to
  carry.
* :mod:`~passes.viz.globe` — ray-traced sphere, projection, depth test.
* :mod:`~passes.viz.scene` — pure drawing primitives over history samples.
* :mod:`~passes.viz.animator` — :class:`TrajectoryAnimator`, the façade a
  notebook calls.
"""

from __future__ import annotations

from passes.viz.animator import Frame, TrajectoryAnimator, video_writer
from passes.viz.ellipsoid import (
    WGS84,
    Ellipsoid,
    ecef_to_geodetic,
    geodetic_to_ecef,
    local_vertical,
    ray_ellipsoid,
)
from passes.viz.globe import (
    DEFAULT_TEXTURE,
    Camera,
    load_texture,
    look_at,
    project,
    render,
    sun_direction,
    to_device,
)
from passes.viz.history import SimulationHistory
from passes.viz.imagery import BlueMarble, default_blue_marble
from passes.viz.pacing import (
    PacingProfile,
    PacingWeights,
    attention_density,
    uniform_pacing,
)
from passes.viz.scene import (
    NOSE_AXIS,
    ChaseRig,
    SceneStyle,
    draw_horizon_ring,
    draw_marker,
    draw_sites,
    draw_track,
    draw_vehicle,
    ease,
    geodetic_to_cartesian,
    globe_plate,
    glyph_polylines,
    glyph_world,
    horizon_ring,
    site_status,
    starfield,
)
from passes.viz.terrain import ElevationSample, Terrain, default_terrain

__all__ = [
    "DEFAULT_TEXTURE",
    "NOSE_AXIS",
    "WGS84",
    "BlueMarble",
    "Camera",
    "ChaseRig",
    "ElevationSample",
    "Ellipsoid",
    "Frame",
    "PacingProfile",
    "PacingWeights",
    "SceneStyle",
    "SimulationHistory",
    "Terrain",
    "TrajectoryAnimator",
    "attention_density",
    "default_blue_marble",
    "default_terrain",
    "draw_horizon_ring",
    "draw_marker",
    "draw_sites",
    "draw_track",
    "draw_vehicle",
    "ease",
    "ecef_to_geodetic",
    "geodetic_to_cartesian",
    "geodetic_to_ecef",
    "globe_plate",
    "glyph_polylines",
    "glyph_world",
    "horizon_ring",
    "load_texture",
    "local_vertical",
    "look_at",
    "project",
    "ray_ellipsoid",
    "render",
    "site_status",
    "starfield",
    "sun_direction",
    "to_device",
    "uniform_pacing",
    "video_writer",
]
