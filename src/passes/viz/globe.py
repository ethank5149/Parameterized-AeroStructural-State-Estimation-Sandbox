"""Ray-traced globe rendering, for trajectory visualisation.

Why this is not a Matplotlib surface plot
-----------------------------------------

The obvious way to draw an Earth in Matplotlib is ``plot_surface`` with a
texture passed as ``facecolors``. It works, and it is why the first version
of the campaign animation looked the way it did: **each mesh quad is filled
with one flat colour**, so the visible resolution is the *mesh* resolution,
not the texture's. A 4096x2048 Blue Marble image downsampled onto a 256x128
mesh throws away 99.6 % of its pixels and leaves visible facets on every
coastline.

Raising the mesh resolution does not fix it. The cost grows as the square,
Matplotlib sorts every quad by depth with a painter's algorithm, and once
trajectory lines are added the sort produces the other familiar artefact:
arcs that pass *through* the planet appearing in front of it, because
Matplotlib compares whole artists rather than fragments.

So this module does the direct thing instead. For every output pixel it
intersects a camera ray with the body, converts the hit point to a geodetic
latitude and longitude, and samples the imagery bilinearly. The result is
limited by the texture and the output size, never by a mesh; occlusion is a
per-point depth test rather than an artist sort; and the whole thing is
vectorised NumPy, so a 1600x900 frame takes tens of milliseconds.

The body is an ellipsoid
------------------------

It used to be a sphere, and that was self-consistent and wrong. The
:mod:`~passes.viz.ellipsoid` note has the measurement: treating geodetic
latitude as geocentric displaces a surface point at 45 degrees by 21.4 km.
The intersection is solved in the scaled space where the ellipsoid *is* the
unit sphere, so it costs the same as the sphere version did, and the
surface normal is the true geodetic vertical rather than the position
direction.

A sphere is still available — pass a float radius, or an
:class:`~passes.viz.ellipsoid.Ellipsoid` with zero flattening — because a
history built on a spherical model must be drawn on the sphere it was built
on. Mixing an ellipsoidal picture with spherical physics moves the ground
out from under the trajectory, which is the failure this whole layer exists
to avoid.

Textures carry their own footprint
----------------------------------

Every image handed in is a :class:`~passes.viz.imagery.Texture`: pixels plus
the degree box they cover. That is what lets a full-globe mosaic and a
native-resolution launch-pad crop be composited in one pass — the renderer
samples the finest texture that covers each pixel and falls back outward,
so a close-up gets 15-arc-second Blue Marble where it has it and the global
mosaic everywhere else, with a feathered join rather than a visible tile
edge.

What is modelled, and what is decoration
----------------------------------------

Two things here are **data**:

* **Relief shading** from GMTED2010. The surface normal is perturbed by the
  measured terrain slope, so the Himalaya and the Andes are lit as they
  are lit, not stippled on.
* **Terrain displacement**, optionally: the ray is marched onto the real
  elevation instead of onto the reference ellipsoid. Off by default,
  because at orbital range Everest is under a pixel; on for close-ups,
  where it is the whole picture.

The rest is presentation and is calibrated against nothing:

* Lambertian diffuse with a soft terminator, because a hard day/night edge
  on a smooth body reads as a rendering bug.
* Night side as a dimmed, blue-shifted day texture. Real night lights would
  need a second image.
* An atmospheric limb, the cheapest cue that the object is a planet with
  air rather than a billiard ball.
* A specular highlight weighted toward dark texels so land does not gleam.

Where it runs
-------------

Every per-pixel expression below is written against the array API that
NumPy and CuPy share, so :func:`render` takes the same ``backend`` argument
as the batched integrator in :mod:`passes.batch.backend` and runs on either
device. Textures must already live on the requested backend — see
:func:`to_device` — because re-uploading them once per frame would cost more
than the render it was meant to accelerate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from passes.batch.backend import Backend, get_array_module, to_numpy
from passes.viz.ellipsoid import WGS84, Ellipsoid, ray_ellipsoid
from passes.viz.imagery import Texture
from passes.viz.terrain import ReliefMap

__all__ = [
    "DEFAULT_TEXTURE",
    "Camera",
    "as_ellipsoid",
    "load_texture",
    "look_at",
    "project",
    "render",
    "sun_direction",
    "to_device",
]

_FloatArray = NDArray[np.float64]


def _default_texture() -> Path:
    """Locate ``assets/blue_marble.jpg`` relative to the installed package.

    Resolved by walking up from this module rather than from the working
    directory, because a notebook runs from its own folder and a test from
    the repository root. A path relative to the process CWD works in one of
    those and fails in the other, which is exactly how the first version of
    this broke.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "assets" / "blue_marble.jpg"
        if candidate.is_file():
            return candidate
    return Path("assets/blue_marble.jpg")


#: The packaged legacy Blue Marble texture: equirectangular, north-up,
#: longitude -180 at column zero. Superseded by
#: :func:`~passes.viz.imagery.default_blue_marble`, which reads the Next
#: Generation GeoTIFFs at 15 arc-seconds; this remains the fallback for an
#: install without the reference archive.
DEFAULT_TEXTURE = _default_texture()


def as_ellipsoid(surface: Ellipsoid | float) -> Ellipsoid:
    """Coerce a radius or an ellipsoid to an :class:`Ellipsoid`.

    A bare float means a sphere, and it means it exactly: zero flattening,
    so every ellipsoidal expression downstream degenerates to the spherical
    one rather than being special-cased.
    """
    if isinstance(surface, Ellipsoid):
        return surface
    radius = float(surface)
    if not (np.isfinite(radius) and radius > 0.0):
        msg = f"radius must be finite and positive, got {surface}"
        raise ValueError(msg)
    return Ellipsoid(semi_major=radius, flattening=0.0, name="sphere")


def load_texture(path: str | Path = DEFAULT_TEXTURE) -> Texture:
    """Load an equirectangular image as a global :class:`Texture`.

    Raises
    ------
    FileNotFoundError
        With a message naming what the file should be, since a missing
        texture is a setup problem rather than a bug and the fix is to
        supply an equirectangular image.
    """
    location = Path(path)
    if not location.is_file():
        msg = (
            f"no globe texture at {location}. Expected an equirectangular "
            "(plate carree) image, north-up, spanning -180..180 in longitude "
            "and -90..90 in latitude; NASA's Blue Marble at 4096x2048 is what "
            "this was built against."
        )
        raise FileNotFoundError(msg)
    from PIL import Image

    with Image.open(location) as handle:
        return Texture(np.asarray(handle.convert("RGB"), dtype=np.uint8))


def to_device(
    texture: Texture | Sequence[Texture] | ReliefMap | Any, backend: Backend = "numpy"
) -> Any:
    """Place a texture, a stack of them, or a relief map on a backend, once.

    Kept explicit rather than done inside :func:`render` because the upload
    is the expensive part: an 8192x4096 BMNG mosaic is 100 MB as ``uint8``
    and the relief grids another 33 MB apiece, and moving them per frame
    would swamp the render they were meant to speed up. Upload at set-up,
    keep the handle, pass it to every frame.

    ``uint8`` stays ``uint8`` — the renderer normalises when it samples, and
    a float64 copy of the same mosaic is 800 MB.
    """
    xp = get_array_module(backend)
    if isinstance(texture, Texture):
        return texture.with_data(xp.asarray(texture.data))
    if isinstance(texture, ReliefMap):
        return ReliefMap(
            elevation=xp.asarray(texture.elevation),
            slope_east=xp.asarray(texture.slope_east),
            slope_north=xp.asarray(texture.slope_north),
            exaggeration=texture.exaggeration,
        )
    if isinstance(texture, (list, tuple)):
        return [to_device(item, backend) for item in texture]
    return xp.asarray(texture)


@dataclass(frozen=True)
class Camera:
    """A pinhole camera looking at a point.

    Attributes
    ----------
    position:
        Eye position in the same frame as the geometry (m).
    target:
        Point the camera looks at (m).
    up:
        World up hint; the true up is re-orthogonalised against the view
        direction, so this only has to be non-parallel to it.
    fov:
        Vertical field of view (rad).
    width, height:
        Output size in pixels.
    """

    position: _FloatArray
    target: _FloatArray
    up: _FloatArray
    fov: float = np.deg2rad(35.0)
    width: int = 1280
    height: int = 720

    def basis(self) -> tuple[_FloatArray, _FloatArray, _FloatArray]:
        """Right, true-up and forward unit vectors.

        Degenerate cases are handled rather than left to produce NaNs: if
        the supplied up is parallel to the view direction the camera would
        have no defined roll, so a fallback axis is substituted.
        """
        forward = np.asarray(self.target, dtype=np.float64) - np.asarray(
            self.position, dtype=np.float64
        )
        norm = float(np.linalg.norm(forward))
        if norm == 0.0:
            msg = "camera position and target coincide, so no view direction is defined"
            raise ValueError(msg)
        forward = forward / norm
        hint = np.asarray(self.up, dtype=np.float64)
        right = np.cross(forward, hint)
        if float(np.linalg.norm(right)) < 1e-12:
            hint = np.array([0.0, 0.0, 1.0])
            right = np.cross(forward, hint)
            if float(np.linalg.norm(right)) < 1e-12:
                right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        right = right / float(np.linalg.norm(right))
        true_up = np.cross(right, forward)
        return right, true_up / float(np.linalg.norm(true_up)), forward

    def ground_resolution(self, distance: float) -> float:
        """Metres per pixel on a surface ``distance`` metres away, face on.

        What a caller compares against
        :attr:`~passes.viz.imagery.Texture.ground_resolution` to decide
        whether a native-resolution crop is worth reading, or whether the
        global mosaic already has more detail than the frame can show.
        """
        return float(2.0 * distance * np.tan(0.5 * self.fov) / max(self.height, 1))


def look_at(
    target: _FloatArray,
    distance: float,
    azimuth: float,
    elevation: float,
    up: _FloatArray | None = None,
    **kwargs: object,
) -> Camera:
    """Build a camera orbiting ``target`` at a given range and direction.

    ``azimuth`` is measured about the world z axis from the world x axis,
    and ``elevation`` above the world xy plane — the usual orbit controls.
    """
    centre = np.asarray(target, dtype=np.float64)
    offset = distance * np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )
    return Camera(
        position=centre + offset,
        target=centre,
        up=np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, dtype=np.float64),
        **kwargs,  # type: ignore[arg-type]
    )


def sun_direction(hour_angle: float, declination: float = 0.0) -> _FloatArray:
    """Unit vector toward the sun in the body-fixed frame.

    Parameters
    ----------
    hour_angle:
        Longitude of the subsolar point (rad).
    declination:
        Latitude of the subsolar point (rad); 0 is equinox.
    """
    return np.array(
        [
            np.cos(declination) * np.cos(hour_angle),
            np.cos(declination) * np.sin(hour_angle),
            np.sin(declination),
        ]
    )


# -- sampling ------------------------------------------------------------


def _bilinear(
    grid: Any,
    row: Any,
    column: Any,
    wraps: bool,
    xp: ModuleType,
) -> Any:
    """Bilinear lookup at fractional ``(row, column)`` in pixel-centre units.

    ``wraps`` applies to the column axis only. Rows are always clamped: a
    grid stops at the poles, and wrapping there would sample the opposite
    hemisphere.
    """
    rows, cols = int(grid.shape[0]), int(grid.shape[1])
    r = xp.clip(row, 0.0, rows - 1.0)
    r0 = xp.clip(xp.floor(r).astype(xp.int64), 0, rows - 1)
    r1 = xp.minimum(r0 + 1, rows - 1)
    fr = r - r0

    if wraps:
        c0 = xp.mod(xp.floor(column).astype(xp.int64), cols)
        c1 = xp.mod(c0 + 1, cols)
        fc = column - xp.floor(column)
    else:
        c = xp.clip(column, 0.0, cols - 1.0)
        c0 = xp.clip(xp.floor(c).astype(xp.int64), 0, cols - 1)
        c1 = xp.minimum(c0 + 1, cols - 1)
        fc = c - c0

    if grid.ndim == 3:
        fr, fc = fr[..., None], fc[..., None]
    top = grid[r0, c0] * (1.0 - fc) + grid[r0, c1] * fc
    bottom = grid[r1, c0] * (1.0 - fc) + grid[r1, c1] * fc
    return top * (1.0 - fr) + bottom * fr


def _pixel_coordinates(
    texture: Texture, latitude_deg: Any, longitude_deg: Any, xp: ModuleType
) -> tuple[Any, Any]:
    """Fractional row and column of a degree point, in pixel-centre units.

    The bounds are pixel **edges**, so the centre of row 0 is half a pixel
    inside ``north``. Getting this off by half a pixel was worth 2.4 km on
    the old 4096-row texture.
    """
    rows, cols = texture.shape
    d_lat = (texture.north - texture.south) / rows
    d_lon = (texture.east - texture.west) / cols
    row = (texture.north - latitude_deg) / d_lat - 0.5
    if texture.wraps:
        column = xp.mod(longitude_deg - texture.west, 360.0) / d_lon - 0.5
    else:
        column = (longitude_deg - texture.west) / d_lon - 0.5
    return row, column


def _sample_texture(
    texture: Texture, latitude_deg: Any, longitude_deg: Any, xp: ModuleType
) -> Any:
    """Bilinear RGB lookup, normalised to ``[0, 1]``."""
    row, column = _pixel_coordinates(texture, latitude_deg, longitude_deg, xp)
    colour = _bilinear(texture.data, row, column, texture.wraps, xp)
    # dtype is a NumPy dtype even on a device array, so this test is the
    # same on both backends.
    if not np.issubdtype(texture.data.dtype, np.floating):
        return colour / 255.0
    return colour


def _inside(texture: Texture, latitude_deg: Any, longitude_deg: Any, xp: ModuleType) -> Any:
    """Mask of points strictly inside a texture's box, with a half-pixel margin."""
    if texture.wraps:
        return (latitude_deg >= texture.south) & (latitude_deg <= texture.north)
    lon = xp.mod(longitude_deg - texture.west, 360.0) + texture.west
    return (
        (latitude_deg >= texture.south)
        & (latitude_deg <= texture.north)
        & (lon >= texture.west)
        & (lon <= texture.east)
    )


def _edge_fade(
    texture: Texture, latitude_deg: Any, longitude_deg: Any, feather: float, xp: ModuleType
) -> Any:
    """Ramp from 0 at a texture's border to 1 ``feather`` degrees inside.

    Without it the join between a 15-arc-second crop and a global mosaic is
    a hard rectangle, which reads as a rendering artefact rather than as
    detail. The ramp does not invent pixels; it cross-fades between two
    measurements of the same ground.
    """
    if texture.wraps or feather <= 0.0:
        return _inside(texture, latitude_deg, longitude_deg, xp).astype(xp.float64)
    lon = xp.mod(longitude_deg - texture.west, 360.0) + texture.west
    distance = xp.minimum(
        xp.minimum(latitude_deg - texture.south, texture.north - latitude_deg),
        xp.minimum(lon - texture.west, texture.east - lon),
    )
    return xp.clip(distance / feather, 0.0, 1.0)


def _rays(camera: Camera, xp: ModuleType) -> tuple[_FloatArray, Any]:
    """Origin and per-pixel unit directions for the camera.

    The origin stays on the host — it is three numbers, and keeping it there
    lets the ray coefficients be plain Python floats on either backend.
    """
    right, up, forward = camera.basis()
    aspect = camera.width / camera.height
    half_h = np.tan(0.5 * camera.fov)
    half_w = half_h * aspect
    # Pixel centres, y increasing downward so row 0 is the top of the image.
    xs = xp.linspace(-half_w, half_w, camera.width)
    ys = xp.linspace(half_h, -half_h, camera.height)
    grid_x, grid_y = xp.meshgrid(xs, ys)
    directions = (
        xp.asarray(forward)[None, None, :]
        + grid_x[..., None] * xp.asarray(right)[None, None, :]
        + grid_y[..., None] * xp.asarray(up)[None, None, :]
    )
    directions /= xp.linalg.norm(directions, axis=-1, keepdims=True)
    return np.asarray(camera.position, dtype=np.float64), directions


def _geodetic_normal(points: Any, ellipsoid: Ellipsoid, xp: ModuleType) -> Any:
    """Unit gradient of the ellipsoid's implicit form — the geodetic vertical.

    :math:`\\nabla(x^2/a^2 + y^2/a^2 + z^2/b^2)` is parallel to the surface
    normal, and on an ellipsoid that is *not* parallel to the position
    vector: the two differ by up to 0.19 degrees at mid-latitudes.
    """
    gradient = points * xp.asarray(1.0 / ellipsoid.axes**2)
    return gradient / xp.maximum(
        xp.linalg.norm(gradient, axis=-1, keepdims=True), 1.0e-300
    )


def _altitude(points: Any, ellipsoid: Ellipsoid, xp: ModuleType) -> Any:
    """Signed height above the ellipsoid (m), to first order in the gradient.

    :math:`F/|\\nabla F|` for the implicit form, which is the standard
    first-order distance to a smooth level set. The error is
    :math:`O(h^2 \\kappa)`; at 10 km altitude on a 6,378 km body that is
    metres, well inside a pixel at any range where terrain is visible, and
    it is checked numerically in the tests rather than asserted here.

    Used only by the displacement march. Anything that needs a *reported*
    altitude uses :func:`~passes.viz.ellipsoid.ecef_to_geodetic`, which is
    exact.
    """
    scaled = points * xp.asarray(1.0 / ellipsoid.axes**2)
    residual = xp.sum(points * scaled, axis=-1) - 1.0
    return 0.5 * residual / xp.maximum(xp.linalg.norm(scaled, axis=-1), 1.0e-300)


def _terrain_height(relief: ReliefMap, latitude_deg: Any, longitude_deg: Any,
                    xp: ModuleType) -> Any:
    """Bilinear elevation from a relief map's grid (m)."""
    rows, cols = relief.shape
    row = (90.0 - latitude_deg) / (180.0 / rows) - 0.5
    column = xp.mod(longitude_deg + 180.0, 360.0) / (360.0 / cols) - 0.5
    return _bilinear(relief.elevation, row, column, True, xp)


def _degrees(normals: Any, xp: ModuleType) -> tuple[Any, Any]:
    """Geodetic latitude and longitude in degrees from a surface normal."""
    latitude = xp.rad2deg(xp.arcsin(xp.clip(normals[..., 2], -1.0, 1.0)))
    longitude = xp.rad2deg(xp.arctan2(normals[..., 1], normals[..., 0]))
    return latitude, longitude


def render(
    camera: Camera,
    texture: Texture | Sequence[Texture],
    surface: Ellipsoid | float = WGS84,
    sun: _FloatArray | None = None,
    ambient: float = 0.12,
    atmosphere: float = 0.55,
    night: float = 0.16,
    specular: float = 0.35,
    background: _FloatArray | None = None,
    relief: ReliefMap | None = None,
    displace: bool = False,
    displace_steps: int = 4,
    feather: float = 0.05,
    backend: Backend = "numpy",
) -> tuple[_FloatArray, _FloatArray]:
    """Render the globe, returning an RGB image and a depth buffer.

    Parameters
    ----------
    camera:
        View.
    texture:
        One :class:`~passes.viz.imagery.Texture`, or several. With several,
        each pixel takes the **last** one that covers it, cross-faded over
        ``feather`` degrees at its border — so the natural order is coarse
        to fine: a global mosaic first, then a native-resolution crop of the
        launch site over the top.
    surface:
        The body. An :class:`~passes.viz.ellipsoid.Ellipsoid`, or a float
        radius meaning a sphere. **Must match whatever produced the states
        being drawn over it**; see the module note.
    sun:
        Unit vector toward the sun in the body frame. ``None`` lights the
        scene from the camera, which removes the terminator entirely and is
        occasionally what a diagram wants.
    ambient:
        Floor on the diffuse term, so the night side is not pure black.
    atmosphere:
        Strength of the limb glow. Zero disables it.
    night:
        Brightness of the unlit side relative to full daylight.
    specular:
        Strength of the ocean glint. Zero disables it, which is what a
        diagram wants when the returned colours are being read back rather
        than looked at — the highlight is additive and would corrupt them.
    background:
        RGB image of shape ``(height, width, 3)`` to composite the globe
        over. ``None`` gives a starfield-free dark background.
    relief:
        GMTED2010 slopes from :meth:`~passes.viz.terrain.Terrain.relief`.
        Perturbs the surface normal, so terrain is *lit* rather than
        painted. ``None`` leaves the body smooth.
    displace:
        Move the intersection onto the terrain instead of onto the
        reference ellipsoid. Needs ``relief``. Off by default because at
        orbital range the whole effect is under a pixel and it costs a few
        extra passes over the frame; on, it is what makes a launch-pad or
        impact close-up land on the right ground.
    displace_steps:
        Fixed-point iterations for the displacement march. Three is enough
        for anything but a grazing view; see the notes.
    feather:
        Degrees over which a finer texture fades in at its border.
    backend:
        ``"numpy"`` or ``"cupy"``. On ``"cupy"`` every texture and the
        relief map must already be device-resident from :func:`to_device`;
        the returned image and depth buffer are brought back to the host,
        because their only consumers are Matplotlib and the projection test,
        both of which are host-side.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The image, shape ``(height, width, 3)`` in ``[0, 1]``, and the
        distance from the camera to the surface at each pixel, ``inf`` where
        the ray misses. The depth buffer is what makes correct occlusion of
        overlaid trajectories possible.

    Notes
    -----
    **The displacement march does not silhouette.** It advances each ray to
    where its own altitude equals the terrain under it, which is right for
    ground the camera can see and wrong for ground hidden behind a ridge —
    there is no occlusion test, so a peak does not cast a hard edge over the
    valley behind it. At the grazing incidences where that matters the
    iteration is also least well conditioned, so the step is clamped and
    the result there is the undisplaced ellipsoid. Both limits are visible
    in the same views, and the honest description is that this puts the
    ground at the right *height*, not that it renders a horizon profile.
    """
    ellipsoid = as_ellipsoid(surface)
    xp = get_array_module(backend)
    textures = [texture] if isinstance(texture, Texture) else list(texture)
    if not textures:
        msg = "render needs at least one texture"
        raise ValueError(msg)
    if backend != "numpy":
        for item in textures:
            if isinstance(item.data, np.ndarray):
                msg = (
                    f"backend {backend!r} needs device-resident textures; call "
                    "passes.viz.globe.to_device(texture, backend) once at set-up "
                    "rather than uploading the mosaic per frame"
                )
                raise TypeError(msg)
    if displace and relief is None:
        msg = "displace=True needs a relief map; pass relief=Terrain(...).relief()"
        raise ValueError(msg)

    origin, directions = _rays(camera, xp)
    start = xp.asarray(origin)[None, None, :]

    t_near, hit = ray_ellipsoid(origin, directions, ellipsoid, xp)

    if displace and relief is not None:
        # March from the *terrain envelope*, not from the reference surface,
        # so a ray that clears the ellipsoid but grazes a plateau is not
        # discarded before the march can find it.
        ceiling = float(relief.elevation.max())
        t_march, reached = ray_ellipsoid(
            origin, directions, ellipsoid, xp, inflation=ceiling
        )
        t_march = xp.where(reached, t_march, 0.0)
        error = xp.zeros_like(t_march)
        for _ in range(max(int(displace_steps), 1)):
            points = start + t_march[..., None] * directions
            normals = _geodetic_normal(points, ellipsoid, xp)
            latitude_deg, longitude_deg = _degrees(normals, xp)
            ground = _terrain_height(relief, latitude_deg, longitude_deg, xp)
            error = _altitude(points, ellipsoid, xp) - ground
            # Rate at which altitude falls per metre along the ray. It goes
            # to zero at grazing incidence, where the fixed point has no
            # useful contraction, so the step is simply not taken there.
            descent = -xp.sum(directions * normals, axis=-1)
            t_march = t_march + error / xp.where(descent > 0.05, descent, xp.inf)
        # Accept the march only where it actually landed on the terrain.
        # Everything else — grazing rays, and the envelope's 8.8 km-high
        # false limb around the disc — falls back to the reference surface,
        # which is the honest answer for a method that does not silhouette.
        settled = reached & (t_march > 0.0) & (xp.abs(error) < 200.0)
        t_near = xp.where(settled, t_march, t_near)
        hit = hit | settled

    depth = xp.where(hit, t_near, np.inf)
    points = start + xp.where(hit, t_near, 0.0)[..., None] * directions
    normals = _geodetic_normal(points, ellipsoid, xp)
    latitude_deg, longitude_deg = _degrees(normals, xp)

    albedo = _sample_texture(textures[0], latitude_deg, longitude_deg, xp)
    for finer in textures[1:]:
        blend = _edge_fade(finer, latitude_deg, longitude_deg, feather, xp)[..., None]
        albedo = albedo * (1.0 - blend) + _sample_texture(
            finer, latitude_deg, longitude_deg, xp
        ) * blend

    shading_normals = normals
    if relief is not None:
        # East/north/up at each hit, from the geodetic latitude and
        # longitude that the normal itself defines.
        phi, lam = xp.deg2rad(latitude_deg), xp.deg2rad(longitude_deg)
        sin_phi, cos_phi = xp.sin(phi), xp.cos(phi)
        sin_lam, cos_lam = xp.sin(lam), xp.cos(lam)
        east = xp.stack([-sin_lam, cos_lam, xp.zeros_like(sin_lam)], axis=-1)
        north = xp.stack(
            [-sin_phi * cos_lam, -sin_phi * sin_lam, cos_phi], axis=-1
        )
        rows, cols = relief.shape
        row = (90.0 - latitude_deg) / (180.0 / rows) - 0.5
        column = xp.mod(longitude_deg + 180.0, 360.0) / (360.0 / cols) - 0.5
        slope_e = _bilinear(relief.slope_east, row, column, True, xp)
        slope_n = _bilinear(relief.slope_north, row, column, True, xp)
        tilted = normals - slope_e[..., None] * east - slope_n[..., None] * north
        shading_normals = tilted / xp.maximum(
            xp.linalg.norm(tilted, axis=-1, keepdims=True), 1.0e-300
        )

    light = (
        np.asarray(sun, dtype=np.float64)
        if sun is not None
        else -np.asarray(camera.basis()[2], dtype=np.float64)
    )
    light = light / float(np.linalg.norm(light))
    cosine = xp.einsum("ijk,k->ij", shading_normals, xp.asarray(light))
    # The terminator follows the *reference* surface, not the terrain: a
    # shaded slope that tipped past the terminator would light itself on the
    # night side, which is a rendering artefact rather than alpenglow.
    smooth_cosine = xp.einsum("ijk,k->ij", normals, xp.asarray(light))

    # Soft terminator: a hard step reads as an aliasing bug on a sphere.
    day = xp.clip(cosine, 0.0, 1.0) ** 0.75
    twilight = xp.clip((smooth_cosine + 0.12) / 0.24, 0.0, 1.0)
    diffuse = (ambient + (1.0 - ambient) * day)[..., None]

    lit = albedo * diffuse
    dark = albedo * night * xp.asarray([0.55, 0.65, 1.0])
    shaded = dark + (lit - dark) * twilight[..., None]

    if atmosphere > 0.0:
        view = -directions
        grazing = 1.0 - xp.clip(xp.einsum("ijk,ijk->ij", normals, view), 0.0, 1.0)
        rim = grazing**3 * xp.clip(smooth_cosine + 0.25, 0.0, 1.0)
        shaded = shaded + atmosphere * rim[..., None] * xp.asarray([0.30, 0.52, 0.95])

    if specular > 0.0:
        # Biased toward dark texels so continents do not gleam.
        halfway = xp.asarray(light) - directions
        halfway /= xp.linalg.norm(halfway, axis=-1, keepdims=True)
        spec = xp.clip(xp.einsum("ijk,ijk->ij", normals, halfway), 0.0, 1.0) ** 48.0
        ocean = xp.clip(1.0 - albedo.mean(axis=-1) * 2.2, 0.0, 1.0)
        shaded = shaded + (
            specular * spec * ocean * xp.clip(smooth_cosine, 0.0, 1.0)
        )[..., None]

    if background is None:
        image = xp.zeros((camera.height, camera.width, 3), dtype=xp.float64)
    else:
        if background.shape != (camera.height, camera.width, 3):
            msg = (
                f"background must have shape {(camera.height, camera.width, 3)}, "
                f"got {background.shape}"
            )
            raise ValueError(msg)
        image = xp.asarray(background, dtype=xp.float64)

    image = xp.where(hit[..., None], xp.clip(shaded, 0.0, 1.0), image)
    if backend == "numpy":
        return np.asarray(image), np.asarray(depth)
    return to_numpy(image), to_numpy(depth)


def project(
    points: _FloatArray,
    camera: Camera,
    surface: Ellipsoid | float | None = None,
) -> tuple[_FloatArray, _FloatArray, NDArray[np.bool_]]:
    """Project world points to pixel coordinates.

    Parameters
    ----------
    points:
        Shape ``(n, 3)`` in the same frame as the camera.
    camera:
        View.
    surface:
        Occluding body — an :class:`~passes.viz.ellipsoid.Ellipsoid` or a
        float radius. When given, a point is marked hidden if the body lies
        between it and the camera: the depth test that Matplotlib's
        painter's algorithm cannot do, and the reason trajectories used to
        appear in front of the planet they were behind.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Pixel ``x`` and ``y`` (float, may fall outside the frame), and a
        boolean mask that is ``True`` where the point is in front of the
        camera and not occluded.

    Notes
    -----
    The occlusion test uses the **reference surface**, not the terrain, even
    when the render displaced it. A trajectory clipped by a mountain rather
    than by the limb would be a stronger claim than this layer can support:
    it would need the ray-terrain occlusion that
    :func:`render`'s displacement march explicitly does not do.
    """
    array = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if array.ndim != 2 or array.shape[1] != 3:
        msg = f"points must have shape (n, 3), got {array.shape}"
        raise ValueError(msg)
    right, up, forward = camera.basis()
    origin = np.asarray(camera.position, dtype=np.float64)
    relative = array - origin

    depth = relative @ forward
    in_front = depth > 0.0
    safe = np.where(in_front, depth, 1.0)
    half_h = np.tan(0.5 * camera.fov)
    half_w = half_h * (camera.width / camera.height)
    ndc_x = (relative @ right) / (safe * half_w)
    ndc_y = (relative @ up) / (safe * half_h)
    px = (0.5 * (ndc_x + 1.0)) * (camera.width - 1)
    py = (0.5 * (1.0 - ndc_y)) * (camera.height - 1)

    visible = in_front
    if surface is not None:
        ellipsoid = as_ellipsoid(surface)
        distance = np.linalg.norm(relative, axis=1)
        direction = relative / np.maximum(distance[:, None], 1e-12)
        t_hit, blocked = ray_ellipsoid(origin, direction, ellipsoid)
        occluded = blocked & (t_hit > 1e-6) & (t_hit < distance - 1e-6)
        visible = visible & ~occluded
    return np.asarray(px), np.asarray(py), np.asarray(visible)
