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
intersects a camera ray with the sphere, converts the hit point to a
latitude and longitude, and samples the texture bilinearly. The result is
limited by the texture and the output size, never by a mesh; occlusion is a
per-point depth test rather than an artist sort; and the whole thing is
vectorised NumPy, so a 1600x900 frame takes tens of milliseconds.

What it models
--------------

Enough to look right, and no more — this is a presentation layer, not
radiometry:

* **Lambertian terrain shading** from a sun direction, with a soft
  terminator rather than a hard one, because a hard day/night edge on a
  sphere reads as a rendering bug.
* **Night side** as a dimmed, slightly blue-shifted version of the day
  texture. Real night lights would need a second texture.
* **Atmospheric limb**, a forward-scattering rim that brightens toward the
  edge of the disc. This is the single cheapest cue that the object is a
  planet with air rather than a billiard ball.
* **Specular highlight**, weighted toward dark (ocean) texels so land does
  not gleam.

None of it is calibrated against anything. It is stated here so that nobody
mistakes a pretty frame for a radiative-transfer result.

Where it runs
-------------

Every per-pixel expression below is written against the array API that
NumPy and CuPy share, so :func:`render` takes the same ``backend`` argument
as the batched integrator in :mod:`passes.batch.backend` and runs on either
device. The texture must already live on the requested backend — see
:func:`to_device` — because re-uploading a 4096x2048 texture once per frame
would cost more than the render it was meant to accelerate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from passes.batch.backend import Backend, get_array_module, to_numpy

__all__ = [
    "DEFAULT_TEXTURE",
    "Camera",
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


#: The packaged Blue Marble texture: equirectangular, north-up, longitude
#: -180 at column zero.
DEFAULT_TEXTURE = _default_texture()


def load_texture(path: str | Path = DEFAULT_TEXTURE) -> _FloatArray:
    """Load an equirectangular texture as float RGB in ``[0, 1]``.

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
        return np.asarray(np.asarray(handle.convert("RGB"), dtype=np.float64) / 255.0)


def to_device(texture: _FloatArray, backend: Backend = "numpy") -> Any:
    """Place a texture on the requested backend, once.

    Kept explicit rather than done inside :func:`render` because the upload
    is the expensive part: a 4096x2048x3 float64 texture is 200 MB, and
    moving it per frame would swamp the render it was meant to speed up.
    Upload at set-up, keep the handle, pass it to every frame.
    """
    return get_array_module(backend).asarray(texture)


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


def _sample(texture: Any, latitude: Any, longitude: Any, xp: ModuleType) -> Any:
    """Bilinear texture lookup, wrapping in longitude and clamping in latitude."""
    rows, cols = texture.shape[0], texture.shape[1]
    # Texture row 0 is +90 latitude; column 0 is -180 longitude.
    v = (0.5 - latitude / np.pi) * (rows - 1)
    u = ((longitude + np.pi) / (2.0 * np.pi)) * cols

    v0 = xp.clip(xp.floor(v).astype(xp.int64), 0, rows - 1)
    v1 = xp.clip(v0 + 1, 0, rows - 1)
    u0 = xp.mod(xp.floor(u).astype(xp.int64), cols)
    u1 = xp.mod(u0 + 1, cols)
    fv = (v - v0)[..., None]
    fu = (u - xp.floor(u))[..., None]

    top = texture[v0, u0] * (1.0 - fu) + texture[v0, u1] * fu
    bottom = texture[v1, u0] * (1.0 - fu) + texture[v1, u1] * fu
    return top * (1.0 - fv) + bottom * fv


def _rays(camera: Camera, xp: ModuleType) -> tuple[_FloatArray, Any]:
    """Origin and per-pixel unit directions for the camera.

    The origin stays on the host — it is three numbers, and keeping it there
    lets the ray-sphere coefficients be plain Python floats on either
    backend.
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


def render(
    camera: Camera,
    texture: Any,
    radius: float,
    sun: _FloatArray | None = None,
    ambient: float = 0.12,
    atmosphere: float = 0.55,
    night: float = 0.16,
    specular: float = 0.35,
    background: _FloatArray | None = None,
    backend: Backend = "numpy",
) -> tuple[_FloatArray, _FloatArray]:
    """Render the globe, returning an RGB image and a depth buffer.

    Parameters
    ----------
    camera:
        View.
    texture:
        Equirectangular RGB in ``[0, 1]``, from :func:`load_texture`.
    radius:
        Sphere radius, same units as the camera.
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
    backend:
        ``"numpy"`` or ``"cupy"``. On ``"cupy"`` the texture must already be
        a device array from :func:`to_device`; the returned image and depth
        buffer are brought back to the host, because their only consumers
        are Matplotlib and the projection test, both of which are host-side.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The image, shape ``(height, width, 3)`` in ``[0, 1]``, and the
        distance from the camera to the sphere at each pixel, ``inf`` where
        the ray misses. The depth buffer is what makes correct occlusion of
        overlaid trajectories possible.
    """
    if radius <= 0.0:
        msg = f"radius must be positive, got {radius}"
        raise ValueError(msg)
    xp = get_array_module(backend)
    if backend != "numpy" and isinstance(texture, np.ndarray):
        msg = (
            f"backend {backend!r} needs a device-resident texture; call "
            "passes.viz.globe.to_device(texture, backend) once at set-up "
            "rather than uploading 200 MB per frame"
        )
        raise TypeError(msg)
    origin, directions = _rays(camera, xp)

    # Ray-sphere intersection about the origin: |o + t d|^2 = R^2.
    b = 2.0 * xp.einsum("ijk,k->ij", directions, xp.asarray(origin))
    c = float(origin @ origin) - radius * radius
    discriminant = b * b - 4.0 * c
    hit = discriminant > 0.0
    sqrt_disc = xp.sqrt(xp.where(hit, discriminant, 0.0))
    t_near = 0.5 * (-b - sqrt_disc)
    hit &= t_near > 0.0

    depth = xp.where(hit, t_near, np.inf)
    points = xp.asarray(origin)[None, None, :] + t_near[..., None] * directions
    normals = points / radius

    latitude = xp.arcsin(xp.clip(normals[..., 2], -1.0, 1.0))
    longitude = xp.arctan2(normals[..., 1], normals[..., 0])
    albedo = _sample(texture, latitude, longitude, xp)

    light = (
        np.asarray(sun, dtype=np.float64)
        if sun is not None
        else -np.asarray(camera.basis()[2], dtype=np.float64)
    )
    light = light / float(np.linalg.norm(light))
    cosine = xp.einsum("ijk,k->ij", normals, xp.asarray(light))

    # Soft terminator: a hard step reads as an aliasing bug on a sphere.
    day = xp.clip(cosine, 0.0, 1.0) ** 0.75
    twilight = xp.clip((cosine + 0.12) / 0.24, 0.0, 1.0)
    diffuse = (ambient + (1.0 - ambient) * day)[..., None]

    lit = albedo * diffuse
    dark = albedo * night * xp.asarray([0.55, 0.65, 1.0])
    shaded = dark + (lit - dark) * twilight[..., None]

    if atmosphere > 0.0:
        view = -directions
        grazing = 1.0 - xp.clip(xp.einsum("ijk,ijk->ij", normals, view), 0.0, 1.0)
        rim = grazing**3 * xp.clip(cosine + 0.25, 0.0, 1.0)
        shaded = shaded + atmosphere * rim[..., None] * xp.asarray([0.30, 0.52, 0.95])

    if specular > 0.0:
        # Biased toward dark texels so continents do not gleam.
        halfway = xp.asarray(light) - directions
        halfway /= xp.linalg.norm(halfway, axis=-1, keepdims=True)
        spec = xp.clip(xp.einsum("ijk,ijk->ij", normals, halfway), 0.0, 1.0) ** 48.0
        ocean = xp.clip(1.0 - albedo.mean(axis=-1) * 2.2, 0.0, 1.0)
        shaded = shaded + (specular * spec * ocean * xp.clip(cosine, 0.0, 1.0))[..., None]

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
    radius: float | None = None,
) -> tuple[_FloatArray, _FloatArray, NDArray[np.bool_]]:
    """Project world points to pixel coordinates.

    Parameters
    ----------
    points:
        Shape ``(n, 3)`` in the same frame as the camera.
    camera:
        View.
    radius:
        Occluding sphere radius. When given, a point is marked hidden if the
        sphere lies between it and the camera — the depth test that
        Matplotlib's painter's algorithm cannot do, and the reason
        trajectories previously appeared to pass in front of the planet
        they were behind.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Pixel ``x`` and ``y`` (float, may fall outside the frame), and a
        boolean mask that is ``True`` where the point is in front of the
        camera and not occluded.
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
    if radius is not None:
        # Closest approach of the eye-to-point segment to the sphere centre.
        direction = relative / np.maximum(
            np.linalg.norm(relative, axis=1, keepdims=True), 1e-12
        )
        b = 2.0 * direction @ origin
        c = float(origin @ origin) - radius * radius
        discriminant = b * b - 4.0 * c
        blocked = discriminant > 0.0
        t_hit = 0.5 * (-b - np.sqrt(np.where(blocked, discriminant, 0.0)))
        distance = np.linalg.norm(relative, axis=1)
        occluded = blocked & (t_hit > 1e-6) & (t_hit < distance - 1e-6)
        visible = visible & ~occluded
    return np.asarray(px), np.asarray(py), np.asarray(visible)
