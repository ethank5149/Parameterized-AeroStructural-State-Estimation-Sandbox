"""Scene primitives: everything a frame is built from, as pure functions.

These lived in notebook cells. That was the structural problem named in
:mod:`passes.viz.history`: helpers that reconstructed geometry sat next to
the physics that produced it, were copied between notebooks, and were
covered by no test. Two of the defects found while building the chase
animations were exactly that — a frame index that stopped at 86 % of the
flight, and a camera placed 96 km underground.

So the helpers move here, and every one of them is a **pure function of a
:class:`~passes.viz.history.SimulationHistory` sample plus a camera**.
Nothing in this module integrates anything, propagates anything, or knows
what a trajectory is; it takes states that were computed elsewhere and turns
them into pixels.

The division of labour
----------------------

* :mod:`passes.viz.globe` — ray-traced sphere and point projection. Knows
  about cameras and textures, nothing about vehicles.
* :mod:`passes.viz.scene` (this module) — geometry and Matplotlib overlays
  built from history samples: tracks, markers, the oriented vehicle glyph,
  sensor overlays, the chase rig.
* :mod:`passes.viz.animator` — composes the above over time.

What the glyph does and does not claim
--------------------------------------

:func:`draw_vehicle` is driven by the direction cosine matrix of the
**integrated** attitude quaternion, so its orientation is real state rather
than a heading guess. Two honest caveats, both stated at the call site:

* It is drawn at an exaggerated, camera-relative scale. A 15 m vehicle at a
  500 km stand-off subtends about 30 nanoradians, some 1/200 000 of a pixel
  at any sane field of view. The alternative to exaggeration is an invisible
  vehicle, so the glyph subtends a fixed fraction of the frame and says so.
* In the current :class:`~passes.flight.simulator.FlightSimulator` the
  attitude is integrated torque-free and drag acts along the relative
  velocity, so attitude does **not** feed back into the force model. The
  glyph therefore shows the rotational state that was integrated; it does
  not show an angle of attack the trajectory responded to, because there
  isn't one yet.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.patches import Rectangle
from numpy.typing import NDArray

from passes.batch.backend import Backend
from passes.geodesy import WGS84_MEAN_RADIUS, GeodeticPosition
from passes.orbital.warning import horizon_central_angle
from passes.viz.globe import Camera, project, render, to_device

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from passes.orbital.radar import CoverageResult, RadarSite

__all__ = [
    "NOSE_AXIS",
    "PHASE_COLORS",
    "ChaseRig",
    "SceneStyle",
    "draw_horizon_ring",
    "draw_marker",
    "draw_sites",
    "draw_timeline",
    "draw_track",
    "draw_vehicle",
    "ease",
    "geodetic_to_cartesian",
    "globe_plate",
    "glyph_polylines",
    "glyph_world",
    "horizon_ring",
    "site_status",
    "starfield",
]

_FloatArray = NDArray[np.float64]

#: Colours for the declared flight phases. Named rather than positional so
#: the same leg keeps its colour across profiles of different structure.
PHASE_COLORS: dict[str, str] = {
    "boost": "#FF6B35",
    "ascent": "#FF6B35",
    "parking coast": "#4CC9F0",
    "deorbit coast": "#B388FF",
    "descent": "#B388FF",
    "entry": "#FF3B30",
}

#: Body-frame nose direction assumed by the vehicle glyph. The flight model
#: is torque-free with drag along the relative velocity, so no force term
#: pins a body axis; this is a presentation convention, declared rather than
#: inferred.
NOSE_AXIS = np.array([1.0, 0.0, 0.0])


@dataclass(frozen=True)
class SceneStyle:
    """Colours and weights, in one place so a notebook does not carry them."""

    track: str = "#FF8A3D"
    trail_width: float = 3.2
    track_width: float = 1.3
    track_alpha: float = 0.30
    vehicle: str = "#FFFFFF"
    aimpoint: str = "#FFD166"
    launch: str = "#FFFFFF"
    site_idle: str = "#7CFFB2"
    site_active: str = "#FF3B30"
    site_seen: str = "#FFD166"
    text: str = "#FFFFFF"
    heat_cmap: str = "inferno"


def ease(t: float | _FloatArray) -> _FloatArray:
    """Smoothstep on ``[0, 1]``.

    Linear interpolation between camera states reads as mechanical: the
    acceleration is discontinuous at every keyframe and the eye visibly
    snaps. Smoothstep has zero first derivative at both ends, which is the
    cheapest fix that looks intentional.
    """
    x = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    return np.asarray(x * x * (3.0 - 2.0 * x))


def geodetic_to_cartesian(
    position: GeodeticPosition | Iterable[GeodeticPosition],
    body_radius: float = WGS84_MEAN_RADIUS,
    lift: float = 0.0,
) -> _FloatArray:
    """Geodetic point(s) to Cartesian vectors on a sphere of ``body_radius``.

    Parameters
    ----------
    position:
        One :class:`~passes.geodesy.GeodeticPosition` or an iterable of them.
    body_radius:
        Sphere radius (m).
    lift:
        Extra radius (m). A marker drawn exactly on the surface is half
        buried in it and half occluded by the depth test, which reads as a
        rendering glitch rather than as a site; lifting it clear is
        cosmetic and stated.

    Returns
    -------
    numpy.ndarray
        Shape ``(3,)`` for a single position, ``(n, 3)`` for an iterable.

    Notes
    -----
    Spherical, not ellipsoidal: geodetic latitude is used directly as a
    geocentric one. That is the same approximation the whole
    :mod:`passes.orbital.scenario` comparison rests on, and mixing an
    ellipsoidal marker into a spherical scene would misplace it relative to
    the very trajectories it is drawn against.
    """
    points: list[GeodeticPosition]
    if isinstance(position, GeodeticPosition):
        single, points = True, [position]
    else:
        single, points = False, list(position)
    radii = np.array(
        [body_radius + max(float(p.altitude), 0.0) + float(lift) for p in points]
    )
    latitudes = np.array([float(p.latitude) for p in points])
    longitudes = np.array([float(p.longitude) for p in points])
    cartesian = np.stack(
        [
            radii * np.cos(latitudes) * np.cos(longitudes),
            radii * np.cos(latitudes) * np.sin(longitudes),
            radii * np.sin(latitudes),
        ],
        axis=1,
    )
    return np.asarray(cartesian[0] if single else cartesian)


_STARFIELDS: dict[tuple[int, int, float, int], _FloatArray] = {}


def starfield(
    width: int, height: int, density: float = 0.00035, seed: int = 7
) -> _FloatArray:
    """A faint fixed starfield, cached per size.

    Deterministic because a resampled field shimmers between frames, and
    cached because it is identical for every frame of a sequence — building
    it per frame was pure waste at 130 frames a run. The returned array is
    marked read-only so a caller cannot poison the cache;
    :func:`passes.viz.globe.render` copies its background, so this costs
    nothing.
    """
    key = (int(width), int(height), float(density), int(seed))
    cached = _STARFIELDS.get(key)
    if cached is not None:
        return cached
    rng = np.random.default_rng(seed)
    sky = np.zeros((height, width, 3), dtype=np.float64)
    count = int(width * height * density)
    rows = rng.integers(0, height, count)
    cols = rng.integers(0, width, count)
    magnitude = rng.power(0.35, count)
    tint = 0.75 + 0.25 * rng.random((count, 3))
    sky[rows, cols] = magnitude[:, None] * tint
    sky.flags.writeable = False
    _STARFIELDS[key] = sky
    return sky


@dataclass(frozen=True)
class ChaseRig:
    """A camera that rides behind and above the vehicle, along its velocity.

    The stand-off **scales with the vehicle's own altitude**, which is what
    lets one rig frame a 150 km fractional parking arc and a 1300 km
    minimum-energy apogee without being retuned. A fixed stand-off makes one
    of the two a line in the corner.

    Heading comes from the sampled **velocity**, not from a finite
    difference over an arbitrary look-ahead in the sample array. That
    matters: the old look-ahead of six samples meant the camera's notion of
    "forward" depended on the sampling density, so the same trajectory at
    400 and 900 samples framed differently.

    Attributes
    ----------
    back_scale, back_offset:
        Stand-off behind the vehicle is ``back_scale * altitude +
        back_offset`` (m). The offset is deliberately small — 25 km — so
        that a vehicle on the pad is framed from 25 km rather than from the
        450 km that an offset sized for orbit implies. Altitude does the
        rest: the same rig stands off 675 km at a 250 km parking orbit and
        3,400 km at a 1,300 km apogee.
    lift_scale, lift_offset:
        Height above the vehicle's local vertical, same form (m).
    min_altitude:
        Altitude floor used in those laws only, so a vehicle at the surface
        still gets a finite stand-off rather than sitting inside the eye.
    tighten:
        Fraction by which the rig closes in over the run. Zero holds the
        stand-off constant.
    fov:
        Vertical field of view (rad).
    floor:
        Minimum eye altitude (m). During boost the velocity is steeply
        radial, so stepping *back* along it also steps *down*: at lift-off
        the eye ended up 96 km underground and the ray tracer returned an
        empty frame — a black screen with a floating trajectory and no Earth
        at all. The eye is pushed back out along its own radius when that
        happens.
    lead:
        How far ahead of the vehicle the camera aims, as a fraction of the
        stand-off. Aiming exactly at the vehicle pins it dead centre, which
        wastes the half of the frame the vehicle is flying into.
    """

    back_scale: float = 2.6
    back_offset: float = 25.0e3
    lift_scale: float = 0.9
    lift_offset: float = 10.0e3
    min_altitude: float = 0.0
    tighten: float = 0.25
    fov: float = float(np.deg2rad(42.0))
    floor: float = 3.0e3
    lead: float = 0.35

    def camera(
        self,
        position: _FloatArray,
        velocity: _FloatArray,
        width: int,
        height: int,
        progress: float = 0.0,
        body_radius: float = WGS84_MEAN_RADIUS,
    ) -> Camera:
        """Place the eye for one state.

        Parameters
        ----------
        position, velocity:
            Inertial state (m, m/s), straight from
            :meth:`~passes.viz.history.SimulationHistory.sample`.
        width, height:
            Frame size in pixels.
        progress:
            Fraction of the run completed, used only by ``tighten``.
        body_radius:
            Sphere radius (m), for the eye-altitude floor.
        """
        centre = np.asarray(position, dtype=np.float64)
        radius = float(np.linalg.norm(centre))
        if radius == 0.0:
            msg = "cannot place a chase camera on a vehicle at the body centre"
            raise ValueError(msg)
        up = centre / radius

        heading = np.asarray(velocity, dtype=np.float64)
        speed = float(np.linalg.norm(heading))
        # A stationary sample has no heading; fall back to the local
        # vertical rather than producing NaNs the renderer would silently
        # turn into an empty frame.
        heading = heading / speed if speed > 1e-6 else up

        # Stand off along the *horizontal* part of the heading, not along
        # the heading itself. On a launch the two are completely different:
        # the velocity is straight up, so stepping back along it steps
        # straight down, the eye clamps to its altitude floor, and the
        # vehicle is left as a dot 500 km away on the horizon. Watching
        # that, a launch looks like it begins in mid-air. Standing off
        # horizontally puts the camera beside the pad looking at a rising
        # vehicle, which is what a launch looks like.
        horizontal = heading - float(heading @ up) * up
        extent = float(np.linalg.norm(horizontal))
        if extent > 1e-6:
            horizontal = horizontal / extent
        else:
            # Purely radial: any horizontal direction will do, so take one
            # from the world axis least aligned with the local vertical.
            seed = np.array([0.0, 0.0, 1.0])
            if abs(float(up @ seed)) > 0.9:
                seed = np.array([1.0, 0.0, 0.0])
            horizontal = np.cross(np.cross(up, seed), up)
            horizontal = horizontal / float(np.linalg.norm(horizontal))
        # Blend: on orbit the heading *is* horizontal and the two agree, so
        # this only bites where it has to.
        back_axis = extent * heading + (1.0 - extent) * horizontal
        back_axis = back_axis / float(np.linalg.norm(back_axis))

        altitude = max(radius - body_radius, self.min_altitude)
        shrink = 1.0 - self.tighten * float(ease(progress))
        back = (self.back_scale * altitude + self.back_offset) * shrink
        lift = (self.lift_scale * altitude + self.lift_offset) * shrink

        eye = centre - back * back_axis + lift * up
        eye_radius = float(np.linalg.norm(eye))
        if eye_radius - body_radius < self.floor:
            eye = eye * (body_radius + self.floor) / eye_radius

        return Camera(
            position=eye,
            target=centre + self.lead * back * back_axis,
            up=up,
            fov=self.fov,
            width=int(width),
            height=int(height),
        )


def globe_plate(
    camera: Camera,
    texture: Any,
    body_radius: float = WGS84_MEAN_RADIUS,
    sun: _FloatArray | None = None,
    ambient: float = 0.13,
    atmosphere: float = 0.6,
    night: float = 0.20,
    specular: float = 0.10,
    stars: bool = True,
    backend: Backend = "numpy",
) -> tuple[Figure, Axes]:
    """A rendered globe on a figure whose axes are **pixels**.

    Every overlay in this module projects to pixel coordinates, so the
    background has to be drawn in the same ones — otherwise a marker and the
    limb it sits against are in two different spaces and only agree by
    accident.

    Specular is kept low by default: at globe scale a broad ocean highlight
    reads as a smudge rather than as sun glint, and it competes with the
    tracks drawn over it.
    """
    import matplotlib.pyplot as plt

    image, _ = render(
        camera,
        to_device(texture, backend),
        body_radius,
        sun=sun,
        ambient=ambient,
        atmosphere=atmosphere,
        night=night,
        specular=specular,
        background=starfield(camera.width, camera.height) if stars else None,
        backend=backend,
    )
    figure, ax = plt.subplots(
        figsize=(camera.width / 100, camera.height / 100), dpi=100
    )
    figure.patch.set_facecolor("black")
    ax.imshow(np.clip(image, 0.0, 1.0), interpolation="bilinear", zorder=0)
    ax.set_xlim(0, camera.width)
    ax.set_ylim(camera.height, 0)
    ax.axis("off")
    figure.subplots_adjust(0, 0, 1, 1)
    return figure, ax


def draw_track(
    ax: Axes,
    points: _FloatArray,
    camera: Camera,
    color: str = "#FF8A3D",
    width: float = 2.0,
    alpha: float = 1.0,
    zorder: float = 3.0,
    body_radius: float | None = WGS84_MEAN_RADIUS,
    values: _FloatArray | None = None,
    cmap: str = "inferno",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Draw a projected polyline, broken where the globe occludes it.

    Occluded samples become ``nan``, which breaks the polyline exactly where
    the limb cuts in front of it. This is the depth test Matplotlib's
    painter's algorithm cannot do, and the reason trajectories used to be
    drawn in front of the planet they were behind.

    Parameters
    ----------
    values:
        Optional per-sample scalar — stagnation heat flux, dynamic pressure,
        recession — used to colour the line through ``cmap`` instead of the
        flat ``color``. This is the cheapest way to put a physical quantity
        the simulator actually computed into the picture rather than
        alongside it.
    """
    array = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if array.shape[0] < 2:
        return
    px, py, visible = project(array, camera, radius=body_radius)
    xs = np.where(visible, px, np.nan)
    ys = np.where(visible, py, np.nan)

    if values is None:
        ax.plot(
            xs, ys, color=color, linewidth=width, alpha=alpha,
            zorder=zorder, solid_capstyle="round",
        )
        return

    from matplotlib.collections import LineCollection

    scalars = np.asarray(values, dtype=np.float64)
    if scalars.shape != (array.shape[0],):
        msg = f"values must have one entry per point, got {scalars.shape} for {array.shape}"
        raise ValueError(msg)
    segments = np.stack(
        [np.column_stack([xs[:-1], ys[:-1]]), np.column_stack([xs[1:], ys[1:]])], axis=1
    )
    midpoint = 0.5 * (scalars[:-1] + scalars[1:])
    collection = LineCollection(
        list(segments),
        cmap=cmap,
        linewidths=width,
        alpha=alpha,
        zorder=zorder,
        capstyle="round",
    )
    collection.set_array(midpoint)
    lo = float(np.nanmin(scalars)) if vmin is None else float(vmin)
    hi = float(np.nanmax(scalars)) if vmax is None else float(vmax)
    collection.set_clim(lo, hi if hi > lo else lo + 1.0)
    ax.add_collection(collection)


def draw_marker(
    ax: Axes,
    position: _FloatArray,
    camera: Camera,
    body_radius: float | None = WGS84_MEAN_RADIUS,
    **kwargs: Any,
) -> bool:
    """Scatter one world point, skipping it when the globe hides it.

    Returns whether it was drawn, so a caller can tell "behind the planet"
    from "drawn".
    """
    px, py, visible = project(np.atleast_2d(position), camera, radius=body_radius)
    if not bool(visible[0]):
        return False
    ax.scatter(px[0], py[0], **kwargs)
    return True


# -- vehicle glyph -------------------------------------------------------


def glyph_polylines(n_fins: int = 4) -> list[_FloatArray]:
    """Unit-length body-frame polylines for a slender re-entry body.

    The shape is a cone from a nose at ``+x`` back to a base ring at
    ``-0.35 x``, plus ``n_fins`` fins. Returned in body coordinates with the
    nose at unit distance, so a caller scales once and rotates once.

    Kept as line segments rather than a mesh on purpose: the renderer is a
    sphere ray-tracer with a depth buffer, not a scene graph, and a shaded
    solid would need occlusion against itself that nothing here provides.
    """
    if n_fins < 0:
        msg = f"n_fins must be non-negative, got {n_fins}"
        raise ValueError(msg)
    nose = np.array([1.0, 0.0, 0.0])
    base_x, base_r = -0.35, 0.30

    ring_angles = np.linspace(0.0, 2.0 * np.pi, 25)
    ring = np.stack(
        [
            np.full_like(ring_angles, base_x),
            base_r * np.cos(ring_angles),
            base_r * np.sin(ring_angles),
        ],
        axis=1,
    )
    lines: list[_FloatArray] = [ring]

    # Four generators of the cone, at 90 degrees, so the body reads as a
    # solid of revolution rather than as a circle with a dot.
    for phi in np.linspace(0.0, 2.0 * np.pi, 5)[:-1]:
        rim = np.array([base_x, base_r * np.cos(phi), base_r * np.sin(phi)])
        lines.append(np.stack([nose, rim]))

    for phi in np.linspace(0.0, 2.0 * np.pi, n_fins + 1)[:-1]:
        radial = np.array([0.0, np.cos(phi), np.sin(phi)])
        lines.append(
            np.stack(
                [
                    np.array([base_x + 0.30, 0.0, 0.0]) + base_r * radial,
                    np.array([base_x, 0.0, 0.0]) + 2.0 * base_r * radial,
                    np.array([base_x, 0.0, 0.0]) + base_r * radial,
                ]
            )
        )
    return lines


def glyph_world(
    position: _FloatArray,
    dcm: _FloatArray,
    scale: float,
    n_fins: int = 4,
) -> list[_FloatArray]:
    """The glyph's polylines placed and oriented in the inertial frame.

    Separated from :func:`draw_vehicle` so the orientation can be checked
    arithmetically rather than by looking at a picture: the nose must land
    at ``position + scale * C.T @ NOSE_AXIS`` for the direction cosine
    matrix ``C`` the run actually integrated.

    Parameters
    ----------
    dcm:
        :math:`\\mathbf{C}_E^B`, inertial to body. Body vectors go the other
        way by its transpose, which is its inverse because it is
        orthonormal — a property
        :func:`~passes.dynamics.attitude.dcm_from_quaternion` guarantees and
        which is checked in the tests rather than assumed here.
    """
    centre = np.asarray(position, dtype=np.float64).reshape(3)
    rotation = np.asarray(dcm, dtype=np.float64)
    if rotation.shape != (3, 3):
        msg = f"dcm must be a single 3x3 matrix, got shape {rotation.shape}"
        raise ValueError(msg)
    to_inertial = rotation.T
    return [
        centre + float(scale) * (line @ to_inertial.T)
        for line in glyph_polylines(n_fins)
    ]


def draw_vehicle(
    ax: Axes,
    position: _FloatArray,
    dcm: _FloatArray,
    camera: Camera,
    scale: float | None = None,
    color: str = "#FFFFFF",
    width: float = 1.6,
    zorder: float = 8.0,
    body_radius: float | None = WGS84_MEAN_RADIUS,
    n_fins: int = 4,
    screen_fraction: float = 0.08,
) -> bool:
    """Draw the vehicle oriented by its integrated attitude.

    Parameters
    ----------
    dcm:
        :math:`\\mathbf{C}_E^B`, mapping inertial components to body
        components — exactly what
        :func:`~passes.dynamics.attitude.dcm_from_quaternion` returns and
        what :meth:`~passes.viz.history.SimulationHistory.sample` puts in
        ``state["dcm"]``. Body vectors are taken to inertial by its
        transpose, which for an orthonormal matrix is its inverse.
    scale:
        Nose distance in metres. ``None`` sizes the glyph so it subtends
        ``screen_fraction`` of the frame height at the current camera
        distance.
    screen_fraction:
        Fraction of frame height the glyph occupies when ``scale`` is
        ``None``.

    Returns
    -------
    bool
        Whether anything was drawn. ``False`` when the globe occludes the
        vehicle — a caller that wants a "behind the planet" cue can use it.

    Notes
    -----
    **Not to scale, and it cannot be.** A 15 m vehicle at a 500 km
    stand-off subtends 30 nrad; at a 42-degree field of view over 720 rows
    that is 1/200 000 of a pixel. A true-scale glyph is an empty frame, so
    this one is sized in screen space and the exaggeration is declared here
    rather than implied by the picture.
    """
    centre = np.asarray(position, dtype=np.float64).reshape(3)

    _, _, in_front = project(centre[None, :], camera, radius=body_radius)
    if not bool(in_front[0]):
        return False

    if scale is None:
        distance = float(np.linalg.norm(centre - np.asarray(camera.position)))
        scale = float(screen_fraction * distance * np.tan(0.5 * camera.fov))

    for world in glyph_world(centre, dcm, scale, n_fins):
        px, py, visible = project(world, camera, radius=body_radius)
        ax.plot(
            np.where(visible, px, np.nan),
            np.where(visible, py, np.nan),
            color=color,
            linewidth=width,
            zorder=zorder,
            solid_capstyle="round",
        )
    return True


# -- sensor overlays -----------------------------------------------------


def horizon_ring(
    site: RadarSite,
    altitude: float,
    body_radius: float = WGS84_MEAN_RADIUS,
    samples: int = 181,
) -> _FloatArray:
    """The circle a site can see out to, at one vehicle altitude.

    A small circle of central angle
    :func:`~passes.orbital.warning.horizon_central_angle` about the site,
    drawn at the vehicle's radius rather than on the ground — because that
    is where the boundary actually is. Drawing it on the surface would show
    a footprint some three times too small at 150 km and understate the
    sensor by exactly the amount the fractional-orbital argument turns on.

    Returns
    -------
    numpy.ndarray
        Shape ``(samples, 3)``, closed (last point equals the first).
    """
    lam = float(horizon_central_angle(altitude, site.mask_elevation, body_radius))
    normal = geodetic_to_cartesian(site.position, body_radius)
    normal = normal / float(np.linalg.norm(normal))

    # Any pair of vectors spanning the plane perpendicular to the site.
    seed = np.array([0.0, 0.0, 1.0])
    if abs(float(normal @ seed)) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    east = np.cross(normal, seed)
    east = east / float(np.linalg.norm(east))
    north = np.cross(normal, east)

    phi = np.linspace(0.0, 2.0 * np.pi, samples)
    radius = body_radius + max(float(altitude), 0.0)
    ring = radius * (
        np.cos(lam) * normal[None, :]
        + np.sin(lam) * (np.cos(phi)[:, None] * east[None, :]
                         + np.sin(phi)[:, None] * north[None, :])
    )
    return np.asarray(ring)


def site_status(coverage: CoverageResult | None, name: str, time: float) -> str:
    """``"idle"``, ``"active"`` or ``"seen"`` for one site at one instant.

    ``"active"`` means the vehicle is above that site's mask **now**;
    ``"seen"`` means it was earlier and is not any more. The distinction is
    the point of the overlay: a network's warning is set by the *first*
    detection, so a site that has already seen the vehicle keeps its
    contribution even after the vehicle drops below its horizon.
    """
    if coverage is None:
        return "idle"
    window = coverage.windows.get(name)
    if window is None or not window.detected:
        return "idle"
    if window.first_detection_time <= time <= window.last_detection_time:
        return "active"
    return "seen" if time > window.first_detection_time else "idle"


def draw_sites(
    ax: Axes,
    sites: Sequence[RadarSite],
    camera: Camera,
    coverage: CoverageResult | None = None,
    time: float | None = None,
    body_radius: float = WGS84_MEAN_RADIUS,
    style: SceneStyle | None = None,
    lift: float = 15.0e3,
    size: float = 38.0,
    zorder: float = 5.0,
) -> dict[str, str]:
    """Draw the sensor network, coloured by what each site currently knows.

    With ``coverage`` and ``time`` supplied the markers become a live
    read-out of the detection model rather than decoration: idle, detecting
    now, or has detected. Returns the status of every site so a caller can
    caption the frame from the same computation that coloured it, instead of
    recomputing and risking disagreement.
    """
    palette = style or SceneStyle()
    colors = {
        "idle": palette.site_idle,
        "active": palette.site_active,
        "seen": palette.site_seen,
    }
    statuses: dict[str, str] = {}
    for radar in sites:
        status = "idle" if time is None else site_status(coverage, radar.name, time)
        statuses[radar.name] = status
        draw_marker(
            ax,
            geodetic_to_cartesian(radar.position, body_radius, lift=lift),
            camera,
            body_radius=body_radius,
            s=size * (1.6 if status == "active" else 1.0),
            marker="^",
            c=colors[status],
            edgecolors="black",
            linewidths=0.6,
            zorder=zorder + (1.0 if status == "active" else 0.0),
        )
    return statuses


def draw_horizon_ring(
    ax: Axes,
    site: RadarSite,
    altitude: float,
    camera: Camera,
    body_radius: float = WGS84_MEAN_RADIUS,
    color: str = "#FF3B30",
    width: float = 1.2,
    alpha: float = 0.75,
    zorder: float = 4.0,
) -> None:
    """Project a site's visibility circle at the vehicle's current altitude."""
    draw_track(
        ax,
        horizon_ring(site, altitude, body_radius),
        camera,
        color=color,
        width=width,
        alpha=alpha,
        zorder=zorder,
        body_radius=body_radius,
    )


def draw_timeline(
    ax: Axes,
    phases: Sequence[Any],
    events: Sequence[Any],
    time: float,
    duration: float,
    left: float = 0.06,
    right: float = 0.94,
    bottom: float = 0.055,
    height: float = 0.022,
    style: SceneStyle | None = None,
) -> None:
    """A phase strip with event ticks and a playhead, in axes fractions.

    This exists because of a specific complaint about the animations, and
    it is a fair one: a chase camera at orbital altitude shows the ground
    scrolling and little else, so a viewer cannot tell a parking coast from
    a deorbit coast, or judge how much flight is left. A HUD clock answers
    "when"; it does not answer "where in the plan".

    The strip is **linear in flight time**, deliberately, even when the
    frames themselves are not. Phase-paced playback stretches the short,
    eventful legs, and a timeline that stretched with it would hide exactly
    the distortion it is there to make legible: the playhead visibly
    crawling through the parking coast and sprinting through entry is the
    honest picture of what the pacing is doing.
    """
    palette = style or SceneStyle()
    if duration <= 0.0:
        return
    span = right - left

    ax.add_patch(
        Rectangle((left, bottom), span, height, transform=ax.transAxes,
                  facecolor="#FFFFFF", alpha=0.10, edgecolor="none", zorder=9)
    )
    for phase in phases:
        start = left + span * float(np.clip(phase.start_time / duration, 0.0, 1.0))
        width = span * float(np.clip(phase.duration / duration, 0.0, 1.0))
        if width <= 0.0:
            continue
        colour = PHASE_COLORS.get(phase.name, palette.track)
        ax.add_patch(
            Rectangle((start, bottom), width, height, transform=ax.transAxes,
                      facecolor=colour, alpha=0.75, edgecolor="none", zorder=10)
        )
        if width > 0.07:
            ax.text(start + 0.5 * width, bottom + 0.5 * height, phase.name,
                    transform=ax.transAxes, color="black", fontsize=8,
                    ha="center", va="center", zorder=12, weight="bold")

    for event in events:
        x = left + span * float(np.clip(event.time / duration, 0.0, 1.0))
        ax.plot([x, x], [bottom + height, bottom + height + 0.012],
                transform=ax.transAxes, color="white", linewidth=1.0,
                alpha=0.8, zorder=11)

    playhead = left + span * float(np.clip(time / duration, 0.0, 1.0))
    ax.plot([playhead, playhead], [bottom - 0.008, bottom + height + 0.008],
            transform=ax.transAxes, color="white", linewidth=2.0, zorder=13)
    ax.text(left, bottom - 0.028, "T+0", transform=ax.transAxes, color="#BFD9FF",
            fontsize=8, ha="left", va="top", family="monospace", zorder=12)
    ax.text(right, bottom - 0.028, f"T+{duration / 60:.0f} min", transform=ax.transAxes,
            color="#BFD9FF", fontsize=8, ha="right", va="top",
            family="monospace", zorder=12)
