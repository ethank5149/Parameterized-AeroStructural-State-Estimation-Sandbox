"""One frame, or a sequence of them, from a :class:`SimulationHistory`.

This is the façade the notebooks call. It exists so that the answer to
"where was the vehicle in frame 87?" is a lookup in the history rather than
a reconstruction, and so that the answer is available to a *test* as well as
to a viewer.

Sampled by time, not by index
-----------------------------

:meth:`TrajectoryAnimator.frame_at` takes **seconds**, and
:meth:`~TrajectoryAnimator.render_sequence` lays frames on a uniform time
grid spanning the whole history. That is not a stylistic preference; it
removes a class of bug by construction. The previous notebook indexed with
``frame * (len(samples) // n_frames)``, which truncates — with 900 samples
over 130 frames the stride was 6 and the last frame landed on sample 774 of
899, so every animation stopped at 86 % of the flight, still at 130 km, with
the entire descent missing. A time grid cannot do that: its endpoints *are*
the history's endpoints.

The same change fixes a quieter problem. Trajectory samples are not
uniformly spaced in time — a Keplerian arc sampled in true anomaly bunches
near apogee — so an index-uniform animation runs at a varying, wrong rate.
A time-uniform one plays at a constant multiple of real time, which is
stated in the HUD.

What is drawn, and from what
----------------------------

Everything in a frame traces to something the physics computed:

* the track and the trail, from ``history.positions``;
* the vehicle, oriented by the direction cosine matrix of the integrated
  attitude quaternion when the producer carried one, and drawn as a bare
  marker when it did not (see :class:`SimulationHistory`);
* the trail's colour, optionally from a per-sample series the simulator
  produced — stagnation heat flux, dynamic pressure, recession;
* sensor markers, coloured by
  :func:`~passes.viz.scene.site_status` against a
  :class:`~passes.orbital.radar.CoverageResult`, so the overlay and the
  caption come from one computation rather than two.

Nothing here integrates, propagates, or re-derives any of it.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from passes.batch.backend import Backend
from passes.viz.ellipsoid import WGS84, Ellipsoid, ecef_to_geodetic
from passes.viz.globe import (
    DEFAULT_TEXTURE,
    Camera,
    as_ellipsoid,
    load_texture,
    project,
    render,
    to_device,
)
from passes.viz.history import SimulationHistory
from passes.viz.imagery import BlueMarble, Texture, default_blue_marble
from passes.viz.pacing import PacingWeights, attention_density
from passes.viz.scene import (
    PHASE_COLORS,
    ChaseRig,
    SceneStyle,
    draw_horizon_ring,
    draw_marker,
    draw_sites,
    draw_stack,
    draw_timeline,
    draw_track,
    draw_vehicle,
    starfield,
)
from passes.viz.staging import StagingPlan, stack_polylines
from passes.viz.terrain import ReliefMap, default_terrain

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from passes.orbital.radar import CoverageResult, RadarSite

__all__ = ["Frame", "TrajectoryAnimator", "video_writer"]

_FloatArray = NDArray[np.float64]

#: Containers that get H.264 through Matplotlib's ffmpeg writer.
_VIDEO_SUFFIXES = {".mp4", ".m4v", ".mov", ".mkv"}

#: How to read an ``extras`` series out loud: label, SI divisor, unit. The
#: simulator stores SI, which is right; a HUD reading ``1.04e+06`` is not.
_EXTRA_FORMATS: dict[str, tuple[str, float, str]] = {
    "stagnation_heat_flux": ("heat flux", 1.0e4, "W/cm2"),
    "dynamic_pressure": ("dyn press", 1.0e3, "kPa"),
    "recession": ("recession", 1.0e-3, "mm"),
    "effective_radius": ("nose", 1.0e-3, "mm"),
    "altitude": ("altitude", 1.0e3, "km"),
}


def _format_extra(name: str, value: float) -> str:
    """One HUD line for a per-sample series, in units a reader can hold."""
    label, divisor, unit = _EXTRA_FORMATS.get(name, (name, 1.0, ""))
    return f"{label:9s}{value / divisor:8.2f} {unit}".rstrip()


class ImageryFallback(UserWarning):
    """Raised when the reference archives are absent and a lesser source is used.

    A warning rather than a silent substitution, because the difference is
    not cosmetic: the packaged JPEG is 5 arc-minutes and Blue Marble Next
    Generation is 15 arc-seconds, and a viewer looking at a launch close-up
    should not have to work out from the blur which one they were given.
    """


def _base_imagery(
    texture: Texture | Sequence[Texture] | None, month: int | None
) -> list[Texture]:
    """Global imagery: BMNG for the flight's month, or the packaged fallback."""
    if texture is not None:
        return [texture] if isinstance(texture, Texture) else list(texture)
    try:
        archive = default_blue_marble()
        return [archive.texture(height=4096, month=month or 1)]
    except (FileNotFoundError, ImportError) as reason:
        warnings.warn(
            f"no Blue Marble Next Generation imagery ({reason}); falling back to "
            f"the packaged {DEFAULT_TEXTURE.name} at 5 arc-minutes, which is "
            "1/300th of BMNG's pixel count and will not resolve a launch site",
            ImageryFallback,
            stacklevel=3,
        )
        return [load_texture()]


def _close_up_imagery(
    history: SimulationHistory,
    surface: Ellipsoid | float,
    month: int | None,
    half_width: float = 0.75,
) -> list[Texture]:
    """Native-resolution crops around the first and last points of a flight.

    Those two are where the camera gets close enough for the global mosaic
    to run out: at a 25 km stand-off the visible ground is a third of a
    degree, which the 4096-row mosaic covers with eight pixels.

    Returned in flight order and composited in that order, so where a
    launch and an impact box overlap — a short-range shot — the impact wins,
    which is the one the viewer is looking at when they overlap.
    """
    ellipsoid = as_ellipsoid(surface)
    try:
        archive = default_blue_marble()
    except (FileNotFoundError, ImportError):
        return []
    crops: list[Texture] = []
    for index in (0, -1):
        latitude, longitude, _ = ecef_to_geodetic(history.positions[index], ellipsoid)
        lat_deg = float(np.rad2deg(latitude))
        lon_deg = float(np.rad2deg(longitude))
        south = max(lat_deg - half_width, -90.0)
        north = min(lat_deg + half_width, 90.0)
        west = max(lon_deg - half_width, -180.0)
        east = min(lon_deg + half_width, 180.0)
        if north <= south or east <= west:  # pragma: no cover - polar/antimeridian
            continue
        try:
            crops.append(archive.window((south, north), (west, east), month=month or 1))
        except (FileNotFoundError, ValueError, ImportError):  # pragma: no cover
            continue
    return crops


def _relief_map(relief: ReliefMap | None, wanted: bool) -> ReliefMap | None:
    """GMTED2010 slopes, unless the caller supplied them or the archive is absent."""
    if relief is not None:
        return relief
    if not wanted:
        return None
    try:
        return default_terrain().relief()
    except (FileNotFoundError, ImportError) as reason:
        warnings.warn(
            f"no GMTED2010 elevation archive ({reason}); the globe will be shaded "
            "as a smooth ellipsoid and terrain displacement is off",
            ImageryFallback,
            stacklevel=3,
        )
        return None


@dataclass(frozen=True)
class Frame:
    """One drawn frame, together with the state it was drawn from.

    Carrying the camera and the state out is the whole point: it is what
    lets a test assert that the vehicle appears at the projection of the
    true state, rather than assert that a picture was produced.
    """

    time: float
    state: dict[str, Any]
    camera: Camera
    axes: Axes = field(repr=False)
    surface: Ellipsoid | float = WGS84

    @property
    def position(self) -> _FloatArray:
        return np.asarray(self.state["position"], dtype=np.float64)

    def pixel_of(self, point: _FloatArray) -> tuple[float, float, bool]:
        """Project a world point through this frame's camera."""
        px, py, visible = project(
            np.atleast_2d(point), self.camera, surface=self.surface
        )
        return float(px[0]), float(py[0]), bool(visible[0])

    @property
    def vehicle_pixel(self) -> tuple[float, float, bool]:
        """Where the vehicle is, in pixels, in this frame."""
        return self.pixel_of(self.position)


def video_writer(filename: str | Path, fps: int) -> Any:
    """Pick a Matplotlib writer from the output extension.

    GIF is what the notebooks emitted, and it is a poor container for this
    material: 256 colours per frame turns a soft terminator and a limb glow
    into visible banding, and the files ran to 13-25 MB for 130 frames.
    H.264 carries the gradients and is roughly an order of magnitude
    smaller, so ``.mp4`` is preferred wherever ffmpeg is present.

    Raises
    ------
    RuntimeError
        If a video container is asked for and ffmpeg is not available, so
        the caller learns that rather than silently receiving a GIF with
        different visual properties.
    """
    from matplotlib.animation import FFMpegWriter, PillowWriter

    suffix = Path(filename).suffix.lower()
    if suffix == ".gif":
        return PillowWriter(fps=fps)
    if suffix in _VIDEO_SUFFIXES:
        if not FFMpegWriter.isAvailable():
            msg = (
                f"{suffix} output needs ffmpeg on PATH and it was not found; "
                "install ffmpeg or ask for a .gif"
            )
            raise RuntimeError(msg)
        return FFMpegWriter(
            fps=fps,
            codec="libx264",
            # yuv420p for players that refuse 4:4:4; crf 18 is visually
            # lossless on this material, which is mostly smooth gradient.
            extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow"],
        )
    msg = (
        f"unsupported output extension {suffix!r}; use .gif or one of "
        f"{', '.join(sorted(_VIDEO_SUFFIXES))}"
    )
    raise ValueError(msg)


class TrajectoryAnimator:
    """Draw a :class:`SimulationHistory` on a ray-traced globe.

    Parameters
    ----------
    history:
        The run. Sampled, never re-derived.
    texture:
        Globe imagery, as one :class:`~passes.viz.imagery.Texture` or
        several ordered coarse to fine. Omitted, the animator asks
        :func:`~passes.viz.imagery.default_blue_marble` for the Blue Marble
        Next Generation mosaic **for the month the flight is in**, and falls
        back to the packaged legacy JPEG only if the archive is absent —
        with a stated warning, because 15-arc-second and 5-arc-minute
        imagery are not the same picture and a viewer should not have to
        guess which one they are looking at.
    surface:
        The body — an :class:`~passes.viz.ellipsoid.Ellipsoid` or a float
        radius meaning a sphere. **Must match whatever built the history.**
        The default is WGS84, which is what
        :class:`~passes.flight.simulator.FlightSimulator` integrates
        against: ``EARTH.radius`` is the WGS84 semi-major axis and the field
        carries J2. A history from
        :meth:`~passes.viz.history.SimulationHistory.from_trajectory` is
        spherical by construction and should be drawn on the same float
        radius it was built with.
    month:
        Month to take the Blue Marble imagery from, as a number, a name or a
        date. Snow line, sea ice and Sahel vegetation all move, and BMNG
        exists to record that; ``None`` leaves it at January.
    relief:
        GMTED2010 slopes. Omitted, the animator loads them from the
        reference archive when it is present and shades without them when it
        is not.
    detail:
        Read a native-resolution Blue Marble crop around the launch and
        impact points and composite it over the mosaic while the camera is
        close enough to resolve it. Needs the BMNG archive.
    displace:
        March rays onto the GMTED2010 terrain instead of onto the reference
        ellipsoid whenever the camera is below ``displace_below``. Costs
        roughly a third again per frame in the frames it applies to, and it
        is what puts a launch on the right ground rather than on a smooth
        datum 348 m below it.
    displace_below:
        Camera altitude (m) under which displacement switches on.
    rig:
        Camera rig. See :class:`~passes.viz.scene.ChaseRig`.
    style:
        Colours and weights.
    sites:
        Sensors to overlay.
    coverage:
        Detection result for **this** history against **those** sites, so
        markers and captions agree. Supplying coverage computed from a
        different trajectory is the one way to make this layer lie, and it
        is checked as far as it can be: the coverage clock must lie inside
        the history's span.
    markers:
        Fixed world points to draw, as ``{label: (position, kwargs)}``.
    sun:
        Explicit light direction in the body frame. Overrides ``lighting``.
    lighting:
        How the globe is lit, and all three options are stated cheats
        because the honest one does not work for a chase.

        ``"sun"`` needs an explicit ``sun`` and is physically right, but a
        launch site and an aimpoint half a globe apart cannot both be in
        daylight, so a true subsolar point plays the last minutes of the run
        in the dark. The static plates use it, where the terminator can be
        seen properly.

        ``"camera"`` lights from the eye, which removes the terminator — and
        removes most of the picture with it. A chase camera looks roughly
        *along* the surface, so the ground it can see is at grazing
        incidence to a light coming from behind the lens, and the frame
        comes out nearly black.

        ``"track"`` is the default: the light rides with the vehicle,
        ``key_elevation`` above its local horizon, so the ground beneath it
        stays lit as it moves. It is not a real sun and does not pretend to
        be one; it is a key light on the thing being filmed.
    key_elevation:
        Height of that key light above the local horizon (rad). **Not 90
        degrees**, deliberately: relief shading works on the cosine against
        a tilted normal, so a light along the local vertical meets terrain
        slope at second order and does essentially nothing. Measured over
        the Himalaya, an overhead key changes brightness by 0.4 % against
        the smooth ellipsoid where a key 55 degrees up changes it by 3.7 %.
        The default trades 18 % of the ground brightness for terrain that
        can actually be seen.
    color_by:
        Name of an ``extras`` series to colour the trail by. Ignored with a
        stated warning if the history does not carry it, rather than
        silently drawing a flat line the viewer would read as "no heating".
    trail_seconds:
        Length of the bright trail, in seconds of flight. Seconds rather
        than samples, because sample spacing is not uniform in time and a
        sample-counted trail therefore changes physical length over a run.
    backend:
        ``"numpy"`` or ``"cupy"``; the texture is uploaded once when the
        latter is asked for.
    """

    def __init__(
        self,
        history: SimulationHistory,
        texture: Texture | Sequence[Texture] | None = None,
        surface: Ellipsoid | float = WGS84,
        month: int | str | date | datetime | None = None,
        relief: ReliefMap | None = None,
        detail: bool = True,
        displace: bool = True,
        displace_below: float = 400.0e3,
        staging: StagingPlan | None = None,
        mould_line: tuple[_FloatArray, _FloatArray] | None = None,
        jettisoned: Mapping[str, SimulationHistory] | None = None,
        rig: ChaseRig | None = None,
        style: SceneStyle | None = None,
        sites: Sequence[RadarSite] = (),
        coverage: CoverageResult | None = None,
        markers: Mapping[str, tuple[_FloatArray, Mapping[str, Any]]] | None = None,
        sun: _FloatArray | None = None,
        lighting: str = "track",
        key_elevation: float = float(np.deg2rad(55.0)),
        color_by: str | None = None,
        trail_seconds: float | None = None,
        width: int = 1280,
        height: int = 720,
        ambient: float = 0.22,
        atmosphere: float = 0.55,
        specular: float = 0.06,
        horizon_rings: bool = True,
        timeline: bool = True,
        pacing: str = "attention",
        pacing_exponent: float = 0.45,
        pacing_weights: PacingWeights | None = None,
        backend: Backend = "numpy",
    ) -> None:
        if width < 2 or height < 2:
            msg = f"frame must be at least 2x2 pixels, got {width}x{height}"
            raise ValueError(msg)
        self.history = history
        self.surface: Ellipsoid | float = surface
        if staging is not None and mould_line is None:
            msg = (
                "staging needs the mould line the mass model was built from: "
                "pass mould_line=(stations, radii). Without it the stack has "
                "stage stations but no shape to cut them out of"
            )
            raise ValueError(msg)
        self.staging = staging
        self.mould_line = (
            None
            if mould_line is None
            else (
                np.asarray(mould_line[0], dtype=np.float64),
                np.asarray(mould_line[1], dtype=np.float64),
            )
        )
        self.jettisoned = dict(jettisoned or {})
        if self.jettisoned and staging is not None:
            known = {s.stage for s in staging.separations}
            unknown = set(self.jettisoned) - known
            if unknown:
                msg = (
                    f"jettisoned histories for {sorted(unknown)} name stages the "
                    f"staging plan never separates; it separates {sorted(known)}"
                )
                raise ValueError(msg)
        self.rig = rig or ChaseRig()
        self.style = style or SceneStyle()
        self.sites = tuple(sites)
        self.coverage = coverage
        self.markers = dict(markers or {})
        self.sun = None if sun is None else np.asarray(sun, dtype=np.float64)
        if lighting not in ("track", "camera", "sun"):
            msg = f"lighting must be 'track', 'camera' or 'sun', got {lighting!r}"
            raise ValueError(msg)
        if lighting == "sun" and sun is None:
            msg = "lighting='sun' needs an explicit sun direction"
            raise ValueError(msg)
        self.lighting = lighting
        self.key_elevation = float(key_elevation)
        self.width, self.height = int(width), int(height)
        self.ambient, self.atmosphere, self.specular = ambient, atmosphere, specular
        self.horizon_rings = horizon_rings
        self.timeline = timeline
        if pacing not in ("attention", "phase", "uniform"):
            msg = (
                f"pacing must be 'attention', 'phase' or 'uniform', got "
                f"{pacing!r}"
            )
            raise ValueError(msg)
        self.pacing = pacing
        self.pacing_exponent = float(pacing_exponent)
        self.pacing_weights = pacing_weights or PacingWeights()
        # Built once: it is a handful of finite differences over the history
        # and every frame time and playback rate is read from it.
        self._profile = (
            attention_density(history, self.surface, self.pacing_weights)
            if pacing == "attention"
            else None
        )
        self._playback: tuple[int, int] | None = None
        self.backend: Backend = backend

        # to_device is a no-op when a texture already lives on the requested
        # backend, so callers may hand in either host arrays or device ones
        # they uploaded once and share between animators.
        self.month = None if month is None else BlueMarble.month_of(month)
        base = _base_imagery(texture, self.month)
        self._textures = [to_device(item, backend) for item in base]
        self._detail: Any = None
        if detail and texture is None:
            near = _close_up_imagery(history, self.surface, self.month)
            self._detail = [to_device(item, backend) for item in near]
        self._relief = _relief_map(relief, displace)
        if self._relief is not None:
            self._relief = to_device(self._relief, backend)
        self.displace = bool(displace) and self._relief is not None
        self.displace_below = float(displace_below)
        self._sky = starfield(self.width, self.height)

        self._values: _FloatArray | None = None
        if color_by is not None:
            extras = history.extras or {}
            if color_by not in extras:
                available = ", ".join(sorted(extras)) or "none"
                msg = (
                    f"history {history.label!r} carries no extras series "
                    f"{color_by!r}; available: {available}"
                )
                raise KeyError(msg)
            self._values = np.asarray(extras[color_by], dtype=np.float64)
        self.color_by = color_by

        span = history.duration
        self.trail_seconds = (
            float(trail_seconds) if trail_seconds is not None else 0.25 * span
        )
        if coverage is not None and coverage.detected:
            first = coverage.first_detection_time
            if not history.times[0] - 1e-6 <= first <= history.times[-1] + 1e-6:
                msg = (
                    f"coverage first detection at t={first:.1f} s lies outside the "
                    f"history span [{history.times[0]:.1f}, {history.times[-1]:.1f}] s; "
                    "this coverage was computed from a different trajectory"
                )
                raise ValueError(msg)

    # -- one frame -------------------------------------------------------

    def frame_at(self, time: float, ax: Axes | None = None) -> Frame:
        """Draw the state at ``time`` (s) and return it with its camera.

        ``time`` is clamped to the history's span by
        :meth:`SimulationHistory.sample`, which holds the endpoints rather
        than extrapolating past the end of the physics.
        """
        import matplotlib.pyplot as plt

        history = self.history
        t = float(np.clip(time, history.times[0], history.times[-1]))
        state = history.sample(t)
        progress = (t - history.times[0]) / history.duration

        camera = self.rig.camera(
            state["position"],
            state["velocity"],
            self.width,
            self.height,
            progress=progress,
            surface=self.surface,
        )

        if ax is None:
            fig, ax = plt.subplots(
                figsize=(self.width / 100, self.height / 100), dpi=100
            )
            fig.patch.set_facecolor("black")
            fig.subplots_adjust(0, 0, 1, 1)
        ax.clear()
        ax.axis("off")

        # A key light on the vehicle rather than a sun on the planet: see
        # the `lighting` parameter for why a chase cannot use either of the
        # honest alternatives.
        if self.sun is not None:
            light = self.sun
        elif self.lighting == "track":
            light = self._key_light(state["position"], camera)
        else:
            light = None

        textures, displace = self._imagery_for(camera)
        image, _ = render(
            camera,
            textures,
            self.surface,
            sun=light,
            ambient=self.ambient,
            atmosphere=self.atmosphere,
            specular=self.specular,
            background=self._sky,
            relief=self._relief,
            displace=displace,
            backend=self.backend,
        )
        ax.imshow(np.clip(image, 0.0, 1.0), interpolation="bilinear", zorder=0)
        ax.set_xlim(0, self.width)
        ax.set_ylim(self.height, 0)

        self._draw_paths(ax, camera, t)
        self._draw_vehicle(ax, camera, state, t)
        self._draw_overlays(ax, camera, state, t)
        self._draw_hud(ax, state, t)
        if self.timeline and history.phases:
            draw_timeline(
                ax, history.phases, history.events, t - history.times[0],
                history.duration, style=self.style,
            )
        return Frame(t, state, camera, ax, self.surface)

    def _key_light(self, position: _FloatArray, camera: Camera) -> _FloatArray:
        """The tracking key light: ``key_elevation`` above the local horizon.

        Not straight up, and the reason is measurable. Relief shading works
        on the *cosine* between the light and a tilted normal, so a light
        along the local vertical meets terrain slope at second order and
        does almost nothing: over the Himalaya at 400 km, an overhead key
        changes brightness by 0.4 % where a key 55 degrees up changes it by
        3.7 %. With the light straight overhead the GMTED2010 shading is
        loaded, computed, and invisible.

        The horizontal component follows the camera's own right axis, so
        the terrain is cross-lit and the shadows fall the same way across
        the frame however the vehicle is heading. The cost is that the
        ground under the vehicle is at ``sin(key_elevation)`` of full
        brightness rather than at all of it — 82 % at the default.
        """
        centre = np.asarray(position, dtype=np.float64)
        up = centre / max(float(np.linalg.norm(centre)), 1e-12)
        right = np.asarray(camera.basis()[0], dtype=np.float64)
        side = right - float(right @ up) * up
        extent = float(np.linalg.norm(side))
        if extent < 1e-9:  # pragma: no cover - camera rolled onto the vertical
            return np.asarray(up)
        side = side / extent
        elevation = float(self.key_elevation)
        light = np.sin(elevation) * up + np.cos(elevation) * side
        return np.asarray(light / float(np.linalg.norm(light)))

    def camera_altitude(self, camera: Camera) -> float:
        """Geodetic altitude of the eye (m).

        The quantity both close-up decisions are made on, exposed because a
        test that asserts "detail imagery is used near the pad and not on
        orbit" needs to know where the switch is without re-deriving it.
        """
        return float(ecef_to_geodetic(camera.position, as_ellipsoid(self.surface))[2])

    def _imagery_for(self, camera: Camera) -> tuple[list[Any], bool]:
        """Textures to composite, and whether to displace, for one camera.

        Both switches are decided by what the frame can actually resolve
        rather than by flight phase, so they behave the same on a profile
        that was never labelled. A native-resolution crop is composited only
        while the camera's ground sampling is finer than the global
        mosaic's — below that the mosaic already carries more detail than
        the frame can show, and reading a crop would cost the read for
        nothing.
        """
        textures = list(self._textures)
        displace = self.displace and self.camera_altitude(camera) < self.displace_below
        if not self._detail:
            return textures, displace
        distance = float(np.linalg.norm(np.asarray(camera.position) - np.asarray(camera.target)))
        pixel = camera.ground_resolution(distance)
        for crop in self._detail:
            if pixel < textures[0].ground_resolution and crop.covers(
                *self._look_at_degrees(camera)
            ):
                textures.append(crop)
        return textures, displace

    def _look_at_degrees(self, camera: Camera) -> tuple[float, float]:
        """Geodetic latitude and longitude the camera is aimed at, in degrees."""
        latitude, longitude, _ = ecef_to_geodetic(
            np.asarray(camera.target, dtype=np.float64), as_ellipsoid(self.surface)
        )
        return float(np.rad2deg(latitude)), float(np.rad2deg(longitude))

    def _draw_jettisoned(self, ax: Axes, camera: Camera, t: float) -> None:
        """Spent stages, on the trajectories they were integrated along.

        Each is drawn only from its own separation onward — before that it
        was part of the stack and had no separate trajectory — and only up
        to whichever comes first of the current time and its own impact.
        """
        for name, spent in self.jettisoned.items():
            start = float(spent.times[0])
            if t < start:
                continue
            visible = spent.times <= t
            if int(np.count_nonzero(visible)) >= 2:
                draw_track(
                    ax, spent.positions[visible], camera,
                    color=self.style.jettisoned, width=1.6, alpha=0.8,
                    zorder=3.2, surface=self.surface,
                )
            if t > float(spent.times[-1]):
                # Down, or off the end of its own propagation: mark where it
                # stopped rather than pinning it to the last sample as if it
                # were still flying.
                draw_marker(
                    ax, spent.positions[-1], camera, surface=self.surface,
                    s=34, marker="x", c=self.style.jettisoned,
                    linewidths=1.4, zorder=6,
                )
                continue
            draw_marker(
                ax, spent.sample(t)["position"], camera, surface=self.surface,
                s=44, marker="s", c="none", edgecolors=self.style.jettisoned,
                linewidths=1.6, zorder=6.5, label=name,
            )

    def _draw_paths(self, ax: Axes, camera: Camera, t: float) -> None:
        history, style = self.history, self.style
        draw_track(
            ax, history.positions, camera, color=style.track,
            width=style.track_width, alpha=style.track_alpha,
            zorder=3.0, surface=self.surface,
        )
        self._draw_jettisoned(ax, camera, t)
        # Trail bounded in *time*: the bright segment then has the same
        # physical meaning everywhere in the run, which a fixed sample count
        # does not on a non-uniformly sampled conic.
        window = history.times >= t - self.trail_seconds
        window &= history.times <= t
        if int(np.count_nonzero(window)) <= 1:
            return
        if self._values is None:
            draw_track(
                ax, history.positions[window], camera, color=style.track,
                width=style.trail_width, alpha=0.95, zorder=4.0,
                surface=self.surface,
            )
            return
        # A faithful sequential colour map is black at its low end, and a
        # cold trail drawn in black over a night ocean is invisible — the
        # picture then says "no trail" where the physics says "little
        # heating". A backing stroke keeps the geometry legible while the
        # colour on top still carries the value, so dark reads as cold
        # rather than as absent.
        draw_track(
            ax, history.positions[window], camera, color=style.track,
            width=style.trail_width * 1.9, alpha=0.45, zorder=3.5,
            surface=self.surface,
        )
        draw_track(
            ax, history.positions[window], camera,
            width=style.trail_width, alpha=1.0, zorder=4.0,
            surface=self.surface,
            values=self._values[window], cmap=style.heat_cmap,
            vmin=float(np.min(self._values)), vmax=float(np.max(self._values)),
        )

    def _draw_vehicle(
        self, ax: Axes, camera: Camera, state: Mapping[str, Any], t: float
    ) -> None:
        if "dcm" in state and self.staging is not None and self.mould_line is not None:
            # The stack as the mass model says it stands at this instant: a
            # vehicle that has dropped its first stage is drawn shorter by
            # the length the model says it lost, not by a fixed fraction.
            drawn = draw_stack(
                ax,
                state["position"],
                state["dcm"],
                camera,
                stack_polylines(self.staging, t, *self.mould_line),
                color=self.style.vehicle,
                surface=self.surface,
            )
            if drawn:
                return
        if "dcm" in state:
            draw_vehicle(
                ax, state["position"], state["dcm"], camera,
                color=self.style.vehicle, surface=self.surface,
            )
            return
        # No attitude was computed, so none is drawn. A marker says "point
        # mass"; an oriented glyph from an invented identity quaternion
        # would say something the run never established.
        draw_marker(
            ax, state["position"], camera, surface=self.surface,
            s=90, marker="o", c=self.style.vehicle,
            edgecolors=self.style.track, linewidths=2.0, zorder=7,
        )

    def _draw_overlays(
        self, ax: Axes, camera: Camera, state: Mapping[str, Any], t: float
    ) -> None:
        statuses = draw_sites(
            ax, self.sites, camera, self.coverage, t,
            surface=self.surface, style=self.style,
        )
        if self.horizon_rings:
            altitude = float(
                ecef_to_geodetic(state["position"], as_ellipsoid(self.surface))[2]
            )
            for radar in self.sites:
                if statuses.get(radar.name) == "active":
                    draw_horizon_ring(
                        ax, radar, altitude, camera,
                        surface=self.surface,
                        color=self.style.site_active,
                    )
        for _label, (point, kwargs) in self.markers.items():
            draw_marker(
                ax, point, camera, surface=self.surface, **dict(kwargs)
            )

        # Where each declared event happens, in the world. Past events are
        # solid and the next one hollow, so the picture answers "what just
        # happened" and "what is coming" without reading the HUD.
        for event in self.history.events:
            if event.name in ("lift-off", "impact"):
                continue
            past = event.time <= t
            draw_marker(
                ax, self.history.sample(event.time)["position"], camera,
                surface=self.surface, s=70, marker="D",
                c="#FFD166" if past else "none",
                edgecolors="#FFD166", linewidths=1.4, zorder=6,
            )

    def _draw_hud(self, ax: Axes, state: Mapping[str, Any], t: float) -> None:
        history, style = self.history, self.style
        altitude = float(
            ecef_to_geodetic(state["position"], as_ellipsoid(self.surface))[2]
        )
        speed = float(np.linalg.norm(state["velocity"]))
        phase = history.phase_at(t)
        lines: list[tuple[str, str, int, str]] = [
            (history.label, style.track, 16, "bold"),
        ]
        if phase:
            lines.append((phase.upper(), PHASE_COLORS.get(phase, style.track), 13, "bold"))
        lines += [
            (f"T+{t / 60:6.1f} min", style.text, 14, "normal"),
            (f"altitude {altitude / 1e3:7.1f} km", "#BFD9FF", 12, "normal"),
            (f"speed    {speed / 1e3:7.2f} km/s", "#BFD9FF", 12, "normal"),
            (f"to end   {(history.times[-1] - t) / 60:6.1f} min", "#FFB4A8", 12, "normal"),
        ]
        if self.staging is not None:
            # What is still attached, and what its structure weighs. **Dry**
            # mass, explicitly: the propellant left in a stage part-way
            # through its burn is not something this plan tracks, and a line
            # reading 208 t at parking orbit — the gross figure — would be a
            # worse answer than a narrower true one.
            present = self.staging.present_at(t)
            dry = sum(self.staging.stage(name).dry_mass for name in present)
            lines.append((" + ".join(present) or "nothing left", "#C8D6E5", 11, "normal"))
            lines.append((f"dry mass {dry / 1e3:7.1f} t", "#C8D6E5", 11, "normal"))
        upcoming = history.next_event(t)
        if upcoming is not None:
            lines.append((
                f"next {upcoming.name} in {(upcoming.time - t) / 60:4.1f} min",
                "#FFD166", 11, "normal",
            ))
        if self._playback is not None:
            n_frames, fps = self._playback
            lines.append((
                f"playback x{self.playback_rate(t, n_frames, fps):,.0f} real time",
                "#9AA7B2", 10, "normal",
            ))
        if self.color_by is not None and self.color_by in state:
            lines.append((_format_extra(self.color_by, state[self.color_by]),
                          "#FFD166", 12, "normal"))
        if self.coverage is not None:
            seen = self.coverage.detected and t >= self.coverage.first_detection_time
            lines.append((
                f"DETECTED - {self.coverage.first_detecting_site}" if seen
                else "below radar horizon",
                style.site_active if seen else style.site_idle,
                12,
                "bold",
            ))
        y = 0.965
        for text, color, size, weight in lines:
            ax.text(
                0.015, y, text, transform=ax.transAxes, color=color,
                fontsize=size, va="top", family="monospace", weight=weight,
            )
            y -= 0.046

    # -- a sequence ------------------------------------------------------

    def times(self, n_frames: int) -> _FloatArray:
        """The time grid a sequence of ``n_frames`` is drawn on.

        Exposed so a caller — or a test — can check the endpoints without
        rendering anything. They are the history's own endpoints, which is
        the property the index-strided version failed to have.

        With ``pacing="phase"`` the grid is **piecewise** uniform in time:
        each declared phase gets a share of the frames set by
        ``duration ** pacing_exponent`` rather than by duration alone.
        Uniform pacing is defensible and unwatchable here — a fractional
        orbital profile is 4 % boost, 71 % parking coast and 25 % descent,
        so at 130 frames the entire powered ascent got 5 of them and the
        parking coast got 92 of a scene in which nothing visibly changes.
        The exponent compresses the long quiet leg without deleting it: at
        0.45 the same run gives boost 17, parking 63, deorbit 32, entry 18.

        Implemented as a piecewise-linear warp sampled uniformly, which
        makes the endpoints exact and the grid strictly increasing by
        construction rather than by rounding frame counts and hoping.
        """
        if n_frames < 2:
            msg = f"need at least two frames, got {n_frames}"
            raise ValueError(msg)
        if self._profile is not None:
            return self._profile.grid(int(n_frames))

        start, stop = float(self.history.times[0]), float(self.history.times[-1])
        phases = self.history.phases
        if self.pacing == "uniform" or not phases:
            return np.linspace(start, stop, int(n_frames))

        weights = np.array([max(p.duration, 0.0) ** self.pacing_exponent for p in phases])
        total = float(weights.sum())
        if total <= 0.0:  # pragma: no cover - a zero-length flight
            return np.linspace(start, stop, int(n_frames))
        edges_u = np.concatenate([[0.0], np.cumsum(weights / total)])
        edges_t = np.array([p.start_time for p in phases] + [phases[-1].end_time])
        grid = np.interp(np.linspace(0.0, 1.0, int(n_frames)), edges_u, edges_t)
        grid[0], grid[-1] = start, stop
        return np.asarray(grid)

    def playback_rate(self, time: float, n_frames: int, fps: int) -> float:
        """Multiple of real time the frame at ``time`` plays back at.

        Reported in the HUD because phase pacing deliberately makes it
        non-constant, and a viewer who cannot see the rate cannot tell a
        long coast from a fast one.
        """
        if self._profile is not None:
            return self._profile.rate(float(time), int(n_frames), int(fps))
        phases = self.history.phases
        video = n_frames / max(fps, 1)
        if self.pacing == "uniform" or not phases:
            return float(self.history.duration / video)
        weights = np.array([max(p.duration, 0.0) ** self.pacing_exponent for p in phases])
        share = weights / float(weights.sum())
        for phase, fraction in zip(phases, share, strict=True):
            if phase.contains(time):
                return float(phase.duration / max(fraction * video, 1e-9))
        return float(self.history.duration / video)

    def render_sequence(
        self,
        filename: str | Path,
        seconds: float = 75.0,
        fps: int = 30,
        n_frames: int | None = None,
        dpi: int = 100,
        progress: bool = False,
    ) -> Path:
        """Write the whole history to ``filename``.

        Parameters
        ----------
        seconds:
            **Length of the output video.** The frame count follows from it
            and ``fps``, rather than the other way round, because the frame
            count is not what anyone wants to specify: 130 frames at 20 fps
            is 6.5 seconds, and a 72-minute flight compressed into 6.5
            seconds is unwatchable no matter how the frames are paced. At
            the default 45 s and 30 fps the same flight gets 1,350 frames,
            and phase pacing then gives the powered ascent about six
            seconds of screen time instead of under one.
        fps:
            Frames per second of the output.
        n_frames:
            Explicit frame count, overriding ``seconds``. For tests and for
            quick previews.
        dpi:
            Output resolution multiplier.
        progress:
            Print a line every twenty frames.

        Returns
        -------
        pathlib.Path
            The file written.

        Notes
        -----
        The container is chosen from the extension: ``.gif`` through Pillow,
        ``.mp4``/``.mov``/``.mkv`` through ffmpeg as H.264. See
        :func:`video_writer` on why the latter is preferred.
        """
        if n_frames is None:
            if not (np.isfinite(seconds) and seconds > 0.0):
                msg = f"seconds must be finite and > 0, got {seconds}"
                raise ValueError(msg)
            n_frames = max(round(float(seconds) * int(fps)), 2)
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        out = Path(filename)
        writer = video_writer(out, fps)
        if out.suffix.lower() in _VIDEO_SUFFIXES and (
            self.width % 2 or self.height % 2
        ):
            msg = (
                f"H.264 with yuv420p needs even frame dimensions, got "
                f"{self.width}x{self.height}"
            )
            raise ValueError(msg)

        grid = self.times(n_frames)
        self._playback = (int(n_frames), int(fps))
        fig = plt.figure(figsize=(self.width / 100, self.height / 100), dpi=100)
        fig.patch.set_facecolor("black")
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        ax.axis("off")

        def draw(index: int) -> list[Any]:
            if progress and index % 20 == 0:
                print(f"  frame {index + 1}/{n_frames}  T+{grid[index] / 60:.1f} min")
            self.frame_at(float(grid[index]), ax=ax)
            return []

        FuncAnimation(
            fig, draw, frames=n_frames, interval=1000 // fps, blit=False
        ).save(str(out), writer=writer, dpi=dpi)
        plt.close(fig)
        return out

    def figure_at(self, time: float) -> tuple[Figure, Frame]:
        """A standalone figure for one instant, for a static plate."""
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(self.width / 100, self.height / 100), dpi=100)
        fig.patch.set_facecolor("black")
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        return fig, self.frame_at(time, ax=ax)
