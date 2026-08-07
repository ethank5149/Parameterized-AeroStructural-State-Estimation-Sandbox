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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from passes.batch.backend import Backend
from passes.geodesy import WGS84_MEAN_RADIUS
from passes.viz.globe import Camera, load_texture, project, render, to_device
from passes.viz.history import SimulationHistory
from passes.viz.scene import (
    PHASE_COLORS,
    ChaseRig,
    SceneStyle,
    draw_horizon_ring,
    draw_marker,
    draw_sites,
    draw_timeline,
    draw_track,
    draw_vehicle,
    starfield,
)

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
    body_radius: float = WGS84_MEAN_RADIUS

    @property
    def position(self) -> _FloatArray:
        return np.asarray(self.state["position"], dtype=np.float64)

    def pixel_of(self, point: _FloatArray) -> tuple[float, float, bool]:
        """Project a world point through this frame's camera."""
        px, py, visible = project(
            np.atleast_2d(point), self.camera, radius=self.body_radius
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
        Equirectangular globe texture. Loaded from the packaged Blue Marble
        when omitted, once, and held — reloading a 4096x2048 image per frame
        was the other easy waste in the notebook version.
    body_radius:
        Sphere radius (m). Must match whatever built the history.
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

        ``"track"`` is the default: the light follows the vehicle's own
        local vertical, so the ground directly beneath it is always fully
        lit and stays lit as the vehicle moves. It is not a real sun and
        does not pretend to be one; it is a key light on the thing being
        filmed.
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
        texture: _FloatArray | None = None,
        body_radius: float = WGS84_MEAN_RADIUS,
        rig: ChaseRig | None = None,
        style: SceneStyle | None = None,
        sites: Sequence[RadarSite] = (),
        coverage: CoverageResult | None = None,
        markers: Mapping[str, tuple[_FloatArray, Mapping[str, Any]]] | None = None,
        sun: _FloatArray | None = None,
        lighting: str = "track",
        color_by: str | None = None,
        trail_seconds: float | None = None,
        width: int = 1280,
        height: int = 720,
        ambient: float = 0.22,
        atmosphere: float = 0.55,
        specular: float = 0.06,
        horizon_rings: bool = True,
        timeline: bool = True,
        pacing: str = "phase",
        pacing_exponent: float = 0.45,
        backend: Backend = "numpy",
    ) -> None:
        if width < 2 or height < 2:
            msg = f"frame must be at least 2x2 pixels, got {width}x{height}"
            raise ValueError(msg)
        self.history = history
        self.body_radius = float(body_radius)
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
        self.width, self.height = int(width), int(height)
        self.ambient, self.atmosphere, self.specular = ambient, atmosphere, specular
        self.horizon_rings = horizon_rings
        self.timeline = timeline
        if pacing not in ("phase", "uniform"):
            msg = f"pacing must be 'phase' or 'uniform', got {pacing!r}"
            raise ValueError(msg)
        self.pacing = pacing
        self.pacing_exponent = float(pacing_exponent)
        self._playback: tuple[int, int] | None = None
        self.backend: Backend = backend

        # to_device is a no-op when the texture already lives on the
        # requested backend, so callers may hand in either a host array or a
        # device one they uploaded once and share between animators.
        self._texture = to_device(load_texture() if texture is None else texture, backend)
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
            body_radius=self.body_radius,
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
            position = np.asarray(state["position"], dtype=np.float64)
            light = position / max(float(np.linalg.norm(position)), 1e-12)
        else:
            light = None

        image, _ = render(
            camera,
            self._texture,
            self.body_radius,
            sun=light,
            ambient=self.ambient,
            atmosphere=self.atmosphere,
            specular=self.specular,
            background=self._sky,
            backend=self.backend,
        )
        ax.imshow(np.clip(image, 0.0, 1.0), interpolation="bilinear", zorder=0)
        ax.set_xlim(0, self.width)
        ax.set_ylim(self.height, 0)

        self._draw_paths(ax, camera, t)
        self._draw_vehicle(ax, camera, state)
        self._draw_overlays(ax, camera, state, t)
        self._draw_hud(ax, state, t)
        if self.timeline and history.phases:
            draw_timeline(
                ax, history.phases, history.events, t - history.times[0],
                history.duration, style=self.style,
            )
        return Frame(t, state, camera, ax, self.body_radius)

    def _draw_paths(self, ax: Axes, camera: Camera, t: float) -> None:
        history, style = self.history, self.style
        draw_track(
            ax, history.positions, camera, color=style.track,
            width=style.track_width, alpha=style.track_alpha,
            zorder=3.0, body_radius=self.body_radius,
        )
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
                body_radius=self.body_radius,
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
            body_radius=self.body_radius,
        )
        draw_track(
            ax, history.positions[window], camera,
            width=style.trail_width, alpha=1.0, zorder=4.0,
            body_radius=self.body_radius,
            values=self._values[window], cmap=style.heat_cmap,
            vmin=float(np.min(self._values)), vmax=float(np.max(self._values)),
        )

    def _draw_vehicle(self, ax: Axes, camera: Camera, state: Mapping[str, Any]) -> None:
        if "dcm" in state:
            draw_vehicle(
                ax, state["position"], state["dcm"], camera,
                color=self.style.vehicle, body_radius=self.body_radius,
            )
            return
        # No attitude was computed, so none is drawn. A marker says "point
        # mass"; an oriented glyph from an invented identity quaternion
        # would say something the run never established.
        draw_marker(
            ax, state["position"], camera, body_radius=self.body_radius,
            s=90, marker="o", c=self.style.vehicle,
            edgecolors=self.style.track, linewidths=2.0, zorder=7,
        )

    def _draw_overlays(
        self, ax: Axes, camera: Camera, state: Mapping[str, Any], t: float
    ) -> None:
        statuses = draw_sites(
            ax, self.sites, camera, self.coverage, t,
            body_radius=self.body_radius, style=self.style,
        )
        if self.horizon_rings:
            altitude = float(np.linalg.norm(state["position"])) - self.body_radius
            for radar in self.sites:
                if statuses.get(radar.name) == "active":
                    draw_horizon_ring(
                        ax, radar, altitude, camera,
                        body_radius=self.body_radius,
                        color=self.style.site_active,
                    )
        for _label, (point, kwargs) in self.markers.items():
            draw_marker(
                ax, point, camera, body_radius=self.body_radius, **dict(kwargs)
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
                body_radius=self.body_radius, s=70, marker="D",
                c="#FFD166" if past else "none",
                edgecolors="#FFD166", linewidths=1.4, zorder=6,
            )

    def _draw_hud(self, ax: Axes, state: Mapping[str, Any], t: float) -> None:
        history, style = self.history, self.style
        altitude = float(np.linalg.norm(state["position"])) - self.body_radius
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
        n_frames: int = 130,
        fps: int = 20,
        dpi: int = 100,
        progress: bool = False,
    ) -> Path:
        """Write the whole history to ``filename``.

        The container is chosen from the extension: ``.gif`` through Pillow,
        ``.mp4``/``.mov``/``.mkv`` through ffmpeg as H.264. See
        :func:`video_writer` on why the latter is preferred.

        Returns
        -------
        pathlib.Path
            The file written.
        """
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
