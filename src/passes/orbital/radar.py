"""Early-warning radar coverage, for warning-time analysis.

:mod:`passes.orbital.warning` gives the horizon geometry for *one* site.
This module supplies the other half of the operational question a fractional
orbital profile raises: warning depends not only on how high a trajectory
flies but on **where the sensors are**, and a profile that arrives on an
unusual bearing may be short-warning for a reason that has nothing to do
with its altitude.

Scope, and what this is for
---------------------------

This is a **strategic-stability analysis** tool, of the kind used openly in
the arms-control literature to reason about warning time and crisis
stability. It computes line-of-sight geometry against publicly documented
radar locations, and nothing else.

What it deliberately does not model, because none of it is public and none
of it is geometry: transmit power, aperture, waveform, search fence
orientation, radar cross-section, track-initiation criteria, or the
command-and-control latency between a first return and a decision. Every
number here is therefore an **upper bound on warning**, and a generous one.
A result from this module says "the target was above the horizon", which is
necessary for detection and nowhere near sufficient.

On the site list
----------------

Locations are the published, approximate positions of long-standing
early-warning installations — the sort of thing that appears in
Federation of American Scientists and Union of Concerned Scientists
analyses, in arms-control journals, and on the operators' own public
pages. They are given to about a tenth of a degree, which is far finer than
the model's other approximations deserve, and no attempt is made to
represent which are active, what they are currently tasked with, or how
they are oriented.

Mask elevations are **assumed**, not published. Five degrees is used as a
representative figure for a large ground radar working against terrain,
clutter and refraction; the true values are site-specific and not open.
Because the mask matters most at low altitude — it removes about 21 % of the
visibility radius at 150 km against 9 % at 1300 km — that assumption is
load-bearing for exactly the fractional-orbital case, and is flagged rather
than buried.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from passes.geodesy import WGS84_MEAN_RADIUS, GeodeticPosition, great_circle_range
from passes.orbital.warning import DetectionWindow, detection_window

__all__ = [
    "EARLY_WARNING_SITES",
    "CoverageResult",
    "RadarSite",
    "coverage",
    "site",
]


@dataclass(frozen=True)
class RadarSite:
    """One ground sensor, as a position and a horizon mask.

    Attributes
    ----------
    name:
        Common name.
    position:
        Geodetic location. Approximate and public.
    mask_elevation:
        Assumed minimum working elevation (rad). See the module docstring:
        this is a modelling choice, not a published figure.
    note:
        What the site is, in one line.
    """

    name: str
    position: GeodeticPosition
    mask_elevation: float = np.deg2rad(5.0)
    note: str = ""


def _site(
    name: str, latitude: float, longitude: float, note: str, mask_deg: float = 5.0
) -> RadarSite:
    return RadarSite(
        name=name,
        position=GeodeticPosition(
            np.deg2rad(latitude), np.deg2rad(longitude), 0.0, label=name
        ),
        mask_elevation=np.deg2rad(mask_deg),
        note=note,
    )


#: Long-standing, publicly documented early-warning installations.
#:
#: Approximate positions only, and no claim about operational status,
#: tasking or orientation. Ordered roughly north to south within each group
#: so that a coverage report reads geographically.
EARLY_WARNING_SITES: tuple[RadarSite, ...] = (
    # --- ballistic-missile early warning, northern approaches ---
    _site("Pituffik (Thule)", 76.6, -68.3, "BMEWS/UEWR, northwest Greenland"),
    _site("Clear", 64.3, -149.2, "UEWR, interior Alaska"),
    _site("Fylingdales", 54.4, -0.7, "BMEWS/UEWR, North Yorkshire"),
    # --- perimeter acquisition and space surveillance, CONUS ---
    _site("Cavalier", 48.7, -97.9, "Perimeter acquisition radar, North Dakota"),
    _site("Cape Cod", 41.8, -70.5, "PAVE PAWS/UEWR, Massachusetts"),
    _site("Beale", 39.1, -121.4, "PAVE PAWS/UEWR, California"),
    _site("Eglin", 30.6, -86.2, "AN/FPS-85 phased array, Florida"),
    _site("Shariki", 40.8, 140.3, "AN/TPY-2 forward-based, northern Honshu"),
    # --- Pacific and mid-course instrumentation ---
    _site("Kwajalein", 8.7, 167.7, "Reagan Test Site radars, Marshall Islands"),
    # --- NATO / European ---
    _site("Kürecik", 38.3, 37.8, "AN/TPY-2 forward-based, eastern Turkey"),
    _site("Deveselu", 44.3, 24.4, "Aegis Ashore, southern Romania"),
    _site("Redzikowo", 54.5, 17.1, "Aegis Ashore, northern Poland"),
    _site("Globus II/III", 70.4, 31.1, "Vardo, northern Norway"),
)


def site(name: str) -> RadarSite:
    """Look a site up by name, case-insensitively and by prefix.

    Raises
    ------
    KeyError
        If the name matches no site, or more than one.
    """
    key = name.strip().lower()
    matches = [s for s in EARLY_WARNING_SITES if s.name.lower().startswith(key)]
    if not matches:
        available = ", ".join(s.name for s in EARLY_WARNING_SITES)
        msg = f"no early-warning site matching {name!r}; available: {available}"
        raise KeyError(msg)
    if len(matches) > 1:
        msg = f"{name!r} is ambiguous: {', '.join(s.name for s in matches)}"
        raise KeyError(msg)
    return matches[0]


@dataclass(frozen=True)
class CoverageResult:
    """What a network of sites learns from one trajectory.

    Attributes
    ----------
    windows:
        Per-site detection windows, keyed by site name.
    first_detection_time:
        Earliest detection across the network (s); ``nan`` if undetected by
        every site.
    warning_time:
        Seconds from that earliest detection to the last trajectory sample.
        This is the network's warning, and it is set by whichever single
        site sees the trajectory first — which is why adding sensors far
        from the approach corridor changes nothing.
    first_detecting_site:
        Which site that was; empty string if undetected.
    detecting_sites:
        Names of every site that ever sees the trajectory, in detection
        order.
    """

    windows: dict[str, DetectionWindow] = field(repr=False)
    first_detection_time: float
    warning_time: float
    first_detecting_site: str
    detecting_sites: tuple[str, ...]

    @property
    def detected(self) -> bool:
        return bool(self.detecting_sites)


def coverage(
    times: ArrayLike,
    altitudes: ArrayLike,
    subpoints: list[GeodeticPosition],
    sites: tuple[RadarSite, ...] = EARLY_WARNING_SITES,
    body_radius: float = WGS84_MEAN_RADIUS,
) -> CoverageResult:
    """Run a sampled trajectory past a network of sensors.

    Parameters
    ----------
    times:
        Sample times (s), strictly increasing. The last sample is taken as
        impact, so the warning time is measured to it.
    altitudes:
        Vehicle altitude above the sphere at each sample (m).
    subpoints:
        Sub-vehicle ground point at each sample, same length as ``times``.
    sites:
        Sensors to run against. Defaults to
        :data:`EARLY_WARNING_SITES`.
    body_radius:
        Sphere radius (m) used for the central-angle geometry.

    Returns
    -------
    CoverageResult

    Notes
    -----
    Each site is evaluated independently and the network warning is the
    earliest of them. That is the correct composition for *warning* — one
    site is enough to raise an alarm — but it is not the right composition
    for track quality or discrimination, which need several sites with
    favourable geometry and are not modelled here.
    """
    t = np.asarray(times, dtype=np.float64)
    h = np.asarray(altitudes, dtype=np.float64)
    if t.ndim != 1 or t.shape != h.shape or t.size < 2:
        msg = "times and altitudes must be 1-D arrays of equal length >= 2"
        raise ValueError(msg)
    if len(subpoints) != t.size:
        msg = f"need {t.size} subpoints to match the samples, got {len(subpoints)}"
        raise ValueError(msg)
    if not sites:
        msg = "at least one radar site is required"
        raise ValueError(msg)

    windows: dict[str, DetectionWindow] = {}
    for radar in sites:
        central = np.array(
            [great_circle_range(radar.position, p) / body_radius for p in subpoints]
        )
        windows[radar.name] = detection_window(
            t, h, central, radar.mask_elevation, body_radius
        )

    seen = [(w.first_detection_time, name) for name, w in windows.items() if w.detected]
    if not seen:
        return CoverageResult(windows, float("nan"), float("nan"), "", ())
    seen.sort()
    earliest, first_name = seen[0]
    return CoverageResult(
        windows=windows,
        first_detection_time=float(earliest),
        warning_time=float(t[-1] - earliest),
        first_detecting_site=first_name,
        detecting_sites=tuple(name for _, name in seen),
    )
