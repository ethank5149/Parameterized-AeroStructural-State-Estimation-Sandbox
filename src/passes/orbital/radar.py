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
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from passes.geodesy import WGS84_MEAN_RADIUS, GeodeticPosition, great_circle_range
from passes.orbital.warning import DetectionWindow, detection_window

__all__ = [
    "EARLY_WARNING_SITES",
    "SATELLITE_SENSORS",
    "CoverageResult",
    "RadarSite",
    "SatelliteSensor",
    "SensorCapability",
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
    capability: SensorCapability | None = None


@dataclass(frozen=True)
class SensorCapability:
    """Capabilities and limitations of a sensor, for strategic analysis.

    Attributes
    ----------
    wavelength_band:
        RF band or optical regime the sensor operates in.
    peak_power_kw:
        Transmitted power (kW), a proxy for detection range.
    aperture_m:
        Antenna diameter or primary mirror (m).
    scan_rate_hz:
        How frequently the sensor can revisit a given direction (Hz).
    max_unambiguous_range_km:
        Maximum unambiguous range (km).
    min_detectable_velocity_mps:
        Minimum detectable radial velocity (m/s), relevant for distinguishing
        targets against clutter.
    is_stationary:
        Whether the sensor is ground-fixed (True) or space-based (False).
    coverage_sector:
        Solid angle the sensor can monitor (steradians), or None if omnidirectional.
    note:
        Operational notes about the sensor's capabilities and limitations.
    """

    wavelength_band: str
    peak_power_kw: float = 0.0
    aperture_m: float = 0.0
    scan_rate_hz: float = 0.0
    max_unambiguous_range_km: float = 0.0
    min_detectable_velocity_mps: float = 0.0
    is_stationary: bool = True
    coverage_sector: float | None = None
    note: str = ""


@dataclass(frozen=True)
class SatelliteSensor:
    """A space-based infrared or radar sensor.

    Unlike ground sensors, satellite sensors have a moving field of regard
    and are characterized by their orbit, not a fixed position. They detect
    boost-phase infrared signatures (hot rocket exhaust plumes) against the
    cold background of space.

    Attributes
    ----------
    name:
        Common name.
    orbit_description:
        Brief description of the constellation or orbit.
    wavelength_band:
        IR band (e.g., "MWIR", "LWIR").
    min_detectable_temperature_k:
        Minimum plume temperature detectable (K).
    n_sats:
        Number of satellites in the constellation.
    note:
        Operational notes.
    """

    name: str
    orbit_description: str
    wavelength_band: str
    min_detectable_temperature_k: float
    n_sats: int = 1
    note: str = ""


def _site(
    name: str,
    latitude: float,
    longitude: float,
    note: str,
    mask_deg: float = 5.0,
    capability: SensorCapability | None = None,
) -> RadarSite:
    return RadarSite(
        name=name,
        position=GeodeticPosition(
            np.deg2rad(latitude), np.deg2rad(longitude), 0.0, label=name
        ),
        mask_elevation=np.deg2rad(mask_deg),
        note=note,
        capability=capability,
    )


def _capability(wavelength: str, power_kw: float, aperture_m: float, **kw: Any) -> SensorCapability:
    """Shorthand for building a SensorCapability."""
    return SensorCapability(
        wavelength_band=wavelength,
        peak_power_kw=power_kw,
        aperture_m=aperture_m,
        **kw,
    )


#: SBX-15 (SBX): sea-based X-band radar, mounted on a converted oil platform.
_SBX = _capability(
    "X-band", 13.0, 11.0,
    max_unambiguous_range_km=600,
    min_detectable_velocity_mps=0.1,
    coverage_sector=4.0 * np.pi / 180.0,  # ~4 degree sector
    note="Mobile sea-based X-band radar on a converted oil-rig hull, "
         "used for midcourse discrimination and tracking. Limited scan sector.",
)

#: AN/TPY-2: TPY-2 radar, the backbone of the US BMD sensor architecture.
_TPY2 = _capability(
    "S-band/X-band", 10.0, 3.7,
    max_unambiguous_range_km=1500,
    min_detectable_velocity_mps=0.2,
    is_stationary=True,
    note="Thaad forward-based radar, long-range search and track. "
         "Can detect and track ballistic missiles through terminal phase.",
)

#: SBX-style sea-based radar with mobile deployment capability.
_SBX_MOBILE = _capability(
    "X-band", 13.0, 11.0,
    max_unambiguous_range_km=600,
    min_detectable_velocity_mps=0.1,
    is_stationary=True,
    note="SBX-type X-band on a mobile sea platform, deployed for specific "
         "operations. Limited deployment window and scan sector.",
)


#: Long-standing, publicly documented early-warning installations.
#:
#: Approximate positions only, and no claim about operational status,
#: tasking or orientation. Ordered roughly north to south within each group
#: so that a coverage report reads geographically.
#:
#: Capability annotations are provided where the system's performance is
#: described in open sources. These are upper-bound proxies — power and
#: aperture set detectability, but track-initiation and decision latency
#: are not modelled here.
EARLY_WARNING_SITES: tuple[RadarSite, ...] = (
    # --- ballistic-missile early warning, northern approaches ---
    _site("Pituffik (Thule)", 76.6, -68.3, "BMEWS, northwest Greenland",
          capability=_capability(
              "L-band", 1.6, 18.3,
              max_unambiguous_range_km=2000,
              min_detectable_velocity_mps=0.5,
              note="Early-Phase BMD, detects boost and ascent phases. "
                   "Fixed beam, 5° elevation mask, covers northern approaches.",
          )),
    _site("Clear", 64.3, -149.2, "UEWR, interior Alaska",
          capability=_capability(
              "L-band", 1.6, 22.0,
              max_unambiguous_range_km=2500,
              min_detectable_velocity_mps=0.5,
              note="UEWR detects and tracks ballistic missiles through midcourse. "
                   "Co-located with GREIA launch support radar.",
          )),
    _site("Fylingdales", 54.4, -0.7, "BMEWS/UEWR, North Yorkshire",
          capability=_capability(
              "L-band", 1.6, 20.1,
              max_unambiguous_range_km=2500,
              min_detectable_velocity_mps=0.5,
              note="UK-operated BMEWS site under the NATO BMEWS Cooperative Programme. "
                   "Provides early warning against northern and central trajectories.",
          )),
    # --- THAAD / PATRIOT forward-based sensors ---
    _site("Shariki (Shariki AB)", 40.8, 140.3, "AN/TPY-2 forward-based, northern Honshu",
          capability=_TPY2),
    _site("Kürecik", 38.3, 37.8, "AN/TPY-2 forward-based, eastern Turkey",
          capability=_TPY2),
    _site("Deveselu", 44.3, 24.4, "AN/TPY-2 forward-based, southern Romania",
          capability=_TPY2),
    _site("Redzikowo", 54.5, 17.1, "AN/TPY-2 forward-based, northern Poland",
          capability=_TPY2),
    _site("Inverclyde (Glasgow Prestwick)", 55.9, -4.6,
          note="AN/TPY-2 forward-based, Scotland. Proposed site covering NE Atlantic approaches.",
          capability=_TPY2, mask_deg=3.0),
    # --- PAVE PAWS / Cobra Dane ---
    _site("Cape Cod", 41.8, -70.5, "PAVE PAES/UEWR, Massachusetts",
          capability=_capability(
              "L-band", 1.6, 30.0,
              max_unambiguous_range_km=4000,
              min_detectable_velocity_mps=0.3,
              note="PAVE PAWS detects and tracks ICBM-class targets through MIDCourse. "
                   "One of the most capable ground-based early-warning radars.",
          )),
    _site("Beale", 39.1, -121.4, "PAVE PAES/UEWR, California",
          capability=_capability(
              "L-band", 1.6, 30.0,
              max_unambiguous_range_km=4000,
              min_detectable_velocity_mps=0.3,
              note="PAVE PAWS, western US coverage. Joint with SBX deployments "
                   "for Pacific trajectory coverage.",
          )),
    _site("Cavalier", 48.7, -97.9, "Cobra Dane / STK, North Dakota",
          capability=_capability(
              "UHF", 3.0, 33.5,
              max_unambiguous_range_km=3000,
              min_detectable_velocity_mps=1.0,
              note="Cobra Dane phased array, used for missile defense and "
                   "space surveillance. Can track multiple targets simultaneously.",
          )),
    _site("Eglin", 30.6, -86.2, "AN/FPS-85/FPS-118, Florida",
          capability=_capability(
              "L-band", 1.6, 27.4,
              max_unambiguous_range_km=3000,
              min_detectable_velocity_mps=0.3,
              note="AN/FPS-85 is a large transportable phased array. "
                   "Capable of simultaneous air and missile warning.",
          )),
    # --- NATO / European systems ---
    _site("Globus II/III", 70.4, 31.1, "Vardo, northern Norway",
          capability=_capability(
              "X-band", 0.5, 2.4,
              max_unambiguous_range_km=500,
              min_detectable_velocity_mps=0.5,
              note="AN/TPY-2 variant on the NATO-run Globus site. "
                   "Provides terminal-area detection for northern Europe.",
          )),
    _site("Vidsa (Skjold AB)", 58.5, 8.0, "AN/TPS-77, southern Norway",
          capability=_capability(
              "L-band", 1.0, 12.0,
              max_unambiguous_range_km=450,
              min_detectable_velocity_mps=0.3,
              note="Long-range TPS-77 radar, NATO air defense. "
                   "Can track low-RCS targets at moderate range.",
          )),
    _site("Pyktila (RAF Lossiemouth)", 57.8, -3.6, "AN/TPS-77, northeastern Scotland",
          capability=_capability(
              "L-band", 1.0, 12.0,
              max_unambiguous_range_km=450,
              min_detectable_velocity_mps=0.3,
              note="TPS-77 under NATO control, covers NE Atlantic "
                   "approach corridors.",
          )),
    # --- Pacific / mid-course ---
    _site("Kwajalein", 8.7, 167.7, "Reagan Test Site, Marshall Islands",
          capability=_capability(
              "X-band", 10.0, 3.7,
              max_unambiguous_range_km=1200,
              min_detectable_velocity_mps=0.1,
              note="X-band phased array, mid-course tracking and discrimination. "
                   "Supports SDI and BMDS tests and operations.",
          )),
    _site("Vandenberg SFB", 34.7, -119.7,
          note="SBX-15 / mobile X-band, California coast. "
               "Sea-Based Radar deployed from Vandenberg for Pacific "
               "operations. Limited scan sector (~4°), but high resolution.",
          capability=_SBX, mask_deg=3.0),
    _site("Andersen AFB", 15.1, 167.0,
          note="AN/TPX-2 / SBX support, Guam. "
               "Guam-based SBX deployment for western Pacific coverage. "
               "Rotational rather than permanently sited.",
          capability=_SBX_MOBILE, mask_deg=3.0),
    # --- Southern hemisphere ---
    _site("Exmouth", -23.5, 113.6, "AN/TPS-77, Western Australia",
          capability=_capability(
              "L-band", 1.0, 12.0,
              max_unambiguous_range_km=450,
              min_detectable_velocity_mps=0.3,
              note="Australian Defence Force long-range radar. Covers the "
                   "Indian Ocean approach, the weakest sector in the northern "
                   "hemisphere-biased network.",
          )),
    _site("Cape Town", -33.9, 18.4, "Denel LS-50 / i-Tracer, South Africa",
          capability=_capability(
              "L-band", 0.8, 12.0,
              max_unambiguous_range_km=400,
              min_detectable_velocity_mps=0.4,
              note="Indigenous South African long-range radar, covers the "
                   "South Atlantic. Not integrated into any western "
                   "early-warning network.",
          )),
    # --- Additional NATO and partner systems ---
    _site("Okno (Zelenograd)", 55.9, 43.2,
          note="AN/TPY-2-style, Vladimir region. Russian early-warning radar at "
               "the Okno site, publicly documented. Included for geographic "
               "completeness, not as a US/NATO system.",
          capability=_TPY2),
    _site("Krasnoyarsk", 56.3, 93.0,
          note="UNIFIED GAZEL radar, Siberia. Russian early-warning radar, part "
               "of the Integrated Aerospace Defence Forces network. "
               "Included for coverage model completeness.",
          capability=_capability(
              "UHF", 5.0, 28.0,
              max_unambiguous_range_km=3000,
              min_detectable_velocity_mps=0.5,
          )),
)


#: Space-based IR sensors for boost-phase detection.
#:
#: These are not point positions on the ground; they are satellite
#: constellations that view large swaths of Earth from orbit. They detect
#: the hot rocket plume against the cold background of space, which is
#: complementary to ground-based radar detection.
#:
#: Key limitation: boost-phase IR sensors have a limited dwell time on any
#: given ground track and cannot track through the atmosphere. Their strength
#: is detecting the launch itself, not the terminal trajectory.
SATELLITE_SENSORS: tuple[SatelliteSensor, ...] = (
     SatelliteSensor(
         name="SBIRS GEO (USAF)",
         orbit_description="# USAF Space-Based Infrared System, geostationary orbit.",
         wavelength_band="MWIR/SWIR",
         min_detectable_temperature_k=600.0,
         n_sats=4,
         note="Four satellites providing continuous hemispheric coverage. "
              "Detects boost-phase IR signatures with a few-minute latency. "
              "Limited by cloud cover and atmospheric opacity in the IR band.",
      ),
    SatelliteSensor(
        name="SBIRS LEO (USAF)",
        orbit_description="1000+ km sun-synchronous orbit, ~30 satellites",
        wavelength_band="MWIR",
        min_detectable_temperature_k=400.0,
        n_sats=30,
        note="Planned constellation providing near-global IR surveillance. "
             "Higher update rate than GEO but gaps during orbital transitions.",
    ),
    SatelliteSensor(
        name="SBX-GEO (proposed)",
        orbit_description="Geostationary over equator",
        wavelength_band="LWIR",
        min_detectable_temperature_k=800.0,
        n_sats=2,
        note="Proposed SBX-derived infrared sensor for geostationary "
             "missile warning. Not yet deployed.",
    ),
    SatelliteSensor(
        name="STK-OCs IR (commercial proxy)",
        orbit_description="~500 km LEO, commercial constellation",
        wavelength_band="MWIR",
        min_detectable_temperature_k=300.0,
        n_sats=40,
        note="Commercial satellite constellation providing IR surveillance "
             "capability. Not dedicated to missile warning but can provide "
             "supplementary boost-phase detection.",
    ),
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
