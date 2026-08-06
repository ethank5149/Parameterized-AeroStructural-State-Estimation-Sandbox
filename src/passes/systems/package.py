"""Launch packages: a portable, versioned configuration format.

Every scenario in this framework has so far been assembled in Python — a
``GeodeticPosition`` here, a ``NAMED_ARCHITECTURES`` lookup there, a
scattering of keyword arguments to :func:`passes.systems.budget.evaluate`.
That is fine for one-off analysis and useless for anything that has to be
reviewed, diffed, archived, or handed to someone else.

This module defines the file format that fixes that: a **launch package**,
stating a complete scenario — where it starts, what it is aimed at, what
flies it, what profile it flies, what is watching, and what it is being
optimised for — in one human-readable, machine-validated document. It is
the analogue of the tabulated environment file a thermal-response code is
handed before a run, or the mission data a vehicle is loaded with before
flight: everything chosen up front, in one place, in a form that can be
checked.

The design rules, and why each one is here
-------------------------------------------

**Units live in the key names.** ``latitude_deg``, not ``latitude``;
``parking_altitude_m``, not ``parking_altitude``. This costs verbosity and
buys the one class of error this repository has been bitten by repeatedly: a
number of the right magnitude in the wrong unit. A file saying
``latitude_deg = 0.9`` is obviously wrong; one saying ``latitude = 0.9``
might be radians and might be a survey blunder, and no loader could tell.

**Angles are degrees on disk, radians in memory.** Humans author degrees;
every function in this codebase takes radians. The conversion happens once,
at the boundary.

**The schema version is required.** A package without a ``schema`` key is
rejected rather than guessed at, so that when the format changes, old files
fail loudly at load instead of being silently reinterpreted.

**Vocabularies are closed and checked against the code.** ``architecture``
must name a real entry in
:data:`passes.systems.architecture.NAMED_ARCHITECTURES`, and ``imu_grade``
one in :data:`passes.guidance.inertial.IMU_GRADES`. A typo is an error at
load with the valid options listed, never a silent fallback.

**Two formats, one model.** TOML is the authoring format — comments,
sections, no significant whitespace, and :mod:`tomllib` reads it from the
standard library. JSON is the interchange format. Both round-trip through
the same dataclass, and a test asserts they agree.

What a package deliberately does not contain
---------------------------------------------

No results, and no defaults the code already owns. A package records what
was *chosen*; anything computable from those choices stays computable, so
that a package plus a version of this repository reproduces a run, and a
package alone never disagrees with the code about a derived quantity.

Example
-------

.. code-block:: toml

    schema = "passes.launch-package/1"

    [metadata]
    name = "minimum-energy reference"

    [launch]
    latitude_deg = 45.0
    longitude_deg = -100.0

    [[aimpoints]]
    latitude_deg = 50.0
    longitude_deg = 40.0

    [profile]
    architecture = "ballistic-single"
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import tomllib

from passes.geodesy import GeodeticPosition
from passes.guidance.inertial import IMU_GRADES
from passes.systems.architecture import NAMED_ARCHITECTURES
from passes.systems.budget import MissionRequest

__all__ = [
    "OBJECTIVES",
    "SCHEMA",
    "LaunchPackage",
    "PackageError",
    "Profile",
    "Sensor",
    "Vehicle",
    "load_package",
]

#: The schema identifier every package must carry.
#:
#: Bump the trailing integer on any change that would make an older file
#: mean something different. Adding an *optional* key with a default that
#: reproduces the previous behaviour does not require a bump; changing a
#: default, renaming a key, or altering a unit does.
SCHEMA = "passes.launch-package/1"

#: Objectives a package may ask to be optimised for.
#:
#: Deliberately a closed set. These are the quantities the framework can
#: actually compute end to end; anything else would be a wish stated in a
#: configuration file, which is the failure mode the whole format exists to
#: avoid.
OBJECTIVES: tuple[str, ...] = (
    "warning_time",
    "burnout_speed",
    "flight_time",
    "cep",
)


class PackageError(ValueError):
    """A launch package is malformed, and this says exactly how."""


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        msg = f"{where}: required key {key!r} is missing"
        raise PackageError(msg)
    return mapping[key]


def _number(value: Any, key: str, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{where}: {key!r} must be a number, got {type(value).__name__}"
        raise PackageError(msg)
    number = float(value)
    if not np.isfinite(number):
        msg = f"{where}: {key!r} must be finite, got {number}"
        raise PackageError(msg)
    return number


def _check_keys(mapping: dict[str, Any], allowed: frozenset[str], where: str) -> None:
    """Refuse keys the schema does not define.

    Silently ignoring an unknown key is how a configuration file lies. The
    specific trap this catches is a TOML one and it is easy to fall into:
    a bare ``key = value`` written *after* a ``[table]`` header belongs to
    that table, not to the document root. So a package that looks like it
    sets ``arrival_time_s`` at top level may in fact be setting
    ``vehicle.arrival_time_s``, which means nothing — and without this
    check the loader would use the default and say nothing. That happened
    to the first example package written for this format.
    """
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        msg = (
            f"{where}: unknown key(s) {', '.join(repr(k) for k in unknown)}. "
            f"Known keys here: {', '.join(sorted(allowed))}. "
            "Note that in TOML a bare key written after a [table] header "
            "belongs to that table, not to the document root."
        )
        raise PackageError(msg)


def _choice(value: Any, key: str, where: str, allowed: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in allowed:
        msg = (
            f"{where}: {key!r} must be one of {', '.join(sorted(allowed))}; "
            f"got {value!r}"
        )
        raise PackageError(msg)
    return value


_POSITION_KEYS = frozenset({"latitude_deg", "longitude_deg", "altitude_m", "label"})


def _position(
    entry: dict[str, Any], where: str, extra: frozenset[str] = frozenset()
) -> GeodeticPosition:
    """Read a geodetic position, in degrees, with the units named."""
    _check_keys(entry, _POSITION_KEYS | extra, where)
    latitude = _number(_require(entry, "latitude_deg", where), "latitude_deg", where)
    longitude = _number(_require(entry, "longitude_deg", where), "longitude_deg", where)
    altitude = _number(entry.get("altitude_m", 0.0), "altitude_m", where)
    if not -90.0 <= latitude <= 90.0:
        msg = f"{where}: latitude_deg must lie in [-90, 90], got {latitude}"
        raise PackageError(msg)
    if not -180.0 <= longitude <= 360.0:
        msg = f"{where}: longitude_deg must lie in [-180, 360], got {longitude}"
        raise PackageError(msg)
    return GeodeticPosition(
        float(np.deg2rad(latitude)),
        float((np.deg2rad(longitude) + np.pi) % (2.0 * np.pi) - np.pi),
        altitude,
        str(entry.get("label", "")),
    )


def _position_dict(position: GeodeticPosition) -> dict[str, Any]:
    out: dict[str, Any] = {
        "latitude_deg": float(np.rad2deg(position.latitude)),
        "longitude_deg": float(np.rad2deg(position.longitude)),
    }
    if position.altitude:
        out["altitude_m"] = float(position.altitude)
    if position.label:
        out["label"] = position.label
    return out


@dataclass(frozen=True)
class Vehicle:
    """What flies the profile.

    Attributes
    ----------
    ballistic_coefficient:
        :math:`m/(C_D A)` (kg/m^2). Sets entry deceleration depth, descent
        time, and whether the vehicle crosses roll resonance once or twice.
    lift_to_drag:
        Hypersonic L/D. Zero for a non-lifting reentry vehicle.
    imu_grade:
        A key of :data:`passes.guidance.inertial.IMU_GRADES`. This is the
        single largest lever on the inertial error budget, and it is a
        procurement choice rather than a physical one, which is why it
        belongs in a configuration file.
    boost_burn_time:
        Seconds of powered flight, over which the inertial errors integrate.
    """

    ballistic_coefficient: float = 20.0e3
    lift_to_drag: float = 0.0
    imu_grade: str = "aviation"
    boost_burn_time: float = 300.0


@dataclass(frozen=True)
class Profile:
    """How the trajectory is flown.

    Attributes
    ----------
    architecture:
        A key of :data:`passes.systems.architecture.NAMED_ARCHITECTURES`.
    parking_altitude:
        Fractional-orbital parking arc altitude (m).
    entry_interface_altitude:
        Where the atmosphere is taken to begin (m).
    burnout_flight_path_angle:
        Angle above local horizontal at burnout (rad), or ``None`` for the
        minimum-energy :math:`\\gamma^*`. ``None`` is not the same as a
        number that happens to equal :math:`\\gamma^*`: it says the choice
        was delegated, which survives a change of range.
    ballistic_entry_angle:
        Entry-interface flight-path angle (rad) for the ballistic leg.
    """

    architecture: str = "ballistic-single"
    parking_altitude: float = 150.0e3
    entry_interface_altitude: float = 100.0e3
    burnout_flight_path_angle: float | None = None
    ballistic_entry_angle: float = float(np.deg2rad(21.8))


@dataclass(frozen=True)
class Sensor:
    """One ground sensor the scenario is to be assessed against."""

    name: str
    position: GeodeticPosition
    mask_elevation: float = float(np.deg2rad(5.0))
    note: str = ""


@dataclass(frozen=True)
class LaunchPackage:
    """A complete, portable scenario definition.

    Attributes
    ----------
    launch:
        Launch site.
    aimpoints:
        One or more targets. The farthest sets the range the architecture
        must close.
    profile, vehicle:
        See :class:`Profile` and :class:`Vehicle`.
    arrival_time:
        Seconds from launch at which the aimpoints must be reached.
    sensors:
        Ground sensors for warning analysis. Empty means "use the
        framework's default network", which is recorded as an explicit
        empty list rather than as a magic value.
    objectives:
        What this package is being optimised for, from :data:`OBJECTIVES`.
        Order is meaningful: the first is primary.
    metadata:
        Free-form provenance. Never read by the code.
    """

    launch: GeodeticPosition
    aimpoints: tuple[GeodeticPosition, ...]
    profile: Profile = field(default_factory=Profile)
    vehicle: Vehicle = field(default_factory=Vehicle)
    arrival_time: float = 3600.0
    sensors: tuple[Sensor, ...] = ()
    objectives: tuple[str, ...] = ("warning_time",)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.aimpoints:
            msg = "a launch package needs at least one aimpoint"
            raise PackageError(msg)
        if self.profile.architecture not in NAMED_ARCHITECTURES:
            msg = (
                f"profile.architecture {self.profile.architecture!r} is not a known "
                f"architecture; available: {', '.join(sorted(NAMED_ARCHITECTURES))}"
            )
            raise PackageError(msg)
        if self.vehicle.imu_grade not in IMU_GRADES:
            msg = (
                f"vehicle.imu_grade {self.vehicle.imu_grade!r} is not a known grade; "
                f"available: {', '.join(sorted(IMU_GRADES))}"
            )
            raise PackageError(msg)
        for objective in self.objectives:
            if objective not in OBJECTIVES:
                msg = (
                    f"objective {objective!r} is not computable by this framework; "
                    f"available: {', '.join(OBJECTIVES)}"
                )
                raise PackageError(msg)
        if not (np.isfinite(self.arrival_time) and self.arrival_time > 0.0):
            msg = f"arrival_time_s must be finite and > 0, got {self.arrival_time}"
            raise PackageError(msg)
        if self.profile.entry_interface_altitude >= self.profile.parking_altitude:
            msg = (
                "profile.entry_interface_altitude_m must sit below "
                "profile.parking_altitude_m, got "
                f"{self.profile.entry_interface_altitude} and "
                f"{self.profile.parking_altitude}"
            )
            raise PackageError(msg)

    # -- construction ----------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LaunchPackage:
        """Build from a parsed TOML or JSON mapping."""
        schema = data.get("schema")
        if schema != SCHEMA:
            msg = (
                f"expected schema {SCHEMA!r}, got {schema!r}. A package without a "
                "matching schema key is refused rather than guessed at, so that a "
                "format change fails loudly instead of being reinterpreted."
            )
            raise PackageError(msg)

        _check_keys(
            data,
            frozenset(
                {
                    "schema",
                    "metadata",
                    "launch",
                    "aimpoints",
                    "profile",
                    "vehicle",
                    "sensors",
                    "arrival_time_s",
                    "objectives",
                }
            ),
            "root",
        )
        launch = _position(_require(data, "launch", "launch"), "launch")
        raw_aims = _require(data, "aimpoints", "aimpoints")
        if not isinstance(raw_aims, list) or not raw_aims:
            msg = "aimpoints must be a non-empty array of tables"
            raise PackageError(msg)
        aimpoints = tuple(
            _position(entry, f"aimpoints[{i}]") for i, entry in enumerate(raw_aims)
        )

        raw_profile = data.get("profile", {})
        _check_keys(
            raw_profile,
            frozenset(
                {
                    "architecture",
                    "parking_altitude_m",
                    "entry_interface_altitude_m",
                    "burnout_flight_path_angle_deg",
                    "ballistic_entry_angle_deg",
                }
            ),
            "profile",
        )
        angle = raw_profile.get("burnout_flight_path_angle_deg")
        profile = Profile(
            architecture=_choice(
                raw_profile.get("architecture", "ballistic-single"),
                "architecture",
                "profile",
                tuple(NAMED_ARCHITECTURES),
            ),
            parking_altitude=_number(
                raw_profile.get("parking_altitude_m", 150.0e3),
                "parking_altitude_m",
                "profile",
            ),
            entry_interface_altitude=_number(
                raw_profile.get("entry_interface_altitude_m", 100.0e3),
                "entry_interface_altitude_m",
                "profile",
            ),
            burnout_flight_path_angle=(
                None
                if angle is None
                else float(
                    np.deg2rad(
                        _number(angle, "burnout_flight_path_angle_deg", "profile")
                    )
                )
            ),
            ballistic_entry_angle=float(
                np.deg2rad(
                    _number(
                        raw_profile.get("ballistic_entry_angle_deg", 21.8),
                        "ballistic_entry_angle_deg",
                        "profile",
                    )
                )
            ),
        )

        raw_vehicle = data.get("vehicle", {})
        _check_keys(
            raw_vehicle,
            frozenset(
                {
                    "ballistic_coefficient_kg_m2",
                    "lift_to_drag",
                    "imu_grade",
                    "boost_burn_time_s",
                }
            ),
            "vehicle",
        )
        vehicle = Vehicle(
            ballistic_coefficient=_number(
                raw_vehicle.get("ballistic_coefficient_kg_m2", 20.0e3),
                "ballistic_coefficient_kg_m2",
                "vehicle",
            ),
            lift_to_drag=_number(
                raw_vehicle.get("lift_to_drag", 0.0), "lift_to_drag", "vehicle"
            ),
            imu_grade=_choice(
                raw_vehicle.get("imu_grade", "aviation"),
                "imu_grade",
                "vehicle",
                tuple(IMU_GRADES),
            ),
            boost_burn_time=_number(
                raw_vehicle.get("boost_burn_time_s", 300.0),
                "boost_burn_time_s",
                "vehicle",
            ),
        )

        sensors = tuple(
            Sensor(
                name=str(_require(entry, "name", f"sensors[{i}]")),
                position=_position(
                    entry, f"sensors[{i}]", frozenset({"name", "mask_elevation_deg", "note"})
                ),
                mask_elevation=float(
                    np.deg2rad(
                        _number(
                            entry.get("mask_elevation_deg", 5.0),
                            "mask_elevation_deg",
                            f"sensors[{i}]",
                        )
                    )
                ),
                note=str(entry.get("note", "")),
            )
            for i, entry in enumerate(data.get("sensors", []))
        )

        objectives = tuple(data.get("objectives", ["warning_time"]))
        return cls(
            launch=launch,
            aimpoints=aimpoints,
            profile=profile,
            vehicle=vehicle,
            arrival_time=_number(
                data.get("arrival_time_s", 3600.0), "arrival_time_s", "root"
            ),
            sensors=sensors,
            objectives=objectives,
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_toml(cls, text: str) -> LaunchPackage:
        """Parse the authoring format."""
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as error:
            msg = f"not valid TOML: {error}"
            raise PackageError(msg) from error
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, text: str) -> LaunchPackage:
        """Parse the interchange format."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as error:
            msg = f"not valid JSON: {error}"
            raise PackageError(msg) from error
        if not isinstance(data, dict):
            msg = f"a package must be an object, got {type(data).__name__}"
            raise PackageError(msg)
        return cls.from_dict(data)

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The canonical mapping, with every angle back in degrees.

        Ordered so a written file reads top-down the way a reviewer would
        want: what it is, where it starts, what it is aimed at, how it
        flies, what flies it, what is watching, what it is for.
        """
        data: dict[str, Any] = {"schema": SCHEMA}
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        data["launch"] = _position_dict(self.launch)
        data["aimpoints"] = [_position_dict(p) for p in self.aimpoints]
        profile: dict[str, Any] = {
            "architecture": self.profile.architecture,
            "parking_altitude_m": self.profile.parking_altitude,
            "entry_interface_altitude_m": self.profile.entry_interface_altitude,
            "ballistic_entry_angle_deg": float(
                np.rad2deg(self.profile.ballistic_entry_angle)
            ),
        }
        if self.profile.burnout_flight_path_angle is not None:
            profile["burnout_flight_path_angle_deg"] = float(
                np.rad2deg(self.profile.burnout_flight_path_angle)
            )
        data["profile"] = profile
        data["vehicle"] = {
            "ballistic_coefficient_kg_m2": self.vehicle.ballistic_coefficient,
            "lift_to_drag": self.vehicle.lift_to_drag,
            "imu_grade": self.vehicle.imu_grade,
            "boost_burn_time_s": self.vehicle.boost_burn_time,
        }
        data["arrival_time_s"] = self.arrival_time
        if self.sensors:
            data["sensors"] = [
                {
                    "name": s.name,
                    **_position_dict(s.position),
                    "mask_elevation_deg": float(np.rad2deg(s.mask_elevation)),
                    **({"note": s.note} if s.note else {}),
                }
                for s in self.sensors
            ]
        data["objectives"] = list(self.objectives)
        return data

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def to_toml(self) -> str:
        """Write the authoring format.

        Requires ``tomli-w``; JSON is always available and round-trips
        identically, so a missing optional dependency costs formatting
        rather than function.
        """
        try:
            import tomli_w
        except ImportError as error:  # pragma: no cover - optional dependency
            msg = (
                "writing TOML needs the optional 'tomli-w' package; "
                "to_json() is always available and round-trips identically"
            )
            raise PackageError(msg) from error
        return tomli_w.dumps(self.to_dict())

    # -- use -------------------------------------------------------------

    def mission_request(self) -> MissionRequest:
        """The form :func:`passes.systems.budget.evaluate` consumes."""
        return MissionRequest(
            launch_site=self.launch,
            aimpoints=self.aimpoints,
            arrival_time=self.arrival_time,
        )

    def architecture(self):  # type: ignore[no-untyped-def]
        """The named architecture this package selects."""
        return NAMED_ARCHITECTURES[self.profile.architecture]

    def with_profile(self, **changes: Any) -> LaunchPackage:
        """A copy with profile fields replaced — for sweeps.

        Returns a new package rather than mutating, so a sweep cannot
        contaminate the baseline it started from.
        """
        return replace(self, profile=replace(self.profile, **changes))

    def summary(self) -> str:
        """One-screen human summary, for logs and notebook output."""
        name = self.metadata.get("name", "(unnamed)")
        aims = ", ".join(
            p.label or f"{np.rad2deg(p.latitude):.1f},{np.rad2deg(p.longitude):.1f}"
            for p in self.aimpoints
        )
        angle = (
            "minimum-energy"
            if self.profile.burnout_flight_path_angle is None
            else f"{np.rad2deg(self.profile.burnout_flight_path_angle):.2f} deg"
        )
        return (
            f"{name}\n"
            f"  launch       {np.rad2deg(self.launch.latitude):+.2f}, "
            f"{np.rad2deg(self.launch.longitude):+.2f}"
            f"{' (' + self.launch.label + ')' if self.launch.label else ''}\n"
            f"  aimpoints    {aims}\n"
            f"  architecture {self.profile.architecture}\n"
            f"  burnout fpa  {angle}\n"
            f"  vehicle      beta={self.vehicle.ballistic_coefficient:,.0f} kg/m2, "
            f"L/D={self.vehicle.lift_to_drag:g}, IMU={self.vehicle.imu_grade}\n"
            f"  sensors      {len(self.sensors) or 'framework default network'}\n"
            f"  objectives   {', '.join(self.objectives)}"
        )


def load_package(path: str | Path) -> LaunchPackage:
    """Load a package, choosing the parser from the file suffix.

    ``.toml`` and ``.json`` are recognised; anything else is refused rather
    than sniffed, because guessing a format is how a malformed file becomes
    a plausible-looking scenario.
    """
    location = Path(path)
    text = location.read_text(encoding="utf-8")
    suffix = location.suffix.lower()
    if suffix == ".toml":
        return LaunchPackage.from_toml(text)
    if suffix == ".json":
        return LaunchPackage.from_json(text)
    msg = f"unrecognised package suffix {suffix!r}; expected .toml or .json"
    raise PackageError(msg)
