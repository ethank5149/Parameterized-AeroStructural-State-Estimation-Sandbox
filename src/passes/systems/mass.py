"""Vehicle mass distribution: where the mass is, and where it goes.

The flight simulator carried mass as a single scalar and its drag area as a
constant. Neither survives contact with a staged liquid vehicle: an RS-28
class ICBM leaves the pad at 208 t and stages at roughly a tenth of that,
and its centre of mass travels metres down the body as the tanks drain. This
module supplies the distribution, built from the outer mould line for shape
and from published aggregates for magnitude.

What is anchored to what
------------------------

The distinction matters more than any individual number, so it is structural
here rather than a footnote:

* **Totals are OSINT.** Gross 208.1 t, propellant 178-190 t, throw weight
  ~10 t, length 35.3-35.5 m. These are consistently reported and are the
  only Sarmat-specific masses in the open domain.
* **The stage split is inferred** from the R-36M2/Voevoda, the documented
  predecessor sharing the Makeyev tandem architecture: stage 1 dry
  13-15 t, stage 2 dry 3-5 t, and a propellant split near 4:1. No
  Sarmat-specific stage masses have entered open reporting, and this module
  does not pretend otherwise — :attr:`Stage.provenance` records, per
  number, whether it was reported or inferred.
* **The axial distribution is geometry.** Structure is spread along the body
  in proportion to local surface area, which is what a monocoque or
  orthogrid shell does; propellant fills its tank from the aft end, which is
  where it settles under thrust.

A geometric inconsistency, stated
---------------------------------

The bundled mesh is **not dimensionally faithful** to the published vehicle.
Its length is right to 1.1 % but its barrel diameter is 3.469 m against a
published 3.0 m — 15.6 % oversized, L/D 10.1 against 11.8 — and the
separation rings divide it into stage-1 and stage-2 tank volumes in the
ratio 1.20:1 where a tandem vehicle burning the same propellants in both
stages needs about 4:1.

So the mesh is a representative external model, and the mass split is
**not** taken from its volumes. Tank extents are solved to hold the stated
propellant at the stated density, and :meth:`VehicleMassModel.audit` reports
how much of each geometric bay that fills. A fill fraction outside roughly
0.6-0.95 is the model telling you the geometry and the masses disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "MassState",
    "Stage",
    "VehicleMassModel",
    "sarmat_mass_model",
]

_FloatArray = NDArray[np.float64]

#: Bulk density of a storable UDMH / N2O4 propellant load (kg/m3), at the
#: oxidiser-rich mixture ratio these engines run. N2O4 is 1,443 and UDMH
#: 791; the bulk figure follows from the ratio, not from either alone.
STORABLE_BULK_DENSITY = 1190.0


@dataclass(frozen=True)
class Stage:
    """One separable element, with its mass and where that mass sits.

    Stations are **metres aft of the nose tip**, increasing rearward, which
    is the convention the mesh's body axes use once the nose is at the
    origin.

    Attributes
    ----------
    name:
        What this element is.
    forward, aft:
        Extent along the body (m aft of the tip).
    dry_mass:
        Structure, engines and residuals (kg).
    propellant_mass:
        Usable propellant at ignition (kg).
    tank_forward, tank_aft:
        Extent of the propellant volume. Defaults to the whole element.
    provenance:
        Per-field note recording whether each number was reported or
        inferred, so a downstream result can say which of its inputs were
        measurements.
    """

    name: str
    forward: float
    aft: float
    dry_mass: float
    propellant_mass: float = 0.0
    tank_forward: float | None = None
    tank_aft: float | None = None
    provenance: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.aft > self.forward:
            msg = (
                f"stage {self.name!r} must extend aft: got forward={self.forward}, "
                f"aft={self.aft} (stations are metres aft of the nose)"
            )
            raise ValueError(msg)
        for label in ("dry_mass", "propellant_mass"):
            value = float(getattr(self, label))
            if not (np.isfinite(value) and value >= 0.0):
                msg = f"{self.name!r} {label} must be finite and >= 0, got {value}"
                raise ValueError(msg)
        if self.dry_mass <= 0.0:
            msg = f"stage {self.name!r} has no dry mass"
            raise ValueError(msg)

    @property
    def length(self) -> float:
        return float(self.aft - self.forward)

    @property
    def gross_mass(self) -> float:
        return float(self.dry_mass + self.propellant_mass)

    @property
    def tank(self) -> tuple[float, float]:
        forward = self.forward if self.tank_forward is None else self.tank_forward
        aft = self.aft if self.tank_aft is None else self.tank_aft
        return float(forward), float(aft)


@dataclass(frozen=True)
class MassState:
    """The vehicle's mass properties at one instant."""

    mass: float
    centre_of_mass: float
    """Metres aft of the nose tip."""
    roll_inertia: float
    """About the long axis (kg m²)."""
    pitch_inertia: float
    """About a transverse axis through the centre of mass (kg m²)."""
    stages_present: tuple[str, ...]
    propellant_remaining: dict[str, float]

    def inertia_tensor(self) -> _FloatArray:
        """Body-axis tensor about the centre of mass, nose along +x.

        Diagonal by symmetry: this is a body of revolution, so the products
        of inertia vanish and the two transverse moments are equal.
        """
        return np.diag(
            [self.roll_inertia, self.pitch_inertia, self.pitch_inertia]
        ).astype(np.float64)


class VehicleMassModel:
    """Stages plus an area profile, giving mass properties at any burn state.

    Parameters
    ----------
    stages:
        Elements, forward to aft.
    stations, radii:
        Outer radius against station (m aft of the tip), from
        :meth:`~passes.geometry.mesh.VehicleMesh.station_profile`. Used to
        distribute structure by local surface area and to convert propellant
        volume into an axial extent.
    """

    def __init__(
        self,
        stages: tuple[Stage, ...] | list[Stage],
        stations: _FloatArray,
        radii: _FloatArray,
    ) -> None:
        if not stages:
            msg = "a mass model needs at least one stage"
            raise ValueError(msg)
        order = np.argsort(np.asarray(stations, dtype=np.float64))
        self.stations = np.asarray(stations, dtype=np.float64)[order]
        self.radii = np.asarray(radii, dtype=np.float64)[order]
        if self.stations.size < 2:
            msg = "need at least two stations to describe a body"
            raise ValueError(msg)
        self.stages = tuple(stages)
        names = [s.name for s in self.stages]
        if len(set(names)) != len(names):
            msg = f"stage names must be unique, got {names}"
            raise ValueError(msg)

    # -- geometry helpers -------------------------------------------------

    def _radius(self, station: _FloatArray) -> _FloatArray:
        return np.asarray(np.interp(station, self.stations, self.radii))

    def _sample(self, forward: float, aft: float, count: int = 400) -> _FloatArray:
        return np.linspace(float(forward), float(aft), int(count))

    def volume(self, forward: float, aft: float) -> float:
        """Enclosed volume of a body-of-revolution slice (m³)."""
        x = self._sample(forward, aft)
        return float(np.trapezoid(np.pi * self._radius(x) ** 2, x))

    def _shell_moments(self, forward: float, aft: float) -> tuple[float, float, float]:
        """Unit-mass first and second moments of a *surface* slice.

        Structure is spread in proportion to local circumference, which is
        what a shell of roughly constant thickness weighs per unit length.
        Returns (centroid, roll integral, pitch integral) per unit mass.
        """
        x = self._sample(forward, aft)
        r = self._radius(x)
        weight = 2.0 * np.pi * r
        total = float(np.trapezoid(weight, x))
        if total <= 0.0:  # pragma: no cover - a zero-radius slice
            middle = 0.5 * (forward + aft)
            return middle, 0.0, 0.0
        centroid = float(np.trapezoid(weight * x, x) / total)
        # Thin shell of radius r: roll inertia per unit mass is r^2.
        roll = float(np.trapezoid(weight * r**2, x) / total)
        pitch = float(np.trapezoid(weight * (0.5 * r**2 + (x - centroid) ** 2), x) / total)
        return centroid, roll, pitch

    def _solid_moments(self, forward: float, aft: float) -> tuple[float, float, float]:
        """Unit-mass moments of a *filled* slice — propellant."""
        x = self._sample(forward, aft)
        r = self._radius(x)
        weight = np.pi * r**2
        total = float(np.trapezoid(weight, x))
        if total <= 0.0:
            middle = 0.5 * (forward + aft)
            return middle, 0.0, 0.0
        centroid = float(np.trapezoid(weight * x, x) / total)
        roll = float(np.trapezoid(weight * 0.5 * r**2, x) / total)
        pitch = float(
            np.trapezoid(weight * (0.25 * r**2 + (x - centroid) ** 2), x) / total
        )
        return centroid, roll, pitch

    def propellant_front(self, stage: Stage, remaining: float, density: float) -> float:
        """Station of the propellant surface for a given remaining mass.

        Liquid settles **aft** under thrust, so the tank empties from the
        front: the remaining propellant occupies the rear of the tank and
        its surface moves rearward as the burn proceeds. Modelling it as a
        uniformly thinning column instead would hold the propellant centroid
        fixed and lose the centre-of-mass travel entirely, which for this
        vehicle is metres.
        """
        forward, aft = stage.tank
        if remaining <= 0.0:
            return aft
        target = float(remaining) / float(density)
        # Bisect on the surface station: volume aft of it equals the target.
        low, high = forward, aft
        for _ in range(60):
            middle = 0.5 * (low + high)
            if self.volume(middle, aft) > target:
                low = middle
            else:
                high = middle
        return 0.5 * (low + high)

    # -- the state --------------------------------------------------------

    def state(
        self,
        burned: dict[str, float] | None = None,
        jettisoned: tuple[str, ...] | list[str] = (),
        density: float = STORABLE_BULK_DENSITY,
    ) -> MassState:
        """Mass, centre of mass and inertia for a given burn and staging state.

        Parameters
        ----------
        burned:
            Propellant consumed per stage (kg). Missing stages are unburnt.
        jettisoned:
            Stages already separated and no longer part of the vehicle.
        density:
            Bulk propellant density (kg/m³).
        """
        spent = dict(burned or {})
        gone = set(jettisoned)
        for name in list(spent) + list(gone):
            if name not in {s.name for s in self.stages}:
                known = ", ".join(s.name for s in self.stages)
                msg = f"unknown stage {name!r}; this vehicle has: {known}"
                raise ValueError(msg)

        masses: list[float] = []
        centroids: list[float] = []
        rolls: list[float] = []
        pitches: list[float] = []
        present: list[str] = []
        left: dict[str, float] = {}

        for stage in self.stages:
            if stage.name in gone:
                continue
            present.append(stage.name)
            used = float(np.clip(spent.get(stage.name, 0.0), 0.0, stage.propellant_mass))
            remaining = stage.propellant_mass - used
            left[stage.name] = remaining

            centroid, roll, pitch = self._shell_moments(stage.forward, stage.aft)
            masses.append(stage.dry_mass)
            centroids.append(centroid)
            rolls.append(stage.dry_mass * roll)
            pitches.append(stage.dry_mass * pitch)

            if remaining > 0.0:
                _, tank_aft = stage.tank
                front = self.propellant_front(stage, remaining, density)
                centroid, roll, pitch = self._solid_moments(front, tank_aft)
                masses.append(remaining)
                centroids.append(centroid)
                rolls.append(remaining * roll)
                pitches.append(remaining * pitch)

        if not masses:
            msg = "every stage has been jettisoned; there is no vehicle left"
            raise ValueError(msg)

        mass = float(np.sum(masses))
        centre = float(np.sum(np.asarray(masses) * np.asarray(centroids)) / mass)
        roll_total = float(np.sum(rolls))
        # Parallel axis to the vehicle centre of mass.
        pitch_total = float(
            np.sum(
                np.asarray(pitches)
                + np.asarray(masses) * (np.asarray(centroids) - centre) ** 2
            )
        )
        return MassState(
            mass=mass,
            centre_of_mass=centre,
            roll_inertia=roll_total,
            pitch_inertia=pitch_total,
            stages_present=tuple(present),
            propellant_remaining=left,
        )

    # -- diagnostics ------------------------------------------------------

    @property
    def gross_mass(self) -> float:
        return float(sum(s.gross_mass for s in self.stages))

    def audit(self, density: float = STORABLE_BULK_DENSITY) -> list[dict[str, float | str | bool]]:
        """Per stage: does the stated propellant fit the geometry?

        The check that catches a mesh and a mass table describing different
        vehicles. A fill fraction far outside 0.6-0.95 means the tank extent
        needed to hold the stated propellant does not match the bay the
        separation rings define.
        """
        rows: list[dict[str, float | str | bool]] = []
        for stage in self.stages:
            forward, aft = stage.tank
            bay = self.volume(forward, aft)
            needed = stage.propellant_mass / float(density)
            fill = (needed / bay) if bay > 0.0 else float("inf")
            rows.append(
                {
                    "stage": stage.name,
                    "bay_volume": bay,
                    "propellant_volume": needed,
                    "fill_fraction": fill,
                    "gross_mass": stage.gross_mass,
                    "feasible": bool(fill <= 1.0),
                }
            )
        return rows

    def check(self, density: float = STORABLE_BULK_DENSITY) -> None:
        """Raise if any stage's propellant cannot fit its bay.

        Not advisory. A fill fraction above one is not a modelling
        preference, it is a stage carrying more propellant than it has room
        for, and every mass property computed from it — centre of mass,
        inertia, the whole burn history — is then fiction.

        This is the check that decided where the bundled vehicle's
        separation planes are. Reading the rings as
        payload/bus, bus/stage-2, stage-2/stage-1 puts 142 t of propellant
        into a 72 m3 bay: **fill 1.66, impossible**. Shifting the reading
        one ring forward — so the ring 23.4 m aft is a stage-1 *intertank*
        frame rather than a separation plane, which is exactly what a
        tandem oxidiser/fuel stage carries — gives 0.90 and 0.73, both of
        which are ordinary booster tank fill fractions. The geometry chose
        between two interpretations that documentation could not.
        """
        bad = [row for row in self.audit(density) if not row["feasible"]]
        if not bad:
            return
        detail = "; ".join(
            f"{row['stage']} needs {float(row['propellant_volume']):.1f} m3 "
            f"in a {float(row['bay_volume']):.1f} m3 bay "
            f"(fill {float(row['fill_fraction']):.2f})"
            for row in bad
        )
        msg = f"propellant does not fit the geometry: {detail}"
        raise ValueError(msg)


def sarmat_mass_model(
    stations: _FloatArray,
    radii: _FloatArray,
    gross_mass: float = 208.1e3,
    propellant_mass: float = 178.0e3,
    throw_weight: float = 10.0e3,
    separations: tuple[float, float, float] = (6.810, 13.412, 23.091),
) -> VehicleMassModel:
    """An RS-28 class mass model, anchored to open reporting.

    Parameters
    ----------
    stations, radii:
        Outer mould line, metres aft of the tip.
    gross_mass, propellant_mass, throw_weight:
        The three OSINT aggregates. Propellant is quoted between 178 t
        (RussianSpaceWeb via FRS) and 190 t (a Plesetsk environmental
        assessment); the lower figure is the default and the spread is
        worth carrying through any result that depends on it.
    separations:
        Stations of the three separation planes, forward to aft: bus from
        payload, stage 2 from bus, stage 1 from stage 2. Defaults are the
        raised rings measured on the bundled mesh. The fourth ring, aft of
        these, is an engine-bay structural frame and is **not** a separation
        plane.

    Notes
    -----
    The split between stages is *inferred*, not reported. Open sources give
    no Sarmat stage masses at all. These follow the R-36M2, which shares the
    architecture: dry masses in a 3.5:1 ratio between the two boosters and
    propellant near 4:1, with the bus taking the balance of the throw
    weight. Every such number is tagged in :attr:`Stage.provenance`.
    """
    if not (0.0 < throw_weight < gross_mass):
        msg = f"throw weight {throw_weight} must lie in (0, {gross_mass})"
        raise ValueError(msg)
    booster_gross = gross_mass - throw_weight
    booster_dry = booster_gross - propellant_mass
    if booster_dry <= 0.0:
        msg = (
            f"propellant {propellant_mass / 1e3:.1f} t and throw weight "
            f"{throw_weight / 1e3:.1f} t exceed the gross {gross_mass / 1e3:.1f} t; "
            "the boosters would have negative structure"
        )
        raise ValueError(msg)

    bus_cut, stage2_cut, stage1_cut = separations
    aft = float(np.max(stations))

    # R-36M2 proportions: dry 3.5:1, propellant 4:1 between the boosters.
    dry_1 = booster_dry * 3.5 / 4.5
    dry_2 = booster_dry / 4.5
    prop_1 = propellant_mass * 4.0 / 5.0
    prop_2 = propellant_mass / 5.0
    # The throw weight is the bus and what it carries; the bus's own
    # propellant is a small fraction of it.
    bus_propellant = 0.18 * throw_weight
    bus_dry = 0.32 * throw_weight
    payload = throw_weight - bus_propellant - bus_dry

    reported = "OSINT aggregate (208.1 t gross, 178-190 t propellant, ~10 t throw)"
    inferred = "inferred from R-36M2 proportions; no Sarmat stage data is public"
    return VehicleMassModel(
        stages=(
            Stage(
                name="payload",
                forward=0.0, aft=bus_cut,
                dry_mass=payload,
                provenance={"dry_mass": reported},
            ),
            Stage(
                name="bus",
                forward=bus_cut, aft=stage2_cut,
                dry_mass=bus_dry, propellant_mass=bus_propellant,
                provenance={"dry_mass": inferred, "propellant_mass": inferred},
            ),
            Stage(
                name="stage2",
                forward=stage2_cut, aft=stage1_cut,
                dry_mass=dry_2, propellant_mass=prop_2,
                provenance={"dry_mass": inferred, "propellant_mass": inferred},
            ),
            Stage(
                name="stage1",
                forward=stage1_cut, aft=aft,
                dry_mass=dry_1, propellant_mass=prop_1,
                provenance={"dry_mass": inferred, "propellant_mass": inferred},
            ),
        ),
        stations=stations,
        radii=radii,
    )
