"""Wind as a field the vehicle flies through, not a perturbation bolted on.

Everything aerodynamic in this codebase is a function of the *air-relative*
velocity, and up to now the code has quietly assumed the air is still. It
never is. The number that sizes a launch vehicle's structure is
:math:`q\\alpha` — dynamic pressure times angle of attack — and in a still
atmosphere a gravity-turn ascent flies at :math:`\\alpha \\approx 0` by
construction, so :math:`q\\alpha` is identically zero and the load case does
not exist. Put a 60 m/s jet-stream core at 11 km in front of a vehicle
travelling at 400 m/s and the angle of attack is 8.5 degrees at very nearly
the worst possible moment.

So wind is a first-class field here, in **ENU** components (east, north, up)
at a site, and :func:`relative_velocity` is the one function that converts a
ground-referenced velocity into the one the aerodynamics actually sees.

**Interpolation is monotone (PCHIP), not spline.** A cubic spline through a
measured wind profile overshoots at every shear layer, and an overshoot in
wind is a spurious angle of attack at exactly the altitudes where the real
shear already makes :math:`q\\alpha` large. Monotone interpolation cannot
invent an extremum between two samples; a spline can, and does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import scipy.interpolate
from numpy.typing import ArrayLike, NDArray

from passes.aerodynamics.closure import smoothstep

__all__ = [
    "NoWind",
    "TabulatedWind",
    "WindField",
    "relative_velocity",
    "wind_incidence",
]

_FloatArray = NDArray[np.float64]


class WindField(Protocol):
    """Anything that reports an ENU wind vector at an altitude."""

    name: str

    def velocity(self, altitude: ArrayLike) -> _FloatArray:  # pragma: no cover
        """Wind velocity (m/s) in east-north-up components, shape ``(..., 3)``."""
        ...


@dataclass(frozen=True)
class NoWind:
    """A still atmosphere — the assumption made explicit so it can be seen."""

    name: str = "still air"

    def velocity(self, altitude: ArrayLike) -> _FloatArray:
        z = np.asarray(altitude, dtype=np.float64)
        return np.zeros((*z.shape, 3))

    def shear(self, altitude: ArrayLike) -> _FloatArray:
        z = np.asarray(altitude, dtype=np.float64)
        return np.zeros((*z.shape, 3))


@dataclass(frozen=True)
class TabulatedWind:
    """A measured or modelled wind profile against altitude.

    Attributes
    ----------
    altitude:
        Strictly increasing geometric altitudes (m).
    east, north, up:
        Wind components at those altitudes (m/s). ``up`` defaults to zero:
        reanalysis vertical velocity is two orders of magnitude below the
        horizontal components and is not carried unless supplied.
    ceiling:
        Altitude above which the wind is taken as zero (m). Data ends
        somewhere — 1 hPa is about 48 km — and the profile is faded to zero
        with a :math:`C^2` smoothstep between the top sample and this
        altitude. That fade is **a statement of ignorance, not a model of the
        mesosphere**; it is placed where it cannot matter, because dynamic
        pressure at 60 km on an ascent is under 100 Pa and a 50 m/s error
        there moves nothing.
    """

    altitude: _FloatArray
    east: _FloatArray
    north: _FloatArray
    up: _FloatArray | None = None
    ceiling: float = 60.0e3
    name: str = "tabulated wind"
    _interpolants: tuple[scipy.interpolate.PchipInterpolator, ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        z = np.asarray(self.altitude, dtype=np.float64)
        if z.ndim != 1 or z.size < 2:
            msg = f"altitude must be a 1-D array of at least 2 samples, got shape {z.shape}"
            raise ValueError(msg)
        if np.any(np.diff(z) <= 0.0):
            msg = "altitude must be strictly increasing"
            raise ValueError(msg)
        components = [np.asarray(self.east, dtype=np.float64),
                      np.asarray(self.north, dtype=np.float64)]
        components.append(
            np.zeros_like(z) if self.up is None else np.asarray(self.up, dtype=np.float64)
        )
        for label, values in zip(("east", "north", "up"), components, strict=True):
            if values.shape != z.shape:
                msg = f"{label} must match altitude shape {z.shape}, got {values.shape}"
                raise ValueError(msg)
            if np.any(~np.isfinite(values)):
                msg = f"{label} contains non-finite values"
                raise ValueError(msg)
        if self.ceiling <= float(z[-1]):
            msg = (
                f"ceiling {self.ceiling:g} m must be above the top sample "
                f"{float(z[-1]):g} m — there is no room to fade the profile out"
            )
            raise ValueError(msg)
        object.__setattr__(
            self,
            "_interpolants",
            tuple(
                scipy.interpolate.PchipInterpolator(z, values, extrapolate=False)
                for values in components
            ),
        )

    @property
    def top(self) -> float:
        """Highest altitude with data (m)."""
        return float(np.asarray(self.altitude)[-1])

    @property
    def bottom(self) -> float:
        """Lowest altitude with data (m)."""
        return float(np.asarray(self.altitude)[0])

    def _fade(self, altitude: _FloatArray) -> _FloatArray:
        """Weight that carries the profile from full strength to zero."""
        span = self.ceiling - self.top
        return np.asarray(
            1.0 - smoothstep((altitude - self.top) / span)
        )

    def velocity(self, altitude: ArrayLike) -> _FloatArray:
        """ENU wind (m/s), shape ``(..., 3)``.

        Held constant below the lowest sample — the profile's bottom is a
        reanalysis level near 100 m, and the surface layer below it does not
        change a launch load — and faded to zero above the top.
        """
        z = np.asarray(altitude, dtype=np.float64)
        clamped = np.clip(z, self.bottom, self.top)
        stacked = np.stack([f(clamped) for f in self._interpolants], axis=-1)
        return np.asarray(stacked * self._fade(z)[..., np.newaxis])

    def shear(self, altitude: ArrayLike) -> _FloatArray:
        """:math:`d\\mathbf{V}_w/dz` (1/s), shape ``(..., 3)``.

        Wind shear, not wind, is what a control system has to reject: a
        uniform wind is a trim offset while a shear layer is a transient the
        vehicle flies through in a second or two.
        """
        z = np.asarray(altitude, dtype=np.float64)
        inside = (z >= self.bottom) & (z <= self.top)
        clamped = np.clip(z, self.bottom, self.top)
        derivative = np.stack(
            [f.derivative()(clamped) for f in self._interpolants], axis=-1
        )
        derivative = derivative * inside[..., np.newaxis]
        # Product rule through the fade, so the reported shear is the shear of
        # what `velocity` actually returns rather than of the raw table.
        span = self.ceiling - self.top
        fade = self._fade(z)[..., np.newaxis]
        t = np.clip((z - self.top) / span, 0.0, 1.0)
        fade_slope = -(30.0 * t**2 * (1.0 - t) ** 2) / span
        values = np.stack([f(clamped) for f in self._interpolants], axis=-1)
        return np.asarray(derivative * fade + values * fade_slope[..., np.newaxis])

    def speed(self, altitude: ArrayLike) -> _FloatArray:
        """Horizontal wind speed (m/s)."""
        wind = self.velocity(altitude)
        return np.asarray(np.hypot(wind[..., 0], wind[..., 1]))

    def bearing(self, altitude: ArrayLike) -> _FloatArray:
        """Meteorological wind direction — the bearing the wind comes *from* (rad).

        The convention is worth stating because it is inverted relative to the
        vector: a "westerly" is a wind *from* the west, which is a vector
        pointing east.
        """
        wind = self.velocity(altitude)
        return np.asarray(np.arctan2(-wind[..., 0], -wind[..., 1]) % (2.0 * np.pi))


def relative_velocity(
    ground_velocity: ArrayLike, wind: ArrayLike
) -> _FloatArray:
    """Air-relative velocity: :math:`\\mathbf{V}_\\infty = \\mathbf{V}_g - \\mathbf{V}_w`.

    Both arguments are in the same local frame — ENU at the vehicle. This is a
    one-line function on purpose: it is the single place the ground-relative
    and air-relative velocities are distinguished, and having it named means
    a call site that forgot to subtract the wind is visible as a call site
    that does not use it.
    """
    v = np.asarray(ground_velocity, dtype=np.float64)
    w = np.asarray(wind, dtype=np.float64)
    if v.shape[-1] != 3 or w.shape[-1] != 3:
        msg = f"both velocities must be 3-vectors, got {v.shape} and {w.shape}"
        raise ValueError(msg)
    return np.asarray(v - w)


def wind_incidence(ground_velocity: ArrayLike, wind: ArrayLike) -> _FloatArray:
    """Angle between the ground-relative and air-relative velocity (rad).

    For a vehicle steering on its ground-relative velocity — which a gravity
    turn does — this *is* the angle of attack the wind induces, and
    multiplying it by dynamic pressure gives the :math:`q\\alpha` that sizes
    the airframe.
    """
    v = np.asarray(ground_velocity, dtype=np.float64)
    relative = relative_velocity(v, wind)
    speed_v = np.linalg.norm(v, axis=-1)
    speed_r = np.linalg.norm(relative, axis=-1)
    denominator = speed_v * speed_r
    cosine = np.sum(v * relative, axis=-1) / np.where(denominator > 0.0, denominator, 1.0)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    # A vehicle at rest has no incidence to speak of, and 0/0 would otherwise
    # come back as a right angle.
    return np.asarray(np.where(denominator > 0.0, angle, 0.0))
