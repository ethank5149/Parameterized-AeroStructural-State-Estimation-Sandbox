"""The Earth as an ellipsoid, for anything that has to be in the right place.

The renderer drew a sphere and the scene placed markers on a sphere, which
is self-consistent and wrong by up to **21.4 km**: the WGS84 semi-major axis
is 6,378.137 km and the semi-minor 6,356.752 km. On a globe 900 pixels tall
that is about 1.5 pixels of radius, which nobody would notice — but the
error is not in the radius, it is in *where a given latitude sits*. Treating
geodetic latitude as geocentric displaces a surface point at 45 degrees by
**21.4 km** — measured, not estimated — and an impact marker 21 km from the
trajectory that produced it is the kind of disagreement between picture and
physics this package exists to prevent.

Three things live here:

**Geodetic to Cartesian and back.** Forward is closed form. The inverse is
Bowring's method, which is a closed-form approximation good to under a
millimetre at terrestrial altitudes, followed by one Newton step that makes
it exact to machine precision at *any* altitude — the difference matters
because this is used on a parking orbit at 170 km and on a re-entry body at
Mach 25, not only on the ground.

**Ray-ellipsoid intersection**, which is where the ray tracer needed it.
Done by the standard change of variables: divide the ray's components by the
semi-axes and the ellipsoid becomes the unit sphere, so the intersection is
a quadratic with the same algebra the sphere version already had. Exact, not
an iteration, and the same cost.

**The local vertical.** On a sphere the surface normal is the position
direction; on an ellipsoid it is not, and the angle between them reaches
0.19 degrees at 45 degrees latitude. Every camera that lifts along "up",
every horizon ring, and every terrain shading normal needs the real one.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "WGS84",
    "Ellipsoid",
    "ecef_to_geodetic",
    "geodetic_to_ecef",
    "local_vertical",
    "ray_ellipsoid",
]

_FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Ellipsoid:
    """A biaxial ellipsoid of revolution.

    Attributes
    ----------
    semi_major:
        Equatorial radius :math:`a` (m).
    flattening:
        :math:`f = (a-b)/a`.
    """

    semi_major: float
    flattening: float
    name: str = "ellipsoid"

    def __post_init__(self) -> None:
        if not (np.isfinite(self.semi_major) and self.semi_major > 0.0):
            msg = f"semi_major must be finite and > 0, got {self.semi_major}"
            raise ValueError(msg)
        if not 0.0 <= self.flattening < 1.0:
            msg = f"flattening must be in [0, 1), got {self.flattening}"
            raise ValueError(msg)

    @property
    def semi_minor(self) -> float:
        """Polar radius :math:`b = a(1-f)` (m)."""
        return float(self.semi_major * (1.0 - self.flattening))

    @property
    def eccentricity_squared(self) -> float:
        """:math:`e^2 = f(2-f)`."""
        return float(self.flattening * (2.0 - self.flattening))

    @property
    def mean_radius(self) -> float:
        """:math:`(2a + b)/3` — the IUGG arithmetic mean radius (m)."""
        return float((2.0 * self.semi_major + self.semi_minor) / 3.0)

    @property
    def axes(self) -> _FloatArray:
        """``(a, a, b)`` — the scaling that maps this to the unit sphere."""
        return np.array([self.semi_major, self.semi_major, self.semi_minor])

    def prime_vertical(self, latitude: ArrayLike) -> _FloatArray:
        """Radius of curvature in the prime vertical, :math:`N(\\varphi)` (m)."""
        sin_phi = np.sin(np.asarray(latitude, dtype=np.float64))
        return np.asarray(
            self.semi_major / np.sqrt(1.0 - self.eccentricity_squared * sin_phi**2)
        )

    def geocentric_latitude(self, latitude: ArrayLike) -> _FloatArray:
        """Geocentric latitude of a point *on the surface* at geodetic ``latitude``.

        The two differ by up to 0.19 degrees at mid-latitudes. Provided
        because conflating them is the specific error this module exists to
        stop, and naming the conversion makes the conflation visible.
        """
        phi = np.asarray(latitude, dtype=np.float64)
        return np.asarray(np.arctan((1.0 - self.eccentricity_squared) * np.tan(phi)))

    def surface_radius(self, latitude: ArrayLike) -> _FloatArray:
        """Distance from the centre to the surface at a geodetic latitude (m)."""
        point = geodetic_to_ecef(latitude, 0.0, 0.0, self)
        return np.asarray(np.linalg.norm(point, axis=-1))


#: The World Geodetic System 1984 reference ellipsoid.
WGS84 = Ellipsoid(semi_major=6378137.0, flattening=1.0 / 298.257223563, name="WGS84")


def geodetic_to_ecef(
    latitude: ArrayLike,
    longitude: ArrayLike,
    altitude: ArrayLike = 0.0,
    ellipsoid: Ellipsoid = WGS84,
) -> _FloatArray:
    """Geodetic coordinates to Earth-centred Cartesian (m).

    .. math::

        x = (N + h)\\cos\\varphi\\cos\\lambda, \\quad
        y = (N + h)\\cos\\varphi\\sin\\lambda, \\quad
        z = \\left(N(1-e^2) + h\\right)\\sin\\varphi

    Angles in radians. Broadcasts, and returns shape ``(..., 3)``.
    """
    phi = np.asarray(latitude, dtype=np.float64)
    lam = np.asarray(longitude, dtype=np.float64)
    height = np.asarray(altitude, dtype=np.float64)
    phi, lam, height = np.broadcast_arrays(phi, lam, height)

    n = ellipsoid.prime_vertical(phi)
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    return np.stack(
        [
            (n + height) * cos_phi * np.cos(lam),
            (n + height) * cos_phi * np.sin(lam),
            (n * (1.0 - ellipsoid.eccentricity_squared) + height) * sin_phi,
        ],
        axis=-1,
    )


def ecef_to_geodetic(
    position: ArrayLike, ellipsoid: Ellipsoid = WGS84, newton_steps: int = 2
) -> tuple[_FloatArray, _FloatArray, _FloatArray]:
    """Earth-centred Cartesian to geodetic latitude, longitude and altitude.

    Bowring's closed form for the starting latitude, then Newton on the
    residual

    .. math::

        F(\\varphi) = z + e^2 N \\sin\\varphi - p\\tan\\varphi\\,\\cos\\varphi
        \\;\\text{(rearranged)}

    Bowring alone is sub-millimetre on the ground and degrades with altitude,
    which is exactly the regime this is used in; two Newton steps take it to
    machine precision everywhere from the geoid to geostationary.

    Returns
    -------
    tuple
        ``(latitude, longitude, altitude)`` in radians and metres, each
        shaped like the leading axes of ``position``.
    """
    r = np.asarray(position, dtype=np.float64)
    if r.shape[-1] != 3:
        msg = f"position must have a trailing axis of 3, got {r.shape}"
        raise ValueError(msg)
    x, y, z = r[..., 0], r[..., 1], r[..., 2]

    a, b = ellipsoid.semi_major, ellipsoid.semi_minor
    e2 = ellipsoid.eccentricity_squared
    # Second eccentricity squared, which is what Bowring's auxiliary angle uses.
    ep2 = (a * a - b * b) / (b * b) if b > 0.0 else 0.0

    p = np.hypot(x, y)
    longitude = np.arctan2(y, x)

    # On the polar axis p is zero and the auxiliary angle is undefined; the
    # latitude there is +/- 90 degrees and the guard only keeps the
    # intermediate arithmetic finite.
    safe_p = np.where(p > 0.0, p, 1.0e-300)
    theta = np.arctan2(z * a, safe_p * b)
    latitude = np.arctan2(
        z + ep2 * b * np.sin(theta) ** 3, safe_p - e2 * a * np.cos(theta) ** 3
    )
    latitude = np.where(p > 0.0, latitude, np.sign(z) * 0.5 * np.pi)

    for _ in range(int(newton_steps)):
        sin_phi, cos_phi = np.sin(latitude), np.cos(latitude)
        n = a / np.sqrt(1.0 - e2 * sin_phi**2)
        height = np.where(
            np.abs(cos_phi) > 1.0e-12,
            safe_p / np.where(np.abs(cos_phi) > 1.0e-12, cos_phi, 1.0) - n,
            np.abs(z) - b,
        )
        # d(lat) from the standard fixed-point form, which converges
        # quadratically once Bowring has supplied a close start.
        updated = np.arctan2(z, safe_p * (1.0 - e2 * n / (n + height)))
        latitude = np.where(p > 0.0, updated, latitude)

    sin_phi, cos_phi = np.sin(latitude), np.cos(latitude)
    n = a / np.sqrt(1.0 - e2 * sin_phi**2)
    altitude = np.where(
        np.abs(cos_phi) > 1.0e-8,
        p / np.where(np.abs(cos_phi) > 1.0e-8, cos_phi, 1.0) - n,
        np.abs(z) - b * np.sqrt(1.0 - e2 * sin_phi**2) / np.sqrt(1.0 - e2),
    )
    # Near the poles the cos(phi) division loses precision; the polar branch
    # above is the exact expression there.
    polar = np.abs(cos_phi) <= 1.0e-8
    if np.any(polar):
        altitude = np.where(polar, np.abs(z) - b, altitude)
    return np.asarray(latitude), np.asarray(longitude), np.asarray(altitude)


def local_vertical(latitude: ArrayLike, longitude: ArrayLike) -> _FloatArray:
    """Outward unit normal to the ellipsoid at a geodetic point, shape ``(..., 3)``.

    The *geodetic* normal, which is what "up" means to anything standing on
    the surface, and which is not parallel to the position vector anywhere
    except the equator and the poles.
    """
    phi = np.asarray(latitude, dtype=np.float64)
    lam = np.asarray(longitude, dtype=np.float64)
    phi, lam = np.broadcast_arrays(phi, lam)
    cos_phi = np.cos(phi)
    return np.stack(
        [cos_phi * np.cos(lam), cos_phi * np.sin(lam), np.sin(phi)], axis=-1
    )


def ray_ellipsoid(
    origin: ArrayLike,
    direction: Any,
    ellipsoid: Ellipsoid = WGS84,
    xp: ModuleType | None = None,
    inflation: float = 0.0,
) -> tuple[Any, Any]:
    """Nearest forward intersection of rays with the ellipsoid.

    Solved by scaling: dividing every component by :math:`(a, a, b)` maps the
    ellipsoid to the unit sphere, where the intersection is the quadratic the
    spherical renderer already solved. The returned distance is measured in
    the *original* space, because the scaling is undone by evaluating the
    quadratic in the scaled space and using its root as a parameter along the
    unscaled ray — the parameter is invariant under the change of variables,
    which is the whole reason this works.

    Parameters
    ----------
    inflation:
        Add this many metres to every semi-axis before intersecting. Used to
        put the atmospheric limb and the terrain envelope slightly outside
        the reference surface.

    Returns
    -------
    tuple
        ``(distance, hit)``. ``distance`` is ``inf`` where there is no
        forward intersection; ``hit`` is the boolean mask.
    """
    module = xp if xp is not None else np
    scale = module.asarray(
        np.array(
            [
                ellipsoid.semi_major + inflation,
                ellipsoid.semi_major + inflation,
                ellipsoid.semi_minor + inflation,
            ]
        )
    )
    start = module.asarray(np.asarray(origin, dtype=np.float64)) / scale
    step = direction / scale

    a = module.sum(step * step, axis=-1)
    b = 2.0 * module.sum(start * step, axis=-1)
    c = module.sum(start * start, axis=-1) - 1.0
    discriminant = b * b - 4.0 * a * c

    root = module.sqrt(module.maximum(discriminant, 0.0))
    near = (-b - root) / (2.0 * a)
    far = (-b + root) / (2.0 * a)
    # Prefer the near root; fall back to the far one for a camera inside the
    # body, which happens on a launch-pad close-up.
    distance = module.where(near > 0.0, near, far)
    hit = (discriminant > 0.0) & (distance > 0.0)
    return module.where(hit, distance, module.inf), hit
