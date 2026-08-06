"""Allen-Eggers closed-form ballistic entry.

Implements the classical steep-entry solution of

* H. J. Allen and A. J. Eggers Jr., "A Study of the Motion and Aerodynamic
  Heating of Missiles Entering the Earth's Atmosphere at High Supersonic
  Speeds," NACA TN 4047, 1957,

as presented and exercised by P. Gallais, *Atmospheric Re-Entry Vehicle
Mechanics* (Springer, 2007), which cites it as [ALL] and works it as his
Exercise 24.

This module exists to replace a constant. The mission budget charged a
ballistic entry a flat 300 km of ground range between entry interface and
impact -- the only leg in the whole budget whose range was neither computed
nor sourced, justified as "roughly geometry-fixed for a steep entry", which
is true enough to be plausible and not true enough to be derived.

It is derived here, and the justification turns out to be *nearly* right for
one particular entry angle and wrong elsewhere: the range is
:math:`h_E/\\tan\\gamma_E`, so 300 km corresponds to a 21.8 degree entry
from 120 km, and a 45 degree entry covers 120 km instead. Holding it fixed
understates steep entries by a factor of 2.5.

What the solution assumes, and why the assumptions are self-consistent
----------------------------------------------------------------------

Allen and Eggers take a steep, non-lifting, drag-dominated entry through an
exponential atmosphere and neglect gravity against drag. Under those
conditions the flight-path angle stays essentially constant, which is what
makes the problem integrable: with :math:`\\gamma` fixed, altitude is a
proxy for time and the velocity equation separates.

.. math::

    \\frac{\\mathrm{d}V}{\\mathrm{d}h}
      = \\frac{\\rho V}{2\\beta\\sin\\gamma}
    \\;\\Longrightarrow\\;
    V(h) = V_E \\exp\\!\\left[-\\frac{H\\,[\\rho(h) - \\rho(h_E)]}
                                    {2\\beta\\sin\\gamma}\\right]

with :math:`\\beta = m/(C_D A)` the ballistic coefficient, :math:`H` the
density scale height and :math:`\\gamma` the (positive) flight-path angle
below horizontal.

Neglecting gravity is the assumption that fails first, and it fails at
shallow angles -- gravity turns a shallow trajectory, so :math:`\\gamma`
does not stay constant and the range is no longer
:math:`h_E/\\tan\\gamma_E`. :func:`ballistic_entry_range` therefore refuses
below a stated angle rather than returning a number that looks fine.

The result that does not depend on the vehicle
----------------------------------------------

The peak deceleration is

.. math:: a_\\max = \\frac{V_E^2 \\sin\\gamma}{2 e H}

which contains **no vehicle property at all**. Ballistic coefficient sets
*where* the peak happens, not how hard it is: a heavier or sleeker vehicle
penetrates deeper before decelerating, into denser air, and the two effects
cancel exactly. This is the sharpest check available on the whole
derivation, and it is the one this module verifies against numerical
integration of the unapproximated equations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "EXPONENTIAL_ATMOSPHERE_EARTH",
    "MINIMUM_BALLISTIC_ENTRY_ANGLE",
    "BallisticEntry",
    "ExponentialAtmosphere",
    "allen_eggers_velocity",
    "ballistic_entry_range",
    "peak_deceleration",
    "peak_deceleration_altitude",
]

_FloatArray = NDArray[np.float64]

#: Below this flight-path angle the constant-gamma assumption underpinning
#: Allen-Eggers stops paying for itself: gravity turns the trajectory faster
#: than drag can hold it, and the geometric range formula overstates the
#: true range by more than half.
#:
#: The floor is set from measurement, not judgement. Against numerical
#: integration of the full point-mass equations, the range error at 15
#: degrees is 18 % for a heavy reentry vehicle and 51 % for a light capsule;
#: at 10 degrees it is 36 % and 73 %. Fifteen degrees is where the useful
#: case is still defensible and the marginal one is already flagged by the
#: error table in :func:`ballistic_entry_range`.
#:
#: Gallais's own worked cases sit at 20 and 60 degrees, with 13.6 degrees
#: for Mars Pathfinder -- a low-beta capsule, and precisely the combination
#: this floor excludes.
MINIMUM_BALLISTIC_ENTRY_ANGLE = np.deg2rad(15.0)


@dataclass(frozen=True)
class ExponentialAtmosphere:
    """Isothermal exponential density model :math:`\\rho = \\rho_s e^{-h/H}`.

    Attributes
    ----------
    sea_level_density:
        :math:`\\rho_s` (kg/m^3).
    scale_height:
        :math:`H` (m).
    """

    sea_level_density: float
    scale_height: float

    def __post_init__(self) -> None:
        if not (np.isfinite(self.sea_level_density) and self.sea_level_density > 0.0):
            msg = f"sea_level_density must be finite and > 0, got {self.sea_level_density}"
            raise ValueError(msg)
        if not (np.isfinite(self.scale_height) and self.scale_height > 0.0):
            msg = f"scale_height must be finite and > 0, got {self.scale_height}"
            raise ValueError(msg)

    def density(self, altitude: float | _FloatArray) -> _FloatArray:
        """Density (kg/m^3) at geometric altitude (m)."""
        h = np.asarray(altitude, dtype=np.float64)
        return np.asarray(self.sea_level_density * np.exp(-h / self.scale_height))


#: The model Gallais uses throughout: rho_s = 1.39 kg/m^3, H = 7000 m.
#:
#: The sea-level density is deliberately *not* 1.225. An exponential fit
#: matched to the density at altitudes that matter for entry heating
#: overshoots at the ground, and 1.39 is the extrapolated intercept of that
#: fit rather than a measurement. Substituting 1.225 to make the ground
#: value "right" would make the fit worse everywhere the vehicle actually
#: decelerates.
EXPONENTIAL_ATMOSPHERE_EARTH = ExponentialAtmosphere(1.39, 7000.0)


@dataclass(frozen=True)
class BallisticEntry:
    """A steep non-lifting entry, in the Allen-Eggers approximation.

    Attributes
    ----------
    entry_velocity:
        :math:`V_E` (m/s) at the entry interface.
    entry_angle:
        :math:`\\gamma_E` (rad), positive below the horizontal.
    entry_altitude:
        :math:`h_E` (m), the entry interface.
    ballistic_coefficient:
        :math:`\\beta = m/(C_D A)` (kg/m^2).
    atmosphere:
        Density model.
    """

    entry_velocity: float
    entry_angle: float
    entry_altitude: float = 120.0e3
    ballistic_coefficient: float = 7500.0
    atmosphere: ExponentialAtmosphere = EXPONENTIAL_ATMOSPHERE_EARTH

    def __post_init__(self) -> None:
        if not (np.isfinite(self.entry_velocity) and self.entry_velocity > 0.0):
            msg = f"entry_velocity must be finite and > 0, got {self.entry_velocity}"
            raise ValueError(msg)
        if not (np.isfinite(self.entry_altitude) and self.entry_altitude > 0.0):
            msg = f"entry_altitude must be finite and > 0, got {self.entry_altitude}"
            raise ValueError(msg)
        if not (np.isfinite(self.ballistic_coefficient) and self.ballistic_coefficient > 0.0):
            msg = (
                "ballistic_coefficient must be finite and > 0, got "
                f"{self.ballistic_coefficient}"
            )
            raise ValueError(msg)
        if not np.isfinite(self.entry_angle) or not (
            MINIMUM_BALLISTIC_ENTRY_ANGLE <= self.entry_angle < 0.5 * np.pi
        ):
            msg = (
                f"entry_angle must lie in [{np.rad2deg(MINIMUM_BALLISTIC_ENTRY_ANGLE):.0f}, "
                f"90) degrees, got {np.rad2deg(self.entry_angle):.2f}. Allen-Eggers "
                "assumes a constant flight-path angle, which requires drag to dominate "
                "gravity; a shallow entry violates that and the closed form is wrong "
                "rather than approximate."
            )
            raise ValueError(msg)

    @property
    def ground_range(self) -> float:
        """Entry interface to impact, along the ground (m)."""
        return float(self.entry_altitude / np.tan(self.entry_angle))

    @property
    def peak_deceleration(self) -> float:
        """Peak deceleration (m/s^2). Independent of ballistic coefficient."""
        return peak_deceleration(
            self.entry_velocity, self.entry_angle, self.atmosphere.scale_height
        )

    @property
    def peak_deceleration_altitude(self) -> float:
        """Altitude of peak deceleration (m). This *is* set by the vehicle."""
        return peak_deceleration_altitude(
            self.entry_angle, self.ballistic_coefficient, self.atmosphere
        )

    def velocity(self, altitude: float | _FloatArray) -> _FloatArray:
        """Velocity (m/s) at altitude, from :func:`allen_eggers_velocity`."""
        return allen_eggers_velocity(
            altitude,
            self.entry_velocity,
            self.entry_angle,
            self.ballistic_coefficient,
            self.entry_altitude,
            self.atmosphere,
        )

    @property
    def terminal_velocity(self) -> float:
        """Sea-level terminal velocity :math:`\\sqrt{2\\beta g/\\rho_s}` (m/s).

        Not part of Allen-Eggers -- it is here to say when Allen-Eggers has
        stopped applying. The closed form neglects gravity, which is sound
        only while drag dominates it, i.e. while the vehicle is travelling
        far faster than terminal. Once it is not, the real vehicle settles at
        terminal velocity while the closed form keeps decelerating toward
        zero.
        """
        return float(
            np.sqrt(2.0 * self.ballistic_coefficient * 9.80665 / self.atmosphere.sea_level_density)
        )

    @property
    def impact_velocity(self) -> float:
        """Velocity at sea level (m/s), where Allen-Eggers is applicable.

        The fraction of entry velocity surviving to the ground separates a
        warhead-class ballistic coefficient from a capsule: at
        :math:`\\beta = 7500` kg/m^2 about 27 % survives -- 1.8 km/s, still
        hypersonic -- while at :math:`\\beta = 500` kg/m^2 the closed form
        returns essentially zero.

        **That zero is not a result.** It is the gravity-free approximation
        running past its own validity: a real vehicle would settle at
        :attr:`terminal_velocity`, around 30 m/s for a light capsule.
        Compare the two before quoting this, and use
        :attr:`allen_eggers_applicable_at_impact` to check.
        """
        return float(self.velocity(0.0))

    @property
    def allen_eggers_applicable_at_impact(self) -> bool:
        """Whether :attr:`impact_velocity` can be believed.

        True when the closed-form impact velocity is comfortably
        supersonic relative to terminal -- taken here as five times
        terminal, below which the neglected gravity term is no longer a
        small correction to drag.
        """
        return self.impact_velocity > 5.0 * self.terminal_velocity


def allen_eggers_velocity(
    altitude: float | _FloatArray,
    entry_velocity: float,
    entry_angle: float,
    ballistic_coefficient: float,
    entry_altitude: float = 120.0e3,
    atmosphere: ExponentialAtmosphere = EXPONENTIAL_ATMOSPHERE_EARTH,
) -> _FloatArray:
    """Velocity against altitude, Allen-Eggers closed form.

    .. math::

        V(h) = V_E \\exp\\!\\left[-\\frac{H[\\rho(h)-\\rho(h_E)]}
                                       {2\\beta\\sin\\gamma}\\right]

    The entry-interface density is retained rather than set to zero. It is
    small, but dropping it makes :math:`V(h_E) \\ne V_E`, which turns an
    exact boundary condition into an approximate one for no benefit.

    Parameters
    ----------
    altitude:
        Geometric altitude (m), scalar or array.
    entry_velocity, entry_angle, entry_altitude:
        :math:`V_E` (m/s), :math:`\\gamma_E` (rad, positive downward),
        :math:`h_E` (m).
    ballistic_coefficient:
        :math:`\\beta = m/(C_D A)` (kg/m^2).
    atmosphere:
        Density model.

    Returns
    -------
    numpy.ndarray
        Velocity (m/s).
    """
    if not (np.isfinite(entry_angle) and 0.0 < entry_angle < 0.5 * np.pi):
        msg = f"entry_angle must lie in (0, 90) degrees, got {np.rad2deg(entry_angle):.2f}"
        raise ValueError(msg)
    if not (np.isfinite(ballistic_coefficient) and ballistic_coefficient > 0.0):
        msg = f"ballistic_coefficient must be finite and > 0, got {ballistic_coefficient}"
        raise ValueError(msg)
    if not (np.isfinite(entry_velocity) and entry_velocity > 0.0):
        msg = f"entry_velocity must be finite and > 0, got {entry_velocity}"
        raise ValueError(msg)

    h = np.asarray(altitude, dtype=np.float64)
    delta_rho = atmosphere.density(h) - atmosphere.density(entry_altitude)
    exponent = -atmosphere.scale_height * delta_rho / (
        2.0 * ballistic_coefficient * np.sin(entry_angle)
    )
    return np.asarray(entry_velocity * np.exp(exponent))


def peak_deceleration(
    entry_velocity: float,
    entry_angle: float,
    scale_height: float = EXPONENTIAL_ATMOSPHERE_EARTH.scale_height,
) -> float:
    """Peak deceleration :math:`V_E^2\\sin\\gamma/(2eH)` (m/s^2).

    Notable for what is *absent*: no ballistic coefficient, no mass, no
    area, no drag coefficient. A denser vehicle penetrates deeper before it
    decelerates, and the extra density it meets exactly offsets its greater
    resistance to being slowed.

    A consequence worth stating plainly, because it inverts an intuition:
    the crew-survivability limit of a ballistic entry cannot be improved by
    changing the vehicle. It is set by how fast and how steeply you arrive.
    """
    if not (np.isfinite(entry_velocity) and entry_velocity > 0.0):
        msg = f"entry_velocity must be finite and > 0, got {entry_velocity}"
        raise ValueError(msg)
    if not (np.isfinite(entry_angle) and 0.0 < entry_angle < 0.5 * np.pi):
        msg = f"entry_angle must lie in (0, 90) degrees, got {np.rad2deg(entry_angle):.2f}"
        raise ValueError(msg)
    if not (np.isfinite(scale_height) and scale_height > 0.0):
        msg = f"scale_height must be finite and > 0, got {scale_height}"
        raise ValueError(msg)
    return float(entry_velocity**2 * np.sin(entry_angle) / (2.0 * np.e * scale_height))


def peak_deceleration_altitude(
    entry_angle: float,
    ballistic_coefficient: float,
    atmosphere: ExponentialAtmosphere = EXPONENTIAL_ATMOSPHERE_EARTH,
) -> float:
    """Altitude (m) at which deceleration peaks.

    Differentiating :math:`a = \\rho V^2/2\\beta` with :math:`V(\\rho)` from
    :func:`allen_eggers_velocity` puts the peak at
    :math:`\\rho_{\\mathrm{crit}} = \\beta\\sin\\gamma/H`, i.e. at

    .. math:: h = H \\ln\\!\\frac{\\rho_s H}{\\beta\\sin\\gamma}

    This is the half of the problem the vehicle *does* control. A high
    ballistic coefficient drives the peak lower -- and for a heavy, steep
    entry the formula can return a negative altitude, meaning the vehicle
    reaches the ground still accelerating into denser air, never having
    peaked. That is returned as-is rather than clamped: a negative answer is
    a meaningful statement about the trajectory, and clamping it to zero
    would hide the case where the ground arrives first. It is also the case
    in which :func:`peak_deceleration` stops applying, since the maximum is
    then an endpoint rather than a stationary point.

    Notes
    -----
    This carried a spurious factor of two (:math:`2\\beta\\sin\\gamma/H`)
    when first written, which placed every peak exactly :math:`H\\ln 2`
    = 4.85 km too low. It was caught by the numerical cross-check in the
    tests, not by inspection -- the error is invisible in
    :func:`peak_deceleration`, whose value is unaffected because
    :math:`a_\\max` is stationary there by construction.
    """
    if not (np.isfinite(entry_angle) and 0.0 < entry_angle < 0.5 * np.pi):
        msg = f"entry_angle must lie in (0, 90) degrees, got {np.rad2deg(entry_angle):.2f}"
        raise ValueError(msg)
    if not (np.isfinite(ballistic_coefficient) and ballistic_coefficient > 0.0):
        msg = f"ballistic_coefficient must be finite and > 0, got {ballistic_coefficient}"
        raise ValueError(msg)
    critical_density = ballistic_coefficient * np.sin(entry_angle) / atmosphere.scale_height
    return float(
        atmosphere.scale_height * np.log(atmosphere.sea_level_density / critical_density)
    )


def ballistic_entry_range(
    entry_angle: float,
    entry_altitude: float = 120.0e3,
) -> float:
    """Ground range (m) from entry interface to impact.

    :math:`R = h_E/\\tan\\gamma_E`, which follows directly from the constant
    flight-path angle Allen-Eggers assumes: a trajectory that does not turn
    is a straight line, and a straight line's ground range is its altitude
    over its tangent.

    This replaces a hardcoded 300 km. The constant was equivalent to
    assuming a 21.8 degree entry from 120 km, so it was reasonable for a
    shallow ballistic arc and understated a steep one by up to 2.5x.

    It is an upper bound, and by a measured amount
    -----------------------------------------------

    The real trajectory *steepens*: gravity turns it downward faster than
    drag can hold the flight-path angle, so the vehicle arrives short of
    where a straight line would put it. This function therefore always
    overstates the range. Against numerical integration of the full
    point-mass equations from 120 km at 6.5 km/s, the overstatement is

    .. code-block:: text

        gamma_E   beta=65   beta=500   beta=2000   beta=7500
          10 deg    72.6 %     52.1 %      41.3 %      35.5 %
          15 deg    51.2 %     32.4 %      22.6 %      18.2 %
          20 deg    41.2 %     23.5 %      14.3 %      11.1 %
          30 deg    31.9 %     15.2 %       7.0 %       5.4 %
          45 deg    25.9 %     10.1 %       3.2 %       2.8 %
          60 deg    23.1 %      7.8 %       2.0 %       1.8 %

    Two readings of that table matter. The error falls with both steepness
    and ballistic coefficient, so it is smallest exactly where this function
    is meant to be used -- a heavy reentry vehicle on a steep arc, where it
    is a few percent. And it never gets small for a low-:math:`\\beta`
    capsule at any angle, because such a vehicle decelerates high and then
    falls nearly vertically, which is not a straight line at all.

    The ballistic coefficient is deliberately not an argument. Taking one
    would imply this function corrects for it, and it does not; the table
    above is how the caller judges whether the bound is tight enough.

    Parameters
    ----------
    entry_angle:
        :math:`\\gamma_E` (rad), positive below horizontal. Must be at least
        :data:`MINIMUM_BALLISTIC_ENTRY_ANGLE` -- see the module docstring for
        why a shallow entry is refused rather than approximated.
    entry_altitude:
        :math:`h_E` (m). Defaults to the 120 km interface Gallais uses.

    Returns
    -------
    float
        Ground range (m), an upper bound.
    """
    if not np.isfinite(entry_angle) or not (
        MINIMUM_BALLISTIC_ENTRY_ANGLE <= entry_angle < 0.5 * np.pi
    ):
        msg = (
            f"entry_angle must lie in [{np.rad2deg(MINIMUM_BALLISTIC_ENTRY_ANGLE):.0f}, 90) "
            f"degrees, got {np.rad2deg(entry_angle):.2f}. Below that the flight-path "
            "angle does not stay constant and R = h/tan(gamma) is wrong, not merely "
            "imprecise."
        )
        raise ValueError(msg)
    if not (np.isfinite(entry_altitude) and entry_altitude > 0.0):
        msg = f"entry_altitude must be finite and > 0, got {entry_altitude}"
        raise ValueError(msg)
    return float(entry_altitude / np.tan(entry_angle))
