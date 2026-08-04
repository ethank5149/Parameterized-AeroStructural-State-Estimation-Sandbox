"""Powered hypersonic cruise: range, the cruise-climb, and where it beats glide.

Everything else in :mod:`passes.guidance` spends energy the vehicle already
has. A cruise vehicle carries a propulsion system and adds energy back, and
that changes the range problem from an accounting exercise into a
mass-fraction one. The governing result is Bréguet's, which for steady
level flight at speed :math:`V` with specific impulse :math:`I_{sp}` and
lift-to-drag ratio :math:`L/D` gives

.. math::

    R = \\frac{V I_{sp} g_0}{1} \\cdot \\frac{L}{D}
        \\ln\\!\\left(\\frac{m_i}{m_f}\\right)
      \\;\\;\\text{(scramjet form: } I_{sp}\\text{ per unit fuel mass)}.

Three consequences follow, and all three are measurable rather than
rhetorical.

**Range is logarithmic in mass ratio and linear in everything else — and
the usual gloss on that is backwards.** It is tempting to say fuel has
diminishing returns while :math:`L/D` does not. Measured, the opposite
holds over the interesting range: at Mach 8, doubling :math:`L/D` from 4 to
8 multiplies range by exactly 2.00, while doubling the fuel fraction from
0.30 to 0.60 multiplies it by **2.57**. The reason is that
:math:`\\ln(1/(1-f))` is *convex* in fuel fraction — its derivative
:math:`1/(1-f)` grows — so doubling :math:`f` more than doubles the
logarithm. The diminishing return is in **mass ratio**, not in fuel
fraction: doubling :math:`m_i/m_f` always adds exactly :math:`\\ln 2`,
whatever it was before. Those are different statements about different
variables, and conflating them misprices tankage against aerodynamics.

**The cruise-climb is not an optimisation, it is a consequence.** Level
flight at constant speed requires lift to equal weight. As fuel burns the
weight falls, so either the vehicle flies at constant altitude and sheds
lift by reducing incidence — moving off its best :math:`L/D` — or it
climbs to thinner air and holds the same aerodynamic state. The second is
the Bréguet assumption, and :func:`cruise_climb_altitude` shows how far it
actually climbs: for a 30% fuel fraction in an exponential atmosphere the
answer is :math:`H \\ln(1/0.7) \\approx 3.0` km, which is small enough to
be flown and large enough that ignoring it costs range.

**Airbreathing beats rocket cruise only where the atmosphere cooperates.**
:math:`I_{sp}` for a scramjet falls with Mach number as the captured air's
own kinetic energy grows relative to the heat that can be added, while a
rocket's is constant. :func:`cruise_range` takes either as a callable, and
:func:`crossover_mach` finds where they trade.

Cruise versus glide
-------------------

The comparison this module exists to support is against
:mod:`passes.guidance.entry`. A glider's range comes from its initial
energy and its :math:`L/D`; a cruiser's comes from fuel. They scale
differently, and :func:`cruise_versus_glide` reports the fuel fraction at
which cruise overtakes an unpowered glide of the same vehicle from the same
initial state. That number, not any absolute range, is the honest way to
compare two things that buy range with different currencies.

Scope
-----

Steady, level, unaccelerated cruise with a fixed :math:`L/D` and an
exponential atmosphere, matching :mod:`passes.guidance.entry`. The
acceleration and descent phases at either end are outside this — a real
range figure must add them, and they subtract. Thermal limits, which in
practice are what actually bound a hypersonic cruiser's speed, are not
modelled here; :mod:`passes.aerothermal` is where that constraint lives.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.optimize
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "CruiseComparison",
    "CruiseVehicle",
    "crossover_mach",
    "cruise_altitude",
    "cruise_climb_altitude",
    "cruise_dynamic_pressure",
    "cruise_range",
    "cruise_versus_glide",
    "scramjet_specific_impulse",
]

_FloatArray = NDArray[np.float64]

_G0 = 9.80665
_H_SCALE = 8500.0
_RHO0 = 1.225
#: Representative hypersonic speed of sound (m/s) in the cruise corridor.
#: The stratosphere is close to isothermal between 20 and 50 km, which is
#: what makes a single value defensible here rather than lazy.
_SOUND_SPEED = 295.0


@dataclass(frozen=True)
class CruiseVehicle:
    """A powered hypersonic cruiser in steady level flight.

    Attributes
    ----------
    wing_loading:
        :math:`m/S` (kg/m²) at the start of cruise.
    lift_to_drag:
        Cruise :math:`L/D`. The single most valuable parameter in the range
        equation, entering linearly.
    lift_coefficient:
        :math:`C_L` held through the cruise. Together with wing loading it
        fixes the dynamic pressure the vehicle must fly at, and therefore
        the altitude.
    fuel_fraction:
        Fuel mass as a fraction of initial mass, in ``[0, 1)``. Enters
        range only through :math:`\\ln(1/(1-f))`, so it has sharply
        diminishing returns.
    """

    wing_loading: float
    lift_to_drag: float
    lift_coefficient: float
    fuel_fraction: float

    def __post_init__(self) -> None:
        for name in ("wing_loading", "lift_to_drag", "lift_coefficient"):
            value = float(getattr(self, name))
            if not (np.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and > 0, got {value}")
        if not (0.0 <= self.fuel_fraction < 1.0):
            raise ValueError(
                f"fuel_fraction must lie in [0, 1); a vehicle cannot burn its "
                f"whole mass. Got {self.fuel_fraction}"
            )

    @property
    def mass_ratio(self) -> float:
        """:math:`m_i / m_f`, the quantity range is logarithmic in."""
        return 1.0 / (1.0 - self.fuel_fraction)


def scramjet_specific_impulse(
    mach: ArrayLike, sea_level: float = 3500.0, decay_mach: float = 12.0
) -> _FloatArray:
    """Scramjet specific impulse (s), falling with Mach number.

    A deliberately simple decay, :math:`I_{sp} = I_0 (1 - M / M_{decay})`
    clipped at zero. The *shape* is what matters for the trades here — an
    airbreather's efficiency collapses as the captured air's own kinetic
    energy grows relative to the heat that can be added to it — and a
    detailed cycle model would change the crossover Mach number without
    changing which way the trade goes.

    Notes
    -----
    This is a stand-in, not a cycle analysis, and the numbers should not be
    quoted as a performance claim for any real engine. It is here so that
    :func:`crossover_mach` has something to work against.
    """
    m = np.asarray(mach, dtype=np.float64)
    if np.any(m < 0.0):
        raise ValueError("mach must be non-negative")
    if not (np.isfinite(decay_mach) and decay_mach > 0.0):
        raise ValueError(f"decay_mach must be finite and > 0, got {decay_mach}")
    return np.asarray(np.maximum(sea_level * (1.0 - m / decay_mach), 0.0))


def cruise_dynamic_pressure(vehicle: CruiseVehicle) -> float:
    """Dynamic pressure (Pa) required for level flight.

    From :math:`L = W`: :math:`q = (m/S) g_0 / C_L`. This is the constraint
    that ties a cruise vehicle to an altitude band, and it is why a
    hypersonic cruiser flies where it does rather than wherever is
    convenient.
    """
    return float(vehicle.wing_loading * _G0 / vehicle.lift_coefficient)


def cruise_altitude(vehicle: CruiseVehicle, speed: float) -> float:
    """Altitude (m) at which level flight closes, in the exponential model.

    Inverts :math:`q = \\tfrac12 \\rho V^2` for density and then the
    atmosphere for altitude.
    """
    if not (np.isfinite(speed) and speed > 0.0):
        raise ValueError(f"speed must be finite and > 0, got {speed}")
    density = 2.0 * cruise_dynamic_pressure(vehicle) / speed**2
    if density >= _RHO0:
        # Level flight would need denser air than exists at sea level, so
        # there is no altitude at which this vehicle closes at this speed.
        return 0.0
    return float(-_H_SCALE * np.log(density / _RHO0))


def cruise_climb_altitude(vehicle: CruiseVehicle) -> float:
    """Altitude gained (m) over the cruise while holding aerodynamic state.

    Level flight needs lift to equal weight, so as fuel burns the vehicle
    must either give up lift — moving off its best :math:`L/D` — or climb
    into thinner air and hold the same :math:`C_L` and Mach. The second is
    what Bréguet assumes, and in an exponential atmosphere the climb is
    exactly :math:`H \\ln(m_i/m_f)`, independent of the vehicle.

    That independence is the useful part: the climb depends only on the
    fuel fraction and the scale height, so it is roughly 3 km for a 30%
    fuel fraction whatever the vehicle, which is small enough to fly and
    large enough that treating cruise as constant-altitude loses range.
    """
    return float(_H_SCALE * np.log(vehicle.mass_ratio))


def cruise_range(
    vehicle: CruiseVehicle,
    speed: float,
    specific_impulse: float | Callable[[float], float],
) -> float:
    """Bréguet range (m) for steady level cruise.

    Parameters
    ----------
    specific_impulse:
        Either a constant (s) — appropriate for a rocket — or a callable
        of Mach number, for an airbreather whose efficiency depends on
        flight speed.

    Notes
    -----
    Returns zero for a vehicle carrying no fuel rather than raising: an
    unfuelled cruiser has no cruise range, which is a meaningful answer and
    the correct limit of the formula.
    """
    if not (np.isfinite(speed) and speed > 0.0):
        raise ValueError(f"speed must be finite and > 0, got {speed}")
    if callable(specific_impulse):
        isp = float(np.asarray(specific_impulse(speed / _SOUND_SPEED)).item())
    else:
        isp = float(specific_impulse)
    if not (np.isfinite(isp) and isp >= 0.0):
        raise ValueError(f"specific impulse must be finite and >= 0, got {isp}")
    return float(speed * isp * _G0 * vehicle.lift_to_drag * np.log(vehicle.mass_ratio) / _G0)


def crossover_mach(
    vehicle: CruiseVehicle,
    rocket_isp: float = 450.0,
    airbreather: Callable[[float], float] | None = None,
    bracket: tuple[float, float] = (2.0, 20.0),
) -> float | None:
    """Mach at which a rocket overtakes an airbreather on cruise range.

    Below it the airbreather wins because it carries no oxidiser; above it
    the airbreather's specific impulse has decayed far enough that the
    rocket's constant value is worth more. Returns ``None`` when no
    crossover exists inside ``bracket``, which is a real possibility and
    better reported than bracketed into a fictitious root.
    """
    engine = scramjet_specific_impulse if airbreather is None else airbreather

    def scalar_engine(mach_number: float) -> float:
        return float(np.asarray(engine(mach_number)).item())

    def difference(mach: float) -> float:
        speed = mach * _SOUND_SPEED
        return cruise_range(vehicle, speed, scalar_engine) - cruise_range(
            vehicle, speed, rocket_isp
        )

    low, high = float(bracket[0]), float(bracket[1])
    if not low < high:
        raise ValueError(f"bracket must be increasing, got {bracket}")
    if difference(low) * difference(high) > 0.0:
        return None
    return float(scipy.optimize.brentq(difference, low, high, xtol=1e-10))


@dataclass(frozen=True)
class CruiseComparison:
    """Where powered cruise overtakes an unpowered glide of the same vehicle."""

    glide_range: float
    """Unpowered range (m) from the same initial energy."""
    cruise_range_at_fraction: float
    """Powered range (m) at the vehicle's own fuel fraction."""
    breakeven_fuel_fraction: float | None
    """Fuel fraction at which the two are equal, or ``None`` if cruise never
    catches the glide within a physically admissible fraction."""


def cruise_versus_glide(
    vehicle: CruiseVehicle,
    speed: float,
    specific_impulse: float | Callable[[float], float],
    glide_range: float,
) -> CruiseComparison:
    """Fuel fraction at which cruise range equals a given glide range.

    Comparing absolute ranges between a cruiser and a glider is close to
    meaningless — they buy range with different currencies, one from fuel
    and one from initial energy. The break-even fuel fraction is the honest
    comparison, because it states the price of matching the glide in the
    only unit that transfers.

    Returns ``None`` for the break-even when no admissible fuel fraction
    suffices, which happens whenever the glide is long and the
    characteristic length :math:`V I_{sp} (L/D)` is short.
    """
    if not (np.isfinite(glide_range) and glide_range > 0.0):
        raise ValueError(f"glide_range must be finite and > 0, got {glide_range}")

    def powered(fraction: float) -> float:
        trial = CruiseVehicle(
            wing_loading=vehicle.wing_loading,
            lift_to_drag=vehicle.lift_to_drag,
            lift_coefficient=vehicle.lift_coefficient,
            fuel_fraction=fraction,
        )
        return cruise_range(trial, speed, specific_impulse)

    breakeven: float | None
    ceiling = 0.95
    if powered(ceiling) < glide_range:
        breakeven = None
    else:
        breakeven = float(
            scipy.optimize.brentq(lambda f: powered(f) - glide_range, 1e-9, ceiling, xtol=1e-12)
        )
    return CruiseComparison(
        glide_range=float(glide_range),
        cruise_range_at_fraction=powered(vehicle.fuel_fraction),
        breakeven_fuel_fraction=breakeven,
    )
