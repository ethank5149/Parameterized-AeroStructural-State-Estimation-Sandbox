"""Terminal homing: what it actually takes to reach a small CEP.

Inertial guidance alone does not get a hypersonic vehicle near its aimpoint.
The accumulated error at the end of midcourse is kilometre-class, and no
refinement of the *inertial* chain removes it, because the error is in
knowing where the target is relative to the vehicle, not in flying the
trajectory. Closing that gap needs a sensor, and then enough control
authority to act on what the sensor says.

The binding constraint is divert, not the seeker
------------------------------------------------

It is natural to assume the seeker's angular accuracy sets terminal
accuracy. Usually it does not. Acquire at range :math:`R` with angular
one-sigma :math:`\\sigma_\\theta`, and the lateral position uncertainty is
:math:`\\sigma_\\theta R` — for a 1 mrad seeker at 20 km that is 20 m, which
already meets most requirements. What binds instead is time and
acceleration. A vehicle at 6 km/s that acquires at 20 km has 3.3 s left. To
null a 1200 m lateral error in that time, proportional navigation demands
roughly

.. math::

    a_{\\text{lat}} \\approx \\frac{N' \\, \\Delta}{t_{go}^2},

which at :math:`N' = 3` is about 330 m/s², or **34 g**. A hypersonic glide
vehicle does not have 34 g of lateral acceleration; its lift is bounded by
dynamic pressure and its own structure. So the achievable miss is set by
what the vehicle can divert to, and the seeker is usually the cheap part.

Three floors, and the answer is the largest
-------------------------------------------

:func:`terminal_homing` returns the binding one by name rather than a bare
number, because the remedy differs completely:

* **Divert-limited** — the error at acquisition is larger than the vehicle
  can null in the time remaining. Fix by reducing the *handover* error
  (better midcourse, or a mid-flight update), by acquiring earlier, or by
  buying lateral acceleration.
* **Seeker-limited** — the loop nulls the handover error but cannot resolve
  better than :math:`\\sigma_\\theta` at the range where the loop stops
  responding. Fix with a better seeker.
* **Loop-limited** — too few guidance time constants remain after
  acquisition for the loop to converge at all. Fix by acquiring earlier or
  by tightening the autopilot.
* **Target-location-limited** — the aimpoint's own coordinates are not
  known any better than this. No sensor and no manoeuvre can beat it,
  because the vehicle would be flying accurately to the wrong place. Fix
  with better survey, or with a seeker that homes on the *scene* rather
  than on a coordinate, which changes the problem rather than the number.

That last floor deserves emphasis because it is the one most often left
out of an accuracy budget, and because :mod:`passes.geodesy` shows how
easily it is blown: confusing geodetic with geocentric latitude displaces
an aimpoint by 21 km, and using the wrong arrival epoch displaces it by
28 km per minute. A 10 m CEP is a statement about survey and timekeeping
before it is a statement about guidance.

Scope and honesty
-----------------

This is an engineering scaling, not an adjoint miss-distance analysis. The
homing loop is represented by a single time constant and a navigation
ratio; a real design would run the adjoint and carry glint, radome
refraction and target motion separately. What the model is good for is
sizing — telling you which of the three floors you are against, and by how
much — and it is deliberately not dressed as more than that.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import scipy.integrate
from numpy.typing import NDArray

_FloatArray = NDArray[np.float64]

__all__ = [
    "MissLimit",
    "Seeker",
    "TerminalEngagement",
    "achievable_cep",
    "homing_miss",
    "required_lateral_acceleration",
    "terminal_homing",
]


class MissLimit(Enum):
    """Which floor sets the achievable miss."""

    DIVERT = "divert-limited"
    SEEKER = "seeker-limited"
    LOOP = "loop-limited"
    TARGET_LOCATION = "target-location-limited"


@dataclass(frozen=True)
class Seeker:
    """A terminal sensor.

    Attributes
    ----------
    angular_sigma:
        One-sigma line-of-sight angular error (rad). A 1 mrad seeker is
        good; 0.1 mrad is excellent.
    acquisition_range:
        Range (m) at which the target is first held well enough to guide
        on. This is usually the dominant parameter, because it sets how
        much time the loop has.
    """

    angular_sigma: float
    acquisition_range: float

    def __post_init__(self) -> None:
        for name in ("angular_sigma", "acquisition_range"):
            value = float(getattr(self, name))
            if not (np.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and > 0, got {value}")

    def lateral_sigma_at(self, distance: float) -> float:
        """Lateral position uncertainty (m) at a given range."""
        return float(self.angular_sigma * float(distance))


def homing_miss(
    flight_time: float,
    time_constant: float,
    navigation_ratio: float = 3.0,
    heading_error_rate: float = 0.0,
    target_acceleration: float = 0.0,
) -> float:
    """Miss distance (m) of a linearised single-lag PN homing loop.

    Integrates the closed loop of Zarchan, *Tactical and Strategic Missile
    Guidance*, Ch. 6 — the same system his adjoint method evaluates, solved
    directly rather than by adjoint. For the linear model the two agree
    exactly; the adjoint is faster when sweeping flight times, and direct
    integration is clearer when the flight time is fixed.

    .. code-block:: text

        y''   = n_T - n_L
        n_L'  = (n_c - n_L) / tau,   n_c = N' (y + y' t_go) / t_go^2

    where ``y + y' t_go`` is the zero-effort miss.

    Verified against three statements in the source: miss from an initial
    heading error vanishes for :math:`t_F/\\tau \\gtrsim 10`; normalised
    miss from a step target manoeuvre peaks at finite :math:`t_F/\\tau`
    rather than growing without bound; and doubling the guidance time
    constant raises manoeuvre-induced miss by more than an order of
    magnitude — measured here as a factor of **12.0**, against Zarchan's
    "more than an order of magnitude" for the same case.

    Parameters
    ----------
    heading_error_rate:
        Initial lateral velocity error (m/s).
    target_acceleration:
        Step manoeuvre (m/s²) held through the engagement.
    """
    t_f = float(flight_time)
    tau = float(time_constant)
    if not (np.isfinite(t_f) and t_f > 0.0):
        raise ValueError(f"flight_time must be finite and > 0, got {t_f}")
    if not (np.isfinite(tau) and tau > 0.0):
        raise ValueError(f"time_constant must be finite and > 0, got {tau}")
    if not (np.isfinite(navigation_ratio) and navigation_ratio > 0.0):
        raise ValueError(f"navigation_ratio must be finite and > 0, got {navigation_ratio}")
    # The guidance command diverges as t_go -> 0, which is a property of the
    # law and not of the integration. Stopping a hair short is the standard
    # treatment and is why miss distance is finite at all.
    epsilon = 1e-4 * t_f

    def rhs(time: float, state: _FloatArray) -> list[float]:
        position, velocity, applied = state
        t_go = max(t_f - time, epsilon)
        zero_effort = position + velocity * t_go
        commanded = navigation_ratio * zero_effort / t_go**2
        return [
            velocity,
            float(target_acceleration) - applied,
            (commanded - applied) / tau,
        ]

    solution = scipy.integrate.solve_ivp(
        rhs,
        (0.0, t_f - epsilon),
        np.array([0.0, float(heading_error_rate), 0.0]),
        rtol=1e-10,
        atol=1e-12,
    )
    if not solution.success:  # pragma: no cover - stiff only if misconfigured
        raise RuntimeError(f"homing loop integration failed: {solution.message}")
    return float(solution.y[0, -1])


def required_lateral_acceleration(
    handover_error: float, time_to_go: float, navigation_ratio: float = 3.0
) -> float:
    """Lateral acceleration (m/s²) needed to null an error before impact.

    Proportional navigation removes an initial displacement
    :math:`\\Delta` with a peak demand of order :math:`N'\\Delta/t_{go}^2`.
    The result is what makes terminal accuracy a *vehicle* problem rather
    than a sensor one: it grows as the inverse square of the time
    remaining, so halving the acquisition range quadruples the demand.
    """
    delta = float(handover_error)
    t_go = float(time_to_go)
    if not (np.isfinite(delta) and delta >= 0.0):
        raise ValueError(f"handover_error must be finite and >= 0, got {delta}")
    if not (np.isfinite(t_go) and t_go > 0.0):
        raise ValueError(f"time_to_go must be finite and > 0, got {t_go}")
    if not (np.isfinite(navigation_ratio) and navigation_ratio > 0.0):
        raise ValueError(f"navigation_ratio must be finite and > 0, got {navigation_ratio}")
    return float(navigation_ratio * delta / t_go**2)


@dataclass(frozen=True)
class TerminalEngagement:
    """What the terminal phase achieves, and what limits it."""

    sigma: float
    """Achievable one-sigma miss (m)."""
    limited_by: MissLimit
    time_to_go: float
    """Seconds of homing available after acquisition."""
    time_constants: float
    """How many guidance time constants that is. Below about 5 the loop
    cannot converge and the handover error survives largely intact."""
    required_acceleration: float
    """Lateral acceleration (m/s²) the handover error demands."""
    available_acceleration: float
    seeker_floor: float
    """Miss (m) the seeker alone would permit."""

    @property
    def acceleration_margin(self) -> float:
        """Available over required. Below 1 the engagement is divert-limited."""
        if self.required_acceleration == 0.0:
            return float("inf")
        return self.available_acceleration / self.required_acceleration


def terminal_homing(
    seeker: Seeker,
    handover_sigma: float,
    closing_speed: float,
    lateral_acceleration: float,
    guidance_time_constant: float = 0.5,
    navigation_ratio: float = 3.0,
    minimum_time_constants: float = 5.0,
    target_location_sigma: float = 0.0,
) -> TerminalEngagement:
    """Achievable terminal miss, and which of the three floors sets it.

    Parameters
    ----------
    handover_sigma:
        One-sigma error (m) inherited from midcourse at acquisition.
    closing_speed:
        Speed (m/s) toward the target; sets how long acquisition range
        buys.
    lateral_acceleration:
        What the vehicle can actually pull (m/s²). For a glide vehicle
        this is bounded by lift at the terminal dynamic pressure.
    guidance_time_constant:
        Autopilot plus seeker-filter lag (s). The loop needs several of
        these to converge.
    target_location_sigma:
        One-sigma uncertainty (m) in the aimpoint's own coordinates. No
        sensor or manoeuvre beats this, because the vehicle would be
        flying accurately to the wrong place.

    Notes
    -----
    The seeker floor is evaluated at the range where the loop stops
    responding — about one time constant of closing distance — rather than
    at acquisition. Evaluating it at acquisition range is a common and
    badly optimistic error, since the seeker is far more accurate in linear
    terms when it is close.
    """
    for name, value in (
        ("handover_sigma", handover_sigma),
        ("closing_speed", closing_speed),
        ("lateral_acceleration", lateral_acceleration),
        ("guidance_time_constant", guidance_time_constant),
    ):
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be finite and > 0, got {value}")

    time_to_go = seeker.acquisition_range / float(closing_speed)
    time_constants = time_to_go / float(guidance_time_constant)

    # The seeker cannot help inside the last time constant, so its floor is
    # its angular error at that range, not at acquisition.
    freeze_range = float(closing_speed) * float(guidance_time_constant)
    seeker_floor = seeker.lateral_sigma_at(freeze_range)

    location = float(target_location_sigma)
    if not (np.isfinite(location) and location >= 0.0):
        raise ValueError(f"target_location_sigma must be finite and >= 0, got {location}")

    required = required_lateral_acceleration(handover_sigma, time_to_go, navigation_ratio)
    available = float(lateral_acceleration)

    if time_constants < minimum_time_constants:
        # Too few time constants for the loop to converge. The surviving
        # fraction is *computed* from the linearised loop rather than
        # modelled as an exponential decay: the handover error enters as an
        # initial lateral rate, and `homing_miss` returns what is left of
        # it. An exponential was the first model here and it is smooth
        # where the real response is not.
        rate = handover_sigma / max(time_to_go, 1e-6)
        residual = abs(
            homing_miss(
                time_to_go,
                float(guidance_time_constant),
                navigation_ratio=navigation_ratio,
                heading_error_rate=rate,
            )
        )
        sigma = float(residual)
        return TerminalEngagement(
            sigma=max(sigma, seeker_floor, location),
            limited_by=(
                MissLimit.TARGET_LOCATION
                if location >= max(sigma, seeker_floor)
                else MissLimit.LOOP
            ),
            time_to_go=time_to_go,
            time_constants=time_constants,
            required_acceleration=required,
            available_acceleration=available,
            seeker_floor=seeker_floor,
        )

    if required > available:
        # Divert-limited: only the fraction of the error the vehicle can
        # actually null is removed.
        removable = available / required
        sigma = float(handover_sigma * (1.0 - removable))
        return TerminalEngagement(
            sigma=max(sigma, seeker_floor, location),
            limited_by=(
                MissLimit.TARGET_LOCATION
                if location >= max(sigma, seeker_floor)
                else MissLimit.DIVERT
            ),
            time_to_go=time_to_go,
            time_constants=time_constants,
            required_acceleration=required,
            available_acceleration=available,
            seeker_floor=seeker_floor,
        )

    return TerminalEngagement(
        sigma=max(seeker_floor, location),
        limited_by=(MissLimit.TARGET_LOCATION if location >= seeker_floor else MissLimit.SEEKER),
        time_to_go=time_to_go,
        time_constants=time_constants,
        required_acceleration=required,
        available_acceleration=available,
        seeker_floor=seeker_floor,
    )


def achievable_cep(engagement: TerminalEngagement) -> float:
    """Circular error probable (m) for a circular terminal dispersion.

    Terminal homing tends to leave a nearly isotropic residual — unlike
    the midcourse error it replaces — because the loop acts equally in both
    lateral axes. The circular Rayleigh constant is therefore appropriate
    here specifically, and is not a general licence to use it.
    """
    return float(engagement.sigma * np.sqrt(2.0 * np.log(2.0)))
