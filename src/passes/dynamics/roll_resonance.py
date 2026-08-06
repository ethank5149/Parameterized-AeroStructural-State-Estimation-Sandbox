"""Roll dynamics and roll-through-resonance for a ballistic reentry vehicle.

Implements

* F. J. Regan, *Re-Entry Vehicle Dynamics* (AIAA, 1984), §13.2
  (Eqs. 13.8-13.12, roll rate history) and §13.5 / Ch. X
  (Eqs. 13.77-13.79, the resonance condition).

Why this exists
---------------

The mission budget's ballistic dispersion entry was an order-of-magnitude
figure "chosen to exercise the accounting", and its justification was that
no model here produced one. That was true because the vehicle was a point
mass with a ballistic coefficient: it had no attitude, so the classical
reentry-vehicle dispersion mechanisms were not small in our model, they
were **absent**. This module supplies the first of them.

Resonance, and why a reentry vehicle crosses it twice
-----------------------------------------------------

A statically stable vehicle has an undamped pitch frequency

.. math:: \\omega_{n\\alpha} = \\sqrt{-M_\\alpha/I_y}
        = V_\\infty\\sqrt{\\tfrac12 \\rho\\, P_s},

where :math:`P_s = -C_{m\\alpha} S d / I_y` is Regan's *stability factor*,
carrying units of m/kg. Spinning at roll rate :math:`p`, the motion becomes
singular in the undamped limit when

.. math:: \\omega_{n\\alpha} = \\sqrt{1 - I_x/I_y}\\; p

(Regan Eq. 13.79). Since :math:`I_x/I_y` is of order 0.1 for a slender
reentry body, this is close to :math:`\\omega_{n\\alpha} = p`, and any trim angle of
attack — from a centre-of-gravity offset, an asymmetric ablation pattern, a
manufacturing tolerance — is amplified as the condition is approached.

The structurally important point is that :math:`\\omega_{n\\alpha}` is **not
monotone** through an entry. It scales as :math:`V\\sqrt{\\rho}`, and on the
way down :math:`\\rho` rises while :math:`V` falls, so the pitch frequency
climbs, peaks, and falls again. A vehicle at fixed roll rate therefore
crosses the resonance condition **twice** — "first" (high) and "second"
(low) resonance — or, if its peak pitch frequency never reaches the roll
rate, not at all. Which of those happens is decided by the ballistic
coefficient, because that sets how deep the vehicle gets before it slows.

Verification
------------

Regan works a case: :math:`V_E = 5` km/s, entry angle 75 degrees,
:math:`P_s = 3.73\\times10^{-3}` m/kg, roll rate 18 rad/s. He reports one
resonance for a ballistic factor of :math:`6\\times10^4` Pa and two for
:math:`6\\times10^3` Pa, the latter at 37 km and about 11 km. Driving this
module with the independently verified Allen-Eggers profile of
:mod:`passes.flight.ballistic_entry` reproduces that: **one crossing at
37.0 km** for the heavy case, **two at 36.5 and 10.2 km** for the light one.
The count is right in both, and the altitudes land within a kilometre or
three on an exponential atmosphere rather than the tabulated model of his
Chapter II.

One inconsistency in the source is worth recording: he gives the first
resonance as "about 34 km" for the heavy vehicle and then "as before, at
37 km" for the light one, though the two are the same quantity and his own
argument implies they should barely differ. This module computes 37.0 and
36.5, which is consistent with the "as before" and not with the 34.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "ResonanceCrossing",
    "pitch_frequency",
    "resonance_condition_ratio",
    "resonance_crossings",
    "roll_rate",
    "steady_state_roll_rate",
    "trim_amplification",
]

_FloatArray = NDArray[np.float64]


def pitch_frequency(
    velocity: ArrayLike,
    density: ArrayLike,
    stability_factor: float,
) -> _FloatArray:
    """Undamped non-rolling pitch frequency (rad/s).

    .. math:: \\omega_{n\\alpha} = V_\\infty \\sqrt{\\tfrac12 \\rho_\\infty P_s}

    Parameters
    ----------
    velocity, density:
        Freestream speed (m/s) and density (kg/m^3), scalar or array.
    stability_factor:
        :math:`P_s = -C_{m\\alpha} S d / I_y` (m/kg), positive for a
        statically stable vehicle. Regan uses
        :math:`3.73\\times10^{-3}` m/kg for his worked reentry body.

    Returns
    -------
    numpy.ndarray
        Pitch frequency (rad/s).
    """
    if not (np.isfinite(stability_factor) and stability_factor > 0.0):
        msg = (
            f"stability_factor must be finite and > 0, got {stability_factor}. "
            "A non-positive value is a statically unstable vehicle, which has "
            "no real pitch frequency and no resonance to cross."
        )
        raise ValueError(msg)
    speed = np.asarray(velocity, dtype=np.float64)
    rho = np.asarray(density, dtype=np.float64)
    if np.any(speed < 0.0) or np.any(rho < 0.0):
        msg = "velocity and density must be non-negative"
        raise ValueError(msg)
    return np.asarray(speed * np.sqrt(0.5 * rho * stability_factor))


def resonance_condition_ratio(inertia_ratio: float) -> float:
    """:math:`\\sqrt{1 - I_x/I_y}` — Regan Eq. (13.79).

    Resonance occurs where :math:`\\omega_{n\\alpha}` equals this factor
    times the roll rate. For a slender reentry body :math:`I_x/I_y \\approx
    0.1`, giving 0.95, which is why the condition is usually quoted as
    simply "pitch frequency equals roll rate" — an approximation good to
    5 % and worth keeping visible rather than absorbing.

    Parameters
    ----------
    inertia_ratio:
        :math:`I_x/I_y`, in [0, 1). Above 1 the body is a disc rather than a
        rod and this treatment does not apply.
    """
    ratio = float(inertia_ratio)
    if not (np.isfinite(ratio) and 0.0 <= ratio < 1.0):
        msg = f"inertia_ratio must lie in [0, 1), got {ratio}"
        raise ValueError(msg)
    return float(np.sqrt(1.0 - ratio))


def steady_state_roll_rate(
    roll_moment_coefficient: float,
    roll_damping_coefficient: float,
    velocity: float,
    reference_length: float,
) -> float:
    """:math:`p_{ss} = -C_{l0} V_\\infty/(C_{lp} d)` — Regan Eq. (13.12).

    The rate a driving roll moment winds the vehicle up to, once damping
    balances it. Independent of inertia and of dynamic pressure: those set
    how *fast* it is reached, not what it is.

    Parameters
    ----------
    roll_moment_coefficient:
        :math:`C_{l0}`, the driving moment. For a reentry vehicle with no
        fins this comes from asymmetry — a centre-of-gravity offset from the
        centreline, or an uneven ablation pattern. Regan is explicit that
        there is no general method to predict it.
    roll_damping_coefficient:
        :math:`C_{lp}`, which **must be negative** for the roll to settle
        rather than diverge.
    velocity, reference_length:
        :math:`V_\\infty` (m/s) and :math:`d` (m).
    """
    clp = float(roll_damping_coefficient)
    if not (np.isfinite(clp) and clp < 0.0):
        msg = (
            f"roll_damping_coefficient must be finite and < 0, got {clp}. "
            "A non-negative C_lp is a vehicle whose roll rate diverges."
        )
        raise ValueError(msg)
    for name, value in (
        ("velocity", float(velocity)),
        ("reference_length", float(reference_length)),
    ):
        if not (np.isfinite(value) and value > 0.0):
            msg = f"{name} must be finite and > 0, got {value}"
            raise ValueError(msg)
    return float(-roll_moment_coefficient * float(velocity) / (clp * float(reference_length)))


def roll_rate(
    time: ArrayLike,
    initial_roll_rate: float,
    steady_state: float,
    time_constant: float,
) -> _FloatArray:
    """:math:`p(t) = [p(0) - p_{ss}]e^{\\lambda t} + p_{ss}` — Eq. (13.11).

    Parameters
    ----------
    time:
        Seconds from the initial condition.
    initial_roll_rate, steady_state:
        :math:`p(0)` and :math:`p_{ss}` (rad/s).
    time_constant:
        :math:`-1/\\lambda` (s), positive, with
        :math:`\\lambda = C_{lp} q S d^2/(I_x V_\\infty)`. Taken as a time
        constant rather than as :math:`\\lambda` so that the sign convention
        cannot be got wrong silently: a negative value here is rejected,
        whereas a positive :math:`\\lambda` would quietly produce a
        diverging roll.
    """
    tau = float(time_constant)
    if not (np.isfinite(tau) and tau > 0.0):
        msg = f"time_constant must be finite and > 0, got {tau}"
        raise ValueError(msg)
    t = np.asarray(time, dtype=np.float64)
    return np.asarray((initial_roll_rate - steady_state) * np.exp(-t / tau) + steady_state)


def trim_amplification(
    pitch_frequency_value: ArrayLike,
    roll_rate_value: float,
    damping_ratio: float,
    inertia_ratio: float = 0.1,
) -> _FloatArray:
    """Amplification of a static trim angle near resonance.

    A single-degree-of-freedom resonant response, driven at the roll rate:

    .. math::

        \\frac{\\alpha}{\\alpha_{\\text{trim}}}
          = \\Big[\\big(1 - r^2\\big)^2 + \\big(2\\zeta r\\big)^2\\Big]^{-1/2},
        \\qquad r = \\frac{\\sqrt{1 - I_x/I_y}\\;p}{\\omega_{n\\alpha}}

    Regan does not give this closed form — he states only that the response
    is singular without damping and "considerably" amplified with the low
    damping ratios reentry vehicles have. The standard second-order form is
    used here so that the amplification is a number rather than an
    adjective, and it is labelled as our construction rather than his.

    What it is good for is the *shape* and the scale: at
    :math:`\\zeta = 0.05` the peak is a factor of 10, and the width in
    frequency is of order :math:`\\zeta`, so a vehicle sweeps through the
    peak quickly and the true amplification depends on the sweep rate. This
    function is a quasi-steady bound, not a swept-resonance solution.

    Parameters
    ----------
    pitch_frequency_value:
        :math:`\\omega_{n\\alpha}` (rad/s), scalar or array.
    roll_rate_value:
        :math:`p` (rad/s).
    damping_ratio:
        :math:`\\zeta`, typically 0.02-0.25 for a reentry vehicle.
    inertia_ratio:
        :math:`I_x/I_y`.
    """
    zeta = float(damping_ratio)
    if not (np.isfinite(zeta) and zeta > 0.0):
        msg = (
            f"damping_ratio must be finite and > 0, got {zeta}. Undamped "
            "resonance is singular, which is a statement about the model "
            "rather than a number to return."
        )
        raise ValueError(msg)
    omega = np.asarray(pitch_frequency_value, dtype=np.float64)
    if np.any(omega <= 0.0):
        msg = "pitch_frequency_value must be positive"
        raise ValueError(msg)
    r = resonance_condition_ratio(inertia_ratio) * abs(float(roll_rate_value)) / omega
    return np.asarray(1.0 / np.sqrt((1.0 - r**2) ** 2 + (2.0 * zeta * r) ** 2))


@dataclass(frozen=True)
class ResonanceCrossing:
    """One crossing of the resonance condition during an entry.

    Attributes
    ----------
    altitude:
        Where it happens (m).
    pitch_frequency:
        :math:`\\omega_{n\\alpha}` there (rad/s), equal to the resonance
        condition by construction.
    ascending:
        ``True`` where the pitch frequency is still rising with decreasing
        altitude — the "first" or high resonance — and ``False`` on the way
        back down, the "second" or low one. The distinction matters for
        dispersion: a first-resonance excursion happens in thin air but has
        the rest of the trajectory to act over, while a second-resonance
        excursion is aerodynamically far more forceful but has little time
        left to steer the impact point.
    """

    altitude: float
    pitch_frequency: float
    ascending: bool


def resonance_crossings(
    altitudes: ArrayLike,
    velocities: ArrayLike,
    densities: ArrayLike,
    roll_rate_value: float,
    stability_factor: float,
    inertia_ratio: float = 0.1,
) -> tuple[ResonanceCrossing, ...]:
    """Find where an entry trajectory crosses the resonance condition.

    Scans a descending trajectory for sign changes of
    :math:`\\omega_{n\\alpha} - \\sqrt{1-I_x/I_y}\\,p` and interpolates each
    linearly. Returns them ordered by decreasing altitude, so the first
    entry is the "first resonance".

    **Zero, one or two crossings are all physical**, and which occurs is set
    by the ballistic coefficient. A vehicle whose peak pitch frequency never
    reaches the roll rate never resonates; a heavy one may only clip the
    condition once; a light one decelerates high, so its pitch frequency
    peaks above the roll rate and comes back down through it.

    Parameters
    ----------
    altitudes, velocities, densities:
        Trajectory samples, same length. ``altitudes`` need not be sorted;
        the result is ordered internally.
    roll_rate_value:
        :math:`p` (rad/s), taken as constant. Regan notes explicitly that
        the conclusions change when the roll rate varies appreciably over
        the trajectory, which :func:`roll_rate` can supply.
    stability_factor, inertia_ratio:
        As above.
    """
    h = np.asarray(altitudes, dtype=np.float64)
    v = np.asarray(velocities, dtype=np.float64)
    rho = np.asarray(densities, dtype=np.float64)
    if not (h.shape == v.shape == rho.shape) or h.ndim != 1 or h.size < 2:
        msg = "altitudes, velocities and densities must be 1-D arrays of equal length >= 2"
        raise ValueError(msg)

    order = np.argsort(h)
    h, v, rho = h[order], v[order], rho[order]
    omega = pitch_frequency(v, rho, stability_factor)
    threshold = resonance_condition_ratio(inertia_ratio) * abs(float(roll_rate_value))
    residual = omega - threshold

    crossings: list[ResonanceCrossing] = []
    for i in range(residual.size - 1):
        lo, hi = residual[i], residual[i + 1]
        if lo == 0.0:
            fraction = 0.0
        elif lo * hi < 0.0:
            fraction = lo / (lo - hi)
        else:
            continue
        crossings.append(
            ResonanceCrossing(
                altitude=float(h[i] + fraction * (h[i + 1] - h[i])),
                pitch_frequency=float(threshold),
                # Ascending in altitude means descending in time, so the
                # pitch frequency is falling here iff residual is rising.
                ascending=bool(hi > lo),
            )
        )
    return tuple(sorted(crossings, key=lambda c: -c.altitude))
