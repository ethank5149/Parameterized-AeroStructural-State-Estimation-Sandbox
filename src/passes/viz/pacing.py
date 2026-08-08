"""How fast to play each second of flight, decided by the flight.

A trajectory animation has one hard problem that is not rendering: a
fractional orbital profile is **4 % boost, 71 % parking coast and 25 %
descent**, and the interesting parts are the 4 % and the last minute of the
25 %. Played at uniform rate the ascent is five frames out of a hundred and
thirty, and the viewer watches a dot cross a blue circle for most of a
minute.

The previous answer weighted each *declared phase* by
``duration ** 0.45``. That helped, and it has a structural flaw: it takes
its cue from labels rather than from the trajectory, so it gives a
uniformly-boring phase the same treatment as one containing a staging
event, and it cannot slow down for something that happens in the middle of a
long leg. A deorbit burn inside a two-hour coast is invisible to it.

What this does instead
----------------------

It reads the **history's own arrays** and builds an attention density —
frames per second of flight — from quantities that are large exactly when
something is visibly happening:

``climb``
    :math:`|\\dot h|`, the altitude rate. Large on ascent and on terminal
    descent; near zero on a coast. This is the term that gets the launch and
    the impact watched in full.
``manoeuvre``
    The **specific force**, :math:`|\\dot{\\mathbf v} + \\mu\\hat r/r^2|` —
    acceleration with gravity removed. The gravity subtraction is the whole
    value of this term: on a ballistic coast the raw acceleration magnitude
    is 8.7 m/s² everywhere and carries no information at all, so a first
    version of this weighted the parking orbit as heavily as the boost.
    What is left after subtracting gravity is thrust and drag, which is
    exactly what "something is happening" means. Computed **only when the
    producer carries velocities** — see :func:`attention_density`.
``ground``
    Angular rate of the sub-vehicle point. Distinguishes a fast low pass
    from a slow high one, which the altitude rate alone does not. Weighted
    lightly by default: in orbit it is large and nearly constant, so leaning
    on it makes a coast look busy.
``attitude``
    Body angular rate, when the producer carries attitude. Tumbling,
    re-orientation for a burn, and separation transients.
``proximity``
    Nearness to the ground, :math:`1/(1 + h/h_{\\rm ref})`. Unlike every
    other term this is not a rate — it says that being *near the surface* is
    interesting regardless of what is happening, which is the direct
    expression of "show the launch and the impact in full". It is what stops
    a terminal descent from being compressed just because a conic
    trajectory's last minute has no dramatic derivative.
``events``
    A kernel around every declared event, because a discrete moment —
    ignition, separation, impact — has no derivative signature of its own at
    a coarse sample rate but is exactly what should be lingered on.

Each is normalised by its own robust scale and compressed by a power law, so
a 3 g boost does not consume the entire frame budget of a run that also
contains a 40 g entry. They are summed on top of a **floor**, which is the
guarantee that no part of the flight is skipped: at ``floor=0.04`` the
quietest second of a coast still gets 4 % of the frame rate of the busiest.

The frame times are then the inverse cumulative distribution of that
density, sampled uniformly. That makes the endpoints exact and the grid
strictly increasing by construction — not by rounding per-phase frame counts
and hoping they add up, which is how the previous version could and did land
short of the end of the flight.

Nothing here invents data. Every term is a finite difference of an array the
simulator produced, or a kernel around an event the simulator declared.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from passes.viz.history import SimulationHistory

__all__ = ["PacingProfile", "PacingWeights", "attention_density", "uniform_pacing"]

_FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PacingWeights:
    """Relative importance of each attention term.

    Defaults chosen so that on a fractional orbital profile the boost, the
    deorbit burn and the last minute of entry each get a visible share, and
    the parking coast is compressed but not deleted. They are weights on
    *compressed, robustly normalised* signals, so a weight of 1 does not
    mean "this term dominates", it means "this term contributes one unit of
    interest when it is at its own typical large value".
    """

    climb: float = 1.0
    manoeuvre: float = 0.8
    ground: float = 0.0
    """Off by default: in orbit it is large and nearly constant, so it adds a
    pedestal to the coast rather than contrast anywhere."""
    proximity: float = 1.6
    proximity_scale: float = 60.0e3
    """:math:`h_{\\rm ref}` for the proximity term (m)."""
    attitude: float = 0.25
    events: float = 1.2
    floor: float = 0.04
    """Density floor as a fraction of the mean.

    The guarantee that nothing is skipped, and the main dial between
    "watchable" and "faithful". At 0.04 the quietest second of a coast still
    gets 4 % of the frame rate of the busiest, which on a 72-minute
    fractional profile is the difference between a parking orbit that takes
    59 % of the video and one that takes 22 %.
    """
    compression: float = 1.0
    """Exponent applied to each normalised signal. See :func:`_normalise`."""
    ceiling: float = 8.0
    """Cap on a normalised signal, in units of the flight's own busy level."""
    event_window: float = 20.0
    """Half-width in seconds of the kernel around each event."""
    smoothing: float = 8.0
    """Gaussian smoothing of the density, in seconds.

    Without it the density inherits every wiggle of the finite differences
    and the playback rate visibly stutters. With too much, a staging event
    is smeared into the coast around it.
    """

    def __post_init__(self) -> None:
        if not 0.0 < self.floor <= 1.0:
            msg = f"floor must be in (0, 1], got {self.floor}"
            raise ValueError(msg)
        for name in ("climb", "manoeuvre", "ground", "attitude", "events",
                     "proximity"):
            if float(getattr(self, name)) < 0.0:
                msg = f"{name} weight must be >= 0, got {getattr(self, name)}"
                raise ValueError(msg)
        if self.event_window <= 0.0 or self.smoothing < 0.0:
            msg = "event_window must be > 0 and smoothing >= 0"
            raise ValueError(msg)
        if not 0.0 < self.compression <= 4.0:
            msg = f"compression must be in (0, 4], got {self.compression}"
            raise ValueError(msg)
        if self.ceiling <= 1.0:
            msg = f"ceiling must be > 1, got {self.ceiling}"
            raise ValueError(msg)
        if self.proximity_scale <= 0.0:
            msg = f"proximity_scale must be > 0, got {self.proximity_scale}"
            raise ValueError(msg)


@dataclass(frozen=True)
class PacingProfile:
    """A per-sample attention density and the frame grid it implies."""

    times: _FloatArray
    density: _FloatArray
    """Relative frames per second of flight; strictly positive."""
    terms: dict[str, _FloatArray] = field(default_factory=dict, repr=False)
    """Each normalised contribution, kept so a plot can show why."""

    def __post_init__(self) -> None:
        t = np.asarray(self.times, dtype=np.float64)
        d = np.asarray(self.density, dtype=np.float64)
        if t.ndim != 1 or t.size < 2:
            msg = f"times must be a 1-D array of 2+ samples, got shape {t.shape}"
            raise ValueError(msg)
        if d.shape != t.shape:
            msg = f"density shape {d.shape} does not match times {t.shape}"
            raise ValueError(msg)
        if np.any(np.diff(t) <= 0.0):
            msg = "times must be strictly increasing"
            raise ValueError(msg)
        if np.any(d <= 0.0) or np.any(~np.isfinite(d)):
            msg = "density must be finite and strictly positive everywhere"
            raise ValueError(msg)

    @property
    def cumulative(self) -> _FloatArray:
        """Normalised cumulative attention, 0 at the start and 1 at the end."""
        integral = np.concatenate(
            [[0.0], np.cumsum(0.5 * (self.density[1:] + self.density[:-1])
                              * np.diff(self.times))]
        )
        return np.asarray(integral / integral[-1])

    def grid(self, n_frames: int) -> _FloatArray:
        """Frame times: the inverse cumulative distribution, sampled uniformly."""
        if n_frames < 2:
            msg = f"need at least two frames, got {n_frames}"
            raise ValueError(msg)
        grid = np.interp(
            np.linspace(0.0, 1.0, int(n_frames)), self.cumulative, self.times
        )
        grid[0], grid[-1] = self.times[0], self.times[-1]
        return np.asarray(grid)

    def rate(self, time: float, n_frames: int, fps: int) -> float:
        """Multiple of real time the frame at ``time`` plays back at.

        The whole point of non-uniform pacing is that this is not constant,
        and a viewer who cannot see it cannot tell a slow pass from a fast
        one — which is why the animator puts it in the HUD.
        """
        video = max(n_frames / max(fps, 1), 1e-9)
        span = float(self.times[-1] - self.times[0])
        mean = float(
            np.trapezoid(self.density, self.times) / span
        ) if span > 0.0 else 1.0
        local = float(np.interp(float(time), self.times, self.density))
        return float(span / video * mean / max(local, 1e-12))

    def share(self, start: float, stop: float) -> float:
        """Fraction of the frames spent between two times."""
        low, high = np.interp(
            [float(start), float(stop)], self.times, self.cumulative
        )
        return float(high - low)


def _normalise(
    signal: _FloatArray,
    exponent: float = 1.0,
    ceiling: float = 8.0,
    quiet: float = 10.0,
    busy: float = 90.0,
) -> _FloatArray:
    """Scale a signal against *this flight's own* quiet and busy levels.

    .. math::

        \\hat x = \\min\\!\\left(
        \\frac{x - P_{10}}{P_{90} - P_{10}},\\; c\\right)^{p}

    Three deliberate choices, each of which a simpler version got wrong:

    **The low percentile is subtracted, not just divided out.** Dividing by
    :math:`P_{90}` alone maps the *typical* sample to 1, and on a profile
    that is 70 % parking coast the typical sample *is* the coast — so the
    coast came out as interesting as everything else and took 59 % of the
    frames. Subtracting :math:`P_{10}` makes each term measure departure
    from this flight's own quiet level, which is the thing worth watching.

    **Clipped at a ceiling.** The boost's specific force is 55 times the
    coast's; unclipped it takes the entire frame budget and the rest of the
    flight flickers past. Eight is roughly "an order of magnitude above
    normal is as interesting as it needs to be".

    **Exponent at or above one.** An earlier version used 0.4, on the
    reasoning that compression prevents any one phase dominating. It does
    the opposite of what was wanted: for a normalised signal in
    :math:`[0,1]`, a fractional power *raises* the small values —
    :math:`0.1^{0.4} = 0.40` — so it was actively promoting the quiet parts
    it was meant to compress.
    """
    values = np.abs(np.asarray(signal, dtype=np.float64))
    values = np.where(np.isfinite(values), values, 0.0)
    low = float(np.percentile(values, quiet))
    high = float(np.percentile(values, busy))
    span = high - low
    if not np.isfinite(span) or span <= 0.0:
        return np.zeros_like(values)
    scaled = np.clip((values - low) / span, 0.0, float(ceiling))
    return np.asarray(scaled ** float(exponent))


def _smooth(values: _FloatArray, times: _FloatArray, width: float) -> _FloatArray:
    """Gaussian smoothing on a possibly non-uniform time grid.

    Done as an explicit weighted sum rather than a convolution because the
    history's samples are not uniformly spaced — an adaptive integrator
    emits them where it needed them — and convolving unequal spacings
    silently weights the dense regions more.
    """
    if width <= 0.0 or times.size < 3:
        return values
    # Truncated at three standard deviations; beyond that the weight is
    # under 1 % and the cost is quadratic in the number of samples.
    span = 3.0 * width
    out: _FloatArray = np.empty_like(values)
    for index, centre in enumerate(times):
        low = np.searchsorted(times, centre - span, side="left")
        high = np.searchsorted(times, centre + span, side="right")
        window = times[low:high]
        weight = np.exp(-0.5 * ((window - centre) / width) ** 2)
        total = float(weight.sum())
        window_values = values[low:high]
        out[index] = (
            float(np.dot(weight, window_values) / total)
            if total > 0.0
            else float(values[index])
        )
    return out


def attention_density(
    history: SimulationHistory,
    body_radius: float,
    weights: PacingWeights | None = None,
    gravitational_parameter: float = 3.986004418e14,
) -> PacingProfile:
    """Build a pacing profile from a run's own state history.

    Parameters
    ----------
    body_radius:
        Used only to turn positions into altitudes for the climb term.
    gravitational_parameter:
        :math:`\\mu`, used to subtract gravity from the acceleration so the
        manoeuvre term measures specific force rather than free fall.
    """
    weights = weights if weights is not None else PacingWeights()
    times = np.asarray(history.times, dtype=np.float64)
    if times.size < 3:
        return uniform_pacing(history)

    positions = np.asarray(history.positions, dtype=np.float64)
    altitude = history.altitudes(body_radius)
    terms: dict[str, _FloatArray] = {}

    # --- climb: |dh/dt|
    climb = np.gradient(altitude, times)
    terms["climb"] = weights.climb * _normalise(climb, weights.compression, weights.ceiling)

    # --- manoeuvre: specific force, but only where it can be measured
    #
    # This needs one derivative of a carried velocity. A producer that
    # carries only positions would need *two* finite differences, and on a
    # sampled conic that is differentiation noise rather than physics: on
    # the 170x250 km parking orbit it came back as 0.85 m/s^2 of "specific
    # force" where the true value is zero, which made the coast the second
    # most interesting thing in the flight and took it 48 % of the frames.
    # So the term is reported as unavailable rather than approximated —
    # the same rule `has_attitude` follows.
    if history.velocities is not None:
        velocity = np.asarray(history.velocities, dtype=np.float64)
        acceleration = np.gradient(velocity, times, axis=0)
        radius_now = np.maximum(np.linalg.norm(positions, axis=1), 1.0)
        gravity = -gravitational_parameter * positions / radius_now[:, None] ** 3
        terms["manoeuvre"] = weights.manoeuvre * _normalise(
            np.linalg.norm(acceleration - gravity, axis=1),
            weights.compression,
            weights.ceiling,
        )
    else:
        terms["manoeuvre"] = np.zeros_like(times)

    # --- ground: angular rate of the sub-vehicle point
    radius = np.maximum(np.linalg.norm(positions, axis=1), 1e-9)
    unit = positions / radius[:, None]
    swept = np.linalg.norm(np.gradient(unit, times, axis=0), axis=1)
    terms["ground"] = weights.ground * _normalise(swept, weights.compression, weights.ceiling)

    # --- attitude: body rate, only when attitude is genuinely carried
    if history.has_attitude and history.quaternions is not None:
        quaternions = np.asarray(history.quaternions, dtype=np.float64)
        rate = np.linalg.norm(np.gradient(quaternions, times, axis=0), axis=1)
        terms["attitude"] = weights.attitude * _normalise(
            rate, weights.compression, weights.ceiling
        )
    else:
        terms["attitude"] = np.zeros_like(times)

    # --- proximity: near the ground is interesting on its own
    terms["proximity"] = weights.proximity / (
        1.0 + np.maximum(altitude, 0.0) / weights.proximity_scale
    )

    # --- events: a Gaussian bump on each declared moment
    bumps = np.zeros_like(times)
    for event in history.events:
        bumps += np.exp(
            -0.5 * ((times - float(event.time)) / weights.event_window) ** 2
        )
    terms["events"] = weights.events * bumps

    total = sum(terms.values())
    total = _smooth(np.asarray(total), times, weights.smoothing)
    mean = float(np.mean(total))
    if not np.isfinite(mean) or mean <= 0.0:
        return uniform_pacing(history)
    density = total + weights.floor * mean
    return PacingProfile(times=times, density=np.asarray(density), terms=terms)


def uniform_pacing(history: SimulationHistory) -> PacingProfile:
    """Constant density — real time, evenly sampled.

    Kept as a named option rather than a special case: comparing an animation
    against its uniform version is how you tell whether the pacing is helping
    or hiding something.
    """
    times = np.asarray(history.times, dtype=np.float64)
    return PacingProfile(times=times, density=np.ones_like(times))
