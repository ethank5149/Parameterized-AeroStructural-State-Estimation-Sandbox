"""Inertial injection error: deriving boost dispersion from IMU grade.

Boost injection error is the last stated specification in the accuracy
chain, and it is the one input every terminal-accuracy number rests on. It
need not be assumed: for a boost lasting a few hundred seconds it follows
from the inertial measurement unit's own error sources by three
propagation laws that are textbook and that scale differently in time,
which is what makes the dominant term identifiable.

The three laws, and why the exponents matter
--------------------------------------------

Over a burn short compared with the Schuler period of about 84 minutes,
the error dynamics have not yet begun to oscillate and the growth is
polynomial:

.. math::

    \\delta r_{\\text{accel}} = \\tfrac{1}{2} b_a t^2, \\qquad
    \\delta r_{\\text{align}} = \\tfrac{1}{2} g\\, \\delta\\theta\\, t^2,
    \\qquad
    \\delta r_{\\text{gyro}}  = \\tfrac{1}{6} g\\, \\varepsilon\\, t^3.

An accelerometer bias integrates twice into position. An initial
*misalignment* tilts the platform so a component of gravity is read as
acceleration, which integrates the same way. A gyro drift rate tilts the
platform progressively, so the false gravity component grows linearly and
integrates *three* times.

The differing exponents are the useful part. A gyro term that is
negligible on a short burn dominates a long one, and the crossover is
computable rather than a matter of judgement:
:func:`dominant_error_source` reports which term leads and
:func:`gyro_dominates_after` gives the burn duration at which the gyro
overtakes the accelerometer.

What this derives and what it does not
--------------------------------------

The *propagation* is derived; the *component specifications* are not, and
this module takes them as inputs because an IMU grade is a procurement
decision rather than a physical constant. Representative grades are given
in :data:`IMU_GRADES` with their provenance stated as "conventional
industry bands, not a measured source" — because that is what they are.
Verifying those bands needs the strapdown-navigation literature, and the
canonical references are named in the project roadmap rather than
paraphrased here.

Velocity error matters more than position
-----------------------------------------

For an injection the position error at burnout is usually the smaller
problem. A velocity error persists and is amplified by the transfer that
follows: :mod:`passes.orbital.fobs` measures a sensitivity of 850 to 3484
seconds depending on perigee depth, so 1 m/s of injection velocity error
becomes kilometres at the entry interface. :func:`injection_error` returns
both, and the velocity term is the one to watch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "IMU_GRADES",
    "ImuGrade",
    "InjectionError",
    "dominant_error_source",
    "gyro_dominates_after",
    "injection_error",
]

#: Standard gravity (m/s²), the lever that turns a tilt into a false
#: acceleration and therefore the reason attitude errors matter at all.
_G0 = 9.80665
#: Degrees per hour to radians per second.
_DEG_PER_HOUR = np.pi / 180.0 / 3600.0
#: Micro-g to m/s².
_MICRO_G = 1e-6 * _G0


@dataclass(frozen=True)
class ImuGrade:
    """Error specification of an inertial measurement unit.

    Attributes
    ----------
    accelerometer_bias:
        One-sigma bias (m/s²).
    gyro_drift:
        One-sigma drift rate (rad/s).
    alignment:
        One-sigma initial platform misalignment (rad).
    label:
        Grade name, carried through so a result can be read back against
        the assumption that produced it.
    """

    accelerometer_bias: float
    gyro_drift: float
    alignment: float
    label: str = ""

    def __post_init__(self) -> None:
        for name in ("accelerometer_bias", "gyro_drift", "alignment"):
            value = float(getattr(self, name))
            if not (np.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be finite and >= 0, got {value}")

    @classmethod
    def from_engineering_units(
        cls,
        accelerometer_bias_micro_g: float,
        gyro_drift_deg_per_hour: float,
        alignment_arcsec: float,
        label: str = "",
    ) -> ImuGrade:
        """Construct from the units instruments are actually specified in."""
        return cls(
            accelerometer_bias=accelerometer_bias_micro_g * _MICRO_G,
            gyro_drift=gyro_drift_deg_per_hour * _DEG_PER_HOUR,
            alignment=alignment_arcsec * np.pi / (180.0 * 3600.0),
            label=label,
        )


#: Instrument grades from Groves, *Principles of GNSS, Inertial, and
#: Multisensor Integrated Navigation Systems*, 2nd ed., **Table 4.1**
#: ("Typical Accelerometer and Gyro Biases for Different Grades of IMU").
#:
#: The taxonomy is his, and adopting it corrected ours. Earlier revisions
#: used "strategic / navigation / tactical", of which only the last is a
#: term Groves uses, and the guessed numbers sat in roughly the right
#: places under the wrong names: our "navigation" (50 ug, 0.01 deg/hr)
#: lands inside his *aviation* band, while our "strategic" (5 ug) was twice
#: as optimistic as his best listed grade, *marine*.
#:
#: Where the table gives a range the better end is taken, with the range
#: recorded alongside so the pessimistic end is not lost.
#:
#: **Alignment is not from Groves.** Table 4.1 covers instrument biases
#: only; initial platform misalignment depends on the alignment procedure
#: and its duration rather than on the instrument, so those values remain
#: stated inputs.
IMU_GRADES: dict[str, ImuGrade] = {
    # Table 4.1: 0.01 mg, 0.001 deg/hr.
    "marine": ImuGrade.from_engineering_units(
        accelerometer_bias_micro_g=10.0,
        gyro_drift_deg_per_hour=0.001,
        alignment_arcsec=5.0,
        label="marine",
    ),
    # Table 4.1: 0.03-0.1 mg, 0.01 deg/hr.
    "aviation": ImuGrade.from_engineering_units(
        accelerometer_bias_micro_g=30.0,
        gyro_drift_deg_per_hour=0.01,
        alignment_arcsec=20.0,
        label="aviation",
    ),
    # Table 4.1: 0.1-1 mg, 0.1 deg/hr.
    "intermediate": ImuGrade.from_engineering_units(
        accelerometer_bias_micro_g=100.0,
        gyro_drift_deg_per_hour=0.1,
        alignment_arcsec=40.0,
        label="intermediate",
    ),
    # Table 4.1: 1-10 mg, 1-100 deg/hr.
    "tactical": ImuGrade.from_engineering_units(
        accelerometer_bias_micro_g=1000.0,
        gyro_drift_deg_per_hour=1.0,
        alignment_arcsec=60.0,
        label="tactical",
    ),
    # Table 4.1: >3 mg, >100 deg/hr.
    "consumer": ImuGrade.from_engineering_units(
        accelerometer_bias_micro_g=3000.0,
        gyro_drift_deg_per_hour=100.0,
        alignment_arcsec=300.0,
        label="consumer",
    ),
}


@dataclass(frozen=True)
class InjectionError:
    """Position and velocity error at burnout, by contributing source."""

    position: float
    """Total one-sigma position error (m), root-sum-square."""
    velocity: float
    """Total one-sigma velocity error (m/s), root-sum-square."""
    from_accelerometer: float
    from_alignment: float
    from_gyro: float
    """Per-source position contributions (m)."""
    burn_time: float
    grade: str = ""

    @property
    def contributions(self) -> dict[str, float]:
        return {
            "accelerometer": self.from_accelerometer,
            "alignment": self.from_alignment,
            "gyro": self.from_gyro,
        }


def injection_error(grade: ImuGrade, burn_time: float) -> InjectionError:
    """Injection error at burnout from an IMU grade and a burn duration.

    Valid while the burn is short against the Schuler period of about
    84 minutes, which every boost phase is. Beyond that the error dynamics
    oscillate rather than grow polynomially and these expressions stop
    describing them; the function refuses rather than extrapolating.
    """
    t = float(burn_time)
    if not (np.isfinite(t) and t > 0.0):
        raise ValueError(f"burn_time must be finite and > 0, got {t}")
    schuler_period = 2.0 * np.pi * np.sqrt(6371e3 / _G0)
    if t > 0.25 * schuler_period:
        raise ValueError(
            f"burn_time {t:.6g} s exceeds a quarter of the Schuler period "
            f"({0.25 * schuler_period:.0f} s); beyond that the inertial error "
            f"dynamics oscillate rather than grow polynomially and these "
            f"expressions no longer describe them"
        )

    from_accelerometer = 0.5 * grade.accelerometer_bias * t**2
    from_alignment = 0.5 * _G0 * grade.alignment * t**2
    from_gyro = _G0 * grade.gyro_drift * t**3 / 6.0

    velocity = float(
        np.sqrt(
            (grade.accelerometer_bias * t) ** 2
            + (_G0 * grade.alignment * t) ** 2
            + (0.5 * _G0 * grade.gyro_drift * t**2) ** 2
        )
    )
    position = float(np.sqrt(from_accelerometer**2 + from_alignment**2 + from_gyro**2))
    return InjectionError(
        position=position,
        velocity=velocity,
        from_accelerometer=float(from_accelerometer),
        from_alignment=float(from_alignment),
        from_gyro=float(from_gyro),
        burn_time=t,
        grade=grade.label,
    )


def dominant_error_source(grade: ImuGrade, burn_time: float) -> str:
    """Which of the three terms contributes most position error."""
    contributions = injection_error(grade, burn_time).contributions
    return max(contributions, key=lambda key: contributions[key])


def gyro_dominates_after(grade: ImuGrade) -> float:
    """Burn duration (s) at which gyro drift overtakes accelerometer bias.

    From :math:`g \\varepsilon t^3/6 = b_a t^2/2`, so
    :math:`t = 3 b_a / (g \\varepsilon)`. This is the number that decides
    which instrument to spend money on for a given burn, and it depends on
    the *ratio* of the two specifications rather than on either alone.
    """
    if grade.gyro_drift == 0.0:
        return float("inf")
    return float(3.0 * grade.accelerometer_bias / (_G0 * grade.gyro_drift))
