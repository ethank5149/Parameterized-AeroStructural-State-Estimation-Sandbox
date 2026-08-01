"""A PICA-like charring ablator for FIAT-formulation runs.

The synthetic material in :func:`passes.thermal.material.demo_material`
exists to exercise the spectral solver's manufactured solutions, and its
kinetics are deliberately fast so that decomposition is well resolved in
a short run. That makes it useless for realistic ablation: it releases
225 kg/m³ of pyrolysis gas with a millisecond time constant, driving
:math:`B'_g` into the tens, an order of magnitude above anything a real
ablator produces.

This module supplies a material of realistic *magnitude*, so that
recession, gas flux and :math:`B'` land where a phenolic-impregnated
carbon ablator actually puts them.

Provenance, stated per property
-------------------------------

**Published, from the MEDLI2 material-response paper** (Monk et al.,
"MEDLI2 Material Response Model Development and Validation", in
``reference/``), Heritage PICA model row:

===========================================  ==========
virgin density                                274 kg/m³
room-temperature virgin conductivity          0.174 W/(m K)
room-temperature char conductivity            0.224 W/(m K)
===========================================  ==========

**Reconstructed** from the published composition of PICA — a FiberForm
carbon preform impregnated with phenolic resin — by splitting the
274 kg/m³ into a non-decomposing carbon skeleton and a resin that chars
to roughly half its mass. The split reproduces the published virgin
density exactly and puts the char density at 227 kg/m³.

**Representative, not published**: the Arrhenius triplets, the
temperature slopes on conductivity and specific heat, and the pyrolysis
gas enthalpy.

The kinetics deserve a specific warning. No published Arrhenius triplets
for PICA appear anywhere in ``reference/`` — the MEDLI2 paper
characterises conductivity, specific heat and density and says nothing
about decomposition rates, and the MSL reconstruction paper notes that
"no kinetic rate-limited recession model for PICA exists that is
sufficiently validated for use in TPS design". Rather than invent three
numbers, the triplets here are pinned to *stated, checkable* targets and
those targets are asserted in the test suite: at a 20 K/min scan the
composite loses 2% of its decomposable mass by **557 K**, peaks in rate
at **799 K**, and leaves a char yield of **227/274 = 0.8285** — the last
being a consequence of the published bulk densities rather than a free
parameter. :mod:`passes.thermal.fiat.kinetics` provides the forward TGA
model and the fitter to replace all of this the moment a real scan is
available.

.. warning::

   This is a *PICA-like* material, not PICA. Results computed with it
   describe the solver, not the material, and must not be reported as
   PICA predictions. Closing a recession comparison against a published
   PICA case requires that case's own property set — see
   ``docs/FIAT-reference-data.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from passes.thermal.material import (
    ArrheniusComponent,
    CharringMaterial,
    LinearBlendProperty,
)

__all__ = [
    "HERITAGE_PICA_CONDUCTIVITY",
    "MEDLI2_PICA_CONDUCTIVITY",
    "MEDLI2_PICA_VIRGIN_DENSITY",
    "ONE_ATMOSPHERE",
    "PICA_CHAR_CONDUCTIVITY_RT",
    "PICA_LIKE_CHAR_DENSITY",
    "PICA_VIRGIN_CONDUCTIVITY_RT",
    "PICA_VIRGIN_DENSITY",
    "PressureConductivity",
    "pica_like_material",
    "structural_material",
]

_FloatArray = NDArray[np.float64]

#: Standard atmosphere, Pa — the upper anchor of the published table.
ONE_ATMOSPHERE = 101325.0
#: MEDLI2 flight-lot PICA virgin bulk density, kg/m³ — published.
MEDLI2_PICA_VIRGIN_DENSITY = 292.0


@dataclass(frozen=True)
class PressureConductivity:
    """Conductivity of a porous ablator as a function of pressure.

    PICA is a carbon preform with most of its volume as gas-filled pore
    space, so its conductivity depends on the pressure of the gas in
    those pores as well as on temperature and char fraction. The MEDLI2
    material-response paper tabulates that dependence directly (Table 3,
    "PICA Room Temperature Properties"), giving virgin and char
    conductivity at **both** 1 atm and 0.001 atm.

    Interpolation is linear in :math:`\\log p`, which is the natural
    variable for a Knudsen-regime transition and the only defensible
    choice given two anchors three decades apart. Outside the anchors the
    value is held constant rather than extrapolated: a two-point fit says
    nothing about the behaviour beyond its own endpoints, and entry
    trajectories routinely go below 0.001 atm.

    Attributes
    ----------
    low_pressure, high_pressure:
        The two tabulated pressures (Pa), ``low < high``.
    low, high:
        :class:`~passes.thermal.material.LinearBlendProperty` at each.
    """

    low_pressure: float
    high_pressure: float
    low: LinearBlendProperty
    high: LinearBlendProperty

    def __post_init__(self) -> None:
        if not (0.0 < self.low_pressure < self.high_pressure):
            raise ValueError(
                f"need 0 < low_pressure < high_pressure, got "
                f"{self.low_pressure} / {self.high_pressure}"
            )

    def value(
        self, temperature: ArrayLike, char_fraction: ArrayLike, pressure: float
    ) -> _FloatArray:
        """Conductivity (W/(m K)) at the given state and pressure."""
        if not (np.isfinite(pressure) and pressure > 0.0):
            raise ValueError(f"pressure must be finite and > 0, got {pressure}")
        span = np.log(self.high_pressure / self.low_pressure)
        w = float(np.clip(np.log(pressure / self.low_pressure) / span, 0.0, 1.0))
        return np.asarray(
            (1.0 - w) * self.low.value(temperature, char_fraction)
            + w * self.high.value(temperature, char_fraction)
        )

    def d_temperature(
        self, temperature: ArrayLike, char_fraction: ArrayLike, pressure: float
    ) -> _FloatArray:
        span = np.log(self.high_pressure / self.low_pressure)
        w = float(np.clip(np.log(pressure / self.low_pressure) / span, 0.0, 1.0))
        return np.asarray(
            (1.0 - w) * self.low.d_temperature(temperature, char_fraction)
            + w * self.high.d_temperature(temperature, char_fraction)
        )

    def d_char_fraction(
        self, temperature: ArrayLike, char_fraction: ArrayLike, pressure: float
    ) -> _FloatArray:
        span = np.log(self.high_pressure / self.low_pressure)
        w = float(np.clip(np.log(pressure / self.low_pressure) / span, 0.0, 1.0))
        return np.asarray(
            (1.0 - w) * self.low.d_char_fraction(temperature, char_fraction)
            + w * self.high.d_char_fraction(temperature, char_fraction)
        )


def _rt_conductivity(
    virgin_1atm: float,
    char_1atm: float,
    virgin_low: float,
    char_low: float,
    virgin_slope: float = 1.5e-4,
    char_slope: float = 4.5e-4,
) -> PressureConductivity:
    """Build a pressure-dependent conductivity from one row of Table 3.

    The table gives room-temperature values only, so the temperature
    slopes are supplied separately and are *representative*: the
    published data pins the 300 K intercepts at both pressures and
    nothing else.
    """
    return PressureConductivity(
        low_pressure=0.001 * ONE_ATMOSPHERE,
        high_pressure=ONE_ATMOSPHERE,
        low=LinearBlendProperty(
            virgin_low - 300.0 * virgin_slope,
            virgin_slope,
            char_low - 300.0 * char_slope,
            char_slope,
        ),
        high=LinearBlendProperty(
            virgin_1atm - 300.0 * virgin_slope,
            virgin_slope,
            char_1atm - 300.0 * char_slope,
            char_slope,
        ),
    )


#: Heritage PICA model, MEDLI2 paper Table 3 — all four values published.
#:
#: Note the direction. This model has virgin conductivity **rising** from
#: 0.174 to 0.520 W/(m K) as pressure falls from 1 atm to 0.001 atm — a
#: factor of three *increase* with decreasing pore-gas pressure, which is
#: the opposite of what a porous medium normally does. The MEDLI2
#: re-measurement in the same table has it falling by 25%, the expected
#: direction. The two disagree by a **factor of 4.1** at 0.001 atm, which
#: is the regime that governs entry. Both are provided; neither is
#: presented as correct.
HERITAGE_PICA_CONDUCTIVITY = _rt_conductivity(0.174, 0.224, 0.520, 0.202)
#: MEDLI2 re-measured PICA model, same table — all four values published.
MEDLI2_PICA_CONDUCTIVITY = _rt_conductivity(0.169, 0.169, 0.127, 0.143)

#: Heritage PICA virgin bulk density, kg/m³ — MEDLI2 paper, published.
PICA_VIRGIN_DENSITY = 274.0
#: Heritage PICA room-temperature virgin conductivity, W/(m K) — published.
PICA_VIRGIN_CONDUCTIVITY_RT = 0.174
#: Heritage PICA room-temperature char conductivity, W/(m K) — published.
PICA_CHAR_CONDUCTIVITY_RT = 0.224
#: Char density of the reconstructed composition, kg/m³ — *not* published.
PICA_LIKE_CHAR_DENSITY = 227.0

# Composition reconstruction. FIAT Eq. (7) is
# rho = Gamma (rho_A + rho_B) + (1 - Gamma) rho_C, so with Gamma = 1/2 the
# resin contributes (rho_A + rho_B)/2 and the carbon skeleton rho_C/2. Taking
# the FiberForm preform at 180 kg/m3 and the phenolic at 94 kg/m3 reproduces
# the published 274 exactly; the resin charring to half its mass puts the char
# density at 180 + 47 = 227 kg/m3.
_GAMMA = 0.5
_RESIN_TOTAL = 2.0 * 94.0
_CARBON_TOTAL = 2.0 * 180.0


def pica_like_material(heritage: bool = False) -> CharringMaterial:
    """A charring ablator with PICA's published bulk properties.

    Parameters
    ----------
    heritage:
        Use the Heritage PICA conductivity row of Table 3 instead of the
        MEDLI2 re-measured row. They differ by a factor of four at
        0.001 atm — see :data:`HERITAGE_PICA_CONDUCTIVITY`.

    Two resin components decomposing over overlapping temperature bands,
    plus a non-decomposing carbon skeleton — the three-component model of
    FIAT Eq. (7), used as intended rather than as three arbitrary
    reactions.
    """
    return CharringMaterial(
        # Low-temperature resin fraction: the lighter volatiles, off by ~700 K.
        resin_a=ArrheniusComponent(
            pre_exponential=1.4e4,
            activation_energy=7.1e4,
            reaction_order=3.0,
            virgin_density=0.30 * _RESIN_TOTAL,
            char_density=0.15 * _RESIN_TOTAL,
        ),
        # High-temperature resin fraction: the phenolic backbone, ~1100 K.
        resin_b=ArrheniusComponent(
            pre_exponential=4.5e9,
            activation_energy=1.70e5,
            reaction_order=3.0,
            virgin_density=0.70 * _RESIN_TOTAL,
            char_density=0.35 * _RESIN_TOTAL,
        ),
        # Carbon preform: present in Eq. (7) but inert. A zero
        # pre-exponential is FIAT's own way of writing a non-decomposing
        # component; the char density is held a hair below virgin only
        # because the material model requires a strict inequality.
        filler=ArrheniusComponent(
            pre_exponential=0.0,
            activation_energy=1.0e5,
            reaction_order=1.0,
            virgin_density=_CARBON_TOTAL,
            char_density=_CARBON_TOTAL * (1.0 - 1e-9),
        ),
        resin_fraction=_GAMMA,
        # The 1 atm column of Table 3. A ply given
        # `conductivity=MEDLI2_PICA_CONDUCTIVITY` overrides this with the
        # full pressure dependence; this field is the sea-level fallback for
        # code paths that have no pressure to hand.
        conductivity=(
            HERITAGE_PICA_CONDUCTIVITY if heritage else MEDLI2_PICA_CONDUCTIVITY
        ).high,
        specific_heat=LinearBlendProperty(1100.0, 0.32, 1250.0, 0.30),
        gas_specific_heat=2100.0,
        gas_enthalpy_offset=-2.2e6,
        gas_enthalpy_slope=2100.0,
        solid_enthalpy_offset=-1.1e6,
        solid_enthalpy_slope=1400.0,
        emissivity_virgin=0.85,
        emissivity_char=0.90,
    )


def structural_material(
    density: float = 1600.0,
    conductivity: float = 0.5,
    specific_heat: float = 900.0,
) -> CharringMaterial:
    """A non-decomposing substructure ply (bondline, honeycomb, laminate).

    FIAT's stacks routinely end in structure that conducts and stores
    heat but neither pyrolyses nor ablates. Expressing that as a
    :class:`~passes.thermal.material.CharringMaterial` with zero
    pre-exponentials — rather than as a separate type — is how FIAT's own
    material database does it, and it keeps one code path through the
    solver.
    """
    if not density > 0.0:
        raise ValueError(f"density must be > 0, got {density}")
    if not conductivity > 0.0:
        raise ValueError(f"conductivity must be > 0, got {conductivity}")
    if not specific_heat > 0.0:
        raise ValueError(f"specific_heat must be > 0, got {specific_heat}")
    # Eq. (7) is rho = Gamma(rho_A + rho_B) + (1 - Gamma) rho_C, and the
    # material model forbids a zero virgin density, so the two resin slots
    # carry a vanishing mass and the carbon slot carries the rest. With
    # Gamma = 1/2 that reproduces `density` to a part in 10^6.
    trace = 1.0e-6 * density
    inert = ArrheniusComponent(
        pre_exponential=0.0,
        activation_energy=1.0e5,
        reaction_order=1.0,
        virgin_density=trace,
        char_density=trace * (1.0 - 1e-9),
    )
    return CharringMaterial(
        resin_a=inert,
        resin_b=inert,
        filler=ArrheniusComponent(
            pre_exponential=0.0,
            activation_energy=1.0e5,
            reaction_order=1.0,
            virgin_density=2.0 * (density - trace),
            char_density=2.0 * (density - trace) * (1.0 - 1e-9),
        ),
        resin_fraction=0.5,
        conductivity=LinearBlendProperty(conductivity, 0.0, conductivity, 0.0),
        specific_heat=LinearBlendProperty(specific_heat, 0.0, specific_heat, 0.0),
        gas_specific_heat=1000.0,
        gas_enthalpy_offset=0.0,
        gas_enthalpy_slope=1000.0,
        solid_enthalpy_offset=0.0,
        solid_enthalpy_slope=specific_heat,
        emissivity_virgin=0.85,
        emissivity_char=0.85,
    )
