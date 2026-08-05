"""In-depth pore pressure from Darcy flow of pyrolysis gas.

The framework carries a *pressure-dependent* conductivity for PICA — the
MEDLI2 model of :mod:`passes.thermal.fiat.materials`, which changes by a
factor of 4.1 between 1 atm and 0.001 atm — and until now handed it the
**boundary-layer edge pressure** at every depth. That is the wrong
pressure. The conductivity is a function of the gas pressure *in the
pores*, and pore pressure is set by pyrolysis gas fighting its way out
through the material's own permeability. It peaks inside the pyrolysis
zone and can exceed surface pressure substantially; Ahn et al., via
Rabinovitch §IV.A, put it at "a factor of two or higher" for the Pioneer
Venus probe.

The governing form is quadratic in pressure
-------------------------------------------

Steady compressible Darcy flow through a porous slab, with the gas obeying
the ideal law :math:`\\rho_g = PM/(RT)`:

.. math::

    \\dot m = -\\frac{\\rho_g K}{\\mu}\\frac{\\mathrm{d}P}{\\mathrm{d}x}
            = -\\frac{P M K}{R T \\mu}\\frac{\\mathrm{d}P}{\\mathrm{d}x}
    \\;\\Longrightarrow\\;
    \\frac{\\mathrm{d}(P^2)}{\\mathrm{d}x} = -\\frac{2 \\dot m R T \\mu}{M K}.

Solving for :math:`P^2` rather than :math:`P` is not a trick — it is the
natural variable, because the gas density is itself proportional to
pressure and the two factors combine. Integrating inward from the surface,

.. math::

    P^2(x) = P_s^2 + \\int_0^x \\frac{2 \\dot m(x') R T(x') \\mu}{M K}
             \\,\\mathrm{d}x',

with :math:`\\dot m(x)` the accumulated gas mass flux, which is zero at the
back face and grows as the flow gathers everything decomposing above it.

The permeability we do not have
-------------------------------

**PICA's permeability is not in this repository's references, and this
module does not pretend otherwise.** Park & Lawrence's measurements (in
``reference/``) are for MX4926 carbon *cloth* phenolic — a dense rocket
nozzle liner near 1.45 g/cm³ — and give :math:`10^{-17}` to
:math:`10^{-21}` m². PICA is a low-density fibrous preform near 270 kg/m³
at roughly 90 % porosity, and is more permeable by many orders of
magnitude; using a nozzle-liner number for it would be wrong in the
direction that matters most. Lachaud and Mansour cite Marschall & Milos,
*Gas Permeability of Rigid Fibrous Refractory Insulations*, for the
correct value; that paper is not held here.

So permeability is an explicit argument with no default, and
:func:`pore_pressure_sensitivity` exists to answer the question that
missing number raises: *does it change anything?* Sweeping it across the
plausible range and watching what the conductivity does is a better use of
an unknown than picking a value and hoping.

The measured answer: it is real, and it is small
------------------------------------------------

That sweep has been run, against real pyrolysis profiles taken from the
solver on two of the Milos & Chen arcjet conditions. The pore-to-surface
pressure ratio is large — up to 31 at low permeability — but the
conductivity error it causes is not:

.. code-block:: text

    surface p    permeability   peak P/Ps   virgin k error
    27.3 kPa     1e-10 m^2          1.06            0.1 %
    27.3 kPa     1e-11 m^2          1.53            1.0 %
    27.3 kPa     1e-13 m^2         11.61            3.0 %
     2.3 kPa     1e-10 m^2          3.3             2.9 %
     2.3 kPa     1e-11 m^2         10.0             5.6 %
     2.3 kPa     1e-12 m^2         31.5             8.4 %

Two things bound it. The MEDLI2 model interpolates in *log* pressure, so a
tenfold pressure error is a fraction of one decade out of the three its
anchors span. And it **clamps** at the 1 atm anchor rather than
extrapolating — a decision taken when it was implemented, for unrelated
reasons, which turns out to cap this error too. The effect grows as
surface pressure falls, exactly as expected, and even at the lowest arcjet
pressure with an implausibly tight permeability it stays under 10 %.

**This module is therefore a diagnostic, not a correction, and is
deliberately not wired into the solver.** Doing so would inject an
unmeasured parameter into the main solve path to buy a change smaller than
the 27 % experimental scatter the recession comparison already lives
inside. Introducing an unknown to fix a known-small error makes the answer
less defensible, not more. If Marschall & Milos is ever obtained and PICA
turns out to sit below :math:`10^{-12}` m², this becomes worth revisiting
and the sweep above says by how much.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "PORE_PRESSURE_REFERENCES",
    "PorePressureProfile",
    "pore_pressure",
    "pore_pressure_sensitivity",
]

_FloatArray = NDArray[np.float64]

#: Universal gas constant (J/mol/K).
_R_UNIVERSAL = 8.314462618

#: Measured permeabilities, with the material each belongs to.
#:
#: The point of this table is the *gap* in it. Park & Lawrence is a real
#: measurement of the wrong material; the PICA entry is a placeholder
#: bracket inferred from porosity, not a measurement, and is labelled so
#: that nothing quotes it as one.
PORE_PRESSURE_REFERENCES: dict[str, tuple[float, float, str]] = {
    "MX4926 carbon cloth phenolic": (
        1e-21,
        1e-17,
        "Park & Lawrence, AIAA 2003-5242, measured 22-260 C. Dense RSRM "
        "nozzle liner near 1.45 g/cm3 -- NOT PICA, and not a substitute "
        "for it.",
    ),
    "PICA (placeholder bracket)": (
        1e-12,
        1e-10,
        "NOT MEASURED. An order-of-magnitude bracket for a ~90% porous "
        "fibrous preform, used only to sweep. The measured value is in "
        "Marschall & Milos, 'Gas Permeability of Rigid Fibrous Refractory "
        "Insulations', which this repository does not hold.",
    ),
}


@dataclass(frozen=True)
class PorePressureProfile:
    """Pore pressure through the stack, and where it peaks."""

    depth: _FloatArray
    """Distance from the heated surface (m), increasing inward."""
    pressure: _FloatArray
    """Pore gas pressure (Pa) at each depth."""
    surface_pressure: float
    permeability: float

    @property
    def peak_pressure(self) -> float:
        return float(np.max(self.pressure))

    @property
    def peak_ratio(self) -> float:
        """Peak pore pressure over surface pressure.

        The single number that says whether feeding surface pressure to a
        pressure-dependent property is defensible. At 1.0 it is exact; the
        further above, the worse the substitution.
        """
        return self.peak_pressure / self.surface_pressure

    @property
    def peak_depth(self) -> float:
        return float(self.depth[int(np.argmax(self.pressure))])


def pore_pressure(
    depth: ArrayLike,
    gas_source: ArrayLike,
    temperature: ArrayLike,
    surface_pressure: float,
    permeability: float,
    viscosity: float = 4.0e-5,
    molar_mass: float = 0.0103,
) -> PorePressureProfile:
    """Pore pressure profile from a pyrolysis gas source distribution.

    Parameters
    ----------
    depth:
        Distance from the heated surface (m), increasing inward and
        strictly increasing. ``depth[0]`` is the surface.
    gas_source:
        Volumetric pyrolysis gas generation rate (kg/m³/s) at each depth,
        i.e. :math:`-\\partial \\rho / \\partial t` from the solid.
    temperature:
        Local temperature (K) at each depth.
    surface_pressure:
        Boundary-layer edge pressure (Pa), the outflow boundary condition.
    permeability:
        Darcy permeability (m²). **No default** — see the module docstring
        for why, and :data:`PORE_PRESSURE_REFERENCES` for what is and is
        not measured.
    viscosity:
        Pyrolysis gas dynamic viscosity (Pa s). The default is a
        representative high-temperature value; the result depends on it
        only linearly and through the same combination as permeability, so
        the two cannot be separated by this model.
    molar_mass:
        Gas molar mass (kg/mol). The default is what the equilibrium solve
        returns for PICA pyrolysis gas at 2000 K, which is light because
        the mixture is mostly H2.

    Notes
    -----
    Quasi-steady: the gas is assumed to leave as fast as it is made, with
    no storage term. That is the standard treatment and it is good wherever
    the pyrolysis front moves slowly against the gas transit time, which is
    everywhere except the first instants of a step heat load.
    """
    x = np.asarray(depth, dtype=np.float64)
    source = np.asarray(gas_source, dtype=np.float64)
    temp = np.asarray(temperature, dtype=np.float64)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("depth must be a 1-D array with at least two points")
    if source.shape != x.shape or temp.shape != x.shape:
        raise ValueError(
            f"gas_source and temperature must match depth: got "
            f"{source.shape} and {temp.shape} against {x.shape}"
        )
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("depth must be strictly increasing from the surface")
    if np.any(source < 0.0):
        raise ValueError(
            "gas_source must be non-negative; a negative source is gas being "
            "absorbed by the solid, which pyrolysis does not do"
        )
    if not (np.isfinite(surface_pressure) and surface_pressure > 0.0):
        raise ValueError(f"surface_pressure must be finite and > 0, got {surface_pressure}")
    for name, value in (
        ("permeability", permeability),
        ("viscosity", viscosity),
        ("molar_mass", molar_mass),
    ):
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be finite and > 0, got {value}")

    # Accumulated mass flux at each depth: everything generated *below*
    # this point has already passed through it on its way out, so the flux
    # is the reverse cumulative integral of the source. Zero at the back
    # face by construction, which is the impermeable-backface condition.
    widths = np.diff(x)
    generated = 0.5 * (source[:-1] + source[1:]) * widths
    # The zero belongs at the *back* face, not the surface: nothing has
    # passed through the deepest node, and the flux grows outward as the
    # flow gathers everything decomposing behind it. Putting it at the
    # surface instead inverts the profile.
    flux = np.concatenate([np.cumsum(generated[::-1])[::-1], [0.0]])

    # Integrate d(P^2)/dx inward. The integrand is evaluated at midpoints
    # and accumulated, so P^2 is exact for a piecewise-linear source.
    coefficient = 2.0 * _R_UNIVERSAL * viscosity / (molar_mass * permeability)
    integrand = coefficient * flux * temp
    midpoint_integrand = 0.5 * (integrand[:-1] + integrand[1:])
    squared = np.concatenate(
        [[surface_pressure**2], surface_pressure**2 + np.cumsum(midpoint_integrand * widths)]
    )
    return PorePressureProfile(
        depth=x,
        pressure=np.sqrt(squared),
        surface_pressure=float(surface_pressure),
        permeability=float(permeability),
    )


def pore_pressure_sensitivity(
    depth: ArrayLike,
    gas_source: ArrayLike,
    temperature: ArrayLike,
    surface_pressure: float,
    permeabilities: ArrayLike,
    **kwargs: float,
) -> dict[float, float]:
    """Peak-to-surface pressure ratio against permeability.

    The question a missing measurement actually raises is not "what is the
    number" but "does the answer depend on it". Returns the peak ratio for
    each permeability, so the sweep can be read directly: a ratio near 1
    everywhere means feeding surface pressure to a pressure-dependent
    property was harmless, and a ratio that climbs means it was not.
    """
    values = np.atleast_1d(np.asarray(permeabilities, dtype=np.float64))
    return {
        float(k): pore_pressure(
            depth, gas_source, temperature, surface_pressure, float(k), **kwargs
        ).peak_ratio
        for k in values
    }
