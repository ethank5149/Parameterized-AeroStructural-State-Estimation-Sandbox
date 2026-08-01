"""Published PICA pyrolysis kinetics, and why FIAT's Eq. (8) cannot hold them all.

Two calibrated models from Torres-Herrador and co-workers are
implemented here, together with the conversion needed to express either
in FIAT's rate normalisation.

Sources, both in ``reference/``:

* **[TH2020]** Torres-Herrador, Coheur, Panerai, Magin, Arnst, Mansour &
  Blondeau, "Competitive kinetic model for the pyrolysis of the Phenolic
  Impregnated Carbon Ablator," *Aerospace Science and Technology* **100**
  (2020) 105784.
* **[TH2019]** Torres-Herrador, Meurisse, Panerai, Blondeau, Lachaud,
  Bessire, Magin & Mansour, "A high heating rate pyrolysis model for the
  Phenolic Impregnated Carbon Ablator (PICA) based on mass spectroscopy
  experiments," *J. Analytical and Applied Pyrolysis* **141** (2019)
  104625.

The structural finding
----------------------

FIAT Eq. (8) models pyrolysis as **independent parallel reactions**, one
per solid component. [TH2020] states plainly that this form cannot
reproduce PICA's measured behaviour across heating rates:

    "In solid-phase pyrolysis, it is usually observed that as the heating
    rate increases, the decomposition curves shift towards higher
    temperatures. This behavior is commonly attributed to the thermal lag
    effects and can be usually reproduced assuming independent parallel
    reactions. However, different experimental evidences show that this
    is not the case for the pyrolysis of carbon/phenolic. For example,
    Stokes observed that at heating rates higher than 300 K/min the
    pyrolysis peak shifted towards *lower* temperatures."

and, of the parallel formulation, that it is

    "not able to reproduce this effect due to their mathematical
    formulation."

This is not a calibration problem. A sum of independent first-order-ish
Arrhenius terms always shifts its peak *up* with heating rate, because
each term does. Reproducing a downward shift requires two reactions
**competing for the same reactant**, so that the faster,
higher-activation-energy path steals reactant from the slower one as the
rate climbs.

It matters for entry rather than for laboratory work. [TH2020] notes
that across the MSL heat shield "values as high as 60000 K/min and as low
as 60 K/min can be found", while "most of flight heating rates are
outside the realm of legacy TGA measurements used for calibration, rarely
exceeding tens of K/min". A parallel model calibrated at 10 K/min is
being extrapolated three or four decades.

:func:`competitive_mass_fraction` integrates [TH2020]'s scheme, which is outside
FIAT Eq. (8)'s model form and is provided as an independent integrator.
:func:`parallel_pica_resin` implements [TH2019]'s six-reaction parallel
set, which *is* in Eq. (8)'s form and drops straight into the solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.integrate
from numpy.typing import ArrayLike, NDArray

from passes.thermal.material import GAS_CONSTANT, ArrheniusComponent

__all__ = [
    "COMPETITIVE_PICA_BAYESIAN",
    "COMPETITIVE_PICA_DETERMINISTIC",
    "PARALLEL_PICA_RESIN",
    "CompetitivePica",
    "ParallelReaction",
    "advancement_to_fiat_rate",
    "competitive_mass_fraction",
    "parallel_pica_resin",
]

_FloatArray = NDArray[np.float64]


# --------------------------------------------------------------------------
# [TH2020] competitive scheme
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CompetitivePica:
    """[TH2020] competitive mechanism for PICA pyrolysis.

    The reaction network of its Fig. 4 and Eq. (22):

    .. code-block:: text

        rho_1  --k11-->  rho_2*  --k21-->  (1-g5) rho_4  +  g5 rho_5^gas
          |
          +----k12-->    rho_3*  --k31-->  (1-g7) rho_6  +  g7 rho_7^gas

    ``k11`` is the slow, low-activation-energy path that dominates at low
    heating rate; ``k12`` is the fast, high-activation-energy path that
    takes over at high heating rate and *starves* the first branch. That
    competition for the shared reactant ``rho_1`` is the whole mechanism,
    and it is what produces the downward peak shift.

    Attributes are :math:`\\log_{10} A` (s⁻¹) and :math:`E` (J/mol), as
    tabulated, plus the two independent gas mass coefficients. Mass
    conservation fixes the solid coefficients: [TH2020] states
    :math:`\\gamma_{i,j,l+1} = 1 - \\gamma_{i,j,l}`.
    """

    log10_a11: float
    e11: float
    log10_a12: float
    e12: float
    log10_a21: float
    e21: float
    log10_a31: float
    e31: float
    gamma_gas_2: float
    """:math:`\\gamma_{2,1,5}`, gas fraction of the low-rate branch."""
    gamma_gas_3: float
    """:math:`\\gamma_{3,1,7}`, gas fraction of the high-rate branch."""
    provenance: str = ""

    def __post_init__(self) -> None:
        for name in ("gamma_gas_2", "gamma_gas_3"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1), got {value}")
        if self.e11 >= self.e12:
            raise ValueError(
                "the competitive mechanism requires E11 < E12: the slow branch "
                "must start earlier for the fast branch to take over as the "
                "heating rate rises. Got "
                f"E11={self.e11:.6g}, E12={self.e12:.6g}"
            )

    def rates(self, temperature: float) -> tuple[float, float, float, float]:
        """``(k11, k12, k21, k31)`` at ``temperature`` (1/s)."""
        rt = GAS_CONSTANT * float(temperature)
        k = [
            10.0**log_a * float(np.exp(-e / rt))
            for log_a, e in (
                (self.log10_a11, self.e11),
                (self.log10_a12, self.e12),
                (self.log10_a21, self.e21),
                (self.log10_a31, self.e31),
            )
        ]
        return k[0], k[1], k[2], k[3]

    def char_yield_limits(self) -> tuple[float, float]:
        """Char yield in the slow-branch and fast-branch limits.

        A consequence of the mechanism worth noticing: because the two
        branches have different gas coefficients, **char yield is a
        function of heating rate**. FIAT Eq. (8) makes it a constant of
        the material.
        """
        return 1.0 - self.gamma_gas_2, 1.0 - self.gamma_gas_3


#: [TH2020] Table 1 — deterministic optimisation. E in J/mol.
COMPETITIVE_PICA_DETERMINISTIC = CompetitivePica(
    log10_a11=2.019,
    e11=32618.482,
    log10_a12=14.292,
    e12=143273.910,
    log10_a21=0.442,
    e21=51783.980,
    log10_a31=0.993,
    e31=31087.851,
    gamma_gas_2=0.163,
    gamma_gas_3=0.244,
    provenance="Torres-Herrador et al. 2020, Table 1 (deterministic optimisation)",
)

#: [TH2020] Table 2 — posterior means from Bayesian inference. E in J/mol.
COMPETITIVE_PICA_BAYESIAN = CompetitivePica(
    log10_a11=2.4768,
    e11=26811.37,
    log10_a12=23.4935,
    e12=183938.42,
    log10_a21=0.2219,
    e21=48796.41,
    log10_a31=1.1969,
    e31=33566.43,
    gamma_gas_2=0.1648,
    gamma_gas_3=0.3190,
    provenance="Torres-Herrador et al. 2020, Table 2 (Bayesian posterior mean)",
)

#: [TH2020] Table 2 — posterior standard deviations, same ordering.
#:
#: Kept because the paper's own conclusion is that two of these are badly
#: identified: the coefficient of variation on :math:`A_{2,1}` is 0.56,
#: and the paper attributes the high correlation between :math:`A` and
#: :math:`E` for reactions (2,1) and (3,1) to "the kinetic compensation
#: effect". Propagating the means alone would hide that.
COMPETITIVE_PICA_UNCERTAINTY = {
    "log10_a11": 0.3027,
    "e11": 893.61,
    "log10_a12": 1.1618,
    "e12": 2369.64,
    "log10_a21": 0.1238,
    "e21": 1723.16,
    "log10_a31": 0.0821,
    "e31": 976.07,
    "gamma_gas_2": 0.0038,
    "gamma_gas_3": 0.0703,
}


def competitive_mass_fraction(
    model: CompetitivePica,
    temperatures: ArrayLike,
    heating_rate: float,
    initial_temperature: float = 300.0,
) -> _FloatArray:
    """Residual solid mass fraction of a constant-rate scan, [TH2020] Eq. (22).

    Integrates the seven-species network along
    :math:`T = T_0 + \\beta\\theta` and returns the solid fraction
    :math:`\\rho_1 + \\rho_2^* + \\rho_3^* + \\rho_4 + \\rho_6`, normalised
    to the initial reactant.
    """
    t = np.asarray(temperatures, dtype=np.float64)
    if t.ndim != 1 or t.size < 2 or np.any(np.diff(t) <= 0.0):
        raise ValueError("temperatures must be strictly increasing with >= 2 points")
    if not (np.isfinite(heating_rate) and heating_rate > 0.0):
        raise ValueError(f"heating_rate must be finite and > 0, got {heating_rate}")
    if t[0] < initial_temperature:
        raise ValueError("temperatures must start at or above initial_temperature")

    g5, g7 = model.gamma_gas_2, model.gamma_gas_3

    def rhs(temp: float, y: _FloatArray) -> _FloatArray:
        k11, k12, k21, k31 = model.rates(temp)
        r1, r2, r3 = y[0], y[1], y[2]
        return (
            np.array(
                [
                    -(k11 + k12) * r1,
                    k11 * r1 - k21 * r2,
                    k12 * r1 - k31 * r3,
                    (1.0 - g5) * k21 * r2,
                    g5 * k21 * r2,
                    (1.0 - g7) * k31 * r3,
                    g7 * k31 * r3,
                ]
            )
            / heating_rate
        )

    solution = scipy.integrate.solve_ivp(
        rhs,
        (initial_temperature, float(t[-1])),
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        t_eval=t,
        method="LSODA",
        rtol=1e-10,
        atol=1e-13,
    )
    if not solution.success:  # pragma: no cover - integrator failure
        raise RuntimeError(f"competitive integration failed: {solution.message}")
    solid = solution.y[[0, 1, 2, 3, 5], :].sum(axis=0)
    return np.asarray(solid)


# --------------------------------------------------------------------------
# [TH2019] parallel scheme — FIAT Eq. (8) compatible
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ParallelReaction:
    """One row of [TH2019] Table 2.

    Attributes
    ----------
    density_loss_fraction:
        :math:`F_{i,j}`, "the fraction of density that is lost when
        reaction :math:`R_{i,j}` reaches completion" ([TH2019] Eq. 5).
    log10_a:
        :math:`\\log_{10} A` (s⁻¹).
    activation_energy_kj:
        :math:`E` (kJ/mol), **as tabulated** — note the unit.
    order:
        :math:`n`, the reaction order.
    """

    density_loss_fraction: float
    log10_a: float
    activation_energy_kj: float
    order: float


#: [TH2019] Table 2 — species-based model, six parallel reactions.
#:
#: Calibrated against the high-heating-rate mass-spectrometry data of
#: Bessire and Minton. The activation energies are in **kJ/mol** in the
#: source and are converted on use; the reaction orders are unusually
#: high (4.2 to 10), which is characteristic of a lumped multi-species
#: fit and is faithful to the table.
PARALLEL_PICA_RESIN = (
    ParallelReaction(0.060, 6.59, 77.6, 5.65),
    ParallelReaction(0.009, 6.96, 61.3, 9.96),
    ParallelReaction(0.203, 6.71, 95.1, 4.23),
    ParallelReaction(0.187, 6.67, 103.0, 4.38),
    ParallelReaction(0.026, 6.58, 113.9, 6.68),
    ParallelReaction(0.059, 6.35, 175.2, 8.85),
)


def advancement_to_fiat_rate(
    log10_a: float, order: float, virgin_density: float, char_density: float
) -> float:
    """Convert an advancement-form pre-exponential to FIAT's normalisation.

    The two literatures normalise the rate differently, and the
    difference is a pure power of the decomposable fraction — silent,
    dimensionally invisible, and worth several orders of magnitude at the
    reaction orders [TH2019] reports.

    [TH2019] and the biomass literature write the rate in terms of
    reaction advancement :math:`\\chi`:

    .. math::

        \\frac{d\\chi}{dt} = A_P e^{-E/RT}(1-\\chi)^n, \\qquad
        \\rho = \\rho_v - \\chi(\\rho_v - \\rho_r).

    FIAT Eq. (8) writes it against the *virgin* density:

    .. math::

        \\frac{d\\rho}{dt} = -A_F e^{-E/RT}\\rho_v
        \\left(\\frac{\\rho - \\rho_r}{\\rho_v}\\right)^{n}.

    Equating the two gives

    .. math::

        A_F = A_P\\left(\\frac{\\rho_v}{\\rho_v - \\rho_r}\\right)^{n-1}.

    At :math:`n = 1` they coincide, which is why the trap only bites for
    the high-order fits.
    """
    if not char_density < virgin_density:
        raise ValueError("need char_density < virgin_density")
    if not virgin_density > 0.0:
        raise ValueError("virgin_density must be > 0")
    ratio = virgin_density / (virgin_density - char_density)
    return float(10.0**log10_a * ratio ** (order - 1.0))


def parallel_pica_resin(
    resin_density: float,
    reactions: tuple[ParallelReaction, ...] = PARALLEL_PICA_RESIN,
) -> list[ArrheniusComponent]:
    """[TH2019] Table 2 as FIAT Eq. (8) components.

    Each tabulated reaction becomes one
    :class:`~passes.thermal.material.ArrheniusComponent` carrying
    :math:`F_{i,j}` of the resin mass, with its char density set so the
    component loses exactly that fraction, and its pre-exponential
    converted through :func:`advancement_to_fiat_rate`.

    Parameters
    ----------
    resin_density:
        Mass of decomposing resin per unit volume of the *resin phase*
        (kg/m³). [TH2019]'s :math:`F` values sum to 0.544, so this model
        describes the resin, not the composite: applied to PICA's
        published 94 kg/m³ of phenolic in 274 kg/m³ of material it
        implies a composite mass loss of
        :math:`0.544 \\times 94/274 = 18.7\\%`, against the
        :math:`(274-227)/274 = 17.2\\%` implied by the published virgin
        and char bulk densities. Those agree to about a percent and a
        half, which is a genuine cross-check between two unrelated
        sources — but the identification of :math:`F` with a
        resin-normalised fraction is an inference from Eq. (5), not a
        statement the paper makes in words.
    """
    if not (np.isfinite(resin_density) and resin_density > 0.0):
        raise ValueError(f"resin_density must be finite and > 0, got {resin_density}")
    total = sum(r.density_loss_fraction for r in reactions)
    if not 0.0 < total < 1.0:
        raise ValueError(f"density loss fractions must sum into (0, 1), got {total}")

    components: list[ArrheniusComponent] = []
    for r in reactions:
        # This component holds the share of resin mass its own F represents,
        # and loses all of it; splitting the resin by F rather than giving
        # every component the whole resin is what makes the F values mean
        # what Eq. (5) says they mean.
        share = r.density_loss_fraction / total
        virgin = resin_density * share
        # Each reaction runs to completion, so the residual is what the
        # reaction does not volatilise; with the F-weighted split above that
        # is the non-decomposing remainder of this component's share.
        char = virgin * (1.0 - total)
        components.append(
            ArrheniusComponent(
                pre_exponential=advancement_to_fiat_rate(
                    r.log10_a, r.order, virgin, char
                ),
                activation_energy=r.activation_energy_kj * 1.0e3,
                reaction_order=r.order,
                virgin_density=virgin,
                char_density=char,
            )
        )
    return components
