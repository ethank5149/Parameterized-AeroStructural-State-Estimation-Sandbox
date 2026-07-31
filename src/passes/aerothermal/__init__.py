"""Hypersonic aerothermodynamic correlations (Paper II, §4).

Stagnation convective heating by Fay–Riddell with the Lewis exponent
stated (Eq. 4.1), the modified-Newtonian stagnation velocity gradient
(Eq. 4.2), the Sutton–Graves correlation kept explicitly distinct as the
screening fallback, Tauber–Sutton shock-layer radiation with its
tabulated velocity function as a required input (Eq. 4.4), the Lees
distribution over the planform with stagnation-region matching
(Eq. 4.5), and the single-temperature recession balance for
non-pyrolyzing leading edges (Eq. 4.6). All quantities SI.
"""

from __future__ import annotations

from passes.aerothermal.distribution import lees_distribution
from passes.aerothermal.radiative import TauberSuttonRadiation
from passes.aerothermal.recession import stefan_recession_rate
from passes.aerothermal.stagnation import (
    fay_riddell,
    newtonian_velocity_gradient,
    sutton_graves,
)

__all__ = [
    "TauberSuttonRadiation",
    "fay_riddell",
    "lees_distribution",
    "newtonian_velocity_gradient",
    "stefan_recession_rate",
    "sutton_graves",
]
