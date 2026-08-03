"""PICA surface thermochemistry, digitised from published ACE output.

Quinn, Pickard, Bernstein, Yee, Koo & Radovitzky, "Validation of a Charring
Ablator Material Response Code Against Oxyacetylene Torch Experiments on
PICA Samples," publish an ACE-generated :math:`B'` table for **PICA** as
their Fig. 5 — :math:`B'_c` against wall temperature in Fig. 5a and wall
enthalpy in Fig. 5b, each at five pyrolysis-gas rates.

This is the first surface chemistry in this package that is PICA rather
than a surrogate. The Mutation++ path in
:mod:`passes.thermal.fiat.mutationpp` generates tables for TACOT, the
open stand-in that exists because real ablator decks are restricted; this
one is the material itself.

Provenance and its limits
-------------------------

The data is **digitised from a figure**, and that sets a floor on what it
can support:

* Reading a curve off a log-scaled axis carries a few percent of its own
  error, which is comparable to the 5% criteria such tables get used to
  test. Any comparison made with this table has to state that.
* The tabulated :math:`B'_g` levels are 0.0001, 0.0005, 0.001, 0.005 and
  0.01 — which the paper notes "effectively coalesce into a single curve"
  and which are therefore transcribed once — plus 0.05, 0.1, 0.5 and 1.0.
  The coalesced curve is carried at both 0 and 0.01, since a run begins
  with no pyrolysis gas and the source's own statement covers the
  interval.
* There is **one pressure**. The oxyacetylene torch runs at ambient, and
  the figure carries no pressure parameter, so the table is built as
  explicitly pressure-independent rather than pretending to a dependence
  it does not have.
* The two figures do not span the same temperature range: :math:`h_w` is
  drawn from 500 K, :math:`B'_c` only from about 1030 K. Below the
  :math:`B'_c` curve's start there is no ablation to speak of, so it is
  continued at its lowest digitised value rather than extrapolated.

Ordinates in Fig. 5a run over four decades, so the curve is resampled in
:math:`\\log B'_c`; Fig. 5b is linear in MJ/kg and is resampled directly.
Both are monotone-smoothed first, because tracing a plotted line produces
several ordinates at the same abscissa and a spline through raw digitiser
output oscillates between them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from passes.thermal.fiat.bprime import BPrimeTable

__all__ = [
    "QUINN_GAS_RATES",
    "PicaSurfaceTable",
    "read_quinn_bprime",
]

_FloatArray = NDArray[np.float64]

#: The :math:`B'_g` values of Quinn et al. Fig. 5, ascending.
#:
#: The figure draws six curves at five distinct values, because
#: :math:`B'_g = 0.0001` through :math:`0.01` — the paper's words —
#: "effectively coalesce into a single curve".
#:
#: That coalescence is what licenses the zero node. A run starts with no
#: pyrolysis gas at all, so the solver queries :math:`B'_g \to 0` on its
#: first step; without a node there the table's no-extrapolation guard
#: fires on a state that is physically the most ordinary one there is.
#: Since the source states the curve is unchanged across two decades
#: below 0.01, continuing it to zero asserts nothing the figure does not
#: already show. The coalesced curve is therefore placed at **both** 0
#: and 0.01, and the interval between them is flat by construction rather
#: than by interpolation.
QUINN_GAS_RATES = (0.0, 0.01, 0.05, 0.1, 0.5, 1.0)

_MJ_TO_J = 1.0e6


@dataclass(frozen=True)
class PicaSurfaceTable:
    """A digitised PICA B' table with its provenance attached."""

    table: BPrimeTable
    temperatures: _FloatArray
    gas_rates: _FloatArray
    char_rates: _FloatArray
    """Shape ``(n_gas_rates, n_temperatures)``."""
    wall_enthalpies: _FloatArray
    provenance: str

    @property
    def digitisation_uncertainty(self) -> float:
        """Indicative relative uncertainty of a figure-traced ordinate.

        Not measured — a stated assumption, carried so that anything
        comparing against this table has a number to quote rather than an
        implicit claim of exactness.
        """
        return 0.05


def _monotone_curve(
    raw: _FloatArray, grid: _FloatArray, logarithmic: bool
) -> _FloatArray:
    """Average duplicate abscissae, enforce monotonicity, resample.

    Digitiser output has several ordinates per abscissa where the traced
    line is steep. Averaging them and then enforcing a non-decreasing
    ordinate removes the tracing jitter without imposing a functional
    form; both :math:`B'_c` and :math:`h_w` are physically monotone in
    wall temperature over this range.
    """
    order = np.argsort(raw[:, 0])
    x, y = raw[order, 0], raw[order, 1]
    # Collapse repeated abscissae to their mean.
    unique_x, inverse = np.unique(np.round(x, 6), return_inverse=True)
    mean_y = np.zeros(unique_x.size)
    np.add.at(mean_y, inverse, y)
    counts = np.bincount(inverse, minlength=unique_x.size)
    mean_y /= counts
    if logarithmic:
        mean_y = np.log(np.maximum(mean_y, 1e-30))
    mean_y = np.maximum.accumulate(mean_y)
    out = np.interp(grid, unique_x, mean_y, left=mean_y[0], right=mean_y[-1])
    return np.asarray(np.exp(out) if logarithmic else out)


def read_quinn_bprime(
    directory: str | Path,
    *,
    temperatures: _FloatArray | None = None,
    pressure: float = 101325.0,
    method: str = "cubic",
) -> PicaSurfaceTable:
    """Assemble the PICA table from the digitised Fig. 5a and 5b files.

    Parameters
    ----------
    directory:
        Holds ``Quinn-et-al-Fig5a_Bprime_g=*.csv`` and the matching 5b
        files, each a headerless ``temperature, value`` CSV. In this
        repository that is ``reference/transcribed-data``.
    temperatures:
        Output temperature axis (K). Defaults to 250–3800 K in 25 K
        steps, which covers a torch run from ambient to just short of the
        figure's upper end. The default deliberately starts below room
        temperature: a table whose lower bound is exactly the initial
        condition trips its own no-extrapolation guard on the first step.
    pressure:
        Recorded on the table. The figure carries no pressure parameter,
        so the result is pressure-independent by construction.
    """
    root = Path(directory)
    grid = (
        np.arange(250.0, 3801.0, 25.0)
        if temperatures is None
        else np.asarray(temperatures, dtype=np.float64)
    )
    if grid.ndim != 1 or grid.size < 4 or np.any(np.diff(grid) <= 0.0):
        raise ValueError("temperatures must be strictly increasing with >= 4 points")

    b_c = np.zeros((len(QUINN_GAS_RATES), grid.size))
    h_w = np.zeros_like(b_c)
    for i, rate in enumerate(QUINN_GAS_RATES):
        coalesced = rate <= 0.01
        a_path = _find(root, "Fig5a", rate, coalesced)
        b_path = _find(root, "Fig5b", rate, coalesced)
        b_c[i] = _monotone_curve(
            np.loadtxt(a_path, delimiter=","), grid, logarithmic=True
        )
        h_w[i] = _monotone_curve(
            np.loadtxt(b_path, delimiter=","), grid, logarithmic=False
        ) * _MJ_TO_J

    table = BPrimeTable(
        [pressure],
        np.asarray(QUINN_GAS_RATES),
        grid,
        b_c[None, :, :],
        h_w[None, :, :],
        method=method,
    )
    return PicaSurfaceTable(
        table=table,
        temperatures=grid,
        gas_rates=np.asarray(QUINN_GAS_RATES),
        char_rates=b_c,
        wall_enthalpies=h_w,
        provenance=(
            "PICA, ACE-generated, digitised from Quinn et al., 'Validation of a "
            "Charring Ablator Material Response Code Against Oxyacetylene Torch "
            "Experiments on PICA Samples', Fig. 5a and 5b. Single pressure "
            f"({pressure:.6g} Pa, ambient); figure-traced, so ordinates carry "
            "several percent of digitisation error."
        ),
    )


def _find(root: Path, figure: str, rate: float, coalesced: bool) -> Path:
    """Locate the CSV for one curve, tolerating the transcription naming."""
    if coalesced:
        matches = sorted(root.glob(f"Quinn-et-al-{figure}_Bprime_g=0.01+*.csv"))
        if matches:
            return matches[0]
    pattern = f"Quinn-et-al-{figure}_Bprime_g={rate:g}.csv"
    path = root / pattern
    if path.exists():
        return path
    # ``0.5`` may have been written ``0.50``; fall back to a numeric match.
    for candidate in root.glob(f"Quinn-et-al-{figure}_Bprime_g=*.csv"):
        text = re.search(r"g=([0-9.]+)(?:\.csv|\+)", candidate.name)
        if text and abs(float(text.group(1)) - rate) < 1e-12:
            return candidate
    raise FileNotFoundError(
        f"no digitised curve for {figure} at B'_g = {rate:g} under {root}"
    )
