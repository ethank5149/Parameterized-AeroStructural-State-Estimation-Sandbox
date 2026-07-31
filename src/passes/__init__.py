"""PASSES — Parameterized AeroStructural State-Estimation Sandbox.

Implementation of the fixed-grid spectral formulation of Knox (2026),
Paper I (``passes-updated.tex``). This package currently covers roadmap
items 1 and 2:

- :mod:`passes.spectral` — Chebyshev–Gauss–Lobatto collocation operators
  built by the direct recurrence with the negative-sum trick (Paper I,
  Appendix A), Clenshaw–Curtis quadrature, and barycentric interpolation.
- :mod:`passes.structures` — the variable-rigidity Euler–Bernoulli
  stiffness operator (Paper I, Eq. 3.5), free-free boundary conditions by
  null-space projection (Paper I, §3.2), the reduced generalized
  eigenproblem, and the temporal-integration strategies of Paper I, §3.6.
- :mod:`passes.guidance` — the numerically stable time-to-go root with
  the non-intercept guard (Paper I, §4.3) and the AC-APN command law.
- :mod:`passes.verification` — executable verification tasks V1, V3, V6
  from Paper I, §8.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
