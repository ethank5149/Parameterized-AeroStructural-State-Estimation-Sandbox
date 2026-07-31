"""Executable verification tasks from Paper I, §8.

Each module runs one verification task against the acceptance criterion
stated in the paper *before any results existed*, and writes a markdown
report plus machine-readable CSV into a results directory:

- :mod:`passes.verification.v1_structural` — V1: conditioning of the
  reduced stiffness operator versus :math:`N`; free-free frequencies
  against the analytic uniform-beam solution.
- :mod:`passes.verification.v3_integrators` — V3: achieved step size and
  wall clock for explicit, modally truncated, and IMEX strategies.
- :mod:`passes.verification.v6_tgo` — V6: the stable time-to-go form
  against the textbook form in double and single precision as
  :math:`\\hat{A}_c \\to 0`.

Run all three with ``python -m passes.verification``.
"""

from __future__ import annotations

from passes.verification.common import VerificationReport

__all__ = ["VerificationReport"]
