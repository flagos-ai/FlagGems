"""bitwise_or__opt: retired C05 GSL layer, now a thin alias.

The grid-stride-loop candidate below was superseded on 2026-11-13 by the flat
grid-stride ladder in ``bitwise_or.py`` (ops/__init__.py imports
``bitwise_or_scalar_`` from there since the 2026-11-16 round-3 repoint); the
GSL wrapper predated the flat ladder, kept bool loads on the slow i1 lowering
and measured 0.47-0.58x on the core bool buckets.  Delegating keeps this
module's public name on the live fast path so any future re-wiring inherits
the current kernels instead of resurrecting the stale candidate.
"""

from flag_gems.runtime.backend._enflame.gcu400.ops.bitwise_or import bitwise_or_scalar_

__all__ = ["bitwise_or_scalar_"]
