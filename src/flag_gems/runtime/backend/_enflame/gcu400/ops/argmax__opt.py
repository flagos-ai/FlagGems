"""argmax entry point (the `__opt` module name is kept and imported by ops/__init__.py).

History: this file started as a tuning candidate that left argmax.py untouched and called the
(inner: G48/BM8/TILE_N=npo2(N)/nw4; non_inner: TILE_K=512/TILE_N=64/nw4).

Since 2026-11-17 both kernels were rewritten for correctness defects (see argmax.py comments):
  * tl.max(return_indices=True) on 2-D tiles misplaces indices when K spans multiple tiles;
  * a fixed 48-CTA grid-stride drove the error rate close to 100%;
  * computed int64 index vectors were stored with int32 strides.
The rewritten kernels use a 1-D row loop and int32 indices, so the signature no longer carries
TILE_N/BLOCK_M/GRID_SIZE/ONE_TILE_PER_CTA and this module's explicit call went stale.
The tuning knobs converged into argmax.py's `_inner_config` / `_non_inner_config`
(overridable via FG_ARGMAX_INNER_TILE / FG_ARGMAX_NON_INNER_TILE), so this simply delegates to
the production implementation, avoiding a TypeError from an entry pointing at a deleted signature.
"""

from flag_gems.runtime.backend._enflame.gcu400.ops.argmax import argmax

__all__ = ["argmax"]
