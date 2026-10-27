import logging

from flag_gems.runtime.backend._kunlunxin.ops.matmuladd import matmuladd  # noqa: F401

logger = logging.getLogger(__name__)

# NOTE (kunlunxin): this module is the measured path for `flag_gems.matmuladd`
# -- same-named entries in the vendor `fused` package shadow the `ops` package
# (`runtime/backend/__init__.py::get_customized_ops` appends fused after ops).
# Upstream's `d312aa024` filled it with a delegate to addmm, which predates
# `ops/matmuladd.py` (introduced by this branch). The wide-N tile implementation
# there is measurably faster on the official core shapes: balanced 0.887 vs
# 0.70 for the delegate (2026-09-18, 3x fresh cache, dev4; evidence:
# artifacts/op-perf-batch-2026-09/evidence/matmuladd-ops-path-20260918/).
# Re-export it so the measured path uses the faster implementation.
