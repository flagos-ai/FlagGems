# Kunlunxin (XPU) override of `negative`.
#
# `negative` is functionally identical to `neg` (both compute -x), and kunlunxin
# already ships a tuned override for neg (`_kunlunxin/ops/neg.py`, a
# `pointwise_dynamic` with the XPU-tuned CodeGenConfig). But `negative` was NOT
# overridden, so it fell back to the generic hand-written kernel
# (`ops/negative.py`) with a fixed `BLOCK_SIZE=1024` + `grid=cdiv(n,1024)` and no
# XPU tuning -> launch-bound / discrete slow path on large shapes (IR baseline
# `harness/perf_ir_3/ir-negative-dev5.log`: large shapes gems 0.04-0.17, plus
# per-shape first-compile warmup spikes to 240-390ms).
#
# Fix: reuse the exact tuned neg recipe by delegating to it. Zero algorithm
# change (both are -x), zero correctness risk.
import logging

from .neg import neg_func

logger = logging.getLogger(__name__)


def negative(A):
    logger.debug("GEMS_KUNLUNXIN NEGATIVE")
    return neg_func(A)
