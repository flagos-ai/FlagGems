# Kunlunxin (XPU) override of `negative_`.
#
# `negative_` is the in-place alias of `neg_` (x = -x, no allocation, no
# numerical rounding: pure sign flip). The generic `flag_gems.ops.negative_`
# binds the *generic* `ops/neg.neg_` at import time, so it never reaches this
# backend's tuned neg kernel. On XPU the generic neg path is a discrete/unvec
# masked kernel: ~2.6ns/element (16.7M fp32 = ~39ms) vs the kunlunxin neg_
# (CodeGenConfig tile 65536, unroll 8) at ~0.086ms and torch native ~0.15ms.
# Alias semantics: input and output are the same tensor, handled by
# `neg_func(A, out0=A)` exactly like the in-place `neg_` override.
import logging

from .neg import neg_func

logger = logging.getLogger(__name__)


def negative_(A):
    logger.debug("GEMS_KUNLUNXIN NEGATIVE_")
    return neg_func(A, out0=A)