import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def reciprocal_func(x):
    y = 1.0 / x.to(tl.float32)
    # The mthreads backend saturates fp16 results that exceed the finite
    # range to 65504.0 instead of producing +/-inf like PyTorch, so restore
    # the IEEE-754 overflow semantics for float16 outputs.
    if tl.constexpr(x.dtype == tl.float16):
        pos_overflow = y >= 65504.0
        neg_overflow = y <= -65504.0
        y = tl.where(
            pos_overflow,
            float("inf"),
            tl.where(neg_overflow, float("-inf"), y),
        )
    return y


def reciprocal(A):
    logger.debug("GEMS_MTHREADS RECIPROCAL")
    return reciprocal_func(A)


def reciprocal_(A):
    logger.debug("GEMS_MTHREADS RECIPROCAL_")
    return reciprocal_func(A, out0=A)
