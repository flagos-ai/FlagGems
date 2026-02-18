import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# logaddexp: log(exp(x) + exp(y))
# Numerically stable form: max(x,y) + log(exp(x-max) + exp(y-max))
# ---------------------------------------------------------------------------
@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def logaddexp_func(x, y):
    x = x.to(tl.float32)
    y = y.to(tl.float32)
    mx = tl.maximum(x, y)
    return mx + tl.log(tl.exp(x - mx) + tl.exp(y - mx))


def logaddexp(A, B):
    logger.debug("GEMS LOGADDEXP")
    return logaddexp_func(A, B)


def logaddexp_(A, B):
    logger.debug("GEMS LOGADDEXP_")
    logaddexp_func(A, B, out0=A)
    return A


# ---------------------------------------------------------------------------
# logaddexp2: log2(2^x + 2^y)
# Numerically stable form: max(x,y) + log2(exp2(x-max) + exp2(y-max))
# ---------------------------------------------------------------------------
@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def logaddexp2_func(x, y):
    x = x.to(tl.float32)
    y = y.to(tl.float32)
    mx = tl.maximum(x, y)
    return mx + tl.log2(tl.exp2(x - mx) + tl.exp2(y - mx))


def logaddexp2(A, B):
    logger.debug("GEMS LOGADDEXP2")
    return logaddexp2_func(A, B)


def logaddexp2_(A, B):
    logger.debug("GEMS LOGADDEXP2_")
    logaddexp2_func(A, B, out0=A)
    return A
