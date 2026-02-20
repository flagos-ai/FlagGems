import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# logaddexp: log(exp(x) + exp(y))
# Numerically stable: m + log(1 + exp(-|x - y|)), where m = max(x, y)
# ---------------------------------------------------------------------------
@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def logaddexp_func(x, y):
    x = x.to(tl.float32)
    y = y.to(tl.float32)
    m = tl.maximum(x, y)
    delta = x - y
    return m + tl.log(1.0 + tl.exp(-tl.abs(delta)))


def logaddexp(A, B):
    logger.debug("GEMS LOGADDEXP")
    return logaddexp_func(A, B)


def logaddexp_out(A, B, *, out):
    logger.debug("GEMS LOGADDEXP_OUT")
    logaddexp_func(A, B, out0=out)
    return out


# ---------------------------------------------------------------------------
# logaddexp2: log2(2^x + 2^y)
# Numerically stable: m + log2(1 + exp2(-(x - y) * sign)), where m = max(x, y)
# ---------------------------------------------------------------------------
@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def logaddexp2_func(x, y):
    x = x.to(tl.float32)
    y = y.to(tl.float32)
    m = tl.maximum(x, y)
    delta = x - y
    return m + tl.log2(1.0 + tl.exp2(-tl.abs(delta)))


def logaddexp2(A, B):
    logger.debug("GEMS LOGADDEXP2")
    return logaddexp2_func(A, B)


def logaddexp2_out(A, B, *, out):
    logger.debug("GEMS LOGADDEXP2_OUT")
    logaddexp2_func(A, B, out0=out)
    return out
