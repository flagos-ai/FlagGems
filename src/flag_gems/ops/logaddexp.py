import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


# logaddexp(a, b) = max(a, b) + log(1 + exp(-|a - b|))
#
# The shifted form keeps the argument of exp in (-inf, 0] which avoids
# overflow when both inputs are large.  fp32 accumulation gives stable
# results for fp16/bf16 inputs.
#
# Edge cases:
#   * a == b == +inf  -> diff = NaN, but result must be +inf
#   * a == b == -inf  -> diff = NaN, but result must be -inf
#   * one input +inf  -> diff = +inf, exp(-diff) = 0, correction = 0 -> max
# We mask out the NaN-from-inf-minus-inf case by selecting on
# ``delta == delta`` (False only when delta is NaN) and falling back to
# the max, which is the correct limit for both ``+inf`` cases.
@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def logaddexp_func(x, y):
    x_fp32 = x.to(tl.float32)
    y_fp32 = y.to(tl.float32)
    m = tl.maximum(x_fp32, y_fp32)
    delta = tl.abs(x_fp32 - y_fp32)
    correction = tl.log(1.0 + tl.exp(-delta))
    # Inf - Inf (same sign) makes ``delta`` NaN; the limit is the signed
    # infinity itself, which equals ``m``.
    base = tl.where(delta == delta, m + correction, m)
    # ``tl.maximum`` follows IEEE 754 fmax semantics and does NOT propagate
    # NaN — a NaN input would silently return the other operand here, but
    # torch.logaddexp returns NaN.  Force NaN propagation explicitly:
    # ``x + y`` is NaN whenever either input is NaN.
    nan_mask = (x_fp32 != x_fp32) | (y_fp32 != y_fp32)
    return tl.where(nan_mask, x_fp32 + y_fp32, base)


def logaddexp(self, other):
    logger.debug("GEMS LOGADDEXP")
    return logaddexp_func(self, other)


def logaddexp_out(self, other, out):
    logger.debug("GEMS LOGADDEXP_OUT")
    logaddexp_func(self, other, out0=out)
    return out
