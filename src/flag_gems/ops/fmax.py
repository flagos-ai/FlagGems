import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


# fmax(x, y) returns the element-wise maximum with IEEE NaN handling:
#   * If exactly one input is NaN, return the other (non-NaN) value.
#   * If both inputs are NaN, return NaN.
# This is in contrast to torch.maximum which propagates NaN if *either* input
# is NaN. We compute in float32 for fp16/bf16 to ensure comparisons remain
# well-defined; pointwise_dynamic handles the final type promotion.
@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def fmax_func(x, y):
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    x_nan = x_f32 != x_f32
    y_nan = y_f32 != y_f32
    both_nan = x_nan & y_nan
    # tl.maximum on NaN inputs is implementation-defined; choose explicitly.
    base = tl.where(x_f32 > y_f32, x_f32, y_f32)
    # Override one-sided NaN cases.
    base = tl.where(x_nan, y_f32, base)
    base = tl.where(y_nan, x_f32, base)
    base = tl.where(both_nan, float("nan"), base)
    return base


def fmax(self, other):
    logger.debug("GEMS FMAX")
    return fmax_func(self, other)


def fmax_out(self, other, out):
    logger.debug("GEMS FMAX_OUT")
    fmax_func(self, other, out0=out)
    return out
