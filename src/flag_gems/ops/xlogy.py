import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


# PyTorch xlogy semantics (per torch.xlogy docs):
#     out_i = NaN                       if y_i is NaN  (wins over x == 0)
#             0                         if x_i == 0 and y_i is not NaN
#             x_i * log(y_i)            otherwise
# All other IEEE-754 outcomes (log(0) = -inf, log(negative) = NaN, NaN
# propagation through x) fall out naturally from the direct computation.
# Compute in float32 for fp16/bf16 to preserve log accuracy; pointwise_dynamic
# casts the result back per type promotion.
@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def xlogy_func(x, y):
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    result = x_f32 * tl.log(y_f32)
    result = tl.where(x_f32 == 0.0, 0.0, result)
    result = tl.where(y_f32 != y_f32, float("nan"), result)
    return result


def xlogy(self, other):
    logger.debug("GEMS XLOGY")
    return xlogy_func(self, other)


def xlogy_out(self, other, out):
    logger.debug("GEMS XLOGY_OUT")
    xlogy_func(self, other, out0=out)
    return out
