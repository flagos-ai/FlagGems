import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def logaddexp_func(x, y):
    x_f = x.to(tl.float32)
    y_f = y.to(tl.float32)

    m = tl.maximum(x_f, y_f)
    abs_diff = tl.abs(x_f - y_f)

    exp_u = tl.exp(-abs_diff)
    res = tl.where(abs_diff > 13.0, m, m + tl.log(1.0 + exp_u))
    is_pos_inf = (x_f == float("inf")) | (y_f == float("inf"))
    is_both_neg_inf = m == -float("inf")

    out = tl.where(
        is_pos_inf,
        float("inf"),
        tl.where(is_both_neg_inf, -float("inf"), res),
    )
    return out


def logaddexp(x, y):
    logger.debug("GEMS LOGADDEXP")
    return logaddexp_func(x, y)
