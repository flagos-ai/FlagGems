# src/flag_gems/ops/cosh.py
import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def cosh_func(x):
    x_f32 = x.to(tl.float32)
    # cosh(x) = (exp(x) + exp(-x)) / 2
    return 0.5 * (tl.exp(x_f32) + tl.exp(-x_f32))


def cosh(A):
    logger.debug("GEMS COSH")
    return cosh_func(A)
