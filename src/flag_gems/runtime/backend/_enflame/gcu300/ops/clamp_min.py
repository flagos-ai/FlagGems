import logging

import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def clamp_min_func(x, min_val):
    return tl.maximum(x, min_val)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def clamp_min_scalar_func(x, min_val):
    return tl.maximum(x, min_val)


def clamp_min(A, min_val):
    logger.debug("GEMS_ENFLAME CLAMP_MIN")
    if isinstance(min_val, (int, float)):
        return clamp_min_scalar_func(A, min_val)
    return clamp_min_func(A, min_val)


def clamp_min_(A, min_val):
    logger.debug("GEMS_ENFLAME CLAMP_MIN_")
    if isinstance(min_val, (int, float)):
        clamp_min_scalar_func(A, min_val, out0=A)
    else:
        clamp_min_func(A, min_val, out0=A)
    return A
