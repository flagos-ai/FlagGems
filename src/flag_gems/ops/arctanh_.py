import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


# arctanh(x) = 0.5 * log((1+x) / (1-x))
@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def arctanh_inplace_func(x):
    xf = x.to(tl.float32)
    return (0.5 * tl.log((1.0 + xf) / (1.0 - xf))).to(x.dtype)


def arctanh_(A):
    logger.debug("GEMS ARCTANH_")
    arctanh_inplace_func(A, out0=A)
    return A
