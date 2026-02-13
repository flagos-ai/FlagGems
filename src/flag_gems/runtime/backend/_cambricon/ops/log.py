import logging

import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def log_func(x, inplace):
    return tl.log(x.to(tl.float32))


def log(A):
    logger.debug("GEMS_CAMBRICON LOG")
    return log_func(A, False)
