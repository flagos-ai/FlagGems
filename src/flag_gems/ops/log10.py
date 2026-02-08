import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def log10_func(x):
    M_LOG10E = 0.43429448190325176
    return tl.log((x.to(tl.float32))) * M_LOG10E


def log10(x):
    logger.debug("GEMS LOG10")
    return log10_func(x)


def log10_(x):
    logger.debug("GEMS LOG10_")
    log10_func(x, out0=x)
    return x
