import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def erf_func(x):
    output = tl.math.erf(x.to(tl.float32))
    return output


def erf(x):
    logger.debug("GEMS ERF")
    return erf_func(x)


def erf_(x):
    logger.debug("GEMS ERF_")
    return erf_func(x, out0=x)
