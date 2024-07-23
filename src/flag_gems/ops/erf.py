import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def erf_func(x):
    output = tl.math.erf(x.to(tl.float32))
    return output


def erf(x):
    logging.debug("GEMS ERF")
    return erf_func(x)
