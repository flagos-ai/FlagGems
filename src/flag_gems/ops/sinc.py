import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def sinc_func(x):
    # sinc(x) = sin(pi*x) / (pi*x), with sinc(0) = 1
    pix = x.to(tl.float32) * 3.141592653589793
    return tl.where(
        pix == 0.0,
        1.0,
        tl.sin(pix) / pix,
    )


def sinc(A):
    logger.debug("GEMS SINC")
    return sinc_func(A)


def sinc_(A):
    logger.debug("GEMS SINC_")
    sinc_func(A, out0=A)
    return A
