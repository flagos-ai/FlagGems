import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_bessel_y1 = tl_extra_shim.y1

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def special_bessel_y1_func(x):
    return _bessel_y1(x.to(tl.float32)).to(x.dtype)


def special_bessel_y1(A):
    logger.debug("GEMS SPECIAL_BESSEL_Y1")
    return special_bessel_y1_func(A)


def special_bessel_y1_(A):
    logger.debug("GEMS SPECIAL_BESSEL_Y1_")
    special_bessel_y1_func(A, out0=A)
    return A


def special_bessel_y1_out(A, *, out=None):
    logger.debug("GEMS SPECIAL_BESSEL_Y1_OUT")
    if out is None:
        return special_bessel_y1_func(A)
    special_bessel_y1_func(A, out0=out)
    return out
