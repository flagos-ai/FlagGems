import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def frac_func(x):
    xf = x.to(tl.float32)
    trunc_x = tl.where(xf >= 0, tl.floor(xf), tl.ceil(xf))
    return (xf - trunc_x).to(x.dtype)


def frac(x):
    logger.debug("GEMS FRAC")
    return frac_func(x)


def frac_(x):
    logger.debug("GEMS FRAC_")
    return frac_func(x, out0=x)


def frac_out(x, *, out):
    logger.debug("GEMS FRAC_OUT")
    return frac_func(x, out0=out)
