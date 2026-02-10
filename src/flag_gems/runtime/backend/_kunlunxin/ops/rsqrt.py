import logging

import triton
import triton.language as tl
import triton.language.extra.xpu.libdevice as xpu

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def rsqrt_func(x):
    return xpu.rsqrt(x.to(tl.float32))


def rsqrt(A):
    logger.debug("GEMS RSQRT")
    return rsqrt_func(A)


def rsqrt_(A):
    logger.debug("GEMS RSQRT_")
    return rsqrt_func(A, out0=A)
