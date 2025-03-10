import logging

import triton
import triton.language as tl
from triton.language.extra.xpu.libdevice import erf as _erf

from ..utils.pointwise_dynamic import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def erf_func(x):
    output = _erf(x.to(tl.float32))
    return output


def erf(x):
    logging.debug("GEMS ERF")
    return erf_func(x)


def erf_(x):
    logging.debug("GEMS ERF_")
    return erf_func(x, out0=x)
