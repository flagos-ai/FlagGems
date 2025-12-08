import logging

import triton
import triton.language as tl
from triton.language.extra.mlu.libdevice import fast_erf as _erf

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def erf_func(x, inplace):
    output = _erf(x.to(tl.float32))
    return output


def erf(x):
    logger.debug("GEMS_CAMBRICON ERF")
    return erf_func(x, False)


def erf_(x):
    logger.debug("GEMS_CAMBRICON ERF_")
    return erf_func(x, True, out0=x)
