import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_isinf = tl_extra_shim.isinf

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")])
@triton.jit
def isneginf_func(x):
    x_fp32 = x.to(tl.float32)
    return _isinf(x_fp32) & (x_fp32 < 0)


def isneginf(A):
    logger.debug("GEMS ISNEGINF")
    return isneginf_func(A)


def isneginf_out(A, *, out=None):
    logger.debug("GEMS ISNEGINF_OUT")
    if out is None:
        return isneginf_func(A)
    isneginf_func(A, out0=out)
    return out
