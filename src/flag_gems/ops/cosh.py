import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_cosh = tl_extra_shim.cosh
logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def cosh_func(x):
    return _cosh(x.to(tl.float32))


def cosh(A):
    logger.debug("GEMS COSH")
    return cosh_func(A)


def cosh_(A):
    logger.debug("GEMS COSH_")
    cosh_func(A, out0=A)
    return A
