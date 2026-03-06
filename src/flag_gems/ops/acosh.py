import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_acosh = tl_extra_shim.acosh
logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def acosh_kernel(x):
    return _acosh(x.to(tl.float32))


def acosh(A):
    logger.debug("GEMS ACOSH")
    return acosh_kernel(A)


def acosh_(A):
    logger.debug("GEMS ACOSH_")
    acosh_kernel(A, out0=A)
    return A
