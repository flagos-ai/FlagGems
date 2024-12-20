import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic, tl_extra_shim

_isinf = tl_extra_shim.isinf


@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")])
@triton.jit
def isinf_func(x):
    return _isinf(x.to(tl.float32))


def isinf(A):
    logging.debug("GEMS ISINF")
    return isinf_func(A)
