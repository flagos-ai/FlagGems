import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 0, "DEFAULT")])
@triton.jit
def minimum_kernel(X, Y):
    if X.dtype == tl.bfloat16:
        X = X.to(tl.float32)
        Y = Y.to(tl.float32)
    return tl.minimum(X, Y)


def minimum(X, Y):
    logging.debug("GEMS MINIMUM")
    assert X.is_cuda and Y.is_cuda
    return minimum_kernel(X, Y)
