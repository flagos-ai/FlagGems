import triton
import triton.language as tl
import logging
from ..utils import libentry, pointwise_dynamic


@pointwise_dynamic
@triton.jit
def pow_func(x, exponent):
    return tl.math.pow(x.to(tl.float32), exponent)


def pow_tensor_tensor(A, exponent):
    logging.debug("GEMS POW_TENSOR_TENSOR")
    O = pow_func(A, exponent)
    return O
