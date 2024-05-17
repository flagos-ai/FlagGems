import triton
import triton.language as tl
import logging
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def rsqrt_func(x):
    return 1.0 / tl.sqrt(x.to(tl.float32))


def rsqrt(A):
    logging.debug("GEMS RSQRT")
    O = rsqrt_func(A)
    return O
