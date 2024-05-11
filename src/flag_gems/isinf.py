import torch
import triton
import triton.language as tl
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def isinf_func(x):
    return tl.math.isinf(x.to(tl.float32))


def isinf(A):
    if __debug__:
        print("GEMS ISINF")
    O = isinf_func(A)
    return O
