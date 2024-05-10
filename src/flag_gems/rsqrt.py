import torch
import triton
import triton.language as tl

from flag_gems.utils.pointwise_dynamic import pointwise_dynamic

@pointwise_dynamic
@triton.jit
def rsqrt_func(x):
    return 1.0 / tl.sqrt(x.to(tl.float32))


def rsqrt(A):
    if __debug__:
        print("GEMS RSQRT")
    O = rsqrt_func(A)
    return O
