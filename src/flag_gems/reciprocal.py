import torch
import triton
import triton.language as tl

from flag_gems.utils.pointwise_dynamic import pointwise_dynamic

@pointwise_dynamic
@triton.jit
def reciprocal_func(x):
    return 1.0 / x.to(tl.float32)


def reciprocal(A):
    if __debug__:
        print("GEMS RECIPROCAL")
    O = reciprocal_func(A)
    return O
