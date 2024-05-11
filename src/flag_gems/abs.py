import torch
import triton
import triton.language as tl
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def abs_func(x):
    return tl.abs(x)


def abs(A):
    if __debug__:
        print("GEMS ABS")
    O = abs_func(A)
    return O
