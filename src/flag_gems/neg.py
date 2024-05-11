import torch
import triton
import triton.language as tl
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def neg_func(x):
    return -x


def neg(A):
    if __debug__:
        print("GEMS NEG")
    O = neg_func(A)
    return O
