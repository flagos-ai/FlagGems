import torch
import triton
import triton.language as tl
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def bitwise_not_func(x):
    return ~x


def bitwise_not(A):
    if __debug__:
        print("GEMS BITWISE NOT")
    O = bitwise_not_func(A)
    return O
