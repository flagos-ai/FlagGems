import torch
import triton
import triton.language as tl
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def sin_func(x):
    return tl.sin(x.to(tl.float32))


def sin(A):
    if __debug__:
        print("GEMS SIN")
    O = sin_func(A)
    return O
