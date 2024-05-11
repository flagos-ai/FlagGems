import torch
import triton
import triton.language as tl
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def cos_func(x):
    return tl.cos(x.to(tl.float32))


def cos(A):
    if __debug__:
        print("GEMS COS")
    O = cos_func(A)
    return O
