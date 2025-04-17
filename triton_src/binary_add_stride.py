"""
A template for pointwise computation of C-contiguous array in triton language.
Simply copy and modify the operation in pointwise_kernel & dtype of the input
tensor and run it. Then collect an inspect the generated ptx & SASS code to
learn the mapping between

`triton builtin function -> ptx -> SASS`

to get some understanding of ptx and SASS.
"""

import triton
from triton import language as tl


@triton.jit
def binary_pointwise_kernel(X, Y, Out, n, sx, sy, so, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < n

    x = tl.load(X + offsets * sx, mask=mask)
    y = tl.load(Y + offsets * sy, mask=mask)
    o = x + y
    tl.store(Out + offsets * so, o, mask=mask)
