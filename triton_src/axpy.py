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
def binary_pointwise_kernel(X, Y, Out, n, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < n

    x = tl.load(X + offsets, mask=mask)
    if Y is not None:
        y = tl.load(Y + offsets, mask=mask)
        x += y
    tl.store(Out + offsets, x, mask=mask)


if __name__ == "__main__":
    import torch

    x = torch.randn(4096, device="cuda")
    y = None
    out = torch.empty_like(x)
    binary_pointwise_kernel(x, y, out, 4096, 1024)
