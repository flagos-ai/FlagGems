"""
A template for pointwise computation of C-contiguous array in triton language.
Simply copy and modify the operation in pointwise_kernel & dtype of the input
tensor and run it. Then collect an inspect the generated ptx & SASS code to
learn the mapping between

`triton builtin function -> ptx -> SASS`

to get some understanding of ptx and SASS.
"""

import torch
import triton
from triton import language as tl


@triton.jit
def binary_pointwise_kernel(X, Y, Out, n, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < n

    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)
    o = x + y
    tl.store(Out + offsets, o, mask=mask)


def binary_add_tensor(x, y):
    dtype = torch.promote_types(x.dtype, y.dtype)
    x, y = torch.broadcast_tensors(x, y)
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x, dtype=dtype)
    n = out.numel()
    BLOCK_N = 1024
    grid = (triton.cdiv(n, BLOCK_N), 1, 1)
    binary_pointwise_kernel[grid](
        x, y, out, n, BLOCK_N=BLOCK_N, num_warps=8, num_stages=1
    )
    return out
