"""square_out 候选 v2：在 v1 基础上把小 shape 回退公共实现。

v1 的 3 轮 benchmark：overall 4.15x / fp16 4.32x / fp16_large 4.34x，
但小 shape (64x64) 有 0.920~0.987 的轻微回归，触发逐 case p10>=0.95
门禁而 FAIL。v2 让 n<65536 直接走公共 pointwise_dynamic 实现，
消除该回归；其余保持 v1 的大 BLOCK 手写 kernel。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _square_kernel(X, Y, N, BLOCK: tl.constexpr, HP: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    x = tl.load(X + off, mask=m, other=0)
    if HP:
        xf = x.to(tl.float32)
        y = xf * xf
    else:
        y = x * x
    tl.store(Y + off, y, mask=m)


def _public(A, out):
    from flag_gems.ops.square import square_out as _impl

    return _impl(A, out=out)


def square_out(A, *, out=None):
    if out is None:
        out = torch.empty_like(A)
    n = A.numel()
    # 小 shape 与非连续输入回退公共实现，避免任何分档回归 / 语义退化
    if n < 65536 or not A.is_contiguous() or not out.is_contiguous():
        return _public(A, out)
    hp = A.dtype in (torch.float16, torch.bfloat16)
    if n < (1 << 24):
        BLOCK, nw = 16384, 2
    else:
        BLOCK, nw = 32768, 4
    grid = (triton.cdiv(n, BLOCK),)
    if grid[0] > 65535:
        BLOCK = 65536
        grid = (triton.cdiv(n, BLOCK),)
    _square_kernel[grid](A, out, n, BLOCK=BLOCK, HP=hp, num_warps=nw)
    return out
