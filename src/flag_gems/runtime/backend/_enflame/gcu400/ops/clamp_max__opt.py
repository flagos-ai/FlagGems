"""clamp_max_ 候选 v1：inplace 大 BLOCK 手写 kernel。

生产 gcu400 实现（KernelGen）走 pointwise_dynamic，ENFLAME
max_grid_size=(12,1,1) 让大 shape 并行度不足。沿用 square/square_out
已验证的分档策略，n<65536 回退公共实现避免微秒级回归。
生产语义：tl.minimum(maxi, x.to(tl.float32))，标量以 fp32 比较后写回原 dtype。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _clamp_max_kernel(X, N, MAXI: tl.constexpr, BLOCK: tl.constexpr,
                      TO_F32: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    x = tl.load(X + off, mask=m, other=0)
    if TO_F32:
        y = tl.minimum(MAXI, x.to(tl.float32))
    else:
        y = tl.minimum(MAXI, x)
    tl.store(X + off, y, mask=m)


def clamp_max_(A, max_value):
    n = A.numel()
    if n < 65536 or not A.is_contiguous():
        from flag_gems.runtime.backend._enflame.gcu400.ops.clamp_max import (
            clamp_max_ as _public,
        )
        return _public(A, max_value)
    to_f32 = A.dtype in (torch.float16, torch.bfloat16)
    if n < (1 << 24):
        BLOCK, nw = 16384, 2
    else:
        BLOCK, nw = 32768, 4
    grid = (triton.cdiv(n, BLOCK),)
    if grid[0] > 65535:
        BLOCK = 65536
        grid = (triton.cdiv(n, BLOCK),)
    _clamp_max_kernel[grid](A, n, float(max_value), BLOCK=BLOCK,
                            TO_F32=to_f32, num_warps=nw)
    return A
