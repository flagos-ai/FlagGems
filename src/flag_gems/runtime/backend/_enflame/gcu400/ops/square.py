"""square 候选 v1：复用 square_out 已验证的大 BLOCK 手写 kernel。

square 与 square_out 共用公共 ops/square.py 的 pointwise_dynamic codegen，
ENFLAME max_grid_size=(12,1,1) 让大 shape 并行度不足（实测 5.95ms vs
torch 0.76ms）。本候选沿用 square_out_v2 的分档策略，out=None 时分配新张量。
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


def square(A):
    n = A.numel()
    # 回退必须先于任何分配，否则小 shape 会多付一次 empty_like 开销
    if n < 65536 or not A.is_contiguous():
        from flag_gems.ops.square import square as _public
        return _public(A)
    out = torch.empty_like(A)
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
