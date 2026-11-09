"""floor_out 候选 v1：1D 手写 kernel，大 shape 用 B131072/nw4。

生产 floor_func 走 pointwise_dynamic：tl.floor(x.to(tl.float32)).to(x.dtype)，
ENFLAME max_grid_size=(12,1,1) 让大 shape 并行度不足。
gt_scalar_ 上已验证 2^24/2^30 档 B131072/nw4 显著优于 B32768/nw4；
微秒级小 shape 复用生产同一底层调用，避免额外 Python 判断造成回归。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _floor_out_kernel(X, Out, N, BLOCK: tl.constexpr):
    out_offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = out_offset < N
    x = tl.load(X + out_offset, mask=in_range, other=0)
    result = tl.floor(x.to(tl.float32)).to(x.dtype)
    tl.store(Out + out_offset, result, mask=in_range)


def floor_out(A, *, out=None):
    if out is None:
        from flag_gems.ops.floor import floor_out as _public
        return _public(A, out=None)
    n = A.numel()
    if n < 65536 or not A.is_contiguous() or not out.is_contiguous():
        from flag_gems.ops.floor import floor_func
        floor_func(A, out0=out)
        return out
    BLOCK, num_warps = 131072, 4
    grid = (triton.cdiv(n, BLOCK),)
    if grid[0] > 65535:
        BLOCK = 262144
        grid = (triton.cdiv(n, BLOCK),)
    _floor_out_kernel[grid](A, out, n, BLOCK=BLOCK, num_warps=num_warps)
    return out
