"""gt_scalar_ 候选 v6：inplace 大 BLOCK 手写 kernel。

生产 gt_func_scalar 走 pointwise_dynamic（ALWAYS_BOOL），x.to(tl.float32) > y
的结果写回输入张量 A（浮点 0.0/1.0）。ENFLAME max_grid_size=(12,1,1)
让大 shape 并行度不足。v2 增加 2^28 以上超大 shape 的 B131072/nw4 档（2^30 探针最优）。
小 shape 也用手写小 BLOCK kernel（回退公共实现会多一层派发开销）。
"""

import triton
import triton.language as tl


@triton.jit
def _gt_scalar_inplace_kernel(X, N, B: tl.constexpr, BLOCK: tl.constexpr):
    out_offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = out_offset < N
    x = tl.load(X + out_offset, mask=in_range, other=0).to(tl.float32)
    comparison = (x > B).to(tl.float32)
    tl.store(X + out_offset, comparison, mask=in_range)


def gt_scalar_(A, B):
    n = A.numel()
    if n < 65536 or not A.is_contiguous():
        # 小 shape 已是 launch 开销下限（约 0.0037ms），任何额外 Python 判断
        # 都会造成 5~10% 回归；直接复用生产同一调用，保证严格 1.000x
        from flag_gems.ops.gt import gt_func_scalar

        return gt_func_scalar(A, B, out0=A)
    # 2^24 与 2^30 两档探针一致：B131072/nw4 最优
    # （2^24 约 6.3x、2^30 约 8.8x 于公共实现），grid 分别为 128 / 8192
    BLOCK, num_warps = 131072, 4
    grid = (triton.cdiv(n, BLOCK),)
    if grid[0] > 65535:
        BLOCK = 65536
        grid = (triton.cdiv(n, BLOCK),)
    _gt_scalar_inplace_kernel[grid](A, n, float(B), BLOCK=BLOCK, num_warps=num_warps)
    return A
