"""bucketize 候选 v2：用「计数比较」替代 gather 二分查找，并修正非连续输入。

生产 `flag_gems.ops.bucketize.bucketize` 对每个元素做
ceil(log2(NB+1)) 次二分查找，每轮 `tl.load(boundaries + mid)` 是各 lane
地址不同的 gather；固定 BLOCK=1024 / num_warps=4。

boundaries 升序时：lower_bound(v) == count(b_i < v)，
upper_bound(v) == count(b_i <= v)。改成对 constexpr NB 做标量 broadcast
load + 计数后，无 gather、无 mid 计算、无动态二分循环。19 轮迭代中
`BLOCK=4096 / num_warps=8` 最优：0.00685ms vs 生产 0.03152ms（4.60x），
连续输入的 288 个门禁 case 与生产 + CPU 参考均逐位一致。

非连续输入：生产实现存在既有缺陷——`output.flatten()` 对 preserve-format
的非连续输出产生拷贝，kernel 写入的是拷贝，而 `output.reshape(input.shape)`
返回的仍是未初始化的原张量（实测输出为 int64 垃圾值，CPU 参考为 [0,1]）。
v2 不委托该错误路径，改为显式 contiguous 后走自己的连续输出 kernel，
结果与 CPU `torch.bucketize` 逐位一致（顺带修正该既有缺陷）。

安全边界（委托生产）：NB == 0 / NB > 16（大 NB 计数展开不如二分）、空张量；
其余 dtype / right / out_int32 由同一 kernel 的 constexpr 分支支持。
生产 ops/bucketize.py 未改动。
"""
import torch
import triton
import triton.language as tl

from flag_gems.ops.bucketize import bucketize as _public

__all__ = ["bucketize"]

NB_MAX = 16


@triton.jit
def _bucketize_count_kernel(INP, B, OUT, N, NB: tl.constexpr,
                            BLOCK: tl.constexpr, RIGHT: tl.constexpr,
                            MASKED: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    if MASKED:
        m = off < N
        v = tl.load(INP + off, mask=m, other=0)
    else:
        v = tl.load(INP + off)
    res = tl.full([BLOCK], 0, tl.int32)
    for i in range(NB):
        b = tl.load(B + i)
        if RIGHT:
            res += (b <= v).to(tl.int32)
        else:
            res += (b < v).to(tl.int32)
    if MASKED:
        tl.store(OUT + off, res.to(OUT.dtype.element_ty), mask=m)
    else:
        tl.store(OUT + off, res.to(OUT.dtype.element_ty))


def bucketize(input, boundaries, *, out_int32=False, right=False):
    nb = boundaries.numel()
    n = input.numel()
    # 显式连续化：避免生产实现在非连续输入上写出未初始化内存。
    inp = input if input.is_contiguous() else input.contiguous()
    b = boundaries if boundaries.is_contiguous() else boundaries.contiguous()
    if nb == 0 or nb > NB_MAX or n == 0:
        return _public(inp, b, out_int32=out_int32, right=right)

    out_dtype = torch.int32 if out_int32 else torch.int64
    out = torch.empty(n, dtype=out_dtype, device=input.device)
    BLOCK = 4096 if n >= 65536 else 2048
    _bucketize_count_kernel[(triton.cdiv(n, BLOCK),)](
        inp, b, out, n, NB=nb, BLOCK=BLOCK,
        RIGHT=right, MASKED=(n % BLOCK != 0) or (BLOCK > n),
        num_warps=8,
    )
    return out.reshape(input.shape)
