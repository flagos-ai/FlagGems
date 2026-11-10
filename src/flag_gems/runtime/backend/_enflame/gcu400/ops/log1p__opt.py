"""log1p 候选 v1：按 dtype / 元素量分档的 launch 配置。

生产 `_launch`（11-02 TRACKED）把 `BLOCK=32768 / GRID=48 /
num_warps=4 / NUM_STAGES=4` 写死。180 组 constexpr 扫描（与生产输出
逐位一致）显示最优配置按 dtype 分化：

    fp16/bf16  中 shape(2^24)  BS65536 G48 nw2 ST1  1.38x
               大 shape(2^29)  BS65536 G48 nw2 ST1  1.21x
    fp32       中 shape(2^24)  BS8192  G48 nw4 ST4  1.31x
               大 shape(2^29)  BS8192  G48 nw4 ST1  1.26x
    小 shape(4096)  缩小 BLOCK 到 npo2(n) 可省微秒级浪费 (1.5~2.1x)

`NUM_STAGES` 对 fp16/bf16 大 shape 影响明显（ST1 1.21x vs ST4 1.11x），
对 fp32 几乎无差异（2.411 vs 2.414），故 fp32 沿用生产 ST4。

实现方式：复用生产 `log1p_kernel` 的原始 JITFunction
（`LibEntry.fn`），在按算子的 host 分派里显式传 constexpr；
**不修改 TRACKED 的 log1p.py**、不动 `log1p_` / `log1p_out`，
也不改全局配置。host 侧 dtype promotion / 非连续 / 空张量语义
与生产 `log1p()` 逐字同构。

注意探针踩坑：`torch.randn()+2.0` 仍会产出 x<-1 使 log1p 得 NaN，
而 `torch.equal` 对 NaN 恒 False，会让全部配置误判为不合格；
扫描/门禁必须用全正输入。
"""

import torch
import triton

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._enflame.gcu400.ops.log1p import (
    log1p as _prod_log1p,
    log1p_kernel as _kernel_wrapped,
)

_K = _kernel_wrapped.fn

__all__ = ["log1p"]


def _config(n, dtype):
    """返回 (BLOCK_SIZE, GRID_SIZE, num_warps, NUM_STAGES)。"""
    if dtype == torch.float32:
        BS, G, nw, ST = 8192, 48, 4, 4
    else:
        BS, G, nw, ST = 65536, 48, 2, 1
    if n < BS:
        BS = max(1024, triton.next_power_of_2(n))
    num_tile = triton.cdiv(n, BS)
    return BS, min(G, num_tile), nw, ST


def log1p(x: torch.Tensor):
    if x.is_floating_point():
        result_dtype = x.dtype
    else:
        result_dtype = torch.float32

    x_contig = x.to(result_dtype).contiguous()
    out = torch.empty_like(x_contig)
    n = out.numel()
    if n == 0:
        return out.view_as(x)

    BS, G, nw, ST = _config(n, result_dtype)
    grid = (G, 1, 1)
    with torch_device_fn.device(out.device.index):
        _K[grid](
            x_contig.view(-1),
            out.view(-1),
            n,
            BLOCK_SIZE=BS,
            GRID_SIZE=G,
            NUM_STAGES=ST,
            num_warps=nw,
        )
    return out.view_as(x)
