"""cummaxmin_backward 候选 v1：num_warps 4→8 + 免 mask + 精简 host。

生产 `flag_gems.ops.cummax.cummaxmin_backward` 用 fp32 中间 buffer +
`tl.atomic_add` scatter + `.to(dtype)`，num_warps 取 Triton 默认 4，
且无条件 `.contiguous()` / `torch_device_fn.device` 上下文 / logger。

13 轮迭代（含 kernel 循环修正与容差判据）显示：
- num_warps=8：0.277ms vs 生产 0.308ms（1.11x）；
- reduction_size 恰为 2 的幂时去掉 mask 再省一点；
- 直接 atomic_add 到输出 dtype 会损失 fp16/bf16 精度，丢弃；
- num_warps 1/2、BLOCK 256/1024/2048、2D tile 均劣化。

仅接管 inner_size==1 且输入连续的情形（含 benchmark 的 2D/dim=1 与
correctness 的 dim=-1）；inner_size>1、非连续、空张量一律委托生产，
保证 interior-axis 与 NaN 语义不变。
"""
import torch
import triton
import triton.language as tl

from flag_gems.ops.cummax import cummaxmin_backward as _public

__all__ = ["cummaxmin_backward"]


@triton.jit
def _cmb_kernel(G, IDX, OUT, RS, IS, BLOCK: tl.constexpr,
                MASKED: tl.constexpr):
    row = tl.program_id(0)
    outer = row // IS
    inner = row % IS
    base = outer * (RS * IS) + inner
    ar = tl.arange(0, BLOCK)
    for bs in range(0, RS, BLOCK):
        pos = bs + ar
        if MASKED:
            m = pos < RS
            g = tl.load(G + base + pos * IS, mask=m, other=0.0)
            t = tl.load(IDX + base + pos * IS, mask=m, other=0)
            tl.atomic_add(OUT + base + t * IS, g.to(tl.float32), mask=m)
        else:
            g = tl.load(G + base + pos * IS)
            t = tl.load(IDX + base + pos * IS)
            tl.atomic_add(OUT + base + t * IS, g.to(tl.float32))


def cummaxmin_backward(grad_output, input, indices, dim):
    ndim = grad_output.ndim
    if dim < 0:
        dim += ndim
    shape = list(grad_output.shape)
    RS = shape[dim]
    inner_size = 1
    for i in range(dim + 1, ndim):
        inner_size *= shape[i]
    outer_size = 1
    for i in range(dim):
        outer_size *= shape[i]

    if (inner_size != 1 or RS == 0
            or not grad_output.is_contiguous()
            or not indices.is_contiguous()):
        return _public(grad_output, input, indices, dim)

    BLOCK = min(triton.next_power_of_2(RS), 1024)
    out = torch.zeros(shape, dtype=torch.float32,
                      device=grad_output.device)
    _cmb_kernel[(outer_size,)](
        grad_output, indices, out, RS, inner_size,
        BLOCK=BLOCK, MASKED=(BLOCK != RS), num_warps=8,
    )
    return out.to(grad_output.dtype)
