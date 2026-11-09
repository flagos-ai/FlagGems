"""full_like 候选 v1：手写 tl.full kernel 替代 pointwise_dynamic codegen。

生产 gcu400 `full_like.py` 走 `logger.debug` + `check_dtype` + `torch.empty`
+ `pointwise_dynamic` codegen 生成的 `full_func_scalar`。13 轮迭代显示
手写 `tl.full` kernel（BLOCK=4096 / num_warps=8）为 0.00396ms vs 生产
0.01838ms（4.64x），所有 case 与生产逐位一致；aten `out.fill_` 对照
0.00461ms，Triton 版更快，故主要路径保持 Triton 实现。

安全边界（全部委托生产 `full_like`，保证语义不变）：
- 非 GCU 设备、显式 device 非 gcu；
- dtype 不属于 fp16/fp32/bf16（int/bool/float64 等）；
- fill_value 是 Tensor（0-d tensor 走 codegen 的 tensor 分支）；
- layout / pin_memory / memory_format 非 None。
float16/fp32/bfloat16 的溢出检查内联并缓存 finfo，避免 codegen 的
Python 层开销，同时保留生产 check_dtype 的语义。
"""
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime.backend._enflame.gcu400.ops.full_like import (
    full_like as _public,
)

__all__ = ["full_like"]

BLOCK = 4096
NUM_WARPS = 8
_FINFO = {
    torch.float16: torch.finfo(torch.float16),
    torch.float32: torch.finfo(torch.float32),
    torch.bfloat16: torch.finfo(torch.bfloat16),
}


@triton.jit
def _fill_kernel(OUT, N, VAL, BLOCK_SIZE: tl.constexpr,
                 MASKED: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    v = tl.full([BLOCK_SIZE], VAL, OUT.dtype.element_ty)
    if MASKED:
        tl.store(OUT + off, v, mask=off < N)
    else:
        tl.store(OUT + off, v)


def full_like(x, fill_value, *, dtype=None, layout=None, device=None,
              pin_memory=None, memory_format=None):
    dt = x.dtype if dtype is None else dtype
    dev = x.device if device is None else torch.device(device)
    if (dev.type != "gcu" or dt not in _FINFO
            or isinstance(fill_value, torch.Tensor)
            or layout is not None or pin_memory is not None
            or memory_format is not None):
        return _public(x, fill_value, dtype=dtype, layout=layout,
                       device=device, pin_memory=pin_memory,
                       memory_format=memory_format)

    # 与生产 check_dtype 相同的溢出检查（finfo 已缓存）
    if (not isinstance(fill_value, bool)
            and not (math.isinf(fill_value) or math.isnan(fill_value))):
        fi = _FINFO[dt]
        if fill_value < fi.min or fill_value > fi.max:
            raise RuntimeError(
                f"value cannot be converted to type {dt} without overflow")

    n = x.numel()
    out = torch.empty(x.size(), device=dev, dtype=dt)
    if n == 0:
        return out
    _fill_kernel[(triton.cdiv(n, BLOCK),)](
        out, n, fill_value,
        BLOCK_SIZE=BLOCK, MASKED=(n % BLOCK != 0) or (BLOCK > n),
        num_warps=NUM_WARPS,
    )
    return out
