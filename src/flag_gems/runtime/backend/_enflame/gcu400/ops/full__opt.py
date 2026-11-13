# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""GCU400 专属 ``full``：手写 tl.full kernel 替代 pointwise_dynamic codegen。

生产 ``gcu400/ops/full.py`` 走 logger + check_dtype + torch.empty +
pointwise_dynamic codegen 生成的 full_func_scalar；纯填充算子的耗时几乎
全部在 codegen 的 Python 层。这里对 GCU 上的 fp16/fp32/bf16 标量填充
改走一次 Triton launch，其余 dtype / Tensor fill_value / 非 GCU 设备
一律委托生产实现，保证 dtype policy 与溢出检查语义不变。
"""

import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime.backend._enflame.gcu400.ops.full import full as _public

__all__ = ["full"]

# --- TUNING PARAMS START ---
BLOCK = 65536
NUM_WARPS = 4
NUM_STAGES = 1
FIXED_GRID = 0
# --- TUNING PARAMS END ---

_FINFO = {
    torch.float16: torch.finfo(torch.float16),
    torch.float32: torch.finfo(torch.float32),
    torch.bfloat16: torch.finfo(torch.bfloat16),
}


@triton.jit
def _full_kernel(
    OUT,
    N,
    VAL,
    BLOCK_SIZE: tl.constexpr,
    MASKED: tl.constexpr,
    FIXED_GRID: tl.constexpr,
):
    """每个 program 生产一段常量向量；FIXED_GRID>0 时核内跨步复用 program。"""
    arange = tl.arange(0, BLOCK_SIZE)
    values = tl.full([BLOCK_SIZE], VAL, OUT.dtype.element_ty)
    if FIXED_GRID > 0:
        pid = tl.program_id(0)
        num_programs = tl.num_programs(0)
        for block_id in tl.range(pid, tl.cdiv(N, BLOCK_SIZE), num_programs):
            offs = block_id * BLOCK_SIZE + arange
            if MASKED:
                tl.store(OUT + offs, values, mask=offs < N)
            else:
                tl.store(OUT + offs, values)
    else:
        offs = tl.program_id(0) * BLOCK_SIZE + arange
        if MASKED:
            tl.store(OUT + offs, values, mask=offs < N)
        else:
            tl.store(OUT + offs, values)


def full(size, fill_value, *, dtype=None, layout=None, device=None, pin_memory=None):
    dev = torch.device(device) if device is not None else torch.device("cpu")
    if dtype is None:
        if isinstance(fill_value, bool):
            dt = torch.bool
        elif isinstance(fill_value, int):
            dt = torch.int64
        else:
            dt = torch.get_default_dtype()
    else:
        dt = dtype

    if (
        dev.type != "gcu"
        or dt not in _FINFO
        or isinstance(fill_value, torch.Tensor)
        or layout is not None
        or pin_memory is not None
    ):
        return _public(
            size,
            fill_value,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
        )

    # 与生产 check_dtype 同语义：bool 标量在浮点 dtype 下先转 int，再做溢出检查。
    if isinstance(fill_value, bool):
        fill_value = int(fill_value)
    if not (math.isinf(fill_value) or math.isnan(fill_value)):
        finfo = _FINFO[dt]
        if fill_value < finfo.min or fill_value > finfo.max:
            raise RuntimeError(
                f"value cannot be converted to type {dt} without overflow"
            )

    out = torch.empty(size, device=dev, dtype=dt)
    num_elems = out.numel()
    if num_elems == 0:
        return out
    if FIXED_GRID > 0:
        grid = (min(triton.cdiv(num_elems, BLOCK), FIXED_GRID),)
    else:
        grid = (triton.cdiv(num_elems, BLOCK),)
    _full_kernel[grid](
        out,
        num_elems,
        fill_value,
        BLOCK_SIZE=BLOCK,
        MASKED=(num_elems % BLOCK != 0) or (BLOCK > num_elems),
        FIXED_GRID=FIXED_GRID,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out
