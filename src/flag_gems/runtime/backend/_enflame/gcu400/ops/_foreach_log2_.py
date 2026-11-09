# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""GCU400 专属 ``_foreach_log2_``（in-place）。

公共 ``foreach_unary`` 执行器用二维 grid 承载 chunk table；GCU400 的
grid.y 硬件上限为 255，大 shape 下会报 ``out of resource: grid.y``。
这里对浮点 tensor 列表改走一维固定 grid + 核内跨步的 log2 kernel，
grid.x 恒为 48，既解除硬件限制也避免每条 tensor 建 metadata 的开销；
非浮点/复数/integral 输入仍回退公共 ``apply_unary`` 以保持 dtype policy。
"""

import logging
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_unary import apply_unary
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# 由优化迭代驱动脚本按轮次生成；编译期配置，不改变算子语义。
FIXED_GRID = 48
BLOCK_SIZE = 8192
NUM_WARPS = 4
USE_DEVICE_CONTEXT = 0
SKIP_EMPTY_TENSORS = 0
REVERSE_LAUNCH_ORDER = 1
FALLBACK_TO_PUBLIC = 0
# --- TUNING PARAMS END ---

_FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _foreach_log2_flat_kernel(
    data_ptr,
    N,
    BLOCK: tl.constexpr,
    FIXED_GRID: tl.constexpr,
):
    """一维固定 grid 跨步写回 log2，支持任意长度 tensor。"""
    arange = tl.arange(0, BLOCK)
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for block_id in tl.range(pid, tl.cdiv(N, BLOCK), num_programs):
        offs = block_id * BLOCK + arange
        mask = offs < N
        value = tl.load(data_ptr + offs, mask=mask).to(tl.float32)
        tl.store(data_ptr + offs, tl.log2(value), mask=mask)


def _foreach_log2_(self):
    logger.debug("GEMS_GCU400 FOREACH_LOG2_")
    if FALLBACK_TO_PUBLIC or not all(
        tensor.dtype in _FLOAT_DTYPES and not tensor.dtype.is_complex
        for tensor in self
    ):
        return apply_unary("log2", self, inplace=True)

    tensors = self
    if SKIP_EMPTY_TENSORS:
        tensors = [tensor for tensor in self if tensor.numel() > 0]
    if REVERSE_LAUNCH_ORDER:
        tensors = list(reversed(tensors))

    device_context = (
        torch_device_fn.device(self[0].device) if USE_DEVICE_CONTEXT else nullcontext()
    )
    with device_context:
        for tensor in tensors:
            num_elems = tensor.numel()
            grid = (min(triton.cdiv(num_elems, BLOCK_SIZE), FIXED_GRID),)
            _foreach_log2_flat_kernel[grid](
                tensor,
                num_elems,
                BLOCK_SIZE,
                FIXED_GRID,
                num_warps=NUM_WARPS,
            )
    return None
