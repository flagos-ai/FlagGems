# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GCU400 专属 ``special_bessel_j1``：手写 1D flat Triton kernel。

公共实现使用 ``@pointwise_dynamic`` codegen；该 codegen 在 GCU 上会做多维索引
分解，产生大量标量整除/取模。这里改为一维线性索引 + grid-stride，数值路径
仍复用 ``tl_extra_shim.j1``（内部先升到 fp32 再计算），保持精度口径一致。
"""

import logging
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# 由优化迭代驱动脚本按轮次生成；编译期配置，不改变算子语义。
BLOCK_SIZE = 8192
NUM_WARPS = 4
FIXED_GRID = 48
NUM_STAGES = 1
USE_CONTIGUOUS = 1
ALLOC_OUT_MODE = 0
USE_DEVICE_CONTEXT = 1
# --- TUNING PARAMS END ---


@libentry()
@triton.jit(do_not_specialize=["N"])
def _j1_flat_kernel(
    x_ptr,
    out_ptr,
    N,
    BLOCK: tl.constexpr,
    FIXED_GRID: tl.constexpr,
):
    """一维线性遍历输入；bessel J1 由公共 shim 在 fp32 上计算。"""
    arange = tl.arange(0, BLOCK)
    pid = tl.program_id(0)
    if FIXED_GRID:
        num_programs = tl.num_programs(0)
        for block_id in tl.range(pid, tl.cdiv(N, BLOCK), num_programs):
            offs = block_id * BLOCK + arange
            mask = offs < N
            x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
            tl.store(out_ptr + offs, tl_extra_shim.j1(x), mask=mask)
    else:
        offs = pid * BLOCK + arange
        mask = offs < N
        x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
        tl.store(out_ptr + offs, tl_extra_shim.j1(x), mask=mask)


def special_bessel_j1(A):
    logger.debug("GEMS_GCU400 SPECIAL_BESSEL_J1")
    inp = A.contiguous() if USE_CONTIGUOUS else A
    if ALLOC_OUT_MODE:
        out = torch.empty(tuple(inp.shape), dtype=inp.dtype, device=inp.device)
    else:
        out = torch.empty_like(inp)
    num_elems = inp.numel()
    if num_elems == 0:
        return out
    if FIXED_GRID:
        grid = (min(triton.cdiv(num_elems, BLOCK_SIZE), FIXED_GRID),)
    else:
        grid = (triton.cdiv(num_elems, BLOCK_SIZE),)
    device_context = (
        torch_device_fn.device(inp.device) if USE_DEVICE_CONTEXT else nullcontext()
    )
    with device_context:
        _j1_flat_kernel[grid](
            inp,
            out,
            num_elems,
            BLOCK_SIZE,
            FIXED_GRID,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
    return out
