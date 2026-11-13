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

"""GCU400 专属 ``_foreach_lerp__List``：逐 tensor 三输入一维 flat kernel。

公共 ``foreach_ternary`` / ``foreach_binary_list`` 依赖设备侧 chunk table 与
二维 grid，在 GCU400 上需要 int64 指针表且容易超出 grid.y 的 255 上限。
这里改成逐 tensor 的一维固定 grid + 核内跨步，计算统一提升 fp32。
"""

import logging
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_binary import _apply_lerp
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---


# 由优化迭代驱动脚本按轮次生成；编译期/启动期配置，不改变算子语义。
FIXED_GRID = 48
BLOCK_SIZE = 65536
NUM_WARPS = 4
NUM_STAGES = 1
USE_DEVICE_CONTEXT = 1
REVERSE_LAUNCH_ORDER = 0
FALLBACK_TO_PUBLIC = 0
# --- TUNING PARAMS END ---

_FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


@libentry()
@triton.jit(do_not_specialize=["N", "SCALAR"])
def _foreach_lerp_list_flat_kernel(
    p0, p1, p2, out_ptr, N, SCALAR, BLOCK: tl.constexpr, FIXED_GRID: tl.constexpr
):
    """一维固定 grid 跨步执行 _foreach_lerp__List。"""
    arange = tl.arange(0, BLOCK)
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for block_id in tl.range(pid, tl.cdiv(N, BLOCK), num_programs):
        offs = block_id * BLOCK + arange
        mask = offs < N
        a = tl.load(p0 + offs, mask=mask).to(tl.float32)
        b = tl.load(p1 + offs, mask=mask).to(tl.float32)
        c = tl.load(p2 + offs, mask=mask).to(tl.float32)
        result = a + c * (b - a)
        tl.store(out_ptr + offs, result, mask=mask)


def _can_use_flat(tensors):
    """所有 tensor 必须是同 dtype 的连续浮点张量。"""
    if not tensors or len(tensors) == 0:
        return False
    first = tensors[0]
    if first.dtype not in _FLOAT_DTYPES or first.dtype.is_complex:
        return False
    for tensor in tensors:
        if tensor is None or tensor.dtype != first.dtype:
            return False
        if not tensor.is_contiguous() or tensor.shape != first.shape:
            return False
    return True


def _flat_ready(self, others, thirds):
    if FALLBACK_TO_PUBLIC:
        return False
    if len(self) == 0 or len(self) != len(others):
        return False
    if thirds is not None and len(thirds) != len(self):
        return False
    groups = [self, others] + ([thirds] if thirds is not None else [])
    for group in groups:
        if not _can_use_flat(group):
            return False
    for index in range(len(self)):
        if others[index].shape != self[index].shape:
            return False
        if thirds is not None and thirds[index].shape != self[index].shape:
            return False
    return True


def _run_flat(self, others, thirds, scalar, inplace):
    indices = list(range(len(self)))
    if REVERSE_LAUNCH_ORDER:
        indices.reverse()
    if not indices:
        return [] if not inplace else None
    outputs = [None] * len(self) if not inplace else None
    device_context = (
        torch_device_fn.device(self[0].device) if USE_DEVICE_CONTEXT else nullcontext()
    )
    with device_context:
        for index in indices:
            first, second = self[index], others[index]
            third = thirds[index] if thirds is not None else second
            num_elems = first.numel()
            out_tensor = first if inplace else torch.empty_like(first)
            # 空 tensor 无元素可算，跳过 kernel launch；但输出列表必须仍保留
            # 这个空结果，保持与输入一一对应（foreach 的返回长度等于输入长度）。
            if num_elems == 0:
                if not inplace:
                    outputs[index] = out_tensor
                continue
            grid = (min(triton.cdiv(num_elems, BLOCK_SIZE), FIXED_GRID),)
            _foreach_lerp_list_flat_kernel[grid](
                first,
                second,
                third,
                out_tensor,
                num_elems,
                scalar,
                BLOCK_SIZE,
                FIXED_GRID,
                num_warps=NUM_WARPS,
                num_stages=NUM_STAGES,
            )
            if not inplace:
                outputs[index] = out_tensor
    return None if inplace else outputs


def _foreach_lerp__List(self, tensors1, weight):
    logger.debug("GEMS_GCU400 FOREACH_LERP__LIST")
    if not _flat_ready(
        self, tensors1, weight if isinstance(weight, (list, tuple)) else None
    ):
        return _apply_lerp(self, tensors1, weight, True)
    return _run_flat(self, tensors1, weight, 0.0, inplace=True)
