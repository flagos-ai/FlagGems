# Copyright 2026, The FlagOS Contributors.
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
"""GCU400 专属 ``_foreach_sin``：逐 tensor 一维 flat kernel。

公共 ``foreach_unary`` 执行器依赖设备侧 chunk table 与二维 grid，在 GCU400
上既需要 int64 指针表又容易超出 grid.y 的 255 上限。这里改成逐 tensor 的
一维固定 grid + 核内跨步，计算统一提升 fp32 保证与公共实现的精度一致。
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


# 由优化迭代驱动脚本按轮次生成；编译期/启动期配置，不改变算子语义。
FIXED_GRID = 48
BLOCK_SIZE = 2048
NUM_WARPS = 4
NUM_STAGES = 1
USE_DEVICE_CONTEXT = 1
REVERSE_LAUNCH_ORDER = 0
FALLBACK_TO_PUBLIC = 0
# --- TUNING PARAMS END ---

_FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _foreach_sin_flat_kernel(
    in_ptr, out_ptr, N, BLOCK: tl.constexpr, FIXED_GRID: tl.constexpr
):
    """一维固定 grid 跨步执行逐元素 sin。

    in_ptr 与 out_ptr 只在 out-of-place 变体下不同；避免 host 侧 clone 在
    超大 tensor 上触发 torch_gcu copy kernel 的 grid.x 上限。
    """
    arange = tl.arange(0, BLOCK)
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for block_id in tl.range(pid, tl.cdiv(N, BLOCK), num_programs):
        offs = block_id * BLOCK + arange
        mask = offs < N
        value = tl.load(in_ptr + offs, mask=mask).to(tl.float32)
        result = tl.sin(value)
        tl.store(out_ptr + offs, result, mask=mask)


def _can_use_flat_kernel(self):
    """仅接受非空的连续浮点 tensor 列表。"""
    if len(self) == 0:
        return False
    try:
        for tensor in self:
            if tensor.dtype not in _FLOAT_DTYPES or tensor.dtype.is_complex:
                return False
            if not tensor.is_contiguous():
                return False
    except AttributeError:
        # 非 Tensor 元素：返回 False 让调用方回退公共 apply_unary 对齐异常。
        return False
    return True


def _run_flat_kernel(self, inplace):
    if FALLBACK_TO_PUBLIC or not _can_use_flat_kernel(self):
        return apply_unary("sin", self, inplace)
    tensors = list(self)
    if not tensors:
        return [] if not inplace else None

    # 反序只改变 launch 次序，out-of-place 的输出必须保持与原列表一致。
    launch_indices = list(range(len(tensors)))
    if REVERSE_LAUNCH_ORDER:
        launch_indices.reverse()
    outputs = [None] * len(tensors) if not inplace else None
    device_context = (
        torch_device_fn.device(tensors[0].device)
        if USE_DEVICE_CONTEXT
        else nullcontext()
    )
    with device_context:
        for index in launch_indices:
            tensor = tensors[index]
            num_elems = tensor.numel()
            out_tensor = tensor if inplace else torch.empty_like(tensor)
            # 空 tensor 无元素可算，跳过 kernel launch；但输出列表必须仍保留
            # 这个空结果，保持与输入一一对应（foreach 的返回长度等于输入长度）。
            if num_elems == 0:
                if not inplace:
                    outputs[index] = out_tensor
                continue
            grid = (min(triton.cdiv(num_elems, BLOCK_SIZE), FIXED_GRID),)
            _foreach_sin_flat_kernel[grid](
                tensor,
                out_tensor,
                num_elems,
                BLOCK_SIZE,
                FIXED_GRID,
                num_warps=NUM_WARPS,
                num_stages=NUM_STAGES,
            )
            if not inplace:
                outputs[index] = out_tensor
    return None if inplace else outputs


def _foreach_sin(self):
    logger.debug("GEMS_GCU400 _FOREACH_SIN")
    return _run_flat_kernel(self, inplace=False)
