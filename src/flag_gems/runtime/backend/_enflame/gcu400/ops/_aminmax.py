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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._aminmax import _aminmax as _generic_aminmax
from flag_gems.ops._aminmax import _aminmax_out as _generic_aminmax_out
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_max, get_dtype_min

logger = logging.getLogger(__name__)

__all__ = ["_aminmax", "_aminmax_out"]

# GCU400 归约亲和参数：固定 grid 匹配 SIP 拓扑，核内跨步循环复用 CTA；
# 大 BLOCK 摊薄 DTE 启动开销。取值由批量 A/B 扫描确定。
_GCU400_AMINMAX_USE_SPECIALIZED = True
_GCU400_AMINMAX_GRID = 24
_GCU400_AMINMAX_BLOCK = 16384
_GCU400_AMINMAX_NUM_WARPS = 4


@libentry()
@triton.jit
def _gcu400_aminmax_partial_kernel(
    inp,
    min_partial,
    max_partial,
    num_elements,
    BLOCK: tl.constexpr,
    GRID: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    IS_FLOAT: tl.constexpr,
):
    """第一阶段：固定数量的 CTA 跨步扫描全张量，各写出一个局部极值。"""
    program_id = tl.program_id(0)
    min_fill = get_dtype_max(ACC_DTYPE)
    max_fill = get_dtype_min(ACC_DTYPE)

    # 累加器恒为 fp32 / int32：GCU400 编译器无法在 bf16 上构造 inf 常量。
    tile_min = tl.full((BLOCK,), min_fill, ACC_DTYPE)
    tile_max = tl.full((BLOCK,), max_fill, ACC_DTYPE)
    nan_count = tl.zeros((BLOCK,), tl.int32)
    num_tiles = tl.cdiv(num_elements, BLOCK)

    for tile_index in range(program_id, num_tiles, GRID):
        element_offset = tile_index * BLOCK + tl.arange(0, BLOCK)
        valid_mask = element_offset < num_elements
        # other=0 避免越界填充在窄 dtype 上触发类型转换；掩码后再补极值。
        values = tl.load(inp + element_offset, mask=valid_mask, other=0).to(ACC_DTYPE)
        values_for_min = tl.where(valid_mask, values, min_fill)
        values_for_max = tl.where(valid_mask, values, max_fill)
        tile_min = tl.minimum(tile_min, values_for_min)
        tile_max = tl.maximum(tile_max, values_for_max)
        if IS_FLOAT:
            # tl.minimum/tl.maximum 会丢弃 NaN，需显式计数并在写出时污染结果。
            nan_count += (values != values).to(tl.int32)

    partial_min = tl.min(tile_min)
    partial_max = tl.max(tile_max)
    if IS_FLOAT:
        tile_has_nan = tl.sum(nan_count) > 0
        nan_value = float("nan")
        partial_min = tl.where(tile_has_nan, nan_value, partial_min)
        partial_max = tl.where(tile_has_nan, nan_value, partial_max)

    tl.store(min_partial + program_id, partial_min.to(min_partial.type.element_ty))
    tl.store(max_partial + program_id, partial_max.to(max_partial.type.element_ty))


@libentry()
@triton.jit
def _gcu400_aminmax_final_kernel(
    min_partial,
    max_partial,
    min_out,
    max_out,
    GRID: tl.constexpr,
    BLOCK_FINAL: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    IS_FLOAT: tl.constexpr,
):
    """第二阶段：单个 CTA 在 fp32 / int32 上归约第一阶段的部分极值。"""
    partial_offset = tl.arange(0, BLOCK_FINAL)
    valid_mask = partial_offset < GRID
    min_fill = get_dtype_max(ACC_DTYPE)
    max_fill = get_dtype_min(ACC_DTYPE)
    partial_mins = tl.load(min_partial + partial_offset, mask=valid_mask, other=0).to(
        ACC_DTYPE
    )
    partial_maxs = tl.load(max_partial + partial_offset, mask=valid_mask, other=0).to(
        ACC_DTYPE
    )
    partial_mins = tl.where(valid_mask, partial_mins, min_fill)
    partial_maxs = tl.where(valid_mask, partial_maxs, max_fill)

    final_min = tl.min(partial_mins)
    final_max = tl.max(partial_maxs)
    if IS_FLOAT:
        # 任一部分结果带 NaN，两个输出都必须是 NaN（与 torch._aminmax 一致）。
        has_nan = (
            tl.sum((partial_mins != partial_mins).to(tl.int32))
            + tl.sum((partial_maxs != partial_maxs).to(tl.int32))
        ) > 0
        nan_value = float("nan")
        final_min = tl.where(has_nan, nan_value, final_min)
        final_max = tl.where(has_nan, nan_value, final_max)

    tl.store(min_out, final_min.to(min_out.type.element_ty))
    tl.store(max_out, final_max.to(max_out.type.element_ty))


def _gcu400_aminmax_run(inp, out0=None, out1=None):
    """GCU400 专属 driver：处理输入/view 与 ``.out`` 语义后启动两阶段归约。"""
    if inp.numel() == 0:
        # ATen 对空张量拒绝归约，而不是返回正负无穷。
        raise RuntimeError(
            "aminmax(): cannot compute aminmax over an empty dimension as the "
            "operation has no identity."
        )

    # kernel 以平坦偏移索引，非连续 / expand 视图必须先物化为稠密副本。
    inp = inp.contiguous()
    num_elements = inp.numel()
    dtype = inp.dtype
    is_float = inp.is_floating_point()

    if out0 is None:
        min_out = torch.empty([], dtype=dtype, device=inp.device)
        max_out = torch.empty([], dtype=dtype, device=inp.device)
    else:
        for name, output in (("out0", out0), ("out1", out1)):
            if output.dtype != dtype:
                raise RuntimeError(
                    f"Expected out tensor to have dtype {dtype}, but got "
                    f"{output.dtype} instead"
                )
            if output.device != inp.device:
                raise RuntimeError(
                    f"Expected out tensor to have device {inp.device}, but got "
                    f"{output.device} instead"
                )
        # ATen 会把形状不符的输出原地 resize 成标量，并保持对象别名。
        if out0.shape != torch.Size([]):
            out0.resize_(())
        if out1.shape != torch.Size([]):
            out1.resize_(())
        min_out, max_out = out0, out1

    grid = _GCU400_AMINMAX_GRID
    block = _GCU400_AMINMAX_BLOCK
    acc_dtype = tl.float32 if is_float else tl.int32
    min_partial = torch.empty((grid,), dtype=dtype, device=inp.device)
    max_partial = torch.empty((grid,), dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        _gcu400_aminmax_partial_kernel[(grid,)](
            inp,
            min_partial,
            max_partial,
            num_elements,
            block,
            grid,
            acc_dtype,
            IS_FLOAT=is_float,
            num_warps=_GCU400_AMINMAX_NUM_WARPS,
        )
        _gcu400_aminmax_final_kernel[(1,)](
            min_partial,
            max_partial,
            min_out,
            max_out,
            grid,
            triton.next_power_of_2(grid),
            acc_dtype,
            IS_FLOAT=is_float,
            num_warps=_GCU400_AMINMAX_NUM_WARPS,
        )
    return min_out, max_out


def _aminmax(inp):
    """gcu400 ``_aminmax``。

    GCU 不支持 fp64：torch_gcu 已把 fp64 aten 用 float 替换，但生成的 Triton
    kernel 中 fp64 NaN 检测无法合法化。对 fp32 视图归约后再恢复输入 dtype。
    """
    if inp.dtype == torch.float64:
        min_out, max_out = _gcu400_aminmax_run(inp.to(torch.float32))
        return min_out.to(torch.float64), max_out.to(torch.float64)
    if not _GCU400_AMINMAX_USE_SPECIALIZED:
        return _generic_aminmax(inp)
    return _gcu400_aminmax_run(inp)


def _aminmax_out(inp, *, out0, out1):
    if inp.dtype == torch.float64:
        min_out, max_out = _aminmax(inp)
        out0.resize_(())
        out1.resize_(())
        out0.copy_(min_out)
        out1.copy_(max_out)
        return out0, out1
    if not _GCU400_AMINMAX_USE_SPECIALIZED:
        return _generic_aminmax_out(inp, out0=out0, out1=out1)
    return _gcu400_aminmax_run(inp, out0=out0, out1=out1)
