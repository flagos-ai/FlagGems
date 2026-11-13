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

"""GCU400 专属 _foreach_mul / _foreach_mul_（全部 4 种 overload）。

公共 foreach 执行器用二维 grid 承载 tensor 列表，GCU400 的 ``grid.y``
上限 255 在大 shape 下会 ``out of resource``。浮点连续列表改走
``_foreach_flat`` 的一维固定 grid / metadata 融合 kernel；非浮点、
非连续或广播等形态仍委托公共执行器保持 dtype policy 与异常语义。
"""

import logging
from contextlib import nullcontext

import torch

from flag_gems.ops._foreach_binary import _apply_list, _apply_scalar, _apply_tensor
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.foreach import _device_meta

from ._foreach_flat import (
    _TL_DTYPES,
    FALLBACK,
    MODE_LIST,
    MODE_SCALAR,
    MODE_SCALAR_LIST,
    MODE_TENSOR,
    OP_MUL,
    _fused_grid,
    can_use_flat,
    fused_binary_list_kernel,
    run_binary,
)

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# 由优化迭代驱动脚本按轮次生成；均为编译期/启动期配置，不改变算子语义。
FIXED_GRID = 48
BLOCK_SIZE = 8192
NUM_WARPS = 4
NUM_STAGES = 1
USE_DEVICE_CONTEXT = 0
FP32_PROMOTE = 0
FUSED_LAUNCH = 1
FALLBACK_TO_PUBLIC = 0
# .List 专用：用公共 _device_meta 缓存替代每次重建 int64 metadata。
USE_CACHED_META = 1
# 小 shape 走公共 binary 执行器（其二维 grid 未越界且缓存更完整）。
PUBLIC_SMALL_SHAPE = 0
SMALL_SHAPE_MAX_NUMEL = 261120
# --- TUNING PARAMS END ---


def _run_mul_list_cached(self, other, outputs):
    """复用公共 _device_meta 缓存构造 int64 metadata，避免每次同步 H2D。

    metadata 的四段布局（输入/输出/第二操作数地址 + 各元素数）必须与
    _foreach_flat._fused_meta 完全一致，否则融合 kernel 会读错指针。
    """
    num_tensors = len(self)
    pointers = (
        [tensor.data_ptr() for tensor in self]
        + [tensor.data_ptr() for tensor in outputs]
        + [other_tensor.data_ptr() for other_tensor in other]
        + [tensor.numel() for tensor in self]
    )
    meta = _device_meta(pointers, self[0].device)
    grid = (_fused_grid(self, FIXED_GRID, BLOCK_SIZE),)
    device_context = (
        torch_device_fn.device(self[0].device) if USE_DEVICE_CONTEXT else nullcontext()
    )
    with device_context:
        fused_binary_list_kernel[grid](
            meta,
            num_tensors,
            OP_MUL,
            _TL_DTYPES[self[0].dtype],
            FP32_PROMOTE,
            BLOCK_SIZE,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
    return outputs


def _foreach_mul_List(self, other):
    logger.debug("GEMS_GCU400 _FOREACH_MUL_LIST")

    def fallback():
        return _apply_list("mul", self, other, inplace=False)

    if FALLBACK_TO_PUBLIC:
        return fallback()
    if (
        PUBLIC_SMALL_SHAPE
        and self
        and max(t.numel() for t in self) <= SMALL_SHAPE_MAX_NUMEL
    ):
        return fallback()
    if USE_CACHED_META and can_use_flat(self, other, MODE_LIST):
        fused_ready = all(
            other_tensor.numel() == tensor.numel()
            for tensor, other_tensor in zip(self, other)
        )
        if fused_ready:
            outputs = [torch.empty_like(tensor) for tensor in self]
            return _run_mul_list_cached(self, other, outputs)
    result = run_binary(
        self,
        other,
        OP_MUL,
        MODE_LIST,
        False,
        fixed_grid=FIXED_GRID,
        block_size=BLOCK_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        use_device_context=USE_DEVICE_CONTEXT,
        fp32_promote=FP32_PROMOTE,
        fused=FUSED_LAUNCH,
    )
    if result is FALLBACK:
        return fallback()
    return result


def _foreach_mul_Scalar(self, scalar):
    logger.debug("GEMS_GCU400 _FOREACH_MUL_SCALAR")

    def fallback():
        return _apply_scalar("mul", self, scalar, inplace=False)

    if FALLBACK_TO_PUBLIC:
        return fallback()
    result = run_binary(
        self,
        scalar,
        OP_MUL,
        MODE_SCALAR,
        False,
        fixed_grid=FIXED_GRID,
        block_size=BLOCK_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        use_device_context=USE_DEVICE_CONTEXT,
        fp32_promote=FP32_PROMOTE,
        fused=FUSED_LAUNCH,
    )
    if result is FALLBACK:
        return fallback()
    return result


def _foreach_mul_ScalarList(self, scalars):
    logger.debug("GEMS_GCU400 _FOREACH_MUL_SCALARLIST")

    def fallback():
        return _apply_scalar("mul", self, scalars, inplace=False)

    if FALLBACK_TO_PUBLIC:
        return fallback()
    result = run_binary(
        self,
        scalars,
        OP_MUL,
        MODE_SCALAR_LIST,
        False,
        fixed_grid=FIXED_GRID,
        block_size=BLOCK_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        use_device_context=USE_DEVICE_CONTEXT,
        fp32_promote=FP32_PROMOTE,
        fused=FUSED_LAUNCH,
    )
    if result is FALLBACK:
        return fallback()
    return result


def _foreach_mul_Tensor(self, other):
    logger.debug("GEMS_GCU400 _FOREACH_MUL_TENSOR")

    def fallback():
        return _apply_tensor("mul", self, other, inplace=False)

    if FALLBACK_TO_PUBLIC:
        return fallback()
    result = run_binary(
        self,
        other,
        OP_MUL,
        MODE_TENSOR,
        False,
        fixed_grid=FIXED_GRID,
        block_size=BLOCK_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        use_device_context=USE_DEVICE_CONTEXT,
        fp32_promote=FP32_PROMOTE,
        fused=FUSED_LAUNCH,
    )
    if result is FALLBACK:
        return fallback()
    return result


def _foreach_mul__List(self, other):
    logger.debug("GEMS_GCU400 _FOREACH_MUL__LIST")

    def fallback():
        return _apply_list("mul", self, other, inplace=True)

    if FALLBACK_TO_PUBLIC:
        return fallback()
    result = run_binary(
        self,
        other,
        OP_MUL,
        MODE_LIST,
        True,
        fixed_grid=FIXED_GRID,
        block_size=BLOCK_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        use_device_context=USE_DEVICE_CONTEXT,
        fp32_promote=FP32_PROMOTE,
        fused=FUSED_LAUNCH,
    )
    if result is FALLBACK:
        return fallback()
    return result


def _foreach_mul__Scalar(self, scalar):
    logger.debug("GEMS_GCU400 _FOREACH_MUL__SCALAR")

    def fallback():
        return _apply_scalar("mul", self, scalar, inplace=True)

    if FALLBACK_TO_PUBLIC:
        return fallback()
    result = run_binary(
        self,
        scalar,
        OP_MUL,
        MODE_SCALAR,
        True,
        fixed_grid=FIXED_GRID,
        block_size=BLOCK_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        use_device_context=USE_DEVICE_CONTEXT,
        fp32_promote=FP32_PROMOTE,
        fused=FUSED_LAUNCH,
    )
    if result is FALLBACK:
        return fallback()
    return result


def _foreach_mul__ScalarList(self, scalars):
    logger.debug("GEMS_GCU400 _FOREACH_MUL__SCALARLIST")

    def fallback():
        return _apply_scalar("mul", self, scalars, inplace=True)

    if FALLBACK_TO_PUBLIC:
        return fallback()
    result = run_binary(
        self,
        scalars,
        OP_MUL,
        MODE_SCALAR_LIST,
        True,
        fixed_grid=FIXED_GRID,
        block_size=BLOCK_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        use_device_context=USE_DEVICE_CONTEXT,
        fp32_promote=FP32_PROMOTE,
        fused=FUSED_LAUNCH,
    )
    if result is FALLBACK:
        return fallback()
    return result


def _foreach_mul__Tensor(self, other):
    logger.debug("GEMS_GCU400 _FOREACH_MUL__TENSOR")

    def fallback():
        return _apply_tensor("mul", self, other, inplace=True)

    if FALLBACK_TO_PUBLIC:
        return fallback()
    result = run_binary(
        self,
        other,
        OP_MUL,
        MODE_TENSOR,
        True,
        fixed_grid=FIXED_GRID,
        block_size=BLOCK_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        use_device_context=USE_DEVICE_CONTEXT,
        fp32_promote=FP32_PROMOTE,
        fused=FUSED_LAUNCH,
    )
    if result is FALLBACK:
        return fallback()
    return result
