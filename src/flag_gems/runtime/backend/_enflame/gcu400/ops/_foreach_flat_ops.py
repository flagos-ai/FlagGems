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
"""GCU400 专属 foreach 二元实现：用一维 grid 规避 ``grid.y`` 硬件上限。

公共 codegen 执行器以二维 ``grid=(block, tensor)`` 承载 tensor 列表，
GCU400 的 ``grid.y`` 上限为 255，benchmark 的大 shape 会直接抛
``out of resource: grid.y``。这里复用 ``_foreach_flat`` 的一维固定
grid / metadata 融合 kernel 完成同类计算；非浮点、非连续、广播等
形态仍委托公共执行器，保证 dtype policy 与异常语义不变。
"""

import logging
import struct

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_binary import _apply_list, _apply_scalar
from flag_gems.utils import libentry

from ._foreach_flat import (
    _TL_DTYPES,
    FALLBACK,
    MODE_LIST,
    MODE_SCALAR,
    MODE_SCALAR_LIST,
    OP_ADD,
    OP_DIV,
    OP_MIN,
    can_use_flat,
    run_binary,
)

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def fused_add_scalarlist_kernel(
    meta_ptr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    FP32_PROMOTE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """ScalarList 加法的单 H2D 融合 kernel。

    ``meta_ptr`` 三段依次为输入地址、各 tensor 元素数、各 float32 标量的
    位模式（以 int64 承载）。相比通用 ``run_binary`` 少一次标量数组 H2D，
    同步拷贝的固定开销可减半；grid.x 按 ``(tensor, part)`` 线性展开。
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tensor_id = pid % NT
    part_id = pid // NT
    num_parts = tl.cdiv(num_programs, NT)

    a_raw = tl.load(meta_ptr + tensor_id)
    num_elems = tl.load(meta_ptr + NT + tensor_id)
    scalar_raw = tl.load(meta_ptr + 2 * NT + tensor_id)
    a_ptr = a_raw.to(tl.pointer_type(A_DT))
    scalar = tl.cast(scalar_raw.to(tl.int32), tl.float32, bitcast=True)

    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(part_id, tl.cdiv(num_elems, BLOCK), num_parts):
        offs = block_id * BLOCK + arange
        mask = offs < num_elems
        a = tl.load(a_ptr + offs, mask=mask)
        if FP32_PROMOTE:
            a = a.to(tl.float32)
        tl.store(a_ptr + offs, a + scalar, mask=mask)


# --- TUNING PARAMS START (foreach_div_list_) ---
# 编译期/启动期配置，按优化轮次调整，不改变算子语义。
FIXED_GRID_DIV = 48
BLOCK_SIZE_DIV = 8192
NUM_WARPS_DIV = 4
NUM_STAGES_DIV = 2
USE_DEVICE_CONTEXT_DIV = 0
FP32_PROMOTE_DIV = 1
FUSED_LAUNCH_DIV = 1
FALLBACK_TO_PUBLIC_DIV = 0
# --- TUNING PARAMS END (foreach_div_list_) ---


def _foreach_div__List(self, other):
    """原地 ``self /= other``（逐 tensor 配对），一维融合 kernel 实现。"""
    logger.debug("GEMS_GCU400 _FOREACH_DIV__LIST")

    def fallback():
        return _apply_list("div", self, other, inplace=True)

    if FALLBACK_TO_PUBLIC_DIV:
        return fallback()
    result = run_binary(
        self,
        other,
        OP_DIV,
        MODE_LIST,
        True,
        fixed_grid=FIXED_GRID_DIV,
        block_size=BLOCK_SIZE_DIV,
        num_warps=NUM_WARPS_DIV,
        num_stages=NUM_STAGES_DIV,
        use_device_context=USE_DEVICE_CONTEXT_DIV,
        fp32_promote=FP32_PROMOTE_DIV,
        fused=FUSED_LAUNCH_DIV,
    )
    if result is FALLBACK:
        return fallback()
    return result


# --- TUNING PARAMS START (foreach_add_scalar_list_) ---
# 编译期/启动期配置，按优化轮次调整，不改变算子语义。
FIXED_GRID_ADD = 48
BLOCK_SIZE_ADD = 8192
NUM_WARPS_ADD = 4
NUM_STAGES_ADD = 2
USE_DEVICE_CONTEXT_ADD = 0
FP32_PROMOTE_ADD = 1
FUSED_LAUNCH_ADD = 1
FALLBACK_TO_PUBLIC_ADD = 0
USE_SINGLE_H2D_ADD = 1
USE_META_CACHE_ADD = 1
META_CACHE_LIMIT_ADD = 32
# --- TUNING PARAMS END (foreach_add_scalar_list_) ---

# 只读 metadata 缓存：key 由设备、每个 tensor 的地址/元素数与标量位模式组成，
# 不含任何张量数据；kernel 每次仍完整读取并计算，仅省去重复的 H2D 同步拷贝。
_META_CACHE = {}


def _get_cached_meta(values, device):
    cache_key = (str(device),) + tuple(values)
    cached = _META_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if len(_META_CACHE) >= META_CACHE_LIMIT_ADD:
        _META_CACHE.clear()
    cached = torch.tensor(values, dtype=torch.int64, device=device)
    _META_CACHE[cache_key] = cached
    return cached


def _run_add_scalarlist_single_h2d(self, scalars):
    """把地址表与标量位模式合并为一次 H2D，避免两次同步拷贝。"""
    if not can_use_flat(self, scalars, MODE_SCALAR_LIST):
        return FALLBACK
    num_tensors = len(self)
    scalar_bits = [
        struct.unpack("<i", struct.pack("<f", float(value)))[0] for value in scalars
    ]
    meta_values = (
        [tensor.data_ptr() for tensor in self]
        + [tensor.numel() for tensor in self]
        + scalar_bits
    )
    if USE_META_CACHE_ADD:
        meta = _get_cached_meta(meta_values, self[0].device)
    else:
        meta = torch.tensor(meta_values, dtype=torch.int64, device=self[0].device)
    blocks_per_tensor = triton.cdiv(
        max(tensor.numel() for tensor in self), BLOCK_SIZE_ADD
    )
    grid_size = max(1, min(FIXED_GRID_ADD, num_tensors * blocks_per_tensor))
    fused_add_scalarlist_kernel[(grid_size,)](
        meta,
        num_tensors,
        _TL_DTYPES[self[0].dtype],
        FP32_PROMOTE_ADD,
        BLOCK_SIZE_ADD,
        num_warps=NUM_WARPS_ADD,
        num_stages=NUM_STAGES_ADD,
    )
    return None


def _foreach_add__ScalarList(self, scalars):
    """原地 ``self += scalars[i]``，一维融合 kernel 承载整个标量列表。"""
    logger.debug("GEMS_GCU400 _FOREACH_ADD__SCALARLIST")

    def fallback():
        return _apply_scalar("add", self, scalars, inplace=True)

    if FALLBACK_TO_PUBLIC_ADD:
        return fallback()
    if USE_SINGLE_H2D_ADD:
        single_h2d_result = _run_add_scalarlist_single_h2d(self, scalars)
        if single_h2d_result is not FALLBACK:
            return single_h2d_result
        return fallback()
    result = run_binary(
        self,
        scalars,
        OP_ADD,
        MODE_SCALAR_LIST,
        True,
        fixed_grid=FIXED_GRID_ADD,
        block_size=BLOCK_SIZE_ADD,
        num_warps=NUM_WARPS_ADD,
        num_stages=NUM_STAGES_ADD,
        use_device_context=USE_DEVICE_CONTEXT_ADD,
        fp32_promote=FP32_PROMOTE_ADD,
        fused=FUSED_LAUNCH_ADD,
    )
    if result is FALLBACK:
        return fallback()
    return result


@libentry()
@triton.jit
def fused_clamp_max_scalar_kernel(
    meta_ptr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    FP32_PROMOTE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """clamp_max 的单 H2D 融合 kernel。

    ``meta_ptr`` 前三段依次为输入地址、各 tensor 元素数、唯一的 float32
    标量位模式（以 int64 承载）。aten 的 clamp_max 语义是逐元素取最小值，
    因此编译期分派到 ``tl.minimum``。地址表与标量合并为一次 H2D，并支持
    只读 metadata 缓存以消除重复调用的同步拷贝。
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tensor_id = pid % NT
    part_id = pid // NT
    num_parts = tl.cdiv(num_programs, NT)

    a_raw = tl.load(meta_ptr + tensor_id)
    num_elems = tl.load(meta_ptr + NT + tensor_id)
    scalar_raw = tl.load(meta_ptr + 2 * NT)
    a_ptr = a_raw.to(tl.pointer_type(A_DT))
    scalar = tl.cast(scalar_raw.to(tl.int32), tl.float32, bitcast=True)

    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(part_id, tl.cdiv(num_elems, BLOCK), num_parts):
        offs = block_id * BLOCK + arange
        mask = offs < num_elems
        a = tl.load(a_ptr + offs, mask=mask)
        if FP32_PROMOTE:
            result = tl.minimum(a.to(tl.float32), scalar)
        else:
            result = tl.minimum(a, scalar.to(A_DT))
        tl.store(a_ptr + offs, result, mask=mask)


# --- TUNING PARAMS START (foreach_clamp_max_scalar_) ---
# 编译期/启动期配置，按优化轮次调整，不改变算子语义。
FIXED_GRID_CLAMP = 48
BLOCK_SIZE_CLAMP = 8192
NUM_WARPS_CLAMP = 4
NUM_STAGES_CLAMP = 2
USE_DEVICE_CONTEXT_CLAMP = 0
FP32_PROMOTE_CLAMP = 0
FUSED_LAUNCH_CLAMP = 1
FALLBACK_TO_PUBLIC_CLAMP = 0
USE_SINGLE_H2D_CLAMP = 1
USE_META_CACHE_CLAMP = 1
META_CACHE_LIMIT_CLAMP = 128
# --- TUNING PARAMS END (foreach_clamp_max_scalar_) ---


def _run_clamp_max_scalar_single_h2d(self, scalar):
    """地址表、元素数与唯一标量位模式合并为一次 H2D。"""
    if not can_use_flat(self, scalar, MODE_SCALAR):
        return FALLBACK
    num_tensors = len(self)
    scalar_bits = struct.unpack("<i", struct.pack("<f", float(scalar)))[0]
    meta_values = (
        [tensor.data_ptr() for tensor in self]
        + [tensor.numel() for tensor in self]
        + [scalar_bits]
    )
    if USE_META_CACHE_CLAMP:
        if len(_META_CACHE) >= META_CACHE_LIMIT_CLAMP:
            _META_CACHE.clear()
        cache_key = (str(self[0].device),) + tuple(meta_values)
        meta = _META_CACHE.get(cache_key)
        if meta is None:
            meta = torch.tensor(meta_values, dtype=torch.int64, device=self[0].device)
            _META_CACHE[cache_key] = meta
    else:
        meta = torch.tensor(meta_values, dtype=torch.int64, device=self[0].device)
    blocks_per_tensor = triton.cdiv(
        max(tensor.numel() for tensor in self), BLOCK_SIZE_CLAMP
    )
    grid_size = max(1, min(FIXED_GRID_CLAMP, num_tensors * blocks_per_tensor))
    fused_clamp_max_scalar_kernel[(grid_size,)](
        meta,
        num_tensors,
        _TL_DTYPES[self[0].dtype],
        FP32_PROMOTE_CLAMP,
        BLOCK_SIZE_CLAMP,
        num_warps=NUM_WARPS_CLAMP,
        num_stages=NUM_STAGES_CLAMP,
    )
    return None


def _foreach_clamp_max__Scalar(self, scalar):
    """原地 ``self = minimum(self, scalar)``；aten 的 clamp_max 语义为取最小值。"""
    logger.debug("GEMS_GCU400 _FOREACH_CLAMP_MAX__SCALAR")

    def fallback():
        return _apply_scalar("clamp_max", self, scalar, inplace=True)

    if FALLBACK_TO_PUBLIC_CLAMP:
        return fallback()
    if USE_SINGLE_H2D_CLAMP:
        single_h2d_result = _run_clamp_max_scalar_single_h2d(self, scalar)
        if single_h2d_result is not FALLBACK:
            return single_h2d_result
        return fallback()
    result = run_binary(
        self,
        scalar,
        OP_MIN,
        MODE_SCALAR,
        True,
        fixed_grid=FIXED_GRID_CLAMP,
        block_size=BLOCK_SIZE_CLAMP,
        num_warps=NUM_WARPS_CLAMP,
        num_stages=NUM_STAGES_CLAMP,
        use_device_context=USE_DEVICE_CONTEXT_CLAMP,
        fp32_promote=FP32_PROMOTE_CLAMP,
        fused=FUSED_LAUNCH_CLAMP,
    )
    if result is FALLBACK:
        return fallback()
    return result
