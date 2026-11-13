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

"""GCU400 专属 foreach ScalarList 最大值类算子。

``_foreach_clamp_min_.ScalarList`` 与 ``_foreach_maximum.ScalarList`` 在公共
codegen 执行器里用二维 grid 承载列表，GCU400 的 ``grid.y <= 255`` 在
16×(1024×1024) 上直接 ``OutOfResources``，benchmark 无任何读数。本文件把
输入地址、元素数与每个 float32 标量位模式打包进同一张 int64 metadata，
用一维固定 grid 的 ``@triton.jit`` kernel 一次 launch 完成整个列表。

原地 overload 的 metadata 完全只读且不含张量数据，可以安全按
``(device, 地址, numel, 标量位模式)`` 缓存，消除重复的同步 H2D；
out-of-place 的输出地址每次分配不同，单独用一张小表传入，静态表仍命中缓存。
非浮点、非连续、标量个数不匹配等形态回退公共执行器，dtype policy 与
异常语义不变。
"""

import logging
import struct

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_binary import _apply_scalar
from flag_gems.utils import libentry

from ._foreach_flat import (
    _TL_DTYPES,
    FALLBACK,
    MODE_SCALAR_LIST,
    OP_MAX,
    can_use_flat,
    run_binary,
)

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# 编译期/启动期配置，按优化轮次调整，不改变算子语义。
FIXED_GRID = 48
BLOCK_SIZE = 8192
NUM_WARPS = 4
NUM_STAGES = 2
FP32_PROMOTE = 1
FUSED_LAUNCH = 1
FALLBACK_TO_PUBLIC = 0
USE_SINGLE_H2D = 1
USE_META_CACHE = 1
META_CACHE_LIMIT = 32
# --- TUNING PARAMS END ---


@libentry()
@triton.jit
def fused_max_scalarlist_kernel(
    meta_ptr,
    out_ptr_meta,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    FP32_PROMOTE: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_OUT: tl.constexpr,
):
    """一次 launch 对整个列表做 ``max(tensor, scalar_i)``。

    ``meta_ptr`` 三段依次为输入地址、各元素数、各 fp32 标量的位模式；
    ``HAS_OUT=1`` 时从 ``out_ptr_meta`` 读取输出地址，否则原地写回输入。
    grid.x 按 ``(tensor, part)`` 线性展开，规避 ``grid.y <= 255`` 限制。
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tensor_id = pid % NT
    part_id = pid // NT
    num_parts = tl.cdiv(num_programs, NT)

    in_raw = tl.load(meta_ptr + tensor_id)
    num_elems = tl.load(meta_ptr + NT + tensor_id)
    scalar_raw = tl.load(meta_ptr + 2 * NT + tensor_id)
    if HAS_OUT:
        out_raw = tl.load(out_ptr_meta + tensor_id)
    else:
        out_raw = in_raw
    in_ptr = in_raw.to(tl.pointer_type(A_DT))
    out_ptr = out_raw.to(tl.pointer_type(A_DT))
    scalar = tl.cast(scalar_raw.to(tl.int32), tl.float32, bitcast=True)

    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(part_id, tl.cdiv(num_elems, BLOCK), num_parts):
        offs = block_id * BLOCK + arange
        mask = offs < num_elems
        value = tl.load(in_ptr + offs, mask=mask)
        if FP32_PROMOTE:
            result = tl.maximum(value.to(tl.float32), scalar)
        else:
            result = tl.maximum(value, scalar.to(value.dtype))
        tl.store(out_ptr + offs, result, mask=mask)


# 只读 metadata 缓存：key 只含设备、地址、元素数与标量位模式，不含张量数据；
# kernel 每次仍完整读取输入并重新计算，地址复用或内容被原地修改都不影响正确性。
_META_CACHE = {}


def _get_cached_meta(values, device):
    cache_key = (str(device),) + tuple(values)
    cached = _META_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if len(_META_CACHE) >= META_CACHE_LIMIT:
        _META_CACHE.clear()
    cached = torch.tensor(values, dtype=torch.int64, device=device)
    _META_CACHE[cache_key] = cached
    return cached


def _run_max_scalarlist(self, scalars, inplace):
    """单 H2D 快路径；不适用形态返回 FALLBACK 交给公共执行器。"""
    if not can_use_flat(self, scalars, MODE_SCALAR_LIST):
        return FALLBACK
    num_tensors = len(self)
    scalar_bits = [
        struct.unpack("<i", struct.pack("<f", float(value)))[0] for value in scalars
    ]
    static_values = (
        [tensor.data_ptr() for tensor in self]
        + [tensor.numel() for tensor in self]
        + scalar_bits
    )
    if USE_META_CACHE:
        meta = _get_cached_meta(static_values, self[0].device)
    else:
        meta = torch.tensor(static_values, dtype=torch.int64, device=self[0].device)

    if inplace:
        outputs = self
        out_ptr_meta = meta
        has_out = 0
    else:
        outputs = [torch.empty_like(tensor) for tensor in self]
        out_ptr_meta = torch.tensor(
            [tensor.data_ptr() for tensor in outputs],
            dtype=torch.int64,
            device=self[0].device,
        )
        has_out = 1

    blocks_per_tensor = triton.cdiv(max(tensor.numel() for tensor in self), BLOCK_SIZE)
    grid_size = max(1, min(FIXED_GRID, num_tensors * blocks_per_tensor))
    fused_max_scalarlist_kernel[(grid_size,)](
        meta,
        out_ptr_meta,
        num_tensors,
        _TL_DTYPES[self[0].dtype],
        FP32_PROMOTE,
        BLOCK_SIZE,
        has_out,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return None if inplace else outputs


def _dispatch(self, scalars, inplace, name):
    """按 tuning 开关在公共执行器、通用融合与单 H2D 快路径之间分派。"""
    if FALLBACK_TO_PUBLIC:
        return _apply_scalar(name, self, scalars, inplace=inplace)
    if USE_SINGLE_H2D:
        result = _run_max_scalarlist(self, scalars, inplace)
        if result is not FALLBACK:
            return result
        return _apply_scalar(name, self, scalars, inplace=inplace)
    result = run_binary(
        self,
        scalars,
        OP_MAX,
        MODE_SCALAR_LIST,
        inplace,
        fixed_grid=FIXED_GRID,
        block_size=BLOCK_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        use_device_context=0,
        fp32_promote=FP32_PROMOTE,
        fused=FUSED_LAUNCH,
    )
    if result is FALLBACK:
        return _apply_scalar(name, self, scalars, inplace=inplace)
    return result


def _foreach_clamp_min__ScalarList(self, scalars):
    logger.debug("GEMS_GCU400 _FOREACH_CLAMP_MIN__SCALARLIST")
    return _dispatch(self, scalars, True, "clamp_min")


def _foreach_clamp_min_ScalarList(self, scalars):
    logger.debug("GEMS_GCU400 _FOREACH_CLAMP_MIN_SCALARLIST")
    return _dispatch(self, scalars, False, "clamp_min")
