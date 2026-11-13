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
"""GCU400 专属 ``_foreach_addcmul_.ScalarList`` 修复实现。

公共 codegen 执行器用二维 ``grid=(block, tensor)`` 承载 tensor 列表，
GCU400 的 ``grid.y <= 255`` 上限在 16×(1024×1024) 的 benchmark 形态下
直接抛 ``OutOfResources``，算子完全不可用。这里把 self / tensor1 /
tensor2 三组地址、各 tensor 元素数与每个 float32 标量位模式打包进
一张 int64 metadata，用一维固定 grid 的 ``@triton.jit`` kernel 一次
launch 完成整个列表的原地 ``self += value * tensor1 * tensor2``。

非浮点、非连续、列表长度或元素数不匹配等形态回退公共执行器，
dtype policy 与异常语义保持不变；metadata 只读且不含张量数据，
可按 (device, 地址, 元素数, 标量位模式) 缓存，消除重复的同步 H2D。
"""

import logging
import struct
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ._foreach_binary import _apply_ternary
from ._foreach_flat import _TL_DTYPES, FLOAT_DTYPES

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# 编译期/启动期配置，按优化轮次调整，不改变算子语义。
FIXED_GRID = 48
BLOCK_SIZE = 8192
NUM_WARPS = 4
NUM_STAGES = 2
FP32_PROMOTE = 1
USE_DEVICE_CONTEXT = 0
FALLBACK_TO_PUBLIC = 0
USE_META_CACHE = 1
META_CACHE_LIMIT = 32
USE_SCALAR_CONSTEXPR = 0
GRID_MODE = 0
PACK_NUMEL_SCALAR = 0
# --- TUNING PARAMS END ---


@libentry()
@triton.jit
def fused_addcmul_scalarlist_kernel(
    meta_ptr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    FP32_PROMOTE: tl.constexpr,
    USE_CONST_SCALAR: tl.constexpr,
    SCALAR_VALUE: tl.constexpr,
    PACKED_META: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """一次 launch 对整个列表做原地 addcmul。

    ``meta_ptr`` 依次存放 self / tensor1 / tensor2 的地址（各 NT 项）、
    元素数（NT 项）与 float32 标量位模式（NT 项，int64 承载）。
    ``grid.x`` 按 ``(tensor, part)`` 线性展开，规避 ``grid.y`` 上限。
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tensor_id = pid % NT
    part_id = pid // NT
    num_parts = tl.cdiv(num_programs, NT)

    a_raw = tl.load(meta_ptr + tensor_id)
    b_raw = tl.load(meta_ptr + NT + tensor_id)
    c_raw = tl.load(meta_ptr + 2 * NT + tensor_id)
    numel_scalar_raw = tl.load(meta_ptr + 3 * NT + tensor_id)

    a_ptr = a_raw.to(tl.pointer_type(A_DT))
    b_ptr = b_raw.to(tl.pointer_type(A_DT))
    c_ptr = c_raw.to(tl.pointer_type(A_DT))

    if PACKED_META:
        # 低 32 位是 float32 标量位模式，高位是元素数。
        num_elems = numel_scalar_raw >> 32
        scalar_raw = numel_scalar_raw & 0xFFFFFFFF
    else:
        num_elems = numel_scalar_raw
        scalar_raw = tl.load(meta_ptr + 4 * NT + tensor_id)
    if USE_CONST_SCALAR:
        value = SCALAR_VALUE
    else:
        value = tl.cast(scalar_raw.to(tl.int32), tl.float32, bitcast=True)

    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(part_id, tl.cdiv(num_elems, BLOCK), num_parts):
        offs = block_id * BLOCK + arange
        mask = offs < num_elems
        a = tl.load(a_ptr + offs, mask=mask)
        b = tl.load(b_ptr + offs, mask=mask)
        c = tl.load(c_ptr + offs, mask=mask)
        if FP32_PROMOTE:
            # 低精度先提升 fp32 计算，store 时隐式降回输出 dtype。
            a = a.to(tl.float32)
            b = b.to(tl.float32)
            c = c.to(tl.float32)
        tl.store(a_ptr + offs, a + value * b * c, mask=mask)


# 只读 metadata 缓存：key 由设备、三组地址、元素数与标量位模式组成，
# 不含任何张量数据；kernel 每次仍完整读取并计算，仅省去重复的 H2D 同步拷贝。
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


def _can_use_fused(self, tensor1, tensor2, scalars):
    """仅当三组列表都是同 dtype 的连续浮点张量且标量列表合法时走快路径。"""
    if not isinstance(scalars, (list, tuple)) or len(scalars) != len(self):
        return False
    if not self or len(tensor1) != len(self) or len(tensor2) != len(self):
        return False
    dtype = self[0].dtype
    if dtype not in FLOAT_DTYPES:
        return False
    for tensor in self:
        if tensor.dtype != dtype or not tensor.is_contiguous() or tensor.numel() == 0:
            return False
    for first, second, third in zip(self, tensor1, tensor2):
        if (
            second.dtype != dtype
            or third.dtype != dtype
            or not second.is_contiguous()
            or not third.is_contiguous()
            or second.numel() != first.numel()
            or third.numel() != first.numel()
        ):
            return False
    return all(isinstance(value, (int, float)) for value in scalars)


def _foreach_addcmul__ScalarList(self, tensor1, tensor2, value=1):
    """原地 ``self += scalars[i] * tensor1[i] * tensor2[i]``。"""
    logger.debug("GEMS_GCU400 _FOREACH_ADDCMUL__SCALARLIST")

    def fallback():
        return _apply_ternary("addcmul", self, tensor1, tensor2, value, True)

    if FALLBACK_TO_PUBLIC or not _can_use_fused(self, tensor1, tensor2, value):
        return fallback()

    num_tensors = len(self)
    scalar_bits = [
        struct.unpack("<i", struct.pack("<f", float(item)))[0] for item in value
    ]
    if PACK_NUMEL_SCALAR:
        # 把元素数与 float32 标量位模式压进同一个 int64，metadata 由 5 段缩到 4 段。
        packed = [
            (tensor.numel() << 32) | (bits & 0xFFFFFFFF)
            for tensor, bits in zip(self, scalar_bits)
        ]
        meta_values = (
            [tensor.data_ptr() for tensor in self]
            + [tensor.data_ptr() for tensor in tensor1]
            + [tensor.data_ptr() for tensor in tensor2]
            + packed
        )
    else:
        meta_values = (
            [tensor.data_ptr() for tensor in self]
            + [tensor.data_ptr() for tensor in tensor1]
            + [tensor.data_ptr() for tensor in tensor2]
            + [tensor.numel() for tensor in self]
            + scalar_bits
        )
    device = self[0].device
    if USE_META_CACHE:
        meta = _get_cached_meta(meta_values, device)
    else:
        meta = torch.tensor(meta_values, dtype=torch.int64, device=device)

    blocks_per_tensor = triton.cdiv(max(tensor.numel() for tensor in self), BLOCK_SIZE)
    if GRID_MODE == 1:
        grid_size = max(1, num_tensors * blocks_per_tensor)
    else:
        grid_size = max(1, min(FIXED_GRID, num_tensors * blocks_per_tensor))

    # 标量列表取值全都相同时可用编译期常量，省去一次 metadata 读取与位转换；
    # constexpr 取值变化会触发一次新的 Triton 特化，故仅作可选策略。
    use_const_scalar = bool(USE_SCALAR_CONSTEXPR and len(set(value)) == 1)
    scalar_value = float(value[0]) if use_const_scalar else 0.0

    device_context = (
        torch_device_fn.device(device) if USE_DEVICE_CONTEXT else nullcontext()
    )
    with device_context:
        fused_addcmul_scalarlist_kernel[(grid_size,)](
            meta,
            num_tensors,
            _TL_DTYPES[self[0].dtype],
            FP32_PROMOTE,
            use_const_scalar,
            scalar_value,
            PACK_NUMEL_SCALAR,
            BLOCK_SIZE,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
    return None
