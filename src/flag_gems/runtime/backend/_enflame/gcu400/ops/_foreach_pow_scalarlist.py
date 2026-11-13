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
"""GCU400 专属 ``_foreach_pow.ScalarList`` 修复实现。

公共 codegen 执行器的标量 pow 内核在 fp16 / bf16 上会让 GCU 的 LLVM
后端生成无法选择的 ``select_cc f16`` 节点，编译直接失败；大 shape 的
二维 ``grid.y`` 也会超过 255 的硬件上限。这里把输入 / 输出地址、元素数
与每个 float32 指数位模式打包进一张 int64 metadata，用一维固定 grid 的
``@triton.jit`` kernel 一次 launch 完成整个列表：pow 的两个操作数都先
提升到 fp32 再调用 libdevice powf，完全避开低精度标量比较。

非浮点、非连续、标量数量不匹配等形态回退公共执行器，dtype policy 与
异常语义保持不变；metadata 只读且不含张量数据，可按地址与标量缓存。
"""

import logging
import struct
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

from ._foreach_binary import _apply_scalar
from ._foreach_flat import _TL_DTYPES, MODE_SCALAR_LIST, can_use_flat

logger = logging.getLogger(__name__)

# Triton 3.6 的 tl tensor 没有 __pow__；复用公共实现同款 GCU libdevice powf。
_pow = tl_extra_shim.pow

# --- TUNING PARAMS START ---
# 编译期/启动期配置，按优化轮次调整，不改变算子语义。
FIXED_GRID = 48
BLOCK_SIZE = 8192
NUM_WARPS = 4
NUM_STAGES = 2
USE_DEVICE_CONTEXT = 0
FALLBACK_TO_PUBLIC = 0
USE_META_CACHE = 1
META_CACHE_LIMIT = 32
USE_SCALAR_CONSTEXPR = 0
GRID_MODE = 0
PACK_NUMEL_SCALAR = 0
USE_SQRT_HALF_EXPONENT = 1
# --- TUNING PARAMS END ---


@libentry()
@triton.jit
def fused_pow_scalarlist_kernel(
    meta_ptr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    USE_CONST_SCALAR: tl.constexpr,
    SCALAR_VALUE: tl.constexpr,
    PACKED_META: tl.constexpr,
    USE_SQRT_HALF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """一次 launch 对整个列表做 out-of-place ``self ** scalars[i]``。

    ``meta_ptr`` 依次存放输入地址、输出地址（各 NT 项）、元素数与
    float32 指数位模式。``grid.x`` 按 ``(tensor, part)`` 线性展开。
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tensor_id = pid % NT
    part_id = pid // NT
    num_parts = tl.cdiv(num_programs, NT)

    a_raw = tl.load(meta_ptr + tensor_id)
    out_raw = tl.load(meta_ptr + NT + tensor_id)
    numel_scalar_raw = tl.load(meta_ptr + 2 * NT + tensor_id)

    a_ptr = a_raw.to(tl.pointer_type(A_DT))
    out_ptr = out_raw.to(tl.pointer_type(A_DT))

    if PACKED_META:
        # 低 32 位是 float32 指数位模式，高位是元素数。
        num_elems = numel_scalar_raw >> 32
        scalar_raw = numel_scalar_raw & 0xFFFFFFFF
    else:
        num_elems = numel_scalar_raw
        scalar_raw = tl.load(meta_ptr + 3 * NT + tensor_id)
    if USE_CONST_SCALAR:
        exponent = SCALAR_VALUE
    else:
        exponent = tl.cast(scalar_raw.to(tl.int32), tl.float32, bitcast=True)

    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(part_id, tl.cdiv(num_elems, BLOCK), num_parts):
        offs = block_id * BLOCK + arange
        mask = offs < num_elems
        a = tl.load(a_ptr + offs, mask=mask)
        # powf 只接受 fp32 操作数；store 时隐式降回输出 tensor 的 dtype。
        base_fp32 = a.to(tl.float32)
        if USE_SQRT_HALF:
            # pow(x, 0.5) 与 sqrt(x) 在 IEEE 语义下逐点一致（负底数同为 NaN），
            # 用单条 sqrt 指令替代整个 libdevice powf，避免 log2/exp2 与边界修正。
            result = tl.sqrt(base_fp32)
        else:
            result = _pow(base_fp32, exponent)
        tl.store(out_ptr + offs, result, mask=mask)


# 只读 metadata 缓存：key 由设备、输入/输出地址、元素数与指数位模式组成，
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


def _foreach_pow_ScalarList(self, scalars):
    """out-of-place ``[self[i] ** scalars[i] for i in range(len(self))]``。"""
    logger.debug("GEMS_GCU400 _FOREACH_POW_SCALARLIST")

    def fallback():
        return _apply_scalar("pow", self, scalars, inplace=False)

    if FALLBACK_TO_PUBLIC or not can_use_flat(self, scalars, MODE_SCALAR_LIST):
        return fallback()

    num_tensors = len(self)
    outputs = [torch.empty_like(tensor) for tensor in self]
    exponent_bits = [
        struct.unpack("<i", struct.pack("<f", float(item)))[0] for item in scalars
    ]
    if PACK_NUMEL_SCALAR:
        # 把元素数与 float32 指数位模式压进同一个 int64，metadata 由 4 段缩到 3 段。
        packed = [
            (tensor.numel() << 32) | (bits & 0xFFFFFFFF)
            for tensor, bits in zip(self, exponent_bits)
        ]
        meta_values = (
            [tensor.data_ptr() for tensor in self]
            + [tensor.data_ptr() for tensor in outputs]
            + packed
        )
    else:
        meta_values = (
            [tensor.data_ptr() for tensor in self]
            + [tensor.data_ptr() for tensor in outputs]
            + [tensor.numel() for tensor in self]
            + exponent_bits
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

    # 指数列表取值全都相同时可用编译期常量，省去一次 metadata 读取与位转换。
    use_const_scalar = bool(USE_SCALAR_CONSTEXPR and len(set(scalars)) == 1)
    scalar_value = float(scalars[0]) if use_const_scalar else 0.0
    # 仅当整个标量列表都恰为 0.5 时才可用 sqrt 替换，其它指数仍走通用 powf。
    use_sqrt_half = bool(
        USE_SQRT_HALF_EXPONENT and len(set(scalars)) == 1 and float(scalars[0]) == 0.5
    )

    device_context = (
        torch_device_fn.device(device) if USE_DEVICE_CONTEXT else nullcontext()
    )
    with device_context:
        fused_pow_scalarlist_kernel[(grid_size,)](
            meta,
            num_tensors,
            _TL_DTYPES[self[0].dtype],
            use_const_scalar,
            scalar_value,
            PACK_NUMEL_SCALAR,
            use_sqrt_half,
            BLOCK_SIZE,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
    return outputs
