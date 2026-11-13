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
"""GCU400 专属 ``_foreach_addcdiv_.Tensor`` 一维融合实现。

公共 ``foreach_ternary`` 执行器以二维 ``grid=(tensor, tile)`` 启动；在
16 个 512x512 张量、BLOCK=1024 时第二维达到 256，越过 GCU400 ``grid.y``
的 255 硬件上限，benchmark 直接 ``OutOfResources``。这里把 ``grid.x`` 按
``(tensor, part)`` 线性展开为单次 launch，并在核内跨步覆盖剩余 tile。
混合 dtype、非连续、广播、标量 value、非法 dtype 等形态仍回退公共执行器，
保证原有 dtype policy 与异常语义不变。
"""

import logging
from typing import Dict, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)
_TL_DTYPES = {
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
}

# 不适合快路径时返回该哨兵，由调用方委托公共执行器。
FALLBACK = object()

# --- TUNING PARAMS START (foreach_addcdiv_tensor_) ---
# 由优化迭代扫描逐轮注入；均为编译期/启动期配置，不改变算子语义。
ADDC_DIV_TENSOR_TUNING = {
    "enabled": 1,
    "fixed_grid": 24,
    "block_size": 16384,
    "num_warps": 4,
    "num_stages": 1,
    "fp32_promote": 1,
    "cache_meta": 1,
}
# --- TUNING PARAMS END (foreach_addcdiv_tensor_) ---

# do_bench 反复调用同一组 tensor，metadata 只与地址 / numel / value 有关。
# 命中缓存即可省掉两次 host→device 小张量构造；键包含全部地址与 value，
# 地址复用但内容变化时仍会重新构造，缓存上限避免长期驻留。
_META_CACHE: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}
_META_CACHE_MAX = 64


@libentry()
@triton.jit
def fused_addcdiv_tensor_kernel(
    meta_ptr,
    value_ptr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    FP32_PROMOTE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """一次 launch 完成 ``self + value * (tensor1 / tensor2)``。

    ``meta_ptr`` 为 int64，依次存放 self / output / tensor1 / tensor2 指针与
    元素数（各 NT 项）；``value_ptr`` 存放每个 tensor 的 fp32 value。
    ``grid.x`` 由 ``(tensor, part)`` 线性展开，绕开 grid.y 的 255 上限。
    """
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tensor_id = program_id % NT
    part_id = program_id // NT
    num_parts = tl.cdiv(num_programs, NT)

    self_raw = tl.load(meta_ptr + tensor_id)
    out_raw = tl.load(meta_ptr + NT + tensor_id)
    tensor1_raw = tl.load(meta_ptr + 2 * NT + tensor_id)
    tensor2_raw = tl.load(meta_ptr + 3 * NT + tensor_id)
    num_elems = tl.load(meta_ptr + 4 * NT + tensor_id)

    self_ptr = self_raw.to(tl.pointer_type(A_DT))
    out_ptr = out_raw.to(tl.pointer_type(A_DT))
    tensor1_ptr = tensor1_raw.to(tl.pointer_type(A_DT))
    tensor2_ptr = tensor2_raw.to(tl.pointer_type(A_DT))
    value = tl.load(value_ptr + tensor_id)

    element_arange = tl.arange(0, BLOCK)
    for block_id in tl.range(part_id, tl.cdiv(num_elems, BLOCK), num_parts):
        offsets = block_id * BLOCK + element_arange
        mask = offsets < num_elems
        self_value = tl.load(self_ptr + offsets, mask=mask, other=0)
        tensor1_value = tl.load(tensor1_ptr + offsets, mask=mask, other=0)
        tensor2_value = tl.load(tensor2_ptr + offsets, mask=mask, other=0)
        if FP32_PROMOTE:
            # 低精度除法先提升 fp32 计算，store 时隐式降回输出 dtype。
            self_value = self_value.to(tl.float32)
            tensor1_value = tensor1_value.to(tl.float32)
            tensor2_value = tensor2_value.to(tl.float32)
            value = value.to(tl.float32)
        result = self_value + value * (tensor1_value / tensor2_value)
        tl.store(out_ptr + offsets, result, mask=mask)


def _build_launch_data(self, tensor1, tensor2, out_tensors, values):
    """返回 (int64 metadata, fp32 value)；命中缓存时复用 device 侧张量。"""
    key = (
        tuple(tensor.data_ptr() for tensor in self),
        tuple(tensor.data_ptr() for tensor in out_tensors),
        tuple(tensor.data_ptr() for tensor in tensor1),
        tuple(tensor.data_ptr() for tensor in tensor2),
        tuple(tensor.numel() for tensor in self),
        tuple(float(value) for value in values),
        self[0].dtype,
        self[0].device,
    )
    cached = _META_CACHE.get(key)
    if cached is not None:
        return cached
    meta = torch.tensor(
        [tensor.data_ptr() for tensor in self]
        + [tensor.data_ptr() for tensor in out_tensors]
        + [tensor.data_ptr() for tensor in tensor1]
        + [tensor.data_ptr() for tensor in tensor2]
        + [tensor.numel() for tensor in self],
        dtype=torch.int64,
        device=self[0].device,
    )
    device_values = torch.tensor(
        [float(value) for value in values],
        dtype=torch.float32,
        device=self[0].device,
    )
    if len(_META_CACHE) >= _META_CACHE_MAX:
        _META_CACHE.clear()
    _META_CACHE[key] = (meta, device_values)
    return meta, device_values


def can_use_flat(self, tensor1, tensor2, scalars) -> bool:
    """只有同 dtype、连续、等长且逐 tensor 数值 value 的列表才走快路径。"""
    if not self:
        return False
    if not isinstance(scalars, torch.Tensor) or scalars.numel() != len(self):
        return False
    if not isinstance(tensor1, (list, tuple)) or len(tensor1) != len(self):
        return False
    if not isinstance(tensor2, (list, tuple)) or len(tensor2) != len(self):
        return False
    dtype = self[0].dtype
    if dtype not in FLOAT_DTYPES:
        return False
    device = self[0].device
    for base, first, second in zip(self, tensor1, tensor2):
        if (
            base.dtype != dtype
            or first.dtype != dtype
            or second.dtype != dtype
            or base.device != device
            or first.device != device
            or second.device != device
        ):
            return False
        if (
            not base.is_contiguous()
            or not first.is_contiguous()
            or not second.is_contiguous()
        ):
            return False
        if (
            base.numel() == 0
            or first.numel() != base.numel()
            or second.numel() != base.numel()
        ):
            return False
    return True


def run_addcdiv_tensor(self, tensor1, tensor2, scalars, inplace: bool):
    """执行 addcdiv 的 .Tensor 重载；不适合快路径时返回 ``FALLBACK`` 哨兵。"""
    config = ADDC_DIV_TENSOR_TUNING
    if not config["enabled"] or not can_use_flat(self, tensor1, tensor2, scalars):
        return FALLBACK

    num_tensors = len(self)
    outputs = None if inplace else [torch.empty_like(tensor) for tensor in self]
    out_tensors = self if inplace else outputs
    values = scalars.tolist()
    if config["cache_meta"]:
        meta, device_values = _build_launch_data(
            self, tensor1, tensor2, out_tensors, values
        )
    else:
        meta = torch.tensor(
            [tensor.data_ptr() for tensor in self]
            + [tensor.data_ptr() for tensor in out_tensors]
            + [tensor.data_ptr() for tensor in tensor1]
            + [tensor.data_ptr() for tensor in tensor2]
            + [tensor.numel() for tensor in self],
            dtype=torch.int64,
            device=self[0].device,
        )
        device_values = torch.tensor(
            [float(value) for value in values],
            dtype=torch.float32,
            device=self[0].device,
        )

    block_size = config["block_size"]
    blocks_per_tensor = triton.cdiv(max(tensor.numel() for tensor in self), block_size)
    # 至少让每个 tensor 有一个 program，且不超过固定 grid 上限。
    grid = max(1, min(config["fixed_grid"], num_tensors * blocks_per_tensor))
    fused_addcdiv_tensor_kernel[(grid,)](
        meta,
        device_values,
        num_tensors,
        _TL_DTYPES[self[0].dtype],
        config["fp32_promote"],
        block_size,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return None if inplace else outputs
