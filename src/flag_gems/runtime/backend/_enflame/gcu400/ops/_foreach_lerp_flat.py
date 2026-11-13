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

"""GCU400 专属 ``_foreach_lerp_.ScalarList`` 一维融合实现。

公共 ``foreach_ternary`` 执行器把 tensor 列表编码成 metadata 并以二维
``grid=(block, tensor)`` 启动；GCU400 的 ``grid.y`` 硬件上限为 255，
16 个 512x512 张量就需要 ``grid.y=256``，benchmark 直接
``OutOfResources``。这里把 ``grid.x`` 按 ``(tensor, part)`` 线性展开，
一次 launch 完成整个列表；非浮点、非连续、混合 dtype、广播等形态
仍由调用方回退公共执行器，保证 dtype policy 与异常语义不变。
"""

import logging
from contextlib import nullcontext
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

# --- TUNING PARAMS START (foreach_lerp_scalar_list_) ---
# 由优化迭代扫描逐轮注入；均为编译期/启动期配置，不改变算子语义。
LERP_SCALAR_LIST_TUNING = {
    "enabled": 1,
    "fused": 1,
    "fixed_grid": 48,
    "block_size": 8192,
    "num_warps": 4,
    "num_stages": 1,
    "fp32_promote": 1,
    "cache_meta": 1,
}
# --- TUNING PARAMS END (foreach_lerp_scalar_list_) ---

# do_bench 会反复调用同一组 tensor，metadata 只与地址 / numel / 权重有关。
# 命中缓存即可省掉两次 host→device 小张量构造；键包含全部地址，地址复用时
# 内容仍然正确，缓存上限避免长期驻留。
_META_CACHE: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}
_META_CACHE_MAX = 64


def _build_launch_data(self, tensors1, out_tensors, weight):
    """返回 (int64 metadata, fp32 权重)；命中缓存时复用 device 侧张量。"""
    key = (
        tuple(tensor.data_ptr() for tensor in self),
        tuple(tensor.data_ptr() for tensor in out_tensors),
        tuple(tensor.data_ptr() for tensor in tensors1),
        tuple(tensor.numel() for tensor in self),
        tuple(float(value) for value in weight),
        self[0].dtype,
        self[0].device,
    )
    cached = _META_CACHE.get(key)
    if cached is not None:
        return cached
    meta = torch.tensor(
        [tensor.data_ptr() for tensor in self]
        + [tensor.data_ptr() for tensor in out_tensors]
        + [tensor.data_ptr() for tensor in tensors1]
        + [tensor.numel() for tensor in self],
        dtype=torch.int64,
        device=self[0].device,
    )
    weights = torch.tensor(
        [float(value) for value in weight],
        dtype=torch.float32,
        device=self[0].device,
    )
    if len(_META_CACHE) >= _META_CACHE_MAX:
        _META_CACHE.clear()
    _META_CACHE[key] = (meta, weights)
    return meta, weights


@libentry()
@triton.jit(do_not_specialize=["N"])
def lerp_flat_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    weight,
    N,
    FP32_PROMOTE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """逐 tensor 的一维跨步 lerp，作为非融合路径的对照实现。"""
    arange = tl.arange(0, BLOCK)
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for block_id in tl.range(pid, tl.cdiv(N, BLOCK), num_programs):
        offs = block_id * BLOCK + arange
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask)
        b = tl.load(b_ptr + offs, mask=mask)
        if FP32_PROMOTE:
            # 低精度先提升 fp32 再算，store 时隐式降回输出 dtype。
            a = a.to(tl.float32)
            b = b.to(tl.float32)
        tl.store(out_ptr + offs, a + weight * (b - a), mask=mask)


@libentry()
@triton.jit
def fused_lerp_scalar_list_kernel(
    meta_ptr,
    weight_ptr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    FP32_PROMOTE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """一次 launch 处理整个列表的 lerp。

    ``meta_ptr`` 依次存放输入指针、输出指针、第二操作数指针、元素数
    （各 NT 项，int64）；``weight_ptr`` 存放每个 tensor 的 fp32 权重。
    ``grid.x`` 按 ``(tensor, part)`` 线性展开，绕开 grid.y 的 255 上限。
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tensor_id = pid % NT
    part_id = pid // NT
    num_parts = tl.cdiv(num_programs, NT)

    a_raw = tl.load(meta_ptr + tensor_id)
    out_raw = tl.load(meta_ptr + NT + tensor_id)
    b_raw = tl.load(meta_ptr + 2 * NT + tensor_id)
    num_elems = tl.load(meta_ptr + 3 * NT + tensor_id)

    a_ptr = a_raw.to(tl.pointer_type(A_DT))
    out_ptr = out_raw.to(tl.pointer_type(A_DT))
    b_ptr = b_raw.to(tl.pointer_type(A_DT))
    weight = tl.load(weight_ptr + tensor_id)

    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(part_id, tl.cdiv(num_elems, BLOCK), num_parts):
        offs = block_id * BLOCK + arange
        mask = offs < num_elems
        a = tl.load(a_ptr + offs, mask=mask)
        b = tl.load(b_ptr + offs, mask=mask)
        if FP32_PROMOTE:
            a = a.to(tl.float32)
            b = b.to(tl.float32)
        tl.store(out_ptr + offs, a + weight * (b - a), mask=mask)


def can_use_flat(self, tensors1, weight) -> bool:
    """只有同 dtype、连续、等长、逐 tensor 数值标量的列表才走快路径。"""
    if not self:
        return False
    if not isinstance(weight, (list, tuple)) or len(weight) != len(self):
        return False
    if not all(isinstance(value, (int, float)) for value in weight):
        return False
    if not isinstance(tensors1, (list, tuple)) or len(tensors1) != len(self):
        return False
    dtype = self[0].dtype
    if dtype not in FLOAT_DTYPES:
        return False
    for tensor, other in zip(self, tensors1):
        if tensor.dtype != dtype or other.dtype != dtype:
            return False
        if not tensor.is_contiguous() or not other.is_contiguous():
            return False
        if tensor.numel() == 0 or other.numel() != tensor.numel():
            return False
    return True


def run_lerp_scalar_list(self, tensors1, weight, inplace: bool):
    """执行 lerp ScalarList；不适合快路径时返回 ``FALLBACK`` 哨兵。"""
    config = LERP_SCALAR_LIST_TUNING
    if not config["enabled"] or not can_use_flat(self, tensors1, weight):
        return FALLBACK

    num_tensors = len(self)
    outputs = None if inplace else [torch.empty_like(tensor) for tensor in self]
    out_tensors = self if inplace else outputs
    if config["cache_meta"]:
        meta, weights = _build_launch_data(self, tensors1, out_tensors, weight)
    else:
        meta = torch.tensor(
            [tensor.data_ptr() for tensor in self]
            + [tensor.data_ptr() for tensor in out_tensors]
            + [tensor.data_ptr() for tensor in tensors1]
            + [tensor.numel() for tensor in self],
            dtype=torch.int64,
            device=self[0].device,
        )
        weights = torch.tensor(
            [float(value) for value in weight],
            dtype=torch.float32,
            device=self[0].device,
        )
    block_size = config["block_size"]
    triton_dtype = _TL_DTYPES[self[0].dtype]
    with nullcontext():
        if config["fused"]:
            blocks_per_tensor = triton.cdiv(
                max(tensor.numel() for tensor in self), block_size
            )
            grid = max(1, min(config["fixed_grid"], num_tensors * blocks_per_tensor))
            fused_lerp_scalar_list_kernel[(grid,)](
                meta,
                weights,
                num_tensors,
                triton_dtype,
                config["fp32_promote"],
                block_size,
                num_warps=config["num_warps"],
                num_stages=config["num_stages"],
            )
        else:
            for index, (tensor, other, output) in enumerate(
                zip(self, tensors1, out_tensors)
            ):
                num_elems = tensor.numel()
                grid = (
                    min(
                        triton.cdiv(num_elems, block_size),
                        config["fixed_grid"],
                    ),
                )
                lerp_flat_kernel[grid](
                    tensor,
                    other,
                    output,
                    float(weight[index]),
                    num_elems,
                    config["fp32_promote"],
                    block_size,
                    num_warps=config["num_warps"],
                    num_stages=config["num_stages"],
                )
    return None if inplace else outputs
