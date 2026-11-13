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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)

NUM_SIPS = 24
FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)

# 大 shape 由访存与 launch 并行度主导：4096 元素/block、4 warps 为 13 轮扫描最优。
FLAT_BLOCK_SIZE = 4096
FLAT_NUM_WARPS = 4
FLAT_GRID_MULTIPLIER = 2


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def trunc_func(x):
    """非连续张量与整数 dtype 的通用回退路径，保持原有 extern 语义。"""
    x_fp32 = x.to(tl.float32)
    t = tl_extra_shim.trunc(x_fp32)
    # GCU400 的取整 extern 会把 -0.5 / -0.0 变成 +0.0，用输入符号位重建零结果。
    xi = x_fp32.to(tl.int32, bitcast=True)
    ti = t.to(tl.int32, bitcast=True)
    ti = tl.where(t == 0.0, xi & -2147483648, ti)
    t = ti.to(tl.float32, bitcast=True)
    return t.to(x.dtype)


@libentry()
@triton.jit(do_not_specialize=["num_elements"])
def trunc_flat_kernel(input_ptr, output_ptr, num_elements, BLOCK: tl.constexpr):
    """连续浮点张量快路径：单次 launch、grid-stride 完成向零取整。"""
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    element_offsets = tl.arange(0, BLOCK)
    num_blocks = tl.cdiv(num_elements, BLOCK)

    for block_index in tl.range(program_id, num_blocks, num_programs):
        linear_offsets = block_index * BLOCK + element_offsets
        valid_mask = linear_offsets < num_elements
        input_values = tl.load(input_ptr + linear_offsets, mask=valid_mask)
        input_fp32 = input_values.to(tl.float32)

        # IEEE 向零取整；GCU400 的 floor/ceil 不保留负零，需重建零结果符号位。
        truncated_fp32 = tl.where(
            input_fp32 >= 0.0, tl.floor(input_fp32), tl.ceil(input_fp32)
        )
        input_bits = input_fp32.to(tl.int32, bitcast=True)
        truncated_bits = truncated_fp32.to(tl.int32, bitcast=True)
        truncated_bits = tl.where(
            truncated_fp32 == 0.0, input_bits & -2147483648, truncated_bits
        )
        result_fp32 = truncated_bits.to(tl.float32, bitcast=True)

        tl.store(
            output_ptr + linear_offsets,
            result_fp32.to(input_values.dtype),
            mask=valid_mask,
        )


def _launch_flat(inp, out):
    num_elements = inp.numel()
    grid_size = min(
        triton.cdiv(num_elements, FLAT_BLOCK_SIZE),
        NUM_SIPS * FLAT_GRID_MULTIPLIER,
    )
    with torch_device_fn.device(inp.device):
        trunc_flat_kernel[(grid_size,)](
            inp,
            out,
            num_elements,
            BLOCK=FLAT_BLOCK_SIZE,
            num_warps=FLAT_NUM_WARPS,
        )


def _use_flat_path(tensor):
    return tensor.dtype in FLOAT_DTYPES and tensor.is_contiguous()


def trunc(A):
    logger.debug("GEMS_ENFLAME TRUNC")
    if _use_flat_path(A):
        out = torch.empty_like(A)
        _launch_flat(A, out)
        return out
    return trunc_func(A)


def trunc_(A):
    logger.debug("GEMS_ENFLAME TRUNC_")
    if _use_flat_path(A):
        _launch_flat(A, A)
        return A
    trunc_func(A, out0=A)
    return A
