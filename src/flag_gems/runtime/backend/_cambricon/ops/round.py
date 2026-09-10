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

from ..utils import TOTAL_CORE_NUM

logger = logging.getLogger(__name__)


@triton.jit
def round_half_to_even_impl(x):
    """Round to nearest with ties to even (round half to even).
    x must be fp32."""
    r = tl.floor(x)
    d = x - r

    is_odd = tl.abs(r - 2.0 * tl.floor(r / 2.0)) > 0.5

    return tl.where((d > 0.5) | ((tl.abs(d - 0.5) < 1e-10) & is_odd), r + 1.0, r)


@triton.jit
def round_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    decimals: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_FP32: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    total_blocks: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(0)

    for block_id in range(pid, total_blocks, num_jobs):
        block_start = block_id * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)

        if decimals == 0:
            out = x
            if IS_FP32:
                out = round_half_to_even_impl(x)
            elif IS_FP16:
                x_fp32 = tl.cast(x, tl.float32)
                out = tl.cast(round_half_to_even_impl(x_fp32), tl.float16)
            elif IS_BF16:
                x_fp32 = tl.cast(x, tl.float32)
                out = tl.cast(round_half_to_even_impl(x_fp32), tl.bfloat16)
        else:
            scale = 10.0**decimals
            if IS_FP32:
                x_scaled = x * scale
                out = round_half_to_even_impl(x_scaled) / scale
            elif IS_FP16:
                x_fp32 = tl.cast(x, tl.float32)
                x_scaled = x_fp32 * scale
                out = tl.cast(round_half_to_even_impl(x_scaled) / scale, tl.float16)
            elif IS_BF16:
                x_fp32 = tl.cast(x, tl.float32)
                x_scaled = x_fp32 * scale
                out = tl.cast(round_half_to_even_impl(x_scaled) / scale, tl.bfloat16)
            else:
                out = x

        tl.store(out_ptr + offsets, out, mask=mask)


def round_func(input, decimals=0):
    if not isinstance(input, torch.Tensor):
        raise TypeError("round expects a torch.Tensor.")

    if input.is_complex():
        raise TypeError("round is not supported for complex tensors.")

    if input.dtype in [torch.int32, torch.int64, torch.int16, torch.int8]:
        return input.clone()

    if not input.is_contiguous():
        raise ValueError(
            "round Triton kernel currently supports only contiguous tensors."
        )

    n_elements = input.numel()
    if n_elements == 0:
        return input

    dtype = input.dtype
    IS_FP32 = dtype == torch.float32
    IS_FP16 = dtype == torch.float16
    IS_BF16 = dtype == torch.bfloat16

    output = torch.empty_like(input)

    BLOCK_SIZE = 1024
    num_warps = 1
    MAX_GRID_SIZE = TOTAL_CORE_NUM // num_warps

    total_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = min(total_blocks, MAX_GRID_SIZE)

    with torch_device_fn.device(input.device):
        round_kernel[(grid,)](
            input,
            output,
            n_elements,
            decimals,
            BLOCK_SIZE=BLOCK_SIZE,
            IS_FP32=IS_FP32,
            IS_FP16=IS_FP16,
            IS_BF16=IS_BF16,
            total_blocks=total_blocks,
            num_warps=num_warps,
        )
    return output


def round(input, decimals=0):
    logger.debug("GEMS_CAMBRICON ROUND")
    return round_func(input, decimals=decimals)


def round_out(input, *, decimals=0, out=None):
    logger.debug("GEMS_CAMBRICON ROUND_OUT")
    if out is None:
        return round_func(input, decimals=decimals)

    if not isinstance(input, torch.Tensor):
        raise TypeError("round expects a torch.Tensor.")

    if input.is_complex():
        raise TypeError("round is not supported for complex tensors.")

    if input.dtype in [torch.int32, torch.int64, torch.int16, torch.int8]:
        out.copy_(input)
        return out

    if not input.is_contiguous():
        raise ValueError(
            "round Triton kernel currently supports only contiguous tensors."
        )

    n_elements = input.numel()
    if n_elements == 0:
        return out

    dtype = input.dtype
    IS_FP32 = dtype == torch.float32
    IS_FP16 = dtype == torch.float16
    IS_BF16 = dtype == torch.bfloat16

    BLOCK_SIZE = 1024
    num_warps = 1
    MAX_GRID_SIZE = TOTAL_CORE_NUM // num_warps

    total_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = min(total_blocks, MAX_GRID_SIZE)

    with torch_device_fn.device(input.device):
        round_kernel[(grid,)](
            input,
            out,
            n_elements,
            decimals,
            BLOCK_SIZE=BLOCK_SIZE,
            IS_FP32=IS_FP32,
            IS_FP16=IS_FP16,
            IS_BF16=IS_BF16,
            total_blocks=total_blocks,
            num_warps=num_warps,
        )
    return out


def round_(input, *, decimals=0):
    logger.debug("GEMS_CAMBRICON ROUND_")
    if not isinstance(input, torch.Tensor):
        raise TypeError("round expects a torch.Tensor.")

    if input.is_complex():
        raise TypeError("round is not supported for complex tensors.")

    if input.dtype in [torch.int32, torch.int64, torch.int16, torch.int8]:
        return input

    if not input.is_contiguous():
        raise ValueError(
            "round Triton kernel currently supports only contiguous tensors."
        )

    n_elements = input.numel()
    if n_elements == 0:
        return input

    dtype = input.dtype
    IS_FP32 = dtype == torch.float32
    IS_FP16 = dtype == torch.float16
    IS_BF16 = dtype == torch.bfloat16

    BLOCK_SIZE = 1024
    num_warps = 1
    MAX_GRID_SIZE = TOTAL_CORE_NUM // num_warps

    total_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = min(total_blocks, MAX_GRID_SIZE)

    with torch_device_fn.device(input.device):
        round_kernel[(grid,)](
            input,
            input,
            n_elements,
            decimals,
            BLOCK_SIZE=BLOCK_SIZE,
            IS_FP32=IS_FP32,
            IS_FP16=IS_FP16,
            IS_BF16=IS_BF16,
            total_blocks=total_blocks,
            num_warps=num_warps,
        )
    return input
