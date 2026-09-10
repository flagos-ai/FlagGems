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
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24


@triton.jit
def round_half_to_even_impl(x):
    r = tl.floor(x)
    d = x - r
    is_odd = tl.abs(r - 2.0 * tl.floor(r / 2.0)) > 0.5
    return tl.where((d > 0.5) | ((tl.abs(d - 0.5) < 1e-10) & is_odd), r + 1.0, r)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def round_kernel(
    x_ptr,
    out_ptr,
    N_total,
    decimals: tl.constexpr,
    IS_FP32: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(pid, (N_total + BLOCK - 1) // BLOCK, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask)

        if decimals == 0:
            out = x
            if IS_FP32:
                out = round_half_to_even_impl(x)
            elif IS_FP16:
                out = tl.cast(
                    round_half_to_even_impl(tl.cast(x, tl.float32)), tl.float16
                )
            elif IS_BF16:
                out = tl.cast(
                    round_half_to_even_impl(tl.cast(x, tl.float32)), tl.bfloat16
                )
        else:
            scale = 10.0**decimals
            if IS_FP32:
                out = round_half_to_even_impl(x * scale) / scale
            elif IS_FP16:
                x_fp32 = tl.cast(x, tl.float32)
                out = tl.cast(
                    round_half_to_even_impl(x_fp32 * scale) / scale, tl.float16
                )
            elif IS_BF16:
                x_fp32 = tl.cast(x, tl.float32)
                out = tl.cast(
                    round_half_to_even_impl(x_fp32 * scale) / scale, tl.bfloat16
                )
            else:
                out = x

        tl.store(out_ptr + off, out, mask=mask)


def _launch_round(input, output, decimals):
    N_total = input.numel()
    if N_total == 0:
        return

    dtype = input.dtype
    IS_FP32 = dtype == torch.float32
    IS_FP16 = dtype == torch.float16
    IS_BF16 = dtype == torch.bfloat16
    BLOCK = 8192
    grid_size = min((N_total + BLOCK - 1) // BLOCK, NUM_SIPS * 2)

    with torch_device_fn.device(input.device):
        round_kernel[(grid_size,)](
            input,
            output,
            N_total,
            decimals,
            IS_FP32=IS_FP32,
            IS_FP16=IS_FP16,
            IS_BF16=IS_BF16,
            BLOCK=BLOCK,
            num_warps=4,
        )


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

    output = torch.empty_like(input)
    _launch_round(input, output, decimals)
    return output


def round(input, decimals=0):
    logger.debug("GEMS_ENFLAME ROUND")
    return round_func(input, decimals=decimals)


def round_out(input, *, decimals=0, out=None):
    logger.debug("GEMS_ENFLAME ROUND_OUT")
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
    _launch_round(input, out, decimals)
    return out


def round_(input, *, decimals=0):
    logger.debug("GEMS_ENFLAME ROUND_")
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
    _launch_round(input, input, decimals)
    return input
