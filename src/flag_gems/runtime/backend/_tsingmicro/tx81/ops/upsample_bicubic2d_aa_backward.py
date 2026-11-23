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
import math
from collections import OrderedDict

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_WEIGHT_CACHE_MAX_ENTRIES = 128
_weight_matrix_cache = OrderedDict()


def _compute_scale(input_size, output_size, align_corners, scale=None):
    if align_corners:
        return float(input_size - 1) / (output_size - 1) if output_size > 1 else 0.0
    return (
        (1.0 / scale)
        if scale is not None and scale > 0
        else float(input_size) / output_size
    )


def _cubic_aa_filter_scalar(x):
    x = abs(float(x))
    if x < 1.0:
        return (1.5 * x - 2.5) * x * x + 1.0
    if x < 2.0:
        return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    return 0.0


def _get_weight_matrix(
    input_size,
    output_size,
    align_corners,
    scale_arg,
    dtype,
    device,
):
    key = (
        int(input_size),
        int(output_size),
        bool(align_corners),
        None if scale_arg is None else float(scale_arg),
        dtype,
        str(device),
    )
    cached = _weight_matrix_cache.get(key)
    if cached is not None:
        _weight_matrix_cache.move_to_end(key)
        return cached

    scale = _compute_scale(input_size, output_size, align_corners, scale_arg)
    support = 2.0 * scale if scale >= 1.0 else 2.0
    invscale = 1.0 / scale if scale >= 1.0 else 1.0
    weight_cpu = torch.zeros((output_size, input_size), dtype=torch.float32)
    for output_idx in range(output_size):
        center = scale * (output_idx + 0.5)
        start = max(math.floor(center - support + 0.5), 0)
        end = min(math.floor(center + support + 0.5), input_size)
        values = [
            _cubic_aa_filter_scalar((input_idx - center + 0.5) * invscale)
            for input_idx in range(start, end)
        ]
        total = sum(values)
        if total != 0.0:
            for input_idx, value in zip(range(start, end), values):
                weight_cpu[output_idx, input_idx] = value / total

    weight = weight_cpu.to(device=device, dtype=dtype)
    if len(_weight_matrix_cache) >= _WEIGHT_CACHE_MAX_ENTRIES:
        _weight_matrix_cache.popitem(last=False)
    _weight_matrix_cache[key] = weight
    return weight


@triton.jit
def _width_backward_kernel(
    grad_ptr,
    temp_ptr,
    weight_ptr,
    total_rows,
    W_in,
    W_out: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    iws = tl.program_id(0) * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = iws < W_in
    row_start = tl.program_id(1)
    row_step = tl.num_programs(1)
    for row in range(row_start, total_rows, row_step):
        acc = tl.zeros((BLOCK_W,), tl.float32)
        for ow in tl.static_range(0, W_out):
            grad = tl.load(grad_ptr + row * W_out + ow).to(tl.float32)
            weight = tl.load(weight_ptr + ow * W_in + iws, mask=mask, other=0.0)
            acc += grad * weight
        tl.store(temp_ptr + row * W_in + iws, acc, mask=mask)


@triton.jit
def _height_backward_kernel(
    temp_ptr,
    output_ptr,
    weight_ptr,
    total_rows,
    H_in,
    W_in,
    H_out: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    iws = tl.program_id(0) * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = iws < W_in
    row_start = tl.program_id(1)
    row_step = tl.num_programs(1)
    for row in range(row_start, total_rows, row_step):
        ih = row % H_in
        nc = row // H_in
        acc = tl.zeros((BLOCK_W,), tl.float32)
        for oh in tl.static_range(0, H_out):
            weight = tl.load(weight_ptr + oh * H_in + ih)
            value = tl.load(
                temp_ptr + (nc * H_out + oh) * W_in + iws,
                mask=mask,
                other=0.0,
            )
            acc += value * weight
        tl.store(output_ptr + row * W_in + iws, acc, mask=mask)


def _upsample_bicubic2d_aa_backward(
    grad_output: torch.Tensor,
    output_size,
    input_size,
    align_corners: bool,
    scales_h=None,
    scales_w=None,
) -> torch.Tensor:
    logger.debug("GEMS_TSINGMICRO _UPSAMPLE_BICUBIC2D_AA_BACKWARD")

    if len(input_size) != 4:
        raise ValueError("input_size must contain N, C, H and W")
    if len(output_size) != 2:
        raise ValueError("output_size must contain H and W")

    N, C, H_in, W_in = (int(value) for value in input_size)
    H_out, W_out = (int(value) for value in output_size)
    expected_shape = (N, C, H_out, W_out)
    if tuple(grad_output.shape) != expected_shape:
        raise ValueError(
            f"grad_output shape {tuple(grad_output.shape)} != {expected_shape}"
        )

    if N * C == 0 or min(H_in, W_in, H_out, W_out) == 0:
        return grad_output.new_zeros((N, C, H_in, W_in))

    weight_w = _get_weight_matrix(
        W_in,
        W_out,
        align_corners,
        scales_w,
        torch.float32,
        grad_output.device,
    )
    weight_h = _get_weight_matrix(
        H_in,
        H_out,
        align_corners,
        scales_h,
        torch.float32,
        grad_output.device,
    )

    nc = N * C
    block_w = min(128, triton.next_power_of_2(max(W_in, 1)))
    num_warps = 2 if block_w <= 64 else 4
    grid_x = triton.cdiv(W_in, block_w)

    grad = grad_output.contiguous()
    temp = torch.empty(
        (nc, H_out, W_in), dtype=torch.float32, device=grad_output.device
    )
    width_rows = nc * H_out
    _width_backward_kernel[(grid_x, min(width_rows, 4096))](
        grad,
        temp,
        weight_w,
        width_rows,
        W_in,
        W_out=W_out,
        BLOCK_W=block_w,
        num_warps=num_warps,
    )

    output = torch.empty(
        (nc, H_in, W_in), dtype=grad_output.dtype, device=grad_output.device
    )
    height_rows = nc * H_in
    _height_backward_kernel[(grid_x, min(height_rows, 4096))](
        temp,
        output,
        weight_h,
        height_rows,
        H_in,
        W_in,
        H_out=H_out,
        BLOCK_W=block_w,
        num_warps=num_warps,
    )
    return output.reshape(N, C, H_in, W_in)
