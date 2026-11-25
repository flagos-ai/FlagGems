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

import math
from typing import Sequence

import torch
import triton
import triton.language as tl

from flag_gems.ops.rms_norm_w8a16_fp8 import (
    rms_norm_w8a16_fp8 as rms_norm_w8a16_fp8_generic,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_fp8_rows_kernel(
    output,
    activation,
    weight,
    scale,
    M,
    N: tl.constexpr,
    eps,
    WEIGHT_STRIDE: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, COLS)
    offsets = rows[:, None] * N + cols[None, :]
    valid = (rows[:, None] < M) & (cols[None, :] < N)
    values = tl.load(activation + offsets, valid, other=0.0).to(tl.float32)
    inverse_rms = tl.rsqrt(tl.sum(values * values, axis=1) / N + eps)
    # Broadcast the live dequantized weight across the rows in this program.
    if WEIGHT_STRIDE == 1:
        weights = tl.load(weight + cols, cols < N, other=0.0).to(tl.float32)
    else:
        # Decode bytes to avoid compiler-dependent FP8 conversion on strided loads.
        bits = tl.load(weight + cols * WEIGHT_STRIDE, cols < N, other=0).to(tl.uint32)
        exponent = (bits >> 3) & 15
        mantissa = bits & 7
        normal = ((exponent + 120) << 23) | (mantissa << 20)
        subnormal = (mantissa.to(tl.float32) * 2.0**-9).to(tl.uint32, bitcast=True)
        magnitude = tl.where(exponent == 0, subnormal, normal)
        magnitude = tl.where((bits & 127) == 127, 0x7FC00000, magnitude)
        weights = (magnitude | ((bits & 128) << 24)).to(tl.float32, bitcast=True)
    scales = tl.load(
        scale + (cols // GROUP_SIZE) * SCALE_STRIDE, cols < N, other=0.0
    ).to(tl.float32)
    normalized = values * inverse_rms[:, None]
    tl.store(output + offsets, normalized * (weights * scales)[None, :], valid)


def rms_norm_w8a16_fp8(
    x: torch.Tensor,
    normalized_shape: Sequence[int],
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    eps: float = 1e-5,
    group_size: int = 128,
) -> torch.Tensor:
    width = math.prod(normalized_shape)
    if width > 2048 or not x.is_contiguous():
        return rms_norm_w8a16_fp8_generic(
            x, normalized_shape, weight_fp8, weight_scale, eps, group_size
        )

    row_count = x.numel() // width
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if row_count == 0:
        return output
    if weight_fp8.stride(-1) != 1:
        weight_fp8 = weight_fp8.view(torch.uint8)
    if row_count < 128 and width > 256:
        rows, warps = 1, 4
    elif row_count < 128:
        rows, warps = 2, 2
    elif width <= 256:
        rows, warps = 8, 2
    else:
        rows, warps = 8, 4
    with torch_device_fn.device(x.device):
        rms_norm_fp8_rows_kernel[(triton.cdiv(row_count, rows),)](
            output,
            x,
            weight_fp8,
            weight_scale,
            row_count,
            width,
            eps,
            weight_fp8.stride(-1),
            weight_scale.stride(-1),
            group_size,
            rows,
            triton.next_power_of_2(width),
            num_warps=warps,
            enable_fp_fusion=False,
        )
    return output
