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
from typing import Any, Optional

import torch
import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

erf = tl_extra_shim.erf
exp = tl_extra_shim.exp
pow = tl_extra_shim.pow
tanh = tl_extra_shim.tanh

logger = logging.getLogger(__name__)


@triton.jit
def geglu_kernel(
    input_ptr,
    output_ptr,
    M,
    H,
    stride_in_m,
    stride_in_h,
    stride_out_m,
    stride_out_h,
    ROWS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Row-tiled GEGLU: each program handles ROWS rows, accessing elements within a row in
    BLOCK_H chunks in order.

    On GCU400 both halves (gate/up) of each row are contiguous in memory (the first H elements
    are gate, the last H are up), so row-wise tiling makes every program issue large contiguous
    accesses, about 7x faster than the previous BLOCK_M x BLOCK_H 2-D strided small tiles
    (measured 4096x2048 fp16: 0.91ms -> 0.115ms). BLOCK_H is the power of two of H (capped at
    2048); when H is not a power of two, the h-direction mask covers the tail.
    """
    pid = tl.program_id(0)
    first_row = pid * ROWS
    offs_h = tl.arange(0, BLOCK_H)
    for row_offset in tl.static_range(ROWS):
        row = first_row + row_offset
        if row < M:
            row_base = row * stride_in_m
            for col_start in range(0, H, BLOCK_H):
                cols = col_start + offs_h
                col_mask = cols < H
                x_a = tl.load(
                    input_ptr + row_base + cols * stride_in_h, mask=col_mask, other=0.0
                ).to(tl.float32)
                x_b = tl.load(
                    input_ptr + row_base + (cols + H) * stride_in_h,
                    mask=col_mask,
                    other=0.0,
                ).to(tl.float32)
                # tanh approximation of GELU; use x*x instead of pow(x,2) to avoid the slow libdevice path
                tanh_out = tanh(0.79788456 * x_a * (1.0 + 0.044715 * x_a * x_a))
                out = (0.5 * x_a * (1.0 + tanh_out)) * x_b
                tl.store(
                    output_ptr + row * stride_out_m + cols * stride_out_h,
                    out,
                    mask=col_mask,
                )


@triton.jit
def dgeglu_kernel(
    grad_out_ptr,
    input_ptr,
    grad_in_ptr,
    M,
    H,
    stride_grad_out_m,
    stride_grad_out_h,
    stride_in_m,
    stride_in_h,
    stride_grad_in_m,
    stride_grad_in_h,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

    mask = (offs_m[:, None] < M) & (offs_h[None, :] < H)

    grad_out_ptr = (
        grad_out_ptr
        + offs_m[:, None] * stride_grad_out_m
        + offs_h[None, :] * stride_grad_out_h
    )
    input_a_ptr = (
        input_ptr + offs_m[:, None] * stride_in_m + offs_h[None, :] * stride_in_h
    )
    input_b_ptr = (
        input_ptr + offs_m[:, None] * stride_in_m + (offs_h[None, :] + H) * stride_in_h
    )
    grad_a_ptr = (
        grad_in_ptr
        + offs_m[:, None] * stride_grad_in_m
        + offs_h[None, :] * stride_grad_in_h
    )
    grad_b_ptr = (
        grad_in_ptr
        + offs_m[:, None] * stride_grad_in_m
        + (offs_h[None, :] + H) * stride_grad_in_h
    )

    grad_out = tl.load(grad_out_ptr, mask=mask, other=0.0).to(tl.float32)
    x_a = tl.load(input_a_ptr, mask=mask, other=0.0).to(tl.float32)
    x_b = tl.load(input_b_ptr, mask=mask, other=0.0).to(tl.float32)

    tanh_out = tanh(0.79788456 * x_a * (1 + 0.044715 * pow(x_a, 2)))
    gelu_out = 0.5 * x_a * (1 + tanh_out)

    # dgelu/dx
    sech2 = 1 - pow(tanh_out, 2)
    dgelu = 0.5 * (1 + tanh_out) + 0.5 * x_a * sech2 * 0.79788456 * (
        1 + 3 * 0.044715 * pow(x_a, 2)
    )

    grad_a = grad_out * x_b * dgelu
    grad_b = grad_out * gelu_out

    tl.store(grad_a_ptr, grad_a.to(x_a.dtype), mask=mask)
    tl.store(grad_b_ptr, grad_b.to(x_a.dtype), mask=mask)


def geglu(input_tensor: torch.Tensor, quantizer: Optional[Any] = None) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME GEGLU")
    shape = input_tensor.shape
    H = shape[-1] // 2
    M = input_tensor.numel() // (2 * H)

    input_2d = input_tensor.contiguous().view(M, 2 * H)
    output_2d = torch.empty(M, H, device=input_tensor.device, dtype=input_tensor.dtype)

    # GCU400: fix the target CTA count (128 measured as the best residency point) and infer from the row count the
    # number of rows
    # each program handles; cap ROWS at 32 to avoid excessive static_range unrolling.
    rows_per_program = min(32, max(1, triton.cdiv(M, 128)))
    block_h = min(2048, triton.next_power_of_2(H))

    grid = (triton.cdiv(M, rows_per_program),)
    geglu_kernel[grid](
        input_2d,
        output_2d,
        M,
        H,
        input_2d.stride(0),
        input_2d.stride(1),
        output_2d.stride(0),
        output_2d.stride(1),
        ROWS=rows_per_program,
        BLOCK_H=block_h,
        num_warps=8,
    )
    return output_2d.view(*shape[:-1], H)


def dgeglu(
    grad_output: torch.Tensor,
    input_tensor: torch.Tensor,
    quantizer: Optional[Any] = None,
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME DGEGLU")
    shape = input_tensor.shape
    H = shape[-1] // 2
    M = input_tensor.numel() // (2 * H)

    grad_out_2d = grad_output.contiguous().view(M, H)
    input_2d = input_tensor.contiguous().view(M, 2 * H)
    grad_in_2d = torch.empty_like(input_2d)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(H, META["BLOCK_SIZE_H"]),
    )

    dgeglu_kernel[grid](
        grad_out_2d,
        input_2d,
        grad_in_2d,
        M,
        H,
        grad_out_2d.stride(0),
        grad_out_2d.stride(1),
        input_2d.stride(0),
        input_2d.stride(1),
        grad_in_2d.stride(0),
        grad_in_2d.stride(1),
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_H=64,
    )
    # print(dgeglu)
    return grad_in_2d.view_as(input_tensor)
