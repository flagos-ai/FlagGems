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

logger = logging.getLogger(__name__)


def _select_params(n):
    if n <= 256:
        return 256, 2
    if n <= 1024:
        return 256, 4
    if n <= 4096:
        return 512, 4
    return 1024, 4


def _estimate_output_size(n, minlength):
    estimate = max(8192, n * 4, minlength)
    estimate = min(estimate, 65536)
    return max(estimate, minlength)


@triton.jit
def fused_max_bincount_kernel(
    input_ptr,
    max_ptr,
    output_ptr,
    n_elements,
    output_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(input_ptr + offsets, mask=mask, other=0)

    local_max = tl.max(vals, axis=0)
    tl.atomic_max(max_ptr, local_max)

    safe_mask = mask & (vals < output_size)
    tl.atomic_add(output_ptr + vals, 1, mask=safe_mask)


@triton.jit
def bincount_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(input_ptr + offsets, mask=mask, other=0)
    tl.atomic_add(output_ptr + vals, 1, mask=mask)


@triton.jit
def fused_max_bincount_weights_fp32_kernel(
    input_ptr,
    weights_ptr,
    max_ptr,
    output_ptr,
    n_elements,
    output_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(input_ptr + offsets, mask=mask, other=0)
    w = tl.load(weights_ptr + offsets, mask=mask, other=0.0)
    w_fp32 = w.to(tl.float32)

    local_max = tl.max(vals, axis=0)
    tl.atomic_max(max_ptr, local_max)

    safe_mask = mask & (vals < output_size)
    tl.atomic_add(output_ptr + vals, w_fp32, mask=safe_mask)


@triton.jit
def bincount_weights_fp32_kernel(
    input_ptr,
    weights_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(input_ptr + offsets, mask=mask, other=0)
    w = tl.load(weights_ptr + offsets, mask=mask, other=0.0)
    w_fp32 = w.to(tl.float32)
    tl.atomic_add(output_ptr + vals, w_fp32, mask=mask)


def _fused_bincount_launch(
    input_contig,
    weights_contig,
    n,
    pre_size,
    minlength,
    out_dtype,
    grid,
    BLOCK_SIZE,
    num_warps,
):
    max_tensor = torch.zeros(1, dtype=torch.int32, device=input_contig.device)
    compute_dtype = torch.float32 if weights_contig is not None else torch.int32

    output = torch.zeros(pre_size, dtype=compute_dtype, device=input_contig.device)

    if weights_contig is None:
        fused_max_bincount_kernel[grid](
            input_contig,
            max_tensor,
            output,
            n,
            pre_size,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    else:
        fused_max_bincount_weights_fp32_kernel[grid](
            input_contig,
            weights_contig,
            max_tensor,
            output,
            n,
            pre_size,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

    max_val = int(max_tensor.item())
    needed_size = max(max_val + 1, minlength)

    if needed_size <= pre_size:
        return output[:needed_size]

    output = torch.zeros(needed_size, dtype=compute_dtype, device=input_contig.device)
    if weights_contig is None:
        bincount_kernel[grid](
            input_contig,
            output,
            n,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    else:
        bincount_weights_fp32_kernel[grid](
            input_contig,
            weights_contig,
            output,
            n,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    return output


def bincount(input, weights=None, minlength=0):
    assert input.dim() == 1
    assert minlength >= 0

    if weights is not None:
        assert weights.shape == input.shape

    n = input.numel()

    if n == 0:
        if weights is not None:
            return torch.zeros(minlength, dtype=weights.dtype, device=input.device)
        return torch.zeros(minlength, dtype=torch.int64, device=input.device)

    input_contig = input.contiguous()
    if input_contig.dtype == torch.int64:
        input_contig = input_contig.to(torch.int32)

    weights_contig = weights.contiguous() if weights is not None else None
    if weights_contig is not None and weights_contig.dtype == torch.float64:
        weights_contig = weights_contig.to(torch.float32)

    BLOCK_SIZE, num_warps = _select_params(n)
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    pre_size = _estimate_output_size(n, minlength)
    out_dtype = weights.dtype if weights is not None else torch.int64

    output = _fused_bincount_launch(
        input_contig,
        weights_contig,
        n,
        pre_size,
        minlength,
        out_dtype,
        grid,
        BLOCK_SIZE,
        num_warps,
    )

    if weights is None:
        output = output.to(torch.int64)
    elif weights.dtype != torch.float32:
        output = output.to(dtype=weights.dtype)

    return output
