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

from flag_gems.utils import libentry

from ..heuristics_config_utils import dreglu_dswiglu_config, reglu_swiglu_config

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def swiglu_kernel(
    x_ptr,
    y_ptr,
    M,
    N_OUT,
    stride_x_m,
    stride_x_n,
    stride_y_m,
    stride_y_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    x_ptr_a = x_ptr + offs_m[:, None] * stride_x_m + offs_n[None, :] * stride_x_n
    x_ptr_b = (
        x_ptr + offs_m[:, None] * stride_x_m + (offs_n[None, :] + N_OUT) * stride_x_n
    )
    y_ptr = y_ptr + offs_m[:, None] * stride_y_m + offs_n[None, :] * stride_y_n
    if NEED_MASK:
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_OUT)
        block_a = tl.load(x_ptr_a, mask=mask, other=0.0).to(tl.float32)
        block_b = tl.load(x_ptr_b, mask=mask, other=0.0).to(tl.float32)
        silu_a = block_a * tl.sigmoid(block_a)
        tl.store(y_ptr, (silu_a * block_b).to(y_ptr.dtype.element_ty), mask=mask)
    else:
        block_a = tl.load(x_ptr_a).to(tl.float32)
        block_b = tl.load(x_ptr_b).to(tl.float32)
        silu_a = block_a * tl.sigmoid(block_a)
        tl.store(y_ptr, (silu_a * block_b).to(y_ptr.dtype.element_ty))


def swiglu(input_tensor: torch.Tensor, quantizer: Optional[Any] = None) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN SWIGLU_FORWARD")
    shape = input_tensor.shape
    if input_tensor.dim() < 1:
        raise ValueError("Input tensor must have at least 1 dimension.")
    last_dim = shape[-1]
    if last_dim % 2 != 0:
        raise ValueError(
            f"The last dimension of the input tensor must be even, but got {last_dim}."
        )
    N_OUT = last_dim // 2
    M = input_tensor.numel() // last_dim
    if input_tensor.numel() == 0:
        output_shape = (*shape[:-1], N_OUT)
        return torch.empty(
            output_shape, device=input_tensor.device, dtype=input_tensor.dtype
        )
    input_2d = input_tensor.contiguous().view(M, last_dim)
    output_2d = torch.empty(
        (M, N_OUT), device=input_tensor.device, dtype=input_tensor.dtype
    )
    block_m, block_n, num_warps = reglu_swiglu_config(input_tensor.dtype, M, N_OUT)
    need_mask = (M % block_m != 0) or (N_OUT % block_n != 0)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N_OUT, block_n))
    swiglu_kernel[grid](
        input_2d,
        output_2d,
        M,
        N_OUT,
        input_2d.stride(0),
        input_2d.stride(1),
        output_2d.stride(0),
        output_2d.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        NEED_MASK=need_mask,
        num_warps=num_warps,
    )
    output_shape = (*shape[:-1], N_OUT)
    return output_2d.view(output_shape)


__all__ = ["swiglu", "dswiglu"]


@libentry()
@triton.jit
def dswiglu_kernel(
    grad_output_ptr,
    input_ptr,
    grad_input_ptr,
    M,
    N,
    stride_grad_out_m,
    stride_grad_out_n,
    stride_in_m,
    stride_in_n,
    stride_grad_in_m,
    stride_grad_in_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    grad_output_ptr += (
        offs_m[:, None] * stride_grad_out_m + offs_n[None, :] * stride_grad_out_n
    )
    input_ptr_a = (
        input_ptr + offs_m[:, None] * stride_in_m + offs_n[None, :] * stride_in_n
    )
    input_ptr_b = (
        input_ptr + offs_m[:, None] * stride_in_m + (offs_n[None, :] + N) * stride_in_n
    )
    grad_input_ptr_a = (
        grad_input_ptr
        + offs_m[:, None] * stride_grad_in_m
        + offs_n[None, :] * stride_grad_in_n
    )
    grad_input_ptr_b = (
        grad_input_ptr
        + offs_m[:, None] * stride_grad_in_m
        + (offs_n[None, :] + N) * stride_grad_in_n
    )
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if NEED_MASK:
        grad_out = tl.load(grad_output_ptr, mask=mask, other=0.0).to(tl.float32)
        block_a = tl.load(input_ptr_a, mask=mask, other=0.0).to(tl.float32)
        block_b = tl.load(input_ptr_b, mask=mask, other=0.0).to(tl.float32)
        sig = tl.sigmoid(block_a)
        silu_a = block_a * sig
        d_silu_a = sig + block_a * sig * (1.0 - sig)
        tl.store(grad_input_ptr_a, grad_out * d_silu_a * block_b, mask=mask)
        tl.store(grad_input_ptr_b, grad_out * silu_a, mask=mask)
    else:
        grad_out = tl.load(grad_output_ptr).to(tl.float32)
        block_a = tl.load(input_ptr_a).to(tl.float32)
        block_b = tl.load(input_ptr_b).to(tl.float32)
        sig = tl.sigmoid(block_a)
        silu_a = block_a * sig
        d_silu_a = sig + block_a * sig * (1.0 - sig)
        tl.store(grad_input_ptr_a, grad_out * d_silu_a * block_b)
        tl.store(grad_input_ptr_b, grad_out * silu_a)


def dswiglu(
    grad_output: torch.Tensor,
    input_tensor: torch.Tensor,
    quantizer: Optional[Any] = None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN DSWIGLU")
    shape = input_tensor.shape
    if input_tensor.dim() < 1:
        raise ValueError("Input tensor must have at least 1 dimension.")
    if shape[-1] % 2 != 0:
        raise ValueError(
            f"The last dimension of the input tensor must be even, but got {shape[-1]}."
        )
    if shape[:-1] != grad_output.shape[:-1] or shape[-1] != 2 * grad_output.shape[-1]:
        raise ValueError(
            f"Shape mismatch: input {shape} vs grad_output {grad_output.shape}"
        )
    N = shape[-1] // 2
    if input_tensor.numel() == 0:
        return torch.empty(shape, device=input_tensor.device, dtype=input_tensor.dtype)
    M = input_tensor.numel() // shape[-1]
    grad_output_2d = grad_output.contiguous().view(M, N)
    input_2d = input_tensor.contiguous().view(M, 2 * N)
    grad_input = torch.empty_like(input_2d)
    block_m, block_n, num_warps = dreglu_dswiglu_config(input_tensor.dtype, M, N)
    need_mask = (M % block_m != 0) or (N % block_n != 0)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    dswiglu_kernel[grid](
        grad_output_2d,
        input_2d,
        grad_input,
        M,
        N,
        grad_output_2d.stride(0),
        grad_output_2d.stride(1),
        input_2d.stride(0),
        input_2d.stride(1),
        grad_input.stride(0),
        grad_input.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        NEED_MASK=need_mask,
        num_warps=num_warps,
    )
    return grad_input.view(shape)
