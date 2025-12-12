import logging
from typing import Any, Optional

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("gated_activation"),
    key=["M", "N"],
)
@triton.jit
def dreglu_kernel(
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
    grad_out = tl.load(grad_output_ptr, mask=mask, other=0.0)
    block_a = tl.load(input_ptr_a, mask=mask, other=0.0)
    block_b = tl.load(input_ptr_b, mask=mask, other=0.0)
    relu_a = tl.maximum(block_a, 0.0)
    d_relu_a = tl.where(block_a > 0, 1.0, 0.0)
    grad_a = grad_out * d_relu_a * block_b
    grad_b = grad_out * relu_a
    tl.store(grad_input_ptr_a, grad_a, mask=mask)
    tl.store(grad_input_ptr_b, grad_b, mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("gated_activation"),
    key=["M", "N_OUT"],
)
@triton.jit
def reglu_kernel(
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
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_OUT)
    block_a = tl.load(x_ptr_a, mask=mask, other=0.0)
    block_b = tl.load(x_ptr_b, mask=mask, other=0.0)
    gate = tl.where(block_a > 0, block_a, 0.0)
    output = gate * block_b
    tl.store(y_ptr, output, mask=mask)


def reglu(input_tensor: torch.Tensor, quantizer: Optional[Any] = None) -> torch.Tensor:
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
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N_OUT, META["BLOCK_N"]),
    )
    reglu_kernel[grid](
        input_2d,
        output_2d,
        M,
        N_OUT,
        input_2d.stride(0),
        input_2d.stride(1),
        output_2d.stride(0),
        output_2d.stride(1),
    )
    output_shape = (*shape[:-1], N_OUT)
    return output_2d.view(output_shape)


def dreglu(
    grad_output: torch.Tensor,
    input_tensor: torch.Tensor,
    quantizer: Optional[Any] = None,
) -> torch.Tensor:
    shape = input_tensor.shape
    if shape[:-1] != grad_output.shape[:-1] or shape[-1] != 2 * grad_output.shape[-1]:
        raise ValueError(
            f"Shape mismatch: input {shape} vs grad_output {grad_output.shape}"
        )
    M = grad_output.numel() // grad_output.shape[-1]
    N = grad_output.shape[-1]
    grad_output_2d = grad_output.contiguous().view(M, N)
    input_2d = input_tensor.contiguous().view(M, 2 * N)
    grad_input = torch.empty_like(input_2d)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )
    dreglu_kernel[grid](
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
    )
    return grad_input.view(shape)
