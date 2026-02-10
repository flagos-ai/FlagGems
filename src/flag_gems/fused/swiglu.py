import logging
from typing import Any, Optional

import torch
import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

sigmoid = tl.sigmoid
exp = tl_extra_shim.exp
pow = tl_extra_shim.pow

logger = logging.getLogger(__name__)


@triton.jit
def swiglu_kernel(
    input_ptr,
    output_ptr,
    M,
    H,
    stride_in_m,
    stride_in_h,
    stride_out_m,
    stride_out_h,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

    mask = (offs_m[:, None] < M) & (offs_h[None, :] < H)

    input_a_ptr = (
        input_ptr + offs_m[:, None] * stride_in_m + offs_h[None, :] * stride_in_h
    )
    input_b_ptr = (
        input_ptr + offs_m[:, None] * stride_in_m + (offs_h[None, :] + H) * stride_in_h
    )
    output_ptr = (
        output_ptr + offs_m[:, None] * stride_out_m + offs_h[None, :] * stride_out_h
    )

    x_a = tl.load(input_a_ptr, mask=mask, other=0.0).to(tl.float32)
    x_b = tl.load(input_b_ptr, mask=mask, other=0.0).to(tl.float32)

    silu_x_a = x_a * sigmoid(x_a)
    out = silu_x_a * x_b

    tl.store(output_ptr, out.to(x_a.dtype), mask=mask)


@triton.jit
def dswiglu_kernel(
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

    sig = sigmoid(x_a)
    silu = x_a * sig
    d_silu = sig + x_a * sig * (1 - sig)

    grad_a = grad_out * x_b * d_silu
    grad_b = grad_out * silu

    tl.store(grad_a_ptr, grad_a.to(x_a.dtype), mask=mask)
    tl.store(grad_b_ptr, grad_b.to(x_a.dtype), mask=mask)


def swiglu(input_tensor: torch.Tensor, quantizer: Optional[Any] = None) -> torch.Tensor:
    if input_tensor.shape[-1] % 2 != 0:
        raise ValueError(
            f"The last dimension of must be even number, got {input_tensor.shape[-1]}"
        )
    if not input_tensor.is_cuda:
        raise ValueError("Only CUDA tensor is supported by SwiGLU")

    shape = input_tensor.shape
    H = shape[-1] // 2
    M = input_tensor.numel() // (2 * H)
    input_2d = input_tensor.contiguous().view(M, 2 * H)
    output_2d = torch.empty(M, H, device=input_tensor.device, dtype=input_tensor.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(H, META["BLOCK_SIZE_H"]),
    )

    swiglu_kernel[grid](
        input_2d,
        output_2d,
        M,
        H,
        input_2d.stride(0),
        input_2d.stride(1),
        output_2d.stride(0),
        output_2d.stride(1),
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_H=64,
    )

    return output_2d.view(*shape[:-1], H)


def dswiglu(
    grad_output: torch.Tensor,
    input_tensor: torch.Tensor,
    quantizer: Optional[Any] = None,
) -> torch.Tensor:
    shape = input_tensor.shape
    assert (
        shape[-1] % 2 == 0
    ), f"The last dimension of input_tensor must be an even number, got {shape[-1]}"
    H = shape[-1] // 2
    M = input_tensor.numel() // (2 * H)
    grad_out_2d = grad_output.contiguous().view(M, H)
    input_2d = input_tensor.contiguous().view(M, 2 * H)
    grad_in_2d = torch.empty_like(input_2d)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(H, META["BLOCK_SIZE_H"]),
    )

    dswiglu_kernel[grid](
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
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_H=64,
    )

    return grad_in_2d.view_as(input_tensor)


__all__ = ["swiglu", "dswiglu"]
