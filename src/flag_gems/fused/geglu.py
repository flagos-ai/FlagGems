import logging

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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

    mask = (offs_m[:, None] < M) & (offs_h[None, :] < H)

    # input 切分为 x_a, x_b
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

    gelu_out = 0.5 * x_a * (1 + tanh(0.79788456 * x_a * (1 + 0.044715 * pow(x_a, 2))))
    out = gelu_out * x_b

    tl.store(output_ptr, out.to(tl.float32), mask=mask)


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

    # GELU 公式及其导数
    tanh_out = tanh(0.79788456 * x_a * (1 + 0.044715 * pow(x_a, 2)))
    gelu_out = 0.5 * x_a * (1 + tanh_out)

    # dgelu/dx
    sech2 = 1 - pow(tanh_out, 2)
    dgelu = 0.5 * (1 + tanh_out) + 0.5 * x_a * sech2 * 0.79788456 * (
        1 + 3 * 0.044715 * pow(x_a, 2)
    )

    # 反向传播
    grad_a = grad_out * x_b * dgelu
    grad_b = grad_out * gelu_out

    tl.store(grad_a_ptr, grad_a.to(x_a.dtype), mask=mask)
    tl.store(grad_b_ptr, grad_b.to(x_a.dtype), mask=mask)


def geglu(input_tensor: torch.Tensor) -> torch.Tensor:
    shape = input_tensor.shape
    H = shape[-1] // 2
    M = input_tensor.numel() // (2 * H)

    input_2d = input_tensor.contiguous().view(M, 2 * H)
    output_2d = torch.empty(M, H, device=input_tensor.device, dtype=input_tensor.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(H, META["BLOCK_SIZE_H"]),
    )

    geglu_kernel[grid](
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


def dgeglu(grad_output: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
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
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_H=64,
    )

    return grad_in_2d.view_as(input_tensor)
