import logging
from typing import Any, Optional

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry, tl_extra_shim

erf = tl_extra_shim.erf
exp = tl_extra_shim.exp
tanh = tl_extra_shim.tanh

logger = logging.getLogger(__name__)


@libentry()
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
    NEED_MASK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

    input_a_ptr = (
        input_ptr + offs_m[:, None] * stride_in_m + offs_h[None, :] * stride_in_h
    )
    input_b_ptr = (
        input_ptr + offs_m[:, None] * stride_in_m + (offs_h[None, :] + H) * stride_in_h
    )
    output_ptr = (
        output_ptr + offs_m[:, None] * stride_out_m + offs_h[None, :] * stride_out_h
    )

    if NEED_MASK:
        mask = (offs_m[:, None] < M) & (offs_h[None, :] < H)
        x_a = tl.load(input_a_ptr, mask=mask, other=0.0).to(tl.float32)
        x_b = tl.load(input_b_ptr, mask=mask, other=0.0).to(tl.float32)
        gelu_out = 0.5 * x_a * (1 + tanh(0.79788456 * x_a * (1 + 0.044715 * x_a * x_a)))
        tl.store(output_ptr, gelu_out * x_b, mask=mask)
    else:
        x_a = tl.load(input_a_ptr).to(tl.float32)
        x_b = tl.load(input_b_ptr).to(tl.float32)
        gelu_out = 0.5 * x_a * (1 + tanh(0.79788456 * x_a * (1 + 0.044715 * x_a * x_a)))
        tl.store(output_ptr, gelu_out * x_b)


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

    tanh_out = tanh(0.79788456 * x_a * (1 + 0.044715 * x_a * x_a))
    gelu_out = 0.5 * x_a * (1 + tanh_out)

    sech2 = 1 - tanh_out * tanh_out
    dgelu = 0.5 * (1 + tanh_out) + 0.5 * x_a * sech2 * 0.79788456 * (
        1 + 3 * 0.044715 * x_a * x_a
    )

    grad_a = grad_out * x_b * dgelu
    grad_b = grad_out * gelu_out

    tl.store(grad_a_ptr, grad_a.to(x_a.dtype), mask=mask)
    tl.store(grad_b_ptr, grad_b.to(x_a.dtype), mask=mask)


@libentry()
@triton.jit
def geglu_pair_kernel(
    input_ptr,
    output_ptr,
    num_tasks,
    H: tl.constexpr,
    TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < num_tasks
    a_off = (tid // H) * H
    x_a = tl.load(input_ptr + tid + a_off, mask=mask).to(tl.float32)
    x_b = tl.load(input_ptr + tid + a_off + H, mask=mask).to(tl.float32)
    gelu_out = 0.5 * x_a * (1 + tanh(0.79788456 * x_a * (1 + 0.044715 * x_a * x_a)))
    tl.store(output_ptr + tid, gelu_out * x_b, mask=mask)


def _pick_geglu_config(dtype, M, H):
    f32 = dtype == torch.float32
    bf16 = dtype == torch.bfloat16
    if H >= 2048:
        if H >= 8192:
            if f32:
                return 1, 8192, 8
            return (1, 16384, 8) if H % 16384 == 0 else (1, 8192, 8)
        if H >= 4096:
            return 1, 4096, 8
        return (1, 2048, 8) if bf16 else (4, 2048, 8)
    if H > 64:
        if f32:
            if M >= 32768:
                return 64, 512, 4
            return (8, 512, 4) if M >= 4096 else (2, 512, 4)
        if M >= 32768:
            return 16, 1024, 4
        return (8, 1024, 4) if M >= 4096 else (1, 1024, 4)
    if H == 1:
        if f32:
            return 8, 128, 4
        return (32, 128, 4) if M <= 1024 else (16, 64, 4)
    if M < 256:
        return (1, 256, 4) if f32 else (1, 512, 4)
    if M > 1024:
        return 64, 256, 4
    return 8, 512, 4


def _pick_geglu_pair_tile(dtype, M, H):
    if dtype == torch.bfloat16:
        return 512 if H <= 64 else 1024
    if dtype == torch.float32 and M >= 65536:
        return 2048
    return 1024


def geglu(input_tensor: torch.Tensor, quantizer: Optional[Any] = None) -> torch.Tensor:
    shape = input_tensor.shape
    if input_tensor.dim() < 1:
        raise ValueError("Input tensor must have at least 1 dimension.")
    last_dim = shape[-1]
    if last_dim % 2 != 0:
        raise ValueError(
            f"The last dimension of the input tensor must be even, but got {last_dim}."
        )
    H = last_dim // 2
    output_shape = (*shape[:-1], H)
    if input_tensor.numel() == 0:
        return torch.empty(
            output_shape, device=input_tensor.device, dtype=input_tensor.dtype
        )
    M = input_tensor.numel() // last_dim

    input_2d = input_tensor.contiguous().view(M, last_dim)
    output_2d = torch.empty(M, H, device=input_tensor.device, dtype=input_tensor.dtype)

    if H <= 1024:
        tile = _pick_geglu_pair_tile(input_tensor.dtype, M, H)
        num_tasks = M * H
        geglu_pair_kernel[(triton.cdiv(num_tasks, tile),)](
            input_2d,
            output_2d,
            num_tasks,
            H=H,
            TILE=tile,
            num_warps=4,
        )
        return output_2d.view(output_shape)

    block_m, block_h, num_warps = _pick_geglu_config(input_tensor.dtype, M, H)
    need_mask = (M % block_m != 0) or (H % block_h != 0)
    grid = (triton.cdiv(M, block_m), triton.cdiv(H, block_h))

    geglu_kernel[grid](
        input_2d,
        output_2d,
        M,
        H,
        input_2d.stride(0),
        input_2d.stride(1),
        output_2d.stride(0),
        output_2d.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_H=block_h,
        NEED_MASK=need_mask,
        num_warps=num_warps,
    )
    return output_2d.view(output_shape)


def _pick_dgeglu_config(dtype, M, H):
    if H >= 2048:
        if H % 8192 == 0:
            return 1, 8192, 8
        if H % 4096 == 0:
            return 1, 4096, 8
        return 4, 2048, 8
    if H >= 256:
        return 1, 1024, 8
    if H == 1:
        return 8, 128, 4
    return 8, 512, 4


def dgeglu(
    grad_output: torch.Tensor,
    input_tensor: torch.Tensor,
    quantizer: Optional[Any] = None,
) -> torch.Tensor:
    shape = input_tensor.shape
    H = shape[-1] // 2
    M = input_tensor.numel() // (2 * H)

    grad_out_2d = grad_output.contiguous().view(M, H)
    input_2d = input_tensor.contiguous().view(M, 2 * H)
    grad_in_2d = torch.empty_like(input_2d)

    block_m, block_h, num_warps = _pick_dgeglu_config(input_tensor.dtype, M, H)
    grid = (triton.cdiv(M, block_m), triton.cdiv(H, block_h))

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
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_H=block_h,
        num_warps=num_warps,
    )
    return grad_in_2d.view_as(input_tensor)
