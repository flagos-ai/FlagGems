import logging
from typing import Any, Optional

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def heur_tile_m(args):
    return triton.cdiv(args["M"], 12)


def heru_tile_n(args):
    import builtins

    return builtins.min(args["N"], 8192)


@libentry()
@triton.jit(do_not_specialize=["num_tasks"])
def dreglu_kernel(
    grad_output_ptr,
    input_ptr,
    grad_input_ptr,
    num_tasks,
    N: tl.constexpr,
    TILE: tl.constexpr,
    TILES_PER_CTA: tl.constexpr,
    ONE_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    if ONE_TILE:
        tid = pid * TILE + tl.arange(0, TILE)
        mask = tid < num_tasks
        a_off = (tid // N) * N
        grad_out = tl.load(grad_output_ptr + tid, mask=mask).to(tl.float32)
        block_a = tl.load(input_ptr + tid + a_off, mask=mask).to(tl.float32)
        block_b = tl.load(input_ptr + tid + a_off + N, mask=mask).to(tl.float32)
        relu_a = tl.maximum(block_a, 0.0)
        d_relu_a = tl.where(block_a > 0, 1.0, 0.0)
        grad_a = grad_out * d_relu_a * block_b
        grad_b = grad_out * relu_a
        tl.store(
            grad_input_ptr + tid + a_off,
            grad_a.to(input_ptr.type.element_ty),
            mask=mask,
        )
        tl.store(
            grad_input_ptr + tid + a_off + N,
            grad_b.to(input_ptr.type.element_ty),
            mask=mask,
        )
    else:
        num_ctas = tl.num_programs(0)
        for j in range(0, TILES_PER_CTA):
            tile_id = pid + j * num_ctas
            tid = tile_id * TILE + tl.arange(0, TILE)
            mask = tid < num_tasks
            a_off = (tid // N) * N
            grad_out = tl.load(grad_output_ptr + tid, mask=mask).to(tl.float32)
            block_a = tl.load(input_ptr + tid + a_off, mask=mask).to(tl.float32)
            block_b = tl.load(input_ptr + tid + a_off + N, mask=mask).to(tl.float32)
            relu_a = tl.maximum(block_a, 0.0)
            d_relu_a = tl.where(block_a > 0, 1.0, 0.0)
            grad_a = grad_out * d_relu_a * block_b
            grad_b = grad_out * relu_a
            tl.store(
                grad_input_ptr + tid + a_off,
                grad_a.to(input_ptr.type.element_ty),
                mask=mask,
            )
            tl.store(
                grad_input_ptr + tid + a_off + N,
                grad_b.to(input_ptr.type.element_ty),
                mask=mask,
            )


@libentry()
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


@libentry()
@triton.jit
def reglu_pair_kernel(
    input_ptr,
    output_ptr,
    num_tasks,
    N_OUT: tl.constexpr,
    TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < num_tasks
    a_off = (tid // N_OUT) * N_OUT
    x_a = tl.load(input_ptr + tid + a_off, mask=mask).to(tl.float32)
    x_b = tl.load(input_ptr + tid + a_off + N_OUT, mask=mask).to(tl.float32)
    gate = tl.maximum(x_a, 0.0)
    tl.store(output_ptr + tid, (gate * x_b).to(input_ptr.type.element_ty), mask=mask)


def _pick_reglu_pair_tile(dtype, M, N_OUT):
    if dtype == torch.bfloat16:
        return 512 if N_OUT <= 64 else 1024
    if dtype == torch.float16:
        return 1024 if N_OUT <= 1024 else 16384
    if N_OUT > 1024:
        return 8192
    return 2048 if M >= 65536 else 1024


def _pick_reglu_config(dtype, M, N_OUT):
    if N_OUT >= 2048 and M >= 1024:
        if dtype == torch.float32:
            if N_OUT >= 65536:
                return 1, 8192, 8
            elif N_OUT >= 4096:
                return 1, 4096, 8
            else:
                return 1, 2048, 8
        elif dtype == torch.bfloat16:
            if N_OUT >= 65536:
                return 1, 16384, 16
            elif N_OUT >= 4096:
                return 1, 4096, 8
            else:
                return 1, 2048, 8
        return 8, 1024, 4
    if N_OUT <= 64:
        if M < 256:
            return 1, 1024, 4
        if dtype == torch.float32:
            return 8, 1024, 4
        return 8, 512, 4
    return (16, 1024, 4) if M >= 8192 else (1, 1024, 4)


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
    if input_tensor.numel() == 0:
        output_shape = (*shape[:-1], N_OUT)
        return torch.empty(
            output_shape, device=input_tensor.device, dtype=input_tensor.dtype
        )
    M = input_tensor.numel() // last_dim
    input_2d = input_tensor.contiguous().view(M, last_dim)
    output_2d = torch.empty(
        (M, N_OUT), device=input_tensor.device, dtype=input_tensor.dtype
    )
    tile = _pick_reglu_pair_tile(input_tensor.dtype, M, N_OUT)
    num_tasks = M * N_OUT
    reglu_pair_kernel[(triton.cdiv(num_tasks, tile),)](
        input_2d,
        output_2d,
        num_tasks,
        N_OUT=N_OUT,
        TILE=tile,
        num_warps=4,
    )
    output_shape = (*shape[:-1], N_OUT)
    return output_2d.view(output_shape)


def _pick_dreglu_config(dtype, M, N):
    f16 = dtype == torch.float16
    f32 = dtype == torch.float32
    if N >= 2048:
        if f16:
            if N >= 65536:
                return 1, 16384, 16
            if N == 4096:
                return 1, 4096, 8
            return 342, 2048, 4
        if f32:
            if N >= 65536:
                return 4, 8192, 8
            if N == 4096:
                return 1, 4096, 8
            return 4, 2048, 8
        if N >= 65536:
            return 1, 16384, 8
        if N == 4096:
            return 1, 4096, 8
        return 4, 2048, 4
    if N <= 64:
        if N == 1:
            if M <= 1024:
                return (8, 64, 4) if f32 else (16, 64, 8)
            return (6, 32, 4) if f32 else (32, 64, 4)
        if N == 16:
            if M <= 1024:
                return (1, 1024, 4) if f32 else (4, 256, 4)
            return (8, 1024, 4) if f32 else (4, 256, 4)
        if N == 32:
            return (1, 1024, 4) if f32 else ((1, 256, 4) if f16 else (1, 1024, 4))
        return (8, 256, 4)
    if f16:
        if M >= 32768:
            return 342, 2048, 4
        return 1, 2048, 8
    if M >= 32768:
        return (8, 1024, 4) if f32 else (1, 1024, 4)
    if f32:
        return (8, 1024, 4) if M > 1024 else (1, 1024, 4)
    return 1, 1024, 4


def dreglu(
    grad_output: torch.Tensor,
    input_tensor: torch.Tensor,
    quantizer: Optional[Any] = None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN DREGLU")
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
    num_tasks = grad_output_2d.numel()
    if num_tasks == 0:
        return grad_input.view(shape)
    num_ctas = 12
    num_tiles = num_ctas
    tile = triton.next_power_of_2(triton.cdiv(num_tasks, num_tiles))
    tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    dreglu_kernel[(num_ctas, 1, 1)](
        grad_output_2d,
        input_2d,
        grad_input,
        num_tasks,
        N=N,
        TILE=tile,
        TILES_PER_CTA=tiles_per_cta,
        ONE_TILE=tiles_per_cta == 1,
    )
    return grad_input.view(shape)
