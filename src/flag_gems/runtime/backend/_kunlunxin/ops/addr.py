import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@libentry()
@triton.jit(do_not_specialize=["beta", "alpha"])
def addr_kernel_1d(
    input_ptr,
    vec1_ptr,
    vec2_ptr,
    output_ptr,
    beta,
    alpha,
    M,
    N,
    stride_input_m,
    stride_input_n,
    stride_output_m,
    stride_output_n,
    BLOCK_SIZE: tl.constexpr,
):
    """1D kernel that processes one row at a time for better parallelism."""
    pid = tl.program_id(0)
    row_idx = pid

    if row_idx >= M:
        return

    # Load vec1 element for this row
    vec1_val = tl.load(vec1_ptr + row_idx).to(tl.float32)

    # Process columns in blocks
    for col_start in range(0, N, BLOCK_SIZE):
        col_offs = col_start + tl.arange(0, BLOCK_SIZE)
        col_mask = col_offs < N

        # Load vec2 elements
        vec2_vals = tl.load(vec2_ptr + col_offs, mask=col_mask, other=0.0).to(
            tl.float32
        )

        # Load input elements
        input_ptrs = input_ptr + row_idx * stride_input_m + col_offs * stride_input_n
        input_vals = tl.load(input_ptrs, mask=col_mask, other=0.0).to(tl.float32)

        # Compute result
        result = beta * input_vals + alpha * (vec1_val * vec2_vals)

        # Store result
        output_ptrs = (
            output_ptr + row_idx * stride_output_m + col_offs * stride_output_n
        )
        tl.store(output_ptrs, result, mask=col_mask)


@libentry()
@triton.jit(do_not_specialize=["beta", "alpha"])
def addr_kernel_2d(
    input_ptr,
    vec1_ptr,
    vec2_ptr,
    output_ptr,
    beta,
    alpha,
    M,
    N,
    stride_input_m,
    stride_input_n,
    stride_vec1,
    stride_vec2,
    stride_output_m,
    stride_output_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    vec1_ptrs = vec1_ptr + offs_m * stride_vec1
    vec2_ptrs = vec2_ptr + offs_n * stride_vec2

    mask_m = offs_m < M
    mask_n = offs_n < N

    vec1 = tl.load(vec1_ptrs, mask=mask_m, other=0.0).to(tl.float32)
    vec2 = tl.load(vec2_ptrs, mask=mask_n, other=0.0).to(tl.float32)

    input_ptrs = (
        input_ptr + offs_m[:, None] * stride_input_m + offs_n[None, :] * stride_input_n
    )

    mask_2d = mask_m[:, None] & mask_n[None, :]
    input_val = tl.load(input_ptrs, mask=mask_2d, other=0.0).to(tl.float32)

    result = beta * input_val + alpha * (vec1[:, None] * vec2[None, :])

    output_ptrs = (
        output_ptr
        + offs_m[:, None] * stride_output_m
        + offs_n[None, :] * stride_output_n
    )
    tl.store(output_ptrs, result, mask=mask_2d)


def addr(input, vec1, vec2, *, beta=1, alpha=1):
    logger.debug("GEMS ADDR")
    if vec1.dim() != 1 or vec2.dim() != 1:
        raise ValueError("addr: expected 1-D vectors")

    M = vec1.shape[0]
    N = vec2.shape[0]
    output_shape = (M, N)

    try:
        input_broadcasted = torch.broadcast_to(input, output_shape)
    except RuntimeError:
        raise ValueError(
            f"addr: input tensor of shape {input.shape} cannot be broadcast to output shape {output_shape}"
        )
    out = torch.empty(output_shape, device=input.device, dtype=input.dtype)

    # Use 1D kernel for small M (row-parallel), 2D kernel for larger matrices
    if M <= 128:
        BLOCK_SIZE = 1024
        grid = (M,)
        with torch_device_fn.device(input.device):
            addr_kernel_1d[grid](
                input_broadcasted,
                vec1,
                vec2,
                out,
                beta,
                alpha,
                M,
                N,
                input_broadcasted.stride(0),
                input_broadcasted.stride(1),
                out.stride(0),
                out.stride(1),
                BLOCK_SIZE=BLOCK_SIZE,
            )
    else:
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
        grid = (
            triton.cdiv(M, BLOCK_SIZE_M),
            triton.cdiv(N, BLOCK_SIZE_N),
        )
        with torch_device_fn.device(input.device):
            addr_kernel_2d[grid](
                input_broadcasted,
                vec1,
                vec2,
                out,
                beta,
                alpha,
                M,
                N,
                input_broadcasted.stride(0),
                input_broadcasted.stride(1),
                vec1.stride(0),
                vec2.stride(0),
                out.stride(0),
                out.stride(1),
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
            )
    return out
