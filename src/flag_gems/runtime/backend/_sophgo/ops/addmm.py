"""
Sophgo TPU-specific addmm operator implementation.

Fix notes:
The original addmm kernel failed to compile on Sophgo TPU because:
1. Pointer updates in loops (a_ptrs += ...) generate scf.for iter_args pattern,
   which PPL ShapeInference cannot handle correctly.
2. Generated ppl.get_value operation (scalar pointer dereference) fails on TPU.

Solution:
1. Remove autotune, use fixed block size.
2. Recompute addresses on each iteration, avoid pointer updates.
3. Use 2D tensor mode for loading and storing.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle


@libentry()
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    alpha,
    beta,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_im,
    stride_in,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Sophgo TPU-specific addmm kernel.
    Computes: beta * bias + alpha * (mat1 @ mat2)

    Key modifications:
    - Recompute addresses on each loop iteration, avoiding pointer updates that produce iter_args.
    - Use 2D tensor mode for loading and storing.
    """
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)

    # Calculate row and column offsets for the current block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # K dimension loop - recompute addresses each iteration, avoid pointer updates
    num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
    for k_idx in range(0, num_k_blocks):
        # Calculate offset for the current K block
        offs_k = k_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        # Recompute A and B addresses (avoid pointer updates)
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # 2D tensor load with mask
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Matrix multiplication accumulation
        accumulator += tl.dot(a, b, allow_tf32=False)

    # Load bias (using 2D broadcasted bias)
    bias_ptrs = bias_ptr + offs_m[:, None] * stride_im + offs_n[None, :] * stride_in
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    bias = tl.load(bias_ptrs, mask=out_mask, other=0.0)

    # Compute final result: beta * bias + alpha * (mat1 @ mat2)
    c = accumulator * alpha + bias * beta
    c = c.to(bias.dtype)

    # Store result (using 2D tensor store)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=out_mask)


def addmm(bias, mat1, mat2, *, beta=1, alpha=1):
    """
    Sophgo TPU-specific addmm implementation.
    Computes: beta * bias + alpha * (mat1 @ mat2)

    Args:
        bias: Bias tensor, shape (N,) or (M, N).
        mat1: First matrix, shape (M, K).
        mat2: Second matrix, shape (K, N).
        beta: Scaling factor for bias, defaults to 1.
        alpha: Scaling factor for matrix multiplication result, defaults to 1.

    Returns:
        Output tensor, shape (M, N).
    """
    logging.debug("GEMS ADDMM (Sophgo TPU)")

    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"

    M, K = mat1.shape
    _, N = mat2.shape

    mat1 = mat1.contiguous()
    mat2 = mat2.contiguous()
    out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)

    # Broadcast bias to output shape and ensure contiguous
    bias = bias.broadcast_to(out.shape).contiguous()

    # Use fixed block sizes (consistent with tune_configs.yaml configuration)
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 8

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    with torch_device_fn.device(mat1.device):
        addmm_kernel[grid](
            mat1,
            mat2,
            bias,
            out,
            alpha,
            beta,
            M,
            N,
            K,
            mat1.stride(0),
            mat1.stride(1),
            mat2.stride(0),
            mat2.stride(1),
            bias.stride(0),
            bias.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )

    return out
