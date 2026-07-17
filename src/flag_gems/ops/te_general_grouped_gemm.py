# Copyright (c) 2024, FlagGems. All rights reserved.
# Licensed under the Apache License, Version 2.0

"""
te_general_grouped_gemm operator implementation using Triton.

This implements a grouped GEMM operation compatible with TransformerEngine's
te_general_grouped_gemm interface. The grouped GEMM computes multiple independent
matrix multiplications in parallel.

Interface:
    te_general_grouped_gemm(
        A_list: List[Tensor],  # List of input matrices A
        transa: bool,          # Whether to transpose A
        B_list: List[Tensor],  # List of input matrices B
        transb: bool,          # Whether to transpose B
        out_list: List[Tensor], # List of output matrices
        out_dtype: Optional[DType] = None,
        m_splits: Optional[List[int]] = None,
        bias: Optional[List[Tensor]] = None,
        bias_dtype: Optional[DType] = None,
        single_output: bool = False,
        pre_gelu_out: Optional[List[Tensor]] = None,
        grad: bool = False,
        workspace: Optional[List[Tensor]] = None,
        workspace_size: int = 0,
        accumulate: bool = False,
        use_split_accumulator: bool = False,
        math_sm_count: int = 0,
    ) -> List[Tensor]
"""

import logging
from typing import List, Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def grouped_gemm_kernel(
    # Pointers to matrices
    A_ptr,
    B_ptr,
    C_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # Strides for A
    stride_am,
    stride_ak,
    # Strides for B
    stride_bk,
    stride_bn,
    # Strides for C
    stride_cm,
    stride_cn,
    # Accumulate flag
    accumulate: tl.constexpr,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Compute a single GEMM: C = A @ B (with optional transpose and accumulate).

    The caller is responsible for setting up strides correctly based on
    transpose flags.
    """
    pid = tl.program_id(0)

    # Compute grid dimensions
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # Group tiles for better L2 cache reuse
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    # Block start positions
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # Pointers to first block of A and B
    A_block_ptr = A_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B_block_ptr = B_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load A and B blocks with masking
        a_mask = (rm[:, None] < M) & (rk[None, :] + k * BLOCK_K < K)
        b_mask = (rk[:, None] + k * BLOCK_K < K) & (rn[None, :] < N)

        a = tl.load(A_block_ptr, mask=a_mask, other=0.0)
        b = tl.load(B_block_ptr, mask=b_mask, other=0.0)

        # Compute matmul for this block
        acc += tl.dot(a, b)

        # Advance pointers
        A_block_ptr += BLOCK_K * stride_ak
        B_block_ptr += BLOCK_K * stride_bk

    # Handle accumulate
    c_offset = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)

    if accumulate:
        c_existing = tl.load(C_ptr + c_offset, mask=c_mask, other=0.0)
        acc += c_existing

    # Store result
    tl.store(C_ptr + c_offset, acc.to(C_ptr.dtype.element_ty), mask=c_mask)


def _compute_single_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    transa: bool,
    transb: bool,
    accumulate: bool,
):
    """
    Compute a single GEMM operation using Triton kernel.

    Args:
        A: Input matrix A
        B: Input matrix B
        C: Output matrix C (will be written to)
        transa: If True, use A^T
        transb: If True, use B^T
        accumulate: If True, add to existing C values
    """
    # Get output dimensions from C
    M, N = C.shape

    # Determine K and strides based on transpose flags
    if transa:
        # A is (K, M), we use A^T which is (M, K)
        K = A.shape[0]
        stride_am = A.stride(1)  # stride to next row of A^T = stride in M dim of A
        stride_ak = A.stride(0)  # stride to next col of A^T = stride in K dim of A
    else:
        # A is (M, K)
        K = A.shape[1]
        stride_am = A.stride(0)
        stride_ak = A.stride(1)

    if transb:
        # B is (N, K), we use B^T which is (K, N)
        stride_bk = B.stride(1)  # stride to next row of B^T
        stride_bn = B.stride(0)  # stride to next col of B^T
    else:
        # B is (K, N)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)

    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # Kernel configuration
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    GROUP_M = 8

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    grouped_gemm_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        accumulate,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
    )


def te_general_grouped_gemm(
    A: List[torch.Tensor],
    transa: bool,
    B: List[torch.Tensor],
    transb: bool,
    out: List[torch.Tensor],
    out_dtype=None,
    m_splits: Optional[List[int]] = None,
    bias: Optional[List[torch.Tensor]] = None,
    bias_dtype=None,
    single_output: bool = False,
    pre_gelu_out: Optional[List[torch.Tensor]] = None,
    grad: bool = False,
    workspace: Optional[List[torch.Tensor]] = None,
    workspace_size: int = 0,
    accumulate: bool = False,
    use_split_accumulator: bool = False,
    math_sm_count: int = 0,
) -> List[torch.Tensor]:
    """
    Grouped GEMM operation compatible with TransformerEngine's te_general_grouped_gemm.

    Computes multiple independent matrix multiplications:
        out[i] = A[i] @ B[i]  (with optional transposes)

    Args:
        A: List of input matrices A
        transa: If True, transpose each A matrix
        B: List of input matrices B
        transb: If True, transpose each B matrix
        out: List of output matrices (will be written to)
        out_dtype: Output data type (optional, uses out[0].dtype if not specified)
        m_splits: Optional list of M dimension splits for single_output mode
        bias: Optional list of bias tensors to add
        bias_dtype: Data type for bias
        single_output: If True, out is a single tensor split by m_splits
        pre_gelu_out: Optional list of tensors for pre-GELU output
        grad: If True, this is a backward pass (affects bias handling)
        workspace: Optional workspace tensors for cuBLAS
        workspace_size: Size of workspace
        accumulate: If True, add to existing output values
        use_split_accumulator: Use split accumulator for better numerical stability
        math_sm_count: Number of SMs to use for computation

    Returns:
        bias: The bias tensors (or grad_bias if grad=True)
    """
    num_gemms = len(A)

    if num_gemms == 0:
        return bias if bias is not None else []

    # Validate inputs
    assert len(B) == num_gemms, "A and B must have same length"
    if not single_output:
        assert len(out) == num_gemms, "out must have same length as A"

    # Handle single_output mode
    if single_output and m_splits is not None:
        out_tensor = out[0] if isinstance(out, list) else out
        out_views = []
        start_idx = 0
        for i in range(num_gemms):
            size = m_splits[i]
            out_views.append(out_tensor[start_idx : start_idx + size])
            start_idx += size
        out = out_views

    # Compute each GEMM
    for i in range(num_gemms):
        _compute_single_gemm(
            A[i],
            B[i],
            out[i],
            transa,
            transb,
            accumulate,
        )

        # Add bias if provided
        if bias is not None and len(bias) > i and bias[i].numel() > 0:
            if not grad:
                # Forward: add bias to output
                out[i] += bias[i]

    # Handle pre_gelu_out (store output before GELU activation)
    # This is used for backward pass computation
    if pre_gelu_out is not None:
        for i in range(num_gemms):
            if len(pre_gelu_out) > i and pre_gelu_out[i].numel() > 0:
                pre_gelu_out[i].copy_(out[i])

    return bias if bias is not None else []


__all__ = ["te_general_grouped_gemm"]
