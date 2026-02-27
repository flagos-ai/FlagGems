import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def _fused_rms_norm_grad_dx_kernel(
    X,  # pointer to the input
    DY,
    INV_RMS,  # pointer to inverse rms
    DX,  # pointer to the output
    W,  # pointer to the weights
    dx_stride_r,
    dx_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    DX += pid * dx_stride_r
    X += pid * x_stride_r
    DY += pid * x_stride_r
    INV_RMS += pid

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
    inv_rms = tl.load(INV_RMS).to(tl.float32)
    dy = tl.load(DY + cols * x_stride_c, mask, other=0.0).to(tl.float32)
    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)

    dy = dy * w

    normalized_buf = x * inv_rms
    row_sum_stats = tl.sum(normalized_buf * dy, axis=0)

    norm_val = normalized_buf / N
    dx = (dy - norm_val * row_sum_stats) * inv_rms

    tl.store(DX + cols * dx_stride_c, dx, mask=mask)


@libentry()
@triton.jit
def _fused_rms_norm_grad_dw_kernel(
    X,  # pointer to the input
    DY,
    INV_RMS,  # pointer to inverse rms
    DW,  # pointer to the output
    dx_stride_r,
    dx_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    M,  # number of rows in X
    N,  # number of columns in X
    ROW_BLOCK_SIZE: tl.constexpr,
    COL_BLOCK_SIZE: tl.constexpr,
):
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)

    row_start = row_pid * ROW_BLOCK_SIZE
    col_start = col_pid * COL_BLOCK_SIZE

    offset = row_start * x_stride_r + col_start * x_stride_c
    X += offset
    DY += offset
    INV_RMS += row_start

    rows = tl.arange(0, ROW_BLOCK_SIZE)
    cols = tl.arange(0, COL_BLOCK_SIZE)

    row_mask = (row_start + rows) < M
    col_mask = (col_start + cols) < N

    x = tl.load(
        X + rows[:, None] * x_stride_r + cols[None, :] * x_stride_c,
        row_mask[:, None] & col_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    inv_rms = tl.load(INV_RMS + rows, row_mask, other=0.0).to(tl.float32)
    dy = tl.load(
        DY + rows[:, None] * x_stride_r + cols[None, :] * x_stride_c,
        row_mask[:, None] & col_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    d_weight = x * dy * inv_rms[:, None]
    # Sum over rows (axis=0) - masked rows are 0 (from other=0.0 in load), so sum is correct
    # The mask ensures invalid rows contribute 0 to the sum
    partial_dweight_sum = tl.sum(d_weight, axis=0)

    tl.store(
        DW + row_pid * N + col_start + cols,
        partial_dweight_sum,
        mask=col_mask,
    )


def _fused_rms_norm_backward(
    grad_out,
    input,
    normalized_shape,
    inv_rms,
    weight,
    eps=1e-5,
):
    """
    Compute the backward pass for fused RMS normalization.

    Args:
        grad_out: Gradient of the output tensor
        input: Input tensor from the forward pass
        normalized_shape: Shape of the normalized dimensions
        inv_rms: Inverse RMS values from the forward pass
        weight: Weight tensor (gamma)
        eps: Epsilon for numerical stability (unused in backward but kept for API consistency)

    Returns:
        Tuple of (input_grad, weight_grad)
    """
    logger.debug("GEMS _FUSED_RMS_NORM_BACKWARD")

    dim = input.ndim - len(normalized_shape)
    M = math.prod(input.shape[:dim])
    N = math.prod(normalized_shape)

    BLOCK_SIZE = triton.next_power_of_2(N)
    input = input.contiguous()
    grad_out = grad_out.contiguous()
    weight = weight.contiguous()
    dx = torch.empty_like(input)

    with torch_device_fn.device(input.device):
        _fused_rms_norm_grad_dx_kernel[M,](
            input, grad_out, inv_rms, dx, weight, N, 1, N, 1, N, eps, BLOCK_SIZE
        )

    ROW_BLOCK_SIZE = 16
    COL_BLOCK_SIZE = 256
    row_block_num = triton.cdiv(M, ROW_BLOCK_SIZE)
    col_block_num = triton.cdiv(N, COL_BLOCK_SIZE)

    partial_buffer = torch.empty(
        (row_block_num, N), dtype=torch.float32, device=input.device
    )

    with torch_device_fn.device(input.device):
        _fused_rms_norm_grad_dw_kernel[row_block_num, col_block_num](
            input,
            grad_out,
            inv_rms,
            partial_buffer,
            N,
            1,
            N,
            1,
            M,
            N,
            ROW_BLOCK_SIZE,
            COL_BLOCK_SIZE,
        )
        dw = (
            torch.sum(partial_buffer, dim=0, dtype=torch.float32)
            .to(input.dtype)
            .reshape(-1)
        )

    return dx, dw
