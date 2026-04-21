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
@triton.jit
def _pdist_backward_p2_kernel(
    grad_ptr,
    x_ptr,
    pdist_ptr,
    out_ptr,
    N,
    M,
    stride_x,
    stride_grad_out,
    BLOCK_M: tl.constexpr,
):
    """Backward kernel for p=2 case."""
    pid_n = tle.program_id(1)
    pid_m = tle.program_id(0)

    # Each block computes gradient for one row of x
    row_offset = pid_n * stride_x
    x_ptr_row = x_ptr + row_offset
    out_ptr_row = out_ptr + pid_n * stride_grad_out

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    # Load x[pid_n, :]
    x_vals = tl.load(x_ptr_row + m_offsets, mask=m_mask, other=0.0).to(tl.float32)

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # For each pair (i, j) where i < j or i > j, compute contribution
    # pdist output indices: for row i, pairs are (i, i+1), (i, i+2), ..., (i, n-1)
    # and (0, i), (1, i), ..., (i-1, i) for j < i

    # First: pairs where this row is the first index (i, j) with i < j
    for j in range(pid_n + 1, N):
        # Load x[j, :]
        xj_offset = j * stride_x
        xj_vals = tl.load(x_ptr + xj_offset + m_offsets, mask=m_mask, other=0.0).to(tl.float32)

        diff = x_vals - xj_vals

        # pdist index for (pid_n, j)
        pdist_idx = pid_n * N - (pid_n * (pid_n + 1)) // 2 + (j - pid_n - 1)
        g = tl.load(grad_ptr + pdist_idx).to(tl.float32)
        dist = tl.load(pdist_ptr + pdist_idx).to(tl.float32)

        # Avoid division by zero
        safe_dist = tl.where(dist != 0.0, dist, 1.0)
        scale = g / safe_dist
        contrib = diff * scale
        acc += contrib

    # Second: pairs where this row is the second index (j, i) with j < i
    for j in range(0, pid_n):
        # Load x[j, :]
        xj_offset = j * stride_x
        xj_vals = tl.load(x_ptr + xj_offset + m_offsets, mask=m_mask, other=0.0).to(tl.float32)

        diff = x_vals - xj_vals

        # pdist index for (j, pid_n)
        pdist_idx = j * N - (j * (j + 1)) // 2 + (pid_n - j - 1)
        g = tl.load(grad_ptr + pdist_idx).to(tl.float32)
        dist = tl.load(pdist_ptr + pdist_idx).to(tl.float32)

        # Avoid division by zero
        safe_dist = tl.where(dist != 0.0, dist, 1.0)
        scale = g / safe_dist
        contrib = diff * scale
        acc += contrib

    tl.store(out_ptr_row + m_offsets, acc.to(out_ptr.dtype.element_ty), mask=m_mask)


@libentry()
@triton.jit
def _pdist_backward_p1_kernel(
    grad_ptr,
    x_ptr,
    out_ptr,
    N,
    M,
    stride_x,
    stride_grad_out,
    BLOCK_M: tl.constexpr,
):
    """Backward kernel for p=1 case (L1 norm)."""
    pid_n = tle.program_id(1)
    pid_m = tle.program_id(0)

    row_offset = pid_n * stride_x
    x_ptr_row = x_ptr + row_offset
    out_ptr_row = out_ptr + pid_n * stride_grad_out

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    x_vals = tl.load(x_ptr_row + m_offsets, mask=m_mask, other=0.0).to(tl.float32)

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Pairs where this row is first (i, j) with i < j
    for j in range(pid_n + 1, N):
        xj_offset = j * stride_x
        xj_vals = tl.load(x_ptr + xj_offset + m_offsets, mask=m_mask, other=0.0).to(tl.float32)

        diff = x_vals - xj_vals
        sign = tl.where(diff > 0, 1.0, tl.where(diff < 0, -1.0, 0.0))

        pdist_idx = pid_n * N - (pid_n * (pid_n + 1)) // 2 + (j - pid_n - 1)
        g = tl.load(grad_ptr + pdist_idx).to(tl.float32)

        acc += g * sign

    # Pairs where this row is second (j, i) with j < i
    for j in range(0, pid_n):
        xj_offset = j * stride_x
        xj_vals = tl.load(x_ptr + xj_offset + m_offsets, mask=m_mask, other=0.0).to(tl.float32)

        diff = x_vals - xj_vals
        sign = tl.where(diff > 0, 1.0, tl.where(diff < 0, -1.0, 0.0))

        pdist_idx = j * N - (j * (j + 1)) // 2 + (pid_n - j - 1)
        g = tl.load(grad_ptr + pdist_idx).to(tl.float32)

        acc += g * sign

    tl.store(out_ptr_row + m_offsets, acc.to(out_ptr.dtype.element_ty), mask=m_mask)


@libentry()
@triton.jit
def _pdist_backward_general_kernel(
    grad_ptr,
    x_ptr,
    pdist_ptr,
    out_ptr,
    N,
    M,
    p_val,
    stride_x,
    stride_grad_out,
    BLOCK_M: tl.constexpr,
):
    """Backward kernel for general p value."""
    pid_n = tle.program_id(1)
    pid_m = tle.program_id(0)

    row_offset = pid_n * stride_x
    x_ptr_row = x_ptr + row_offset
    out_ptr_row = out_ptr + pid_n * stride_grad_out

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    x_vals = tl.load(x_ptr_row + m_offsets, mask=m_mask, other=0.0).to(tl.float32)

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Pairs where this row is first (i, j) with i < j
    for j in range(pid_n + 1, N):
        xj_offset = j * stride_x
        xj_vals = tl.load(x_ptr + xj_offset + m_offsets, mask=m_mask, other=0.0).to(tl.float32)

        diff = x_vals - xj_vals
        abs_diff = tl.abs(diff)
        sign = tl.where(diff > 0, 1.0, tl.where(diff < 0, -1.0, 0.0))

        pdist_idx = pid_n * N - (pid_n * (pid_n + 1)) // 2 + (j - pid_n - 1)
        g = tl.load(grad_ptr + pdist_idx).to(tl.float32)
        c = tl.load(pdist_ptr + pdist_idx).to(tl.float32)

        safe_c = tl.where(c == 0.0, 1.0, c)
        ratio = abs_diff / safe_c

        # Compute ratio^(p-1) using exp and log
        safe_ratio = tl.where(ratio == 0.0, 0.0, ratio)
        log_ratio = tl.where(safe_ratio > 0.0, tl.log(safe_ratio), 0.0)
        powered = tl.where(safe_ratio > 0.0, tl.exp((p_val - 1.0) * log_ratio), 0.0)

        contrib = tl.where(c == 0.0, 0.0, g * sign * powered)
        acc += contrib

    # Pairs where this row is second (j, i) with j < i
    for j in range(0, pid_n):
        xj_offset = j * stride_x
        xj_vals = tl.load(x_ptr + xj_offset + m_offsets, mask=m_mask, other=0.0).to(tl.float32)

        diff = x_vals - xj_vals
        abs_diff = tl.abs(diff)
        sign = tl.where(diff > 0, 1.0, tl.where(diff < 0, -1.0, 0.0))

        pdist_idx = j * N - (j * (j + 1)) // 2 + (pid_n - j - 1)
        g = tl.load(grad_ptr + pdist_idx).to(tl.float32)
        c = tl.load(pdist_ptr + pdist_idx).to(tl.float32)

        safe_c = tl.where(c == 0.0, 1.0, c)
        ratio = abs_diff / safe_c

        safe_ratio = tl.where(ratio == 0.0, 0.0, ratio)
        log_ratio = tl.where(safe_ratio > 0.0, tl.log(safe_ratio), 0.0)
        powered = tl.where(safe_ratio > 0.0, tl.exp((p_val - 1.0) * log_ratio), 0.0)

        contrib = tl.where(c == 0.0, 0.0, g * sign * powered)
        acc += contrib

    tl.store(out_ptr_row + m_offsets, acc.to(out_ptr.dtype.element_ty), mask=m_mask)


@libentry()
@triton.jit
def _pdist_backward_inf_kernel(
    grad_ptr,
    x_ptr,
    pdist_ptr,
    out_ptr,
    N,
    M,
    stride_x,
    stride_grad_out,
    BLOCK_M: tl.constexpr,
):
    """Backward kernel for p=inf case."""
    pid_n = tle.program_id(1)
    pid_m = tle.program_id(0)

    row_offset = pid_n * stride_x
    x_ptr_row = x_ptr + row_offset
    out_ptr_row = out_ptr + pid_n * stride_grad_out

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    x_vals = tl.load(x_ptr_row + m_offsets, mask=m_mask, other=0.0).to(tl.float32)

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Pairs where this row is first (i, j) with i < j
    for j in range(pid_n + 1, N):
        xj_offset = j * stride_x
        xj_vals = tl.load(x_ptr + xj_offset + m_offsets, mask=m_mask, other=0.0).to(tl.float32)

        diff = x_vals - xj_vals
        abs_diff = tl.abs(diff)

        pdist_idx = pid_n * N - (pid_n * (pid_n + 1)) // 2 + (j - pid_n - 1)
        g = tl.load(grad_ptr + pdist_idx).to(tl.float32)
        c = tl.load(pdist_ptr + pdist_idx).to(tl.float32)

        is_max = tl.where(abs_diff == c, 1.0, 0.0)
        sign = tl.where(diff > 0, 1.0, tl.where(diff < 0, -1.0, 0.0))

        acc += g * sign * is_max

    # Pairs where this row is second (j, i) with j < i
    for j in range(0, pid_n):
        xj_offset = j * stride_x
        xj_vals = tl.load(x_ptr + xj_offset + m_offsets, mask=m_mask, other=0.0).to(tl.float32)

        diff = x_vals - xj_vals
        abs_diff = tl.abs(diff)

        pdist_idx = j * N - (j * (j + 1)) // 2 + (pid_n - j - 1)
        g = tl.load(grad_ptr + pdist_idx).to(tl.float32)
        c = tl.load(pdist_ptr + pdist_idx).to(tl.float32)

        is_max = tl.where(abs_diff == c, 1.0, 0.0)
        sign = tl.where(diff > 0, 1.0, tl.where(diff < 0, -1.0, 0.0))

        acc += g * sign * is_max

    tl.store(out_ptr_row + m_offsets, acc.to(out_ptr.dtype.element_ty), mask=m_mask)


def _pdist_backward(grad, x, p, pdist):
    """Compute gradient of pdist forward pass."""
    logger.debug("GEMS _PDIST_BACKWARD")

    assert x.ndim == 2, "pdist only supports 2D input"
    N = x.shape[0]
    M = x.shape[1]

    grad = grad.contiguous()
    x = x.contiguous()
    pdist = pdist.contiguous()

    out = torch.empty_like(x)

    BLOCK_M = min(triton.next_power_of_2(M), 1024)
    grid = (triton.cdiv(M, BLOCK_M), N)

    with torch_device_fn.device(x.device):
        if p == 2.0:
            _pdist_backward_p2_kernel[grid](
                grad,
                x,
                pdist,
                out,
                N,
                M,
                x.stride(0),
                out.stride(0),
                BLOCK_M=BLOCK_M,
            )
        elif p == 1.0:
            _pdist_backward_p1_kernel[grid](
                grad,
                x,
                out,
                N,
                M,
                x.stride(0),
                out.stride(0),
                BLOCK_M=BLOCK_M,
            )
        elif math.isinf(p):
            _pdist_backward_inf_kernel[grid](
                grad,
                x,
                pdist,
                out,
                N,
                M,
                x.stride(0),
                out.stride(0),
                BLOCK_M=BLOCK_M,
            )
        else:
            _pdist_backward_general_kernel[grid](
                grad,
                x,
                pdist,
                out,
                N,
                M,
                p,
                x.stride(0),
                out.stride(0),
                BLOCK_M=BLOCK_M,
            )

    return out