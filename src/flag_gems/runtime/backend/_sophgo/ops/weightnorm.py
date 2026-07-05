"""
Sophgo TPU-specific weightnorm operator implementation.

Fix notes:
Adapted for Sophgo TPU by:
1. Removed autotune, use fixed block sizes instead (Sophgo TPU does not support autotune).
2. Recompute addresses on each loop iteration, avoiding pointer updates that produce
   scf.for iter_args pattern which PPL ShapeInference cannot handle correctly.
3. Use 2D tensor mode for loading and storing.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# Fixed block sizes for Sophgo TPU (no autotune)
BLOCK_ROW_SIZE = 16
BLOCK_COL_SIZE = 32


@libentry()
@triton.jit(do_not_specialize=["eps"])
def weight_norm_kernel_last(
    output,
    norm,
    v,
    g,
    M,
    N,
    eps,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    tx = tl.arange(0, BLOCK_COL_SIZE)[:, None]
    bx = tle.program_id(axis=0) * BLOCK_COL_SIZE
    col_offset = bx + tx
    col_mask = col_offset < N

    ty = tl.arange(0, BLOCK_ROW_SIZE)[None, :]
    v_block = tl.zeros([BLOCK_COL_SIZE, BLOCK_ROW_SIZE], dtype=tl.float32)
    for base in range(0, M, BLOCK_ROW_SIZE):
        # Recompute offsets each iteration (Sophgo TPU compatible)
        row_offset = base + ty
        mask = row_offset < M and col_mask
        v_value = tl.load(v + row_offset * N + col_offset, mask=mask).to(tl.float32)
        v_block += v_value * v_value

    # Use rsqrt (hardware SFU) instead of sqrt + div.
    normalized = tl.rsqrt(tl.sum(v_block, axis=1) + eps)
    norm_val = 1.0 / normalized
    tl.store(norm + col_offset, norm_val[:, None], mask=col_mask)
    g_value = tl.load(g + col_offset, mask=col_mask).to(tl.float32)

    for base in range(0, M, BLOCK_ROW_SIZE):
        # Recompute offsets each iteration (Sophgo TPU compatible)
        row_offset = base + ty
        mask = row_offset < M and col_mask
        v_value = tl.load(v + row_offset * N + col_offset, mask=mask).to(tl.float32)
        v_vec = v_value * normalized[:, None]
        out = v_vec * g_value
        tl.store(output + row_offset * N + col_offset, out, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def weight_norm_kernel_first(
    output,
    norm,
    v,
    g,
    M,
    N,
    eps,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    ty = tl.arange(0, BLOCK_ROW_SIZE)[:, None]
    by = tle.program_id(axis=0) * BLOCK_ROW_SIZE
    row_offset = by + ty
    row_mask = row_offset < M

    tx = tl.arange(0, BLOCK_COL_SIZE)[None, :]
    v_block = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    for base in range(0, N, BLOCK_COL_SIZE):
        # Recompute offsets each iteration (Sophgo TPU compatible)
        col_offset = base + tx
        mask = col_offset < N and row_mask
        v_value = tl.load(v + row_offset * N + col_offset, mask=mask).to(tl.float32)
        v_block += v_value * v_value

    # Use rsqrt (hardware SFU) instead of sqrt + div.
    normalized = tl.rsqrt(tl.sum(v_block, axis=1) + eps)
    norm_val = 1.0 / normalized
    tl.store(norm + row_offset, norm_val[:, None], mask=row_mask)
    g_value = tl.load(g + row_offset, mask=row_mask).to(tl.float32)

    for base in range(0, N, BLOCK_COL_SIZE):
        # Recompute offsets each iteration (Sophgo TPU compatible)
        col_offset = base + tx
        mask = col_offset < N and row_mask
        v_value = tl.load(v + row_offset * N + col_offset, mask=mask).to(tl.float32)
        v_vec = v_value * normalized[:, None]
        out = v_vec * g_value
        tl.store(output + row_offset * N + col_offset, out, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def weight_norm_bwd_kernel_last(
    v_grad,
    g_grad,
    w,
    v,
    g,
    norm,
    M,
    N,
    eps,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    tx = tl.arange(0, BLOCK_COL_SIZE)[:, None]
    bx = tle.program_id(axis=0) * BLOCK_COL_SIZE
    col_offset = tx + bx
    col_mask = col_offset < N

    g_value = tl.load(g + col_offset, mask=col_mask).to(tl.float32)
    norm_value = tl.load(norm + col_offset, mask=col_mask).to(tl.float32)

    ty = tl.arange(0, BLOCK_ROW_SIZE)[None, :]

    vw_block = tl.zeros([BLOCK_COL_SIZE, BLOCK_ROW_SIZE], dtype=tl.float32)
    for base in range(0, M, BLOCK_ROW_SIZE):
        # Recompute offsets each iteration (Sophgo TPU compatible)
        row_offset = base + ty
        mask = row_offset < M and col_mask
        v_value = tl.load(v + row_offset * N + col_offset, mask=mask).to(tl.float32)
        w_value = tl.load(w + row_offset * N + col_offset, mask=mask).to(tl.float32)
        vw_block += v_value * w_value
    vw_sum = tl.sum(vw_block, 1)[:, None]

    for base in range(0, M, BLOCK_ROW_SIZE):
        # Recompute offsets each iteration (Sophgo TPU compatible)
        row_offset = base + ty
        mask = row_offset < M and col_mask
        v_value = tl.load(v + row_offset * N + col_offset, mask=mask).to(tl.float32)
        w_value = tl.load(w + row_offset * N + col_offset, mask=mask).to(tl.float32)
        v_grad_value = g_value * (
            w_value / (norm_value + eps)
            - v_value / (norm_value * norm_value * norm_value + eps) * vw_sum
        )
        tl.store(v_grad + row_offset * N + col_offset, v_grad_value, mask=mask)

    g_grad_value = vw_sum / (norm_value + eps)
    tl.store(g_grad + col_offset, g_grad_value, mask=col_mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def weight_norm_bwd_kernel_first(
    v_grad,
    g_grad,
    w,
    v,
    g,
    norm,
    M,
    N,
    eps,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    ty = tl.arange(0, BLOCK_ROW_SIZE)[:, None]
    by = tle.program_id(axis=0) * BLOCK_ROW_SIZE
    row_offset = by + ty
    row_mask = row_offset < M

    g_value = tl.load(g + row_offset, mask=row_mask).to(tl.float32)
    norm_value = tl.load(norm + row_offset, mask=row_mask).to(tl.float32)

    tx = tl.arange(0, BLOCK_COL_SIZE)[None, :]

    v_block = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    for base in range(0, N, BLOCK_COL_SIZE):
        # Recompute offsets each iteration (Sophgo TPU compatible)
        col_offset = base + tx
        mask = col_offset < N and row_mask
        v_value = tl.load(v + row_offset * N + col_offset, mask=mask).to(tl.float32)
        w_value = tl.load(w + row_offset * N + col_offset, mask=mask).to(tl.float32)
        v_block += v_value * w_value
    vw_sum = tl.sum(v_block, 1)[:, None]

    for base in range(0, N, BLOCK_COL_SIZE):
        # Recompute offsets each iteration (Sophgo TPU compatible)
        col_offset = base + tx
        mask = col_offset < N and row_mask
        v_value = tl.load(v + row_offset * N + col_offset, mask=mask).to(tl.float32)
        w_value = tl.load(w + row_offset * N + col_offset, mask=mask).to(tl.float32)
        v_grad_value = g_value * (
            w_value / (norm_value + eps)
            - v_value / (norm_value * norm_value * norm_value + eps) * vw_sum
        )
        tl.store(v_grad + row_offset * N + col_offset, v_grad_value, mask=mask)

    g_grad_value = vw_sum / (norm_value + eps)
    tl.store(g_grad + row_offset, g_grad_value, mask=row_mask)


def weight_norm_interface(v, g, dim=0):
    """
    Sophgo TPU-specific weight_norm forward implementation.

    Args:
        v: Weight tensor.
        g: Gain tensor.
        dim: Dimension along which to compute weight norm.

    Returns:
        Tuple of (output, norm) tensors.
    """
    logging.debug("GEMS WEIGHT NORM INTERFACE FORWARD (Sophgo TPU)")
    v = v.contiguous()
    g = g.contiguous()
    output = torch.empty_like(v)
    norm = torch.empty_like(
        g, dtype=torch.float32
    )  # fp32: avoid precision loss between fwd sqrt and bwd load
    if dim == 0:
        M = v.shape[0]
        N = math.prod(v.shape[1:])
        grid = (triton.cdiv(M, BLOCK_ROW_SIZE),)
        with torch_device_fn.device(v.device):
            weight_norm_kernel_first[grid](
                output,
                norm,
                v,
                g,
                M,
                N,
                eps=torch.finfo(torch.float32).tiny,
                BLOCK_ROW_SIZE=BLOCK_ROW_SIZE,
                BLOCK_COL_SIZE=BLOCK_COL_SIZE,
            )
    elif dim == v.ndim - 1:
        M = math.prod(v.shape[:-1])
        N = v.shape[dim]
        grid = (triton.cdiv(N, BLOCK_COL_SIZE),)
        with torch_device_fn.device(v.device):
            weight_norm_kernel_last[grid](
                output,
                norm,
                v,
                g,
                M,
                N,
                eps=torch.finfo(torch.float32).tiny,
                BLOCK_ROW_SIZE=BLOCK_ROW_SIZE,
                BLOCK_COL_SIZE=BLOCK_COL_SIZE,
            )
    return output, norm


def weight_norm_interface_backward(w_grad, saved_v, saved_g, saved_norms, dim):
    """
    Sophgo TPU-specific weight_norm backward implementation.

    Args:
        w_grad: Gradient of the output.
        saved_v: Saved v tensor from forward.
        saved_g: Saved g tensor from forward.
        saved_norms: Saved norms from forward.
        dim: Dimension along which weight norm was computed.

    Returns:
        Tuple of (v_grad, g_grad) tensors.
    """
    logging.debug("GEMS WEIGHT NORM INTERFACE BACKWARD (Sophgo TPU)")
    w_grad = w_grad.contiguous()
    saved_v = saved_v.contiguous()
    saved_g = saved_g.contiguous()
    saved_norms = saved_norms.contiguous()
    v_grad = torch.empty_like(saved_v)
    g_grad = torch.empty_like(saved_g)

    if dim == 0:
        M = saved_v.shape[0]
        N = math.prod(saved_v.shape[1:])
        grid = (triton.cdiv(M, BLOCK_ROW_SIZE),)
        with torch_device_fn.device(saved_v.device):
            weight_norm_bwd_kernel_first[grid](
                v_grad,
                g_grad,
                w_grad,
                saved_v,
                saved_g,
                saved_norms,
                M,
                N,
                eps=torch.finfo(torch.float32).tiny,
                BLOCK_ROW_SIZE=BLOCK_ROW_SIZE,
                BLOCK_COL_SIZE=BLOCK_COL_SIZE,
            )
    elif dim == saved_v.ndim - 1:
        M = math.prod(saved_v.shape[:dim])
        N = saved_v.shape[dim]
        grid = (triton.cdiv(N, BLOCK_COL_SIZE),)
        with torch_device_fn.device(saved_v.device):
            weight_norm_bwd_kernel_last[grid](
                v_grad,
                g_grad,
                w_grad,
                saved_v,
                saved_g,
                saved_norms,
                M,
                N,
                eps=torch.finfo(torch.float32).tiny,
                BLOCK_ROW_SIZE=BLOCK_ROW_SIZE,
                BLOCK_COL_SIZE=BLOCK_COL_SIZE,
            )
    return v_grad, g_grad
