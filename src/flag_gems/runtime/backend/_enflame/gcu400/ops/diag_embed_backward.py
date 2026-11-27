# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""GCU400-specific ``diag_embed_backward``.

The shared implementation is a flat gather that reads the ``(offset, dim1, dim2)`` diagonal of
``grad_output``. Every lane re-derives the source address from three device-resident metadata
vectors, so each element pays a chain of integer divisions against *runtime* divisors plus three
global loads per view axis, and the host pays three tiny ``torch.tensor`` uploads per call. On
GCU400 that index arithmetic, not the one-element-per-address gather, sets the latency floor.

Two changes remove it:

* The metadata is compile-time constant: view sizes and strides are ``constexpr`` scalars, so the
  per-lane ``%`` / ``//`` chains fold into constant multiply-shift sequences and the three uploads
  disappear.
* When ``BLOCK`` divides ``diag_len`` (the usual case) a whole block sits inside one view row, so
  the view index is decomposed once per program as *scalar* work; only shapes whose ``diag_len``
  is not a multiple of the block fall back to per-lane decomposition.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# Generated per optimization round; all compile-time / launch-time config, semantics unchanged.
LANE_BYTES = 4096
MIN_BLOCK = 64
MAX_BLOCK = 4096
NUM_WARPS = 1
NUM_STAGES = 1
# Row packing folds several whole view rows into one program block, which is only possible while
# the diagonal length is a power of two (the block width stays a power of two as well).
PACK_ROWS = 1
# --- TUNING PARAMS END ---

# Launch modes: one view row per block, several packed rows per block, or the masked flat path
# used when the diagonal length is not a power of two. The kernel only sees the numbers, since
# Triton kernels cannot read plain module globals.
MODE_SINGLE_ROW = tl.constexpr(0)
MODE_PACKED_ROWS = tl.constexpr(1)
MODE_GENERAL = tl.constexpr(2)

# A tensor ranks up to 8, so the diagonal view keeps at most 6 axes. Padding a shorter rank with
# size 1 / stride 0 keeps the decomposition branch-free.
_MAX_VIEW = 6


@libentry()
@triton.jit
def _diag_embed_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    DIAG_LEN: tl.constexpr,
    DIAG_STRIDE: tl.constexpr,
    DIAG_BASE: tl.constexpr,
    N_VIEW: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    S4: tl.constexpr,
    S5: tl.constexpr,
    T0: tl.constexpr,
    T1: tl.constexpr,
    T2: tl.constexpr,
    T3: tl.constexpr,
    T4: tl.constexpr,
    T5: tl.constexpr,
    N_TOTAL: tl.constexpr,
    MODE: tl.constexpr,
    WIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gather the diagonal into the contiguous gradient, one flat block per program.

    ``MODE`` picks how the block maps onto the view rows: one row per block, several whole rows
    per block (``DIAG_LEN`` a power of two), or the masked flat path.
    """
    pid = tl.program_id(axis=0)
    lane = tl.arange(0, BLOCK)

    if MODE == MODE_SINGLE_ROW:
        blocks_per_row: tl.constexpr = DIAG_LEN // BLOCK
        row = pid // blocks_per_row
        diagonal_index = (pid % blocks_per_row) * BLOCK + lane
        flat_index = row * DIAG_LEN + diagonal_index
    elif MODE == MODE_PACKED_ROWS:
        rows_per_block: tl.constexpr = BLOCK // DIAG_LEN
        # ``DIAG_LEN`` is a power of two here, so both ops are shifts.
        row = pid * rows_per_block + lane // DIAG_LEN
        diagonal_index = lane % DIAG_LEN
        flat_index = pid * BLOCK + lane
    else:
        flat_index = pid * BLOCK + lane
        valid = flat_index < N_TOTAL
        row = flat_index // DIAG_LEN
        diagonal_index = flat_index % DIAG_LEN

    if N_VIEW > 0:
        # View-row -> grad_output offset. The view index is row-major, so the decomposition runs
        # from the last axis backwards; padding (size 1, stride 0) contributes nothing, so all six
        # axes are unrolled unconditionally.
        remaining = row
        coord_5 = remaining % S5
        remaining = remaining // S5
        coord_4 = remaining % S4
        remaining = remaining // S4
        coord_3 = remaining % S3
        remaining = remaining // S3
        coord_2 = remaining % S2
        remaining = remaining // S2
        coord_1 = remaining % S1
        remaining = remaining // S1
        coord_0 = remaining % S0
        row_base = (
            coord_0 * T0
            + coord_1 * T1
            + coord_2 * T2
            + coord_3 * T3
            + coord_4 * T4
            + coord_5 * T5
        )
    else:
        row_base = tl.zeros_like(row)

    if WIDE:
        source = (
            row_base.to(tl.int64)
            + diagonal_index.to(tl.int64) * DIAG_STRIDE
            + DIAG_BASE
        )
        destination = flat_index.to(tl.int64)
    else:
        source = row_base + diagonal_index * DIAG_STRIDE + DIAG_BASE
        destination = flat_index

    if MODE == MODE_GENERAL:
        value = tl.load(grad_output_ptr + source, mask=valid, other=0)
        tl.store(grad_input_ptr + destination, value, mask=valid)
    else:
        # Both row modes tile the output exactly, so every lane is a valid gather.
        value = tl.load(grad_output_ptr + source)
        tl.store(grad_input_ptr + destination, value)


def _highest_power_of_two_divisor(value):
    divisor = 1
    while value % (divisor * 2) == 0:
        divisor *= 2
    return divisor


def _choose_block(diag_len, element_size):
    """Pick the block of lanes and the launch mode that folds the diagonal onto blocks.

    * A power-of-two ``diag_len`` lets whole view rows be packed into one program, up to the lane
      budget: few rows per wave costs a per-lane shift to recover the row number, but cuts the
      program count (and so the number of waves) by the same factor.
    * A diagonal carrying a small odd factor cannot be packed; it keeps the widest dividing block
      when one exists, otherwise the masked flat path (a 1-lane block would idle the machine).
    """
    lane_target = max(MIN_BLOCK, LANE_BYTES // max(element_size, 1))
    budget = min(MAX_BLOCK, triton.next_power_of_2(lane_target))
    divisor = _highest_power_of_two_divisor(diag_len)
    if divisor == diag_len:
        # Power-of-two diagonal: pack as many whole rows as the budget allows.
        if PACK_ROWS and budget > diag_len:
            return budget - budget % diag_len, MODE_PACKED_ROWS
        return min(budget, diag_len), MODE_SINGLE_ROW
    if divisor >= MIN_BLOCK:
        return min(divisor, MAX_BLOCK, budget), MODE_SINGLE_ROW
    return budget, MODE_GENERAL


def _diag_embed_backward_impl(
    grad_output: torch.Tensor, x: torch.Tensor, offset: int, dim1: int, dim2: int
):
    row_off = max(-offset, 0)
    col_off = max(offset, 0)
    rank = grad_output.ndim
    shape = list(grad_output.shape)
    strides = [1] * rank
    for axis in range(rank - 2, -1, -1):
        strides[axis] = strides[axis + 1] * shape[axis + 1]

    diag_len = min(shape[dim1] + min(offset, 0), shape[dim2] - max(offset, 0))
    if diag_len <= 0:
        return torch.empty_like(x)

    view_sizes = [shape[axis] for axis in range(rank) if axis != dim1 and axis != dim2]
    src_strides = [
        strides[axis] for axis in range(rank) if axis != dim1 and axis != dim2
    ]
    n_view = len(view_sizes)
    n_view_total = 1
    for size in view_sizes:
        n_view_total *= size
    n_total = n_view_total * diag_len

    # Fixed six-axis decomposition: size 1 needs no divide, stride 0 adds nothing.
    sizes = view_sizes + [1] * (_MAX_VIEW - n_view)
    srcs = src_strides + [0] * (_MAX_VIEW - n_view)

    # The diagonal advances dim1 and dim2 together, so its step and start are both constants.
    diag_stride = strides[dim1] + strides[dim2]
    diag_base = row_off * strides[dim1] + col_off * strides[dim2]
    largest_offset = (n_view_total - 1) * max(srcs) + diag_len * diag_stride + diag_base
    # The 32-bit path is only safe while the *byte* offset stays inside int32;
    # an element-offset check misses buffers above 2 GiB (e.g. 1025x1025x1024
    # fp16), whose byte addressing overflows and faults with a device Sip error.
    element_size = grad_output.element_size()
    wide = largest_offset * element_size >= 2**31 or n_total >= 2**31

    grad_input = torch.empty(
        view_sizes + [diag_len], dtype=grad_output.dtype, device=grad_output.device
    )

    block, mode = _choose_block(diag_len, grad_output.element_size())
    grid = (triton.cdiv(n_total, block),)
    with torch_device_fn.device(grad_output.device):
        _diag_embed_backward_kernel[grid](
            grad_output,
            grad_input,
            DIAG_LEN=diag_len,
            DIAG_STRIDE=diag_stride,
            DIAG_BASE=diag_base,
            N_VIEW=n_view,
            S0=sizes[0],
            S1=sizes[1],
            S2=sizes[2],
            S3=sizes[3],
            S4=sizes[4],
            S5=sizes[5],
            T0=srcs[0],
            T1=srcs[1],
            T2=srcs[2],
            T3=srcs[3],
            T4=srcs[4],
            T5=srcs[5],
            N_TOTAL=n_total,
            MODE=mode,
            WIDE=wide,
            BLOCK=block,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
    return grad_input


def diag_embed_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    offset: int = 0,
    dim1: int = -2,
    dim2: int = -1,
):
    """Gradient of ``torch.diag_embed``: the diagonal of the upstream gradient."""
    logger.debug("GEMS_GCU400 DIAG_EMBED_BACKWARD")

    rank = x.ndim + 1
    a = dim1 % rank
    b = dim2 % rank

    if grad_output.numel() == 0:
        return torch.empty_like(x)

    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()

    return _diag_embed_backward_impl(grad_output, x, offset, a, b)


class DiagEmbedFunction(torch.autograd.Function):
    """autograd bridge so ``diag_embed`` differentiates through the Triton gather kernel."""

    @staticmethod
    def forward(ctx, x, offset, dim1, dim2):
        from .diag_embed import diag_embed

        ctx.offset = offset
        ctx.dim1 = dim1
        ctx.dim2 = dim2
        ctx.save_for_backward(x)
        return diag_embed(x, offset, dim1, dim2)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None
        return (
            diag_embed_backward(grad_output, x, ctx.offset, ctx.dim1, ctx.dim2),
            None,
            None,
            None,
        )
