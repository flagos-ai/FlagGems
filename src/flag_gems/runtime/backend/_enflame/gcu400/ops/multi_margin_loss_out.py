# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GCU400-specific ``multi_margin_loss.out``.

The common ``_compute_forward`` launches a row kernel with ``grid=min(N, 65535)``
(one row per program), which on GCU400 (4 DIE x 6 SIP) pays scheduling and DTE
launch overhead far beyond the actual compute. Route by problem size:

- small row-loss shapes (``N * C <= SMALL_MIN_ELEMENTS``): keep the common
  per-row kernel path (launch-bound; identical to the delivered baseline);
- large row-loss shapes (``reduction == 0``): a 2D rows kernel processes
  ``BLOCK_N x BLOCK_C`` tiles with a fixed grid plus in-kernel grid stride;
- reduced shapes (``reduction != 0``, batched): small problems keep the
  common two-step structure (per-row kernel into an fp32 row-loss buffer,
  then one reduce launch); larger shapes run the rows kernel in partial
  mode, where each program folds its rows into one fp32 partial, and the
  reduce kernel finishes the sum over at most ``FIXED_GRID`` partials
  instead of ``N`` row losses. Tiny 2D tiles carry a large scheduling
  penalty on GCU400, so the partial route is only used once the row count
  can fill ``BLOCK_N >= 4`` tiles beyond one row per program.

All other input validation, dtype policy and out semantics match the common
implementation; per-row accumulation stays fp32 to avoid error amplification.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.multi_margin_loss import (
    _USE_DEVICE_ASSERT,
    _check_inputs,
    _empty_forward,
    _multi_margin_loss_kernel,
    _normalize_p,
    _normalize_reduction,
    _output_shape,
)
from flag_gems.ops.multi_margin_loss import (
    multi_margin_loss_out as _public_multi_margin_loss_out,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---


# Overwritten per round by the optimization driver script; all compile-time /
# launch-time config, semantics unchanged.
FIXED_GRID = 48
NUM_WARPS = 4
REDUCE_WARPS = 1
BLOCK_C_CAP = 512
TILE_ELEMENTS = 16384
# Below this element count the row-loss path keeps the common per-row kernel
# (launch-bound small shapes; the delivered baseline behavior).
SMALL_MIN_ELEMENTS = 2048
FALLBACK_TO_PUBLIC = 0
# --- TUNING PARAMS END ---


@libentry()
@triton.jit(
    debug=True,
    do_not_specialize=["N", "C", "margin"],
)
def _multi_margin_loss_rows_kernel(
    input_ptr,
    target_ptr,
    weight_ptr,
    output_ptr,
    N,
    C,
    margin,
    P: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    USE_DEVICE_ASSERT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Compute per-row losses into ``output_ptr`` with a fixed grid stride."""
    acc_dtype = tl.float64 if input_ptr.type.element_ty == tl.float64 else tl.float32
    margin = margin.to(acc_dtype)
    pid = tl.program_id(0)
    row_stride = tl.num_programs(0) * BLOCK_N

    for row_start in range(pid * BLOCK_N, N, row_stride):
        rows = row_start + tl.arange(0, BLOCK_N)[:, None]
        row_mask = rows < N
        targets = tl.load(target_ptr + rows, mask=row_mask, other=0)
        valid_target = (targets >= 0) & (targets < C)
        if USE_DEVICE_ASSERT:
            invalid_count = tl.sum((row_mask & ~valid_target).to(tl.int32), axis=0)
            invalid_count = tl.sum(invalid_count, axis=0)
            tl.device_assert(
                invalid_count == 0,
                "multi_margin_loss: target index is out of bounds",
            )
        safe_targets = tl.where(valid_target, targets, 0)
        target_values = tl.load(
            input_ptr + rows * C + safe_targets,
            mask=row_mask & valid_target,
            other=0.0,
        ).to(acc_dtype)

        loss_sums = tl.zeros((BLOCK_N,), dtype=acc_dtype)
        for class_start in range(0, C, BLOCK_C):
            classes = class_start + tl.arange(0, BLOCK_C)[None, :]
            class_mask = classes < C
            values = tl.load(
                input_ptr + rows * C + classes,
                mask=row_mask & class_mask,
                other=0.0,
            ).to(acc_dtype)
            z = margin - target_values + values
            active = (
                row_mask
                & class_mask
                & valid_target
                & (classes != safe_targets)
                & (z > 0)
            )
            term = tl.where(active, z, 0.0)
            if P == 2:
                term = term * term
            loss_sums += tl.sum(term, axis=1)

        if HAS_WEIGHT:
            target_weights = tl.load(
                weight_ptr + safe_targets,
                mask=row_mask & valid_target,
                other=0.0,
            ).to(acc_dtype)
        else:
            target_weights = tl.full((BLOCK_N, 1), 1.0, acc_dtype)

        losses = target_weights * loss_sums[:, None] / C.to(acc_dtype)
        tl.store(output_ptr + rows, losses, mask=row_mask)


@libentry()
@triton.jit(
    debug=True,
    do_not_specialize=["N", "C", "margin"],
)
def _multi_margin_loss_rows_partial_kernel(
    input_ptr,
    target_ptr,
    weight_ptr,
    partial_ptr,
    N,
    C,
    margin,
    P: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    USE_DEVICE_ASSERT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Fold every row handled by one program into a single fp32 partial."""
    acc_dtype = tl.float64 if input_ptr.type.element_ty == tl.float64 else tl.float32
    margin = margin.to(acc_dtype)
    pid = tl.program_id(0)
    row_stride = tl.num_programs(0) * BLOCK_N
    # Accumulate per-row-slot partials across tiles as a vector and reduce
    # once after the loop; masked rows already contribute exactly zero.
    partial_vec = tl.zeros((BLOCK_N,), dtype=acc_dtype)

    for row_start in range(pid * BLOCK_N, N, row_stride):
        rows = row_start + tl.arange(0, BLOCK_N)[:, None]
        row_mask = rows < N
        targets = tl.load(target_ptr + rows, mask=row_mask, other=0)
        valid_target = (targets >= 0) & (targets < C)
        if USE_DEVICE_ASSERT:
            invalid_count = tl.sum((row_mask & ~valid_target).to(tl.int32), axis=0)
            invalid_count = tl.sum(invalid_count, axis=0)
            tl.device_assert(
                invalid_count == 0,
                "multi_margin_loss: target index is out of bounds",
            )
        safe_targets = tl.where(valid_target, targets, 0)
        target_values = tl.load(
            input_ptr + rows * C + safe_targets,
            mask=row_mask & valid_target,
            other=0.0,
        ).to(acc_dtype)

        loss_sums = tl.zeros((BLOCK_N,), dtype=acc_dtype)
        for class_start in range(0, C, BLOCK_C):
            classes = class_start + tl.arange(0, BLOCK_C)[None, :]
            class_mask = classes < C
            values = tl.load(
                input_ptr + rows * C + classes,
                mask=row_mask & class_mask,
                other=0.0,
            ).to(acc_dtype)
            z = margin - target_values + values
            active = (
                row_mask
                & class_mask
                & valid_target
                & (classes != safe_targets)
                & (z > 0)
            )
            term = tl.where(active, z, 0.0)
            if P == 2:
                term = term * term
            loss_sums += tl.sum(term, axis=1)

        if HAS_WEIGHT:
            target_weights = tl.load(
                weight_ptr + safe_targets,
                mask=row_mask & valid_target,
                other=0.0,
            ).to(acc_dtype)
        else:
            target_weights = tl.full((BLOCK_N, 1), 1.0, acc_dtype)

        losses = target_weights * loss_sums[:, None] / C.to(acc_dtype)
        partial_vec += tl.sum(losses, axis=1)

    tl.store(partial_ptr + pid, tl.sum(partial_vec, axis=0))


@libentry()
@triton.jit(do_not_specialize=["N"])
def _multi_margin_loss_partials_reduce_kernel(
    partial_ptr,
    output_ptr,
    partial_count,
    N,
    REDUCTION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    acc_dtype = tl.float64 if partial_ptr.type.element_ty == tl.float64 else tl.float32
    total = tl.zeros((), dtype=acc_dtype)
    for start in range(0, partial_count, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < partial_count
        values = tl.load(partial_ptr + offsets, mask=mask, other=0.0).to(acc_dtype)
        total += tl.sum(tl.where(mask, values, 0.0), axis=0)
    if REDUCTION == 1:
        total /= N.to(acc_dtype)
    tl.store(output_ptr, total)


def _prev_pow2(n: int) -> int:
    return 1 << max(0, n.bit_length() - 1)


def _block_c(C: int) -> int:
    return min(BLOCK_C_CAP, triton.next_power_of_2(C))


def _rows_config(N: int, C: int):
    block_c = _block_c(C)
    # BLOCK_N participates in tl.arange, so it must be a power of two. Bound
    # the tile element budget, never exceed the row count, and keep the grid
    # near FIXED_GRID so every program covers a whole number of tiles. Tiny
    # 2D tiles carry a heavy scheduling penalty on GCU400, so never shrink
    # below four rows per tile while the row count allows it.
    block_n = max(
        1,
        min(
            TILE_ELEMENTS // block_c,
            _prev_pow2(max(1, N)),
            triton.next_power_of_2(triton.cdiv(N, FIXED_GRID)),
        ),
    )
    if N >= 4:
        block_n = max(block_n, 4)
    grid = min(triton.cdiv(N, block_n), FIXED_GRID)
    return block_n, block_c, grid


def _compute_forward_gcu400(
    input,
    target,
    weight,
    N,
    C,
    is_batched,
    p,
    margin,
    reduction,
    output_shape,
    destination,
):
    """Route by size: common per-row kernel, 2D rows kernel, or two-step reduce."""
    if N == 0:
        return _empty_forward(input, N, is_batched, reduction)

    output = (
        destination
        if destination is not None
        else torch.empty(output_shape, dtype=input.dtype, device=input.device)
    )
    weight_ptr = input if weight is None else weight
    has_weight = weight is not None

    needs_reduction = is_batched and N > 1 and reduction != 0
    buffer_dtype = torch.float64 if input.dtype == torch.float64 else torch.float32
    with torch_device_fn.device(input.device):
        if not needs_reduction:
            if N * C <= SMALL_MIN_ELEMENTS:
                _multi_margin_loss_kernel[(min(N, FIXED_GRID),)](
                    input,
                    target,
                    weight_ptr,
                    output,
                    N,
                    C,
                    margin,
                    P=p,
                    HAS_WEIGHT=has_weight,
                    USE_DEVICE_ASSERT=_USE_DEVICE_ASSERT,
                    BLOCK_C=_block_c(C),
                    num_warps=NUM_WARPS,
                )
            else:
                block_n, block_c, grid = _rows_config(N, C)
                _multi_margin_loss_rows_kernel[(grid,)](
                    input,
                    target,
                    weight_ptr,
                    output,
                    N,
                    C,
                    margin,
                    P=p,
                    HAS_WEIGHT=has_weight,
                    USE_DEVICE_ASSERT=_USE_DEVICE_ASSERT,
                    BLOCK_N=block_n,
                    BLOCK_C=block_c,
                    num_warps=NUM_WARPS,
                )
        elif N * C <= SMALL_MIN_ELEMENTS or N <= FIXED_GRID:
            # Small problems keep the common two-step structure: one row per
            # program into an fp32 row-loss buffer, then a single reduce.
            row_losses = torch.empty((N,), dtype=buffer_dtype, device=input.device)
            _multi_margin_loss_kernel[(min(N, FIXED_GRID),)](
                input,
                target,
                weight_ptr,
                row_losses,
                N,
                C,
                margin,
                P=p,
                HAS_WEIGHT=has_weight,
                USE_DEVICE_ASSERT=_USE_DEVICE_ASSERT,
                BLOCK_C=_block_c(C),
                num_warps=NUM_WARPS,
            )
            _multi_margin_loss_partials_reduce_kernel[(1,)](
                row_losses,
                output,
                N,
                N,
                REDUCTION=reduction,
                BLOCK_SIZE=triton.next_power_of_2(max(min(N, 512), 16)),
                num_warps=REDUCE_WARPS,
            )
        else:
            block_n, block_c, grid = _rows_config(N, C)
            partials = torch.empty((grid,), dtype=buffer_dtype, device=input.device)
            _multi_margin_loss_rows_partial_kernel[(grid,)](
                input,
                target,
                weight_ptr,
                partials,
                N,
                C,
                margin,
                P=p,
                HAS_WEIGHT=has_weight,
                USE_DEVICE_ASSERT=_USE_DEVICE_ASSERT,
                BLOCK_N=block_n,
                BLOCK_C=block_c,
                num_warps=NUM_WARPS,
            )
            _multi_margin_loss_partials_reduce_kernel[(1,)](
                partials,
                output,
                grid,
                N,
                REDUCTION=reduction,
                BLOCK_SIZE=triton.next_power_of_2(max(grid, 16)),
                num_warps=REDUCE_WARPS,
            )
    return output


def multi_margin_loss_out(
    input: torch.Tensor,
    target: torch.Tensor,
    p=1,
    margin=1,
    weight=None,
    reduction=1,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS_GCU400 MULTI_MARGIN_LOSS_OUT")
    if FALLBACK_TO_PUBLIC:
        return _public_multi_margin_loss_out(
            input, target, p, margin, weight, reduction, out=out
        )
    p = _normalize_p(p)
    reduction = _normalize_reduction(reduction)
    try:
        margin = float(margin)
    except (TypeError, ValueError) as error:
        raise RuntimeError("multi_margin_loss: margin must be a real scalar") from error
    input, target, weight, N, C, is_batched = _check_inputs(input, target, weight)
    if out.device != input.device:
        raise RuntimeError("multi_margin_loss.out: out must be on the input device")
    if out.dtype != input.dtype:
        raise RuntimeError("multi_margin_loss.out: out must have the input dtype")

    shape = _output_shape(N, is_batched, reduction)
    if tuple(out.shape) != shape:
        out.resize_(shape)
    destination = out if out.is_contiguous() else None
    result = _compute_forward_gcu400(
        input,
        target,
        weight,
        N,
        C,
        is_batched,
        p,
        margin,
        reduction,
        shape,
        destination,
    )
    if result is not out:
        out.copy_(result)
    return out
