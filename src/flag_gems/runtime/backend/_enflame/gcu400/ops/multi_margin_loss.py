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
"""GCU400-specific ``multi_margin_loss`` forward/backward implementation.

The shared ``_compute_forward``/``_compute_backward`` launch one kernel row at a time with
``grid=min(N, 65535)``, which on GCU400 (4 DIE x 6 SIP) creates thousands of programs for large
shapes, making scheduling and DTE startup far exceed the actual compute (bench measured ~600us
for (4096,1000) fp32, ~85us with a fixed row-block grid). This dispatches by shape:

- small shapes (``N * C <= ROWS_KERNEL_MIN_ELEMENTS``): keep the shared per-row kernel path
  (scheduling is a small share there, and a 2-D row-block tile degenerating to ``BLOCK_N=1``
  is actually slower);
- large shapes: fixed program count plus an in-kernel 2-D row-block stride (BLOCK_N rows x
  BLOCK_C-class tile persistent loop), launching a reduce kernel once when a reduction is needed.

The remaining input validation, target sync check, dtype policy and out semantics reuse the
shared implementation (same structure as ``multi_margin_loss_out.py``).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.multi_margin_loss import (
    _MAX_BLOCK_C,
    _MAX_GRID_SIZE,
    _REDUCE_BLOCK_SIZE,
    _USE_DEVICE_ASSERT,
    _check_grad_output,
    _check_inputs,
    _empty_forward,
    _multi_margin_loss_backward_kernel,
    _multi_margin_loss_kernel,
    _multi_margin_loss_reduce_kernel,
    _multi_margin_loss_rows_backward_kernel,
    _normalize_p,
    _normalize_reduction,
    _output_shape,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---


# Overwritten per round by the optimization-iteration driver script; all are compile-time/launch-time configs and do not
# change operator semantics.
FIXED_GRID = 48
NUM_WARPS = 4
BLOCK_C_CAP = 1024
REDUCE_BLOCK = 512
TILE_ELEMENTS = 4096
# Shapes below this element count use the common row-wise kernel (for small shapes the scheduling overhead of row-wise
# launches is a small fraction,
# and a 2D row-block tile degenerating to BLOCK_N=1 is actually slower), consistent with the pre-fix path.
ROWS_KERNEL_MIN_ELEMENTS = 2048
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


def _prev_pow2(n: int) -> int:
    return 1 << max(0, n.bit_length() - 1)


def _block_c(C: int) -> int:
    return min(_MAX_BLOCK_C, triton.next_power_of_2(C))


def _rows_config(N: int, C: int):
    block_c = min(BLOCK_C_CAP, triton.next_power_of_2(C))
    # BLOCK_N is used in an in-kernel tl.arange and must be a power of two; when the row count is less than one tile,
    # clamp the edge to the row count.
    block_n = max(1, min(TILE_ELEMENTS // block_c, _prev_pow2(max(1, N))))
    grid = min(triton.cdiv(N, block_n), FIXED_GRID)
    return block_n, block_c, grid


def _row_buffer_dtype(input):
    return torch.float64 if input.dtype == torch.float64 else torch.float32


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
    """Small shapes take the shared per-row kernel; large shapes launch the 2-D row-block kernel."""
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
    if needs_reduction:
        # Accumulate row-wise results in fp32 to avoid amplifying error in the second reduction under low precision.
        row_losses = torch.empty(
            (N,), dtype=_row_buffer_dtype(input), device=input.device
        )
    else:
        row_losses = output

    with torch_device_fn.device(input.device):
        if N * C <= ROWS_KERNEL_MIN_ELEMENTS:
            _multi_margin_loss_kernel[(min(N, _MAX_GRID_SIZE),)](
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
            )
        else:
            block_n, block_c, grid = _rows_config(N, C)
            _multi_margin_loss_rows_kernel[(grid,)](
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
                BLOCK_N=block_n,
                BLOCK_C=block_c,
                num_warps=NUM_WARPS,
            )
        if needs_reduction:
            _multi_margin_loss_reduce_kernel[(1,)](
                row_losses,
                output,
                N,
                REDUCTION=reduction,
                BLOCK_SIZE=REDUCE_BLOCK or _REDUCE_BLOCK_SIZE,
                num_warps=NUM_WARPS,
            )
    return output

    logger.debug("GEMS_ENFLAME MULTI_MARGIN_LOSS")
    logger.debug("GEMS_ENFLAME MULTI_MARGIN_LOSS_OUT")


def _compute_backward_gcu400(
    grad_output,
    input,
    target,
    weight,
    N,
    C,
    p,
    margin,
    reduction,
    grad_input=None,
):
    if grad_input is None:
        grad_input = torch.empty_like(input)
    if N == 0:
        return grad_input

    weight_ptr = input if weight is None else weight
    with torch_device_fn.device(input.device):
        if N * C <= ROWS_KERNEL_MIN_ELEMENTS:
            _multi_margin_loss_backward_kernel[(min(N, _MAX_GRID_SIZE),)](
                grad_output,
                input,
                target,
                weight_ptr,
                grad_input,
                N,
                C,
                margin,
                P=p,
                REDUCTION=reduction,
                HAS_WEIGHT=weight is not None,
                USE_DEVICE_ASSERT=_USE_DEVICE_ASSERT,
                BLOCK_C=_block_c(C),
            )
        else:
            block_n, block_c, grid = _rows_config(N, C)
            _multi_margin_loss_rows_backward_kernel[(grid,)](
                grad_output,
                input,
                target,
                weight_ptr,
                grad_input,
                N,
                C,
                margin,
                P=p,
                REDUCTION=reduction,
                HAS_WEIGHT=weight is not None,
                USE_DEVICE_ASSERT=_USE_DEVICE_ASSERT,
                BLOCK_N=block_n,
                BLOCK_C=block_c,
                num_warps=NUM_WARPS,
            )
    return grad_input


def multi_margin_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    p=1,
    margin=1,
    weight=None,
    reduction=1,
) -> torch.Tensor:
    p = _normalize_p(p)
    reduction = _normalize_reduction(reduction)
    try:
        margin = float(margin)
    except (TypeError, ValueError) as error:
        raise RuntimeError("multi_margin_loss: margin must be a real scalar") from error
    input, target, weight, N, C, is_batched = _check_inputs(input, target, weight)
    return _compute_forward_gcu400(
        input,
        target,
        weight,
        N,
        C,
        is_batched,
        p,
        margin,
        reduction,
        _output_shape(N, is_batched, reduction),
        None,
    )


def multi_margin_loss_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    target: torch.Tensor,
    p,
    margin,
    weight=None,
    reduction=1,
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME MULTI_MARGIN_LOSS_BACKWARD")
    p = _normalize_p(p)
    reduction = _normalize_reduction(reduction)
    try:
        margin = float(margin)
    except (TypeError, ValueError) as error:
        raise RuntimeError("multi_margin_loss: margin must be a real scalar") from error
    input, target, weight, N, C, is_batched = _check_inputs(input, target, weight)
    grad_output = _check_grad_output(grad_output, input, N, is_batched, reduction)
    return _compute_backward_gcu400(
        grad_output,
        input,
        target,
        weight,
        N,
        C,
        p,
        margin,
        reduction,
    )


def multi_margin_loss_backward_out(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    target: torch.Tensor,
    p,
    margin,
    weight=None,
    reduction=1,
    *,
    grad_input: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME MULTI_MARGIN_LOSS_BACKWARD_OUT")
    p = _normalize_p(p)
    reduction = _normalize_reduction(reduction)
    try:
        margin = float(margin)
    except (TypeError, ValueError) as error:
        raise RuntimeError("multi_margin_loss: margin must be a real scalar") from error
    input, target, weight, N, C, is_batched = _check_inputs(input, target, weight)
    grad_output = _check_grad_output(grad_output, input, N, is_batched, reduction)
    if grad_input.device != input.device:
        raise RuntimeError(
            "multi_margin_loss_backward.grad_input: output must be on the input device"
        )
    if grad_input.dtype != input.dtype:
        raise RuntimeError(
            "multi_margin_loss_backward.grad_input: output must have the input dtype"
        )
    if tuple(grad_input.shape) != tuple(input.shape):
        grad_input.resize_(input.shape)

    destination = grad_input if grad_input.is_contiguous() else None
    result = _compute_backward_gcu400(
        grad_output,
        input,
        target,
        weight,
        N,
        C,
        p,
        margin,
        reduction,
        grad_input=destination,
    )
    if result is not grad_input:
        grad_input.copy_(result)
    return grad_input


__all__ = [
    "multi_margin_loss",
    "multi_margin_loss_backward",
    "multi_margin_loss_backward_out",
]
