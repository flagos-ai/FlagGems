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

"""Kunlunxin XPU3 override for ``multi_margin_loss`` and its backward.

The generic kernels accumulate a row inside a loop that also carries the
out-of-range target assertion, and this backend cannot legalize the resulting
``tt.reduce``: the module fails to compile for every dtype and shape, so the
generic file cannot serve this backend at all.

The override keeps the row-wise algorithm -- the class axis is visited in tiles
and the row total is accumulated in a scalar -- but gives every row its own
program instead of looping over rows inside one program, which is the shape this
backend lowers.  The class tile is reduced with a plain ``tl.sum`` over the tile,
and the row total stays a scalar, so no reduction ever spans a loop boundary.

The host side is shared with the generic implementation: the same validation,
the same output shapes, and the same empty-batch behaviour.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.multi_margin_loss import (
    _block_c,
    _check_grad_output,
    _check_inputs,
    _empty_forward,
    _normalize_p,
    _normalize_reduction,
    _output_shape,
    _validate_target_range,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_MAX_GRID_SIZE = 65535
_REDUCE_BLOCK_SIZE = 256


@libentry()
@triton.jit(do_not_specialize=["N", "C", "margin"])
def _multi_margin_loss_kernel(
    input_ptr,
    target_ptr,
    weight_ptr,
    output_ptr,
    N,
    C,
    margin,
    row_start,
    P: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0) + row_start
    target = tl.load(target_ptr + row)

    row_offset = row * C
    target_value = tl.load(input_ptr + row_offset + target).to(tl.float32)

    loss_sum = tl.zeros((), dtype=tl.float32)
    for class_start in range(0, C, BLOCK_C):
        classes = class_start + tl.arange(0, BLOCK_C)
        class_mask = classes < C
        values = tl.load(
            input_ptr + row_offset + classes,
            mask=class_mask,
            other=0.0,
        ).to(tl.float32)
        z = margin - target_value + values
        active = class_mask & (classes != target) & (z > 0)
        term = tl.where(active, z, 0.0)
        if P == 2:
            term = term * term
        loss_sum += tl.sum(term, axis=0)

    if HAS_WEIGHT:
        target_weight = tl.load(weight_ptr + target).to(tl.float32)
    else:
        target_weight = tl.full((), 1.0, tl.float32)

    loss = target_weight * loss_sum / C.to(tl.float32)
    tl.store(output_ptr + row, loss)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _multi_margin_loss_reduce_kernel(
    partial_ptr,
    output_ptr,
    N,
    REDUCTION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    total = tl.zeros((), dtype=tl.float32)
    for start in range(0, N, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        values = tl.load(partial_ptr + offsets, mask=offsets < N, other=0.0)
        total += tl.sum(values, axis=0)
    if REDUCTION == 1:
        total = total / N.to(tl.float32)
    tl.store(output_ptr, total)


@libentry()
@triton.jit(do_not_specialize=["N", "C", "margin"])
def _multi_margin_loss_backward_kernel(
    grad_output_ptr,
    input_ptr,
    target_ptr,
    weight_ptr,
    grad_input_ptr,
    N,
    C,
    margin,
    row_start,
    P: tl.constexpr,
    REDUCTION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0) + row_start
    target = tl.load(target_ptr + row)

    row_offset = row * C
    target_value = tl.load(input_ptr + row_offset + target).to(tl.float32)
    if HAS_WEIGHT:
        target_weight = tl.load(weight_ptr + target).to(tl.float32)
    else:
        target_weight = tl.full((), 1.0, tl.float32)

    if REDUCTION == 0:
        grad_output = tl.load(grad_output_ptr + row).to(tl.float32)
    else:
        grad_output = tl.load(grad_output_ptr).to(tl.float32)

    scale = grad_output * target_weight / C.to(tl.float32)
    if REDUCTION == 1:
        scale = scale / N.to(tl.float32)

    target_grad = tl.zeros((), dtype=tl.float32)
    for class_start in range(0, C, BLOCK_C):
        classes = class_start + tl.arange(0, BLOCK_C)
        class_mask = classes < C
        values = tl.load(
            input_ptr + row_offset + classes,
            mask=class_mask,
            other=0.0,
        ).to(tl.float32)
        z = margin - target_value + values
        active = class_mask & (classes != target) & (z > 0)
        if P == 1:
            grad = tl.where(active, scale, 0.0)
        else:
            grad = tl.where(active, 2.0 * z * scale, 0.0)
        target_grad -= tl.sum(grad, axis=0)
        tl.store(
            grad_input_ptr + row_offset + classes,
            grad,
            mask=class_mask,
        )

    tl.store(grad_input_ptr + row_offset + target, target_grad)


def _compute_forward(
    input,
    target,
    weight,
    N,
    C,
    is_batched,
    p,
    margin,
    reduction,
    output=None,
):
    if N == 0:
        return _empty_forward(input, N, is_batched, reduction)

    output_shape = _output_shape(N, is_batched, reduction)
    if output is None:
        output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    # A real pointer is passed for the no-weight specialization so every
    # supported Triton fork sees a valid pointer argument.
    weight_ptr = input if weight is None else weight
    has_weight = weight is not None
    block_c = _block_c(C)

    with torch_device_fn.device(input.device):
        if not is_batched or N == 1 or reduction == 0:
            # One program per row: the row loop of the generic kernel is what
            # this backend cannot compile once the class loop reduces, so the
            # rows are walked by the grid instead and split across launches
            # when the batch exceeds the largest grid.
            for start in range(0, N, _MAX_GRID_SIZE):
                count = min(_MAX_GRID_SIZE, N - start)
                _multi_margin_loss_kernel[(count,)](
                    input,
                    target,
                    weight_ptr,
                    output,
                    N,
                    C,
                    margin,
                    start,
                    P=p,
                    HAS_WEIGHT=has_weight,
                    BLOCK_C=block_c,
                )
        else:
            partial = torch.empty((N,), dtype=torch.float32, device=input.device)
            for start in range(0, N, _MAX_GRID_SIZE):
                count = min(_MAX_GRID_SIZE, N - start)
                _multi_margin_loss_kernel[(count,)](
                    input,
                    target,
                    weight_ptr,
                    partial,
                    N,
                    C,
                    margin,
                    start,
                    P=p,
                    HAS_WEIGHT=has_weight,
                    BLOCK_C=block_c,
                )
            _multi_margin_loss_reduce_kernel[(1,)](
                partial,
                output,
                N,
                REDUCTION=reduction,
                BLOCK_SIZE=_REDUCE_BLOCK_SIZE,
            )
    return output


def multi_margin_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    p=1,
    margin=1,
    weight=None,
    reduction=1,
) -> torch.Tensor:
    """Compute the multi-margin loss of ``input`` against ``target``."""
    logger.debug("GEMS_KUNLUNXIN MULTI_MARGIN_LOSS")
    p = _normalize_p(p)
    reduction = _normalize_reduction(reduction)
    try:
        margin = float(margin)
    except (TypeError, ValueError) as error:
        raise RuntimeError("multi_margin_loss: margin must be a real scalar") from error
    original_target = target
    input, target, weight, N, C, is_batched = _check_inputs(input, target, weight)
    _validate_target_range(original_target, target, N, C)
    return _compute_forward(
        input,
        target,
        weight,
        N,
        C,
        is_batched,
        p,
        margin,
        reduction,
    )


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
    """Write the multi-margin loss into ``out``."""
    logger.debug("GEMS_KUNLUNXIN MULTI_MARGIN_LOSS_OUT")
    p = _normalize_p(p)
    reduction = _normalize_reduction(reduction)
    try:
        margin = float(margin)
    except (TypeError, ValueError) as error:
        raise RuntimeError("multi_margin_loss: margin must be a real scalar") from error
    original_target = target
    input, target, weight, N, C, is_batched = _check_inputs(input, target, weight)
    _validate_target_range(original_target, target, N, C)
    if out.device != input.device:
        raise RuntimeError("multi_margin_loss.out: out must be on the input device")
    if out.dtype != input.dtype:
        raise RuntimeError("multi_margin_loss.out: out must have the input dtype")

    shape = _output_shape(N, is_batched, reduction)
    if tuple(out.shape) != shape:
        out.resize_(shape)
    destination = out if out.is_contiguous() else None
    result = _compute_forward(
        input,
        target,
        weight,
        N,
        C,
        is_batched,
        p,
        margin,
        reduction,
        output=destination,
    )
    if result is not out:
        out.copy_(result)
    return out


def _compute_backward(
    grad_output,
    input,
    target,
    weight,
    N,
    C,
    is_batched,
    p,
    margin,
    reduction,
    grad_input=None,
):
    if N == 0:
        if grad_input is None:
            return torch.zeros_like(input)
        grad_input.resize_(input.shape)
        return grad_input

    if grad_input is None:
        grad_input = torch.zeros(input.shape, dtype=input.dtype, device=input.device)

    weight_ptr = input if weight is None else weight
    has_weight = weight is not None
    block_c = _block_c(C)

    with torch_device_fn.device(input.device):
        for start in range(0, N, _MAX_GRID_SIZE):
            count = min(_MAX_GRID_SIZE, N - start)
            _multi_margin_loss_backward_kernel[(count,)](
                grad_output,
                input,
                target,
                weight_ptr,
                grad_input,
                N,
                C,
                margin,
                start,
                P=p,
                REDUCTION=reduction,
                HAS_WEIGHT=has_weight,
                BLOCK_C=block_c,
            )
    return grad_input


def multi_margin_loss_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    target: torch.Tensor,
    p=1,
    margin=1,
    weight=None,
    reduction=1,
) -> torch.Tensor:
    """Gradient of the multi-margin loss with respect to ``input``."""
    logger.debug("GEMS_KUNLUNXIN MULTI_MARGIN_LOSS_BACKWARD")
    p = _normalize_p(p)
    reduction = _normalize_reduction(reduction)
    original_target = target
    input, target, weight, N, C, is_batched = _check_inputs(input, target, weight)
    _validate_target_range(original_target, target, N, C)
    grad_output = _check_grad_output(grad_output, input, N, is_batched, reduction)
    return _compute_backward(
        grad_output,
        input,
        target,
        weight,
        N,
        C,
        is_batched,
        p,
        margin,
        reduction,
    )


def multi_margin_loss_backward_out(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    target: torch.Tensor,
    p=1,
    margin=1,
    weight=None,
    reduction=1,
    *,
    grad_input: torch.Tensor,
) -> torch.Tensor:
    """Write the multi-margin loss gradient into ``grad_input``."""
    logger.debug("GEMS_KUNLUNXIN MULTI_MARGIN_LOSS_BACKWARD_OUT")
    p = _normalize_p(p)
    reduction = _normalize_reduction(reduction)
    original_target = target
    input, target, weight, N, C, is_batched = _check_inputs(input, target, weight)
    _validate_target_range(original_target, target, N, C)
    grad_output = _check_grad_output(grad_output, input, N, is_batched, reduction)
    if grad_input.device != input.device:
        raise RuntimeError(
            "multi_margin_loss_backward.out: grad_input must be on the input device"
        )
    if grad_input.dtype != input.dtype:
        raise RuntimeError(
            "multi_margin_loss_backward.out: grad_input must have the input dtype"
        )
    if tuple(grad_input.shape) != tuple(input.shape):
        grad_input.resize_(input.shape)
    destination = grad_input if grad_input.is_contiguous() else None
    result = _compute_backward(
        grad_output,
        input,
        target,
        weight,
        N,
        C,
        is_batched,
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
    "multi_margin_loss_out",
]
