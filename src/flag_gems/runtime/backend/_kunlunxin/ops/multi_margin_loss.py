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


import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.multi_margin_loss import (
    _MAX_GRID_SIZE,
    _REDUCE_BLOCK_SIZE,
    _TARGET_CHECK_BLOCK_SIZE,
    _block_c,
    _check_grad_output,
    _check_inputs,
    _empty_forward,
    _multi_margin_loss_reduce_kernel,
    _multi_margin_loss_validate_target_kernel,
    _normalize_p,
    _normalize_reduction,
    _output_shape,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["N", "C", "margin"])
def _mml_fwd_row_kernel(
    input_ptr,
    target_ptr,
    weight_ptr,
    output_ptr,
    N,
    C,
    margin,
    P: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    acc_dtype = tl.float64 if input_ptr.type.element_ty == tl.float64 else tl.float32
    margin = margin.to(acc_dtype)
    row = tl.program_id(0)

    target = tl.load(target_ptr + row)
    valid_target = (target >= 0) & (target < C)
    safe_target = tl.where(valid_target, target, 0)
    row_offset = row * C
    target_value = tl.load(
        input_ptr + row_offset + safe_target,
        mask=valid_target,
        other=0.0,
    ).to(acc_dtype)

    loss_sum = tl.zeros((), dtype=acc_dtype)
    for class_start in range(0, C, BLOCK_C):
        classes = class_start + tl.arange(0, BLOCK_C)
        class_mask = classes < C
        values = tl.load(
            input_ptr + row_offset + classes,
            mask=class_mask,
            other=0.0,
        ).to(acc_dtype)
        z = margin - target_value + values
        active = class_mask & valid_target & (classes != safe_target) & (z > 0)
        term = tl.where(active, z, 0.0)
        if P == 2:
            term = term * term
        loss_sum += tl.sum(term, axis=0)

    if HAS_WEIGHT:
        target_weight = tl.load(
            weight_ptr + safe_target,
            mask=valid_target,
            other=0.0,
        ).to(acc_dtype)
    else:
        target_weight = tl.full((), 1.0, acc_dtype)

    loss = target_weight * loss_sum / C.to(acc_dtype)
    tl.store(output_ptr + row, loss)


@libentry()
@triton.jit(do_not_specialize=["N", "C", "margin"])
def _mml_bwd_row_kernel(
    grad_output_ptr,
    input_ptr,
    target_ptr,
    weight_ptr,
    grad_input_ptr,
    N,
    C,
    margin,
    P: tl.constexpr,
    REDUCTION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    acc_dtype = tl.float64 if input_ptr.type.element_ty == tl.float64 else tl.float32
    margin = margin.to(acc_dtype)
    row = tl.program_id(0)

    target = tl.load(target_ptr + row)
    valid_target = (target >= 0) & (target < C)
    safe_target = tl.where(valid_target, target, 0)
    row_offset = row * C
    target_value = tl.load(
        input_ptr + row_offset + safe_target,
        mask=valid_target,
        other=0.0,
    ).to(acc_dtype)

    if HAS_WEIGHT:
        target_weight = tl.load(
            weight_ptr + safe_target,
            mask=valid_target,
            other=0.0,
        ).to(acc_dtype)
    else:
        target_weight = tl.full((), 1.0, acc_dtype)

    if REDUCTION == 0:
        grad_output = tl.load(grad_output_ptr + row).to(acc_dtype)
    else:
        grad_output = tl.load(grad_output_ptr).to(acc_dtype)

    scale = grad_output * target_weight / C.to(acc_dtype)
    if REDUCTION == 1:
        scale = scale / N.to(acc_dtype)

    target_grad = tl.zeros((), dtype=acc_dtype)
    for class_start in range(0, C, BLOCK_C):
        classes = class_start + tl.arange(0, BLOCK_C)
        class_mask = classes < C
        values = tl.load(
            input_ptr + row_offset + classes,
            mask=class_mask,
            other=0.0,
        ).to(acc_dtype)
        z = margin - target_value + values
        active = class_mask & valid_target & (classes != safe_target) & (z > 0)
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

    tl.store(
        grad_input_ptr + row_offset + safe_target,
        target_grad,
        mask=valid_target,
    )


def _validate_target_range_xpu(contiguous_target, N, C):
    """Host-synchronous OOB target check (device_assert is unavailable on XPU)."""
    invalid = torch.empty((), dtype=torch.int32, device=contiguous_target.device)
    with torch_device_fn.device(contiguous_target.device):
        _multi_margin_loss_validate_target_kernel[(1,)](
            contiguous_target,
            invalid,
            N,
            C,
            BLOCK_SIZE=_TARGET_CHECK_BLOCK_SIZE,
        )
    if invalid.item() != 0:
        raise RuntimeError("multi_margin_loss: target index is out of bounds")


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

    _validate_target_range_xpu(target, N, C)

    output_shape = _output_shape(N, is_batched, reduction)
    if output is None:
        output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    weight_ptr = input if weight is None else weight
    has_weight = weight is not None
    block_c = _block_c(C)

    if N > _MAX_GRID_SIZE:
        raise RuntimeError(
            "multi_margin_loss: batch size exceeds the supported grid limit "
            f"({_MAX_GRID_SIZE}) on this backend"
        )

    with torch_device_fn.device(input.device):
        if (not is_batched) or N == 1 or reduction == 0:
            _mml_fwd_row_kernel[(N,)](
                input,
                target,
                weight_ptr,
                output,
                N,
                C,
                margin,
                P=p,
                HAS_WEIGHT=has_weight,
                BLOCK_C=block_c,
            )
        else:
            accumulator_dtype = (
                torch.float64 if input.dtype == torch.float64 else torch.float32
            )
            partial = torch.empty((N,), dtype=accumulator_dtype, device=input.device)
            _mml_fwd_row_kernel[(N,)](
                input,
                target,
                weight_ptr,
                partial,
                N,
                C,
                margin,
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


def _compute_backward(
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

    _validate_target_range_xpu(target, N, C)

    weight_ptr = input if weight is None else weight

    if N > _MAX_GRID_SIZE:
        raise RuntimeError(
            "multi_margin_loss_backward: batch size exceeds the supported grid "
            f"limit ({_MAX_GRID_SIZE}) on this backend"
        )

    with torch_device_fn.device(input.device):
        _mml_bwd_row_kernel[(N,)](
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
            BLOCK_C=_block_c(C),
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
    logger.debug("GEMS_KUNLUNXIN MULTI_MARGIN_LOSS")
    p = _normalize_p(p)
    reduction = _normalize_reduction(reduction)
    try:
        margin = float(margin)
    except (TypeError, ValueError) as error:
        raise RuntimeError("multi_margin_loss: margin must be a real scalar") from error
    input, target, weight, N, C, is_batched = _check_inputs(input, target, weight)
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
    logger.debug("GEMS_KUNLUNXIN MULTI_MARGIN_LOSS OUT")
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


def multi_margin_loss_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    target: torch.Tensor,
    p,
    margin,
    weight=None,
    reduction=1,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN MULTI_MARGIN_LOSS BACKWARD")
    p = _normalize_p(p)
    reduction = _normalize_reduction(reduction)
    try:
        margin = float(margin)
    except (TypeError, ValueError) as error:
        raise RuntimeError("multi_margin_loss: margin must be a real scalar") from error
    input, target, weight, N, C, is_batched = _check_inputs(input, target, weight)
    grad_output = _check_grad_output(grad_output, input, N, is_batched, reduction)
    return _compute_backward(
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
    logger.debug("GEMS_KUNLUNXIN MULTI_MARGIN_LOSS BACKWARD OUT")
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
    result = _compute_backward(
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


import sys as _sys  # noqa: E402

_generic_ops_module = _sys.modules.get("flag_gems.ops")
if _generic_ops_module is not None:
    for _name, _fn in (
        ("multi_margin_loss", multi_margin_loss),
        ("multi_margin_loss_out", multi_margin_loss_out),
        ("multi_margin_loss_backward", multi_margin_loss_backward),
        ("multi_margin_loss_backward_out", multi_margin_loss_backward_out),
    ):
        setattr(_generic_ops_module, _name, _fn)
