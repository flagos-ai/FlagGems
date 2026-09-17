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
import weakref

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# FlagGems selects one backend before importing operators. Cache descriptor
# fields that stay fixed for the process instead of resolving them per call.
_DEVICE_NAME = runtime.device.name
_VENDOR_NAME = runtime.device.vendor_name
_SUPPORTS_FP64 = runtime.device.support_fp64
_SUPPORTS_BF16 = runtime.device.support_bf16

_MAX_GRID_SIZE = 65535
_MAX_BLOCK_C = 256
_HYGON_FUSED_REDUCED_MAX_ELEMENTS = 2048
_HYGON_PARTIAL_MAX_BLOCK_C = 1024
_HYGON_PARTIAL_TILE_ELEMENTS = 2048
_HYGON_ROWS_BLOCK_N = 4
_MTHREADS_BACKWARD_BLOCK_SIZE = 256
_MTHREADS_BLOCK_N = 4
_REDUCE_BLOCK_SIZE = 256
_TARGET_CHECK_BLOCK_SIZE = 256
_TARGET_VALIDATION_CACHE_LIMIT = 128
_REDUCTION_CODES = {"none": 0, "mean": 1, "sum": 2}

# Some vendor compilers do not provide a reliable tl.device_assert contract:
# CoreX only prints, MUSA can hang after reporting, and CANN 8.5 removes the
# assertion during lowering. Keep the CUDA/NVIDIA device-assert path, but make
# affected backends report a reliable host error before the loss kernel runs.
_SYNCHRONOUS_TARGET_CHECK_VENDORS = {"ascend", "iluvatar", "mthreads"}
_REQUIRES_SYNCHRONOUS_TARGET_CHECK = _VENDOR_NAME in _SYNCHRONOUS_TARGET_CHECK_VENDORS
_USE_DEVICE_ASSERT = not _REQUIRES_SYNCHRONOUS_TARGET_CHECK
_IS_HYGON_BACKEND = _VENDOR_NAME == "hygon"
_IS_MTHREADS_BACKEND = _VENDOR_NAME == "mthreads"
_TARGET_VALIDATION_CACHE = {}


@libentry()
@triton.jit(do_not_specialize=["N", "C"])
def _multi_margin_loss_validate_target_kernel(
    target_ptr,
    invalid_ptr,
    N,
    C,
    BLOCK_SIZE: tl.constexpr,
):
    invalid = tl.zeros((), dtype=tl.int32)
    for start in range(0, N, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        targets = tl.load(target_ptr + offsets, mask=mask, other=0)
        out_of_bounds = mask & ((targets < 0) | (targets >= C))
        invalid += tl.sum(out_of_bounds.to(tl.int32), axis=0)
    tl.store(invalid_ptr, invalid)


@libentry()
@triton.jit(
    debug=True,
    do_not_specialize=["N", "C", "margin"],
)
def _multi_margin_loss_kernel(
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
    BLOCK_C: tl.constexpr,
):
    acc_dtype = tl.float64 if input_ptr.type.element_ty == tl.float64 else tl.float32
    margin = margin.to(acc_dtype)
    pid = tl.program_id(0)
    program_count = tl.num_programs(0)

    for row in range(pid, N, program_count):
        target = tl.load(target_ptr + row)
        valid_target = (target >= 0) & (target < C)
        if USE_DEVICE_ASSERT:
            tl.device_assert(
                valid_target,
                "multi_margin_loss: target index is out of bounds",
            )

        # Keep every indirect access in range even on compilers where a device
        # assertion is reported asynchronously.
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
def _multi_margin_loss_mthreads_rows_kernel(
    input_ptr,
    target_ptr,
    weight_ptr,
    output_ptr,
    N,
    C,
    margin,
    P: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Compute several rows per program to avoid MUSA scalar-grid stalls."""
    acc_dtype = tl.float64 if input_ptr.type.element_ty == tl.float64 else tl.float32
    margin = margin.to(acc_dtype)
    pid = tl.program_id(0)
    row_stride = tl.num_programs(0) * BLOCK_N

    for row_start in range(pid * BLOCK_N, N, row_stride):
        rows = row_start + tl.arange(0, BLOCK_N)[:, None]
        row_mask = rows < N
        targets = tl.load(target_ptr + rows, mask=row_mask, other=0)
        valid_target = (targets >= 0) & (targets < C)
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
def _multi_margin_loss_hygon_rows_kernel(
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
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
    row_mask = rows < N
    classes = tl.arange(0, BLOCK_C)[None, :]
    class_mask = classes < C

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
    values = tl.load(
        input_ptr + rows * C + classes,
        mask=row_mask & class_mask,
        other=0.0,
    ).to(acc_dtype)
    z = margin - target_values + values
    active = row_mask & class_mask & valid_target & (classes != safe_targets) & (z > 0)
    terms = tl.where(active, z, 0.0)
    if P == 2:
        terms = terms * terms
    loss_sums = tl.sum(terms, axis=1)

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
def _multi_margin_loss_hygon_fused_reduced_kernel(
    input_ptr,
    target_ptr,
    weight_ptr,
    output_ptr,
    N,
    C,
    margin,
    P: tl.constexpr,
    REDUCTION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    USE_DEVICE_ASSERT: tl.constexpr,
    BLOCK_NC: tl.constexpr,
):
    acc_dtype = tl.float64 if input_ptr.type.element_ty == tl.float64 else tl.float32
    margin = margin.to(acc_dtype)
    offsets = tl.arange(0, BLOCK_NC)
    element_count = N * C
    mask = offsets < element_count
    rows = offsets // C
    classes = offsets - rows * C

    targets = tl.load(target_ptr + rows, mask=mask, other=0)
    valid_target = (targets >= 0) & (targets < C)
    if USE_DEVICE_ASSERT:
        invalid_count = tl.sum((mask & ~valid_target).to(tl.int32), axis=0)
        tl.device_assert(
            invalid_count == 0,
            "multi_margin_loss: target index is out of bounds",
        )

    safe_targets = tl.where(valid_target, targets, 0)
    target_values = tl.load(
        input_ptr + rows * C + safe_targets,
        mask=mask & valid_target,
        other=0.0,
    ).to(acc_dtype)
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(acc_dtype)
    z = margin - target_values + values
    active = mask & valid_target & (classes != safe_targets) & (z > 0)
    terms = tl.where(active, z, 0.0)
    if P == 2:
        terms = terms * terms

    if HAS_WEIGHT:
        target_weights = tl.load(
            weight_ptr + safe_targets,
            mask=mask & valid_target,
            other=0.0,
        ).to(acc_dtype)
        terms = terms * target_weights

    total = tl.sum(terms, axis=0) / C.to(acc_dtype)
    if REDUCTION == 1:
        total /= N.to(acc_dtype)
    tl.store(output_ptr, total)


@libentry()
@triton.jit(
    debug=True,
    do_not_specialize=["N", "C", "margin"],
)
def _multi_margin_loss_hygon_partial_2d_kernel(
    input_ptr,
    target_ptr,
    weight_ptr,
    partial_ptr,
    N,
    C,
    margin,
    P: tl.constexpr,
    REDUCTION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    USE_DEVICE_ASSERT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    acc_dtype = tl.float64 if input_ptr.type.element_ty == tl.float64 else tl.float32
    margin = margin.to(acc_dtype)
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
    row_mask = rows < N
    classes = tl.arange(0, BLOCK_C)[None, :]
    class_mask = classes < C

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
    values = tl.load(
        input_ptr + rows * C + classes,
        mask=row_mask & class_mask,
        other=0.0,
    ).to(acc_dtype)
    z = margin - target_values + values
    active = row_mask & class_mask & valid_target & (classes != safe_targets) & (z > 0)
    terms = tl.where(active, z, 0.0)
    if P == 2:
        terms = terms * terms
    loss_sums = tl.sum(terms, axis=1)

    if HAS_WEIGHT:
        target_weights = tl.load(
            weight_ptr + safe_targets,
            mask=row_mask & valid_target,
            other=0.0,
        ).to(acc_dtype)
    else:
        target_weights = tl.full((BLOCK_N, 1), 1.0, acc_dtype)

    losses = target_weights * loss_sums[:, None] / C.to(acc_dtype)
    if REDUCTION == 1:
        losses = losses / N.to(acc_dtype)
    tile_total = tl.sum(tl.where(row_mask, losses, 0.0), axis=0)
    tile_total = tl.sum(tile_total, axis=0)
    tl.store(partial_ptr + pid, tile_total)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _multi_margin_loss_reduce_kernel(
    input_ptr,
    output_ptr,
    N,
    REDUCTION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    acc_dtype = tl.float64 if input_ptr.type.element_ty == tl.float64 else tl.float32
    total = tl.zeros((), dtype=acc_dtype)
    for start in range(0, N, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(acc_dtype)
        total += tl.sum(tl.where(mask, values, 0.0), axis=0)
    if REDUCTION == 1:
        total /= N.to(acc_dtype)
    tl.store(output_ptr, total)


@libentry()
@triton.jit
def _multi_margin_loss_mthreads_flat_backward_kernel(
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
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    element_count = N * C
    element_stride = tl.num_programs(0) * BLOCK_SIZE
    margin = margin.to(tl.float32)

    for start in range(pid * BLOCK_SIZE, element_count, element_stride):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < element_count
        rows = offsets // C
        classes = offsets % C
        targets = tl.load(target_ptr + rows, mask=mask, other=0)
        safe_targets = tl.where((targets >= 0) & (targets < C), targets, 0)
        target_values = tl.load(
            input_ptr + rows * C + safe_targets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        if HAS_WEIGHT:
            target_weights = tl.load(
                weight_ptr + safe_targets,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
        else:
            target_weights = tl.full((BLOCK_SIZE,), 1.0, tl.float32)

        if REDUCTION == 0:
            grad_output = tl.load(
                grad_output_ptr + rows,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
        else:
            grad_output = tl.load(grad_output_ptr).to(tl.float32)

        scale = grad_output * target_weights / C.to(tl.float32)
        if REDUCTION == 1:
            scale /= N.to(tl.float32)
        z = margin - target_values + values
        active = mask & (classes != safe_targets) & (z > 0)
        if P == 1:
            grads = tl.where(active, scale, 0.0)
        else:
            grads = tl.where(active, 2.0 * z * scale, 0.0)
        tl.store(grad_input_ptr + offsets, grads, mask=mask)


@libentry()
@triton.jit
def _multi_margin_loss_mthreads_target_backward_kernel(
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
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    row_stride = tl.num_programs(0) * BLOCK_N
    margin = margin.to(tl.float32)

    for row_start in range(pid * BLOCK_N, N, row_stride):
        rows = row_start + tl.arange(0, BLOCK_N)[:, None]
        row_mask = rows < N
        targets = tl.load(target_ptr + rows, mask=row_mask, other=0)
        safe_targets = tl.where((targets >= 0) & (targets < C), targets, 0)
        target_values = tl.load(
            input_ptr + rows * C + safe_targets,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)

        if HAS_WEIGHT:
            target_weights = tl.load(
                weight_ptr + safe_targets,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
        else:
            target_weights = tl.full((BLOCK_N, 1), 1.0, tl.float32)

        if REDUCTION == 0:
            grad_output = tl.load(
                grad_output_ptr + rows,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
        else:
            grad_output = tl.load(grad_output_ptr).to(tl.float32)

        scale = grad_output * target_weights / C.to(tl.float32)
        if REDUCTION == 1:
            scale /= N.to(tl.float32)

        target_grads = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for class_start in range(0, C, BLOCK_C):
            classes = class_start + tl.arange(0, BLOCK_C)[None, :]
            class_mask = classes < C
            values = tl.load(
                input_ptr + rows * C + classes,
                mask=row_mask & class_mask,
                other=0.0,
            ).to(tl.float32)
            z = margin - target_values + values
            active = row_mask & class_mask & (classes != safe_targets) & (z > 0)
            if P == 1:
                grads = tl.where(active, scale, 0.0)
            else:
                grads = tl.where(active, 2.0 * z * scale, 0.0)
            target_grads -= tl.sum(grads, axis=1)

        tl.store(
            grad_input_ptr + rows * C + safe_targets,
            target_grads[:, None],
            mask=row_mask,
        )


@libentry()
@triton.jit(
    debug=True,
    do_not_specialize=["N", "C", "margin"],
)
def _multi_margin_loss_rows_backward_kernel(
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
                "multi_margin_loss_backward: target index is out of bounds",
            )

        safe_targets = tl.where(valid_target, targets, 0)
        target_values = tl.load(
            input_ptr + rows * C + safe_targets,
            mask=row_mask & valid_target,
            other=0.0,
        ).to(acc_dtype)
        if HAS_WEIGHT:
            target_weights = tl.load(
                weight_ptr + safe_targets,
                mask=row_mask & valid_target,
                other=0.0,
            ).to(acc_dtype)
        else:
            target_weights = tl.full((BLOCK_N, 1), 1.0, acc_dtype)

        if REDUCTION == 0:
            grad_output = tl.load(
                grad_output_ptr + rows,
                mask=row_mask,
                other=0.0,
            ).to(acc_dtype)
        else:
            grad_output = tl.load(grad_output_ptr).to(acc_dtype)

        scale = grad_output * target_weights / C.to(acc_dtype)
        if REDUCTION == 1:
            scale /= N.to(acc_dtype)

        target_grads = tl.zeros((BLOCK_N,), dtype=acc_dtype)
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
            if P == 1:
                grads = tl.where(active, scale, 0.0)
            else:
                grads = tl.where(active, 2.0 * z * scale, 0.0)
            target_grads -= tl.sum(grads, axis=1)
            tl.store(
                grad_input_ptr + rows * C + classes,
                grads,
                mask=row_mask & class_mask,
            )

        tl.store(
            grad_input_ptr + rows * C + safe_targets,
            target_grads[:, None],
            mask=row_mask & valid_target,
        )


@libentry()
@triton.jit(
    debug=True,
    do_not_specialize=["N", "C", "margin"],
)
def _multi_margin_loss_backward_kernel(
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
    USE_DEVICE_ASSERT: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    acc_dtype = tl.float64 if input_ptr.type.element_ty == tl.float64 else tl.float32
    margin = margin.to(acc_dtype)
    pid = tl.program_id(0)
    program_count = tl.num_programs(0)

    for row in range(pid, N, program_count):
        target = tl.load(target_ptr + row)
        valid_target = (target >= 0) & (target < C)
        if USE_DEVICE_ASSERT:
            tl.device_assert(
                valid_target,
                "multi_margin_loss_backward: target index is out of bounds",
            )

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
            scale /= N.to(acc_dtype)

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


def _normalize_p(p) -> int:
    try:
        value = float(p)
    except (TypeError, ValueError) as error:
        raise RuntimeError("multi_margin_loss: p must be 1 or 2") from error
    if value not in (1.0, 2.0):
        raise RuntimeError(f"multi_margin_loss: p must be 1 or 2, got {p}")
    return int(value)


def _normalize_reduction(reduction) -> int:
    if isinstance(reduction, str):
        try:
            return _REDUCTION_CODES[reduction]
        except KeyError:
            pass
    elif isinstance(reduction, int) and reduction in (0, 1, 2):
        return reduction
    raise RuntimeError(
        "multi_margin_loss: reduction must be one of none/mean/sum or 0/1/2"
    )


def _shape_info(input: torch.Tensor) -> tuple[int, int, bool]:
    if input.dim() > 2:
        raise RuntimeError(
            f"multi_margin_loss: expected scalar, 1D, or 2D input, got {input.dim()}D"
        )
    if input.dim() == 0:
        return 1, 1, False
    if input.dim() == 1:
        C = input.shape[0]
        if C == 0:
            raise RuntimeError(
                "multi_margin_loss: expected a non-empty class dimension"
            )
        return 1, C, False

    N, C = input.shape
    if C == 0:
        raise RuntimeError("multi_margin_loss: expected a non-empty class dimension")
    return N, C, True


def _check_inputs(input, target, weight):
    if not isinstance(input, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("multi_margin_loss: input and target must be tensors")
    if input.device.type != _DEVICE_NAME:
        raise RuntimeError(
            f"multi_margin_loss: input must be on a {_DEVICE_NAME} device"
        )
    if input.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise RuntimeError("multi_margin_loss: input must have a floating point dtype")
    if input.dtype == torch.float64 and not _SUPPORTS_FP64:
        raise RuntimeError(
            "multi_margin_loss: float64 is not supported on this backend"
        )
    if input.dtype == torch.bfloat16 and not _SUPPORTS_BF16:
        raise RuntimeError(
            "multi_margin_loss: bfloat16 is not supported on this backend"
        )
    if target.dtype != torch.int64:
        raise RuntimeError("multi_margin_loss: target must have dtype int64")
    if target.device != input.device:
        raise RuntimeError(
            "multi_margin_loss: input and target must be on the same device"
        )

    N, C, is_batched = _shape_info(input)
    if is_batched:
        if tuple(target.shape) != (N,):
            raise RuntimeError(
                "multi_margin_loss: target for a 2D input must have shape [N]"
            )
    elif target.dim() > 1 or target.numel() != 1:
        raise RuntimeError(
            "multi_margin_loss: scalar or 1D input requires exactly one target"
        )

    if weight is not None:
        if not isinstance(weight, torch.Tensor):
            raise TypeError("multi_margin_loss: weight must be a tensor or None")
        if tuple(weight.shape) != (C,):
            raise RuntimeError("multi_margin_loss: weight must have shape [C]")
        if weight.dtype != input.dtype:
            raise RuntimeError(
                "multi_margin_loss: weight must have the same dtype as input"
            )
        if weight.device != input.device:
            raise RuntimeError(
                "multi_margin_loss: weight must be on the same device as input"
            )

    input = input.contiguous()
    original_target = target
    target = target.contiguous()
    weight = None if weight is None else weight.contiguous()
    if N > 0 and _REQUIRES_SYNCHRONOUS_TARGET_CHECK:
        _validate_target_range(original_target, target, N, C)
    return input, target, weight, N, C, is_batched


def _output_shape(N: int, is_batched: bool, reduction: int) -> tuple[int, ...]:
    if is_batched and reduction == 0:
        return (N,)
    return ()


def _block_c(C: int) -> int:
    return min(_MAX_BLOCK_C, triton.next_power_of_2(C))


def _hygon_partial_config(N: int, C: int):
    if C > _HYGON_PARTIAL_MAX_BLOCK_C:
        return None
    block_c = triton.next_power_of_2(C)
    block_n = _HYGON_PARTIAL_TILE_ELEMENTS // block_c
    partial_count = triton.cdiv(N, block_n)
    if partial_count > _MAX_GRID_SIZE:
        return None
    return block_n, block_c, partial_count


def _target_validation_key(target, N, C):
    return (
        _VENDOR_NAME,
        target.device.type,
        target.device.index,
        target.data_ptr(),
        target.numel(),
        target._version,
        N,
        C,
    )


def _validate_target_range(target, contiguous_target, N, C):
    key = _target_validation_key(target, N, C)
    cached_target = _TARGET_VALIDATION_CACHE.get(key)
    if cached_target is not None and cached_target() is target:
        return

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

    # Identity-checking a weak reference prevents stale data_ptr reuse without
    # retaining up to 128 potentially large target tensors. Standard in-place
    # mutations advance _version and therefore produce a cache miss.
    if len(_TARGET_VALIDATION_CACHE) >= _TARGET_VALIDATION_CACHE_LIMIT:
        _TARGET_VALIDATION_CACHE.clear()
    _TARGET_VALIDATION_CACHE[key] = weakref.ref(target)


def _empty_forward(input, N, is_batched, reduction):
    shape = _output_shape(N, is_batched, reduction)
    if reduction == 0:
        return torch.empty(shape, dtype=input.dtype, device=input.device)
    if reduction == 1:
        return torch.full((), float("nan"), dtype=input.dtype, device=input.device)
    return torch.zeros((), dtype=input.dtype, device=input.device)


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
    grid_size = min(N, _MAX_GRID_SIZE)
    block_c = _block_c(C)
    hygon_rows_config = None
    if (
        _IS_HYGON_BACKEND
        and is_batched
        and N > 1
        and reduction == 0
        and C <= _HYGON_PARTIAL_MAX_BLOCK_C
    ):
        rows_block_c = triton.next_power_of_2(C)
        rows_block_n = _HYGON_PARTIAL_TILE_ELEMENTS // rows_block_c
        rows_grid_size = triton.cdiv(N, rows_block_n)
        if rows_grid_size <= _MAX_GRID_SIZE:
            hygon_rows_config = rows_block_n, rows_block_c, rows_grid_size
    hygon_partial_config = None
    if _IS_HYGON_BACKEND and is_batched and N > 1 and reduction != 0:
        hygon_partial_config = _hygon_partial_config(N, C)

    with torch_device_fn.device(input.device):
        if hygon_rows_config is not None:
            rows_block_n, rows_block_c, rows_grid_size = hygon_rows_config
            _multi_margin_loss_hygon_rows_kernel[(rows_grid_size,)](
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
                BLOCK_N=rows_block_n,
                BLOCK_C=rows_block_c,
            )
        elif (
            _IS_HYGON_BACKEND
            and is_batched
            and N > 1
            and reduction != 0
            and N * C <= _HYGON_FUSED_REDUCED_MAX_ELEMENTS
        ):
            _multi_margin_loss_hygon_fused_reduced_kernel[(1,)](
                input,
                target,
                weight_ptr,
                output,
                N,
                C,
                margin,
                P=p,
                REDUCTION=reduction,
                HAS_WEIGHT=has_weight,
                USE_DEVICE_ASSERT=_USE_DEVICE_ASSERT,
                BLOCK_NC=triton.next_power_of_2(N * C),
            )
        elif hygon_partial_config is not None:
            partial_block_n, partial_block_c, partial_count = hygon_partial_config
            accumulator_dtype = (
                torch.float64 if input.dtype == torch.float64 else torch.float32
            )
            partial = torch.empty(
                (partial_count,),
                dtype=accumulator_dtype,
                device=input.device,
            )
            _multi_margin_loss_hygon_partial_2d_kernel[(partial_count,)](
                input,
                target,
                weight_ptr,
                partial,
                N,
                C,
                margin,
                P=p,
                REDUCTION=reduction,
                HAS_WEIGHT=has_weight,
                USE_DEVICE_ASSERT=_USE_DEVICE_ASSERT,
                BLOCK_N=partial_block_n,
                BLOCK_C=partial_block_c,
            )
            _multi_margin_loss_reduce_kernel[(1,)](
                partial,
                output,
                partial_count,
                REDUCTION=2,
                BLOCK_SIZE=_REDUCE_BLOCK_SIZE,
            )
        elif _IS_MTHREADS_BACKEND and is_batched and N > 1:
            grid_size = min(
                triton.cdiv(N, _MTHREADS_BLOCK_N),
                _MAX_GRID_SIZE,
            )
            if reduction == 0:
                row_losses = output
            else:
                row_losses = torch.empty(
                    (N,),
                    dtype=torch.float32,
                    device=input.device,
                )
            _multi_margin_loss_mthreads_rows_kernel[(grid_size,)](
                input,
                target,
                weight_ptr,
                row_losses,
                N,
                C,
                margin,
                P=p,
                HAS_WEIGHT=has_weight,
                BLOCK_N=_MTHREADS_BLOCK_N,
                BLOCK_C=block_c,
            )
            if reduction != 0:
                _multi_margin_loss_reduce_kernel[(1,)](
                    row_losses,
                    output,
                    N,
                    REDUCTION=reduction,
                    BLOCK_SIZE=_REDUCE_BLOCK_SIZE,
                )
        elif not is_batched or N == 1 or reduction == 0:
            _multi_margin_loss_kernel[(grid_size,)](
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
                BLOCK_C=block_c,
            )
        else:
            accumulator_dtype = (
                torch.float64 if input.dtype == torch.float64 else torch.float32
            )
            partial = torch.empty((N,), dtype=accumulator_dtype, device=input.device)
            _multi_margin_loss_kernel[(grid_size,)](
                input,
                target,
                weight_ptr,
                partial,
                N,
                C,
                margin,
                P=p,
                HAS_WEIGHT=has_weight,
                USE_DEVICE_ASSERT=_USE_DEVICE_ASSERT,
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
    logger.debug("GEMS MULTI_MARGIN_LOSS")
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
    logger.debug("GEMS MULTI_MARGIN_LOSS OUT")
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


def _check_grad_output(grad_output, input, N, is_batched, reduction):
    if not isinstance(grad_output, torch.Tensor):
        raise TypeError("multi_margin_loss_backward: grad_output must be a tensor")
    if grad_output.device != input.device:
        raise RuntimeError(
            "multi_margin_loss_backward: grad_output must be on the input device"
        )
    if grad_output.dtype != input.dtype:
        raise RuntimeError(
            "multi_margin_loss_backward: grad_output must have the input dtype"
        )
    expected = _output_shape(N, is_batched, reduction)
    if tuple(grad_output.shape) != expected:
        raise RuntimeError(
            "multi_margin_loss_backward: grad_output has an invalid shape"
        )
    return grad_output.contiguous()


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

    weight_ptr = input if weight is None else weight
    grid_size = min(N, _MAX_GRID_SIZE)
    with torch_device_fn.device(input.device):
        if _IS_MTHREADS_BACKEND and N > 1:
            flat_grid = min(
                triton.cdiv(N * C, _MTHREADS_BACKWARD_BLOCK_SIZE),
                _MAX_GRID_SIZE,
            )
            row_grid = min(
                triton.cdiv(N, _MTHREADS_BLOCK_N),
                _MAX_GRID_SIZE,
            )
            _multi_margin_loss_mthreads_flat_backward_kernel[(flat_grid,)](
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
                BLOCK_SIZE=_MTHREADS_BACKWARD_BLOCK_SIZE,
            )
            _multi_margin_loss_mthreads_target_backward_kernel[(row_grid,)](
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
                BLOCK_N=_MTHREADS_BLOCK_N,
                BLOCK_C=_block_c(C),
            )
        elif _IS_HYGON_BACKEND and N > 1:
            grid_size = min(
                triton.cdiv(N, _HYGON_ROWS_BLOCK_N),
                _MAX_GRID_SIZE,
            )
            _multi_margin_loss_rows_backward_kernel[(grid_size,)](
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
                BLOCK_N=_HYGON_ROWS_BLOCK_N,
                BLOCK_C=_block_c(C),
            )
        else:
            _multi_margin_loss_backward_kernel[(grid_size,)](
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
    return grad_input


def multi_margin_loss_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    target: torch.Tensor,
    p,
    margin,
    weight=None,
    reduction=1,
) -> torch.Tensor:
    logger.debug("GEMS MULTI_MARGIN_LOSS BACKWARD")
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
    logger.debug("GEMS MULTI_MARGIN_LOSS BACKWARD OUT")
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
