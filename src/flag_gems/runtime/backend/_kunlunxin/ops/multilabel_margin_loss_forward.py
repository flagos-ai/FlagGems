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
import sys as _sys

import torch
import triton
import triton.language as tl

from flag_gems.ops.multilabel_margin_loss_forward import (
    _FUSED_CLASS_LIMIT,
    _VALIDATION_BLOCK,
    _check_inputs,
    _empty_batch_result,
    _normalize_reduction,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_LARGE_C_BLOCK = 64


@libentry()
@triton.jit
def _mlml_validate_kernel(
    target_ptr,
    target_lengths_ptr,
    invalid_ptr,
    n_classes,
    BLOCK: tl.constexpr,
):
    """One program per row: validate every target entry and find the active
    prefix length (index of the first ``-1`` sentinel).  A single loop carries
    the ``tl.sum`` / ``tl.min`` reduces (XPU-legal)."""
    row = tl.program_id(0)
    row_base = row * n_classes

    row_stop = n_classes
    row_invalid = tl.zeros((), dtype=tl.int32)
    for target_start in range(0, n_classes, BLOCK):
        offs = target_start + tl.arange(0, BLOCK)
        mask = offs < n_classes
        tids = tl.load(target_ptr + row_base + offs, mask=mask, other=-1)
        valid = (tids == -1) | ((tids >= 0) & (tids < n_classes))
        row_invalid += tl.sum((mask & ~valid).to(tl.int32), axis=0)
        sentinel = tl.where(mask & (tids == -1), offs, n_classes)
        block_stop = tl.min(sentinel, axis=0)
        row_stop = tl.where(block_stop < row_stop, block_stop, row_stop)

    tl.store(target_lengths_ptr + row, row_stop)
    if row_invalid != 0:
        tl.atomic_add(invalid_ptr, 1)


@libentry()
@triton.jit
def _mlml_build_is_target_kernel(
    target_ptr,
    target_lengths_ptr,
    is_target_ptr,
    n_classes,
    class_tiles,
    BLOCK_C: tl.constexpr,
):
    """One program per (row, class-tile).  ``is_target[j] = 1`` iff class ``j``
    equals one of the active target ids.  Built by direct comparison over a 1D
    class vector -- no scatter and no masked atomic (both unreliable here)."""
    pid = tl.program_id(0)
    row = pid // class_tiles
    class_tile = pid - row * class_tiles
    row_base = row * n_classes

    class_off = class_tile * BLOCK_C + tl.arange(0, BLOCK_C)
    class_mask = class_off < n_classes
    target_length = tl.load(target_lengths_ptr + row)

    marked = tl.zeros((BLOCK_C,), dtype=tl.int32)
    for ti in range(0, n_classes):
        use = ti < target_length
        tid = tl.load(target_ptr + row_base + ti)
        valid_id = (tid >= 0) & (tid < n_classes)
        match = class_mask & use & valid_id & (class_off == tid)
        marked = tl.where(match, 1, marked)

    tl.store(
        is_target_ptr + row_base + class_off,
        tl.where(marked != 0, 1.0, 0.0),
        mask=class_mask,
    )


@libentry()
@triton.jit
def _mlml_row_loss_kernel(
    input_ptr,
    target_ptr,
    is_target_ptr,
    target_lengths_ptr,
    row_loss_ptr,
    n_classes,
    BLOCK_C: tl.constexpr,
):
    """One program per row, full class vector held in registers (C small).

    Per-target contributions are added into a 1D lane accumulator; a single
    ``tl.sum(..., axis=0)`` is taken once at the end.  Taking the reduce inside
    the target loop miscompiles on this backend."""
    row = tl.program_id(0)
    row_base = row * n_classes
    target_length = tl.load(target_lengths_ptr + row)

    class_off = tl.arange(0, BLOCK_C)
    class_mask = class_off < n_classes
    class_values = tl.load(
        input_ptr + row_base + class_off, mask=class_mask, other=0.0
    ).to(tl.float32)
    class_is_target = tl.load(
        is_target_ptr + row_base + class_off, mask=class_mask, other=0.0
    )
    non_target = class_mask & (class_is_target == 0.0)

    lane_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for ti in range(0, n_classes):
        use = ti < target_length
        tid = tl.load(target_ptr + row_base + ti)
        valid_id = (tid >= 0) & (tid < n_classes)
        use = use & valid_id
        safe_id = tl.where(use, tid, 0)
        tval = tl.load(input_ptr + row_base + safe_id, mask=use, other=0.0).to(
            tl.float32
        )
        margins = 1.0 - tval + class_values
        contrib = tl.where(non_target & use & (margins > 0.0), margins, 0.0)
        lane_acc += contrib

    acc = tl.sum(lane_acc, axis=0)
    tl.store(row_loss_ptr + row, acc / n_classes)


@libentry()
@triton.jit
def _mlml_accum_tile_kernel(
    input_ptr,
    target_ptr,
    is_target_ptr,
    target_lengths_ptr,
    row_acc_ptr,
    n_classes,
    class_tile,
    BLOCK_C: tl.constexpr,
):
    """Large-C path: one program per row (grid=(n_rows,)); the class-tile index
    is supplied by the host loop.  This tile's contribution is summed into a 1D
    lane accumulator (single ``tl.sum`` at the end) and read-modify-write
    accumulated into ``row_acc_ptr[row]``.

    One program per row is the only structure this backend compiles correctly
    for the per-target loss loop: a multi-program-per-row grid
    (``grid=(n_rows*class_tiles,)`` with ``row = pid // class_tiles``)
    miscompiles the accumulation once the program count grows -- empty rows get
    a spurious non-zero loss.  Hoisting the class-tile iteration onto the host
    keeps the grid at exactly ``n_rows`` programs."""
    row = tl.program_id(0)
    row_base = row * n_classes
    target_length = tl.load(target_lengths_ptr + row)

    class_off = class_tile * BLOCK_C + tl.arange(0, BLOCK_C)
    class_mask = class_off < n_classes
    class_values = tl.load(
        input_ptr + row_base + class_off, mask=class_mask, other=0.0
    ).to(tl.float32)
    class_is_target = tl.load(
        is_target_ptr + row_base + class_off, mask=class_mask, other=0.0
    )
    non_target = class_mask & (class_is_target == 0.0)

    lane_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for ti in range(0, n_classes):
        use = ti < target_length
        tid = tl.load(target_ptr + row_base + ti)
        valid_id = (tid >= 0) & (tid < n_classes)
        use = use & valid_id
        safe_id = tl.where(use, tid, 0)
        tval = tl.load(input_ptr + row_base + safe_id, mask=use, other=0.0).to(
            tl.float32
        )
        margins = 1.0 - tval + class_values
        contrib = tl.where(non_target & use & (margins > 0.0), margins, 0.0)
        lane_acc += contrib

    acc = tl.sum(lane_acc, axis=0)
    old = tl.load(row_acc_ptr + row)
    tl.store(row_acc_ptr + row, old + acc)


@libentry()
@triton.jit
def _mlml_finalize_rows_kernel(
    row_acc_ptr,
    row_loss_ptr,
    n_rows,
    n_classes,
    BLOCK: tl.constexpr,
):
    """Divide each row's accumulated raw loss by ``n_classes`` and cast into the
    (possibly lower-precision) per-row loss buffer."""
    offs = tl.arange(0, BLOCK)
    mask = offs < n_rows
    vals = tl.load(row_acc_ptr + offs, mask=mask, other=0.0)
    tl.store(row_loss_ptr + offs, vals / n_classes, mask=mask)


@libentry()
@triton.jit
def _mlml_reduce_rows_kernel(
    row_loss_ptr,
    output_ptr,
    n_rows,
    divisor,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    mask = offs < n_rows
    vals = tl.load(row_loss_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(vals, axis=0)
    tl.store(output_ptr, total / divisor)


def multilabel_margin_loss_forward(
    input: torch.Tensor, target: torch.Tensor, reduction=1
):
    logger.debug("GEMS_KUNLUNXIN MULTILABEL_MARGIN_LOSS_FORWARD")
    reduction = _normalize_reduction(reduction)
    n_rows, n_classes = _check_inputs(input, target)

    if n_rows == 0:
        return _empty_batch_result(input, n_classes, reduction)

    input_c = input.contiguous()
    target_c = target.contiguous()
    is_target = torch.empty(input.shape, dtype=input.dtype, device=input.device)
    target_lengths = torch.empty((n_rows,), dtype=torch.int64, device=input.device)
    invalid = torch.zeros((), dtype=torch.int32, device=input.device)

    scalar_or_vector = input.ndim <= 1
    reduction_divisor = float(n_rows) if reduction == 1 else 1.0

    small_c = n_classes <= _FUSED_CLASS_LIMIT
    if small_c:
        block_c = triton.next_power_of_2(n_classes)
        class_tiles = 1
    else:
        block_c = _LARGE_C_BLOCK
        class_tiles = triton.cdiv(n_classes, _LARGE_C_BLOCK)

    with torch_device_fn.device(input.device):
        _mlml_validate_kernel[(n_rows,)](
            target_c,
            target_lengths,
            invalid,
            n_classes,
            BLOCK=_VALIDATION_BLOCK,
        )

        if int(invalid.item()) != 0:
            raise RuntimeError(
                "multilabel_margin_loss_forward: target values must be -1 or in [0, C)"
            )

        _mlml_build_is_target_kernel[(n_rows * class_tiles,)](
            target_c,
            target_lengths,
            is_target,
            n_classes,
            class_tiles,
            BLOCK_C=block_c,
        )

        if scalar_or_vector or reduction == 0:
            loss = torch.empty(
                () if scalar_or_vector else (n_rows,),
                dtype=input.dtype,
                device=input.device,
            )
            row_loss = loss
        else:
            loss = torch.empty((), dtype=input.dtype, device=input.device)
            row_loss = torch.empty((n_rows,), dtype=torch.float32, device=input.device)

        if small_c:
            _mlml_row_loss_kernel[(n_rows,)](
                input_c,
                target_c,
                is_target,
                target_lengths,
                row_loss,
                n_classes,
                BLOCK_C=block_c,
            )
        else:
            row_acc = torch.zeros((n_rows,), dtype=torch.float32, device=input.device)
            for class_tile in range(class_tiles):
                _mlml_accum_tile_kernel[(n_rows,)](
                    input_c,
                    target_c,
                    is_target,
                    target_lengths,
                    row_acc,
                    n_classes,
                    class_tile,
                    BLOCK_C=_LARGE_C_BLOCK,
                )
            _mlml_finalize_rows_kernel[(1,)](
                row_acc,
                row_loss,
                n_rows,
                n_classes,
                BLOCK=triton.next_power_of_2(n_rows),
            )

        if not scalar_or_vector and reduction != 0:
            _mlml_reduce_rows_kernel[(1,)](
                row_loss,
                loss,
                n_rows,
                reduction_divisor,
                BLOCK=triton.next_power_of_2(n_rows),
            )

    return loss, is_target


_generic_ops_module = _sys.modules.get("flag_gems.ops")
if _generic_ops_module is not None:
    setattr(
        _generic_ops_module,
        "multilabel_margin_loss_forward",
        multilabel_margin_loss_forward,
    )
