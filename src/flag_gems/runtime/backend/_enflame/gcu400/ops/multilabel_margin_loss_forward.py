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

from flag_gems.runtime import device as runtime_device
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


_VALIDATION_BLOCK = 256
_LOSS_CLASS_BLOCK = 128
_LOSS_TARGET_BLOCK = 8
_FUSED_CLASS_LIMIT = 256
_ILUVATAR_LOSS_TARGET_BLOCK = 32
_ILUVATAR_VECTOR_VALIDATION_MAX_ELEMENTS = 8192
_MAX_ROW_PROGRAMS = 1024
_MAX_LOSS_PROGRAMS = 4096
_ASCEND_DEFAULT_VECTOR_CORES = 40
_DEFAULT_PROGRAM_LIMITS = (_MAX_ROW_PROGRAMS, _MAX_LOSS_PROGRAMS)
_REDUCTION_MAP = {"none": 0, "mean": 1, "sum": 2}
_SUPPORTED_INPUT_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)
_IS_ASCEND = runtime_device.vendor_name == "ascend"
_IS_ILUVATAR = runtime_device.vendor_name == "iluvatar"
_IS_ENFLAME = runtime_device.vendor_name == "enflame"
# Tile width along target. With BLOCK_T=8 a (BLOCK_C=128, BLOCK_T=8) tile has only
# 1024 elements and the target loop iterates C/8 times. L600 sweep (BLOCK_T 8/16/32/64/128 x
# BLOCK_C 128/256, core shapes total latency): 5.64 / 4.20 / 3.65 / 3.48 / 3.50 ms,
# 32~128 is already a plateau, so take the minimum 64 as default.
_ENFLAME_LOSS_TARGET_BLOCK = 64
_BACKEND_LOSS_TARGET_BLOCK = (
    _ILUVATAR_LOSS_TARGET_BLOCK
    if _IS_ILUVATAR
    else (_ENFLAME_LOSS_TARGET_BLOCK if _IS_ENFLAME else _LOSS_TARGET_BLOCK)
)
# Return the maximum target length so the host can truncate target-direction tiles / loops to
# ``max_target_length``: all pairs with ``target_offsets >= target_length`` are
# masked by ``active_target`` and contribute exactly 0, so traversing them is pure waste. enflame uses the same
# `_reduce_invalid_flags_kernel`, with only the extra cost of reading n_rows int64s, so it is
# enabled here too (previously only ascend enabled it, forcing enflame to traverse the full C).
_RETURN_MAX_TARGET_LENGTH = _IS_ASCEND or _IS_ENFLAME
_INVALID_RESULT_DTYPE = torch.int64 if _RETURN_MAX_TARGET_LENGTH else torch.int32
_ascend_vector_cores = None


def _get_program_limits():
    if not _IS_ASCEND:
        return _DEFAULT_PROGRAM_LIMITS

    global _ascend_vector_cores
    if _ascend_vector_cores is None:
        try:
            from triton.runtime import driver

            properties = driver.active.utils.get_device_properties(
                torch_device_fn.current_device()
            )
            _ascend_vector_cores = int(
                properties.get("num_vectorcore", _ASCEND_DEFAULT_VECTOR_CORES)
            )
        except Exception:
            _ascend_vector_cores = _ASCEND_DEFAULT_VECTOR_CORES
        if _ascend_vector_cores <= 0:
            _ascend_vector_cores = _ASCEND_DEFAULT_VECTOR_CORES
    return _ascend_vector_cores, _ascend_vector_cores


@libentry()
@triton.jit
def _build_is_target_kernel(
    target_ptr,
    is_target_ptr,
    target_lengths_ptr,
    invalid_flags_ptr,
    n_rows,
    n_classes,
    BLOCK: tl.constexpr,
):
    """Validate every target value and build the active-prefix membership mask.

    ``is_target_ptr`` is both output and scratch: `_fill_zero_kernel` already cleared it before the call,
    so only 1s are written at hit positions, with no separate int32 membership staging buffer.
    """
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    program_invalid = tl.zeros((), dtype=tl.int32)
    rows_per_program = (n_rows + n_programs - 1) // n_programs
    row_start = pid * rows_per_program
    row_end = tl.minimum(row_start + rows_per_program, n_rows)

    for row in range(row_start, row_end):
        row_base = row * n_classes

        # CPU checks the global target min/max, including entries after the
        # first sentinel. Do the equivalent validation here while separately
        # finding the first -1 that terminates the computational prefix.
        row_stop = n_classes
        row_invalid = tl.zeros((), dtype=tl.int32)
        for target_start in range(0, n_classes, BLOCK):
            target_offsets = target_start + tl.arange(0, BLOCK)
            target_mask = target_offsets < n_classes
            target_ids = tl.load(
                target_ptr + row_base + target_offsets,
                mask=target_mask,
                other=-1,
            )
            valid = (target_ids == -1) | ((target_ids >= 0) & (target_ids < n_classes))
            row_invalid += tl.sum((target_mask & ~valid).to(tl.int32), axis=0)

            sentinel_offsets = tl.where(
                target_mask & (target_ids == -1), target_offsets, n_classes
            )
            block_stop = tl.min(sentinel_offsets, axis=0)
            row_stop = tl.where(block_stop < row_stop, block_stop, row_stop)

        tl.store(target_lengths_ptr + row, row_stop)
        program_invalid += tl.where(row_invalid != 0, 1, 0)

        # A second pass performs only safe stores. Invalid IDs before the
        # sentinel are masked and replaced with zero before pointer arithmetic;
        # entries after the sentinel are ignored. Duplicate IDs intentionally
        # store the same value and remain duplicated in the later loss pass.
        for target_start in range(0, n_classes, BLOCK):
            target_offsets = target_start + tl.arange(0, BLOCK)
            target_mask = target_offsets < n_classes
            target_ids = tl.load(
                target_ptr + row_base + target_offsets,
                mask=target_mask,
                other=-1,
            )
            valid_id = (target_ids >= 0) & (target_ids < n_classes)
            active = target_mask & (target_offsets < row_stop) & valid_id
            safe_ids = tl.where(active, target_ids, 0)
            # The scratch is cleared by a preceding kernel. Overwriting with a
            # constant keeps duplicate IDs deterministic across target blocks;
            # only the final nonzero state matters, not the returned count.
            # (No atomics on GCU: see knowledge coding_rules.)
            tl.store(is_target_ptr + row_base + safe_ids, 1, mask=active)

    tl.store(invalid_flags_ptr + pid, program_invalid != 0)


@libentry()
@triton.jit
def _build_is_target_vector_kernel(
    target_ptr,
    membership_ptr,
    target_lengths_ptr,
    invalid_result_ptr,
    loss_ptr,
    n_rows,
    n_classes,
    INIT_LOSS: tl.constexpr,
    RETURN_MAX_TARGET_LENGTH: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Vectorize bounded row/class validation in one Iluvatar program."""
    if INIT_LOSS:
        tl.store(loss_ptr, 0.0)

    row_offsets_1d = tl.arange(0, BLOCK_R)
    row_offsets = row_offsets_1d[:, None]
    class_offsets = tl.arange(0, BLOCK_C)[None, :]
    element_mask = (row_offsets < n_rows) & (class_offsets < n_classes)
    offsets = row_offsets * n_classes + class_offsets

    target_ids = tl.load(target_ptr + offsets, mask=element_mask, other=-1)

    valid = (target_ids == -1) | ((target_ids >= 0) & (target_ids < n_classes))
    invalid_per_row = tl.sum((element_mask & ~valid).to(tl.int32), axis=1)
    invalid_total = tl.sum(invalid_per_row, axis=0)

    sentinel_offsets = tl.where(
        element_mask & (target_ids == -1), class_offsets, n_classes
    )
    row_stop = tl.min(sentinel_offsets, axis=1)
    valid_rows = row_offsets_1d < n_rows
    if RETURN_MAX_TARGET_LENGTH:
        max_target_length = tl.max(tl.where(valid_rows, row_stop, 0), axis=0)
        validation_result = tl.where(invalid_total != 0, -1, max_target_length)
    else:
        validation_result = tl.where(invalid_total != 0, -1, 0)
    tl.store(invalid_result_ptr, validation_result)
    tl.store(
        target_lengths_ptr + row_offsets_1d,
        row_stop,
        mask=valid_rows,
    )

    valid_id = (target_ids >= 0) & (target_ids < n_classes)
    active = element_mask & (class_offsets < row_stop[:, None]) & valid_id
    safe_ids = tl.where(active, target_ids, 0)
    tl.atomic_add(
        membership_ptr + row_offsets * n_classes + safe_ids,
        1,
        mask=active,
    )


@libentry()
@triton.jit
def _fill_zero_kernel(
    output_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    """Zero an output buffer of any dtype (store implicitly converts by pointer type)."""
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    for block_start in range(pid * BLOCK, n_elements, n_programs * BLOCK):
        offsets = block_start + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        tl.store(output_ptr + offsets, 0, mask=mask)


@libentry()
@triton.jit
def _membership_to_output_kernel(
    membership_ptr,
    is_target_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    for block_start in range(pid * BLOCK, n_elements, n_programs * BLOCK):
        offsets = block_start + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        membership = tl.load(membership_ptr + offsets, mask=mask, other=0)
        tl.store(
            is_target_ptr + offsets,
            tl.where(membership != 0, 1.0, 0.0),
            mask=mask,
        )


@libentry()
@triton.jit
def _fused_row_loss_kernel(
    input_ptr,
    target_ptr,
    is_target_ptr,
    target_lengths_ptr,
    row_loss_ptr,
    n_rows,
    n_classes,
    divisor,
    target_bound,  # maximum valid target length across all rows; beyond it nothing participates in comparison
    ACC_FP64: tl.constexpr,
    ATOMIC_REDUCE: tl.constexpr,
    SKIP_EMPTY_CLASS_TILE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Compute complete rows when C is small enough for a bounded program."""
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    rows_per_program = (n_rows + n_programs - 1) // n_programs
    row_start = pid * rows_per_program
    row_end = tl.minimum(row_start + rows_per_program, n_rows)

    for row in range(row_start, row_end):
        row_base = row * n_classes
        target_length = tl.load(target_lengths_ptr + row)
        if ACC_FP64:
            row_acc = tl.zeros((), dtype=tl.float64)
        else:
            row_acc = tl.zeros((), dtype=tl.float32)

        for class_start in range(0, n_classes, BLOCK_C):
            class_offsets_1d = class_start + tl.arange(0, BLOCK_C)
            if SKIP_EMPTY_CLASS_TILE:
                class_offsets = class_offsets_1d
            else:
                class_offsets = class_offsets_1d[None, :]
            class_mask = class_offsets < n_classes
            class_values = tl.load(
                input_ptr + row_base + class_offsets,
                mask=class_mask,
                other=0.0,
            )
            class_is_target = tl.load(
                is_target_ptr + row_base + class_offsets,
                mask=class_mask,
                other=0.0,
            )
            if not ACC_FP64:
                class_values = class_values.to(tl.float32)
            # GCU P03 workaround: is_target is fp16/bf16 for half inputs and a
            # value comparison on such tiles lowers through an unrelocatable
            # cvt intrinsic (ld.lld ICALL). is_target is exactly {0.0, 1.0},
            # so an arithmetic weight is an exact, comparison-free stand-in.
            non_target_weight = 1.0 - class_is_target

            class_has_non_target = True
            if SKIP_EMPTY_CLASS_TILE:
                non_target = class_mask & (class_is_target == 0.0)
                class_has_non_target = tl.sum(non_target.to(tl.int32), axis=0) != 0

            if class_has_non_target:
                # ``target_bound`` is the global maximum valid target length; offsets beyond it
                # are padding for all rows and contribute exactly 0, so the loop can be safely truncated.
                for target_start in range(0, target_bound, BLOCK_T):
                    target_offsets = target_start + tl.arange(0, BLOCK_T)[:, None]
                    active_target = target_offsets < target_length
                    target_ids = tl.load(
                        target_ptr + row_base + target_offsets,
                        mask=active_target,
                        other=0,
                    )
                    valid_id = (target_ids >= 0) & (target_ids < n_classes)
                    safe_ids = tl.where(active_target & valid_id, target_ids, 0)
                    target_values = tl.load(
                        input_ptr + row_base + safe_ids,
                        mask=active_target & valid_id,
                        other=0.0,
                    )
                    if not ACC_FP64:
                        target_values = target_values.to(tl.float32)

                    pair_mask = active_target & valid_id & class_mask
                    margins = 1.0 - target_values + class_values
                    # Match the native `if (z > 0)` behavior: NaN margins do
                    # not contribute, unlike maximum on some backends. The
                    # non-target filter rides on the {0,1} weight, so padding
                    # lanes (loaded as 0.0 -> weight 1.0) are already masked
                    # out of the hinge by class_mask inside pair_mask.
                    contributions = (
                        tl.where(pair_mask & (margins > 0.0), margins, 0.0)
                        * non_target_weight
                    )
                    row_acc += tl.sum(tl.sum(contributions, axis=1), axis=0)

        row_value = row_acc / n_classes
        if ATOMIC_REDUCE:
            tl.atomic_add(row_loss_ptr, row_value / divisor)
        else:
            tl.store(row_loss_ptr + row, row_value)


@libentry()
@triton.jit
def _tiled_loss_kernel(
    input_ptr,
    target_ptr,
    is_target_ptr,
    target_lengths_ptr,
    partial_ptr,
    n_classes,
    class_tiles,
    target_tiles,
    total_tiles,
    ACC_FP64: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Bound each program to one target/class tile for large-C watchdog safety."""
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    tiles_per_row = class_tiles * target_tiles
    tiles_per_program = (total_tiles + n_programs - 1) // n_programs
    tile_start = pid * tiles_per_program
    tile_end = tl.minimum(tile_start + tiles_per_program, total_tiles)

    for tile in range(tile_start, tile_end):
        row = tile // tiles_per_row
        tile_in_row = tile - row * tiles_per_row
        class_tile = tile_in_row // target_tiles
        target_tile = tile_in_row - class_tile * target_tiles
        row_base = row * n_classes

        class_offsets = class_tile * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
        target_offsets = target_tile * BLOCK_T + tl.arange(0, BLOCK_T)[:, None]
        class_mask = class_offsets < n_classes
        target_length = tl.load(target_lengths_ptr + row)
        active_target = target_offsets < target_length

        # GCU P03 workaround (same as _fused_row_loss_kernel): the half-dtype
        # value comparison lowers through an unrelocatable cvt intrinsic, so
        # the non-target filter rides on the exact {0,1} arithmetic weight.
        # Padding lanes load as 1.0, i.e. weight 0; they are excluded by
        # class_mask inside pair_mask anyway, so this stays exact and lets the
        # same weight double as the empty-class-tile test below.
        class_is_target = tl.load(
            is_target_ptr + row_base + class_offsets,
            mask=class_mask,
            other=1.0,
        )
        non_target_weight = 1.0 - class_is_target

        if ACC_FP64:
            partial = tl.zeros((), dtype=tl.float64)
        else:
            partial = tl.zeros((), dtype=tl.float32)

        # Two kinds of tiles contribute exactly 0, so exiting early skips a whole (BLOCK_C x BLOCK_T) operation:
        # 1) the entire target tile lies outside target_length (active_target all false);
        # 2) all classes in the class tile are targets (non_target_weight all 0),
        #    e.g. the class tile for the first half of benchmark input target=arange(C).
        tile_non_target_weight = tl.sum(tl.sum(non_target_weight, axis=1), axis=0)
        if (target_tile * BLOCK_T < target_length) and (tile_non_target_weight != 0.0):
            target_ids = tl.load(
                target_ptr + row_base + target_offsets,
                mask=active_target,
                other=0,
            )
            valid_id = (target_ids >= 0) & (target_ids < n_classes)
            safe_ids = tl.where(active_target & valid_id, target_ids, 0)
            target_values = tl.load(
                input_ptr + row_base + safe_ids,
                mask=active_target & valid_id,
                other=0.0,
            )
            class_values = tl.load(
                input_ptr + row_base + class_offsets,
                mask=class_mask,
                other=0.0,
            )
            if not ACC_FP64:
                target_values = target_values.to(tl.float32)
                class_values = class_values.to(tl.float32)

            pair_mask = active_target & valid_id & class_mask
            margins = 1.0 - target_values + class_values
            contributions = (
                tl.where(pair_mask & (margins > 0.0), margins, 0.0) * non_target_weight
            )
            partial += tl.sum(tl.sum(contributions, axis=1), axis=0)
        tl.store(partial_ptr + tile, partial)


@libentry()
@triton.jit
def _class_tiled_loss_kernel(
    input_ptr,
    target_ptr,
    is_target_ptr,
    target_lengths_ptr,
    partial_ptr,
    n_classes,
    class_tiles,
    total_tiles,
    ACC_FP64: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Accumulate one partial per class tile, skipping empty comparison tiles."""
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    tiles_per_program = (total_tiles + n_programs - 1) // n_programs
    tile_start = pid * tiles_per_program
    tile_end = tl.minimum(tile_start + tiles_per_program, total_tiles)

    for tile in range(tile_start, tile_end):
        row = tile // class_tiles
        class_tile = tile - row * class_tiles
        row_base = row * n_classes
        class_offsets = class_tile * BLOCK_C + tl.arange(0, BLOCK_C)
        class_mask = class_offsets < n_classes
        class_is_target = tl.load(
            is_target_ptr + row_base + class_offsets,
            mask=class_mask,
            other=0.0,
        )
        non_target = class_mask & (class_is_target == 0.0)

        if ACC_FP64:
            partial = tl.zeros((), dtype=tl.float64)
        else:
            partial = tl.zeros((), dtype=tl.float32)

        if tl.sum(non_target.to(tl.int32), axis=0) != 0:
            class_values = tl.load(
                input_ptr + row_base + class_offsets,
                mask=class_mask,
                other=0.0,
            )
            if not ACC_FP64:
                class_values = class_values.to(tl.float32)
            target_length = tl.load(target_lengths_ptr + row)

            for target_start in range(0, n_classes, BLOCK_T):
                target_offsets = target_start + tl.arange(0, BLOCK_T)[:, None]
                active_target = target_offsets < target_length
                target_ids = tl.load(
                    target_ptr + row_base + target_offsets,
                    mask=active_target,
                    other=0,
                )
                valid_id = (target_ids >= 0) & (target_ids < n_classes)
                safe_ids = tl.where(active_target & valid_id, target_ids, 0)
                target_values = tl.load(
                    input_ptr + row_base + safe_ids,
                    mask=active_target & valid_id,
                    other=0.0,
                )
                if not ACC_FP64:
                    target_values = target_values.to(tl.float32)

                pair_mask = active_target & valid_id & non_target
                margins = 1.0 - target_values + class_values
                contributions = tl.where(pair_mask & (margins > 0.0), margins, 0.0)
                partial += tl.sum(tl.sum(contributions, axis=1), axis=0)

        tl.store(partial_ptr + tile, partial)


@libentry()
@triton.jit
def _reduce_tiled_rows_kernel(
    partial_ptr,
    row_loss_ptr,
    n_rows,
    tiles_per_row,
    n_classes,
    divisor,
    ACC_FP64: tl.constexpr,
    ATOMIC_REDUCE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    rows_per_program = (n_rows + n_programs - 1) // n_programs
    row_start = pid * rows_per_program
    row_end = tl.minimum(row_start + rows_per_program, n_rows)

    for row in range(row_start, row_end):
        if ACC_FP64:
            row_acc = tl.zeros((), dtype=tl.float64)
        else:
            row_acc = tl.zeros((), dtype=tl.float32)
        for tile_start in range(0, tiles_per_row, BLOCK):
            tile_offsets = tile_start + tl.arange(0, BLOCK)
            tile_mask = tile_offsets < tiles_per_row
            values = tl.load(
                partial_ptr + row * tiles_per_row + tile_offsets,
                mask=tile_mask,
                other=0.0,
            )
            if not ACC_FP64:
                values = values.to(tl.float32)
            row_acc += tl.sum(values, axis=0)
        row_value = row_acc / n_classes
        if ATOMIC_REDUCE:
            tl.atomic_add(row_loss_ptr, row_value / divisor)
        else:
            tl.store(row_loss_ptr + row, row_value)


@libentry()
@triton.jit
def _reduce_rows_kernel(
    row_loss_ptr,
    output_ptr,
    n_rows,
    divisor,
    ACC_FP64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if ACC_FP64:
        total = tl.zeros((), dtype=tl.float64)
    else:
        total = tl.zeros((), dtype=tl.float32)

    for row_start in range(0, n_rows, BLOCK):
        row_offsets = row_start + tl.arange(0, BLOCK)
        row_mask = row_offsets < n_rows
        values = tl.load(row_loss_ptr + row_offsets, mask=row_mask, other=0.0)
        if not ACC_FP64:
            values = values.to(tl.float32)
        total += tl.sum(values, axis=0)
    tl.store(output_ptr, total / divisor)


@libentry()
@triton.jit
def _reduce_invalid_flags_kernel(
    invalid_flags_ptr,
    target_lengths_ptr,
    invalid_result_ptr,
    n_flags,
    n_rows,
    RETURN_MAX_TARGET_LENGTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_flags
    flags = tl.load(invalid_flags_ptr + offsets, mask=mask, other=0)
    invalid_total = tl.sum(flags, axis=0)

    if RETURN_MAX_TARGET_LENGTH:
        max_target_length = tl.zeros((), dtype=tl.int64)
        for row_start in range(0, n_rows, BLOCK):
            row_offsets = row_start + tl.arange(0, BLOCK)
            row_mask = row_offsets < n_rows
            lengths = tl.load(
                target_lengths_ptr + row_offsets,
                mask=row_mask,
                other=0,
            )
            max_target_length = tl.maximum(
                max_target_length,
                tl.max(lengths, axis=0),
            )
        validation_result = tl.where(invalid_total != 0, -1, max_target_length)
    else:
        validation_result = tl.where(invalid_total != 0, -1, 0)

    tl.store(invalid_result_ptr, validation_result)


def _normalize_reduction(reduction):
    if isinstance(reduction, str):
        normalized = _REDUCTION_MAP.get(reduction.lower())
        if normalized is not None:
            return normalized
    elif isinstance(reduction, int) and reduction in (0, 1, 2):
        return reduction
    raise ValueError(
        "multilabel_margin_loss_forward: reduction must be none/mean/sum or 0/1/2"
    )


def _check_inputs(input, target):
    if input.ndim not in (0, 1, 2):
        raise RuntimeError(
            "multilabel_margin_loss_forward: input must be scalar, 1D, or 2D"
        )
    if input.shape != target.shape:
        raise RuntimeError(
            "multilabel_margin_loss_forward: target shape must exactly match input shape"
        )
    if input.device != target.device:
        raise RuntimeError(
            "multilabel_margin_loss_forward: input and target must be on the same device"
        )
    if target.dtype != torch.int64:
        raise RuntimeError(
            "multilabel_margin_loss_forward: target must have dtype torch.int64"
        )
    if input.dtype not in _SUPPORTED_INPUT_DTYPES:
        raise RuntimeError(
            "multilabel_margin_loss_forward: input must have a floating-point loss dtype"
        )

    n_rows = input.shape[0] if input.ndim == 2 else 1
    n_classes = input.shape[-1] if input.ndim != 0 else 1
    if n_classes == 0:
        raise RuntimeError(
            "multilabel_margin_loss_forward: class dimension must be non-empty"
        )
    return n_rows, n_classes


def _empty_batch_result(input, n_classes, reduction):
    is_target = torch.empty(input.shape, dtype=input.dtype, device=input.device)
    if reduction == 0:
        loss = torch.empty((0,), dtype=input.dtype, device=input.device)
    elif reduction == 2:
        loss = torch.zeros((), dtype=input.dtype, device=input.device)
    else:
        loss = torch.full((), float("nan"), dtype=input.dtype, device=input.device)
    return loss, is_target


def multilabel_margin_loss_forward(
    input: torch.Tensor, target: torch.Tensor, reduction=1
):
    logger.debug("GEMS_ENFLAME_GCU400 MULTILABEL_MARGIN_LOSS_FORWARD")
    reduction = _normalize_reduction(reduction)
    n_rows, n_classes = _check_inputs(input, target)

    if n_rows == 0:
        return _empty_batch_result(input, n_classes, reduction)

    input_contiguous = input.contiguous()
    target_contiguous = target.contiguous()
    is_target = torch.empty(input.shape, dtype=input.dtype, device=input.device)
    target_lengths = torch.empty((n_rows,), dtype=torch.int64, device=input.device)
    row_program_limit, loss_program_limit = _get_program_limits()
    validation_grid = min(n_rows, row_program_limit)
    validation_block_r = triton.next_power_of_2(n_rows)
    validation_block_c = triton.next_power_of_2(n_classes)
    use_vector_validation = (
        _IS_ILUVATAR
        and validation_block_r * validation_block_c
        <= _ILUVATAR_VECTOR_VALIDATION_MAX_ELEMENTS
    )
    acc_dtype = torch.float64 if input.dtype == torch.float64 else torch.float32
    acc_fp64 = input.dtype == torch.float64

    scalar_or_vector = input.ndim <= 1
    atomic_row_reduce = (
        use_vector_validation
        and input.dtype == torch.float32
        and not scalar_or_vector
        and reduction != 0
    )
    skip_empty_class_tile = _IS_ASCEND and reduction == 2
    row_reduce_block = min(256, triton.next_power_of_2(n_rows)) if _IS_ILUVATAR else 256
    reduction_divisor = float(n_rows) if reduction == 1 else 1.0
    if scalar_or_vector or reduction == 0:
        loss = torch.empty(
            () if scalar_or_vector else (n_rows,),
            dtype=input.dtype,
            device=input.device,
        )
        row_loss = loss
    elif atomic_row_reduce:
        loss = torch.empty((), dtype=input.dtype, device=input.device)
        row_loss = loss
    else:
        loss = torch.empty((), dtype=input.dtype, device=input.device)
        row_loss = torch.empty((n_rows,), dtype=acc_dtype, device=input.device)

    with torch_device_fn.device(input.device):
        invalid_result = torch.empty(
            (),
            dtype=_INVALID_RESULT_DTYPE,
            device=input.device,
        )
        membership_grid = min(
            triton.cdiv(n_rows * n_classes, _VALIDATION_BLOCK),
            loss_program_limit,
        )
        if use_vector_validation:
            membership = torch.empty(
                (n_rows * n_classes,), dtype=torch.int32, device=input.device
            )
            _fill_zero_kernel[(membership_grid,)](
                membership,
                n_rows * n_classes,
                BLOCK=_VALIDATION_BLOCK,
            )
            _build_is_target_vector_kernel[(1,)](
                target_contiguous,
                membership,
                target_lengths,
                invalid_result,
                loss,
                n_rows,
                n_classes,
                INIT_LOSS=atomic_row_reduce,
                RETURN_MAX_TARGET_LENGTH=_RETURN_MAX_TARGET_LENGTH,
                BLOCK_R=validation_block_r,
                BLOCK_C=validation_block_c,
            )
        else:
            invalid_flags = torch.empty(
                (validation_grid,), dtype=torch.int32, device=input.device
            )
            # Use is_target directly as scratch: clear the whole block first, then let the validation kernel
            # write 1s at hit positions. This saves an N*C int32 staging buffer, one read/write of it
            # and one extra kernel launch (previously clear + build + convert, three launches).
            _fill_zero_kernel[(membership_grid,)](
                is_target,
                n_rows * n_classes,
                BLOCK=_VALIDATION_BLOCK,
            )
            _build_is_target_kernel[(validation_grid,)](
                target_contiguous,
                is_target,
                target_lengths,
                invalid_flags,
                n_rows,
                n_classes,
                BLOCK=_VALIDATION_BLOCK,
            )

            invalid_block = triton.next_power_of_2(validation_grid)
            _reduce_invalid_flags_kernel[(1,)](
                invalid_flags,
                target_lengths,
                invalid_result,
                validation_grid,
                n_rows,
                RETURN_MAX_TARGET_LENGTH=_RETURN_MAX_TARGET_LENGTH,
                BLOCK=invalid_block,
            )

        # Synchronizing one scalar is deliberate: invalid targets must raise
        # before any loss kernel is allowed to execute.
        max_target_length = int(invalid_result.item())
        if max_target_length < 0:
            raise RuntimeError(
                "multilabel_margin_loss_forward: target values must be -1 or in [0, C)"
            )

        if use_vector_validation:
            _membership_to_output_kernel[(membership_grid,)](
                membership,
                is_target,
                n_rows * n_classes,
                BLOCK=_VALIDATION_BLOCK,
            )

        if n_classes <= _FUSED_CLASS_LIMIT:
            loss_grid = min(n_rows, row_program_limit)
            _fused_row_loss_kernel[(loss_grid,)](
                input_contiguous,
                target_contiguous,
                is_target,
                target_lengths,
                row_loss,
                n_rows,
                n_classes,
                reduction_divisor,
                max_target_length,
                ACC_FP64=acc_fp64,
                ATOMIC_REDUCE=atomic_row_reduce,
                SKIP_EMPTY_CLASS_TILE=skip_empty_class_tile,
                BLOCK_C=_LOSS_CLASS_BLOCK,
                BLOCK_T=_BACKEND_LOSS_TARGET_BLOCK,
            )
        else:
            class_tiles = triton.cdiv(n_classes, _LOSS_CLASS_BLOCK)
            full_target_tiles = triton.cdiv(n_classes, _BACKEND_LOSS_TARGET_BLOCK)
            # Truncate the number of target-direction tiles exactly by the global maximum valid target length: any
            # pair with ``target_offsets >= target_length`` is
            # masked by active_target and contributes exactly 0, so traversing it is pure waste. Benchmark inputs have
            # target_length = C/2, so this alone removes half the target tiles.
            active_target_tiles = triton.cdiv(
                max_target_length, _BACKEND_LOSS_TARGET_BLOCK
            )
            target_tiles = (
                min(full_target_tiles, active_target_tiles)
                if active_target_tiles > 0
                else 1
            )
            use_class_tiled_loss = _IS_ASCEND and reduction == 2
            if use_class_tiled_loss:
                tiles_per_row = class_tiles
            else:
                tiles_per_row = class_tiles * target_tiles
            total_tiles = n_rows * tiles_per_row
            partial = torch.empty((total_tiles,), dtype=acc_dtype, device=input.device)
            tiled_grid = min(total_tiles, loss_program_limit)
            if use_class_tiled_loss:
                _class_tiled_loss_kernel[(tiled_grid,)](
                    input_contiguous,
                    target_contiguous,
                    is_target,
                    target_lengths,
                    partial,
                    n_classes,
                    class_tiles,
                    total_tiles,
                    ACC_FP64=acc_fp64,
                    BLOCK_C=_LOSS_CLASS_BLOCK,
                    BLOCK_T=_BACKEND_LOSS_TARGET_BLOCK,
                )
            else:
                _tiled_loss_kernel[(tiled_grid,)](
                    input_contiguous,
                    target_contiguous,
                    is_target,
                    target_lengths,
                    partial,
                    n_classes,
                    class_tiles,
                    target_tiles,
                    total_tiles,
                    ACC_FP64=acc_fp64,
                    BLOCK_C=_LOSS_CLASS_BLOCK,
                    BLOCK_T=_BACKEND_LOSS_TARGET_BLOCK,
                )
            row_grid = min(n_rows, row_program_limit)
            _reduce_tiled_rows_kernel[(row_grid,)](
                partial,
                row_loss,
                n_rows,
                tiles_per_row,
                n_classes,
                reduction_divisor,
                ACC_FP64=acc_fp64,
                ATOMIC_REDUCE=atomic_row_reduce,
                BLOCK=256,
            )

        if not scalar_or_vector and reduction != 0 and not atomic_row_reduce:
            _reduce_rows_kernel[(1,)](
                row_loss,
                loss,
                n_rows,
                reduction_divisor,
                ACC_FP64=acc_fp64,
                BLOCK=row_reduce_block,
            )

    return loss, is_target
