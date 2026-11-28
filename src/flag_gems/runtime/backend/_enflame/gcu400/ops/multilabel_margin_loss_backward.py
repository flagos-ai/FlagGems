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
"""GCU400 ``multilabel_margin_loss_backward``.

Same contract and maths as the generic Triton implementation, with three backend
differences.

* The GCU launch descriptor limits ``grid.x`` to 65535, while the generic kernel
  maps one program to one row and needs ``rows`` programs; the spec shapes reach
  204288 rows and that launch is rejected with ``OutOfResources: out of resource:
  grid.x``. Both kernels here walk rows with a grid-stride loop and cap the row
  axis of the grid.
* The generic kernel forms a ``BLOCK x BLOCK`` pair tile with a fixed
  ``BLOCK = 128``, so a 15-class shape spends 98% of its pair lanes on masked
  padding. Here the tile is never wider than the next power of two above the
  class count.
* When the whole class axis fits one tile, the per-row cross-lane reductions that
  decide whether a term can be skipped cost more than the masked outer product
  they would skip, so that path keeps both terms on.

Per row, with ``A`` the active target prefix (stopped at the first ``-1``) and
``NT`` the classes whose ``is_target`` entry is zero::

    g[a] = mult[a] * ( -#{d in NT : 1 - x[a] + x[d] > 0} )      # a in A
    g[d] += sum_{a in A} mult[a] * [1 - x[a] + x[d] > 0]        # d in NT

scaled by ``1/(C * n_mean)`` and by ``grad_output``. ``mult[a]`` is the number of
times class ``a`` appears in ``A``.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Hard limit of the GCU launch descriptor on the first grid dimension.
_MAX_GRID_X = 65535
# Programs on the row axis. Both kernels walk rows with a grid-stride loop, so a
# grid smaller than the row count amortises the fixed per-program cost over
# several rows instead of paying it once per row; 512 measured best on the spec
# shapes, 64 and 4096 were 15-20% slower.
_MAX_ROW_PROGRAMS = 512

_SUPPORTED_INPUT_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)

# Number of target entries compared against one class tile at a time; keeps the
# multiplicity comparison tile at BLOCK_T x BLOCK instead of C x C.
_BLOCK_T = 64
# Candidate widths of the class tile, widest first.
_GRAD_BLOCK_CANDIDATES = (128, 64, 32, 16)


@libentry()
@triton.jit
def _membership_kernel(
    target_ptr,
    mult_ptr,
    num_rows,
    num_classes,
    row_programs,
    target_row_stride,
    target_class_stride,
    BLOCK_T: tl.constexpr,
    BLOCK: tl.constexpr,
    SINGLE_TILE: tl.constexpr,
):
    """Multiplicity of every class id in the active target prefix of a row.

    The active prefix ends at the first negative id, so the prefix length is the
    minimum over the target tiles of the first negative offset. When the class
    axis fits one BLOCK_T tile that scan is a single masked reduction.

    The (row program, class tile) pair is decoded from a flat program id: the
    GCU launch descriptor caps the second grid dimension at 255, which a class
    axis of 4096 already exceeds with the narrowest candidate tile.
    """
    program = tl.program_id(0)
    row_program = program % row_programs
    tile = program // row_programs
    for row in range(row_program, num_rows, row_programs):
        target_row_base = row * target_row_stride
        if SINGLE_TILE:
            offsets = tl.arange(0, BLOCK_T)
            in_range = offsets < num_classes
            target_ids = tl.load(
                target_ptr + target_row_base + offsets * target_class_stride,
                mask=in_range,
                other=-1,
            )
            prefix_end = tl.min(
                tl.where(in_range & (target_ids < 0), offsets, num_classes), axis=0
            )
        else:
            prefix_end = num_classes
            for tile_start in range(0, num_classes, BLOCK_T):
                offsets = tile_start + tl.arange(0, BLOCK_T)
                in_range = offsets < num_classes
                target_ids = tl.load(
                    target_ptr + target_row_base + offsets * target_class_stride,
                    mask=in_range,
                    other=-1,
                )
                prefix_end = tl.minimum(
                    prefix_end,
                    tl.min(
                        tl.where(in_range & (target_ids < 0), offsets, num_classes),
                        axis=0,
                    ),
                )
        class_offsets = tile * BLOCK + tl.arange(0, BLOCK)
        in_range = class_offsets < num_classes
        multiplicity = tl.zeros((BLOCK,), dtype=tl.int32)
        for tile_start in range(0, prefix_end, BLOCK_T):
            entry_offsets = tile_start + tl.arange(0, BLOCK_T)
            entry_valid = entry_offsets < prefix_end
            # -2 never equals a class offset, so masked lanes never count.
            entry_ids = tl.load(
                target_ptr + target_row_base + entry_offsets * target_class_stride,
                mask=entry_valid,
                other=-2,
            )
            multiplicity += tl.sum(
                (
                    (entry_ids[:, None] == class_offsets[None, :])
                    & entry_valid[:, None]
                ).to(tl.int32),
                axis=0,
            )
        tl.store(
            mult_ptr + row * num_classes + class_offsets,
            multiplicity,
            mask=in_range,
        )


@libentry()
@triton.jit
def _grad_kernel(
    input_ptr,
    mult_ptr,
    is_target_ptr,
    grad_input_ptr,
    grad_output_ptr,
    num_rows,
    num_classes,
    row_programs,
    input_row_stride,
    input_class_stride,
    is_target_row_stride,
    is_target_class_stride,
    grad_output_stride,
    n_mean,
    ACC64: tl.constexpr,
    BLOCK: tl.constexpr,
    SINGLE_TILE: tl.constexpr,
):
    """d(loss)/dx for one (row, class tile) pair.

    One program owns a class tile and writes its values once, so no atomics are
    needed. Both gradient terms of a class are sums over the whole class axis,
    hence the other operand is streamed tile by tile; a single pass produces both
    terms.

    See ``_membership_kernel`` for why the pair is decoded from a flat id.
    """
    program = tl.program_id(0)
    row_program = program % row_programs
    tile = program // row_programs
    if ACC64:
        acc_ty: tl.constexpr = tl.float64
    else:
        acc_ty: tl.constexpr = tl.float32

    for row in range(row_program, num_rows, row_programs):
        input_row_base = row * input_row_stride
        is_target_row_base = row * is_target_row_stride
        row_base = row * num_classes
        factor = tl.load(grad_output_ptr + row * grad_output_stride).to(acc_ty)
        scale = 1.0 / (tl.full((), 1.0, dtype=acc_ty) * num_classes * n_mean)
        weight = scale * factor

        class_offsets = tile * BLOCK + tl.arange(0, BLOCK)
        in_range = class_offsets < num_classes
        multiplicity = tl.load(
            mult_ptr + row_base + class_offsets, mask=in_range, other=0
        )
        own_is_target = multiplicity > 0
        # other=1.0 marks padding lanes as targets so they never join NT.
        own_mask = tl.load(
            is_target_ptr + is_target_row_base + class_offsets * is_target_class_stride,
            mask=in_range,
            other=1.0,
        )
        own_non_target = in_range & (own_mask == 0.0)
        own_values = tl.load(
            input_ptr + input_row_base + class_offsets * input_class_stride,
            mask=in_range,
            other=0.0,
        )
        if ACC64:
            accumulator = tl.zeros((BLOCK,), dtype=tl.float64)
            own_multiplicity = multiplicity.to(tl.float64)
        else:
            accumulator = tl.zeros((BLOCK,), dtype=tl.float32)
            own_multiplicity = multiplicity.to(tl.float32)

        # A tile without active targets cannot contribute to the first term and
        # one without non-target classes nothing to the second; checking once
        # keeps the whole pair outer product out of the loop for those rows.
        # With a single class tile those per-row reductions cost more than the
        # masked outer product they would skip, so the fast path keeps both terms
        # on and lets the masks decide.
        if SINGLE_TILE:
            has_own_target = True
            has_own_non_target = True
        else:
            has_own_target = tl.sum(own_is_target.to(tl.int32), axis=0) != 0
            has_own_non_target = tl.sum(own_non_target.to(tl.int32), axis=0) != 0

        if has_own_target or has_own_non_target:
            for other_start in range(0, num_classes, BLOCK):
                other_offsets = other_start + tl.arange(0, BLOCK)
                other_in_range = other_offsets < num_classes
                other_multiplicity = tl.load(
                    mult_ptr + row_base + other_offsets, mask=other_in_range, other=0
                )
                other_mask = tl.load(
                    is_target_ptr
                    + is_target_row_base
                    + other_offsets * is_target_class_stride,
                    mask=other_in_range,
                    other=1.0,
                )
                other_non_target = other_in_range & (other_mask == 0.0)
                other_is_target = other_multiplicity > 0
                if SINGLE_TILE:
                    need_first = has_own_target
                    need_second = has_own_non_target
                else:
                    need_first = has_own_target and (
                        tl.sum(other_non_target.to(tl.int32), axis=0) != 0
                    )
                    need_second = has_own_non_target and (
                        tl.sum(other_is_target.to(tl.int32), axis=0) != 0
                    )
                if need_first or need_second:
                    other_values = tl.load(
                        input_ptr + input_row_base + other_offsets * input_class_stride,
                        mask=other_in_range,
                        other=0.0,
                    )
                    if need_first:
                        # Own class is the active target, the other class the
                        # non-target partner. The hinge is evaluated in the input
                        # element type, exactly as the native kernel.
                        one_own = tl.full((BLOCK,), 1.0, input_ptr.dtype.element_ty)
                        own_margin = (
                            (one_own - own_values)[:, None] + other_values[None, :]
                        ).to(input_ptr.dtype.element_ty)
                        own_as_target = (
                            (own_margin > 0.0)
                            & own_is_target[:, None]
                            & other_non_target[None, :]
                        )
                        accumulator -= tl.sum(
                            tl.where(own_as_target, own_multiplicity[:, None], 0.0),
                            axis=1,
                        )
                    if need_second:
                        # Mirrored set: the other class is the active target and
                        # the own class the non-target partner.
                        if ACC64:
                            other_weight = other_multiplicity.to(tl.float64)
                        else:
                            other_weight = other_multiplicity.to(tl.float32)
                        one_other = tl.full((BLOCK,), 1.0, input_ptr.dtype.element_ty)
                        other_margin = (
                            (one_other - other_values)[:, None] + own_values[None, :]
                        ).to(input_ptr.dtype.element_ty)
                        own_as_non_target = (
                            (other_margin > 0.0)
                            & other_is_target[:, None]
                            & own_non_target[None, :]
                        )
                        accumulator += tl.sum(
                            tl.where(own_as_non_target, other_weight[:, None], 0.0),
                            axis=0,
                        )

        tl.store(
            grad_input_ptr + row_base + class_offsets,
            accumulator * weight,
            mask=in_range,
        )


def _grad_block_for(rows, n_classes):
    """Class-tile width for the gradient grid.

    Both kernels form a BLOCK x BLOCK pair tile, so a tile wider than the class
    axis computes lanes that the mask then throws away: a 128-wide tile on a
    15-class shape wastes 98% of the pair lanes. The tile is therefore never
    wider than the next power of two above the class count.

    Among the widths that fit the class axis the widest always wins on GCU400.
    Narrowing the tile to raise the program count multiplies the number of
    BLOCK x BLOCK pair steps by (128 / BLOCK) ** 2, and the scalar loop
    bookkeeping per step then dominates the vector work: measured on the spec
    shapes, a 16-wide tile took 27.25 ms in total against 2.20 ms for a 128-wide
    tile (12.4x), while 256 is rejected with ``out of resource: dsm memory``.
    ``rows`` is unused now but kept in the signature so call sites stay stable.
    """
    widest_useful = max(16, triton.next_power_of_2(n_classes))
    return min(widest_useful, _GRAD_BLOCK_CANDIDATES[0])


def _strides2d(t):
    """(row_stride, class_stride) for a rank<=2 tensor viewed as (rows, classes)."""
    if t.dim() == 0:
        return 1, 1
    if t.dim() == 1:
        return 0, t.stride(0)
    return t.stride(0), t.stride(1)


def _row_col(t):
    if t.dim() == 0:
        return 1, 1
    if t.dim() == 1:
        return 1, int(t.shape[0])
    return int(t.shape[0]), int(t.shape[1])


def _check_inputs(grad_output, self, target, is_target):
    if self.dim() > 2:
        raise RuntimeError(
            "multilabel_margin_loss_backward: input must be scalar, 1D or 2D"
        )
    if tuple(self.shape) != tuple(target.shape):
        raise RuntimeError(
            "multilabel_margin_loss_backward: target shape must exactly match "
            "input shape"
        )
    if tuple(self.shape) != tuple(is_target.shape):
        raise RuntimeError(
            "multilabel_margin_loss_backward: is_target shape must exactly match "
            "input shape"
        )
    if self.device != target.device or self.device != is_target.device:
        raise RuntimeError(
            "multilabel_margin_loss_backward: input, target and is_target must be "
            "on the same device"
        )
    if grad_output.device != self.device:
        raise RuntimeError(
            "multilabel_margin_loss_backward: grad_output must be on the same "
            "device as input"
        )
    if target.dtype != torch.int64:
        raise RuntimeError(
            "multilabel_margin_loss_backward: target must have dtype torch.int64"
        )
    if self.dtype not in _SUPPORTED_INPUT_DTYPES:
        raise RuntimeError(
            "multilabel_margin_loss_backward: input must have a floating-point "
            "loss dtype"
        )
    if is_target.dtype != self.dtype:
        raise RuntimeError(
            "multilabel_margin_loss_backward: is_target must have the same dtype "
            "as input"
        )

    rows, n_classes = _row_col(self)
    if n_classes == 0:
        raise RuntimeError(
            "multilabel_margin_loss_backward: class dimension must be non-empty"
        )
    return rows, n_classes


def multilabel_margin_loss_backward(
    grad_output: torch.Tensor,
    self: torch.Tensor,
    target: torch.Tensor,
    reduction: int,
    is_target: torch.Tensor,
):
    logger.debug("GEMS_ENFLAME MULTILABEL_MARGIN_LOSS_BACKWARD")
    if isinstance(reduction, bool) or not isinstance(reduction, int):
        raise RuntimeError(
            "multilabel_margin_loss_backward: reduction must be an integer "
            "(0, 1 or 2)"
        )
    if reduction not in (0, 1, 2):
        # The native kernel maps every other integer to "sum".
        reduction = 2

    rows, n_classes = _check_inputs(grad_output, self, target, is_target)
    if rows == 0:
        return torch.empty_like(self)

    # grad_output is per-row only for reduction 0; the scalar reductions get one
    # value. A 1-D tensor of any other length is rejected rather than broadcast.
    if reduction == 0 and grad_output.dim() >= 1 and grad_output.numel() != rows:
        raise RuntimeError(
            "multilabel_margin_loss_backward: grad_output must hold one value per "
            "row for reduction 0"
        )
    grad_output_stride = (
        grad_output.stride(0) if reduction == 0 and grad_output.dim() == 1 else 0
    )

    acc64 = self.dtype == torch.float64
    input_row_stride, input_class_stride = _strides2d(self)
    target_row_stride, target_class_stride = _strides2d(target)
    is_target_row_stride, is_target_class_stride = _strides2d(is_target)

    n_mean = rows if reduction == 1 else 1

    # Rows beyond the row grid are walked by the kernels' grid-stride loop.
    row_programs = min(rows, _MAX_ROW_PROGRAMS, _MAX_GRID_X)
    block = _grad_block_for(rows, n_classes)
    class_tiles = triton.cdiv(n_classes, block)
    # The second grid dimension is capped at 255 by the GCU launch descriptor,
    # which a wide class axis exceeds, so the (row program, class tile) pair is
    # flattened into a single grid axis and decoded inside the kernels.
    grid = (row_programs * class_tiles,)
    # One class tile means the tile masks alone can decide both terms, which
    # removes four per-row cross-lane reductions from the kernels.
    single_tile = class_tiles == 1

    with torch_device_fn.device(self.device):
        multiplicity = torch.empty(
            (rows, n_classes), dtype=torch.int32, device=self.device
        )
        _membership_kernel[grid](
            target,
            multiplicity,
            rows,
            n_classes,
            row_programs,
            target_row_stride,
            target_class_stride,
            BLOCK_T=_BLOCK_T,
            BLOCK=block,
            SINGLE_TILE=single_tile and n_classes <= _BLOCK_T,
        )
        grad_input = torch.empty(
            (rows, n_classes), dtype=self.dtype, device=self.device
        )
        _grad_kernel[grid](
            self,
            multiplicity,
            is_target,
            grad_input,
            grad_output,
            rows,
            n_classes,
            row_programs,
            input_row_stride,
            input_class_stride,
            is_target_row_stride,
            is_target_class_stride,
            grad_output_stride,
            n_mean,
            ACC64=acc64,
            BLOCK=block,
            SINGLE_TILE=single_tile,
        )
    return grad_input.reshape(self.shape)
