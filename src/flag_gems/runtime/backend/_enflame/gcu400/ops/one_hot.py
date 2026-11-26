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

from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# Below this output element count the single-kernel flat comparison path is
# cheaper; above it the two-stage "fill zeroes + scatter ones" path saturates the
# int64 write bandwidth.
_FLAT_MAX_ELEMENTS = 1 << 19
# Fused min/max validation kernel: one program scans the whole index tensor, so
# only use it while that is still cheaper than two parallel aten reductions.
_MINMAX_ONESHOT_MAX = 1 << 18
_MINMAX_BLOCK = 4096
_FLAT_BLOCK = 4096
_FILL_BLOCK = 32768
_SCATTER_BLOCK = 1024
# Single-dimension grid limit
_MAX_GRID_X = 65535


@triton.jit
def one_hot_flat_kernel(
    index_ptr,
    out_ptr,
    num_classes,
    total,
    BLOCK: tl.constexpr,
    NC_SHIFT: tl.constexpr,
):
    """Flat comparison path: block the output elements contiguously and recover (row, col).

    Each program owns BLOCK consecutive int64 output elements and writes them with
    pure contiguous stores, which suits small to medium outputs (launch overhead
    dominates there, so one kernel beats the two-stage form).

    When ``num_classes`` is a power of two the host passes ``NC_SHIFT`` and the
    row/col split becomes a shift and a mask: the backend emulates integer
    division in software, so the per-element div/mod is a measurable share of the
    small-output case.
    """
    # For large outputs the block count exceeds the grid.x limit, so use a 2D grid
    block_id = tle.program_id(0) + tle.program_id(1) * tle.num_programs(0)
    out_offsets = block_id * BLOCK + tl.arange(0, BLOCK)
    mask = out_offsets < total

    if NC_SHIFT >= 0:
        row_index = out_offsets >> NC_SHIFT
        col_index = out_offsets & (num_classes - 1)
    else:
        row_index = out_offsets // num_classes
        col_index = out_offsets % num_classes
    target_class = tl.load(index_ptr + row_index, mask=mask, other=-1)

    tl.store(out_ptr + out_offsets, (target_class == col_index).to(tl.int64), mask=mask)


@triton.jit
def one_hot_minmax_kernel(index_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    """Compute min and max of the class indices in a single launch.

    ``aten::one_hot`` needs both bounds (non-negative and below ``num_classes``).
    Calling ``tensor.max().item()`` and ``tensor.min().item()`` costs two
    launches *and* two device synchronisations per call; one launch that stores
    both scalars lets the host read them with a single sync.
    """
    offsets = tl.arange(0, BLOCK)
    running_min = tl.full((), 9223372036854775807, tl.int64)
    running_max = tl.full((), -9223372036854775808, tl.int64)
    for start in range(0, numel, BLOCK):
        idx = start + offsets
        mask = idx < numel
        vals = tl.load(index_ptr + idx, mask=mask, other=0)
        running_min = tl.minimum(
            running_min, tl.min(tl.where(mask, vals, 9223372036854775807))
        )
        running_max = tl.maximum(
            running_max, tl.max(tl.where(mask, vals, -9223372036854775808))
        )
    tl.store(out_ptr, running_min)
    tl.store(out_ptr + 1, running_max)


@triton.jit
def one_hot_fill_kernel(out_ptr, total, BLOCK: tl.constexpr):
    """Fill the whole output buffer with 0 (pure int64 fill, no computation)."""
    block_id = tle.program_id(0) + tle.program_id(1) * tle.num_programs(0)
    out_offsets = block_id * BLOCK + tl.arange(0, BLOCK)
    tl.store(
        out_ptr + out_offsets,
        tl.zeros([BLOCK], dtype=tl.int64),
        mask=out_offsets < total,
    )


@triton.jit
def one_hot_scatter_kernel(
    index_ptr,
    out_ptr,
    num_classes,
    numel,
    BLOCK: tl.constexpr,
):
    """Write 1 for every valid index: out[i * num_classes + index[i]] = 1."""
    block_id = tle.program_id(0) + tle.program_id(1) * tle.num_programs(0)
    row_offsets = block_id * BLOCK + tl.arange(0, BLOCK)
    mask = row_offsets < numel
    target_class = tl.load(index_ptr + row_offsets, mask=mask, other=0)
    tl.store(
        out_ptr + row_offsets * num_classes + target_class,
        tl.full([BLOCK], 1, tl.int64),
        mask=mask,
    )


def _index_bounds(tensor: torch.Tensor):
    """Return (min, max) of the class indices as Python ints.

    Small tensors take the fused single-launch min/max kernel (one device
    synchronisation instead of two); larger ones keep the aten reductions so the
    whole tensor is still scanned by a parallel grid.
    """
    if tensor.numel() <= _MINMAX_ONESHOT_MAX:
        bounds = torch.empty(2, device=tensor.device, dtype=torch.int64)
        one_hot_minmax_kernel[(1,)](tensor, bounds, tensor.numel(), BLOCK=_MINMAX_BLOCK)
        min_index, max_index = bounds.tolist()
        return int(min_index), int(max_index)
    return int(tensor.min().item()), int(tensor.max().item())


def one_hot(tensor: torch.Tensor, num_classes: int = -1) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME ONE_HOT")
    numel = tensor.numel()
    # Validate exactly like aten::one_hot: `num_classes=-1` infers the class count
    # from the largest index, every index must be non-negative, and every index
    # must stay below `num_classes`. Without these checks out-of-range indices
    # silently write past the row (or produce an empty output), which is what the
    # `pytest.raises(RuntimeError)` contract of the accuracy test checks.
    if numel == 0:
        # On this backend the max reduction on a tensor with no elements aborts
        # the process instead of raising, so mirror torch.nn.functional.one_hot's
        # Python-level error here.
        if num_classes == -1:
            raise RuntimeError(
                "cannot perform reduction function max on tensor with no elements "
                "because the operation does not have an identity"
            )
        return torch.empty(
            (*tensor.shape, num_classes), device=tensor.device, dtype=torch.int64
        )

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    min_index, max_index = _index_bounds(tensor)
    if min_index < 0:
        raise RuntimeError("Class values must be non-negative.")
    if num_classes == -1:
        num_classes = max_index + 1
    elif max_index >= num_classes:
        raise RuntimeError("Class values must be smaller than num_classes.")

    out = torch.empty(
        (*tensor.shape, num_classes), device=tensor.device, dtype=torch.int64
    )

    total = numel * num_classes
    # Power-of-two class counts replace the per-element div/mod with shift + mask.
    is_pow2 = num_classes & (num_classes - 1) == 0
    nc_shift = num_classes.bit_length() - 1 if is_pow2 else -1
    # Small outputs use the single-kernel comparison path, large ones "write zeroes +
    # scatter"; both paths are pure Triton.
    if total <= _FLAT_MAX_ELEMENTS:
        flat_blocks = triton.cdiv(total, _FLAT_BLOCK)
        flat_grid_y = triton.cdiv(flat_blocks, _MAX_GRID_X)
        one_hot_flat_kernel[(triton.cdiv(flat_blocks, flat_grid_y), flat_grid_y)](
            tensor,
            out,
            num_classes,
            total,
            BLOCK=_FLAT_BLOCK,
            NC_SHIFT=nc_shift,
        )
    else:
        fill_blocks = triton.cdiv(total, _FILL_BLOCK)
        fill_grid_y = triton.cdiv(fill_blocks, _MAX_GRID_X)
        one_hot_fill_kernel[(triton.cdiv(fill_blocks, fill_grid_y), fill_grid_y)](
            out,
            total,
            BLOCK=_FILL_BLOCK,
        )
        scatter_blocks = triton.cdiv(numel, _SCATTER_BLOCK)
        scatter_grid_y = triton.cdiv(scatter_blocks, _MAX_GRID_X)
        one_hot_scatter_kernel[
            (triton.cdiv(scatter_blocks, scatter_grid_y), scatter_grid_y)
        ](
            tensor,
            out,
            num_classes,
            numel,
            BLOCK=_SCATTER_BLOCK,
        )
    return out
