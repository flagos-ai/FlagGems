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

import importlib
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

_GATHER_BLOCK_SIZE = 1024


def _gen():
    # `flag_gems.ops.__init__` rebinds the name `unique_dim` to the entry
    # function, so plain attribute access on the package yields the function,
    # not this module. Resolve the real module lazily (the generic module may
    # not be fully imported yet while the vendor package initialises).
    return importlib.import_module("flag_gems.ops.unique_dim")


@triton.jit
def _gather_strided_kernel(
    src,
    dst,
    numel,
    shape,
    strides,
    NDIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Materialise an arbitrarily strided view into a contiguous buffer. The
    # generic unique_dim reaches for `.contiguous()` on a `movedim` view (and
    # on `flat.t()` in the int64 cascade), which redispatches to the native
    # strided copy_ and dies with `invalid device function` on this backend.
    # Each lane owns one contiguous destination element, decodes it into view
    # coordinates and gathers through the view's strides instead.
    #
    # Indices are clamped rather than masked: a tail lane re-reads and
    # re-writes the last element (a redundant write of the correct value),
    # because a masked store past the end of a small buffer segfaults here.
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets_c = tl.minimum(offsets, numel - 1)

    idx = offsets_c
    src_off = offsets_c * 0
    for d in tl.static_range(NDIM - 1, -1, -1):
        size_d = tl.load(shape + d)
        stride_d = tl.load(strides + d)
        coord = idx % size_d
        idx = idx // size_d
        src_off += coord * stride_d

    values = tl.load(src + src_off)
    tl.store(dst + offsets_c, values)


@triton.jit
def _counts_clamped_kernel(
    first_positions,
    counts,
    num_rows,
    num_unique,
    BLOCK_SIZE: tl.constexpr,
):
    # The generic counts kernel loads `first_positions[i + 1]` with
    # `mask=(i + 1) < num_unique, other=num_rows`. On this backend such a
    # masked load leaks whatever really sits one past the end of the buffer
    # instead of `other`, so the last group's count became garbage (e.g. 1
    # instead of 4 whenever every row collapses into one group). Clamp the
    # neighbour index into range and select the `num_rows` sentinel with
    # `tl.where` instead.
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets_c = tl.minimum(offsets, num_unique - 1)

    positions = tl.load(first_positions + offsets_c)
    next_c = tl.minimum(offsets_c + 1, num_unique - 1)
    next_loaded = tl.load(first_positions + next_c)
    is_last = (offsets_c + 1) >= num_unique
    end = tl.where(is_last, positions * 0 + num_rows, next_loaded)
    tl.store(counts + offsets_c, end - positions)


def _materialize_contiguous(view: torch.Tensor) -> torch.Tensor:
    """Contiguous copy of ``view`` without going through native copy_."""
    if view.is_contiguous():
        return view
    out = torch.empty(view.shape, dtype=view.dtype, device=view.device)
    numel = view.numel()
    if numel == 0:
        return out
    shape = torch.tensor(view.shape, dtype=torch.int64, device=view.device)
    strides = torch.tensor(view.stride(), dtype=torch.int64, device=view.device)
    with torch_device_fn.device(view.device):
        _gather_strided_kernel[(triton.cdiv(numel, _GATHER_BLOCK_SIZE),)](
            view,
            out,
            numel,
            shape,
            strides,
            NDIM=view.ndim,
            BLOCK_SIZE=_GATHER_BLOCK_SIZE,
        )
    return out


def _unique_dim_counts(is_first: torch.Tensor, num_rows: int) -> torch.Tensor:
    first_positions = torch.nonzero(is_first, as_tuple=False).flatten()
    num_unique = first_positions.numel()
    counts = torch.empty(num_unique, dtype=torch.int64, device=is_first.device)
    if num_unique == 0:
        return counts
    grid = (triton.cdiv(num_unique, _GATHER_BLOCK_SIZE), 1, 1)
    with torch_device_fn.device(is_first.device.index):
        _counts_clamped_kernel[grid](
            first_positions,
            counts,
            num_rows,
            num_unique,
            BLOCK_SIZE=_GATHER_BLOCK_SIZE,
            num_warps=4,
        )
    return counts


def _lex_argsort_rows_cascade(flat: torch.Tensor) -> torch.Tensor:
    # Same LSD cascade as the generic fallback, but the column-major
    # materialisation goes through the strided gather above instead of
    # `flat.t().contiguous()`.
    generic = _gen()
    num_rows, num_cols = flat.shape
    indices = torch.arange(num_rows, dtype=torch.int64, device=flat.device)
    if num_rows <= 1 or num_cols == 0:
        return indices
    flat_t = _materialize_contiguous(flat.t())
    for col in range(num_cols - 1, -1, -1):
        keys = generic._triton_gather_1d(flat_t[col], indices)
        # The LSD cascade requires a stable sort. The generic code calls
        # `torch.sort(keys, stable=True)`, which under FlagGems interception
        # dispatches to the Triton radix sort - and that raises a kernel
        # exception (719, in scatter_kernel) on this backend even for five
        # keys. `_argsort_keys` is stable too: its rank kernel breaks ties by
        # the original index (`(vals == cur) & (candidates < row)`), so reuse
        # it here.
        perm, _ = generic._argsort_keys(keys)
        indices = generic._triton_gather_1d(indices, perm)
    return indices


def _lex_argsort_rows(flat: torch.Tensor) -> tuple:
    composite = _gen()._lex_argsort_rows_composite(flat)
    if composite is not None:
        return composite
    return _lex_argsort_rows_cascade(flat), False


def unique_dim(
    input: torch.Tensor,
    dim: int,
    sorted: bool = True,
    return_inverse: bool = False,
    return_counts: bool = False,
):
    logger.debug("GEMS_KUNLUNXIN UNIQUE_DIM")
    generic = _gen()
    ndim = input.ndim if input.ndim > 0 else 1
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= max(input.ndim, 1):
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-input.ndim}, {input.ndim - 1}], but got {dim})"
        )

    device = input.device
    size_dim = input.size(dim) if input.ndim > 0 else input.numel()

    if size_dim == 0:
        output = input.clone()
        inverse_indices = torch.empty(0, dtype=torch.int64, device=device)
        counts = torch.empty(0, dtype=torch.int64, device=device)
        return output, inverse_indices, counts

    moved = _materialize_contiguous(input.movedim(dim, 0))
    flat = moved.reshape(size_dim, -1)

    sorted_indices, all_unique = _lex_argsort_rows(flat)

    inverse_indices = torch.empty(0, dtype=torch.int64, device=device)
    counts = torch.empty(0, dtype=torch.int64, device=device)

    if all_unique:
        if return_counts:
            counts = torch.ones(size_dim, dtype=torch.int64, device=device)
        if return_inverse:
            inverse_indices = generic._unique_dim_inverse_from_permutation(
                sorted_indices
            )
        output = generic._unique_dim_gather_output(
            moved, sorted_indices, dim, input.shape
        )
        return _materialize_contiguous(output), inverse_indices, counts

    is_first = generic._unique_dim_first_mask(flat, sorted_indices)
    if return_inverse:
        (
            unique_in_orig,
            inverse_indices,
        ) = generic._unique_dim_unique_indices_and_inverse(sorted_indices, is_first)
    else:
        unique_in_orig = generic._unique_dim_unique_indices(sorted_indices, is_first)

    if return_counts:
        counts = _unique_dim_counts(is_first, size_dim)

    output = generic._unique_dim_gather_output(moved, unique_in_orig, dim, input.shape)

    return _materialize_contiguous(output), inverse_indices, counts
