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

"""Correctness tests for ``aten::_validate_compressed_sparse_indices``.

``compressed_idx`` has shape ``batch + (cdim + 1,)`` and holds one offset per
compressed row/slice plus the terminal offset ``nnz``; ``plain_idx`` has shape
``batch + (nnz,)`` and holds plain indices inside ``[0, dim)``.  The operator
validates that descriptor and returns ``None``.

Unresolved coverage gap (deliberately not represented as tests): a wrong leading
offset, non-monotone offsets, a terminal offset other than ``nnz``, and plain
indices outside ``[0, dim)`` or unordered inside one compressed row are only
observable through illegal index *contents*.  On the CUDA target those contents
take the fatal ``CUDA_KERNEL_ASSERT`` path at
``ValidateCompressedIndicesCommon.h:44``, which aborts the process and poisons
the CUDA context for the rest of the session (measured: exit code 1 with a flood
of ``..._assert: Assertion 'cond && message' failed``), so they cannot run
in-process on a shared device, and the test protocol offers no isolated
execution.  This suite therefore covers valid descriptors and the safe
host/schema errors only; the content workloads stay pending in the local
protocol-gap record and must not be counted as covered.

Spec dimensions that do not apply: the descriptors are integer tensors with no
floating index dtype, so there is no NaN/Inf case and the five value ranges
select slice-length profiles instead of values; ``cdim``/``dim``/``nnz`` are
mandatory Python integers with no schema default, so no call omits them (their
coverage comes from the descriptor geometries -- zero, tiny, empty batch, and
the 32/64-bit limits -- and from the negative tests); the two descriptors must
have exactly matching shapes, so nothing broadcasts; and the result is ``None``,
so there is no backward pass.
"""

import math

import pytest
import torch

import flag_gems

from . import test_utils as tu

# AT_DISPATCH_INDEX_TYPES admits 32/64-bit integer indices only; every other
# index dtype is rejected by the kernel dispatch (see the dtype negative test).
_INDEX_DTYPES = [torch.int32]
if flag_gems.runtime.device.support_int64:
    _INDEX_DTYPES.append(torch.int64)

# Fixed plain-dimension size of the main grid: every slice-length profile below
# stays well inside it.
_GRID_DIM = 8

# Each value range selects one slice-length pattern -- repeated over every
# compressed row -- together with the end of ``[0, dim)`` at which the plain
# indices are anchored: the extreme-range patterns reach ``dim - 1``, the others
# stay near 0.
_RANGE_PLANS = {
    ("-1", "1"): ((1, 2, 1, 2), "low"),
    ("0", "1"): ((1, 2, 3, 4), "low"),
    ("-1", "0"): ((4, 3, 2, 1), "high"),
    ("0", "max"): ((4, 1, 1, 1), "high"),
    ("min", "0"): ((1, 0, 0, 0), "low"),
}

# Width of the widest range profile.  The per-slice anchor shift stays inside
# it, which is what keeps ``first + arange(count)`` inside ``[0, dim]`` for
# every profile.  It describes the range plans only and never limits the
# explicitly requested boundary structures below.
_PROFILE_WIDTH = max(max(profile) for profile, _ in _RANGE_PLANS.values())

# ``shape`` is the shape of ``compressed_idx``; a rank-0 tensor cannot carry a
# compressed dimension, so it is covered by the rank negative test instead.
_VALID_SHAPES = [shape for shape in tu.selected_shapes() if len(shape) > 0]

# ``idx_max_ndims = 8`` in ValidateCompressedIndicesCommon.h selects a static
# shape path for descriptors up to rank 8 and the general path beyond it, for
# both ``is_crow`` directions; these two shapes exercise the two branches.
_HIGH_RANK_SHAPES = [(2,) * 7 + (4,), (2,) * 8 + (4,)]

# Keep both cheap rank-dispatch branches in quick as well as default.
_SHAPE_CASES = _VALID_SHAPES + _HIGH_RANK_SHAPES


def _range_plan(value_range):
    key = (value_range[0], value_range[1])
    if key not in _RANGE_PLANS:
        raise KeyError(f"unsupported value range {value_range!r}")
    return _RANGE_PLANS[key]


def _descriptor(counts, first, index_dtype):
    """Build ``(compressed_idx, plain_idx, cdim, nnz)`` from explicit lengths.

    ``counts`` and ``first`` have shape ``batch + (cdim,)``: the number of plain
    indices in each compressed row and the first plain index of that row.  The
    offsets are the exact prefix sums of ``counts`` and the plain indices are
    ``first + arange(count)``, so the returned ``cdim`` and ``nnz`` are exactly
    the requested ones and no counted structure is rescaled.  Every counter
    (counts, offsets, slot index, row total, cumulative sum) is computed in
    ``index_dtype`` itself, so a device without int64 support never sees an
    int64 staging tensor for an int32 descriptor.
    """
    device = flag_gems.device
    counts = counts.to(device=device, dtype=index_dtype)
    first = first.to(device=device, dtype=index_dtype)
    batch = tuple(counts.shape[:-1])
    cdim = int(counts.shape[-1])
    # No compressed row, no batch slice, or no entry at all: the offsets stay at
    # their zero initializer and the plain buffer is empty (nnz == 0), which is
    # the coherent descriptor for those geometries.
    if cdim == 0 or counts.numel() == 0 or int(counts.max().item()) == 0:
        compressed = torch.zeros(batch + (cdim + 1,), dtype=index_dtype, device=device)
        plain = torch.zeros(batch + (0,), dtype=index_dtype, device=device)
        return compressed, plain, cdim, 0

    flat_counts = counts.reshape(-1, cdim)
    totals = flat_counts.sum(-1, dtype=index_dtype)
    nnz = int(totals[0].item())
    if not bool((totals == nnz).all()):
        raise ValueError("every batch slice must contribute the same nnz")

    slots = torch.arange(
        int(flat_counts.max().item()), dtype=index_dtype, device=device
    )
    mask = slots < counts.unsqueeze(-1)
    plain = (first.unsqueeze(-1) + slots)[mask].view(batch + (nnz,))
    compressed = torch.zeros(batch + (cdim + 1,), dtype=index_dtype, device=device)
    compressed[..., 1:] = flat_counts.cumsum(-1, dtype=index_dtype).reshape(
        batch + (cdim,)
    )
    return compressed, plain, cdim, nnz


def _descriptor_for_range(shape, value_range, index_dtype, dim):
    """Descriptor whose compressed rows follow the range's slice-length plan.

    The row lengths are rotated per batch slice, which keeps ``nnz`` fixed while
    every slice carries a different legal offset sequence.
    """
    device = flag_gems.device
    batch = tuple(shape[:-1])
    cdim = shape[-1] - 1
    if cdim == 0:
        empty = torch.zeros(batch + (0,), dtype=index_dtype, device=device)
        return _descriptor(empty, empty, index_dtype)

    profile, anchor = _range_plan(value_range)
    lengths = [min(profile[row % len(profile)], dim) for row in range(cdim)]
    counts_row = torch.tensor(lengths, dtype=index_dtype, device=device)
    nslice = math.prod(batch) if batch else 1
    rows = torch.arange(cdim, dtype=index_dtype, device=device)
    local = (torch.arange(nslice, dtype=index_dtype, device=device) % cdim).view(
        batch + (1,)
    )
    counts = counts_row[(rows - local) % cdim]
    room = dim - counts
    shift = local % _PROFILE_WIDTH
    if anchor == "high":
        first = (room - shift).clamp(min=0)
    else:
        first = torch.minimum(shift * 2, room)
    target = batch + (cdim,)
    return _descriptor(counts.expand(target), first.expand(target), index_dtype)


def _row_profile(pattern, shape, index_dtype):
    """Repeat a per-row profile over exactly ``shape[-1] - 1`` rows per slice.

    The built descriptor therefore has ``cdim == shape[-1] - 1`` compressed rows
    in every slice of ``shape[:-1]``, and the profile decides both the true
    ``nnz`` and the plain values.
    """
    batch = tuple(shape[:-1])
    cdim = shape[-1] - 1
    rows = [pattern[row % len(pattern)] for row in range(cdim)]
    flat = torch.tensor(rows, dtype=index_dtype, device=flag_gems.device)
    return flat.reshape((1,) * len(batch) + (cdim,)).expand(batch + (cdim,))


def _input_signature(tensor):
    """Storage and view identity of an input, not merely its values.

    Value comparison alone cannot tell an untouched input from one whose storage
    was swapped for an equally valued copy, so the storage pointer, offset, size
    and strides are captured around the candidate call as well.
    """
    return (
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.size()),
        tuple(tensor.stride()),
    )


def _check_descriptor(is_crow, compressed, plain, cdim, dim, nnz):
    """Run the reference on private copies, then the candidate on the inputs.

    The reference gets its own tensors, so a non-target reference device never
    runs the native kernel on (or through) the candidate inputs and a native
    write can never be mistaken for a candidate write; the value comparison
    against those copies plus the storage signatures taken around the candidate
    call are the candidate's input-immutability check.
    """
    ref_compressed = tu.to_reference(compressed)
    ref_plain = tu.to_reference(plain)
    before = (_input_signature(compressed), _input_signature(plain))

    torch.ops.aten._validate_compressed_sparse_indices(
        is_crow, ref_compressed, ref_plain, cdim, dim, nnz
    )
    result = flag_gems._validate_compressed_sparse_indices(
        is_crow, compressed, plain, cdim, dim, nnz
    )

    assert result is None
    after = (_input_signature(compressed), _input_signature(plain))
    assert after == before, "the validator must not swap or resize its inputs"
    tu.assert_result_equal(compressed, ref_compressed)
    tu.assert_result_equal(plain, ref_plain)


@pytest.mark.validate_compressed_sparse_indices
@pytest.mark.parametrize("shape", _SHAPE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("is_crow", [True, False])
def test_validate_compressed_sparse_indices(shape, value_range, index_dtype, is_crow):
    compressed, plain, cdim, nnz = _descriptor_for_range(
        shape, value_range, index_dtype, _GRID_DIM
    )
    _check_descriptor(is_crow, compressed, plain, cdim, _GRID_DIM, nnz)


# ``(case id, compressed_idx shape, per-row plain counts, per-row first plain
# index, dim)``.  The count/first tuples are profiles: each is repeated over
# exactly ``shape[-1] - 1`` compressed rows in every batch slice, so the built
# cdim and nnz always match the requested geometry.  These rows add the
# boundaries the regular grid cannot reach: an empty plain dimension, the
# smallest non-empty one, an empty batch, segments without entries between
# filled ones, and a dim that saturates the 32/64-bit index range.
_BOUNDARY_ROWS = [
    ("dim-zero", (256,), (0,), (0,), 0),
    ("dim-one", (256,), (1,), (0,), 1),
    ("tiny-nnz-top-index", (2,), (1,), (7,), 8),
    ("empty-segments", (256,), (0, 2, 0, 1), (0, 0, 0, 7), 8),
    ("zero-batch", (0, 5), (1,), (0,), 8),
    ("int32-max-dim", (2,), (1,), (2**31 - 2,), 2**31 - 1),
    ("int64-huge-dim", (2, 19, 7), (1,), (2**31 - 1,), 2**31),
]

# ``int32-max-dim`` runs the int32 index kernel at the largest ``dim`` that
# kernel can represent (2**31 - 1).  ``int64-huge-dim`` needs ``dim = 2**31``,
# which is above the int32 range, so it must use the int64 index kernel; its
# stored index ``2**31 - 1`` would fit int32 by itself, the size is what forces
# the wider kernel.
_BOUNDARY_DTYPES = {
    "dim-zero": _INDEX_DTYPES,
    "dim-one": _INDEX_DTYPES,
    "tiny-nnz-top-index": _INDEX_DTYPES,
    "empty-segments": _INDEX_DTYPES,
    "zero-batch": _INDEX_DTYPES,
    "int32-max-dim": [torch.int32],
    "int64-huge-dim": [torch.int64],
}


def _boundary_cases():
    rows = []
    for case_id, shape, counts_row, first_row, dim in _BOUNDARY_ROWS:
        for index_dtype in _BOUNDARY_DTYPES[case_id]:
            if index_dtype in _INDEX_DTYPES:
                rows.append(
                    pytest.param(
                        shape, counts_row, first_row, dim, index_dtype, id=case_id
                    )
                )
    return rows


@pytest.mark.validate_compressed_sparse_indices
@pytest.mark.parametrize(
    "shape, counts_row, first_row, dim, index_dtype", _boundary_cases()
)
@pytest.mark.parametrize("is_crow", [True, False])
def test_validate_compressed_sparse_indices_boundary(
    shape, counts_row, first_row, dim, index_dtype, is_crow
):
    counts = _row_profile(counts_row, shape, index_dtype)
    first = _row_profile(first_row, shape, index_dtype)
    compressed, plain, cdim, nnz = _descriptor(counts, first, index_dtype)
    _check_descriptor(is_crow, compressed, plain, cdim, dim, nnz)


@pytest.mark.validate_compressed_sparse_indices
@pytest.mark.parametrize("field", ["cdim", "nnz"])
@pytest.mark.parametrize("is_crow", [True, False])
def test_validate_compressed_sparse_indices_negative_extent(field, is_crow):
    compressed, plain, cdim, nnz = _descriptor_for_range(
        (256,), ["-1", "1"], torch.int32, _GRID_DIM
    )
    if field == "cdim":
        cdim = -2
    else:
        nnz = -1
    with pytest.raises((RuntimeError, IndexError)):
        flag_gems._validate_compressed_sparse_indices(
            is_crow, compressed, plain, cdim, _GRID_DIM, nnz
        )


@pytest.mark.validate_compressed_sparse_indices
@pytest.mark.parametrize("delta", [1, -1])
@pytest.mark.parametrize("field", ["cdim", "nnz"])
@pytest.mark.parametrize("is_crow", [True, False])
def test_validate_compressed_sparse_indices_length_mismatch(field, delta, is_crow):
    compressed, plain, cdim, nnz = _descriptor_for_range(
        (256,), ["-1", "1"], torch.int32, _GRID_DIM
    )
    if field == "cdim":
        cdim = cdim + delta
    else:
        nnz = nnz + delta
    with pytest.raises((RuntimeError, IndexError)):
        flag_gems._validate_compressed_sparse_indices(
            is_crow, compressed, plain, cdim, _GRID_DIM, nnz
        )


@pytest.mark.validate_compressed_sparse_indices
@pytest.mark.parametrize("is_crow", [True, False])
def test_validate_compressed_sparse_indices_rank_negative(is_crow):
    device = flag_gems.device
    scalar = torch.zeros((), dtype=torch.int32, device=device)
    empty = torch.zeros((0,), dtype=torch.int32, device=device)
    with pytest.raises((RuntimeError, IndexError)):
        flag_gems._validate_compressed_sparse_indices(is_crow, scalar, empty, 0, 8, 0)


@pytest.mark.validate_compressed_sparse_indices
@pytest.mark.parametrize("is_crow", [True, False])
def test_validate_compressed_sparse_indices_int32_overflow_negative(is_crow):
    compressed, plain, cdim, nnz = _descriptor_for_range(
        (2,), ["-1", "1"], torch.int32, _GRID_DIM
    )
    # ``dim`` above INT32_MAX cannot be represented by an int32 index kernel.
    with pytest.raises((RuntimeError, ValueError)):
        flag_gems._validate_compressed_sparse_indices(
            is_crow, compressed, plain, cdim, 2**31, nnz
        )


# AT_DISPATCH_INDEX_TYPES admits int32/int64 only, and the rejection happens
# before any index content is read, so these are safe negative cases.  The
# rejection is the measured nvidia CUDA behaviour, and dtype families the active
# device cannot construct are left out, so the list is decided statically here
# instead of being skipped while the suite runs.
_UNSUPPORTED_INDEX_DTYPES = []
if flag_gems.vendor_name == "nvidia":
    _UNSUPPORTED_INDEX_DTYPES = [
        torch.float32,
        torch.float16,
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.bool,
    ]
    if flag_gems.runtime.device.support_bf16:
        _UNSUPPORTED_INDEX_DTYPES.append(torch.bfloat16)
    if flag_gems.runtime.device.support_fp64:
        _UNSUPPORTED_INDEX_DTYPES.append(torch.float64)
    if flag_gems.runtime.device.support_fp8:
        _UNSUPPORTED_INDEX_DTYPES.extend([torch.float8_e4m3fn, torch.float8_e5m2])


@pytest.mark.validate_compressed_sparse_indices
@pytest.mark.parametrize("index_dtype", _UNSUPPORTED_INDEX_DTYPES)
def test_validate_compressed_sparse_indices_unsupported_dtype_negative(index_dtype):
    compressed, plain, cdim, nnz = _descriptor_for_range(
        (256,), ["-1", "1"], torch.int32, _GRID_DIM
    )
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._validate_compressed_sparse_indices(
            True,
            compressed.to(index_dtype),
            plain.to(index_dtype),
            cdim,
            _GRID_DIM,
            nnz,
        )
