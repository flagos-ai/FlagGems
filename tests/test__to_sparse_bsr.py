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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

_DEVICE = flag_gems.device
_DEV = flag_gems.runtime.device

# Support comes from the static backend capability flags only; nothing in this
# file probes the native operator at import, collection or run time.
_DTYPES = [torch.int8, torch.uint8, torch.float32, torch.float16, torch.int32]
if _DEV.support_bf16:
    _DTYPES.append(torch.bfloat16)
if _DEV.support_int64:
    _DTYPES.append(torch.int64)
if _DEV.support_fp8:
    _DTYPES += [torch.float8_e4m3fn, torch.float8_e5m2]
if utils.fp64_is_supported:
    _DTYPES.append(torch.float64)

_SPECIAL_DTYPES = [torch.float16, torch.float32]
if _DEV.support_bf16:
    _SPECIAL_DTYPES.append(torch.bfloat16)
if utils.fp64_is_supported:
    _SPECIAL_DTYPES.append(torch.float64)
if _DEV.support_fp8:
    _SPECIAL_DTYPES += [torch.float8_e4m3fn, torch.float8_e5m2]

# These fixtures choose int64 for the sparse index tensors they build: the .out
# BSR crow/col buffers are constructed with int64 explicitly, the COO source is
# normalised to int64 by to_sparse(), and the CSR/COO negative inputs come from
# the same construction paths. A backend without the int64 capability cannot
# materialise them, so the families that build them are gated on the static
# backend flag instead of failing inside allocation. The index dtype here is this
# fixture's choice (or the COO normalisation); it is not a universal native
# restriction on constructed BSR/CSR index tensors.
_INT64_OK = bool(_DEV.support_int64)

# The mask helpers index at most a few dozen block rows and columns, so int32 is
# adequate; naming the dtype keeps them off arange's implicit int64 default.
_MASK_DTYPE = torch.int32

# The conversion takes its block grid from the last two sparse axes, so the
# sparse part needs rank >= 2. Rank 0 and rank 1 dense inputs are rejected by the
# native schema (IndexError: Dimension out of range) and are covered by the
# negative table instead of the shape grid.
_GRID_SHAPES = [shape for shape in tu.selected_shapes() if len(shape) >= 2]
_GRID_RANGES = tu.selected_ranges()

# The main grid already visits the quick shape and range, so this supplemental
# block-size sweep is a default-only family.
_BLOCKSIZE_ROWS = tu.selected_cases(
    [
        ((8, 8), (1, 1), 0),
        ((8, 8), (2, 2), 0),
        ((8, 8), (4, 4), 0),
        ((8, 8), (2, 4), 0),
        ((8, 8), (8, 8), 0),
        ((16, 16), (4, 4), 0),
        ((1024, 1024), (16, 16), 0),
        ((20, 320, 15), (4, 5), 0),
        ((16, 128, 64, 60), (4, 4), 0),
        ((8, 8, 8), (2, 2), 1),
        ((16, 8, 8), (4, 4), 1),
        ((8, 8, 8, 4), (4, 2), 2),
        ((2, 4, 4), (2, 2), 0),
        ((2, 4, 8, 8), (2, 4), 1),
    ],
    quick=[],
)
_BLOCKSIZE_DTYPES = [torch.float32, torch.float16, torch.int8]

_PATTERN_ROWS = tu.selected_cases(
    [
        ((8, 8), (2, 2), 0),
        ((6, 16), (2, 2), 0),
        ((8, 8, 8), (4, 4), 1),
        ((2, 4, 4), (2, 2), 0),
        ((2, 4, 8, 8), (2, 4), 1),
        ((16, 8), (4, 4), 0),
    ],
    quick=[],
)
_PATTERN_DTYPES = [torch.float32, torch.float16, torch.int8]

_DEGENERATE_ROWS = tu.selected_cases(
    [
        ((0, 0), (1, 1), 0, "zeros"),
        ((8, 0), (2, 2), 0, "zeros"),
        ((0, 8), (2, 2), 0, "zeros"),
        ((4, 0, 4), (2, 2), 1, "zeros"),
        ((4, 4, 0), (2, 2), 1, "zeros"),
        ((4, 8, 8, 0), (2, 2), 1, "zeros"),
        ((4, 8, 8, 0, 0), (2, 2), 2, "zeros"),
        ((2, 4, 4), (2, 2), 0, "zeros"),
        ((8, 8), (2, 2), 0, "zeros"),
        ((8, 8), (2, 2), 0, "single"),
        ((8, 8), (2, 2), 0, "ones"),
        ((8, 8), (2, 2), 0, "opposite"),
    ],
    quick=[],
)

_VIEW_ROWS = tu.selected_cases(
    [
        ((8, 16), (2, 2), 0, "transpose"),
        ((8, 16), (2, 2), 0, "strided"),
        ((8, 16), (2, 2), 0, "offset"),
        ((4, 8, 8), (2, 2), 1, "batch_transpose"),
        ((8, 4, 8, 8), (2, 4), 2, "hybrid_transpose"),
        ((8, 16), (1, 2), 0, "expand"),
        ((2, 4, 8, 8), (2, 4), 1, "expand"),
    ],
    quick=[],
)

# The .out family constructs its BSR buffers with explicit int64 crow/col index
# tensors, the dtype this fixture chooses for them, so it needs the backend
# int64 capability; the comparison itself is unaffected. One small valid row
# stays in quick so the positive .out path is still smoked there, while the
# other positive rows remain default-only.
_OUT_ROWS = (
    tu.selected_cases(
        [((8, 8), (2, 2), 0), ((8, 16), (2, 2), 0)],
        quick=[((8, 8), (2, 2), 0)],
    )
    if _INT64_OK
    else []
)
# Rows of stored block columns, one tuple per block row, for the .out cases.
# Both .out rows have a 4-block-row grid and the input stores 3 blocks, so the
# declared buffer capacity is below the 16 blocks of a fully specified input.
_OUT_BLOCKS = ((), (0,), (0, 1), ())

# Backward through the native reference is unsupported when the sparse part is
# batched, i.e. rank > 2. The reverse path builds a sparse COO whose
# sparse_dim() counts the batch axes and then requires sparse_dim() == 2 when
# converting back to BSR. Observed on the active device for (2, 4, 8, 8) with
# blocksize [2, 4], dense_dim 1 and for (2, 8, 8) with blocksize [2, 2],
# dense_dim 0: RuntimeError: sparse_coo_to_sparse: conversion from Sparse to
# SparseBsr for input tensors with sparse_dim()!=2 is not supported. The forward
# of those shapes succeeds and reports sparse_dim() == 2, so only unbatched
# shapes are covered here. The 'holes' row adds a small shape whose single stored
# block keeps a non-zero border but a zero interior, so the backward is also
# exercised on a retained block that is not uniformly non-zero.
_BACKWARD_ROWS = tu.selected_cases(
    [
        ((8, 8), (2, 2), 0, "pattern"),
        ((6, 16), (2, 2), 0, "pattern"),
        ((4, 8, 8), (4, 4), 1, "pattern"),
        ((8, 8), (2, 2), 0, "zeros"),
        ((8, 8), (4, 4), 0, "holes"),
    ],
    quick=[],
)

# A sparse COO source converts directly when the dense_dim argument is absent;
# giving it for a sparse input is rejected and stays in the negative table. The
# source itself allocates int64 indices, hence the capability gate.
_COO_ROWS = (
    tu.selected_cases(
        [
            ((8, 8), (2, 2), "omit"),
            ((8, 8), (2, 2), "none"),
            ((6, 16), (2, 2), "omit"),
            ((8, 8), (4, 4), "omit"),
            ((16, 8), (4, 4), "none"),
        ],
        quick=[],
    )
    if _INT64_OK
    else []
)

_NEGATIVE_ROWS = [
    ((), (1, 1), 0, "plain", torch.float32, (RuntimeError, IndexError)),
    ((5,), (1, 1), 0, "plain", torch.float32, (RuntimeError, IndexError)),
    ((8,), (1, 1), 0, "plain", torch.float32, (RuntimeError, IndexError)),
    ((8, 8), (3, 2), 0, "plain", torch.float32, RuntimeError),
    ((8, 8), (2, 3), 0, "plain", torch.float32, RuntimeError),
    ((10, 10), (3, 3), 0, "plain", torch.float32, RuntimeError),
    ((8, 8), (2,), 0, "plain", torch.float32, RuntimeError),
    ((8, 8), (2, 2, 2), 0, "plain", torch.float32, RuntimeError),
    ((8, 8), (0, 2), 0, "plain", torch.float32, RuntimeError),
    ((8, 8), (2, 0), 0, "plain", torch.float32, RuntimeError),
    ((8, 8), (-2, 2), 0, "plain", torch.float32, RuntimeError),
    ((8, 8), (16, 16), 0, "plain", torch.float32, RuntimeError),
    ((8, 8), (8, 16), 0, "plain", torch.float32, RuntimeError),
    ((8, 8), (2.0, 2.0), 0, "plain", torch.float32, (RuntimeError, TypeError)),
    ((8, 8), (0, 0), 0, "plain", torch.float32, RuntimeError),
    ((8, 8), (2, 2), 1, "plain", torch.float32, (RuntimeError, IndexError)),
    ((8, 8), (2, 2), 2, "plain", torch.float32, (RuntimeError, IndexError)),
    ((8, 8), (2, 2), -1, "plain", torch.float32, RuntimeError),
    ((8, 8, 8), (2, 2), 3, "plain", torch.float32, (RuntimeError, IndexError)),
    ((8, 8, 8), (3, 2), 0, "plain", torch.float32, RuntimeError),
    ((8, 8, 10), (2, 3), 1, "plain", torch.float32, RuntimeError),
    ((2, 4, 4), (2, 1), 0, "uneven", torch.float32, RuntimeError),
    ((2, 8, 8), (2, 2), 0, "uneven", torch.float32, RuntimeError),
    ((2, 4, 8, 8), (2, 4), 1, "uneven", torch.float32, RuntimeError),
    ((2, 3, 8, 8), (2, 2), 0, "uneven", torch.float32, RuntimeError),
    ((0, 4, 4), (1, 1), 0, "plain", torch.float32, RuntimeError),
    ((0, 8, 8), (2, 2), 0, "plain", torch.float32, RuntimeError),
    ((0, 4, 4), (2, 2), 0, "plain", torch.float32, RuntimeError),
    ((0, 8, 8, 4), (2, 2), 1, "plain", torch.float32, RuntimeError),
    ((2, 0, 8, 8), (2, 2), 0, "plain", torch.float32, RuntimeError),
]

if _INT64_OK:
    # These rows build a COO (.to_sparse(), whose stored indices are normalised
    # to int64) or a CSR (.to_sparse_csr()) source, so they need the backend
    # int64 capability. A given dense_dim is what rejects a sparse input here;
    # the absent-argument form is valid and is covered positively by the COO
    # family.
    _NEGATIVE_ROWS += [
        ((8, 8), (2, 2), 0, "sparse", torch.float32, RuntimeError),
        ((8, 8), (2, 2), 0, "csr", torch.float32, RuntimeError),
    ]

_OPTIONAL_SHAPES = tu.selected_cases(
    [(1024, 1024), (20, 320, 15), (16, 128, 64, 60)], quick=[]
)


def _grid_input(dtype, shape, value_range):
    """Value-range sample for the shape grid.

    A batched dense input must store the same number of blocks in every batch
    instance (native: 'Expect the same number of specified elements per
    batch'), which independently sampled values do not satisfy, so the batch
    axes repeat a single sampled instance. Rank <= 2 inputs are sampled
    directly.
    """
    if len(shape) <= 2:
        return tu.make_input(dtype, shape, value_range)
    instance = tu.make_input(dtype, (1,) * (len(shape) - 2) + shape[-2:], value_range)
    return instance.repeat(shape[:-2] + (1, 1))


def _block_mask(block_rows, block_cols):
    """Deterministic stored-block mask over one block grid.

    With an interior block row available the first row stays empty and the last
    one is cleared, so the pattern has empty leading and trailing rows with
    uneven counts between them. A grid with one or two block rows has no
    interior row, so row 0 carries the pattern instead of collapsing the whole
    input to zero stored blocks.
    """
    rows = torch.arange(block_rows, dtype=_MASK_DTYPE, device=_DEVICE)
    cols = torch.arange(block_cols, dtype=_MASK_DTYPE, device=_DEVICE)
    if block_rows <= 2:
        length = 1 + rows % block_cols
    else:
        length = torch.where(rows > 0, 1 + (rows - 1) % block_cols, 0)
    mask = cols[None, :] < length[:, None]
    if block_rows > 2:
        mask[-1, :] = False
    return mask


def _blocked_pattern(shape, blocksize, dense_dim, dtype):
    """Dense input with an uneven but batch-uniform set of stored blocks.

    Batched inputs must hold the same number of specified elements in every
    batch instance, which independent random values do not guarantee: one mask
    rolled along the batch keeps that count equal while the retained positions
    move. Stored entries are the magnitudes 1..7 with alternating signs taken
    from the flattened position, which is exact in every dtype used here, so no
    stored block collapses to zero and both the block interiors and the dense
    tail are non-uniform.
    """
    rank = len(shape)
    sparse = tuple(shape[: rank - dense_dim])
    batch = sparse[:-2]
    block_rows = sparse[-2] // blocksize[0]
    block_cols = sparse[-1] // blocksize[1]
    count = 1
    for dim in batch:
        count *= dim
    if count == 0 or block_rows == 0 or block_cols == 0:
        # Natively valid empty form: a zero sparse or dense extent. A zero batch
        # product is rejected natively ('Expected product of batch dimensions to
        # be non-zero') and is covered by the negative table instead.
        return torch.zeros(shape, dtype=dtype, device=_DEVICE)
    mask = _block_mask(block_rows, block_cols)
    masks = torch.stack(
        [torch.roll(mask, b % block_rows, dims=0) for b in range(count)]
    )
    grid = masks.reshape(batch + (block_rows, block_cols)).to(torch.float32)
    grid = grid.repeat_interleave(blocksize[0], dim=-2).repeat_interleave(
        blocksize[1], dim=-1
    )
    if dense_dim:
        tail = tuple(shape[rank - dense_dim :])
        grid = grid.reshape(batch + sparse[-2:] + (1,) * dense_dim).expand(
            batch + sparse[-2:] + tail
        )
    position = torch.arange(grid.numel(), dtype=torch.float32, device=_DEVICE).reshape(
        grid.shape
    )
    magnitude = (position % 7) + 1
    values = torch.where(position % 2 == 0, magnitude, -magnitude)
    return (grid * values).contiguous().to(dtype)


def _holes_input(shape, blocksize, dense_dim, dtype):
    """Small dense input whose single stored block contains interior zeros.

    Only the first block of the block grid is non-zero: its interior entries are
    cleared while its border stays non-zero, so the block is retained but is not
    uniformly non-zero, and every other position is zero. This is the smallest
    input where a stored block holds holes, which the backward has to relocate
    entry by entry instead of by an all-or-nothing block mask.
    """
    if dense_dim:
        raise ValueError("the holes fixture is defined for dense_dim 0")
    inp = torch.zeros(shape, dtype=dtype, device=_DEVICE)
    block_rows = shape[-2] // blocksize[0]
    block_cols = shape[-1] // blocksize[1]
    if block_rows < 1 or block_cols < 1:
        return inp
    ramp = torch.arange(
        1, blocksize[0] * blocksize[1] + 1, dtype=dtype, device=_DEVICE
    ).reshape(tuple(blocksize))
    block = ramp.clone()
    block[1:-1, 1:-1] = 0
    inp[: blocksize[0], : blocksize[1]] = block
    return inp


def _degenerate_input(shape, kind, dtype):
    """Rectangular inputs that exercise the empty and all-specified block sets."""
    inp = torch.zeros(shape, dtype=dtype, device=_DEVICE)
    if kind == "ones":
        inp.fill_(1)
    elif kind == "single":
        inp.reshape(-1)[0] = 2
    elif kind == "opposite":
        # Two opposite signs inside block (0, 0): the conversion stores a block
        # that holds any non-zero entry, it does not test the block sum.
        flat = inp.reshape(-1)
        flat[0] = 2
        flat[1] = -2
    return inp


def _view_input(shape, blocksize, dense_dim, kind, dtype):
    """Return (view, parent) for a strided, offset or expanded view.

    The parent is the full backing tensor, which is what a snapshot has to
    cover: the view alone hides writes into the storage it does not expose.
    """
    if kind == "expand":
        # Zero-stride leading axis. The source keeps every other axis at full
        # extent, so the rank-2 row broadcasts block rows (blocksize[0] == 1)
        # and the rank-4 row broadcasts a batch axis. Expanded dense inputs are
        # natively supported.
        source = (1,) + tuple(shape[1:])
        base = _blocked_pattern(source, blocksize, dense_dim, dtype)
        return base.expand(shape), base
    base = _blocked_pattern(shape, blocksize, dense_dim, dtype)
    if kind == "transpose":
        return base.t(), base
    if kind == "strided":
        return base[:, ::2], base
    if kind == "offset":
        return base[2:], base
    if kind == "batch_transpose":
        return base.transpose(1, 2), base
    return base.transpose(2, 3), base


def _uneven_input(shape, dtype):
    """Batched input whose last batch instance stores no block at all."""
    inp = torch.ones(shape, dtype=dtype, device=_DEVICE)
    tail = 1
    for dim in shape[1:]:
        tail *= dim
    inp.reshape(-1)[tail:] = 0
    return inp


def _structure_input(shape, blocksize, rows, dtype):
    """Rank-2 dense input storing exactly the given block columns per block row."""
    block = tuple(blocksize)
    inp = torch.zeros(shape, dtype=dtype, device=_DEVICE)
    ramp = torch.arange(
        1, block[0] * block[1] + 1, dtype=dtype, device=_DEVICE
    ).reshape(block)
    for row, cols in enumerate(rows):
        for col in cols:
            window = (
                slice(row * block[0], (row + 1) * block[0]),
                slice(col * block[1], (col + 1) * block[1]),
            )
            inp[window] = (row + 1) * ramp
    return inp


def _block_counts(rows):
    return tuple(len(cols) for cols in rows)


def _coo_source(shape, blocksize, dtype):
    """Coalesced rank-2 COO source whose nonempty block columns vary per row."""
    block_rows = shape[0] // blocksize[0]
    block_cols = shape[1] // blocksize[1]
    rows = [tuple(range((row % block_cols) + 1)) for row in range(block_rows)]
    return _structure_input(shape, blocksize, rows, dtype).to_sparse()


def _out_buffer(shape, blocksize, dense_dim, counts, dtype, device, fill):
    """Sparse BSR buffer holding exactly `counts[i]` blocks in block row i.

    crow is non-decreasing and ends at the stored-block count the native .out
    requires the buffer to match exactly; each block row fills its lowest block
    columns, which keeps the indices ascending and in range.
    """
    crow = [0]
    col = []
    for row_count in counts:
        col.extend(range(row_count))
        crow.append(crow[-1] + row_count)
    tail = tuple(shape[len(shape) - dense_dim :]) if dense_dim else ()
    values = torch.full(
        (crow[-1],) + tuple(blocksize) + tail, fill, dtype=dtype, device=device
    )
    # int64 is the dtype this fixture chooses for the BSR index tensors.
    return torch.sparse_bsr_tensor(
        torch.tensor(crow, dtype=torch.int64, device=device),
        torch.tensor(col, dtype=torch.int64, device=device),
        values,
        size=shape,
    )


@pytest.mark.to_sparse_bsr
@pytest.mark.parametrize("shape", _GRID_SHAPES)
@pytest.mark.parametrize("value_range", _GRID_RANGES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test__to_sparse_bsr_value_grid(shape, value_range, dtype):
    inp = _grid_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)
    expected_input = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_sparse_bsr(ref_inp, [1, 1], 0)
    res_out = flag_gems._to_sparse_bsr(inp, [1, 1], 0)

    # Layout and sparse/dense rank are metadata the value comparison of
    # crow/col/values does not cover.
    assert res_out.layout == ref_out.layout
    assert res_out.sparse_dim() == ref_out.sparse_dim()
    assert res_out.dense_dim() == ref_out.dense_dim()
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, expected_input)
    assert res_out.device == inp.device


@pytest.mark.to_sparse_bsr
@pytest.mark.parametrize("shape,blocksize,dense_dim", _BLOCKSIZE_ROWS)
@pytest.mark.parametrize("dtype", _BLOCKSIZE_DTYPES)
def test__to_sparse_bsr_blocksize(shape, blocksize, dense_dim, dtype):
    inp = _grid_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    expected_input = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_sparse_bsr(ref_inp, list(blocksize), dense_dim)
    res_out = flag_gems._to_sparse_bsr(inp, list(blocksize), dense_dim)

    assert res_out.dense_dim() == ref_out.dense_dim()
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, expected_input)
    assert res_out.device == inp.device


@pytest.mark.to_sparse_bsr
@pytest.mark.parametrize("shape", _OPTIONAL_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int8])
def test__to_sparse_bsr_dense_dim_explicit_none(shape, dtype):
    inp = _blocked_pattern(shape, (1, 1), 0, dtype)
    ref_inp = tu.to_reference(inp)
    expected_input = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_sparse_bsr(ref_inp, [1, 1], None)
    res_out = flag_gems._to_sparse_bsr(inp, [1, 1], None)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, expected_input)
    assert res_out.device == inp.device


@pytest.mark.to_sparse_bsr
@pytest.mark.parametrize("shape", _OPTIONAL_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int8])
def test__to_sparse_bsr_dense_dim_omitted(shape, dtype):
    # Omitting dense_dim must behave like the schema default of 0; passing the
    # default explicitly is a different check and is covered above.
    inp = _blocked_pattern(shape, (1, 1), 0, dtype)
    ref_inp = tu.to_reference(inp)
    expected_input = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_sparse_bsr(ref_inp, [1, 1])
    res_out = flag_gems._to_sparse_bsr(inp, [1, 1])

    assert res_out.dense_dim() == ref_out.dense_dim()
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, expected_input)
    assert res_out.device == inp.device


@pytest.mark.to_sparse_bsr
@pytest.mark.parametrize("shape,blocksize,dense_dim", _PATTERN_ROWS)
@pytest.mark.parametrize("dtype", _PATTERN_DTYPES)
def test__to_sparse_bsr_block_pattern(shape, blocksize, dense_dim, dtype):
    inp = _blocked_pattern(shape, blocksize, dense_dim, dtype)
    ref_inp = tu.to_reference(inp)
    expected_input = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_sparse_bsr(ref_inp, list(blocksize), dense_dim)
    res_out = flag_gems._to_sparse_bsr(inp, list(blocksize), dense_dim)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, expected_input)
    assert res_out.device == inp.device


@pytest.mark.to_sparse_bsr
@pytest.mark.parametrize("shape,blocksize,dense_dim,kind", _DEGENERATE_ROWS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int8])
def test__to_sparse_bsr_degenerate(shape, blocksize, dense_dim, kind, dtype):
    inp = _degenerate_input(shape, kind, dtype)
    ref_inp = tu.to_reference(inp)
    expected_input = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_sparse_bsr(ref_inp, list(blocksize), dense_dim)
    res_out = flag_gems._to_sparse_bsr(inp, list(blocksize), dense_dim)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, expected_input)
    assert res_out.device == inp.device


@pytest.mark.to_sparse_bsr
@pytest.mark.parametrize("shape,blocksize,dense_dim,kind", _VIEW_ROWS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int8])
def test__to_sparse_bsr_view(shape, blocksize, dense_dim, kind, dtype):
    inp, parent = _view_input(shape, blocksize, dense_dim, kind, dtype)
    ref_inp = tu.to_reference(inp)
    # The view exposes only part of the parent storage, so the parent is the
    # snapshot that makes a write into the hidden padding visible.
    expected_parent = tu.to_reference(parent)

    ref_out = torch.ops.aten._to_sparse_bsr(ref_inp, list(blocksize), dense_dim)
    res_out = flag_gems._to_sparse_bsr(inp, list(blocksize), dense_dim)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(parent, expected_parent)
    assert res_out.device == inp.device


@pytest.mark.to_sparse_bsr
@pytest.mark.parametrize("shape,blocksize,dense_dim", _OUT_ROWS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test__to_sparse_bsr_out(shape, blocksize, dense_dim, dtype):
    counts = _block_counts(_OUT_BLOCKS)
    # The input stores three blocks, far below the sixteen of a full grid, and
    # both buffers are independent allocations that hold exactly those three
    # blocks the way the native .out requires. Their crow/col differ, so a
    # candidate that refreshes the values without writing the indices cannot
    # match the reference.
    ref_counts = tuple(reversed(counts))
    buf_counts = counts[1:] + counts[:1]
    inp = _structure_input(shape, blocksize, _OUT_BLOCKS, dtype)
    ref_inp = tu.to_reference(inp)
    expected_input = tu.to_reference(inp)
    buf = _out_buffer(shape, blocksize, dense_dim, buf_counts, dtype, _DEVICE, fill=-1)
    ref_buf = _out_buffer(
        shape, blocksize, dense_dim, ref_counts, dtype, ref_inp.device, fill=-2
    )

    res_out = flag_gems._to_sparse_bsr(inp, list(blocksize), dense_dim, out=buf)
    ref_out = torch.ops.aten._to_sparse_bsr.out(
        ref_inp, list(blocksize), dense_dim, out=ref_buf
    )

    assert res_out is buf
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, expected_input)
    assert res_out.device == inp.device


@pytest.mark.to_sparse_bsr
@pytest.mark.skipif(
    not _INT64_OK,
    reason="the .out buffer is constructed with explicit int64 crow/col index"
    " tensors and this backend does not advertise int64 support",
)
def test__to_sparse_bsr_out_block_count_mismatch():
    shape, blocksize, dense_dim = (8, 8), (2, 2), 0
    inp = torch.ones(shape, dtype=torch.float32, device=_DEVICE)
    # Structurally valid buffer (non-decreasing crow, ascending in-range
    # columns) that holds 5 blocks while the input produces 16; the native .out
    # rejects the mismatched capacity instead of resizing the buffer.
    buf = _out_buffer(
        shape, blocksize, dense_dim, (2, 1, 1, 1), torch.float32, _DEVICE, fill=1
    )

    with pytest.raises((RuntimeError, IndexError)):
        flag_gems._to_sparse_bsr(inp, list(blocksize), dense_dim, out=buf)


@pytest.mark.to_sparse_bsr
@pytest.mark.parametrize("shape,blocksize,dense_dim,kind", _BACKWARD_ROWS)
def test__to_sparse_bsr_backward(shape, blocksize, dense_dim, kind):
    dtype = torch.float32
    if kind == "zeros":
        inp = torch.zeros(shape, dtype=dtype, device=_DEVICE)
    elif kind == "holes":
        inp = _holes_input(shape, blocksize, dense_dim, dtype)
    else:
        inp = _blocked_pattern(shape, blocksize, dense_dim, dtype)
    inp = inp.detach().requires_grad_(True)
    ref_inp = tu.to_reference(inp).detach().requires_grad_(True)
    expected_input = tu.to_reference(inp)
    # Distinct upstream values on every path; the backward of this conversion
    # relocates values (each dense element belongs to one block) and never
    # accumulates, so the gradients are compared exactly.
    upstream = torch.arange(
        1.0, float(inp.numel()) + 1.0, dtype=dtype, device=inp.device
    ).reshape(shape)
    ref_upstream = tu.to_reference(upstream)

    ref_out = torch.ops.aten._to_sparse_bsr(ref_inp, list(blocksize), dense_dim)
    res_out = flag_gems._to_sparse_bsr(inp, list(blocksize), dense_dim)
    tu.assert_result_equal(res_out, ref_out)

    (ref_grad,) = torch.autograd.grad(
        ref_out.to_dense(), ref_inp, grad_outputs=ref_upstream
    )
    (grad,) = torch.autograd.grad(res_out.to_dense(), inp, grad_outputs=upstream)
    tu.assert_result_equal(grad, ref_grad)
    tu.assert_result_equal(inp, expected_input)
    assert grad.device == inp.device


@pytest.mark.to_sparse_bsr
@pytest.mark.parametrize("shape,blocksize,call_form", _COO_ROWS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int8])
def test__to_sparse_bsr_sparse_coo_source(shape, blocksize, call_form, dtype):
    # A sparse COO input converts directly when dense_dim is not given; the
    # schema is int? dense_dim=None, so an explicit None is the same absent
    # argument. Giving dense_dim for a sparse input is rejected and stays in the
    # negative table.
    inp = _coo_source(shape, blocksize, dtype)
    ref_inp = tu.to_reference(inp)
    # Raw index/value snapshots, taken without coalescing the source, so an
    # in-place rewrite of the source cannot hide behind a coalesced view.
    expected_indices = tu.to_reference(inp.indices())
    expected_values = tu.to_reference(inp.values())
    expected_coalesced = inp.is_coalesced()

    if call_form == "none":
        ref_out = torch.ops.aten._to_sparse_bsr(ref_inp, list(blocksize), None)
        res_out = flag_gems._to_sparse_bsr(inp, list(blocksize), None)
    else:
        ref_out = torch.ops.aten._to_sparse_bsr(ref_inp, list(blocksize))
        res_out = flag_gems._to_sparse_bsr(inp, list(blocksize))

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp.indices(), expected_indices)
    tu.assert_result_equal(inp.values(), expected_values)
    assert inp.is_coalesced() == expected_coalesced
    assert res_out.device == inp.device


@pytest.mark.to_sparse_bsr
@pytest.mark.parametrize(
    "dtype,scenario",
    tu.selected_cases(list(tu.special_value_cases(_SPECIAL_DTYPES)), quick=[]),
)
def test__to_sparse_bsr_special_values(dtype, scenario):
    # The payload is 5 elements long, so an all-ones blocksize keeps one dense
    # element per block and lets NaN/Inf blocks stay individual.
    inp = tu.make_special_input(dtype, scenario).reshape(5, 1)
    ref_inp = tu.to_reference(inp)
    expected_input = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_sparse_bsr(ref_inp, [1, 1], 0)
    res_out = flag_gems._to_sparse_bsr(inp, [1, 1], 0)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, expected_input)
    assert res_out.device == inp.device


@pytest.mark.to_sparse_bsr
@pytest.mark.parametrize("shape,blocksize,dense_dim,kind,dtype,error", _NEGATIVE_ROWS)
def test__to_sparse_bsr_negative(shape, blocksize, dense_dim, kind, dtype, error):
    if kind == "sparse":
        inp = torch.ones(shape, dtype=dtype, device=_DEVICE).to_sparse()
    elif kind == "csr":
        inp = torch.ones(shape, dtype=dtype, device=_DEVICE).to_sparse_csr()
    elif kind == "uneven":
        inp = _uneven_input(shape, dtype)
    else:
        inp = torch.ones(shape, dtype=dtype, device=_DEVICE)

    with pytest.raises(error):
        flag_gems._to_sparse_bsr(inp, blocksize, dense_dim)
