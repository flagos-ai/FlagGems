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

"""Correctness tests for aten::_to_sparse_bsc.

Axis semantics: the sparse matrix of the result occupies the two dims
immediately before the trailing ``dense_dim`` value dims, and every leading dim
is a batch dim. ``dense_dim=None`` and ``dense_dim=0`` describe the same layout,
so the optional argument is forwarded to the reference and to the candidate
exactly as the case declares it: omitting it makes the native operator read a
different pair of axes, which fails with a divisibility error as soon as those
axes do not tile the block size. A dense_dim above ``rank - 2`` addresses a
sparse axis that does not exist and raises IndexError; a negative or non-int
dense_dim is rejected earlier with RuntimeError; rank < 2 raises IndexError.

Native behaviour the fixtures rely on (measured on the target device):
  * a block whose values are all zero is dropped, so the stored-block count is
    the number of non-zero blocks; the shared sparse comparison reports a
    stored-count mismatch, so the individual cases do not restate it;
  * every batch dim of the result stores the same number of blocks ('Expect the
    same number of specified elements per batch.'). The random value grid is
    therefore split: the families whose layout could let a zero element drop a
    block use the controlled ``_block_input`` fixture, which keeps one rotated
    block occupancy per batch, while the plain value grid keeps random ranges;
  * ``.out`` needs a buffer that already stores as many blocks as the result
    ('torch.copy_: only sparse compressed tensors with the same number of
    specified elements are supported.') and then overwrites it completely,
    indices included, returning that same buffer. A dense-tailed buffer is
    expressible -- ``sparse_bsc_tensor`` takes the dense tail from the trailing
    value dims -- so one ``.out`` row is hybrid;
  * a rank >= 3 input whose batch dims multiply to zero is rejected ('Expected
    product of batch dimensions to be non-zero.'), while a zero extent inside
    the sparse dims is accepted with no stored block -- including a hybrid
    layout, where the empty dim is sparse rather than batched;
  * the backward rows cover the plain, the batched and the batched dense-tailed
    result forms; on the target device each of those forms x {float32, float64,
    bfloat16, float16} produced the gradient of the native reference, and the
    gradient is compared exactly on the reference device with its own device
    checked, so a backend that cannot differentiate one of the forms fails that
    row instead of being skipped.

Case levels: ``--quick`` runs the value grid plus one ``.out`` smoke case and
every negative case; the other positive families are default-only through
``tu.selected_cases(..., quick=[])``. Every positive family is collected only
where the runtime advertises int64 support, because each of their results -- and
the ``.out`` buffers built for them -- is an int64-indexed compressed structure
that this operator itself emits; that flag is static device metadata, so
collection performs no tensor work, and the schema negatives, whose native
rejection happens before any index tensor exists, run everywhere.

Every positive case also compares the candidate's input with the independent
reference copy once the call has returned (the fold only reads) and checks the
result device against the input directly: the shared comparison may move the
reference across devices, so it cannot answer either question.

The operator has a single tensor operand and no scalar parameter, so there is
nothing to broadcast, and no implementation or documented dispatch condition is
available, so implementation-specific kernel branches cannot be established.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# Static runtime capability flags: shared-helper constants that read device
# metadata only, never a tensor, so collection stays free of device work.
_DTYPE_GATES = {
    torch.bfloat16: utils.bf16_is_supported,
    torch.float64: utils.fp64_is_supported,
    torch.int64: utils.int64_is_supported,
    torch.float8_e4m3fn: utils.fp8_is_supported,
    torch.float8_e5m2: utils.fp8_is_supported,
}


def _is_supported(dtype):
    return _DTYPE_GATES.get(dtype, True)


def _gated(dtypes):
    return [dtype for dtype in dtypes if _is_supported(dtype)]


# The spec's nine required dtypes first, then the extra types this operator also
# accepts; the static flags decide which of them the runtime advertises.
_SUPPORTED_DTYPES = _gated(
    [
        torch.int8,
        torch.uint8,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float32,
        torch.bfloat16,
        torch.float16,
        torch.int32,
        torch.int64,
        torch.float64,
        torch.complex64,
        torch.bool,
    ]
)
# The special-value matrix follows the supported float set, so every FP8 dtype
# keeps the scenarios it can represent (e4m3fn has no infinity, e5m2 has nan,
# inf and mixed) instead of being dropped as a family.
_FLOAT_DTYPES = _gated(
    [dtype for dtype in _SUPPORTED_DTYPES if dtype.is_floating_point]
)
_PATTERN_DTYPES = _gated(
    [torch.float32, torch.bfloat16, torch.int8, torch.bool, torch.float8_e4m3fn]
)
_DENSE_DIM_DTYPES = _gated([torch.float32, torch.bfloat16])
_BLOCKSIZE_DTYPES = _gated([torch.float16, torch.float32])
_BACKWARD_DTYPES = _gated([torch.float32, torch.float64, torch.bfloat16, torch.float16])
_VIEW_DTYPES = _gated([torch.float32, torch.float16])
_OUT_DTYPES = _gated([torch.float32, torch.int32])
# The .out rows hand the operator a buffer that already stores blocks, so this
# family depends on int64 twice over; its dtype list encodes that gate, while
# the other negatives build no index tensor and keep running on every runtime.
_OUT_DTYPES_SELECTED = _OUT_DTYPES if utils.int64_is_supported else []

# The five shared ranges drive the value grid; the structural cases below take
# [-1, 1] from the same symbolic table instead of a literal pair.
_CENTERED_RANGE = ["-1", "1"]


def _call_args(blocksize, dense_dim, omit_dense_dim):
    """The native argument list of one case, shared by reference and candidate.

    The optional dense_dim is passed only when the case declares it; the
    omitted spelling stays omitted on both sides, so the candidate is never
    asked to reproduce a layout the reference did not build.
    """
    if omit_dense_dim:
        return (tuple(blocksize),)
    return (tuple(blocksize), dense_dim)


def _value_dims(dense_dim):
    return 0 if dense_dim is None else dense_dim


def _sparse_axes(rank, dense_dim):
    """The two dims holding the sparse matrix of a rank-``rank`` input."""
    value_dims = _value_dims(dense_dim)
    return rank - 2 - value_dims, rank - 1 - value_dims


def _dividing_blocksize(shape, dense_dim):
    """Block shape that exactly tiles the sparse dims of ``shape``."""
    row_axis, col_axis = _sparse_axes(len(shape), dense_dim)
    return tuple(
        next(divisor for divisor in (4, 3, 2, 1) if shape[axis] % divisor == 0)
        for axis in (row_axis, col_axis)
    )


def _dense_dim_row(shape, dense_dim, omit_dense_dim=False):
    return (shape, _dividing_blocksize(shape, dense_dim), dense_dim, omit_dense_dim)


def _positive(rows, quick=()):
    """Collect the positive ``rows`` of a family on this runtime.

    Every positive result is a compressed structure indexed in int64, and the
    controlled fixtures build their dense payload in the dtype under test only,
    so a runtime that does not advertise int64 support cannot produce one: the
    family then collects no case instead of a case it could never execute. The
    flag reads device metadata only, so collection stays free of tensor work.
    """
    if not utils.int64_is_supported:
        return []
    return tu.selected_cases(rows, quick=quick)


def _block_input(shape, blocksize, dense_dim, empty_blocks, dtype):
    """Dense input with a controlled, batch-uniform block occupancy.

    Every block holds a +-1 checkerboard except the ``empty_blocks`` positions,
    which are exactly zero, so the stored-block count is fixed. Stored blocks
    with more than one element also carry one zero element, so a candidate that
    treats any zero element as an empty block -- or that ignores the zero when
    it does store the block -- fails as well. Blocks are enumerated once as
    (batch, row_block, col_block) and written through that same row-major
    coordinate pair. The empty positions are rotated by the batch index, so each
    batch stores the same number of blocks (which the native operator requires)
    at different batch coordinates: a candidate that mixes the batches up still
    fails.
    """
    block_rows, block_cols = blocksize
    row_axis, col_axis = _sparse_axes(len(shape), dense_dim)
    batch_shape = tuple(shape[:row_axis])
    dense_shape = tuple(shape[col_axis + 1 :])
    batches = 1
    for extent in batch_shape:
        batches *= extent
    n_row_blocks = shape[row_axis] // block_rows
    n_col_blocks = shape[col_axis] // block_cols
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    if batches == 0 or n_row_blocks == 0 or n_col_blocks == 0:
        return inp
    empty = set(empty_blocks)
    blocks = [
        (batch, row_block, col_block)
        for batch in range(batches)
        for row_block in range(n_row_blocks)
        for col_block in range(n_col_blocks)
        if ((row_block + batch) % n_row_blocks, col_block) not in empty
    ]
    index = torch.tensor(blocks, dtype=torch.int64, device=flag_gems.device)
    element = torch.arange(block_rows * block_cols, device=flag_gems.device).reshape(
        (1, block_rows, block_cols) + (1,) * len(dense_shape)
    )
    ordinal = torch.arange(len(blocks), device=flag_gems.device).reshape(
        (-1, 1, 1) + (1,) * len(dense_shape)
    )
    values = 1 - 2 * ((element + ordinal) % 2)
    if dense_shape:
        span = 1
        for extent in dense_shape:
            span *= extent
        tail = torch.arange(span, device=flag_gems.device).reshape(
            (1, 1, 1) + dense_shape
        )
        values = values * (1 - 2 * ((tail + ordinal) % 2))
    if block_rows * block_cols > 1:
        values[::3, 0, 0] = 0
    view = inp.reshape(
        (batches, n_row_blocks, block_rows, n_col_blocks, block_cols) + dense_shape
    )
    view[index[:, 0], index[:, 1], :, index[:, 2], :] = values.to(dtype)
    return inp


def _bsc_buffer(shape, blocksize, dense_dim, empty_blocks, dtype, device):
    """Independent BSC tensor assembled from its own components.

    ``.out`` only accepts a buffer that already stores as many blocks as the
    result, so the buffer keeps the result's stored-block count while occupying
    different blocks, and every value is a sentinel: a candidate that returns
    the untouched buffer matches neither the indices nor the values. The dense
    tail of a hybrid result is carried by the trailing dims of the values
    tensor, which is what ``sparse_bsc_tensor`` derives the block size and the
    dense dims from.
    """
    block_rows, block_cols = blocksize
    row_axis, col_axis = _sparse_axes(len(shape), dense_dim)
    batch_shape = tuple(shape[:row_axis])
    dense_shape = tuple(shape[col_axis + 1 :])
    n_row_blocks = shape[row_axis] // block_rows
    n_col_blocks = shape[col_axis] // block_cols
    ccol_indices = [0]
    row_indices = []
    for col_block in range(n_col_blocks):
        row_indices.extend(
            row_block
            for row_block in range(n_row_blocks)
            if (row_block, col_block) not in empty_blocks
        )
        ccol_indices.append(len(row_indices))
    nnz = len(row_indices)
    ccol = torch.tensor(ccol_indices, dtype=torch.int64, device=device)
    row = torch.tensor(row_indices, dtype=torch.int64, device=device)
    values = torch.full(
        (nnz, block_rows, block_cols) + dense_shape, -7, dtype=dtype, device=device
    )
    # A batch-free tensor takes the 1-D indices directly; expanding them to
    # their own shape is a no-op, and to the batch shape replicates them per
    # batch as the batched layout requires.
    return torch.ops.aten.sparse_bsc_tensor(
        ccol.expand(batch_shape + (n_col_blocks + 1,)).contiguous(),
        row.expand(batch_shape + (nnz,)).contiguous(),
        values.expand(batch_shape + values.shape).contiguous(),
        list(shape),
        dtype=dtype,
        device=device,
    )


def _strided_offset_view(shape, dtype):
    """A non-contiguous view plus the backing parent it is taken from.

    The parent prepends a leading axis and doubles the last extent; indexing the
    leading axis out and keeping every second element of the last dim yields the
    requested extent with stride 2 on that dim and a non-zero storage offset, so
    a candidate that assumes a contiguous input reads the wrong elements. The
    parent is returned too, so the case can show that the fold leaves the whole
    backing storage untouched.
    """
    parent = tu.make_input(
        dtype, (2,) + tuple(shape[:-1]) + (2 * shape[-1],), _CENTERED_RANGE
    )
    return parent, parent[1][..., ::2]


def _lazy_conjugate_input(shape):
    """Complex parent plus the lazy conjugate view handed to the operator."""
    real = tu.make_input(torch.float32, shape, _CENTERED_RANGE)
    imag = tu.make_input(torch.float32, shape, _CENTERED_RANGE)
    parent = torch.complex(real, imag)
    return parent, parent.conj()


def _tiled_special_input(shape, dtype, scenario):
    """Special-value payload for ``shape``.

    The shared generator returns a short 1-D payload that is tiled to the number
    of element slots the requested shape needs; the special values (and any
    zeros) are therefore retained unchanged, and the rank-2 shapes used by the
    caller keep the per-batch stored-block rule out of the picture.
    """
    payload = tu.make_special_input(dtype, scenario)
    count = 1
    for extent in shape:
        count *= extent
    reps = (count + payload.numel() - 1) // payload.numel()
    return payload.repeat(reps)[:count].reshape(shape)


# (shape, blocksize, dense_dim, omit_dense_dim). Both spellings of the default
# are exercised: dense_dim omitted from the call and an explicit None.
_STRUCTURE_ROWS = [
    ((2, 19, 7), (19, 7), None, True),
    ((2, 19, 7), (19, 7), None, False),
    ((1024, 1024), (2, 2), None, True),
    ((1024, 1024), (2, 2), None, False),
    ((20, 320, 15), (5, 5), None, True),
    ((20, 320, 15), (5, 5), None, False),
    ((16, 128, 64, 60), (16, 15), None, False),
    ((16, 7, 57, 32, 29), (19, 16), 1, False),
]
_VALUE_GRID_ROWS = _positive(_STRUCTURE_ROWS, quick=[_STRUCTURE_ROWS[0]])
_ALL_ZERO_ROWS = _positive(_STRUCTURE_ROWS)

# (shape, blocksize, dense_dim, omit_dense_dim, empty_blocks): empty blocks cover
# a leading, an interior and a trailing block, a completely empty block column,
# the one-block/full-block cases, a 1x1-block batched tensor, and dense-tail
# layouts whose sparse axes only exist once dense_dim is forwarded.
_PATTERN_ROWS = [
    ((2, 4, 4), (2, 2), None, False, ()),
    ((2, 4, 4), (2, 2), None, True, ()),
    ((2, 4, 4), (2, 2), None, False, ((0, 0),)),
    ((2, 4, 4), (2, 2), None, False, ((1, 1),)),
    ((2, 4, 4), (2, 2), None, False, ((0, 0), (1, 1))),
    ((3, 8, 8), (2, 2), None, False, ((0, 1),)),
    ((3, 8, 8), (2, 2), None, False, ((1, 1), (2, 0))),
    ((3, 8, 8), (2, 2), None, False, ((0, 0), (1, 0), (2, 0), (3, 0))),
    ((4, 8, 8), (2, 4), None, False, ((0, 1), (2, 0))),
    ((2, 19, 7), (1, 1), None, False, ()),
    ((2, 19, 7), (19, 7), None, False, ()),
    ((16, 16), (16, 16), None, False, ()),
    ((16, 32), (16, 16), None, False, ((0, 1),)),
    ((16, 16), (1, 1), None, False, ()),
    ((2, 8, 8, 3), (2, 2), 1, False, ((0, 1), (2, 0), (3, 3))),
    ((2, 8, 8, 3), (2, 2), 1, False, ()),
]
_PATTERN_ROWS_SELECTED = _positive(_PATTERN_ROWS)

# (shape, blocksize, dense_dim, omit_dense_dim): a zero extent inside the sparse
# dims (or no batch dim at all) is a valid input. The single shared range is
# enough because an empty tensor has no element whose range could matter.
_ZERO_EXTENT_ROWS = _positive(
    [
        ((0, 4), (1, 1), None, False),
        ((4, 0), (1, 1), None, False),
        ((0, 0), (1, 1), None, False),
        ((2, 0, 4), (1, 1), None, False),
        ((2, 8, 0, 4), (1, 1), None, False),
        ((0, 4, 3), (1, 1), 1, False),
    ]
)

# Measured native rejection: a rank >= 3 input whose batch dims multiply to zero
# fails with 'to_sparse_bsc: Expected product of batch dimensions to be
# non-zero.' even though the sparse dims are empty. The zero-extent rows above
# keep every batch dim non-zero, so this is a real limit of the operator and not
# the shape of the fixture.
_ZERO_BATCH_ROWS = [((0, 2, 4), (1, 1)), ((0, 0, 4), (1, 1))]

# dense_dim sweeps every valid value 0..rank-2 for each rank; None is the
# default and is equivalent to 0. The controlled fixture keeps the stored-block
# count batch-uniform: with random values and a 1x1 block size, a single zero
# element would otherwise drop a block in one batch only.
_DENSE_DIM_ROWS = _positive(
    [
        ((2, 19, 7), (1, 1), None, True),
        _dense_dim_row((2, 19, 7), 0),
        _dense_dim_row((20, 320, 15), None),
        _dense_dim_row((20, 320, 15), 0),
        _dense_dim_row((20, 320, 15), 1),
        _dense_dim_row((16, 128, 64, 60), None),
        _dense_dim_row((16, 128, 64, 60), 0),
        _dense_dim_row((16, 128, 64, 60), 1),
        _dense_dim_row((16, 128, 64, 60), 2),
        _dense_dim_row((2, 3, 6, 4, 5), 0),
        _dense_dim_row((2, 3, 6, 4, 5), 1),
        _dense_dim_row((2, 3, 6, 4, 5), 2),
        _dense_dim_row((2, 3, 6, 4, 5), 3),
    ]
)

_BLOCKSIZE_ROWS_SELECTED = _positive(
    [
        ((16, 16), (1, 1), None, True),
        ((16, 16), (2, 2), None, True),
        ((16, 16), (4, 4), None, False),
        ((16, 16), (8, 8), None, False),
        ((16, 16), (16, 16), None, False),
        ((20, 320, 15), (1, 1), None, True),
        ((20, 320, 15), (5, 15), None, False),
        ((20, 320, 15), (10, 3), None, False),
        ((20, 320, 15), (20, 5), None, False),
        ((16, 128, 64, 60), (16, 15), None, False),
    ]
)

# Every rank the backward exercises, including the batched (4, 8, 8) result and
# the batched dense-tailed (2, 8, 8, 3) one: the gradient must scatter the
# upstream values back to the same batch coordinates.
_BACKWARD_ROWS = _positive(
    [
        ((64, 64), (4, 4), None, True),
        ((20, 320, 15), (5, 5), 1, False),
        ((16, 128, 64, 60), (16, 16), 2, False),
        ((2, 3, 6, 4, 5), (1, 3), 3, False),
        ((4, 8, 8), (2, 2), None, False),
        ((2, 8, 8, 3), (2, 2), 1, False),
    ]
)

# Rank-2 rows: without a batch dim the single stored-block count cannot differ
# between batches, so special values (including all-zero blocks) stay valid.
_SPECIAL_ROWS = _positive(
    [
        ((4, 4), (2, 2), None, True),
        ((8, 8), (4, 4), None, True),
    ]
)

_VIEW_ROWS = _positive(
    [
        ((16, 16), (2, 2), None, False),
        ((20, 320, 15), (5, 5), None, False),
    ]
)

_LAZY_ROWS = _positive([(8, 8)])

# (shape, blocksize, dense_dim, omit_dense_dim, result empty block, buffer empty
# block): the buffer stores the same number of blocks but occupies a different
# one, and the result overwrites it completely. The last row is dense-tailed, so
# the buffer carries the hybrid value layout as well.
_OUT_ROWS = [
    ((16, 16), (2, 2), None, True, (0, 0), (7, 7)),
    ((20, 320, 15), (5, 5), None, True, (0, 0), (3, 1)),
    ((4, 8, 8), (2, 2), None, True, (1, 1), (0, 0)),
    ((2, 8, 8, 3), (2, 2), 1, False, (1, 1), (0, 3)),
]
_OUT_ROWS_SELECTED = _positive(_OUT_ROWS, quick=[_OUT_ROWS[0]])

# Measured native rejections: a negative dense_dim raises RuntimeError, a
# dense_dim above rank-2 (including one that overflows the index computation)
# raises IndexError, and a non-int dense_dim is rejected before the operator's
# own checks run, so its class may be the schema TypeError instead.
_INVALID_DENSE_DIM_ROWS = [
    ((16, 16), (2, 2), -1, RuntimeError),
    ((16, 16), (2, 2), 1, IndexError),
    ((20, 320, 15), (5, 5), 2, IndexError),
    ((16, 128, 64, 60), (16, 15), 4, IndexError),
    ((2, 3, 6, 4, 5), (1, 3), 2**40, IndexError),
    ((16, 16), (2, 2), 1.5, (RuntimeError, TypeError)),
    ((16, 16), (2, 2), "0", (RuntimeError, TypeError)),
]

# Measured native rejections: a wrong-length, non-positive or non-dividing
# blocksize raises RuntimeError. ``blocksize`` is declared int[2], so for the two
# argument-type rows the rejection is measured as either the schema error
# (RuntimeError) or the list conversion error (TypeError).
_INVALID_BLOCKSIZE_ROWS = [
    ((16, 16), [2], RuntimeError),
    ((16, 16), [2, 2, 2], RuntimeError),
    ((16, 16), [0, 2], RuntimeError),
    ((16, 16), [2, 0], RuntimeError),
    ((16, 16), [-2, 2], RuntimeError),
    ((16, 16), [2, -2], RuntimeError),
    ((16, 16), [3, 3], RuntimeError),
    ((16, 16), [2.0, 2.0], (RuntimeError, TypeError)),
    ((16, 16), "2,2", (RuntimeError, TypeError)),
]


@pytest.mark.to_sparse_bsc
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape,blocksize,dense_dim,omit_dense_dim", _VALUE_GRID_ROWS)
def test__to_sparse_bsc_value_grid(
    shape, blocksize, dense_dim, omit_dense_dim, value_range, dtype
):
    args = _call_args(blocksize, dense_dim, omit_dense_dim)
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_sparse_bsc(ref_inp, *args)
    res_out = flag_gems._to_sparse_bsc(inp, *args)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    assert res_out.device == inp.device


@pytest.mark.to_sparse_bsc
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
@pytest.mark.parametrize("shape,blocksize,dense_dim,omit_dense_dim", _ALL_ZERO_ROWS)
def test__to_sparse_bsc_all_zero(shape, blocksize, dense_dim, omit_dense_dim, dtype):
    args = _call_args(blocksize, dense_dim, omit_dense_dim)
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_sparse_bsc(ref_inp, *args)
    res_out = flag_gems._to_sparse_bsc(inp, *args)

    # Every block is zero, so the result stores none of them.
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    assert res_out.device == inp.device


@pytest.mark.to_sparse_bsc
@pytest.mark.parametrize("dtype", _PATTERN_DTYPES)
@pytest.mark.parametrize(
    "shape,blocksize,dense_dim,omit_dense_dim,empty_blocks", _PATTERN_ROWS_SELECTED
)
def test__to_sparse_bsc_pattern_blocks(
    shape, blocksize, dense_dim, omit_dense_dim, empty_blocks, dtype
):
    args = _call_args(blocksize, dense_dim, omit_dense_dim)
    inp = _block_input(shape, blocksize, dense_dim, empty_blocks, dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_sparse_bsc(ref_inp, *args)
    res_out = flag_gems._to_sparse_bsc(inp, *args)

    # The fold must drop exactly the all-zero blocks and keep the partly zero
    # ones, which the stored-block count of the shared comparison reports.
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    assert res_out.device == inp.device


@pytest.mark.to_sparse_bsc
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
@pytest.mark.parametrize("shape,blocksize,dense_dim,omit_dense_dim", _ZERO_EXTENT_ROWS)
def test__to_sparse_bsc_zero_extent(shape, blocksize, dense_dim, omit_dense_dim, dtype):
    args = _call_args(blocksize, dense_dim, omit_dense_dim)
    inp = tu.make_input(dtype, shape, _CENTERED_RANGE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_sparse_bsc(ref_inp, *args)
    res_out = flag_gems._to_sparse_bsc(inp, *args)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    assert res_out.device == inp.device


@pytest.mark.to_sparse_bsc
@pytest.mark.parametrize("shape,blocksize", _ZERO_BATCH_ROWS)
def test__to_sparse_bsc_zero_batch(shape, blocksize):
    inp = tu.make_input(torch.float32, shape, _CENTERED_RANGE)

    with pytest.raises(RuntimeError):
        flag_gems._to_sparse_bsc(inp, blocksize)


@pytest.mark.to_sparse_bsc
@pytest.mark.parametrize("dtype", _DENSE_DIM_DTYPES)
@pytest.mark.parametrize("shape,blocksize,dense_dim,omit_dense_dim", _DENSE_DIM_ROWS)
def test__to_sparse_bsc_dense_dim(shape, blocksize, dense_dim, omit_dense_dim, dtype):
    args = _call_args(blocksize, dense_dim, omit_dense_dim)
    inp = _block_input(shape, blocksize, dense_dim, (), dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_sparse_bsc(ref_inp, *args)
    res_out = flag_gems._to_sparse_bsc(inp, *args)

    # Which dims became sparse and which stayed dense is this family's subject,
    # and it is visible in the comparison itself: the sparse/dense axis
    # assignment decides the shape of the stored block values, whose shape and
    # content the shared sparse comparison checks.
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    assert res_out.device == inp.device


@pytest.mark.to_sparse_bsc
@pytest.mark.parametrize("dtype", _BLOCKSIZE_DTYPES)
@pytest.mark.parametrize(
    "shape,blocksize,dense_dim,omit_dense_dim", _BLOCKSIZE_ROWS_SELECTED
)
def test__to_sparse_bsc_blocksize(shape, blocksize, dense_dim, omit_dense_dim, dtype):
    args = _call_args(blocksize, dense_dim, omit_dense_dim)
    inp = _block_input(shape, blocksize, dense_dim, (), dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_sparse_bsc(ref_inp, *args)
    res_out = flag_gems._to_sparse_bsc(inp, *args)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    assert res_out.device == inp.device


@pytest.mark.to_sparse_bsc
@pytest.mark.parametrize("dtype", _OUT_DTYPES_SELECTED)
@pytest.mark.parametrize(
    "shape,blocksize,dense_dim,omit_dense_dim,result_empty,out_empty",
    _OUT_ROWS_SELECTED,
)
def test__to_sparse_bsc_out(
    shape, blocksize, dense_dim, omit_dense_dim, result_empty, out_empty, dtype
):
    args = _call_args(blocksize, dense_dim, omit_dense_dim)
    inp = _block_input(shape, blocksize, dense_dim, (result_empty,), dtype)
    ref_inp = tu.to_reference(inp)
    # Two independently built buffers: same stored-block count as the result,
    # different occupied block, sentinel values. Each lives on the device its
    # own side runs on, since the shared comparison moves the result, not the
    # reference.
    res_buffer = _bsc_buffer(
        shape, blocksize, dense_dim, (out_empty,), dtype, inp.device
    )
    ref_buffer = _bsc_buffer(
        shape, blocksize, dense_dim, (out_empty,), dtype, ref_inp.device
    )

    torch.ops.aten._to_sparse_bsc.out(ref_inp, *args, out=ref_buffer)
    res_ret = flag_gems._to_sparse_bsc(inp, *args, out=res_buffer)

    # The candidate must write into the buffer it was given, not a new tensor.
    assert res_ret is res_buffer
    tu.assert_result_equal(res_ret, ref_buffer)
    tu.assert_result_equal(inp, ref_inp)
    assert res_ret.device == inp.device


@pytest.mark.to_sparse_bsc
@pytest.mark.parametrize("dtype", _OUT_DTYPES_SELECTED)
def test__to_sparse_bsc_out_requires_matching_nnz(dtype):
    shape, blocksize = (16, 16), (2, 2)
    inp = _block_input(shape, blocksize, None, ((0, 0),), dtype)
    # The buffer stores every block while the result drops one.
    buffer = _bsc_buffer(shape, blocksize, None, (), dtype, inp.device)

    with pytest.raises(RuntimeError):
        flag_gems._to_sparse_bsc(inp, blocksize, out=buffer)


@pytest.mark.to_sparse_bsc
@pytest.mark.parametrize("shape,blocksize,dense_dim,error", _INVALID_DENSE_DIM_ROWS)
def test__to_sparse_bsc_invalid_dense_dim(shape, blocksize, dense_dim, error):
    inp = tu.make_input(torch.float32, shape, _CENTERED_RANGE)

    with pytest.raises(error):
        flag_gems._to_sparse_bsc(inp, blocksize, dense_dim)


@pytest.mark.to_sparse_bsc
@pytest.mark.parametrize("shape,blocksize,error", _INVALID_BLOCKSIZE_ROWS)
def test__to_sparse_bsc_invalid_blocksize(shape, blocksize, error):
    inp = tu.make_input(torch.float32, shape, _CENTERED_RANGE)

    with pytest.raises(error):
        flag_gems._to_sparse_bsc(inp, blocksize)


@pytest.mark.to_sparse_bsc
@pytest.mark.parametrize("shape", [(), (5,)])
def test__to_sparse_bsc_low_rank(shape):
    # The sparse axes do not exist below rank 2; both ranks are real negatives.
    inp = tu.make_input(torch.float32, shape, _CENTERED_RANGE)

    with pytest.raises(IndexError):
        flag_gems._to_sparse_bsc(inp, (1, 1))


@pytest.mark.to_sparse_bsc
@pytest.mark.parametrize("dtype", _BACKWARD_DTYPES)
@pytest.mark.parametrize("shape,blocksize,dense_dim,omit_dense_dim", _BACKWARD_ROWS)
def test__to_sparse_bsc_backward(shape, blocksize, dense_dim, omit_dense_dim, dtype):
    args = _call_args(blocksize, dense_dim, omit_dense_dim)
    # The candidate keeps the target tensor itself; the reference is a separate
    # differentiable copy on the reference device, so neither side shares a
    # buffer or a graph and the candidate never leaves the target device.
    inp = tu.make_input(dtype, shape, _CENTERED_RANGE)
    inp.requires_grad_(True)
    ref_inp = tu.to_reference(inp)
    # A non-uniform upstream gradient on each side's own device: an identity
    # backward cannot reproduce it.
    up_dense = tu.make_input(dtype, shape, _CENTERED_RANGE)
    upstream = torch.ops.aten._to_sparse_bsc(up_dense, *args)
    ref_upstream = torch.ops.aten._to_sparse_bsc(tu.to_reference(up_dense), *args)

    ref_out = torch.ops.aten._to_sparse_bsc(ref_inp, *args)
    res_out = flag_gems._to_sparse_bsc(inp, *args)
    # A wrong forward already fails here, before the gradient is looked at.
    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)

    (ref_grad,) = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_upstream)
    (res_grad,) = torch.autograd.grad(res_out, inp, grad_outputs=upstream)

    assert res_grad.device == inp.device
    tu.assert_result_equal(res_grad, ref_grad)


@pytest.mark.to_sparse_bsc
@pytest.mark.parametrize("dtype,scenario", tu.special_value_cases(_FLOAT_DTYPES))
@pytest.mark.parametrize("shape,blocksize,dense_dim,omit_dense_dim", _SPECIAL_ROWS)
def test__to_sparse_bsc_special_values(
    shape, blocksize, dense_dim, omit_dense_dim, dtype, scenario
):
    args = _call_args(blocksize, dense_dim, omit_dense_dim)
    inp = _tiled_special_input(shape, dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_sparse_bsc(ref_inp, *args)
    res_out = flag_gems._to_sparse_bsc(inp, *args)

    # NaN, +-inf and the signed zeros must survive the fold: matching NaNs are
    # compared as matching, and inf must stay inf, never a finite value.
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    assert res_out.device == inp.device


@pytest.mark.to_sparse_bsc
@pytest.mark.parametrize("dtype", _VIEW_DTYPES)
@pytest.mark.parametrize("shape,blocksize,dense_dim,omit_dense_dim", _VIEW_ROWS)
def test__to_sparse_bsc_strided_offset_input(
    shape, blocksize, dense_dim, omit_dense_dim, dtype
):
    args = _call_args(blocksize, dense_dim, omit_dense_dim)
    parent, inp = _strided_offset_view(shape, dtype)
    # The reference view is taken from the transferred parent, so it keeps the
    # storage offset and the strides the input view has.
    ref_parent = tu.to_reference(parent)
    ref_inp = ref_parent[1][..., ::2]

    ref_out = torch.ops.aten._to_sparse_bsc(ref_inp, *args)
    res_out = flag_gems._to_sparse_bsc(inp, *args)

    tu.assert_result_equal(res_out, ref_out)
    # The candidate read the same geometry the reference view describes, and it
    # only read: the view metadata and the whole backing storage are unchanged.
    assert (inp.shape, inp.stride(), inp.storage_offset()) == (
        ref_inp.shape,
        ref_inp.stride(),
        ref_inp.storage_offset(),
    )
    tu.assert_result_equal(inp, ref_inp)
    assert res_out.device == inp.device
    tu.assert_result_equal(parent, ref_parent)


@pytest.mark.to_sparse_bsc
@pytest.mark.parametrize("shape", _LAZY_ROWS)
def test__to_sparse_bsc_lazy_conjugate_input(shape):
    parent, inp = _lazy_conjugate_input(shape)
    # Snapshot the parent before either call: the reference view is derived from
    # that snapshot, so both sides start from the same values.
    ref_parent = tu.to_reference(parent)
    ref_inp = ref_parent.conj()

    ref_out = torch.ops.aten._to_sparse_bsc(ref_inp, (2, 2))
    res_out = flag_gems._to_sparse_bsc(inp, (2, 2))

    tu.assert_result_equal(res_out, ref_out)
    # The conjugate bit is still lazy on both views, the logical input values
    # still match the snapshot, and the fold left the parent untouched.
    assert inp.is_conj() and ref_inp.is_conj()
    tu.assert_result_equal(inp, ref_inp)
    assert res_out.device == inp.device
    tu.assert_result_equal(parent, ref_parent)
