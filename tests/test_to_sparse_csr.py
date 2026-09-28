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

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# to_sparse_csr always builds a 2-D CSR tensor (sparse_dim() is 2), so the
# rank-0 and rank-1 spec shapes have no matrix to convert: native rejects both
# (rank 1 fails inside expand, rank 0 reports no dimensions). The rejection is
# covered by the negative tests; the positive grid keeps the rank >= 2 shapes
# of the current level.
_GRID_SHAPES = [shape for shape in tu.selected_shapes() if len(shape) >= 2]

_FP8_DTYPES = {
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
}


def _gate(dtypes):
    # Static backend capability flags only: collection must not allocate or
    # probe a tensor.
    supported = []
    for dtype in dtypes:
        if dtype in _FP8_DTYPES and not utils.fp8_is_supported:
            continue
        if dtype == torch.bfloat16 and not utils.bf16_is_supported:
            continue
        if dtype == torch.int64 and not utils.int64_is_supported:
            continue
        if dtype == torch.float64 and not utils.fp64_is_supported:
            continue
        supported.append(dtype)
    return supported


# int16, complex64 and bool are accepted by native in addition to the nine
# required dtypes.
_DTYPES = _gate(
    tu.REQUIRED_DTYPES + [torch.float64, torch.int16, torch.complex64, torch.bool]
)


def _dense_dim_args(dense_dim):
    # None means omit the argument; native then defaults to dense_dim 0.
    return () if dense_dim is None else (dense_dim,)


def _sparse_dims(shape, dense_dim):
    # The two matrix dimensions of the requested split.
    start = len(shape) - 2 - dense_dim
    return shape[start], shape[start + 1]


def _expand_tail(tiled, shape, start):
    # ``tiled`` already carries the batch axes and the two sparse axes; give it
    # the dense tail and materialize a fresh contiguous tensor of ``shape``.
    batch_shape = tuple(shape[:start])
    tail = tuple(shape[start + 2 :])
    if not tail:
        return tiled.reshape(shape).contiguous()
    per_batch = tiled.numel() // max(1, math.prod(batch_shape))
    flat = tiled.reshape(batch_shape + (per_batch, 1))
    return (
        flat.expand(batch_shape + (per_batch, math.prod(tail)))
        .reshape(shape)
        .contiguous()
    )


def _tile(block, shape, dense_dim):
    # Replicate a 2-D block over the batch axes and the dense tail.
    start = len(shape) - 2 - dense_dim
    batch_shape = tuple(shape[:start])
    tiled = block.reshape((1,) * len(batch_shape) + tuple(block.shape)).expand(
        batch_shape + tuple(block.shape)
    )
    return _expand_tail(tiled, shape, start)


def _make_input(dtype, shape, value_range, dense_dim=0):
    """Dense input for one dense_dim split.

    A batched CSR result must store the same number of elements in every
    batch; an independent random fill cannot guarantee that (rounding, and any
    value range containing zero, change the per-batch count, and native raises
    "Expect the same number of specified elements per batch."). Filling one
    matrix and replicating it keeps the per-batch count identical.
    """
    start = len(shape) - 2 - dense_dim
    if start <= 0:
        return tu.make_input(dtype, shape, value_range)
    rows, cols = _sparse_dims(shape, dense_dim)
    return _tile(tu.make_input(dtype, (rows, cols), value_range), shape, dense_dim)


def _dense_roundtrip(res_out, dtype):
    """Materialize the CSR result, or None where the backend has no kernel.

    Measured on the active backend: to_dense() raises "index_add" not
    implemented for 'Float8_e4m3fn', so fp8 cases rely on the exact
    values / crow / col comparison instead.
    """
    if dtype in _FP8_DTYPES:
        return None
    return res_out.to_dense()


def _assert_csr_metadata(res_out, ref_out, inp, index_dtype):
    """Structural and value checks the shared assertions do not cover.

    ``index_dtype`` is the index dtype the source layout must produce: int64
    for a dense source, the source's own dtype for a compressed source.
    """
    assert res_out.layout == torch.sparse_csr
    assert res_out.dense_dim() == ref_out.dense_dim()
    assert res_out.sparse_dim() == ref_out.sparse_dim() == 2
    assert res_out.shape == ref_out.shape == tuple(inp.shape)
    assert res_out.device == inp.device
    assert res_out.values().dtype == inp.dtype
    assert res_out.crow_indices().dtype == index_dtype
    assert res_out.col_indices().dtype == index_dtype
    tu.assert_result_equal(res_out.crow_indices(), ref_out.crow_indices())
    tu.assert_result_equal(res_out.col_indices(), ref_out.col_indices())
    tu.assert_result_equal(res_out.values(), ref_out.values())


@pytest.mark.to_sparse_csr
@pytest.mark.parametrize("shape", _GRID_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_to_sparse_csr(shape, value_range, dtype):
    inp = _make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_sparse_csr(ref_inp)
    res_out = flag_gems.to_sparse_csr(inp)

    # A dense source produces a new tensor and leaves the input untouched.
    assert res_out is not inp
    tu.assert_result_equal(inp, ref_inp)
    dense = _dense_roundtrip(res_out, dtype)
    if dense is not None:
        tu.assert_result_equal(dense, inp)
    tu.assert_result_equal(res_out, ref_out)
    _assert_csr_metadata(res_out, ref_out, inp, torch.int64)


# Default suite only.
_DENSE_DIM_ROWS = tu.selected_cases(
    [
        ((8, 6), None),
        ((8, 6), 0),
        ((3, 8, 6), None),
        ((3, 8, 6), 0),
        # rank 3, dense_dim 1: sparse_dim 2, dense_dim 1, batch_dims 0 - the
        # whole rank-3 input is one matrix with a dense tail, not a batch.
        ((3, 8, 6), 1),
        ((2, 3, 8, 6), 0),
        ((2, 3, 8, 6), 1),
        ((2, 3, 8, 6), 2),
        ((2, 3, 4, 8, 6), 0),
        ((2, 3, 4, 8, 6), 1),
        ((2, 3, 4, 8, 6), 2),
        ((2, 3, 4, 8, 6), 3),
    ],
    quick=[],
)


@pytest.mark.to_sparse_csr
@pytest.mark.parametrize("shape,dense_dim", _DENSE_DIM_ROWS)
def test_to_sparse_csr_dense_dim(shape, dense_dim):
    # Every valid split 0 <= dense_dim <= rank - 2 is accepted; an omitted
    # dense_dim defaults to 0 and must agree with an explicit 0.
    effective = 0 if dense_dim is None else dense_dim
    inp = _make_input(torch.float32, shape, ["-1", "1"], effective)
    ref_inp = tu.to_reference(inp)
    before = inp.clone()
    args = () if dense_dim is None else (dense_dim,)

    ref_out = torch.ops.aten.to_sparse_csr(ref_inp, *args)
    res_out = flag_gems.to_sparse_csr(inp, *args)

    assert res_out is not inp
    tu.assert_result_equal(inp, before)
    assert res_out.sparse_dim() == 2
    assert res_out.dense_dim() == effective
    assert res_out.shape == ref_out.shape
    tu.assert_result_equal(res_out, ref_out)
    _assert_csr_metadata(res_out, ref_out, inp, torch.int64)


def _non_contiguous_input(dtype, dense_dim):
    """A strided view together with the parent tensor it must read through."""
    if dense_dim == 0:
        parent = tu.make_input(dtype, (8, 24), ["-1", "1"])
        return parent, parent[:, ::4]
    parent = tu.make_input(dtype, (8, 24, dense_dim * 2), ["-1", "1"])
    return parent, parent[:, ::4, ::2]


# Layout family: default suite only.
_NON_CONTIGUOUS_DIMS = tu.selected_cases([0, 1], quick=[])


@pytest.mark.to_sparse_csr
@pytest.mark.parametrize("dense_dim", _NON_CONTIGUOUS_DIMS)
@pytest.mark.parametrize("dtype", _gate([torch.float32, torch.int32, torch.float16]))
def test_to_sparse_csr_non_contiguous(dense_dim, dtype):
    # A strided source must be read through its strides: the same values, not
    # the raw storage, form the CSR output.
    parent, inp = _non_contiguous_input(dtype, dense_dim)
    before = parent.clone()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_sparse_csr(ref_inp, *_dense_dim_args(dense_dim))
    res_out = flag_gems.to_sparse_csr(inp, *_dense_dim_args(dense_dim))

    assert res_out is not inp
    # Neither the view nor any other element of its backing storage is written.
    tu.assert_result_equal(parent, before)
    assert res_out.dense_dim() == dense_dim
    dense = _dense_roundtrip(res_out, dtype)
    if dense is not None:
        tu.assert_result_equal(dense, inp)
    tu.assert_result_equal(res_out, ref_out)
    _assert_csr_metadata(res_out, ref_out, inp, torch.int64)


def _sparse_source(layout, index_dtype):
    values = torch.tensor([3.0, 4.0], device=flag_gems.device)
    if layout == "csr":
        crow = torch.tensor([0, 1, 1, 2, 2], dtype=index_dtype, device=flag_gems.device)
        col = torch.tensor([1, 0], dtype=index_dtype, device=flag_gems.device)
        return torch.sparse_csr_tensor(crow, col, values, size=(4, 4))
    if layout == "csc":
        ccol = torch.tensor([0, 1, 1, 2, 2], dtype=index_dtype, device=flag_gems.device)
        row = torch.tensor([1, 0], dtype=index_dtype, device=flag_gems.device)
        return torch.sparse_csc_tensor(ccol, row, values, size=(4, 4))
    indices = torch.tensor([[0, 3], [1, 0]], dtype=index_dtype, device=flag_gems.device)
    return torch.sparse_coo_tensor(indices, values, size=(4, 4)).coalesce()


def _source_parts(source):
    """Every stored component of a sparse source tensor."""
    if source.layout == torch.sparse_csr:
        return [source.crow_indices(), source.col_indices(), source.values()]
    if source.layout == torch.sparse_csc:
        return [source.ccol_indices(), source.row_indices(), source.values()]
    return [source.indices(), source.values()]


# Sparse compressed sources keep their index dtype; a COO source is promoted to
# int64 by native. The int64 rows are dropped on a backend without static int64
# support so no required structural index tensor exceeds the device capability.
# Default suite only.
_LAYOUT_ROWS = tu.selected_cases(
    [("csr", torch.int32), ("csc", torch.int32)]
    + (
        [
            ("csr", torch.int64),
            ("csc", torch.int64),
            ("coo", torch.int64),
        ]
        if utils.int64_is_supported
        else []
    ),
    quick=[],
)


@pytest.mark.to_sparse_csr
@pytest.mark.parametrize("layout,index_dtype", _LAYOUT_ROWS)
def test_to_sparse_csr_sparse_source(layout, index_dtype):
    source = _sparse_source(layout, index_dtype)
    ref_source = _sparse_source(layout, index_dtype)
    # Independent copies of every stored component: the source's own accessors
    # return live views, so comparing against them after the call would compare
    # mutated storage with itself.
    before = [part.clone() for part in _source_parts(source)]
    before_meta = (source.shape, source.dense_dim(), source.sparse_dim(), source.layout)

    ref_out = torch.ops.aten.to_sparse_csr(ref_source)
    res_out = flag_gems.to_sparse_csr(source)

    tu.assert_result_equal(res_out, ref_out)
    _assert_csr_metadata(res_out, ref_out, source, index_dtype)
    if layout == "csr":
        # An already-CSR source is returned as the same object.
        assert res_out is source
    # The source is never mutated, whichever path native takes: every index
    # component, the stored values and the metadata must be unchanged.
    for current, original in zip(_source_parts(source), before):
        tu.assert_result_equal(current, original)
    assert (
        source.shape,
        source.dense_dim(),
        source.sparse_dim(),
        source.layout,
    ) == before_meta


_EMPTY_DTYPES = _gate([torch.float32, torch.float16, torch.int32, torch.int8])

# Default suite only.
_ZERO_EXTENT_ROWS = tu.selected_cases(
    [
        ((0, 3), 0),
        ((3, 0), 0),
        ((0, 0), 0),
        ((2, 0, 3), 0),
        ((2, 0, 3), 1),
        ((2, 3, 0), 0),
        ((2, 3, 0), 1),
    ],
    quick=[],
)


@pytest.mark.to_sparse_csr
@pytest.mark.parametrize("shape,dense_dim", _ZERO_EXTENT_ROWS)
@pytest.mark.parametrize("dtype", _EMPTY_DTYPES)
def test_to_sparse_csr_zero_extent(shape, dense_dim, dtype):
    # A zero-sized matrix (or zero-sized dense tail) is legal as long as the
    # batch product is non-zero; test_to_sparse_csr_rejects_zero_batch covers
    # the zero-batch-product rejection.
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    before = inp.clone()

    ref_out = torch.ops.aten.to_sparse_csr(ref_inp, dense_dim)
    res_out = flag_gems.to_sparse_csr(inp, dense_dim)

    assert res_out is not inp
    tu.assert_result_equal(inp, before)
    assert res_out.values().numel() == 0
    tu.assert_result_equal(res_out, ref_out)
    _assert_csr_metadata(res_out, ref_out, inp, torch.int64)


# Default suite only.
_ZERO_NNZ_SHAPES = tu.selected_cases(_GRID_SHAPES, quick=[])


@pytest.mark.to_sparse_csr
@pytest.mark.parametrize("shape", _ZERO_NNZ_SHAPES)
@pytest.mark.parametrize("dtype", _EMPTY_DTYPES)
def test_to_sparse_csr_zero_nnz(shape, dtype):
    # An all-zero source stores no elements at all (explicit zeros are
    # dropped), which leaves every crow row pointing at the same value.
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    before = inp.clone()

    ref_out = torch.ops.aten.to_sparse_csr(ref_inp)
    res_out = flag_gems.to_sparse_csr(inp)

    assert res_out is not inp
    tu.assert_result_equal(inp, before)
    assert res_out.values().numel() == 0
    assert res_out.col_indices().numel() == 0
    tu.assert_result_equal(res_out, ref_out)
    _assert_csr_metadata(res_out, ref_out, inp, torch.int64)


def _pattern_block(rows, cols, kind, dtype):
    block = torch.zeros((rows, cols), dtype=dtype, device=flag_gems.device)
    if rows == 0 or cols == 0:
        return block
    if kind == "full_row":
        block[0, :] = 1
    elif kind == "empty_interior":
        block[0, 0] = 1
        block[rows - 1, cols - 1] = 1
    elif kind == "single_column":
        block[:, 0] = 1
    else:  # double_entry
        block[:, 0] = 1
        block[:, cols - 1] = 2
    return block


# Default suite only.
_ROW_PATTERN_ROWS = tu.selected_cases(
    [
        ("full_row", (4, 5), 0),
        ("empty_interior", (4, 5), 0),
        ("single_column", (4, 5), 0),
        ("double_entry", (4, 5), 0),
        ("full_row", (2, 4, 5), 0),
        ("empty_interior", (3, 4, 5), 0),
        ("double_entry", (2, 4, 5), 1),
    ],
    quick=[],
)


@pytest.mark.to_sparse_csr
@pytest.mark.parametrize("kind,shape,dense_dim", _ROW_PATTERN_ROWS)
@pytest.mark.parametrize("dtype", _gate([torch.float32, torch.int16]))
def test_to_sparse_csr_row_patterns(kind, shape, dense_dim, dtype):
    # Fully dense rows, empty rows in the interior and rows with several
    # entries all produce different crow / col layouts.
    rows, cols = _sparse_dims(shape, dense_dim)
    block = _pattern_block(rows, cols, kind, dtype)
    inp = _tile(block, shape, dense_dim)
    ref_inp = tu.to_reference(inp)
    before = inp.clone()

    ref_out = torch.ops.aten.to_sparse_csr(ref_inp, *_dense_dim_args(dense_dim))
    res_out = flag_gems.to_sparse_csr(inp, *_dense_dim_args(dense_dim))

    assert res_out is not inp
    tu.assert_result_equal(inp, before)
    assert res_out.dense_dim() == dense_dim
    dense = _dense_roundtrip(res_out, dtype)
    if dense is not None:
        tu.assert_result_equal(dense, inp)
    tu.assert_result_equal(res_out, ref_out)
    _assert_csr_metadata(res_out, ref_out, inp, torch.int64)


_ROTATED_K = 3


def _rotated_block(rows, cols, dtype):
    """One sparse plane: ``min(cols, 3)`` stored entries per row.

    Entry ``(i, (i + step) % cols)`` carries the value ``1 + step`` for every
    ``step < min(cols, 3)``. For a fixed row those columns are distinct, so
    every row stores the same number of entries and a batched conversion sees
    the same count in each batch plane. Each wrap band of the modular pattern
    is the diagonal of a slice, so the fixture allocates no index tensor of any
    dtype and stays usable on a backend without int64 support.
    """
    block = torch.zeros((rows, cols), dtype=dtype, device=flag_gems.device)
    if rows == 0 or cols == 0:
        return block
    for step in range(min(cols, _ROTATED_K)):
        band = 0
        while True:
            if band == 0:
                row_start, col_start = 0, step
            else:
                row_start, col_start = band * cols - step, 0
            if row_start >= rows or col_start >= cols:
                break
            block[row_start:, col_start:].diagonal().fill_(1 + step)
            band += 1
    return block


def _rotated_batch_input(dtype, shape, dense_dim):
    """Equal stored count per batch, at different columns and values.

    Every batch keeps the same number of entries per row (so the CSR batch
    constraint holds) but the columns are rotated by the batch index and the
    values are scaled by a strictly positive factor, so a candidate that
    computes one batch and reuses it for the others, drops the batch axis or
    permutes the batches cannot match. The rotation uses slicing and torch.cat
    and the scale a float32 scalar ramped in the payload dtype, so no index
    tensor is allocated whatever the payload dtype is.
    """
    start = len(shape) - 2 - dense_dim
    batch_shape = tuple(shape[:start])
    batch_count = math.prod(batch_shape)
    rows, cols = _sparse_dims(shape, dense_dim)
    block = _rotated_block(rows, cols, dtype)
    if start == 0 or cols == 0 or batch_count < 2:
        return _tile(block, shape, dense_dim)
    planes = []
    for index in range(batch_count):
        shift = index % cols
        if shift == 0:
            planes.append(block)
        else:
            planes.append(
                torch.cat((block[:, cols - shift :], block[:, : cols - shift]), dim=1)
            )
    scale = torch.tensor(
        [float(index + 1) for index in range(batch_count)],
        dtype=torch.float32,
        device=flag_gems.device,
    ).to(dtype)
    rolled = torch.stack(planes) * scale.reshape(batch_count, 1, 1)
    return _expand_tail(rolled.reshape(batch_shape + (rows, cols)), shape, start)


# Default suite only.
_BATCHED_ROWS = tu.selected_cases(
    [
        ((2, 4, 6), 0),
        ((3, 2, 4, 6), 0),
        ((2, 3, 4, 6), 0),
        ((2, 3, 4, 5, 6), 0),
        ((2, 3, 4, 5, 6), 1),
        ((3, 2, 4, 5, 6), 1),
    ],
    quick=[],
)


@pytest.mark.to_sparse_csr
@pytest.mark.parametrize("shape,dense_dim", _BATCHED_ROWS)
@pytest.mark.parametrize("dtype", _gate([torch.float32, torch.float16, torch.int32]))
def test_to_sparse_csr_batched_layout(shape, dense_dim, dtype):
    inp = _rotated_batch_input(dtype, shape, dense_dim)
    ref_inp = tu.to_reference(inp)
    before = inp.clone()

    ref_out = torch.ops.aten.to_sparse_csr(ref_inp, dense_dim)
    res_out = flag_gems.to_sparse_csr(inp, dense_dim)

    assert res_out is not inp
    tu.assert_result_equal(inp, before)
    assert res_out.dense_dim() == dense_dim
    batch_count = math.prod(shape[: len(shape) - 2 - dense_dim])
    assert batch_count >= 2
    dense_values = res_out.to_dense().reshape(batch_count, -1)
    assert not torch.equal(dense_values[0], dense_values[1])

    tu.assert_result_equal(res_out.to_dense(), inp)
    tu.assert_result_equal(res_out, ref_out)
    _assert_csr_metadata(res_out, ref_out, inp, torch.int64)


@pytest.mark.to_sparse_csr
@pytest.mark.parametrize(
    "dtype",
    tu.selected_cases(_gate([torch.float32, torch.float16, torch.int32]), quick=[]),
)
def test_to_sparse_csr_drops_explicit_zeros(dtype):
    # Only non-zero entries are stored; an explicit zero in the middle of a row
    # must not appear in col_indices/values.
    inp = torch.tensor(
        [[0.0, 1.0, 0.0, 2.0], [3.0, 0.0, 0.0, 0.0]], device=flag_gems.device
    ).to(dtype)
    ref_inp = tu.to_reference(inp)
    before = inp.clone()

    ref_out = torch.ops.aten.to_sparse_csr(ref_inp)
    res_out = flag_gems.to_sparse_csr(inp)

    assert res_out is not inp
    tu.assert_result_equal(inp, before)
    stored = int(torch.count_nonzero(inp).item())
    assert res_out.values().numel() == stored
    assert res_out.crow_indices()[-1].item() == stored
    tu.assert_result_equal(res_out.to_dense(), inp)
    tu.assert_result_equal(res_out, ref_out)
    _assert_csr_metadata(res_out, ref_out, inp, torch.int64)


_SPECIAL_DTYPES = _gate(
    [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ]
)


@pytest.mark.to_sparse_csr
@pytest.mark.parametrize(
    "dtype,scenario",
    tu.selected_cases(tu.special_value_cases(_SPECIAL_DTYPES), quick=[]),
)
def test_to_sparse_csr_special_values(dtype, scenario):
    payload = tu.make_special_input(dtype, scenario).reshape(1, -1)
    ref_inp = tu.to_reference(payload)
    before = payload.clone()

    ref_out = torch.ops.aten.to_sparse_csr(ref_inp)
    res_out = flag_gems.to_sparse_csr(payload)

    assert res_out is not payload
    tu.assert_result_equal(payload, before)
    tu.assert_result_equal(res_out, ref_out)
    _assert_csr_metadata(res_out, ref_out, payload, torch.int64)


# Default suite only.
_BACKWARD_ROWS = tu.selected_cases(
    [
        ((6, 5), 0),
        ((4, 6, 5), 0),
        ((4, 6, 5), 1),
        ((3, 4, 5, 6), 0),
        ((3, 4, 5, 6), 1),
        ((3, 4, 5, 6), 2),
        ((2, 3, 4, 5), 0),
        ((2, 3, 4, 5), 1),
        ((2, 3, 4, 5), 2),
        ((2, 3, 4, 5, 6), 3),
    ],
    quick=[],
)


def _varying_upstream(dtype, shape, dense_dim):
    """Nonuniform upstream gradient.

    _make_input replicates one matrix over the batch axes and the dense tail,
    so an upstream built from it alone would repeat every batch and tail
    entry and could not detect a permuted batch or tail gradient. A strictly
    positive float32 ramp is applied per batch element and per tail position
    (in a supported width, so no int64 geometry is allocated).
    """
    start = len(shape) - 2 - dense_dim
    batch_shape = tuple(shape[:start])
    tail_shape = tuple(shape[start + 2 :])
    upstream = _make_input(dtype, shape, ["0", "1"], dense_dim)
    batch_ramp = torch.linspace(
        1.0, 2.0, max(1, math.prod(batch_shape)), device=flag_gems.device
    ).reshape(batch_shape + (1,) * (2 + len(tail_shape)))
    upstream = upstream * batch_ramp.to(dtype)
    if tail_shape:
        tail_ramp = torch.linspace(
            1.0, 2.0, math.prod(tail_shape), device=flag_gems.device
        ).reshape((1,) * len(batch_shape) + (1, 1) + tail_shape)
        upstream = upstream * tail_ramp.to(dtype)
    return upstream


@pytest.mark.to_sparse_csr
@pytest.mark.parametrize("shape,dense_dim", _BACKWARD_ROWS)
@pytest.mark.parametrize("dtype", _gate([torch.float32, torch.float16, torch.bfloat16]))
def test_to_sparse_csr_backward(shape, dense_dim, dtype):
    # The result is a relocation of the stored input values, so autograd hands
    # the dense upstream gradient straight back (grad equals upstream exactly),
    # and the forward result must already match the reference before it is
    # differentiated. Every plan passes its requested dense_dim, and each path
    # gets its own upstream tensor on its own device so a reference run on
    # another device stays valid.
    inp = _make_input(dtype, shape, ["-1", "1"], dense_dim).requires_grad_()
    ref_inp = tu.to_reference(inp).detach().requires_grad_()
    before = inp.detach().clone()
    upstream = _varying_upstream(dtype, shape, dense_dim)
    ref_upstream = tu.to_reference(upstream)

    ref_out = torch.ops.aten.to_sparse_csr(ref_inp, dense_dim)
    ref_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_upstream)[0]

    res_out = flag_gems.to_sparse_csr(inp, dense_dim)
    assert res_out.requires_grad
    tu.assert_result_equal(res_out, ref_out)
    res_grad = torch.autograd.grad(res_out, inp, grad_outputs=upstream)[0]

    tu.assert_result_equal(res_grad, ref_grad)
    tu.assert_result_equal(res_grad, ref_upstream)
    tu.assert_result_equal(inp.detach(), before)


@pytest.mark.to_sparse_csr
@pytest.mark.parametrize("shape", [(3,), (5,), (), (1,)])
def test_to_sparse_csr_rejects_low_rank(shape):
    # Native needs a matrix: (3,), (5,) and (1,) fail inside expand with a
    # RuntimeError, and the rank-0 input reaches an indexing expression that
    # reports "Dimension specified as -1 but tensor has no dimensions" as an
    # IndexError. Both classes are measured on the active backend for these
    # rows, hence the tuple.
    inp = torch.zeros(shape, device=flag_gems.device)
    with pytest.raises((RuntimeError, IndexError)):
        flag_gems.to_sparse_csr(inp)


@pytest.mark.to_sparse_csr
@pytest.mark.parametrize(
    "shape,dense_dim",
    [
        ((4, 5), 1),
        ((4, 5), 2),
        ((2, 4, 5), 2),
        ((2, 4, 5), 3),
        ((2, 3, 4, 5), 3),
        ((2, 3, 4, 5), 7),
    ],
)
def test_to_sparse_csr_rejects_dense_dim_out_of_range(shape, dense_dim):
    # Measured natively on the CPU and the CUDA backend, every row above is
    # rejected up front with the same class: RuntimeError
    # "dense_to_sparse_csr: dense_dim argument must be in [0, rank - 2] range,
    # but <dense_dim> is given" ([0,0] for the 2-D rows, [0,1] for (2,4,5),
    # [0,2] for (2,3,4,5)).
    inp = torch.zeros(shape, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.to_sparse_csr(inp, dense_dim)


@pytest.mark.to_sparse_csr
@pytest.mark.parametrize(
    "shape,dense_dim",
    [
        ((0, 3, 4), 0),
        ((2, 0, 3, 4), 0),
        ((0, 0, 3, 4), 1),
    ],
)
def test_to_sparse_csr_rejects_zero_batch(shape, dense_dim):
    # Native: a batched conversion needs a non-zero batch product for the
    # requested dense_dim ("Expected product of batch dimensions to be
    # non-zero."), while a zero matrix or a zero dense tail next to a non-zero
    # batch product stays legal (see test_to_sparse_csr_zero_extent).
    inp = torch.zeros(shape, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.to_sparse_csr(inp, dense_dim)


@pytest.mark.to_sparse_csr
@pytest.mark.parametrize("dense_dim", [0, 1])
def test_to_sparse_csr_rejects_dense_dim_on_csr_source(dense_dim):
    # Native refuses to reinterpret an existing CSR tensor with an explicit
    # dense_dim ("conversion from SparseCsr to SparseCsr with dense_dim
    # argument given is not supported"); the default call returns it as is.
    source = _sparse_source("csr", torch.int32)
    with pytest.raises(RuntimeError):
        flag_gems.to_sparse_csr(source, dense_dim)


@pytest.mark.to_sparse_csr
@pytest.mark.parametrize("bad", [1.5, "0"])
def test_to_sparse_csr_rejects_non_int_dense_dim(bad):
    inp = torch.zeros((4, 5), device=flag_gems.device)
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.to_sparse_csr(inp, bad)


@pytest.mark.to_sparse_csr
def test_to_sparse_csr_rejects_non_tensor():
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.to_sparse_csr(3.14)


@pytest.mark.to_sparse_csr
def test_to_sparse_csr_rejects_block_source():
    # Only CSR/CSC (and COO) sources convert; a BSR source raises
    # "sparse_compressed_to_sparse_csr: expected SparseCsr or SparseCsc layout".
    crow = torch.tensor([0, 1, 1, 2, 2], dtype=torch.int32, device=flag_gems.device)
    col = torch.tensor([1, 0], dtype=torch.int32, device=flag_gems.device)
    values = torch.ones(2, 1, 1, device=flag_gems.device)
    source = torch.sparse_bsr_tensor(crow, col, values, size=(4, 4))
    with pytest.raises(RuntimeError):
        flag_gems.to_sparse_csr(source)
