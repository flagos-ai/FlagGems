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

from . import test_utils as tu

# Static capability flags of the active backend, read without creating a tensor
# or running an operator.
_RUNTIME = flag_gems.runtime.device

_BASE_DTYPES = [torch.int8, torch.uint8, torch.float32, torch.float16, torch.int32]
_BF16_DTYPES = [torch.bfloat16] if _RUNTIME.support_bf16 else []
_INT64_DTYPES = [torch.int64] if _RUNTIME.support_int64 else []
_FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2] if _RUNTIME.support_fp8 else []

# Dense->COO builds its coordinates with the native nonzero kernel. On the
# measured NVIDIA backend that kernel has no FP8 implementation and the failure
# is raised inside the native conversion, not before the operator runs:
#   torch.ops.aten._to_sparse(zeros(4, dtype=float8_e4m3fn)) -> RuntimeError
#       'nonzero_cuda' not implemented for 'Float8_e4m3fn'
#   torch.ops.aten._to_sparse(zeros(4, dtype=float8_e5m2)) -> RuntimeError
#       'nonzero_cuda' not implemented for 'Float8_e5m2'
# The compressed layouts do not use that helper and do convert both formats, so
# the FP8 positives live there. Eligibility is gated on the measured vendor
# only; other backends stay unmeasured instead of being assumed either way.
_COO_FP8_OK = _RUNTIME.support_fp8 and _RUNTIME.vendor_name != "nvidia"


def _int64_gated(cases):
    """Case data gated on the backend's int64 allocation support.

    Every case below ends in a result whose index structure is int64: a COO
    result stores its coordinates in int64, a compressed result its crow/col or
    ccol/row pointers, and the out/layout/backward fixtures additionally
    allocate int64 index buffers and an int64 scatter index. A backend that
    cannot allocate int64 tensors therefore cannot run these families at all, so
    the whole family is emptied statically -- never only the int64 payload row
    while the fixtures still allocate int64 structure.
    """
    return list(cases) if _RUNTIME.support_int64 else []


_COO_DTYPES = _int64_gated(
    _BASE_DTYPES + _BF16_DTYPES + _INT64_DTYPES + (_FP8_DTYPES if _COO_FP8_OK else [])
)
_COMPRESSED_DTYPES = _int64_gated(
    _BASE_DTYPES + _BF16_DTYPES + _INT64_DTYPES + _FP8_DTYPES
)
_SPECIAL_DTYPES = _int64_gated(
    [torch.float32, torch.float16]
    + _BF16_DTYPES
    + ([torch.float64] if _RUNTIME.support_fp64 else [])
)

_COMPRESSED_SHAPE = (1024, 1024)

# Every positive family asserts the same two operator-specific properties after
# the candidate call: the result lives on the input's device, and the tested
# input still holds the sampled values (the conversion is not in place). The
# independent reference input taken before the call is also the immutability
# oracle, so no extra snapshot is allocated. Native output structure -- int64
# COO coordinates, int64 crow/col beside a lower-precision value dtype -- is
# covered by the shared comparison of the whole result.


def _batch_size(batch):
    return math.prod(batch) if batch else 1


def _window_mask(grid, blocksize, device):
    """Equal stored-cell count per batch slice, in the dense logical shape.

    A batched compressed conversion rejects an unequal number of specified
    elements per batch ('Expect the same number of specified elements per
    batch.'), so the same number of cells is selected in every slice while the
    window start shifts per slice, keeping the stored positions non-uniform.
    Block cells are expanded back to rows/columns after the zero-cell short
    circuit, so the returned mask always has the dense operand's shape.
    """
    rows, cols = grid[-2], grid[-1]
    batch = tuple(grid[:-2])
    if blocksize is None:
        br, bc = rows, cols
    else:
        br, bc = rows // blocksize[0], cols // blocksize[1]
    cells = br * bc
    slices = _batch_size(batch)
    if cells == 0:
        selected = torch.zeros((slices, br, bc), dtype=torch.bool, device=device)
    else:
        keep = max(cells // 4, 1)
        starts = (torch.arange(slices, dtype=torch.int32, device=device) * keep) % cells
        offsets = torch.arange(keep, dtype=torch.int32, device=device)
        index = ((starts[:, None] + offsets[None, :]) % cells).long()
        selected = torch.zeros((slices, cells), dtype=torch.bool, device=device)
        selected.scatter_(1, index, True)
        selected = selected.reshape((slices, br, bc))
    selected = selected.reshape(batch + (br, bc))
    if blocksize is None:
        return selected
    return selected.repeat_interleave(blocksize[0], -2).repeat_interleave(
        blocksize[1], -1
    )


def _anchor(value_range, dtype, device):
    """In-range nonzero anchor for the selected cells, or None if none exists.

    A batched compressed input must store the same number of elements in every
    slice, so a selected cell whose sampled value is 0 is replaced by a nonzero
    value that still lies inside ``value_range``. A range whose upper bound is
    not positive cannot use +1, and an unsigned dtype cannot hold a negative
    anchor; such a range clamps to the degenerate [0, 0] fill, where every slice
    stores zero elements and the equal-count rule already holds.
    """
    if tu.resolve_bound(value_range[1], dtype) > 0:
        value = 1.0
    elif value_range[0] == "0":
        value = -1.0
    else:
        value = max(tu.resolve_bound(value_range[0], dtype), -1.0)
    if not (dtype.is_floating_point or dtype.is_complex):
        low, high = (int(bound) for bound in tu.dtype_bounds(dtype))
        value = min(max(int(value), low), high)
        if value == 0:
            return None
    return torch.full((), value, dtype=dtype, device=device)


def _empty_units(masked, mask, blocksize):
    """Selected cells/blocks that hold no nonzero value at all.

    The equal-count rule needs every selected unit to store something, but a
    zero *inside* a unit that already stores a nonzero value is real data and
    keeps its sampled value. Only a unit with no nonzero entry at all loses its
    sampled zeros and is rewritten with the anchor.
    """
    live = masked != 0
    while live.dim() > mask.dim():
        live = live.any(dim=-1)
    if blocksize is not None:
        block_rows, block_cols = blocksize
        rows, cols = live.shape[-2] // block_rows, live.shape[-1] // block_cols
        live = live.reshape(live.shape[:-2] + (rows, block_rows, cols, block_cols))
        live = live.any(dim=-1).any(dim=-2)
        live = live.repeat_interleave(block_rows, -2).repeat_interleave(block_cols, -1)
    return mask & ~live


def _masked_dense(dtype, shape, value_range, blocksize=None, dense_dim=0):
    """Dense operand for a compressed conversion.

    A 2-D grid has no batch slice, so the sampled values (zeros included) are
    used unchanged; only a batched grid needs the equal-count window, and there
    only the selected units that are otherwise all zero receive the anchor.
    """
    dense = tu.make_input(dtype, shape, value_range)
    grid = tuple(shape[: len(shape) - dense_dim])
    if len(grid) <= 2:
        return dense
    mask = _window_mask(grid, blocksize, dense.device)
    in_unit = mask.reshape(grid + (1,) * dense_dim)
    masked = torch.where(in_unit, dense, torch.zeros_like(dense))
    anchor = _anchor(value_range, dtype, dense.device)
    if anchor is None:
        return masked
    fill = _empty_units(masked, mask, blocksize).reshape(grid + (1,) * dense_dim)
    return torch.where(fill, anchor, masked)


def _dense_input(dtype, shape, value_range, layout, blocksize, dense_dim=0):
    """Dense operand for one conversion case; COO stores every nonzero."""
    if layout is torch.sparse_coo:
        return tu.make_input(dtype, shape, value_range)
    return _masked_dense(dtype, shape, value_range, blocksize, dense_dim)


def _pattern_dense(pattern, dtype):
    """(8, 8) input with a controlled number of stored entries.

    'empty_rows' fills a whole row, so 7 rows stay empty; 'empty_cols' fills a
    whole column, so 7 columns stay empty.
    """
    base = tu.make_input(dtype, (8, 8), ["-1", "1"])
    if pattern == "empty":
        return torch.zeros_like(base)
    if pattern == "full":
        return torch.where(base == 0, torch.ones_like(base), base)
    mask = torch.zeros((8, 8), dtype=torch.bool, device=base.device)
    if pattern == "single":
        mask[3, 5] = True
    elif pattern == "empty_rows":
        mask[2, :] = True
    else:  # empty_cols
        mask[:, 2] = True
    masked = torch.where(mask, base, torch.zeros_like(base))
    return torch.where(mask & (masked == 0), torch.ones_like(masked), masked)


def _view(name, source):
    """View used as the tested input; every form changes strides and/or offset."""
    if name == "transpose":
        return source.t()
    if name == "row_stride":
        return source[1::2, :]
    if name == "column_stride":
        return source[:, 1::2]
    return source[2:18, 2:18]


def _sentinel_coo(shape, sparse_dim, nnz, dtype, device):
    """COO buffer the out overload must rewrite completely.

    Coordinates are valid in-range values and every stored value is a marker the
    sampled inputs cannot produce, so a buffer that is left unwritten, or only
    partly written, is reported by the value comparison, and the fixture is
    never a malformed sparse tensor. Coordinates are int64, the dtype COO
    indices always use, and the scalar form uses its legitimate (0, nnz) index
    shape instead of coordinates that do not exist.
    """
    if sparse_dim == 0:
        indices = torch.zeros((0, nnz), dtype=torch.int64, device=device)
    else:
        if any(extent == 0 for extent in shape[:sparse_dim]):
            nnz = 0
        limit = max(min(shape[:sparse_dim]), 1)
        columns = torch.arange(nnz, dtype=torch.int64, device=device) % limit
        indices = columns.reshape(1, nnz).expand(sparse_dim, nnz).contiguous()
    if dtype.is_floating_point:
        marker = 999.0
    else:
        marker = float(tu.dtype_bounds(dtype)[1])
    values = torch.full(
        (nnz,) + tuple(shape[sparse_dim:]), marker, dtype=dtype, device=device
    )
    return torch.sparse_coo_tensor(indices, values, size=shape)


def _kwargs(layout, blocksize=None, dense_dim=0):
    kwargs = {"layout": layout}
    if blocksize is not None:
        kwargs["blocksize"] = blocksize
    if dense_dim:
        kwargs["dense_dim"] = dense_dim
    return kwargs


def _grad_upstream(result, dtype):
    """Finite non-uniform upstream gradient aligned with the stored entries.

    The counter spans the dense tail as well as the coordinates, so every stored
    scalar gets its own value and a backward implementation that permutes or
    repeats dense-tail entries cannot match the reference. The values are
    allocated directly on the result's device -- no host round trip -- in the
    COO form the native gradient accepts for a compressed result.
    """
    coo = result if result.layout == torch.sparse_coo else result.to_sparse_coo()
    indices = coo.coalesce().indices().detach()
    nnz = indices.shape[1]
    tail = tuple(coo.shape[coo.sparse_dim() :])
    values = torch.arange(
        1, nnz * math.prod(tail) + 1, dtype=torch.float32, device=indices.device
    )
    values = values.reshape((nnz,) + tail).to(dtype)
    return torch.sparse_coo_tensor(indices, values, coo.shape, is_coalesced=True)


@pytest.mark.to_sparse
@pytest.mark.parametrize("dtype", _COO_DTYPES)
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test__to_sparse_default(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    # Default call: layout, blocksize and dense_dim are all omitted. The scalar
    # shape () is the native-accepted sparse_dim 0 form.
    ref_out = torch.ops.aten._to_sparse(ref_inp)
    res_out = flag_gems._to_sparse(inp)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    tu.assert_result_equal(inp, ref_inp)


# The 1M compressed grid is a supplemental positive, so quick runs only the
# compact default-parameter smoke and the negative cases.
_COMPRESSED_LAYOUTS = tu.selected_cases(
    [
        (torch.sparse_csr, None),
        (torch.sparse_csc, None),
        (torch.sparse_bsr, [2, 2]),
        (torch.sparse_bsc, [2, 2]),
    ],
    quick=[],
)


@pytest.mark.to_sparse
@pytest.mark.parametrize("dtype", _COMPRESSED_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("layout,blocksize", _COMPRESSED_LAYOUTS)
def test__to_sparse_compressed_layout(layout, blocksize, value_range, dtype):
    inp = _masked_dense(dtype, _COMPRESSED_SHAPE, value_range, blocksize)
    ref_inp = tu.to_reference(inp)

    kwargs = _kwargs(layout, blocksize)
    ref_out = torch.ops.aten._to_sparse.default(ref_inp, **kwargs)
    res_out = flag_gems._to_sparse(inp, **kwargs)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    tu.assert_result_equal(inp, ref_inp)


_EXPLICIT_COO_DTYPES = tu.selected_cases(_COO_DTYPES, quick=[])


@pytest.mark.to_sparse
@pytest.mark.parametrize("dtype", _EXPLICIT_COO_DTYPES)
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test__to_sparse_explicit_coo_layout(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_sparse.default(ref_inp, layout=torch.sparse_coo)
    res_out = flag_gems._to_sparse(inp, layout=torch.sparse_coo)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    tu.assert_result_equal(inp, ref_inp)


# dense_dim must be in [0, rank - 1]; a nonzero value needs a sparse grid of
# rank >= 2, so the extras use rank >= 3 inputs. Zero is passed explicitly here
# because the omitted-argument default call is the main grid above.
_DENSE_DIM_CASES = tu.selected_cases(
    [((1024, 1024), 0), ((8, 12, 6), 1), ((8, 6, 5, 4), 2)], quick=[]
)
_DENSE_DIM_DTYPES = tu.selected_cases(_COO_DTYPES, quick=[])


@pytest.mark.to_sparse
@pytest.mark.parametrize("dtype", _DENSE_DIM_DTYPES)
@pytest.mark.parametrize("shape,dense_dim", _DENSE_DIM_CASES)
def test__to_sparse_dense_dim(shape, dense_dim, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_sparse.default(ref_inp, dense_dim=dense_dim)
    res_out = flag_gems._to_sparse(inp, dense_dim=dense_dim)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    tu.assert_result_equal(inp, ref_inp)


# blocksize must be positive and divide the compressed matrix; the boundary 1
# and larger exact divisors are covered, invalid values are negative cases.
_BLOCKSIZE_CASES = tu.selected_cases(
    [
        (torch.sparse_bsr, [1, 1]),
        (torch.sparse_bsr, [2, 2]),
        (torch.sparse_bsr, [4, 4]),
        (torch.sparse_bsr, [8, 8]),
        (torch.sparse_bsc, [2, 2]),
        (torch.sparse_bsc, [4, 4]),
    ],
    quick=[],
)
_BLOCKSIZE_DTYPES = tu.selected_cases(
    _int64_gated([torch.float32, torch.int32]), quick=[]
)


@pytest.mark.to_sparse
@pytest.mark.parametrize("dtype", _BLOCKSIZE_DTYPES)
@pytest.mark.parametrize("layout,blocksize", _BLOCKSIZE_CASES)
def test__to_sparse_blocksize(layout, blocksize, dtype):
    inp = _masked_dense(dtype, (16, 16), ["-1", "1"], blocksize)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_sparse.default(
        ref_inp, layout=layout, blocksize=blocksize
    )
    res_out = flag_gems._to_sparse(inp, layout=layout, blocksize=blocksize)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    tu.assert_result_equal(inp, ref_inp)


# Rank >= 3 compressed conversions need the equal-count window, so every row is
# built with _masked_dense. The rows carry the batched, hybrid and spec-scale
# variants, reusing the large spec shapes for the batched maps.
_BATCHED_ROWS = tu.selected_cases(
    [
        ((2, 4, 6), torch.sparse_csr, None, 0, ["-1", "1"]),
        ((2, 4, 6), torch.sparse_csc, None, 0, ["-1", "1"]),
        ((2, 4, 6), torch.sparse_bsr, [2, 2], 0, ["-1", "1"]),
        ((2, 4, 6), torch.sparse_bsc, [2, 2], 0, ["-1", "1"]),
        ((2, 3, 5, 7), torch.sparse_csr, None, 0, ["0", "max"]),
        ((2, 3, 5, 7), torch.sparse_bsr, [1, 1], 0, ["min", "0"]),
        ((2, 4, 6), torch.sparse_csr, None, 1, ["-1", "1"]),
        ((2, 4, 6), torch.sparse_bsr, [2, 2], 1, ["-1", "1"]),
        ((2, 3, 4, 5), torch.sparse_csr, None, 2, ["-1", "1"]),
        ((20, 320, 15), torch.sparse_csr, None, 0, ["-1", "1"]),
        ((20, 320, 15), torch.sparse_bsr, [5, 5], 0, ["0", "1"]),
        ((16, 128, 64, 60), torch.sparse_csr, None, 0, ["-1", "1"]),
        ((16, 128, 64, 60), torch.sparse_bsr, [2, 2], 0, ["-1", "0"]),
        ((16, 128, 64, 60), torch.sparse_csr, None, 1, ["-1", "1"]),
        ((16, 7, 57, 32, 29), torch.sparse_csr, None, 0, ["-1", "1"]),
    ],
    quick=[],
)
_BATCHED_DTYPES = _int64_gated([torch.float32, torch.int8])


@pytest.mark.to_sparse
@pytest.mark.parametrize("dtype", _BATCHED_DTYPES)
@pytest.mark.parametrize("shape,layout,blocksize,dense_dim,value_range", _BATCHED_ROWS)
def test__to_sparse_batched_compressed(
    shape, layout, blocksize, dense_dim, value_range, dtype
):
    inp = _masked_dense(dtype, shape, value_range, blocksize, dense_dim)
    ref_inp = tu.to_reference(inp)

    kwargs = _kwargs(layout, blocksize, dense_dim)
    ref_out = torch.ops.aten._to_sparse.default(ref_inp, **kwargs)
    res_out = flag_gems._to_sparse(inp, **kwargs)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    tu.assert_result_equal(inp, ref_inp)


# Rectangular blocks exercise the block-pair axis independently per layout.
_RECT_ROWS = tu.selected_cases(
    [
        ((12, 18), torch.sparse_bsr, [2, 3]),
        ((18, 12), torch.sparse_bsr, [3, 2]),
        ((12, 18), torch.sparse_bsr, [4, 6]),
        ((12, 18), torch.sparse_bsr, [1, 9]),
        ((12, 18), torch.sparse_bsc, [2, 3]),
        ((18, 12), torch.sparse_bsc, [3, 2]),
    ],
    quick=[],
)
_RECT_DTYPES = _int64_gated([torch.float32, torch.int8, torch.uint8])


@pytest.mark.to_sparse
@pytest.mark.parametrize("dtype", _RECT_DTYPES)
@pytest.mark.parametrize("shape,layout,blocksize", _RECT_ROWS)
def test__to_sparse_rectangular_blocks(shape, layout, blocksize, dtype):
    inp = _masked_dense(dtype, shape, ["-1", "1"], blocksize)
    ref_inp = tu.to_reference(inp)

    kwargs = _kwargs(layout, blocksize)
    ref_out = torch.ops.aten._to_sparse.default(ref_inp, **kwargs)
    res_out = flag_gems._to_sparse(inp, **kwargs)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    tu.assert_result_equal(inp, ref_inp)


_DENSITY_PATTERNS = ["empty", "full", "single", "empty_rows", "empty_cols"]
_DENSITY_LAYOUTS = [torch.sparse_coo, torch.sparse_csr, torch.sparse_bsr]
_DENSITY_ROWS = tu.selected_cases(
    [
        (pattern, layout, [2, 2] if layout is torch.sparse_bsr else None)
        for pattern in _DENSITY_PATTERNS
        for layout in _DENSITY_LAYOUTS
    ],
    quick=[],
)
_DENSITY_DTYPES = _int64_gated([torch.float32, torch.int8, torch.uint8])


@pytest.mark.to_sparse
@pytest.mark.parametrize("dtype", _DENSITY_DTYPES)
@pytest.mark.parametrize("pattern,layout,blocksize", _DENSITY_ROWS)
def test__to_sparse_density_patterns(pattern, layout, blocksize, dtype):
    inp = _pattern_dense(pattern, dtype)
    ref_inp = tu.to_reference(inp)

    kwargs = _kwargs(layout, blocksize)
    ref_out = torch.ops.aten._to_sparse.default(ref_inp, **kwargs)
    res_out = flag_gems._to_sparse(inp, **kwargs)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    tu.assert_result_equal(inp, ref_inp)


# Zero extents: (0,) is the rank-1 COO form; a compressed layout keeps a rank-2
# shape whose index and value arrays become empty while the compressed pointer
# arrays keep their required endpoints.
_ZERO_EXTENT_ROWS = tu.selected_cases(
    [
        ((0,), torch.sparse_coo, None),
        ((0, 8), torch.sparse_coo, None),
        ((8, 0), torch.sparse_coo, None),
        ((0, 0), torch.sparse_coo, None),
        ((0, 8), torch.sparse_csr, None),
        ((8, 0), torch.sparse_csr, None),
        ((0, 0), torch.sparse_csr, None),
        ((0, 8), torch.sparse_csc, None),
        ((0, 8), torch.sparse_bsr, [2, 2]),
        ((8, 0), torch.sparse_bsr, [2, 2]),
        ((0, 0), torch.sparse_bsr, [2, 2]),
        ((0, 0), torch.sparse_bsc, [2, 2]),
    ],
    quick=[],
)
_ZERO_EXTENT_DTYPES = _int64_gated([torch.float32, torch.int8])


@pytest.mark.to_sparse
@pytest.mark.parametrize("dtype", _ZERO_EXTENT_DTYPES)
@pytest.mark.parametrize("shape,layout,blocksize", _ZERO_EXTENT_ROWS)
def test__to_sparse_zero_extent(shape, layout, blocksize, dtype):
    inp = _dense_input(dtype, shape, ["-1", "1"], layout, blocksize)
    ref_inp = tu.to_reference(inp)

    kwargs = _kwargs(layout, blocksize)
    ref_out = torch.ops.aten._to_sparse.default(ref_inp, **kwargs)
    res_out = flag_gems._to_sparse(inp, **kwargs)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    tu.assert_result_equal(inp, ref_inp)


# The views below change strides and/or the storage offset (transpose, row
# stride, column stride, offset). A candidate that ignores the view metadata
# converts different values and the comparison fails. Two distinct properties
# are asserted after the candidate call: the view's own logical values
# (inp/ref_inp) and its whole parent allocation (source/ref_source, strided
# gaps and padding included), so replacing the view's values at an unchanged
# stride and offset is caught as well. The reference parent is allocated
# independently before the same view is taken on it.
_VIEW_ROWS = tu.selected_cases(
    [
        (view, shape, layout, [2, 2] if layout is torch.sparse_bsr else None)
        for view, shape in (
            ("transpose", (16, 16)),
            ("row_stride", (16, 16)),
            ("column_stride", (16, 24)),
            ("offset", (20, 20)),
        )
        for layout in (torch.sparse_coo, torch.sparse_csr, torch.sparse_bsr)
    ],
    quick=[],
)
_VIEW_DTYPES = _int64_gated([torch.float32, torch.int32])


@pytest.mark.to_sparse
@pytest.mark.parametrize("dtype", _VIEW_DTYPES)
@pytest.mark.parametrize("view,shape,layout,blocksize", _VIEW_ROWS)
def test__to_sparse_input_views(view, shape, layout, blocksize, dtype):
    source = tu.make_input(dtype, shape, ["-1", "1"])
    ref_source = tu.to_reference(source)

    inp = _view(view, source)
    ref_inp = _view(view, ref_source)

    kwargs = _kwargs(layout, blocksize)
    ref_out = torch.ops.aten._to_sparse.default(ref_inp, **kwargs)
    res_out = flag_gems._to_sparse(inp, **kwargs)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    tu.assert_result_equal(inp, ref_inp)
    tu.assert_result_equal(source, ref_source)
    assert inp.stride() == ref_inp.stride()
    assert inp.storage_offset() == ref_inp.storage_offset()


# The out buffers are allocated on the device that executes the call, one pad
# above the reference nnz, so the overload has to resize the buffer in place;
# the valid in-range marker values report a buffer that was left unwritten. The
# scalar rows use the legitimate (0, nnz) index shape of a sparse_dim 0 result.
_OUT_ROWS = tu.selected_cases(
    _int64_gated(
        [
            (torch.float32, (), 0, 0),
            (torch.float32, (), 5, 0),
            (torch.float32, (4, 5), 0, 0),
            (torch.float32, (4, 5), 5, 0),
            (torch.float32, (6, 4, 3), 5, 0),
            (torch.int32, (4, 5), 5, 0),
            (torch.float32, (0, 5), 5, 0),
            (torch.float32, (8, 6), 5, 1),
        ]
        + ([(torch.float64, (8, 3), 5, 0)] if _RUNTIME.support_fp64 else [])
    ),
    quick=[],
)


@pytest.mark.to_sparse
@pytest.mark.parametrize("dtype,shape,pad,dense_dim", _OUT_ROWS)
def test__to_sparse_out_overload(dtype, shape, pad, dense_dim):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    kwargs = _kwargs(torch.sparse_coo, None, dense_dim)
    ref_nnz = torch.ops.aten._to_sparse.default(ref_inp, **kwargs)._nnz()
    sparse_dim = len(shape) - dense_dim

    ref_out = _sentinel_coo(shape, sparse_dim, ref_nnz + pad, dtype, ref_inp.device)
    ref_res = torch.ops.aten._to_sparse.out(ref_inp, out=ref_out, **kwargs)

    out = _sentinel_coo(shape, sparse_dim, ref_nnz + pad, dtype, inp.device)
    res = flag_gems._to_sparse(inp, out=out, **kwargs)

    assert res is out
    tu.assert_result_equal(res, ref_res)
    assert res.device == inp.device
    tu.assert_result_equal(inp, ref_inp)


# The sparse_dim overload takes the sparse dimension positionally, and the
# packet dispatches that same call form for both paths. The scalar input is the
# native-accepted sparse_dim 0 form (() with sparse_dim 0 stores one value and
# reports sparse_dim 0), which is why the rank-2 negative rows are not
# generalized to it.
_SPARSE_DIM_ROWS = tu.selected_cases(
    [
        ((), 0),
        ((4, 5), 1),
        ((4, 5), 2),
        ((4, 5, 6), 1),
        ((4, 5, 6), 2),
        ((4, 5, 6), 3),
        ((8,), 1),
    ],
    quick=[],
)


@pytest.mark.to_sparse
@pytest.mark.parametrize("dtype", _COO_DTYPES)
@pytest.mark.parametrize("shape,sparse_dim", _SPARSE_DIM_ROWS)
def test__to_sparse_sparse_dim_overload(shape, sparse_dim, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_sparse.sparse_dim(ref_inp, sparse_dim)
    res_out = flag_gems._to_sparse(inp, sparse_dim)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    tu.assert_result_equal(inp, ref_inp)


_SPARSE_DIM_OUT_ROWS = tu.selected_cases(
    _int64_gated(
        [
            ((), 0, 0),
            ((), 0, 5),
            ((4, 5), 1, 0),
            ((4, 5), 1, 5),
            ((4, 5), 2, 5),
            ((4, 5, 6), 1, 5),
            ((4, 5, 6), 2, 5),
        ]
    ),
    quick=[],
)


@pytest.mark.to_sparse
@pytest.mark.parametrize("shape,sparse_dim,pad", _SPARSE_DIM_OUT_ROWS)
def test__to_sparse_sparse_dim_out_overload(shape, sparse_dim, pad):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_nnz = torch.ops.aten._to_sparse.sparse_dim(ref_inp, sparse_dim)._nnz()
    ref_out = _sentinel_coo(
        shape, sparse_dim, ref_nnz + pad, torch.float32, ref_inp.device
    )
    ref_res = torch.ops.aten._to_sparse.sparse_dim_out(ref_inp, sparse_dim, out=ref_out)

    out = _sentinel_coo(shape, sparse_dim, ref_nnz + pad, torch.float32, inp.device)
    res = flag_gems._to_sparse(inp, sparse_dim, out=out)

    assert res is out
    tu.assert_result_equal(res, ref_res)
    assert res.device == inp.device
    tu.assert_result_equal(inp, ref_inp)


# Independent candidate and reference inputs with equal values, and an
# independent upstream gradient per side: neither the graph nor the gradient
# object is shared between the calls. The backward is a pure scatter with no
# accumulation, so the gradients are compared exactly.
_BACKWARD_ROWS = tu.selected_cases(
    _int64_gated(
        [
            (torch.float32, (8, 12), torch.sparse_coo, None, 0),
            (torch.float32, (8, 12), torch.sparse_coo, None, 1),
            (torch.float16, (8, 12), torch.sparse_coo, None, 1),
            (torch.float32, (8, 12), torch.sparse_csr, None, 0),
            (torch.float16, (8, 12), torch.sparse_csr, None, 0),
            (torch.float32, (8, 12), torch.sparse_bsr, [2, 2], 0),
        ]
        + (
            [(torch.float64, (8, 12), torch.sparse_coo, None, 0)]
            if _RUNTIME.support_fp64
            else []
        )
    ),
    quick=[],
)


@pytest.mark.to_sparse
@pytest.mark.parametrize("dtype,shape,layout,blocksize,dense_dim", _BACKWARD_ROWS)
def test__to_sparse_backward(dtype, shape, layout, blocksize, dense_dim):
    res_inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    # tu.to_reference already copies the storage into a detached tensor and
    # restores the input's requires_grad, so the two graphs are independent
    # without a second clone chain.
    ref_inp = tu.to_reference(res_inp)

    kwargs = _kwargs(layout, blocksize, dense_dim)
    ref_out = torch.ops.aten._to_sparse.default(ref_inp, **kwargs)
    res_out = flag_gems._to_sparse(res_inp, **kwargs)
    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == res_inp.device
    tu.assert_result_equal(res_inp, ref_inp)

    ref_grad_out = _grad_upstream(ref_out, dtype)
    res_grad_out = _grad_upstream(res_out, dtype)

    (ref_grad,) = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad_out)
    (res_grad,) = torch.autograd.grad(res_out, res_inp, grad_outputs=res_grad_out)

    tu.assert_result_equal(res_grad, ref_grad)
    assert res_grad.device == res_inp.device


_SPARSE_DIM_BACKWARD_ROWS = tu.selected_cases(
    _int64_gated([((8, 12), 1), ((8, 12), 2)]), quick=[]
)


@pytest.mark.to_sparse
@pytest.mark.parametrize("shape,sparse_dim", _SPARSE_DIM_BACKWARD_ROWS)
def test__to_sparse_sparse_dim_backward(shape, sparse_dim):
    res_inp = tu.make_input(torch.float32, shape, ["-1", "1"]).requires_grad_(True)
    ref_inp = tu.to_reference(res_inp)

    ref_out = torch.ops.aten._to_sparse.sparse_dim(ref_inp, sparse_dim)
    res_out = flag_gems._to_sparse(res_inp, sparse_dim)
    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == res_inp.device
    tu.assert_result_equal(res_inp, ref_inp)

    ref_grad_out = _grad_upstream(ref_out, torch.float32)
    res_grad_out = _grad_upstream(res_out, torch.float32)

    (ref_grad,) = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad_out)
    (res_grad,) = torch.autograd.grad(res_out, res_inp, grad_outputs=res_grad_out)

    tu.assert_result_equal(res_grad, ref_grad)
    assert res_grad.device == res_inp.device


_SPECIAL_COO = tu.selected_cases(tu.special_value_cases(_SPECIAL_DTYPES), quick=[])


@pytest.mark.to_sparse
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_COO)
def test__to_sparse_special_values(dtype, scenario):
    # make_special_input is 1-D, so this is the rank-1 COO call form.
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_sparse(ref_inp)
    res_out = flag_gems._to_sparse(inp)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    tu.assert_result_equal(inp, ref_inp)


# Dense->COO FP8 is the measured vendor gap documented above, so the compressed
# layouts carry those cases. The generator supplies a 1-D payload, viewed as a
# single matrix row.
_SPECIAL_COMPRESSED_LAYOUTS = tu.selected_cases(
    [(torch.sparse_csr, None), (torch.sparse_bsr, [1, 1])], quick=[]
)
_SPECIAL_COMPRESSED_DTYPES = _int64_gated(_SPECIAL_DTYPES + _FP8_DTYPES)
_SPECIAL_COMPRESSED = tu.selected_cases(
    tu.special_value_cases(_SPECIAL_COMPRESSED_DTYPES), quick=[]
)


@pytest.mark.to_sparse
@pytest.mark.parametrize("layout,blocksize", _SPECIAL_COMPRESSED_LAYOUTS)
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_COMPRESSED)
def test__to_sparse_special_values_compressed(layout, blocksize, dtype, scenario):
    inp = tu.make_special_input(dtype, scenario).reshape(1, 5)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_sparse.default(
        ref_inp, layout=layout, blocksize=blocksize
    )
    res_out = flag_gems._to_sparse(inp, layout=layout, blocksize=blocksize)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    tu.assert_result_equal(inp, ref_inp)


# Argument validation, kept in both modes. Every row raises for the native
# operator as probed on the exact overload (invalid layout values, dense_dim out
# of range, blocksize zero/negative on either axis, wrong length, non-dividing
# pairs, blocksize on COO/CSR, CSR with a nonzero dense_dim, and a rank too low
# for CSR).
_NEGATIVE_ROWS = [
    ((4, 5), {"layout": torch.strided}),
    ((4, 5), {"layout": "csr"}),
    ((4, 5), {"layout": 3}),
    ((4, 5), {"dense_dim": 5}),
    ((4, 5), {"dense_dim": -1}),
    ((4, 5), {"dense_dim": 2}),
    ((4, 5), {"dense_dim": 1.5}),
    ((4, 5, 6), {"dense_dim": 3}),
    ((4, 5), {"layout": torch.sparse_bsr, "blocksize": [0, 0]}),
    ((4, 5), {"layout": torch.sparse_bsr, "blocksize": [0, 2]}),
    ((4, 5), {"layout": torch.sparse_bsr, "blocksize": [2, 0]}),
    ((4, 5), {"layout": torch.sparse_bsr, "blocksize": [-2, 2]}),
    ((4, 5), {"layout": torch.sparse_bsr, "blocksize": [2, -2]}),
    ((4, 5), {"layout": torch.sparse_bsr, "blocksize": [-2, -2]}),
    ((4, 5), {"layout": torch.sparse_bsr, "blocksize": [2]}),
    ((4, 5), {"layout": torch.sparse_bsr, "blocksize": [2, 2, 2]}),
    ((4, 5), {"layout": torch.sparse_bsc, "blocksize": [0, 2]}),
    ((4, 5), {"layout": torch.sparse_bsc, "blocksize": [-2, 2]}),
    ((4, 4), {"layout": torch.sparse_bsr, "blocksize": [3, 3]}),
    ((4, 4), {"layout": torch.sparse_coo, "blocksize": [2, 2]}),
    ((4, 4), {"layout": torch.sparse_csr, "blocksize": [2, 2]}),
    ((4, 4), {"layout": torch.sparse_csr, "dense_dim": 1}),
    ((5,), {"layout": torch.sparse_csr}),
    ((), {"layout": torch.sparse_csr}),
]


@pytest.mark.to_sparse
@pytest.mark.parametrize("shape,kwargs", _NEGATIVE_ROWS)
def test__to_sparse_invalid_arguments(shape, kwargs):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError, IndexError)):
        flag_gems._to_sparse(inp, **kwargs)


# sparse_dim validity is rank-dependent, so these rows are not generalized
# across ranks: a rank-2 input admits 1..2 and rejects 0 ('sparse_dim argument
# must be in >0 when self.dim()>0') and 3 ('... must be in [0,2] range, but 3 is
# given'), while the scalar input admits only 0 and rejects 1 ('... must be in
# [0,0] range, but 1 is given').
_NEGATIVE_SPARSE_DIM_ROWS = [
    ((4, 5), 3),
    ((4, 5), 0),
    ((4, 5), -1),
    ((4, 5), 1.5),
    ((4, 5, 6), 4),
    ((), 1),
]


@pytest.mark.to_sparse
@pytest.mark.parametrize("shape,sparse_dim", _NEGATIVE_SPARSE_DIM_ROWS)
def test__to_sparse_invalid_sparse_dim(shape, sparse_dim):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError, IndexError)):
        flag_gems._to_sparse(inp, sparse_dim)
