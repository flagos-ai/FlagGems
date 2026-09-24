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

# crow_indices_copy returns an independent, contiguous copy of sparse indices.
# Prefill out buffers with -1 so missing writes cannot match valid indices.
_CROW_DTYPES = tu.REQUIRED_DTYPES + [torch.int16, torch.float64, torch.bool]
_VALUE_RANGE_DTYPES = [dtype for dtype in _CROW_DTYPES if dtype != torch.bool]

# (layout, size, nnz, block_shape), including batched and empty storage.
_CROW_CASES = tu.selected_cases(
    [
        ("csr", (5, 4), 6, None),
        ("csr", (4, 1), 3, None),
        ("csr", (1, 5), 2, None),
        ("csr", (8, 8), 16, None),
        ("csr", (16, 32), 40, None),
        ("csr", (32, 16), 80, None),
        ("csr", (3, 3), 9, None),
        ("csr", (3, 4), 0, None),
        ("csr_batch", (2, 6, 8), 12, None),
        ("bsr", (4, 6), 4, (2, 2)),
        ("bsr", (8, 8), 8, (2, 2)),
        ("bsr", (6, 6), 6, (3, 2)),
        ("bsr", (4, 6), 0, (2, 2)),
        ("bsr_batch", (2, 4, 6), 6, (2, 2)),
        ("csr_batch", (7, 3, 12, 4, 5), 20, None),
        ("bsr", (12, 12), 12, (3, 4)),
        ("bsr_batch", (2, 8, 12), 6, (4, 4)),
    ],
    quick=[("csr_batch", (2, 19, 7), 20, None)],
)

_CROW_RANGE_CASES = tu.selected_cases(
    [
        ("csr", (5, 4), 6, None),
        ("csr_batch", (2, 6, 8), 12, None),
        ("bsr", (4, 6), 4, (2, 2)),
    ],
    quick=[("csr", (5, 4), 6, None)],
)

# (layout, matrix_shape, values_shape, compressed_indices, plain_indices).
_INDEX_LAYOUT_CASES = [
    (torch.sparse_csr, (4, 6), (4,), [0, 2, 2, 3, 4], [0, 4, 1, 5]),
    (torch.sparse_bsr, (4, 6), (3, 2, 3), [0, 2, 3], [0, 1, 1]),
    (torch.sparse_csr, (4, 6), (0,), [0, 0, 0, 0, 0], []),
    (torch.sparse_bsr, (4, 6), (0, 2, 3), [0, 0, 0], []),
]
_INDEX_CASES = tu.selected_cases(
    [
        (case, batch_shape, dense_shape)
        for case in _INDEX_LAYOUT_CASES
        for batch_shape in [(), (2,), (2, 3)]
        for dense_shape in [(), (2,)]
    ],
    quick=[
        (_INDEX_LAYOUT_CASES[0], (), ()),
        (_INDEX_LAYOUT_CASES[1], (2,), (2,)),
        (_INDEX_LAYOUT_CASES[2], (), ()),
        (_INDEX_LAYOUT_CASES[3], (2,), ()),
    ],
)


def _shape_to_crow_case(shape):
    """Map scalar/1-D shapes to matrices and retain batch dimensions otherwise."""
    if len(shape) == 0:
        size = (1, 1)
    elif len(shape) == 1:
        size = (shape[0], 1)
    else:
        size = tuple(shape)
    rows, cols = size[-2], size[-1]
    nnz = min(rows * cols, 16)
    if len(size) == 2:
        return ("csr", size, nnz, None)
    return ("csr_batch", size, nnz, None)


_SHAPE_CASES = [_shape_to_crow_case(shape) for shape in tu.selected_shapes()]


def _make_crow(n_compressed, nnz):
    counts = torch.full((n_compressed,), nnz // n_compressed, dtype=torch.long)
    counts[: nnz % n_compressed] += 1
    return torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])


def _make_crow_batch(n_batch, n_compressed, nnz):
    return _make_crow(n_compressed, nnz).expand(n_batch, -1).contiguous()


def _make_values(dtype, values_shape, value_range, gen):
    # Seeded bool values ignore the numeric range.
    if dtype == torch.bool:
        return torch.randint(0, 2, values_shape, dtype=dtype, generator=gen).to(
            flag_gems.device
        )
    # Fill integer ranges that clamp to a single value.
    if not (dtype.is_floating_point or dtype.is_complex):
        lo_bound, hi_bound = tu.dtype_bounds(dtype)
        low = int(min(max(tu.resolve_bound(value_range[0], dtype), lo_bound), hi_bound))
        high = int(
            min(max(tu.resolve_bound(value_range[1], dtype), lo_bound), hi_bound)
        )
        if low == high:
            return torch.full(values_shape, low, device=flag_gems.device, dtype=dtype)
    return tu.make_input(dtype, values_shape, list(value_range))


def _make_csr(size, nnz, dtype, gen, device, value_range):
    n_rows, n_cols = size
    assert 0 <= nnz <= n_rows * n_cols
    crow = _make_crow(n_rows, nnz)
    col = torch.arange(nnz) - torch.repeat_interleave(crow[:-1], crow.diff())
    values = _make_values(dtype, (nnz,), value_range, gen)
    return torch.sparse_csr_tensor(crow, col, values, size=size, device=device)


def _make_csr_batch(size, nnz, dtype, gen, device, value_range):
    batch_dims, n_rows, n_cols = size[:-2], size[-2], size[-1]
    assert 0 <= nnz <= n_rows * n_cols
    n_batch = math.prod(batch_dims)
    crow = _make_crow_batch(n_batch, n_rows, nnz)
    col = (
        (torch.arange(nnz) - torch.repeat_interleave(crow[0][:-1], crow[0].diff()))
        .expand(n_batch, -1)
        .contiguous()
    )
    values = _make_values(dtype, (n_batch, nnz), value_range, gen)
    return torch.sparse_csr_tensor(
        crow.view(batch_dims + (n_rows + 1,)),
        col.view(batch_dims + (nnz,)),
        values.view(batch_dims + (nnz,)),
        size=size,
        device=device,
    )


def _make_bsr(size, nnz, blocks, dtype, gen, device, value_range):
    n_rows, n_cols = size
    block_rows, block_cols = blocks
    assert n_rows % block_rows == n_cols % block_cols == 0
    n_row_blocks = n_rows // block_rows
    n_col_blocks = n_cols // block_cols
    assert 0 <= nnz <= n_row_blocks * n_col_blocks
    crow = _make_crow(n_row_blocks, nnz)
    col = torch.arange(nnz) - torch.repeat_interleave(crow[:-1], crow.diff())
    values = _make_values(dtype, (nnz, block_rows, block_cols), value_range, gen)
    return torch.sparse_bsr_tensor(crow, col, values, size=size, device=device)


def _make_bsr_batch(size, nnz, blocks, dtype, gen, device, value_range):
    batch_dims, n_rows, n_cols = size[:-2], size[-2], size[-1]
    block_rows, block_cols = blocks
    assert n_rows % block_rows == n_cols % block_cols == 0
    n_batch = math.prod(batch_dims)
    n_row_blocks = n_rows // block_rows
    n_col_blocks = n_cols // block_cols
    assert 0 <= nnz <= n_row_blocks * n_col_blocks
    crow = _make_crow_batch(n_batch, n_row_blocks, nnz)
    col = (
        (torch.arange(nnz) - torch.repeat_interleave(crow[0][:-1], crow[0].diff()))
        .expand(n_batch, -1)
        .contiguous()
    )
    values = _make_values(
        dtype, (n_batch, nnz, block_rows, block_cols), value_range, gen
    )
    return torch.sparse_bsr_tensor(
        crow.view(batch_dims + (n_row_blocks + 1,)),
        col.view(batch_dims + (nnz,)),
        values.view(batch_dims + (nnz, block_rows, block_cols)),
        size=size,
        device=device,
    )


def _make_input(layout, size, nnz, blocks, dtype, value_range=("-1", "1"), seed=0):
    gen = torch.Generator("cpu").manual_seed(seed)
    if layout == "csr":
        return _make_csr(size, nnz, dtype, gen, flag_gems.device, value_range)
    if layout == "csr_batch":
        return _make_csr_batch(size, nnz, dtype, gen, flag_gems.device, value_range)
    if layout == "bsr":
        return _make_bsr(size, nnz, blocks, dtype, gen, flag_gems.device, value_range)
    return _make_bsr_batch(size, nnz, blocks, dtype, gen, flag_gems.device, value_range)


def _expected_crow_shape(case):
    layout, size, _, blocks = case
    n_rows = size[-2]
    if layout in ("bsr", "bsr_batch"):
        n_compressed = n_rows // blocks[0]
    else:
        n_compressed = n_rows
    if layout in ("csr", "bsr"):
        return (n_compressed + 1,)
    return size[:-2] + (n_compressed + 1,)


def _assert_copy_semantics(res, ref, inp, ref_inp):
    assert res.is_contiguous()
    tu.assert_result_equal(res, ref)
    # Empty storage has a null pointer.
    if res.numel() > 0:
        assert res.data_ptr() != inp.crow_indices().data_ptr()
    utils.gems_assert_equal(inp, ref_inp, equal_nan=True)


def _make_index_layout(case, batch_shape, dense_shape, dtype, index_dtype):
    layout, matrix_shape, values_shape, compressed, plain = case
    compressed = torch.tensor(compressed, dtype=index_dtype, device=flag_gems.device)
    plain = torch.tensor(plain, dtype=index_dtype, device=flag_gems.device)
    compressed = compressed.expand(batch_shape + compressed.shape).contiguous()
    plain = plain.expand(batch_shape + plain.shape).contiguous()
    values = tu.make_input(dtype, batch_shape + values_shape + dense_shape, ["-1", "1"])
    inp = torch.sparse_compressed_tensor(
        compressed,
        plain,
        values,
        size=batch_shape + matrix_shape + dense_shape,
        layout=layout,
        check_invariants=True,
    )
    return inp


@pytest.mark.crow_indices_copy
@pytest.mark.parametrize("case,batch_shape,dense_shape", _INDEX_CASES)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", _CROW_DTYPES)
def test_crow_indices_copy_index_layouts(
    case, batch_shape, dense_shape, dtype, index_dtype
):
    inp = _make_index_layout(case, batch_shape, dense_shape, dtype, index_dtype)
    ref_inp = tu.to_reference(inp)
    ref_out = torch.ops.aten.crow_indices_copy(ref_inp)
    res_out = flag_gems.crow_indices_copy(inp)
    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices_copy_out
@pytest.mark.parametrize("case,batch_shape,dense_shape", _INDEX_CASES)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", _CROW_DTYPES)
def test_crow_indices_copy_out_index_layouts(
    case, batch_shape, dense_shape, dtype, index_dtype
):
    inp = _make_index_layout(case, batch_shape, dense_shape, dtype, index_dtype)
    ref_inp = tu.to_reference(inp)
    out = torch.full_like(inp.crow_indices(), -1)
    ref_out = torch.full_like(ref_inp.crow_indices(), -1)
    torch.ops.aten.crow_indices_copy(ref_inp, out=ref_out)
    res_ret = flag_gems.crow_indices_copy(inp, out=out)
    assert res_ret is out
    _assert_copy_semantics(out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices_copy
@pytest.mark.parametrize("case", _CROW_CASES)
@pytest.mark.parametrize("dtype", _CROW_DTYPES)
def test_crow_indices_copy(case, dtype):
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.crow_indices_copy(ref_inp)
    res_out = flag_gems.crow_indices_copy(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices_copy_out
@pytest.mark.parametrize("case", _CROW_CASES)
@pytest.mark.parametrize("dtype", _CROW_DTYPES)
def test_crow_indices_copy_out(case, dtype):
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype)
    ref_inp = tu.to_reference(inp)
    out = torch.full(
        _expected_crow_shape(case), -1, dtype=torch.long, device=inp.device
    )
    ref_out = torch.full(
        _expected_crow_shape(case), -1, dtype=torch.long, device=ref_inp.device
    )

    torch.ops.aten.crow_indices_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems.crow_indices_copy(inp, out=out)

    # The .out variant must write into and return the out tensor itself.
    assert res_ret is out
    _assert_copy_semantics(out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices_copy
@pytest.mark.parametrize("case", _SHAPE_CASES)
@pytest.mark.parametrize("dtype", _CROW_DTYPES)
def test_crow_indices_copy_spec_shapes(case, dtype):
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.crow_indices_copy(ref_inp)
    res_out = flag_gems.crow_indices_copy(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices_copy_out
@pytest.mark.parametrize("case", _SHAPE_CASES)
@pytest.mark.parametrize("dtype", _CROW_DTYPES)
def test_crow_indices_copy_out_spec_shapes(case, dtype):
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype)
    ref_inp = tu.to_reference(inp)
    out = torch.full(
        _expected_crow_shape(case), -1, dtype=torch.long, device=inp.device
    )
    ref_out = torch.full(
        _expected_crow_shape(case), -1, dtype=torch.long, device=ref_inp.device
    )

    torch.ops.aten.crow_indices_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems.crow_indices_copy(inp, out=out)

    assert res_ret is out
    _assert_copy_semantics(out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices_copy
@pytest.mark.parametrize("case", _CROW_RANGE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _VALUE_RANGE_DTYPES)
def test_crow_indices_copy_value_ranges(case, value_range, dtype):
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype, value_range=value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.crow_indices_copy(ref_inp)
    res_out = flag_gems.crow_indices_copy(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices_copy_out
@pytest.mark.parametrize("case", _CROW_RANGE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _VALUE_RANGE_DTYPES)
def test_crow_indices_copy_out_value_ranges(case, value_range, dtype):
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype, value_range=value_range)
    ref_inp = tu.to_reference(inp)
    out = torch.full(
        _expected_crow_shape(case), -1, dtype=torch.long, device=inp.device
    )
    ref_out = torch.full(
        _expected_crow_shape(case), -1, dtype=torch.long, device=ref_inp.device
    )

    torch.ops.aten.crow_indices_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems.crow_indices_copy(inp, out=out)

    assert res_ret is out
    _assert_copy_semantics(out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices_copy
@pytest.mark.parametrize("dtype", _CROW_DTYPES)
def test_crow_indices_copy_empty_bsr(dtype):
    inp = _make_input("bsr", (4, 6), 0, (2, 2), dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.crow_indices_copy(ref_inp)
    res_out = flag_gems.crow_indices_copy(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices_copy_out
@pytest.mark.parametrize("dtype", _CROW_DTYPES)
def test_crow_indices_copy_out_empty_bsr(dtype):
    inp = _make_input("bsr", (4, 6), 0, (2, 2), dtype)
    ref_inp = tu.to_reference(inp)
    out = torch.full((3,), -1, dtype=torch.long, device=inp.device)
    ref_out = torch.full((3,), -1, dtype=torch.long, device=ref_inp.device)

    torch.ops.aten.crow_indices_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems.crow_indices_copy(inp, out=out)

    assert res_ret is out
    _assert_copy_semantics(out, ref_out, inp, ref_inp)


def _uncoalesced_csr(dtype):
    # Keep the repeated (0, 0) entry in storage order.
    shape = (4, 3)
    crow = torch.tensor([0, 3, 3, 5, 5], dtype=torch.long, device=flag_gems.device)
    cols = torch.tensor([0, 0, 2, 1, 2], dtype=torch.long, device=flag_gems.device)
    values = _make_values(dtype, (5,), ["-1", "1"], torch.Generator("cpu"))
    return torch.sparse_csr_tensor(crow, cols, values, shape)


@pytest.mark.crow_indices_copy
@pytest.mark.parametrize("dtype", _CROW_DTYPES)
def test_crow_indices_copy_unchecked_uncoalesced(dtype):
    inp = _uncoalesced_csr(dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.crow_indices_copy(ref_inp)
    res_out = flag_gems.crow_indices_copy(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices_copy_out
@pytest.mark.parametrize("dtype", _CROW_DTYPES)
def test_crow_indices_copy_out_unchecked_uncoalesced(dtype):
    inp = _uncoalesced_csr(dtype)
    ref_inp = tu.to_reference(inp)
    out = torch.full((5,), -1, dtype=torch.long, device=inp.device)
    ref_out = torch.full((5,), -1, dtype=torch.long, device=ref_inp.device)

    torch.ops.aten.crow_indices_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems.crow_indices_copy(inp, out=out)

    assert res_ret is out
    _assert_copy_semantics(out, ref_out, inp, ref_inp)


def _special_csr(dtype, scenario):
    shape = (3, 4)
    crow = torch.tensor([0, 2, 4, 7], dtype=torch.long, device=flag_gems.device)
    cols = torch.tensor(
        [0, 1, 0, 2, 0, 1, 2], dtype=torch.long, device=flag_gems.device
    )
    values = tu.make_special_input(dtype, scenario).repeat(2)[:7]
    return torch.sparse_csr_tensor(crow, cols, values, shape)


@pytest.mark.crow_indices_copy
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_CROW_DTYPES))
)
def test_crow_indices_copy_nan_inf_values(dtype, scenario):
    inp = _special_csr(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.crow_indices_copy(ref_inp)
    res_out = flag_gems.crow_indices_copy(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices_copy_out
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_CROW_DTYPES))
)
def test_crow_indices_copy_out_nan_inf_values(dtype, scenario):
    inp = _special_csr(dtype, scenario)
    ref_inp = tu.to_reference(inp)
    out = torch.full((4,), -1, dtype=torch.long, device=inp.device)
    ref_out = torch.full((4,), -1, dtype=torch.long, device=ref_inp.device)

    torch.ops.aten.crow_indices_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems.crow_indices_copy(inp, out=out)

    assert res_ret is out
    _assert_copy_semantics(out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices_copy
def test_crow_indices_copy_negative_dense():
    inp = tu.make_input(torch.float32, (3, 4), ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.crow_indices_copy(tu.to_reference(inp))
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.crow_indices_copy(inp)


@pytest.mark.crow_indices_copy
def test_crow_indices_copy_negative_csc():
    ccol_indices = torch.tensor([0, 2, 4], dtype=torch.long, device=flag_gems.device)
    row_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=flag_gems.device)
    values = tu.make_input(torch.float32, (4,), ["-1", "1"])
    inp = torch.sparse_csc_tensor(
        ccol_indices, row_indices, values, (4, 2), device=flag_gems.device
    )
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.crow_indices_copy(tu.to_reference(inp))
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.crow_indices_copy(inp)


@pytest.mark.crow_indices_copy
def test_crow_indices_copy_negative_coo():
    indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    values = torch.ones(2, dtype=torch.float32)
    inp = torch.sparse_coo_tensor(indices, values, (3, 3), device=flag_gems.device)
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        torch.ops.aten.crow_indices_copy(tu.to_reference(inp))
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        flag_gems.crow_indices_copy(inp)


@pytest.mark.crow_indices_copy
def test_crow_indices_copy_negative_non_tensor():
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.crow_indices_copy(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.crow_indices_copy(3.14)


@pytest.mark.crow_indices_copy_out
def test_crow_indices_copy_out_negative_dense():
    inp = tu.make_input(torch.float32, (3, 4), ["-1", "1"])
    out = torch.empty(5, dtype=torch.long, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.crow_indices_copy.out(tu.to_reference(inp), out=out)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.crow_indices_copy(inp, out=out)


@pytest.mark.crow_indices_copy_out
def test_crow_indices_copy_out_negative_csc():
    ccol_indices = torch.tensor([0, 2, 4], dtype=torch.long, device=flag_gems.device)
    row_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=flag_gems.device)
    values = tu.make_input(torch.float32, (4,), ["-1", "1"])
    inp = torch.sparse_csc_tensor(
        ccol_indices, row_indices, values, (4, 2), device=flag_gems.device
    )
    out = torch.empty(5, dtype=torch.long, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.crow_indices_copy.out(tu.to_reference(inp), out=out)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.crow_indices_copy(inp, out=out)


@pytest.mark.crow_indices_copy_out
def test_crow_indices_copy_out_negative_coo():
    indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    values = torch.ones(2, dtype=torch.float32)
    inp = torch.sparse_coo_tensor(indices, values, (3, 3), device=flag_gems.device)
    out = torch.empty(5, dtype=torch.long, device=flag_gems.device)
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        torch.ops.aten.crow_indices_copy.out(tu.to_reference(inp), out=out)
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        flag_gems.crow_indices_copy(inp, out=out)


@pytest.mark.crow_indices_copy_out
def test_crow_indices_copy_out_negative_wrong_dtype():
    inp = _make_input("csr", (5, 4), 6, None, torch.float32)
    ref_inp = tu.to_reference(inp)
    out = torch.empty(6, dtype=torch.float32, device=inp.device)
    ref_out = torch.empty(6, dtype=torch.float32, device=ref_inp.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten.crow_indices_copy.out(ref_inp, out=ref_out)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.crow_indices_copy(inp, out=out)
