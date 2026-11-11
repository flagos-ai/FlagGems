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


def _tensor_metadata(inp):
    return (
        type(inp),
        inp.dtype,
        inp.device,
        inp.shape,
        inp.layout,
        inp.requires_grad,
        inp._version,
    )


def _input_parts(inp):
    names = (
        ("crow_indices", "col_indices")
        if inp.layout
        in (
            torch.sparse_csr,
            torch.sparse_bsr,
        )
        else ("ccol_indices", "row_indices")
    )
    return tuple(getattr(inp, name)() for name in names) + (inp.values(),)


def _part_metadata(inp):
    storage = inp.untyped_storage()
    return (
        _tensor_metadata(inp),
        inp.stride(),
        inp.storage_offset(),
        inp.is_conj(),
        inp.is_neg(),
        storage._cdata,
        inp.data_ptr(),
        storage.data_ptr(),
        storage.nbytes(),
    )


def _capture_input(inp):
    parts = _input_parts(inp)
    # Hold each original storage and capture its addresses as independent ints:
    # a live Storage object alone follows a later resize/rebind.
    return (
        _tensor_metadata(inp),
        inp.sparse_dim(),
        inp.dense_dim(),
        tuple(_part_metadata(part) for part in parts),
        tuple(part.untyped_storage() for part in parts),
    )


def _assert_input_unchanged(inp, expected):
    metadata, sparse_dim, dense_dim, parts, storages = expected
    assert _tensor_metadata(inp) == metadata
    assert (inp.sparse_dim(), inp.dense_dim()) == (sparse_dim, dense_dim)
    assert tuple(_part_metadata(part) for part in _input_parts(inp)) == parts
    # Strong original references stay alive throughout these comparisons.
    assert len(storages) == len(parts)


def _device(device):
    expected = torch.device(device)
    if expected.index is None and expected.type not in ("cpu", "meta"):
        expected = torch.device(
            expected.type, flag_gems.runtime.torch_device_fn.current_device()
        )
    return expected


def _view_from_storage(inp, storage, device):
    if flag_gems.vendor_name == "kunlunxin" and inp.dtype == torch.float64:
        # A typed device factory narrows FP64; a byte view retains its storage.
        raw = torch.empty(0, dtype=torch.uint8, device=device).set_(
            storage, 0, (storage.nbytes(),), (1,)
        )
        out = raw.view(inp.dtype).as_strided(
            inp.size(), inp.stride(), inp.storage_offset()
        )
    else:
        out = torch.empty(0, dtype=inp.dtype, device=device).set_(
            storage, inp.storage_offset(), inp.size(), inp.stride()
        )
    if inp.is_conj():
        out = out.conj()
    if inp.is_neg():
        out = torch._neg_view(out)
    assert out.dtype == inp.dtype and out.device == _device(device)
    assert out.shape == inp.shape and out.stride() == inp.stride()
    assert out.storage_offset() == inp.storage_offset()
    assert (out.is_conj(), out.is_neg()) == (inp.is_conj(), inp.is_neg())
    return out.requires_grad_(inp.requires_grad)


def _copy_to(inp, device):
    # Copy physical bytes before restoring the view and lazy flags. A typed
    # cross-device copy can resolve conjugation or require unavailable kernels.
    nbytes = inp.untyped_storage().nbytes()
    raw = torch.empty(0, dtype=torch.uint8, device=inp.device).set_(
        inp.untyped_storage(), 0, (nbytes,), (1,)
    )
    raw = raw.to(device).clone()
    return _view_from_storage(inp, raw.untyped_storage(), raw.device)


def _check_values(inp, dtype, shape):
    assert inp.shape == torch.Size(shape)
    assert inp.device == _device(flag_gems.device)
    assert inp.dtype == dtype
    return inp


def _cpu_values(dtype, shape, value_range):
    # The original factory recipe, with the same bounds and degenerate fills.
    low, high = (tu.resolve_bound(v, dtype) for v in value_range)
    if not dtype.is_floating_point:
        low, high = int(low), int(high)
        dtype_min, dtype_max = tu.dtype_bounds(dtype)
        low = max(low, int(dtype_min))
        high = min(high, int(dtype_max))
        low = min(low, high)
    if low == high:
        return torch.full(shape, low, dtype=dtype, device="cpu")
    return torch.testing.make_tensor(
        shape, dtype=dtype, device="cpu", low=low, high=high
    )


def _make_payload(dtype, shape, value_range):
    if flag_gems.vendor_name == "kunlunxin" and dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float64,
    ):
        inp = _copy_to(_cpu_values(dtype, shape, value_range), flag_gems.device)
    else:
        inp = tu.make_input(dtype, shape, value_range)
    return _check_values(inp, dtype, shape)


def _make_special_payload(dtype, scenario):
    if (
        flag_gems.vendor_name == "kunlunxin"
        and dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.float64)
    ) or (flag_gems.vendor_name == "ascend" and dtype == torch.float64):
        # Ascend's device-side float64 cast narrows these payloads to float32.
        payloads = {
            "nan": [float("nan"), 0.0, -0.0, 1.0, -1.0],
            "inf": [float("inf"), float("-inf"), 0.0, -0.0, 1.0],
            "mixed": [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
        }
        # Encode and cycle the original five payloads on CPU before transfer.
        value = torch.tensor(payloads[scenario], dtype=torch.float32, device="cpu").to(
            dtype
        )
        return _check_values(
            _copy_to(value.repeat(2)[:7], flag_gems.device), dtype, (7,)
        )
    return _check_values(
        tu.make_special_input(dtype, scenario).repeat(2)[:7], dtype, (7,)
    )


def _assert_values_equal(actual, expected, *, device, equal_nan=False):
    if flag_gems.vendor_name == "kunlunxin" and actual.dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float64,
    ):
        # Retain the original contract before CPU observation; keep FP64
        # precision while decoding FP8 for the missing comparison kernels.
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape and actual.layout == expected.layout
        assert actual.device == device
        assert expected.device == (
            torch.device("cpu") if utils.TO_CPU else actual.device
        )
        actual = _copy_to(actual, "cpu")
        expected = _copy_to(expected, "cpu")
        if actual.dtype != torch.float64:
            actual = actual.float()
            expected = expected.float()
    utils.gems_assert_equal(actual, expected, equal_nan=equal_nan)


# Select every overload and contract case with the operator marker.
pytestmark = pytest.mark.ccol_indices_copy

# ccol_indices_copy returns an independent, contiguous copy of sparse indices.
# Prefill out buffers with -1 so missing writes cannot match valid indices.
_CCOLS_DTYPES = tu.REQUIRED_DTYPES + [torch.int16, torch.float64, torch.bool]
_VALUE_RANGE_DTYPES = [dtype for dtype in _CCOLS_DTYPES if dtype != torch.bool]

# (layout, size, nnz, block_shape), including batched and empty storage.
_CCOLS_CASES = tu.selected_cases(
    [
        ("csc", (5, 4), 6, None),
        ("csc", (4, 1), 3, None),
        ("csc", (1, 5), 2, None),
        ("csc", (8, 8), 16, None),
        ("csc", (16, 32), 40, None),
        ("csc", (32, 16), 80, None),
        ("csc", (3, 3), 9, None),
        ("csc", (3, 4), 0, None),
        ("csc_batch", (2, 6, 8), 12, None),
        ("bsc", (4, 6), 4, (2, 2)),
        ("bsc", (8, 8), 8, (2, 2)),
        ("bsc", (6, 6), 6, (3, 2)),
        ("bsc", (4, 6), 0, (2, 2)),
        ("bsc_batch", (2, 4, 6), 6, (2, 2)),
        ("csc_batch", (7, 3, 12, 4, 5), 20, None),
        ("bsc", (12, 12), 12, (3, 4)),
        ("bsc_batch", (2, 8, 12), 6, (4, 4)),
    ],
    quick=[("csc_batch", (2, 19, 7), 20, None)],
)

_CCOLS_RANGE_CASES = tu.selected_cases(
    [
        ("csc", (5, 4), 6, None),
        ("csc_batch", (2, 6, 8), 12, None),
        ("bsc", (4, 6), 4, (2, 2)),
    ],
    quick=[("csc", (5, 4), 6, None)],
)

# (layout, matrix_shape, values_shape, compressed_indices, plain_indices).
_INDEX_LAYOUT_CASES = [
    (torch.sparse_csc, (6, 4), (4,), [0, 2, 2, 3, 4], [0, 4, 1, 5]),
    (torch.sparse_bsc, (6, 4), (3, 3, 2), [0, 2, 3], [0, 1, 1]),
    (torch.sparse_csc, (6, 4), (0,), [0, 0, 0, 0, 0], []),
    (torch.sparse_bsc, (6, 4), (0, 3, 2), [0, 0, 0], []),
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


def _shape_to_csc_case(shape):
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
        return ("csc", size, nnz, None)
    return ("csc_batch", size, nnz, None)


# Scalar and singleton spec shapes map to the same sparse matrix.
_SHAPE_CASES = list(
    dict.fromkeys(_shape_to_csc_case(shape) for shape in tu.selected_shapes())
)


def _make_ccol(n_compressed, nnz):
    counts = torch.full((n_compressed,), nnz // n_compressed, dtype=torch.long)
    counts[: nnz % n_compressed] += 1
    return torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])


def _make_ccol_batch(n_batch, n_compressed, nnz):
    return _make_ccol(n_compressed, nnz).expand(n_batch, -1).contiguous()


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
    return _make_payload(dtype, values_shape, list(value_range))


def _make_csc(size, nnz, dtype, gen, device, value_range):
    n_rows, n_cols = size
    assert 0 <= nnz <= n_rows * n_cols
    ccol = _make_ccol(n_cols, nnz)
    rows = torch.arange(nnz) - torch.repeat_interleave(ccol[:-1], ccol.diff())
    values = _make_values(dtype, (nnz,), value_range, gen)
    return torch.sparse_csc_tensor(ccol, rows, values, size=size, device=device)


def _make_csc_batch(size, nnz, dtype, gen, device, value_range):
    batch_dims, n_rows, n_cols = size[:-2], size[-2], size[-1]
    assert 0 <= nnz <= n_rows * n_cols
    n_batch = math.prod(batch_dims)
    ccol = _make_ccol_batch(n_batch, n_cols, nnz)
    rows = (
        (torch.arange(nnz) - torch.repeat_interleave(ccol[0][:-1], ccol[0].diff()))
        .expand(n_batch, -1)
        .contiguous()
    )
    values = _make_values(dtype, (n_batch, nnz), value_range, gen)
    return torch.sparse_csc_tensor(
        ccol.view(batch_dims + (n_cols + 1,)),
        rows.view(batch_dims + (nnz,)),
        values.view(batch_dims + (nnz,)),
        size=size,
        device=device,
    )


def _make_bsc(size, nnz, blocks, dtype, gen, device, value_range):
    n_rows, n_cols = size
    block_rows, block_cols = blocks
    assert n_rows % block_rows == n_cols % block_cols == 0
    n_col_blocks = n_cols // block_cols
    n_row_blocks = n_rows // block_rows
    assert 0 <= nnz <= n_row_blocks * n_col_blocks
    ccol = _make_ccol(n_col_blocks, nnz)
    row = torch.arange(nnz) - torch.repeat_interleave(ccol[:-1], ccol.diff())
    values = _make_values(dtype, (nnz, block_rows, block_cols), value_range, gen)
    # torch.sparse_bsc_tensor infers the block size from the trailing dims of
    # the values tensor (values_shape == (nnz, block_rows, block_cols)).
    return torch.sparse_bsc_tensor(ccol, row, values, size=size, device=device)


def _make_bsc_batch(size, nnz, blocks, dtype, gen, device, value_range):
    batch_dims, n_rows, n_cols = size[:-2], size[-2], size[-1]
    block_rows, block_cols = blocks
    assert n_rows % block_rows == n_cols % block_cols == 0
    n_batch = math.prod(batch_dims)
    n_col_blocks = n_cols // block_cols
    n_row_blocks = n_rows // block_rows
    assert 0 <= nnz <= n_row_blocks * n_col_blocks
    ccol = _make_ccol_batch(n_batch, n_col_blocks, nnz)
    row = (
        (torch.arange(nnz) - torch.repeat_interleave(ccol[0][:-1], ccol[0].diff()))
        .expand(n_batch, -1)
        .contiguous()
    )
    values = _make_values(
        dtype, (n_batch, nnz, block_rows, block_cols), value_range, gen
    )
    return torch.sparse_bsc_tensor(
        ccol.view(batch_dims + (n_col_blocks + 1,)),
        row.view(batch_dims + (nnz,)),
        values.view(batch_dims + (nnz, block_rows, block_cols)),
        size=size,
        device=device,
    )


def _make_input(layout, size, nnz, blocks, dtype, value_range=("-1", "1"), seed=0):
    gen = torch.Generator("cpu").manual_seed(seed)
    if layout == "csc":
        return _make_csc(size, nnz, dtype, gen, flag_gems.device, value_range)
    if layout == "csc_batch":
        return _make_csc_batch(size, nnz, dtype, gen, flag_gems.device, value_range)
    if layout == "bsc":
        return _make_bsc(size, nnz, blocks, dtype, gen, flag_gems.device, value_range)
    return _make_bsc_batch(size, nnz, blocks, dtype, gen, flag_gems.device, value_range)


def _expected_ccol_shape(case):
    layout, size, _, blocks = case
    if layout == "csc":
        return (size[-1] + 1,)
    if layout == "csc_batch":
        return size[:-2] + (size[-1] + 1,)
    n_col_blocks = size[-1] // blocks[1]
    if layout == "bsc":
        return (n_col_blocks + 1,)
    return size[:-2] + (n_col_blocks + 1,)


def _assert_copy_semantics(res, ref, inp, ref_inp, expected_device):
    assert inp.dtype == ref_inp.dtype and inp.shape == ref_inp.shape
    assert inp.layout == ref_inp.layout
    assert inp.device == expected_device
    assert ref_inp.device == (torch.device("cpu") if utils.TO_CPU else inp.device)
    assert res.dtype == ref.dtype and res.shape == ref.shape
    assert res.layout == ref.layout == torch.strided and res.device == inp.device
    assert ref.device == ref_inp.device
    # Empty copies also need an independent storage object.
    assert res.untyped_storage()._cdata != inp.ccol_indices().untyped_storage()._cdata
    assert res.is_contiguous()
    tu.assert_result_equal(res, ref)
    assert res.data_ptr() != inp.ccol_indices().data_ptr()
    utils.gems_assert_equal(inp.ccol_indices(), ref_inp.ccol_indices())
    utils.gems_assert_equal(inp.row_indices(), ref_inp.row_indices())
    _assert_values_equal(
        inp.values(), ref_inp.values(), device=expected_device, equal_nan=True
    )


def _make_index_layout(case, batch_shape, dense_shape, dtype, index_dtype):
    layout, matrix_shape, values_shape, compressed, plain = case
    compressed = torch.tensor(compressed, dtype=index_dtype, device=flag_gems.device)
    plain = torch.tensor(plain, dtype=index_dtype, device=flag_gems.device)
    compressed = compressed.expand(batch_shape + compressed.shape).contiguous()
    plain = plain.expand(batch_shape + plain.shape).contiguous()
    values = _make_payload(dtype, batch_shape + values_shape + dense_shape, ["-1", "1"])
    inp = torch.sparse_compressed_tensor(
        compressed,
        plain,
        values,
        size=batch_shape + matrix_shape + dense_shape,
        layout=layout,
        check_invariants=True,
    )
    return inp


_CCOLS_INPUT_DTYPE_CASES = [
    pytest.param(
        dtype,
        marks=pytest.mark.skipif(
            flag_gems.vendor_name == "ascend"
            and dtype in (torch.float8_e4m3fn, torch.float8_e5m2),
            reason="Ascend cannot construct or compare FP8 payloads with device kernels",
        ),
    )
    for dtype in _CCOLS_DTYPES
]


@pytest.mark.ccol_indices_copy
@pytest.mark.parametrize("case,batch_shape,dense_shape", _INDEX_CASES)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", _CCOLS_INPUT_DTYPE_CASES)
def test_ccol_indices_copy_index_layouts(
    case, batch_shape, dense_shape, dtype, index_dtype
):
    inp = _make_index_layout(case, batch_shape, dense_shape, dtype, index_dtype)
    ref_inp = tu.to_reference(inp)
    ref_out = torch.ops.aten.ccol_indices_copy(ref_inp)
    _input_state = _capture_input(inp)
    _expected_device = inp.device
    res_out = flag_gems.ccol_indices_copy(inp)
    _assert_input_unchanged(inp, _input_state)
    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, _expected_device)


@pytest.mark.ccol_indices_copy_out
@pytest.mark.parametrize("case,batch_shape,dense_shape", _INDEX_CASES)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", _CCOLS_INPUT_DTYPE_CASES)
def test_ccol_indices_copy_out_index_layouts(
    case, batch_shape, dense_shape, dtype, index_dtype
):
    inp = _make_index_layout(case, batch_shape, dense_shape, dtype, index_dtype)
    ref_inp = tu.to_reference(inp)
    out = torch.full_like(inp.ccol_indices(), -1)
    ref_out = torch.full_like(ref_inp.ccol_indices(), -1)
    torch.ops.aten.ccol_indices_copy(ref_inp, out=ref_out)
    _input_state = _capture_input(inp)
    _expected_device = inp.device
    res_ret = flag_gems.ccol_indices_copy(inp, out=out)
    _assert_input_unchanged(inp, _input_state)
    assert res_ret is out
    _assert_copy_semantics(out, ref_out, inp, ref_inp, _expected_device)


@pytest.mark.ccol_indices_copy
@pytest.mark.parametrize("case", _CCOLS_CASES)
@pytest.mark.parametrize("dtype", _CCOLS_INPUT_DTYPE_CASES)
def test_ccol_indices_copy(case, dtype):
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.ccol_indices_copy(ref_inp)
    _input_state = _capture_input(inp)
    _expected_device = inp.device
    res_out = flag_gems.ccol_indices_copy(inp)
    _assert_input_unchanged(inp, _input_state)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, _expected_device)


@pytest.mark.ccol_indices_copy_out
@pytest.mark.parametrize("case", _CCOLS_CASES)
@pytest.mark.parametrize("dtype", _CCOLS_INPUT_DTYPE_CASES)
def test_ccol_indices_copy_out(case, dtype):
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype)
    ref_inp = tu.to_reference(inp)
    out = torch.full(
        _expected_ccol_shape(case), -1, dtype=torch.long, device=inp.device
    )
    ref_out = torch.full(
        _expected_ccol_shape(case), -1, dtype=torch.long, device=ref_inp.device
    )

    torch.ops.aten.ccol_indices_copy.out(ref_inp, out=ref_out)
    _input_state = _capture_input(inp)
    _expected_device = inp.device
    res_ret = flag_gems.ccol_indices_copy(inp, out=out)
    _assert_input_unchanged(inp, _input_state)

    # The .out variant must write into and return the out tensor itself.
    assert res_ret is out
    _assert_copy_semantics(out, ref_out, inp, ref_inp, _expected_device)


@pytest.mark.ccol_indices_copy
@pytest.mark.parametrize("case", _SHAPE_CASES)
@pytest.mark.parametrize("dtype", _CCOLS_INPUT_DTYPE_CASES)
def test_ccol_indices_copy_spec_shapes(case, dtype):
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.ccol_indices_copy(ref_inp)
    _input_state = _capture_input(inp)
    _expected_device = inp.device
    res_out = flag_gems.ccol_indices_copy(inp)
    _assert_input_unchanged(inp, _input_state)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, _expected_device)


@pytest.mark.ccol_indices_copy_out
@pytest.mark.parametrize("case", _SHAPE_CASES)
@pytest.mark.parametrize("dtype", _CCOLS_INPUT_DTYPE_CASES)
def test_ccol_indices_copy_out_spec_shapes(case, dtype):
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype)
    ref_inp = tu.to_reference(inp)
    out = torch.full(
        _expected_ccol_shape(case), -1, dtype=torch.long, device=inp.device
    )
    ref_out = torch.full(
        _expected_ccol_shape(case), -1, dtype=torch.long, device=ref_inp.device
    )

    torch.ops.aten.ccol_indices_copy.out(ref_inp, out=ref_out)
    _input_state = _capture_input(inp)
    _expected_device = inp.device
    res_ret = flag_gems.ccol_indices_copy(inp, out=out)
    _assert_input_unchanged(inp, _input_state)

    assert res_ret is out
    _assert_copy_semantics(out, ref_out, inp, ref_inp, _expected_device)


# Only indices are copied; one ordinary values range per dtype is sufficient.
# uint8 keeps nonzero samples when available to detect unintended values writes.
_VALUE_DTYPE_RANGE_CASES = [
    pytest.param(
        dtype,
        value_range,
        marks=pytest.mark.skipif(
            flag_gems.vendor_name == "ascend"
            and dtype in (torch.float8_e4m3fn, torch.float8_e5m2),
            reason="Ascend cannot construct or compare FP8 payloads with device kernels",
        ),
    )
    for dtype in _VALUE_RANGE_DTYPES
    for value_range in (
        [["0", "max"]]
        if dtype == torch.uint8 and ["0", "max"] in tu.selected_ranges()
        else tu.selected_ranges()[:1]
    )
]


@pytest.mark.ccol_indices_copy
@pytest.mark.parametrize("case", _CCOLS_RANGE_CASES)
@pytest.mark.parametrize(
    "dtype, value_range",
    _VALUE_DTYPE_RANGE_CASES,
)
def test_ccol_indices_copy_value_ranges(case, value_range, dtype):
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype, value_range=value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.ccol_indices_copy(ref_inp)
    _input_state = _capture_input(inp)
    _expected_device = inp.device
    res_out = flag_gems.ccol_indices_copy(inp)
    _assert_input_unchanged(inp, _input_state)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, _expected_device)


@pytest.mark.ccol_indices_copy_out
@pytest.mark.parametrize("case", _CCOLS_RANGE_CASES)
@pytest.mark.parametrize(
    "dtype, value_range",
    _VALUE_DTYPE_RANGE_CASES,
)
def test_ccol_indices_copy_out_value_ranges(case, value_range, dtype):
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype, value_range=value_range)
    ref_inp = tu.to_reference(inp)
    out = torch.full(
        _expected_ccol_shape(case), -1, dtype=torch.long, device=inp.device
    )
    ref_out = torch.full(
        _expected_ccol_shape(case), -1, dtype=torch.long, device=ref_inp.device
    )

    torch.ops.aten.ccol_indices_copy.out(ref_inp, out=ref_out)
    _input_state = _capture_input(inp)
    _expected_device = inp.device
    res_ret = flag_gems.ccol_indices_copy(inp, out=out)
    _assert_input_unchanged(inp, _input_state)

    assert res_ret is out
    _assert_copy_semantics(out, ref_out, inp, ref_inp, _expected_device)


@pytest.mark.ccol_indices_copy
@pytest.mark.parametrize("dtype", _CCOLS_INPUT_DTYPE_CASES)
def test_ccol_indices_copy_empty_bsc(dtype):
    inp = _make_input("bsc", (4, 6), 0, (2, 2), dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.ccol_indices_copy(ref_inp)
    _input_state = _capture_input(inp)
    _expected_device = inp.device
    res_out = flag_gems.ccol_indices_copy(inp)
    _assert_input_unchanged(inp, _input_state)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, _expected_device)


@pytest.mark.ccol_indices_copy_out
@pytest.mark.parametrize("dtype", _CCOLS_INPUT_DTYPE_CASES)
def test_ccol_indices_copy_out_empty_bsc(dtype):
    inp = _make_input("bsc", (4, 6), 0, (2, 2), dtype)
    ref_inp = tu.to_reference(inp)
    out = torch.full((4,), -1, dtype=torch.long, device=inp.device)
    ref_out = torch.full((4,), -1, dtype=torch.long, device=ref_inp.device)

    torch.ops.aten.ccol_indices_copy.out(ref_inp, out=ref_out)
    _input_state = _capture_input(inp)
    _expected_device = inp.device
    res_ret = flag_gems.ccol_indices_copy(inp, out=out)
    _assert_input_unchanged(inp, _input_state)

    assert res_ret is out
    _assert_copy_semantics(out, ref_out, inp, ref_inp, _expected_device)


def _uncoalesced_csc(dtype):
    # Keep the repeated (0, 0) entry in storage order.
    shape = (3, 4)
    ccol = torch.tensor([0, 3, 3, 5, 5], dtype=torch.long, device=flag_gems.device)
    rows = torch.tensor([0, 0, 2, 1, 2], dtype=torch.long, device=flag_gems.device)
    values = _make_values(dtype, (5,), ["-1", "1"], torch.Generator("cpu"))
    return torch.sparse_csc_tensor(ccol, rows, values, shape)


@pytest.mark.ccol_indices_copy
@pytest.mark.parametrize("dtype", _CCOLS_INPUT_DTYPE_CASES)
def test_ccol_indices_copy_unchecked_uncoalesced(dtype):
    inp = _uncoalesced_csc(dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.ccol_indices_copy(ref_inp)
    _input_state = _capture_input(inp)
    _expected_device = inp.device
    res_out = flag_gems.ccol_indices_copy(inp)
    _assert_input_unchanged(inp, _input_state)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, _expected_device)


@pytest.mark.ccol_indices_copy_out
@pytest.mark.parametrize("dtype", _CCOLS_INPUT_DTYPE_CASES)
def test_ccol_indices_copy_out_unchecked_uncoalesced(dtype):
    inp = _uncoalesced_csc(dtype)
    ref_inp = tu.to_reference(inp)
    out = torch.full((5,), -1, dtype=torch.long, device=inp.device)
    ref_out = torch.full((5,), -1, dtype=torch.long, device=ref_inp.device)

    torch.ops.aten.ccol_indices_copy.out(ref_inp, out=ref_out)
    _input_state = _capture_input(inp)
    _expected_device = inp.device
    res_ret = flag_gems.ccol_indices_copy(inp, out=out)
    _assert_input_unchanged(inp, _input_state)

    assert res_ret is out
    _assert_copy_semantics(out, ref_out, inp, ref_inp, _expected_device)


def _special_csc(dtype, scenario):
    shape = (3, 4)
    ccol = torch.tensor([0, 2, 4, 6, 7], dtype=torch.long, device=flag_gems.device)
    rows = torch.tensor(
        [0, 1, 0, 2, 1, 2, 0], dtype=torch.long, device=flag_gems.device
    )
    values = _make_special_payload(dtype, scenario)
    return torch.sparse_csc_tensor(ccol, rows, values, shape)


@pytest.mark.ccol_indices_copy
@pytest.mark.parametrize(
    "dtype,scenario",
    [
        pytest.param(
            dtype,
            scenario,
            marks=pytest.mark.skipif(
                flag_gems.vendor_name == "ascend"
                and dtype in (torch.float8_e4m3fn, torch.float8_e5m2),
                reason="Ascend cannot construct or compare FP8 payloads with device kernels",
            ),
        )
        for dtype, scenario in tu.selected_cases(tu.special_value_cases(_CCOLS_DTYPES))
    ],
)
def test_ccol_indices_copy_nan_inf_values(dtype, scenario):
    inp = _special_csc(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.ccol_indices_copy(ref_inp)
    _input_state = _capture_input(inp)
    _expected_device = inp.device
    res_out = flag_gems.ccol_indices_copy(inp)
    _assert_input_unchanged(inp, _input_state)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, _expected_device)


@pytest.mark.ccol_indices_copy_out
@pytest.mark.parametrize(
    "dtype,scenario",
    [
        pytest.param(
            dtype,
            scenario,
            marks=pytest.mark.skipif(
                flag_gems.vendor_name == "ascend"
                and dtype in (torch.float8_e4m3fn, torch.float8_e5m2),
                reason="Ascend cannot construct or compare FP8 payloads with device kernels",
            ),
        )
        for dtype, scenario in tu.selected_cases(tu.special_value_cases(_CCOLS_DTYPES))
    ],
)
def test_ccol_indices_copy_out_nan_inf_values(dtype, scenario):
    inp = _special_csc(dtype, scenario)
    ref_inp = tu.to_reference(inp)
    out = torch.full((5,), -1, dtype=torch.long, device=inp.device)
    ref_out = torch.full((5,), -1, dtype=torch.long, device=ref_inp.device)

    torch.ops.aten.ccol_indices_copy.out(ref_inp, out=ref_out)
    _input_state = _capture_input(inp)
    _expected_device = inp.device
    res_ret = flag_gems.ccol_indices_copy(inp, out=out)
    _assert_input_unchanged(inp, _input_state)

    assert res_ret is out
    _assert_copy_semantics(out, ref_out, inp, ref_inp, _expected_device)


@pytest.mark.ccol_indices_copy
def test_ccol_indices_copy_negative_dense():
    inp = torch.randn(3, 4, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.ccol_indices_copy(tu.to_reference(inp))
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.ccol_indices_copy(inp)


@pytest.mark.ccol_indices_copy
def test_ccol_indices_copy_negative_csr():
    crow = torch.tensor([0, 2, 3], dtype=torch.long)
    cols = torch.tensor([0, 1, 2], dtype=torch.long)
    values = torch.randn(3, dtype=torch.float32)
    inp = torch.sparse_csr_tensor(crow, cols, values, (2, 3), device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.ccol_indices_copy(tu.to_reference(inp))
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.ccol_indices_copy(inp)


@pytest.mark.ccol_indices_copy
def test_ccol_indices_copy_negative_coo():
    indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    values = torch.randn(2, dtype=torch.float32)
    inp = torch.sparse_coo_tensor(indices, values, (3, 3), device=flag_gems.device)
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        torch.ops.aten.ccol_indices_copy(tu.to_reference(inp))
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        flag_gems.ccol_indices_copy(inp)


@pytest.mark.ccol_indices_copy_out
def test_ccol_indices_copy_out_negative_dense():
    inp = torch.randn(3, 4, dtype=torch.float32, device=flag_gems.device)
    out = torch.empty(5, dtype=torch.long, device=inp.device)
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.ccol_indices_copy.out(tu.to_reference(inp), out=out)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.ccol_indices_copy(inp, out=out)


@pytest.mark.ccol_indices_copy_out
def test_ccol_indices_copy_out_negative_csr():
    crow = torch.tensor([0, 2, 3], dtype=torch.long)
    cols = torch.tensor([0, 1, 2], dtype=torch.long)
    values = torch.randn(3, dtype=torch.float32)
    inp = torch.sparse_csr_tensor(crow, cols, values, (2, 3), device=flag_gems.device)
    out = torch.empty(5, dtype=torch.long, device=inp.device)
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.ccol_indices_copy.out(tu.to_reference(inp), out=out)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.ccol_indices_copy(inp, out=out)


@pytest.mark.ccol_indices_copy_out
def test_ccol_indices_copy_out_negative_coo():
    indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    values = torch.randn(2, dtype=torch.float32)
    inp = torch.sparse_coo_tensor(indices, values, (3, 3), device=flag_gems.device)
    out = torch.empty(5, dtype=torch.long, device=inp.device)
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        torch.ops.aten.ccol_indices_copy.out(tu.to_reference(inp), out=out)
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        flag_gems.ccol_indices_copy(inp, out=out)


@pytest.mark.ccol_indices_copy_out
def test_ccol_indices_copy_out_negative_wrong_dtype():
    inp = _make_input("csc", (5, 4), 6, None, torch.float32)
    ref_inp = tu.to_reference(inp)
    out = torch.empty(5, dtype=torch.float32, device=inp.device)
    ref_out = torch.empty(5, dtype=torch.float32, device=ref_inp.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten.ccol_indices_copy.out(ref_inp, out=ref_out)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.ccol_indices_copy(inp, out=out)


@pytest.mark.ccol_indices_copy
@pytest.mark.parametrize("shape,nnz", [((4, 6), 6), ((16, 32), 6), ((5, 5), 6), ((3, 1), 2)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES + utils.BOOL_TYPES)
def test_ccol_indices_copy_master_shapes(shape, nnz, dtype):
    inp = _make_input("csc", shape, nnz, None, dtype)
    reference = tu.to_reference(inp)
    expected = torch.ops.aten.ccol_indices_copy(reference)
    input_state = _capture_input(inp)

    actual = flag_gems.ccol_indices_copy(inp)

    _assert_input_unchanged(inp, input_state)
    _assert_copy_semantics(actual, expected, inp, reference, inp.device)


@pytest.mark.ccol_indices_copy
@pytest.mark.parametrize(
    "case",
    [
        ("bsc", (8, 12), 5, (2, 3)),
        ("bsc", (4, 4), 2, (2, 2)),
        ("bsc_batch", (2, 8, 12), 5, (2, 3)),
    ],
)
def test_ccol_indices_copy_master_compressed_layouts(case):
    inp = _make_input(*case, dtype=torch.float32)
    reference = tu.to_reference(inp)
    expected = torch.ops.aten.ccol_indices_copy(reference)
    input_state = _capture_input(inp)

    actual = flag_gems.ccol_indices_copy(inp)

    _assert_input_unchanged(inp, input_state)
    _assert_copy_semantics(actual, expected, inp, reference, inp.device)


@pytest.mark.ccol_indices_copy
def test_ccol_indices_copy_master_batched_csc():
    device = flag_gems.device
    # Distinct valid pointers detect accidentally copying batch zero twice.
    ccol = torch.tensor(
        [[0, 1, 2, 3, 3, 4, 6], [0, 0, 1, 1, 3, 4, 6]], device=device
    )
    rows = torch.tensor([[0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]], device=device)
    values = torch.randn((2, 6), device=device)
    inp = torch.sparse_csc_tensor(
        ccol, rows, values, size=(2, 4, 6), check_invariants=True
    )
    reference = tu.to_reference(inp)
    expected = torch.ops.aten.ccol_indices_copy(reference)
    input_state = _capture_input(inp)

    actual = flag_gems.ccol_indices_copy(inp)

    _assert_input_unchanged(inp, input_state)
    _assert_copy_semantics(actual, expected, inp, reference, inp.device)


@pytest.mark.ccol_indices_copy
def test_ccol_indices_copy_master_repeated_dispatch():
    inp = _make_input("csc", (8, 10), 12, None, torch.float32, seed=7)
    reference = tu.to_reference(inp)
    expected = torch.ops.aten.ccol_indices_copy(reference)
    input_state = _capture_input(inp)

    first = flag_gems.ccol_indices_copy(inp)
    _assert_copy_semantics(first, expected, inp, reference, inp.device)
    for _ in range(3):
        actual = flag_gems.ccol_indices_copy(inp)
        assert actual.data_ptr() != first.data_ptr()
        assert torch.ops.aten.dim(inp) == 2
        _assert_copy_semantics(actual, expected, inp, reference, inp.device)
    _assert_input_unchanged(inp, input_state)
