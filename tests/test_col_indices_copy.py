# Copyright 2026, The FlagOS Contributors.
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

# col_indices_copy returns an independent, contiguous copy of sparse indices.
# Prefill out buffers with -1 so missing writes cannot match valid indices.
# The dtype and case lists mirror the KernelGen acceptance suite so the
# reported speedup stays comparable to its H20 measurement.
_COLS_DTYPES = (
    [
        torch.int8,
        torch.uint8,
        torch.float32,
        torch.bfloat16,
        torch.float16,
        torch.int32,
        torch.int64,
    ]
    # float64 is gated on backend support (see utils.fp64_is_supported).
    + ([torch.float64] if utils.fp64_is_supported else [])
    + [torch.int16, torch.bool]
)
_VALUE_RANGE_DTYPES = [dtype for dtype in _COLS_DTYPES if dtype != torch.bool]

# (layout, size, nnz, block_shape), including batched and empty storage.
_COLS_CASES = [
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
]

_COLS_RANGE_CASES = [
    ("csr", (5, 4), 6, None),
    ("csr_batch", (2, 6, 8), 12, None),
    ("bsr", (4, 6), 4, (2, 2)),
]

# (layout, matrix_shape, values_shape, compressed_indices, plain_indices).
_INDEX_LAYOUT_CASES = [
    (torch.sparse_csr, (4, 6), (4,), [0, 2, 2, 3, 4], [0, 4, 1, 5]),
    (torch.sparse_bsr, (4, 6), (3, 2, 3), [0, 2, 3], [0, 1, 1]),
    (torch.sparse_csr, (4, 6), (0,), [0, 0, 0, 0, 0], []),
    (torch.sparse_bsr, (4, 6), (0, 2, 3), [0, 0, 0], []),
]
_INDEX_CASES = [
    (case, batch_shape, dense_shape)
    for case in _INDEX_LAYOUT_CASES
    for batch_shape in [(), (2,), (2, 3)]
    for dense_shape in [(), (2,)]
]


def _shape_to_csr_case(shape):
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


# The KernelGen spec's 0~5-dim shape sweep, mapped onto CSR matrices.
_SHAPE_CASES = [
    _shape_to_csr_case(shape)
    for shape in [(1,), (256,), (20, 320, 15), (16, 128, 64, 60)]
]

_VALUE_RANGES = [
    ["-1", "1"],
    ["0", "1"],
    ["-1", "0"],
    ["0", "max"],
    ["min", "0"],
]


def _dtype_bounds(dtype):
    """Return the (min, max) value bounds of ``dtype``."""
    if dtype == torch.bool:
        return 0, 1
    if dtype.is_complex:
        real = {
            torch.complex32: torch.float16,
            torch.complex64: torch.float32,
            torch.complex128: torch.float64,
        }[dtype]
        finfo = torch.finfo(real)
        return finfo.min, finfo.max
    if dtype.is_floating_point:
        finfo = torch.finfo(dtype)
        return finfo.min, finfo.max
    iinfo = torch.iinfo(dtype)
    return iinfo.min, iinfo.max


def _resolve_bound(symbol, dtype):
    """Resolve a range-bound symbol (-1 / 0 / 1 / max / min) for ``dtype``."""
    low, high = _dtype_bounds(dtype)
    table = {
        "-1": -1.0,
        "0": 0.0,
        "1": 1.0,
        "max": high,
        "min": low,
        "max/2": high / 2,
        "min/2": low / 2,
    }
    return table[symbol]


def _make_values(dtype, values_shape, value_range):
    """Generate stored values within ``value_range`` on the FlagGems device."""
    if dtype == torch.bool:
        return torch.randint(0, 2, values_shape, dtype=dtype, device=flag_gems.device)
    low = _resolve_bound(value_range[0], dtype)
    high = _resolve_bound(value_range[1], dtype)
    if not (dtype.is_floating_point or dtype.is_complex):
        low, high = int(low), int(high)
        lo_bound, hi_bound = _dtype_bounds(dtype)
        low = min(max(low, lo_bound), hi_bound)
        high = min(max(high, lo_bound), hi_bound)
        if low == high:
            return torch.full(values_shape, low, device=flag_gems.device, dtype=dtype)
    if low == high:
        return torch.full(values_shape, low, device=flag_gems.device, dtype=dtype)
    return torch.testing.make_tensor(
        values_shape,
        dtype=dtype,
        device=flag_gems.device,
        low=low,
        high=high,
    )


def _make_crow(n_compressed, nnz):
    counts = torch.full((n_compressed,), nnz // n_compressed, dtype=torch.long)
    counts[: nnz % n_compressed] += 1
    return torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])


def _make_crow_batch(n_batch, n_compressed, nnz):
    return _make_crow(n_compressed, nnz).expand(n_batch, -1).contiguous()


def _make_csr(size, nnz, dtype, gen, device, value_range):
    n_rows, n_cols = size
    assert 0 <= nnz <= n_rows * n_cols
    crow = _make_crow(n_rows, nnz)
    col = torch.arange(nnz) - torch.repeat_interleave(crow[:-1], crow.diff())
    values = _make_values(dtype, (nnz,), value_range)
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
    values = _make_values(dtype, (n_batch, nnz), value_range)
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
    values = _make_values(dtype, (nnz, block_rows, block_cols), value_range)
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
    values = _make_values(dtype, (n_batch, nnz, block_rows, block_cols), value_range)
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


def _expected_col_shape(case):
    layout, size, nnz, _ = case
    if layout in ("csr", "bsr"):
        return (nnz,)
    return size[:-2] + (nnz,)


def _assert_copy_semantics(res, ref, inp, ref_inp):
    assert res.is_contiguous()
    utils.gems_assert_equal(res, ref)
    # Empty storage has a null pointer.
    if res.numel() > 0:
        assert res.data_ptr() != inp.col_indices().data_ptr()
    utils.gems_assert_equal(inp, ref_inp, equal_nan=True)


def _make_index_layout(case, batch_shape, dense_shape, dtype, index_dtype):
    layout, matrix_shape, values_shape, compressed, plain = case
    compressed = torch.tensor(compressed, dtype=index_dtype, device=flag_gems.device)
    plain = torch.tensor(plain, dtype=index_dtype, device=flag_gems.device)
    compressed = compressed.expand(batch_shape + compressed.shape).contiguous()
    plain = plain.expand(batch_shape + plain.shape).contiguous()
    values = _make_values(dtype, batch_shape + values_shape + dense_shape, ["-1", "1"])
    inp = torch.sparse_compressed_tensor(
        compressed,
        plain,
        values,
        size=batch_shape + matrix_shape + dense_shape,
        layout=layout,
        check_invariants=True,
    )
    return inp


@pytest.mark.col_indices_copy
@pytest.mark.parametrize("case,batch_shape,dense_shape", _INDEX_CASES)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", _COLS_DTYPES)
def test_col_indices_copy_index_layouts(
    case, batch_shape, dense_shape, dtype, index_dtype
):
    inp = _make_index_layout(case, batch_shape, dense_shape, dtype, index_dtype)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.ops.aten.col_indices_copy(ref_inp)
    res_out = flag_gems.col_indices_copy(inp)
    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.col_indices_copy
@pytest.mark.parametrize("case,batch_shape,dense_shape", _INDEX_CASES)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", _COLS_DTYPES)
def test_col_indices_copy_out_index_layouts(
    case, batch_shape, dense_shape, dtype, index_dtype
):
    inp = _make_index_layout(case, batch_shape, dense_shape, dtype, index_dtype)
    ref_inp = utils.to_reference(inp)
    out = torch.full_like(inp.col_indices(), -1)
    ref_out = torch.full_like(ref_inp.col_indices(), -1)
    torch.ops.aten.col_indices_copy(ref_inp, out=ref_out)
    res_ret = flag_gems.col_indices_copy(inp, out=out)
    assert res_ret is out
    _assert_copy_semantics(out, ref_out, inp, ref_inp)


@pytest.mark.col_indices_copy
@pytest.mark.parametrize("case", _COLS_CASES)
@pytest.mark.parametrize("dtype", _COLS_DTYPES)
def test_col_indices_copy(case, dtype):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.col_indices_copy(ref_inp)
    res_out = flag_gems.col_indices_copy(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.col_indices_copy_out
@pytest.mark.parametrize("case", _COLS_CASES)
@pytest.mark.parametrize("dtype", _COLS_DTYPES)
def test_col_indices_copy_out(case, dtype):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype)
    ref_inp = utils.to_reference(inp)
    out = torch.full(_expected_col_shape(case), -1, dtype=torch.long, device=inp.device)
    ref_out = torch.full(
        _expected_col_shape(case), -1, dtype=torch.long, device=ref_inp.device
    )

    torch.ops.aten.col_indices_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems.col_indices_copy(inp, out=out)

    # The .out variant must write into and return the out tensor itself.
    assert res_ret is out
    _assert_copy_semantics(out, ref_out, inp, ref_inp)


@pytest.mark.col_indices_copy
@pytest.mark.parametrize("case", _SHAPE_CASES)
@pytest.mark.parametrize("dtype", _COLS_DTYPES)
def test_col_indices_copy_spec_shapes(case, dtype):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.col_indices_copy(ref_inp)
    res_out = flag_gems.col_indices_copy(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.col_indices_copy
@pytest.mark.parametrize("case", _SHAPE_CASES)
@pytest.mark.parametrize("dtype", _COLS_DTYPES)
def test_col_indices_copy_out_spec_shapes(case, dtype):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype)
    ref_inp = utils.to_reference(inp)
    out = torch.full(_expected_col_shape(case), -1, dtype=torch.long, device=inp.device)
    ref_out = torch.full(
        _expected_col_shape(case), -1, dtype=torch.long, device=ref_inp.device
    )

    torch.ops.aten.col_indices_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems.col_indices_copy(inp, out=out)

    assert res_ret is out
    _assert_copy_semantics(out, ref_out, inp, ref_inp)


@pytest.mark.col_indices_copy
@pytest.mark.parametrize("case", _COLS_RANGE_CASES)
@pytest.mark.parametrize("value_range", _VALUE_RANGES)
@pytest.mark.parametrize("dtype", _VALUE_RANGE_DTYPES)
def test_col_indices_copy_value_ranges(case, value_range, dtype):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype, value_range=value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.col_indices_copy(ref_inp)
    res_out = flag_gems.col_indices_copy(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.col_indices_copy
@pytest.mark.parametrize("case", _COLS_RANGE_CASES)
@pytest.mark.parametrize("value_range", _VALUE_RANGES)
@pytest.mark.parametrize("dtype", _VALUE_RANGE_DTYPES)
def test_col_indices_copy_out_value_ranges(case, value_range, dtype):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype, value_range=value_range)
    ref_inp = utils.to_reference(inp)
    out = torch.full(_expected_col_shape(case), -1, dtype=torch.long, device=inp.device)
    ref_out = torch.full(
        _expected_col_shape(case), -1, dtype=torch.long, device=ref_inp.device
    )

    torch.ops.aten.col_indices_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems.col_indices_copy(inp, out=out)

    assert res_ret is out
    _assert_copy_semantics(out, ref_out, inp, ref_inp)


@pytest.mark.col_indices_copy
@pytest.mark.parametrize("dtype", _COLS_DTYPES)
def test_col_indices_copy_empty_bsr(dtype):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    inp = _make_input("bsr", (4, 6), 0, (2, 2), dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.col_indices_copy(ref_inp)
    res_out = flag_gems.col_indices_copy(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.col_indices_copy
@pytest.mark.parametrize("dtype", _COLS_DTYPES)
def test_col_indices_copy_out_empty_bsr(dtype):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    inp = _make_input("bsr", (4, 6), 0, (2, 2), dtype)
    ref_inp = utils.to_reference(inp)
    out = torch.full((0,), -1, dtype=torch.long, device=inp.device)
    ref_out = torch.full((0,), -1, dtype=torch.long, device=ref_inp.device)

    torch.ops.aten.col_indices_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems.col_indices_copy(inp, out=out)

    assert res_ret is out
    _assert_copy_semantics(out, ref_out, inp, ref_inp)


def _uncoalesced_csr(dtype):
    # Keep the repeated (0, 0) entry in storage order.
    # Fixed shape for the uncoalesced-group check; indices repeat (0, 0).
    shape = (4, 3)
    crow = torch.tensor([0, 3, 3, 5, 5], dtype=torch.long, device=flag_gems.device)
    cols = torch.tensor([0, 0, 2, 1, 2], dtype=torch.long, device=flag_gems.device)
    values = _make_values(dtype, (5,), ["-1", "1"])
    return torch.sparse_csr_tensor(crow, cols, values, shape)


@pytest.mark.col_indices_copy
@pytest.mark.parametrize("dtype", _COLS_DTYPES)
def test_col_indices_copy_unchecked_uncoalesced(dtype):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    inp = _uncoalesced_csr(dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.col_indices_copy(ref_inp)
    res_out = flag_gems.col_indices_copy(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.col_indices_copy
@pytest.mark.parametrize("dtype", _COLS_DTYPES)
def test_col_indices_copy_out_unchecked_uncoalesced(dtype):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    inp = _uncoalesced_csr(dtype)
    ref_inp = utils.to_reference(inp)
    out = torch.full((5,), -1, dtype=torch.long, device=inp.device)
    ref_out = torch.full((5,), -1, dtype=torch.long, device=ref_inp.device)

    torch.ops.aten.col_indices_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems.col_indices_copy(inp, out=out)

    assert res_ret is out
    _assert_copy_semantics(out, ref_out, inp, ref_inp)


_SPECIAL_PAYLOADS = {
    "nan": [float("nan"), 0.0, -0.0, 1.0, -1.0],
    "inf": [float("inf"), float("-inf"), 0.0, -0.0, 1.0],
    "mixed": [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
}


def _special_value_cases(dtypes):
    """Representable special scenarios per floating dtype."""
    cases = []
    for dtype in dtypes:
        if not dtype.is_floating_point:
            continue
        cases.append((dtype, "nan"))
        cases.extend((dtype, kind) for kind in ("inf", "mixed"))
    return cases


def _special_csr(dtype, scenario):
    # Fixed shape for the special-value CSR check; crow/cols describe 7 stored
    # values with a repeated group so the accessor copy is exercised.
    shape = (3, 4)
    crow = torch.tensor([0, 2, 4, 7], dtype=torch.long, device=flag_gems.device)
    cols = torch.tensor(
        [0, 1, 0, 2, 0, 1, 2], dtype=torch.long, device=flag_gems.device
    )
    base = torch.tensor(
        _SPECIAL_PAYLOADS[scenario], device=flag_gems.device, dtype=torch.float32
    ).to(dtype)
    values = base.repeat(2)[:7]
    return torch.sparse_csr_tensor(crow, cols, values, shape)


@pytest.mark.col_indices_copy
@pytest.mark.parametrize("dtype,scenario", _special_value_cases(_COLS_DTYPES))
def test_col_indices_copy_nan_inf_values(dtype, scenario):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    inp = _special_csr(dtype, scenario)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.col_indices_copy(ref_inp)
    res_out = flag_gems.col_indices_copy(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.col_indices_copy
def test_col_indices_copy_negative_dense():
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    inp = torch.randn(3, 4, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.col_indices_copy(utils.to_reference(inp))
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.col_indices_copy(inp)


@pytest.mark.col_indices_copy
def test_col_indices_copy_negative_csc():
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    ccol_indices = torch.tensor([0, 2, 4], dtype=torch.long, device=flag_gems.device)
    row_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=flag_gems.device)
    values = torch.randn(4, device=flag_gems.device)
    inp = torch.sparse_csc_tensor(
        ccol_indices, row_indices, values, (4, 2), device=flag_gems.device
    )
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.col_indices_copy(utils.to_reference(inp))
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.col_indices_copy(inp)


@pytest.mark.col_indices_copy
def test_col_indices_copy_negative_coo():
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    values = torch.ones(2, dtype=torch.float32)
    inp = torch.sparse_coo_tensor(indices, values, (3, 3), device=flag_gems.device)
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        torch.ops.aten.col_indices_copy(utils.to_reference(inp))
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        flag_gems.col_indices_copy(inp)


@pytest.mark.col_indices_copy
def test_col_indices_copy_negative_non_tensor():
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    with pytest.raises(RuntimeError):
        torch.ops.aten.col_indices_copy(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.col_indices_copy(3.14)


@pytest.mark.col_indices_copy
def test_col_indices_copy_out_negative_dense():
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    inp = torch.randn(3, 4, device=flag_gems.device)
    out = torch.empty(5, dtype=torch.long, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.col_indices_copy.out(utils.to_reference(inp), out=out)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.col_indices_copy(inp, out=out)


@pytest.mark.col_indices_copy
def test_col_indices_copy_out_negative_coo():
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    values = torch.ones(2, dtype=torch.float32)
    inp = torch.sparse_coo_tensor(indices, values, (3, 3), device=flag_gems.device)
    out = torch.empty(5, dtype=torch.long, device=flag_gems.device)
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        torch.ops.aten.col_indices_copy.out(utils.to_reference(inp), out=out)
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        flag_gems.col_indices_copy(inp, out=out)


@pytest.mark.col_indices_copy
def test_col_indices_copy_out_negative_wrong_dtype():
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    inp = _make_input("csr", (5, 4), 6, None, torch.float32)
    ref_inp = utils.to_reference(inp)
    out = torch.empty(6, dtype=torch.float32, device=inp.device)
    ref_out = torch.empty(6, dtype=torch.float32, device=ref_inp.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten.col_indices_copy.out(ref_inp, out=ref_out)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.col_indices_copy(inp, out=out)
