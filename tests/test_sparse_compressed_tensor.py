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

# Build compressed sparse storage from index and value tensors.
# Pass dtype explicitly: these ATen factories default to float32.
# Accelerator inputs also need an explicit device.
_SPARSE_COMPRESSED_DTYPES = (
    [torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2]
    + utils.ALL_FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + [torch.bool]
)
_BLOCK_LAYOUTS = (torch.sparse_bsr, torch.sparse_bsc)
_BLOCK_SIZE = 2

# (layout, shape, nnz, index_dtype).
_LAYOUT_CASES = tu.selected_cases(
    [
        (torch.sparse_csr, (5, 4), 7, torch.int64),
        (torch.sparse_csc, (5, 4), 7, torch.int64),
        (torch.sparse_bsr, (6, 4), 6, torch.int64),
        (torch.sparse_bsc, (4, 6), 6, torch.int64),
        (torch.sparse_csr, (2, 3, 5, 4), 8, torch.int64),
        (torch.sparse_csr, (3, 5, 4), 7, torch.int32),
        (torch.sparse_bsr, (2, 4, 6, 4), 6, torch.int64),
        (torch.sparse_csc, (3, 4, 5, 6), 9, torch.int32),
    ],
    quick=[(torch.sparse_csr, (2, 19, 7), 12, torch.int64)],
)

# (layout, shape, nnz, index_dtype).
_VALUE_CASES = tu.selected_cases(
    [
        (torch.sparse_csr, (5, 4), 7, torch.int64),
        (torch.sparse_bsr, (6, 4), 6, torch.int64),
        (torch.sparse_csr, (2, 3, 5, 4), 8, torch.int64),
        (torch.sparse_bsc, (2, 4, 4, 6), 6, torch.int64),
    ],
    quick=[(torch.sparse_csr, (2, 19, 7), 12, torch.int64)],
)

# Layouts exercised by the size-inferred overload.
_NO_SIZE_CASES = [
    (torch.sparse_csr, (5, 4), 7, torch.int64),
    (torch.sparse_csc, (5, 4), 7, torch.int64),
    (torch.sparse_bsr, (6, 4), 6, torch.int64),
]

# Empty storage for each compressed layout.
_EMPTY_CASES = [
    (torch.sparse_csr, (4, 5)),
    (torch.sparse_csc, (4, 5)),
    (torch.sparse_bsr, (4, 4)),
    (torch.sparse_bsc, (4, 4)),
]


def _range_for_dtype(dtype, value_range):
    # Clamp a negative lower bound to zero for unsigned storage.
    low_symbol, high_symbol = value_range
    if dtype != torch.bool and not dtype.is_floating_point:
        if torch.iinfo(dtype).min == 0 and low_symbol.startswith("-"):
            low_symbol = "0"
    return [low_symbol, high_symbol]


def _make_input(layout, shape, nnz, dtype, index_dtype=torch.int64, value_range=None):
    # Build sorted unique index entries in each compressed segment.
    if value_range is None:
        value_range = ["-1", "1"]
    device = flag_gems.device
    batch = shape[:-2]
    nrows, ncols = shape[-2], shape[-1]
    if layout in _BLOCK_LAYOUTS:
        bs0 = bs1 = _BLOCK_SIZE
    else:
        bs0 = bs1 = 1
    nblocks0, nblocks1 = nrows // bs0, ncols // bs1
    if layout in (torch.sparse_csr, torch.sparse_bsr):
        comp_dim, plain_dim = nblocks0, nblocks1
    else:  # csc / bsc
        comp_dim, plain_dim = nblocks1, nblocks0
    entries = batch + (nnz,)
    assert 0 <= nnz <= comp_dim * plain_dim
    counts = torch.full((comp_dim,), nnz // comp_dim, dtype=torch.long)
    counts[: nnz % comp_dim] += 1
    compressed = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    plain = torch.arange(nnz) - torch.repeat_interleave(compressed[:-1], counts)
    compressed = compressed.expand(batch + (comp_dim + 1,)).contiguous()
    plain = plain.expand(entries).contiguous()
    block_shape = (bs0, bs1) if bs0 > 1 else ()
    values = tu.make_input(
        dtype, entries + block_shape, _range_for_dtype(dtype, value_range)
    )
    return (
        compressed.to(device=device, dtype=index_dtype),
        plain.to(device=device, dtype=index_dtype),
        values.to(device),
    )


def _assert_result(res_out, ref_out, dtype, layout, index_dtype):
    assert res_out.layout == layout
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == dtype
    assert res_out.device.type == flag_gems.device
    assert torch.ops.aten._nnz(res_out) == torch.ops.aten._nnz(ref_out)

    if layout in (torch.sparse_csr, torch.sparse_bsr):
        res_c = torch.ops.aten.crow_indices(res_out)
        ref_c = torch.ops.aten.crow_indices(ref_out)
    else:
        res_c = torch.ops.aten.ccol_indices(res_out)
        ref_c = torch.ops.aten.ccol_indices(ref_out)
    if layout in (torch.sparse_csr, torch.sparse_bsr):
        res_p = torch.ops.aten.col_indices(res_out)
        ref_p = torch.ops.aten.col_indices(ref_out)
    else:
        res_p = torch.ops.aten.row_indices(res_out)
        ref_p = torch.ops.aten.row_indices(ref_out)
    assert res_c.dtype == index_dtype
    assert res_p.dtype == index_dtype
    # Indices are exact integer data.
    utils.gems_assert_equal(res_c, ref_c)
    utils.gems_assert_equal(res_p, ref_p)
    tu.assert_result_equal(res_out.values(), ref_out.values())


@pytest.mark.sparse_compressed_tensor
@pytest.mark.parametrize(
    "shape", [shape for shape in tu.selected_shapes() if len(shape) >= 2]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.float8_e4m3fn])
def test_sparse_compressed_tensor_shape_levels(shape, dtype):
    layout = torch.sparse_csr
    nnz = 8
    compressed, plain, values = _make_input(layout, shape, nnz, dtype)
    ref_compressed = tu.to_reference(compressed)
    ref_plain = tu.to_reference(plain)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten.sparse_compressed_tensor(
        ref_compressed,
        ref_plain,
        ref_values,
        list(shape),
        dtype=dtype,
        layout=layout,
        device=ref_compressed.device,
    )
    res_out = flag_gems.sparse_compressed_tensor(
        compressed,
        plain,
        values,
        list(shape),
        dtype=dtype,
        layout=layout,
        device=compressed.device,
    )

    assert res_out.shape == ref_out.shape == tuple(shape)
    _assert_result(res_out, ref_out, dtype, layout, torch.int64)


@pytest.mark.sparse_compressed_tensor
@pytest.mark.parametrize("case", _LAYOUT_CASES)
@pytest.mark.parametrize("dtype", _SPARSE_COMPRESSED_DTYPES)
def test_sparse_compressed_tensor(case, dtype):
    layout, shape, nnz, index_dtype = case
    compressed, plain, values = _make_input(layout, shape, nnz, dtype, index_dtype)
    ref_compressed = tu.to_reference(compressed)
    ref_plain = tu.to_reference(plain)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten.sparse_compressed_tensor(
        ref_compressed,
        ref_plain,
        ref_values,
        list(shape),
        dtype=dtype,
        layout=layout,
        device=ref_compressed.device,
    )
    res_out = flag_gems.sparse_compressed_tensor(
        compressed,
        plain,
        values,
        list(shape),
        dtype=dtype,
        layout=layout,
        device=compressed.device,
    )

    _assert_result(res_out, ref_out, dtype, layout, index_dtype)


@pytest.mark.sparse_compressed_tensor
@pytest.mark.parametrize("case", _NO_SIZE_CASES)
@pytest.mark.parametrize("dtype", _SPARSE_COMPRESSED_DTYPES)
def test_sparse_compressed_tensor_no_size(case, dtype):
    layout, shape, nnz, index_dtype = case
    compressed, plain, values = _make_input(layout, shape, nnz, dtype, index_dtype)
    ref_compressed = tu.to_reference(compressed)
    ref_plain = tu.to_reference(plain)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten.sparse_compressed_tensor(
        ref_compressed,
        ref_plain,
        ref_values,
        dtype=dtype,
        layout=layout,
        device=ref_compressed.device,
    )
    res_out = flag_gems.sparse_compressed_tensor(
        compressed,
        plain,
        values,
        dtype=dtype,
        layout=layout,
        device=compressed.device,
    )

    _assert_result(res_out, ref_out, dtype, layout, index_dtype)


@pytest.mark.sparse_compressed_tensor
@pytest.mark.parametrize("case", _EMPTY_CASES)
@pytest.mark.parametrize("dtype", _SPARSE_COMPRESSED_DTYPES)
def test_sparse_compressed_tensor_empty(case, dtype):
    layout, shape = case
    compressed, plain, values = _make_input(layout, shape, 0, dtype)
    ref_compressed = tu.to_reference(compressed)
    ref_plain = tu.to_reference(plain)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten.sparse_compressed_tensor(
        ref_compressed,
        ref_plain,
        ref_values,
        list(shape),
        dtype=dtype,
        layout=layout,
        device=ref_compressed.device,
    )
    res_out = flag_gems.sparse_compressed_tensor(
        compressed,
        plain,
        values,
        list(shape),
        dtype=dtype,
        layout=layout,
        device=compressed.device,
    )

    _assert_result(res_out, ref_out, dtype, layout, torch.int64)


@pytest.mark.sparse_compressed_tensor
@pytest.mark.parametrize("case", _VALUE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _SPARSE_COMPRESSED_DTYPES)
def test_sparse_compressed_tensor_value_ranges(case, value_range, dtype):
    layout, shape, nnz, index_dtype = case
    compressed, plain, values = _make_input(
        layout, shape, nnz, dtype, index_dtype, value_range
    )
    ref_compressed = tu.to_reference(compressed)
    ref_plain = tu.to_reference(plain)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten.sparse_compressed_tensor(
        ref_compressed,
        ref_plain,
        ref_values,
        list(shape),
        dtype=dtype,
        layout=layout,
        device=ref_compressed.device,
    )
    res_out = flag_gems.sparse_compressed_tensor(
        compressed,
        plain,
        values,
        list(shape),
        dtype=dtype,
        layout=layout,
        device=compressed.device,
    )

    _assert_result(res_out, ref_out, dtype, layout, index_dtype)


@pytest.mark.sparse_compressed_tensor
@pytest.mark.parametrize(
    "dtype,scenario",
    tu.selected_cases(tu.special_value_cases(_SPARSE_COMPRESSED_DTYPES)),
)
def test_sparse_compressed_tensor_nan_inf_values(dtype, scenario):
    layout = torch.sparse_csr
    shape = (3, 4)
    crow_t = torch.tensor([0, 2, 4, 7], dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(
        [0, 1, 0, 2, 0, 1, 2], dtype=torch.long, device=flag_gems.device
    )
    values = tu.make_special_input(dtype, scenario).repeat(2)[:7]
    ref_crow = tu.to_reference(crow_t)
    ref_col = tu.to_reference(col_t)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten.sparse_compressed_tensor(
        ref_crow,
        ref_col,
        ref_values,
        list(shape),
        dtype=dtype,
        layout=layout,
        device=ref_crow.device,
    )
    res_out = flag_gems.sparse_compressed_tensor(
        crow_t,
        col_t,
        values,
        list(shape),
        dtype=dtype,
        layout=layout,
        device=crow_t.device,
    )

    assert res_out.layout == layout
    assert res_out.shape == ref_out.shape == shape
    assert res_out.dtype == ref_out.dtype == dtype
    assert torch.ops.aten._nnz(res_out) == torch.ops.aten._nnz(ref_out) == 7
    utils.gems_assert_equal(res_out.crow_indices(), ref_out.crow_indices())
    utils.gems_assert_equal(res_out.col_indices(), ref_out.col_indices())
    tu.assert_result_equal(res_out.values(), ref_out.values())


@pytest.mark.sparse_compressed_tensor
@pytest.mark.parametrize("dtype", tu.selected_cases(utils.ALL_FLOAT_DTYPES))
def test_sparse_compressed_tensor_backward(dtype):
    layout, shape, nnz = torch.sparse_csr, (5, 4), 7
    compressed, plain, values = _make_input(layout, shape, nnz, dtype)
    ref_compressed = tu.to_reference(compressed)
    ref_plain = tu.to_reference(plain)
    values.requires_grad_(True)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten.sparse_compressed_tensor(
        ref_compressed,
        ref_plain,
        ref_values,
        list(shape),
        dtype=dtype,
        layout=layout,
        device=ref_compressed.device,
    )
    res_out = flag_gems.sparse_compressed_tensor(
        compressed,
        plain,
        values,
        list(shape),
        dtype=dtype,
        layout=layout,
        device=compressed.device,
    )

    _assert_result(res_out, ref_out, dtype, layout, torch.int64)

    weights = torch.linspace(-1.0, 1.0, 7, dtype=dtype, device=flag_gems.device)
    ref_weights = tu.to_reference(weights)
    grad_ref = torch.autograd.grad((ref_out.values() * ref_weights).sum(), ref_values)[
        0
    ]
    grad_res = torch.autograd.grad((res_out.values() * weights).sum(), values)[0]
    utils.gems_assert_close(grad_res, grad_ref, dtype)


def _negative_csr_inputs(dtype=torch.float32):
    layout, shape, nnz = torch.sparse_csr, (5, 4), 7
    compressed, plain, values = _make_input(layout, shape, nnz, dtype)
    return compressed, plain, values, shape


@pytest.mark.sparse_compressed_tensor
def test_sparse_compressed_tensor_rejects_missing_layout():
    compressed, plain, values, shape = _negative_csr_inputs()
    ref_compressed = tu.to_reference(compressed)
    ref_plain = tu.to_reference(plain)
    ref_values = tu.to_reference(values)

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_compressed_tensor(
            ref_compressed,
            ref_plain,
            ref_values,
            list(shape),
            dtype=torch.float32,
            device=ref_compressed.device,
        )
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.sparse_compressed_tensor(
            compressed,
            plain,
            values,
            list(shape),
            dtype=torch.float32,
            device=compressed.device,
        )


@pytest.mark.sparse_compressed_tensor
def test_sparse_compressed_tensor_rejects_coo_layout():
    compressed, plain, values, shape = _negative_csr_inputs()
    ref_compressed = tu.to_reference(compressed)
    ref_plain = tu.to_reference(plain)
    ref_values = tu.to_reference(values)

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_compressed_tensor(
            ref_compressed,
            ref_plain,
            ref_values,
            list(shape),
            dtype=torch.float32,
            layout=torch.sparse_coo,
            device=ref_compressed.device,
        )
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.sparse_compressed_tensor(
            compressed,
            plain,
            values,
            list(shape),
            dtype=torch.float32,
            layout=torch.sparse_coo,
            device=compressed.device,
        )


@pytest.mark.sparse_compressed_tensor
def test_sparse_compressed_tensor_rejects_dtype_mismatch():
    compressed, plain, values, shape = _negative_csr_inputs()
    ref_compressed = tu.to_reference(compressed)
    ref_plain = tu.to_reference(plain)
    ref_values = tu.to_reference(values)

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_compressed_tensor(
            ref_compressed,
            ref_plain,
            ref_values,
            list(shape),
            dtype=torch.float64,
            layout=torch.sparse_csr,
            device=ref_compressed.device,
        )
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.sparse_compressed_tensor(
            compressed,
            plain,
            values,
            list(shape),
            dtype=torch.float64,
            layout=torch.sparse_csr,
            device=compressed.device,
        )


@pytest.mark.sparse_compressed_tensor
def test_sparse_compressed_tensor_rejects_non_float32_values_without_dtype():
    compressed, plain, values, shape = _negative_csr_inputs(torch.bfloat16)
    ref_compressed = tu.to_reference(compressed)
    ref_plain = tu.to_reference(plain)
    ref_values = tu.to_reference(values)

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_compressed_tensor(
            ref_compressed,
            ref_plain,
            ref_values,
            list(shape),
            layout=torch.sparse_csr,
            device=ref_compressed.device,
        )
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.sparse_compressed_tensor(
            compressed,
            plain,
            values,
            list(shape),
            layout=torch.sparse_csr,
            device=compressed.device,
        )
