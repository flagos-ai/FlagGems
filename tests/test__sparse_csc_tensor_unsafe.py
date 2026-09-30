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

from . import test_utils as tu

# _sparse_csc_tensor_unsafe is the unchecked CSC constructor: it validates
# neither the metadata nor the buffer lengths and stores the caller's three
# tensors by alias. The tests assert the reported metadata (layout, size,
# sparse_dim, dense_dim, nnz, component shape/dtype) and the alias contract,
# and compare the stored index/value buffers through the shared helpers.
# Malformed metadata is compared verbatim and never densified. The constructor
# has no broadcast operands and no kernel backward, so those dimensions do not
# apply here.

_INDEX_DTYPES = [torch.int32, torch.int64]

_VALUE_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
]

_SPEC_SHAPES = [
    (),
    (1,),
    (256,),
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]
_QUICK_SPEC_SHAPES = [(2, 19, 7)]

_SPECIAL_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
]


def _matrix_size(spec_shape):
    # CSC needs the trailing (rows, cols) axes, so the spec's rank-0 and rank-1
    # shapes widen to the smallest valid matrix. The bare rank-0/rank-1 `size`
    # lists with empty components are exercised by the unchecked cases below.
    if len(spec_shape) >= 2:
        return [int(dim) for dim in spec_shape]
    if len(spec_shape) == 1:
        return [max(int(spec_shape[0]), 1), 4]
    return [1, 1]


def _csc_indices(size, per_column, index_dtype=torch.int64, device=None):
    # Compressed column pointers plus sorted, duplicate-free row indices.
    device = flag_gems.device if device is None else device
    batch, rows, cols = list(size[:-2]), int(size[-2]), int(size[-1])
    per = max(0, min(int(per_column), rows))
    ccol = torch.arange(cols + 1, dtype=index_dtype, device=device) * per
    row = torch.arange(per, dtype=index_dtype, device=device).repeat(cols)
    nnz = per * cols
    if batch:
        ccol = ccol.expand(*batch, cols + 1).contiguous()
        row = row.expand(*batch, nnz).contiguous()
    return ccol, row, batch + [nnz]


def _assert_metadata(res_out, ref_out, size, values_device):
    assert res_out.layout == torch.sparse_csc
    assert res_out.dtype == ref_out.dtype
    assert tuple(res_out.shape) == tuple(ref_out.shape) == tuple(size)
    assert res_out.sparse_dim() == ref_out.sparse_dim()
    assert res_out.dense_dim() == ref_out.dense_dim()
    assert torch.ops.aten._nnz(res_out) == torch.ops.aten._nnz(ref_out)
    # The output device follows the requested/input device, not the reference's.
    assert res_out.device == values_device


def _assert_components(res_out, ref_out, index_dtype):
    # Sparse CSC tensors expose no stride() (native raises 'Sparse CSC tensors
    # do not have strides'); the aliased component views are compared instead.
    assert res_out.ccol_indices().dtype == index_dtype
    assert res_out.row_indices().dtype == index_dtype
    assert res_out.ccol_indices().shape == ref_out.ccol_indices().shape
    assert res_out.row_indices().shape == ref_out.row_indices().shape
    assert res_out.values().shape == ref_out.values().shape
    assert res_out.values().dtype == ref_out.values().dtype
    tu.assert_result_equal(res_out.ccol_indices(), ref_out.ccol_indices())
    tu.assert_result_equal(res_out.row_indices(), ref_out.row_indices())
    tu.assert_result_equal(res_out.values(), ref_out.values())


@pytest.mark.sparse_csc_tensor_unsafe
@pytest.mark.parametrize(
    "spec_shape", tu.selected_cases(_SPEC_SHAPES, quick=_QUICK_SPEC_SHAPES)
)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _VALUE_DTYPES)
def test__sparse_csc_tensor_unsafe_values(spec_shape, value_range, dtype):
    size = _matrix_size(spec_shape)
    ccol, row, values_shape = _csc_indices(size, 1)
    values = tu.make_input(dtype, values_shape, value_range)

    ref_ccol = tu.to_reference(ccol)
    ref_row = tu.to_reference(row)
    ref_values = tu.to_reference(values)
    ref_out = torch.ops.aten._sparse_csc_tensor_unsafe(
        ref_ccol,
        ref_row,
        ref_values,
        list(size),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=ref_ccol.device,
    )
    res_out = flag_gems._sparse_csc_tensor_unsafe(
        ccol,
        row,
        values,
        list(size),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )

    _assert_metadata(res_out, ref_out, size, values.device)
    _assert_components(res_out, ref_out, torch.int64)
    # The constructor stores the buffers it is given; they must stay unchanged.
    tu.assert_result_equal(ccol, ref_ccol)
    tu.assert_result_equal(row, ref_row)
    tu.assert_result_equal(values, ref_values)


# (size, entries per column): the second value shapes the compressed structure
# and the dense prefixes exercise the batched sizes.
_STRUCT_CASES = [
    ([3, 2], 1),
    ([5, 7], 3),
    ([1, 256], 1),
    ([2, 3, 4], 1),
    ([2, 2, 3, 4], 2),
    ([16, 7, 57, 32, 29], 1),
]
# Quick keeps every cheap branch (both index dtypes, single-column and batched
# sizes, multi-entry columns); only the large 5-D size stays default-only.
_QUICK_STRUCT_CASES = [
    ([3, 2], 1),
    ([5, 7], 3),
    ([1, 256], 1),
    ([2, 3, 4], 1),
    ([2, 2, 3, 4], 2),
]


@pytest.mark.sparse_csc_tensor_unsafe
@pytest.mark.parametrize(
    "size, per_column", tu.selected_cases(_STRUCT_CASES, quick=_QUICK_STRUCT_CASES)
)
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test__sparse_csc_tensor_unsafe_structure(size, per_column, dtype, index_dtype):
    ccol, row, values_shape = _csc_indices(size, per_column, index_dtype=index_dtype)
    values = tu.make_input(dtype, values_shape, ["-1", "1"])

    ref_ccol = tu.to_reference(ccol)
    ref_row = tu.to_reference(row)
    ref_values = tu.to_reference(values)
    ref_out = torch.ops.aten._sparse_csc_tensor_unsafe(
        ref_ccol,
        ref_row,
        ref_values,
        list(size),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=ref_ccol.device,
    )
    res_out = flag_gems._sparse_csc_tensor_unsafe(
        ccol,
        row,
        values,
        list(size),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )

    _assert_metadata(res_out, ref_out, size, values.device)
    _assert_components(res_out, ref_out, index_dtype)


# Native-accepted malformed metadata (probed on CPU and CUDA): the unchecked
# constructor reports exactly what it is given. A rank-0 ccol or a 2-D values
# tensor raises dense_dim() to 1, and the buffer lengths are not validated.
# A rank-0/rank-1 size keeps the requested shape with empty components.
# (case, size, ccol, row, values shape)
_UNCHECKED_CASES = [
    ("values-longer-than-nnz", [3, 2], [0, 1, 2], [0, 1], [5]),
    ("values-shorter-than-nnz", [3, 2], [0, 1, 2], [0, 1], [1]),
    ("duplicate-rows", [2, 2], [0, 1, 3], [0, 0, 0], [3]),
    ("unsorted-rows", [3, 1], [0, 3, 3], [1, 0, 2], [3]),
    ("ccol-past-nnz", [3, 2], [0, 1, 9], [0, 1], [2]),
    ("short-ccol", [3, 2], [0, 1], [0, 1], [2]),
    ("rank0-ccol", [3, 2], 2, [0, 1], [2]),
    ("size-rank-mismatch", [3, 3, 3], [0, 1, 2], [0, 1], [2]),
    ("size-rank0", [], [0, 1, 2], [0, 1], [2]),
    ("zero-columns", [3, 0], [0, 1, 2], [0, 1], [2]),
    ("zero-rows", [0, 0], [0, 1, 2], [0, 1], [2]),
    ("dense-tail", [3, 2], [0, 2], [0, 1], [2, 1]),
    ("rank1-empty-size", [1073741824], [], [], [0]),
    ("rank0-empty-size", [], [], [], [0]),
]


@pytest.mark.sparse_csc_tensor_unsafe
@pytest.mark.parametrize(
    "case, size, ccol_spec, row_spec, values_shape", _UNCHECKED_CASES
)
def test__sparse_csc_tensor_unsafe_unchecked_metadata(
    case, size, ccol_spec, row_spec, values_shape
):
    ccol = torch.tensor(ccol_spec, dtype=torch.int64, device=flag_gems.device)
    row = torch.tensor(row_spec, dtype=torch.int64, device=flag_gems.device)
    values = tu.make_input(torch.float32, values_shape, ["-1", "1"])

    ref_out = torch.ops.aten._sparse_csc_tensor_unsafe(
        tu.to_reference(ccol),
        tu.to_reference(row),
        tu.to_reference(values),
        list(size),
        dtype=torch.float32,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )
    res_out = flag_gems._sparse_csc_tensor_unsafe(
        ccol,
        row,
        values,
        list(size),
        dtype=torch.float32,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )

    _assert_metadata(res_out, ref_out, size, values.device)
    _assert_components(res_out, ref_out, torch.int64)


@pytest.mark.sparse_csc_tensor_unsafe
def test__sparse_csc_tensor_unsafe_aliases_component_storage():
    # Strided views with non-zero storage offsets: the constructor keeps the
    # buffers themselves, so the components must expose the same storage, shape,
    # stride and offset instead of a normalized copy.
    ccol_base = torch.zeros(9, dtype=torch.int64, device=flag_gems.device)
    ccol_base[::2] = torch.arange(5, dtype=torch.int64, device=flag_gems.device)
    ccol = ccol_base[::2]
    row_base = torch.zeros(9, dtype=torch.int64, device=flag_gems.device)
    row_base[1:9:2] = torch.arange(4, dtype=torch.int64, device=flag_gems.device)
    row = row_base[1:9:2]
    values_base = torch.zeros(9, dtype=torch.float32, device=flag_gems.device)
    values_base[1:9:2] = torch.arange(
        1, 5, dtype=torch.float32, device=flag_gems.device
    )
    values = values_base[1:9:2]
    size = [6, 4]

    ref_out = torch.ops.aten._sparse_csc_tensor_unsafe(
        tu.to_reference(ccol),
        tu.to_reference(row),
        tu.to_reference(values),
        list(size),
        dtype=torch.float32,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )
    res_out = flag_gems._sparse_csc_tensor_unsafe(
        ccol,
        row,
        values,
        list(size),
        dtype=torch.float32,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )

    _assert_metadata(res_out, ref_out, size, values.device)
    _assert_components(res_out, ref_out, torch.int64)

    for produced, source in (
        (res_out.ccol_indices(), ccol),
        (res_out.row_indices(), row),
        (res_out.values(), values),
    ):
        assert (
            produced.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
        )
        assert produced.data_ptr() == source.data_ptr()
        assert produced.shape == source.shape
        assert produced.stride() == source.stride()
        assert produced.storage_offset() == source.storage_offset()

    res_out.values().fill_(7.5)
    assert torch.equal(values, torch.full_like(values, 7.5))


# The schema defaults are dtype=float32, layout=sparse_csc, device=cpu and
# pin_memory; the default device is CPU, so the omitted-argument forms use CPU
# buffers with the reference on the same CPU contract.
_OPTION_CASES = [
    (
        "explicit-kwargs",
        {"dtype": torch.float32, "layout": torch.sparse_csc, "device": "cpu"},
    ),
    ("omit-layout", {"dtype": torch.float32, "device": "cpu"}),
    ("omit-device", {"dtype": torch.float32, "layout": torch.sparse_csc}),
    ("omit-dtype", {"layout": torch.sparse_csc, "device": "cpu"}),
    (
        "pin-memory-false",
        {
            "dtype": torch.float32,
            "layout": torch.sparse_csc,
            "device": "cpu",
            "pin_memory": False,
        },
    ),
    (
        "pin-memory-true",
        {
            "dtype": torch.float32,
            "layout": torch.sparse_csc,
            "device": "cpu",
            "pin_memory": True,
        },
    ),
]


@pytest.mark.sparse_csc_tensor_unsafe
@pytest.mark.parametrize("case, kwargs", _OPTION_CASES)
def test__sparse_csc_tensor_unsafe_optional_kwargs(case, kwargs):
    ccol = torch.tensor([0, 1, 2], dtype=torch.int64, device="cpu")
    row = torch.tensor([0, 1], dtype=torch.int64, device="cpu")
    values = torch.tensor([1.5, -2.25], dtype=torch.float32, device="cpu")
    size = [3, 2]

    ref_out = torch.ops.aten._sparse_csc_tensor_unsafe(
        tu.to_reference(ccol),
        tu.to_reference(row),
        tu.to_reference(values),
        list(size),
        **kwargs,
    )
    res_out = flag_gems._sparse_csc_tensor_unsafe(
        ccol, row, values, list(size), **kwargs
    )

    _assert_metadata(res_out, ref_out, size, values.device)
    _assert_components(res_out, ref_out, torch.int64)
    assert res_out.values().is_pinned() == ref_out.values().is_pinned()


@pytest.mark.sparse_csc_tensor_unsafe
@pytest.mark.parametrize(
    "dtype, scenario",
    tu.selected_cases(tu.special_value_cases(_SPECIAL_DTYPES), quick=[]),
)
def test__sparse_csc_tensor_unsafe_special_values(dtype, scenario):
    values = tu.make_special_input(dtype, scenario)
    ccol = torch.tensor([0, 2, 5], dtype=torch.int64, device=flag_gems.device)
    row = torch.tensor([0, 1, 0, 1, 2], dtype=torch.int64, device=flag_gems.device)
    size = [3, 2]

    ref_out = torch.ops.aten._sparse_csc_tensor_unsafe(
        tu.to_reference(ccol),
        tu.to_reference(row),
        tu.to_reference(values),
        list(size),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )
    res_out = flag_gems._sparse_csc_tensor_unsafe(
        ccol,
        row,
        values,
        list(size),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )

    _assert_metadata(res_out, ref_out, size, values.device)
    _assert_components(res_out, ref_out, torch.int64)


def _basic_buffers(dtype=torch.float32):
    device = flag_gems.device
    ccol = torch.tensor([0, 1, 2], dtype=torch.int64, device=device)
    row = torch.tensor([0, 1], dtype=torch.int64, device=device)
    values = torch.tensor([1.5, -2.25], dtype=dtype, device=device)
    return ccol, row, values


@pytest.mark.sparse_csc_tensor_unsafe
@pytest.mark.parametrize(
    "values_dtype, declared_dtype",
    [(torch.float16, torch.float32), (torch.float64, torch.float32)],
)
def test__sparse_csc_tensor_unsafe_rejects_dtype_mismatch(values_dtype, declared_dtype):
    ccol, row, values = _basic_buffers(values_dtype)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._sparse_csc_tensor_unsafe(
            ccol,
            row,
            values,
            [3, 2],
            dtype=declared_dtype,
            layout=torch.sparse_csc,
            device=flag_gems.device,
        )


@pytest.mark.sparse_csc_tensor_unsafe
def test__sparse_csc_tensor_unsafe_rejects_missing_dtype_for_other_values():
    ccol, row, values = _basic_buffers(torch.float64)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._sparse_csc_tensor_unsafe(
            ccol,
            row,
            values,
            [3, 2],
            layout=torch.sparse_csc,
            device=flag_gems.device,
        )


@pytest.mark.sparse_csc_tensor_unsafe
@pytest.mark.parametrize("layout", [torch.sparse_csr, torch.sparse_coo])
def test__sparse_csc_tensor_unsafe_rejects_foreign_layout(layout):
    ccol, row, values = _basic_buffers()
    with pytest.raises((RuntimeError, TypeError, ValueError, NotImplementedError)):
        flag_gems._sparse_csc_tensor_unsafe(
            ccol,
            row,
            values,
            [3, 2],
            dtype=torch.float32,
            layout=layout,
            device=flag_gems.device,
        )


@pytest.mark.sparse_csc_tensor_unsafe
def test__sparse_csc_tensor_unsafe_rejects_non_list_size():
    ccol, row, values = _basic_buffers()
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems._sparse_csc_tensor_unsafe(
            ccol,
            row,
            values,
            3,
            dtype=torch.float32,
            layout=torch.sparse_csc,
            device=flag_gems.device,
        )


@pytest.mark.sparse_csc_tensor_unsafe
def test__sparse_csc_tensor_unsafe_rejects_unknown_device():
    ccol, row, values = _basic_buffers()
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems._sparse_csc_tensor_unsafe(
            ccol,
            row,
            values,
            [3, 2],
            dtype=torch.float32,
            layout=torch.sparse_csc,
            device="not-a-device",
        )


@pytest.mark.sparse_csc_tensor_unsafe
@pytest.mark.skipif(flag_gems.device == "cpu", reason="needs a non-CPU device")
def test__sparse_csc_tensor_unsafe_rejects_missing_device_on_accelerator():
    ccol, row, values = _basic_buffers()
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._sparse_csc_tensor_unsafe(
            ccol,
            row,
            values,
            [3, 2],
            dtype=torch.float32,
            layout=torch.sparse_csc,
        )


@pytest.mark.sparse_csc_tensor_unsafe
@pytest.mark.skipif(flag_gems.device == "cpu", reason="needs a non-CPU device")
def test__sparse_csc_tensor_unsafe_rejects_cross_device_buffers():
    ccol, row, values = _basic_buffers()
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._sparse_csc_tensor_unsafe(
            ccol,
            row,
            values.to("cpu"),
            [3, 2],
            dtype=torch.float32,
            layout=torch.sparse_csc,
            device=flag_gems.device,
        )


@pytest.mark.sparse_csc_tensor_unsafe
@pytest.mark.skipif(flag_gems.device == "cpu", reason="needs a non-CPU device")
def test__sparse_csc_tensor_unsafe_rejects_pinned_accelerator_buffers():
    ccol, row, values = _basic_buffers()
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._sparse_csc_tensor_unsafe(
            ccol,
            row,
            values,
            [3, 2],
            dtype=torch.float32,
            layout=torch.sparse_csc,
            device=flag_gems.device,
            pin_memory=True,
        )
