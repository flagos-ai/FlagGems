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

# Dtypes this constructor accepts with int64 indices and a matching `dtype`
# argument (probing the real signature rejects the others).
SUPPORTED_DTYPES = [
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
    torch.int16,
    torch.bool,
]

# The shared matrix already drops e4m3fn's inf scenarios (that dtype has no
# infinity); its nan scenario, and nan/inf/mixed for every other floating dtype,
# still apply.
_SPECIAL_VALUE_CASES = tu.selected_cases(
    tu.special_value_cases(
        [dtype for dtype in SUPPORTED_DTYPES if dtype.is_floating_point]
    ),
    quick=[],
)

# This constructor takes no tensor-shape argument: the result's shape is the
# (size, sparse_dim, dense_dim) triple and nnz counts the stored entries, so each
# spec shape is carried by one row that keeps its rank. quick keeps every cheap
# branch (0-dim, singleton, empty nnz, batch, dense tail, 3-dim) and drops only
# the large rows. Broadcast and backward do not apply: there is no elementwise
# pair, and the stored members of the result are the caller's own leaves, so
# autograd reports "element 0 of tensors does not require grad" instead of a
# grad_fn.
_QUICK_SIZE_ROWS = [
    pytest.param(((), 0, 0, 0), id="zero-dim"),
    pytest.param(((1,), 1, 0, 1), id="singleton"),
    pytest.param(((256,), 1, 0, 8), id="1d"),
    pytest.param(((4, 3), 2, 0, 0), id="empty-nnz-batch"),
    pytest.param(((2, 3, 4), 2, 1, 2), id="dense-tail"),
    pytest.param(((2, 19, 7), 3, 0, 5), id="3d"),
]
_LARGE_SIZE_ROWS = [
    pytest.param(((1024, 1024), 2, 0, 8), id="large-2d"),
    pytest.param(((20, 320, 15), 3, 0, 8), id="large-3d"),
    pytest.param(((16, 128, 64, 60), 2, 2, 4), id="large-4d-hybrid"),
    pytest.param(((16, 7, 57, 32, 29), 5, 0, 8), id="large-5d"),
]
_SIZE_ROWS = _QUICK_SIZE_ROWS + _LARGE_SIZE_ROWS


def _make_inputs(size, sparse_dim, nnz, dtype, value_range):
    """Valid COO members: int64 in-range indices plus values of the target type."""
    if nnz == 0 or sparse_dim == 0:
        indices = torch.zeros(
            (sparse_dim, nnz), dtype=torch.int64, device=flag_gems.device
        )
    else:
        indices = torch.stack(
            [
                torch.randint(
                    0, extent, (nnz,), dtype=torch.int64, device=flag_gems.device
                )
                for extent in size[:sparse_dim]
            ]
        )
    values = tu.make_input(dtype, (nnz,) + tuple(size[sparse_dim:]), value_range)
    return indices, values


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
@pytest.mark.parametrize(
    "size_row", tu.selected_cases(_SIZE_ROWS, quick=_QUICK_SIZE_ROWS)
)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test__sparse_coo_tensor_with_dims_and_tensors(size_row, value_range, dtype):
    size, sparse_dim, dense_dim, nnz = size_row
    indices, values = _make_inputs(size, sparse_dim, nnz, dtype, value_range)
    ref_indices = tu.to_reference(indices)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten._sparse_coo_tensor_with_dims_and_tensors(
        sparse_dim,
        dense_dim,
        list(size),
        ref_indices,
        ref_values,
        layout=torch.sparse_coo,
        device=ref_values.device,
        dtype=dtype,
    )
    res_out = flag_gems._sparse_coo_tensor_with_dims_and_tensors(
        sparse_dim,
        dense_dim,
        list(size),
        indices,
        values,
        layout=torch.sparse_coo,
        device=values.device,
        dtype=dtype,
    )

    assert res_out.layout == ref_out.layout == torch.sparse_coo
    assert res_out.dtype == ref_out.dtype == dtype
    assert res_out.shape == ref_out.shape == torch.Size(size)
    assert res_out.sparse_dim() == ref_out.sparse_dim() == sparse_dim
    assert res_out.dense_dim() == ref_out.dense_dim() == dense_dim
    assert res_out._nnz() == ref_out._nnz() == nnz
    assert res_out.device == values.device
    assert res_out._indices().dtype == torch.int64
    assert res_out._indices().shape == torch.Size((sparse_dim, nnz))
    assert res_out._values().shape == torch.Size((nnz,) + tuple(size[sparse_dim:]))
    # nnz == 0 reports coalesced, every other row keeps the inferred flag.
    assert res_out.is_coalesced() == ref_out.is_coalesced()
    tu.assert_result_equal(res_out._indices(), ref_out._indices())
    tu.assert_result_equal(res_out._values(), ref_out._values())


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
@pytest.mark.parametrize("nnz", [0, 4])
@pytest.mark.parametrize("is_coalesced", [None, True, False])
def test__sparse_coo_tensor_with_dims_and_tensors_is_coalesced(nnz, is_coalesced):
    # An explicit is_coalesced is stored verbatim; omitting the schema default is
    # a different call, so it is one of the parametrized cases.
    indices = (
        torch.arange(nnz, dtype=torch.int64, device=flag_gems.device)
        .mul(2)
        .unsqueeze(0)
    )
    values = tu.make_input(torch.float32, (nnz,), ["-1", "1"])
    ref_indices = tu.to_reference(indices)
    ref_values = tu.to_reference(values)
    extra = {} if is_coalesced is None else {"is_coalesced": is_coalesced}

    ref_out = torch.ops.aten._sparse_coo_tensor_with_dims_and_tensors(
        1,
        0,
        [8],
        ref_indices,
        ref_values,
        layout=torch.sparse_coo,
        device=ref_values.device,
        dtype=torch.float32,
        **extra,
    )
    res_out = flag_gems._sparse_coo_tensor_with_dims_and_tensors(
        1,
        0,
        [8],
        indices,
        values,
        layout=torch.sparse_coo,
        device=values.device,
        dtype=torch.float32,
        **extra,
    )

    assert res_out._nnz() == ref_out._nnz() == nnz
    assert res_out.is_coalesced() == ref_out.is_coalesced()
    if is_coalesced is not None:
        assert res_out.is_coalesced() is is_coalesced
    tu.assert_result_equal(res_out._indices(), ref_out._indices())
    tu.assert_result_equal(res_out._values(), ref_out._values())


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
def test__sparse_coo_tensor_with_dims_and_tensors_omits_optional_arguments():
    # `dtype`, `pin_memory` and `is_coalesced` have schema defaults. Omitting
    # `dtype` means the result takes the default dtype and the values must
    # already match it, so this case uses torch.get_default_dtype().
    dtype = torch.get_default_dtype()
    indices, values = _make_inputs((4,), 1, 3, dtype, ["-1", "1"])
    ref_indices = tu.to_reference(indices)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten._sparse_coo_tensor_with_dims_and_tensors(
        1,
        0,
        [4],
        ref_indices,
        ref_values,
        layout=torch.sparse_coo,
        device=ref_values.device,
    )
    res_out = flag_gems._sparse_coo_tensor_with_dims_and_tensors(
        1, 0, [4], indices, values, layout=torch.sparse_coo, device=values.device
    )

    assert res_out.dtype == ref_out.dtype == dtype
    assert res_out._nnz() == ref_out._nnz() == 3
    assert res_out.is_coalesced() == ref_out.is_coalesced()
    tu.assert_result_equal(res_out._indices(), ref_out._indices())
    tu.assert_result_equal(res_out._values(), ref_out._values())


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test__sparse_coo_tensor_with_dims_and_tensors_keeps_input_storage(dtype):
    # The result shares the members' untyped storage and keeps their strides and
    # storage offsets, so a write through the result reaches the caller's tensors.
    indices = (
        torch.arange(6, dtype=torch.int64, device=flag_gems.device).reshape(3, 2).t()
    )
    values = tu.make_input(dtype, (3, 8), ["-1", "1"])[:, :1].squeeze(1)
    ref_indices = tu.to_reference(indices)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten._sparse_coo_tensor_with_dims_and_tensors(
        2,
        0,
        [8, 9],
        ref_indices,
        ref_values,
        layout=torch.sparse_coo,
        device=ref_values.device,
        dtype=dtype,
    )
    res_out = flag_gems._sparse_coo_tensor_with_dims_and_tensors(
        2,
        0,
        [8, 9],
        indices,
        values,
        layout=torch.sparse_coo,
        device=values.device,
        dtype=dtype,
    )

    assert res_out._indices().data_ptr() == indices.data_ptr()
    assert res_out._values().data_ptr() == values.data_ptr()
    assert res_out._indices().stride() == indices.stride()
    assert res_out._values().stride() == values.stride()
    assert res_out._values().storage_offset() == values.storage_offset()
    tu.assert_result_equal(res_out._indices(), ref_out._indices())
    tu.assert_result_equal(res_out._values(), ref_out._values())

    res_out._values().fill_(0)
    assert bool((values == 0).all())


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_VALUE_CASES)
def test__sparse_coo_tensor_with_dims_and_tensors_special_values(dtype, scenario):
    # Stored FP8 payloads are compared through the shared helper, which widens
    # them only inside the comparison.
    nnz = 5
    values = tu.make_special_input(dtype, scenario)
    indices = torch.arange(nnz, dtype=torch.int64, device=flag_gems.device).repeat(2, 1)
    ref_indices = tu.to_reference(indices)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten._sparse_coo_tensor_with_dims_and_tensors(
        2,
        0,
        [nnz, nnz],
        ref_indices,
        ref_values,
        layout=torch.sparse_coo,
        device=ref_values.device,
        dtype=dtype,
    )
    res_out = flag_gems._sparse_coo_tensor_with_dims_and_tensors(
        2,
        0,
        [nnz, nnz],
        indices,
        values,
        layout=torch.sparse_coo,
        device=values.device,
        dtype=dtype,
    )

    assert res_out.dtype == ref_out.dtype == dtype
    tu.assert_result_equal(res_out._values(), ref_out._values())
    tu.assert_result_equal(res_out._indices(), ref_out._indices())


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test__sparse_coo_tensor_with_dims_and_tensors_out(dtype):
    # `.out` writes the members into an existing COO tensor and returns it, so
    # the buffer is built with a different nnz and different content than the
    # source to show that the stored indices and values are replaced.
    size, sparse_dim, dense_dim, nnz = (4, 3), 1, 1, 3
    indices, values = _make_inputs(size, sparse_dim, nnz, dtype, ["-1", "1"])
    buf_indices, buf_values = _make_inputs(size, sparse_dim, nnz - 1, dtype, ["0", "1"])
    ref_indices = tu.to_reference(indices)
    ref_values = tu.to_reference(values)
    ref_buf_indices = tu.to_reference(buf_indices)
    ref_buf_values = tu.to_reference(buf_values)

    ref_buffer = torch.ops.aten._sparse_coo_tensor_with_dims_and_tensors(
        sparse_dim,
        dense_dim,
        list(size),
        ref_buf_indices,
        ref_buf_values,
        layout=torch.sparse_coo,
        device=ref_buf_values.device,
        dtype=dtype,
    )
    res_buffer = flag_gems._sparse_coo_tensor_with_dims_and_tensors(
        sparse_dim,
        dense_dim,
        list(size),
        buf_indices,
        buf_values,
        layout=torch.sparse_coo,
        device=buf_values.device,
        dtype=dtype,
    )
    ref_out = torch.ops.aten._sparse_coo_tensor_with_dims_and_tensors.out(
        sparse_dim, dense_dim, list(size), ref_indices, ref_values, out=ref_buffer
    )
    res_out = flag_gems._sparse_coo_tensor_with_dims_and_tensors(
        sparse_dim, dense_dim, list(size), indices, values, out=res_buffer
    )

    assert res_out is res_buffer
    assert res_out.shape == ref_out.shape == torch.Size(size)
    assert res_out.sparse_dim() == ref_out.sparse_dim() == sparse_dim
    assert res_out.dense_dim() == ref_out.dense_dim() == dense_dim
    assert res_out._nnz() == ref_out._nnz() == nnz
    tu.assert_result_equal(res_out._indices(), ref_out._indices())
    tu.assert_result_equal(res_out._values(), ref_out._values())


def _coo_members():
    """Valid members for a 2-sparse-dim, no-dense-dim result."""
    return _make_inputs((4, 5), 2, 3, torch.float32, ["-1", "1"])


def _invalid_call(
    indices, values, *, sparse_dim=2, dense_dim=0, size=(4, 5), **overrides
):
    """Call the candidate with every argument valid except the overridden one."""
    kwargs = {
        "layout": torch.sparse_coo,
        "device": values.device,
        "dtype": values.dtype,
    }
    kwargs.update(overrides)
    return flag_gems._sparse_coo_tensor_with_dims_and_tensors(
        sparse_dim, dense_dim, list(size), indices, values, **kwargs
    )


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
def test__sparse_coo_tensor_with_dims_and_tensors_rejects_non_int64_indices():
    indices, values = _coo_members()
    with pytest.raises(RuntimeError):
        _invalid_call(indices.int(), values)


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
def test__sparse_coo_tensor_with_dims_and_tensors_rejects_non_2d_indices():
    indices, values = _coo_members()
    with pytest.raises(RuntimeError):
        _invalid_call(indices.reshape(-1), values)


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
def test__sparse_coo_tensor_with_dims_and_tensors_rejects_nnz_mismatch():
    indices, values = _coo_members()
    with pytest.raises(RuntimeError):
        _invalid_call(indices, values[:-1])


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
def test__sparse_coo_tensor_with_dims_and_tensors_rejects_size_rank_mismatch():
    indices, values = _coo_members()
    with pytest.raises(RuntimeError):
        _invalid_call(indices, values, size=(4, 5, 6))


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
def test__sparse_coo_tensor_with_dims_and_tensors_rejects_indices_row_mismatch():
    indices, values = _coo_members()
    with pytest.raises(RuntimeError):
        _invalid_call(indices, values, sparse_dim=1)


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
def test__sparse_coo_tensor_with_dims_and_tensors_rejects_negative_sparse_dim():
    indices, values = _coo_members()
    with pytest.raises(RuntimeError):
        _invalid_call(indices, values, sparse_dim=-1)


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
def test__sparse_coo_tensor_with_dims_and_tensors_rejects_negative_dense_dim():
    indices, values = _coo_members()
    with pytest.raises(RuntimeError):
        _invalid_call(indices, values, dense_dim=-1)


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
def test__sparse_coo_tensor_with_dims_and_tensors_rejects_dtype_mismatch():
    indices, values = _coo_members()
    with pytest.raises(RuntimeError):
        _invalid_call(indices, values, dtype=torch.float64)


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
def test__sparse_coo_tensor_with_dims_and_tensors_rejects_non_tensor_indices():
    _, values = _coo_members()
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _invalid_call([[0, 1, 2], [0, 1, 2]], values)


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
@pytest.mark.parametrize("layout", [None, torch.strided])
def test__sparse_coo_tensor_with_dims_and_tensors_rejects_non_sparse_layout(layout):
    indices, values = _coo_members()
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _invalid_call(indices, values, layout=layout)


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
def test__sparse_coo_tensor_with_dims_and_tensors_rejects_pin_memory():
    # CUDA members cannot be pinned, so pin_memory=True is invalid here.
    indices, values = _coo_members()
    with pytest.raises(RuntimeError):
        _invalid_call(indices, values, pin_memory=True)


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
def test__sparse_coo_tensor_with_dims_and_tensors_rejects_dense_out():
    indices, values = _coo_members()
    dense = torch.zeros(4, 5, device=flag_gems.device, dtype=values.dtype)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._sparse_coo_tensor_with_dims_and_tensors(
            2, 0, [4, 5], indices, values, out=dense
        )
