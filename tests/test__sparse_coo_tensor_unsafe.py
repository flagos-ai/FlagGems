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

# _sparse_coo_tensor_unsafe is a metadata constructor: it wraps the caller's
# coordinate and value storages in a new COO TensorImpl and never touches the
# values numerically, so the value-range grid applies to ``values`` only.
# There is no broadcast form (the operands are a coordinate buffer, a value
# buffer and a size list) and no scalar operand, so neither dimension appears.

_NNZ = 4  # stored entries per value-range case

# Probed as accepted ``values`` dtypes on the active backend: the nine spec
# dtypes plus the other value types COO storage supports.
_VALUE_DTYPES = [
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
    torch.bool,
    torch.complex64,
]

_FLOAT_DTYPES = [
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.float64,
]

_BACKWARD_DTYPES = [torch.float32, torch.float16, torch.bfloat16, torch.float64]


def _coo_indices(size, nnz):
    """In-bounds coordinate rows for ``nnz`` entries, shaped (sparse_dim, nnz)."""
    device = flag_gems.device
    if not size:
        return torch.empty((0, nnz), dtype=torch.int64, device=device)
    rows = [
        torch.randint(0, int(dim), (nnz,), dtype=torch.int64, device=device)
        for dim in size
    ]
    return torch.stack(rows, 0)


def _assert_sparse_equal(res_out, ref_out):
    # A COO comparison alone is not enough: coordinates can be permuted while
    # the dense materialization stays equal, so indices and values are compared
    # component by component as well.
    tu.assert_result_equal(res_out._indices(), ref_out._indices())
    tu.assert_result_equal(res_out._values(), ref_out._values())
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.sparse_coo_tensor_unsafe
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _VALUE_DTYPES)
def test_sparse_coo_tensor_unsafe(shape, value_range, dtype):
    size = [int(dim) for dim in shape]
    indices = _coo_indices(size, _NNZ)
    values = tu.make_input(dtype, (_NNZ,), value_range)

    ref_out = torch.ops.aten._sparse_coo_tensor_unsafe(
        tu.to_reference(indices), tu.to_reference(values), size
    )
    res_out = flag_gems._sparse_coo_tensor_unsafe(indices, values, size)

    _assert_sparse_equal(res_out, ref_out)
    # The shared assertions do not describe the sparse/dense split of the result.
    assert res_out.sparse_dim() == len(size)
    assert res_out.dense_dim() == 0
    assert res_out._nnz() == _NNZ


# (size, sparse_dim, dense_shape, nnz, default is_coalesced). The default is
# measured: a freshly built tensor reports coalesced below two stored entries.
_STRUCTURE_ROWS = [
    ((5, 6), 2, (), 3, False),
    ((5, 4), 1, (4,), 3, False),
    ((7, 3, 5), 1, (3, 5), 2, False),
    ((5, 6), 2, (), 0, True),
    ((5, 6), 2, (), 1, True),
    ((4, 5, 6), 3, (), 2, False),
    ((), 0, (), 1, True),
]


@pytest.mark.sparse_coo_tensor_unsafe
@pytest.mark.parametrize(
    "size,sparse_dim,dense_shape,nnz,is_coalesced", _STRUCTURE_ROWS
)
def test_sparse_coo_tensor_unsafe_structure(
    size, sparse_dim, dense_shape, nnz, is_coalesced
):
    indices = _coo_indices(size[:sparse_dim], nnz)
    values = tu.make_input(torch.float32, (nnz,) + dense_shape, ["-1", "1"])

    ref_out = torch.ops.aten._sparse_coo_tensor_unsafe(
        tu.to_reference(indices), tu.to_reference(values), list(size)
    )
    res_out = flag_gems._sparse_coo_tensor_unsafe(indices, values, list(size))

    _assert_sparse_equal(res_out, ref_out)
    assert res_out.shape == torch.Size(size)
    assert res_out.sparse_dim() == sparse_dim
    assert res_out.dense_dim() == len(dense_shape)
    assert res_out._nnz() == nnz
    assert res_out.is_coalesced() is is_coalesced


@pytest.mark.sparse_coo_tensor_unsafe
@pytest.mark.parametrize("is_coalesced", [True, False])
def test_sparse_coo_tensor_unsafe_explicit_coalesced(is_coalesced):
    # is_coalesced overrides the measured default; both values must be honoured
    # even though the coordinates arrive unsorted.
    size = [20, 320, 15]
    indices = _coo_indices(size, _NNZ)
    values = tu.make_input(torch.float32, (_NNZ,), ["-1", "1"])

    ref_out = torch.ops.aten._sparse_coo_tensor_unsafe(
        tu.to_reference(indices),
        tu.to_reference(values),
        size,
        is_coalesced=is_coalesced,
    )
    res_out = flag_gems._sparse_coo_tensor_unsafe(
        indices, values, size, is_coalesced=is_coalesced
    )

    _assert_sparse_equal(res_out, ref_out)
    assert res_out.is_coalesced() is is_coalesced


@pytest.mark.sparse_coo_tensor_unsafe
@pytest.mark.parametrize(
    "size", [[20, 320, 15], (20, 320, 15), torch.Size([20, 320, 15])]
)
def test_sparse_coo_tensor_unsafe_size_container(size):
    indices = _coo_indices(list(size), _NNZ)
    values = tu.make_input(torch.float32, (_NNZ,), ["-1", "1"])

    ref_out = torch.ops.aten._sparse_coo_tensor_unsafe(
        tu.to_reference(indices), tu.to_reference(values), size
    )
    res_out = flag_gems._sparse_coo_tensor_unsafe(indices, values, size)

    _assert_sparse_equal(res_out, ref_out)


@pytest.mark.sparse_coo_tensor_unsafe
def test_sparse_coo_tensor_unsafe_size_keyword():
    size = [20, 320, 15]
    indices = _coo_indices(size, _NNZ)
    values = tu.make_input(torch.float32, (_NNZ,), ["-1", "1"])

    ref_out = torch.ops.aten._sparse_coo_tensor_unsafe(
        tu.to_reference(indices), tu.to_reference(values), size=size
    )
    res_out = flag_gems._sparse_coo_tensor_unsafe(indices, values, size=size)

    _assert_sparse_equal(res_out, ref_out)


@pytest.mark.sparse_coo_tensor_unsafe
def test_sparse_coo_tensor_unsafe_keeps_caller_storage():
    size = [20, 320, 15]
    # Non-contiguous coordinates and an offset slice of a larger buffer: the
    # result must adopt the caller's storages and metadata rather than re-pack
    # them into fresh contiguous buffers.
    indices = _coo_indices(size, _NNZ).t().contiguous().t()
    assert not indices.is_contiguous()
    buffer = torch.zeros(_NNZ + 2, dtype=torch.float32, device=flag_gems.device)
    buffer[2:] = torch.arange(_NNZ, dtype=torch.float32, device=flag_gems.device)
    values = buffer[2:]

    ref_out = torch.ops.aten._sparse_coo_tensor_unsafe(
        tu.to_reference(indices), tu.to_reference(values), size
    )
    res_out = flag_gems._sparse_coo_tensor_unsafe(indices, values, size)

    _assert_sparse_equal(res_out, ref_out)
    assert res_out._indices().data_ptr() == indices.data_ptr()
    assert res_out._indices().stride() == indices.stride()
    assert res_out._indices().storage_offset() == indices.storage_offset()
    assert res_out._values().data_ptr() == values.data_ptr()
    assert res_out._values().stride() == values.stride()
    assert res_out._values().storage_offset() == values.storage_offset()

    # One shared storage per component: a write through the result is visible
    # in the buffer the caller still holds.
    res_out._values().fill_(2.5)
    assert torch.equal(values, torch.full_like(values, 2.5))


@pytest.mark.sparse_coo_tensor_unsafe
def test_sparse_coo_tensor_unsafe_accepts_out_of_range_indices():
    # "unsafe" means no coordinate bounds checking: rows past the logical size
    # are stored unchanged instead of being rejected.
    size = [5, 6]
    indices = torch.tensor([[7, 1], [0, 9]], dtype=torch.int64, device=flag_gems.device)
    values = tu.make_input(torch.float32, (2,), ["-1", "1"])

    ref_out = torch.ops.aten._sparse_coo_tensor_unsafe(
        tu.to_reference(indices), tu.to_reference(values), size
    )
    res_out = flag_gems._sparse_coo_tensor_unsafe(indices, values, size)

    _assert_sparse_equal(res_out, ref_out)
    assert res_out._nnz() == 2
    assert torch.equal(res_out._indices(), indices)


@pytest.mark.sparse_coo_tensor_unsafe
@pytest.mark.parametrize(
    "dtype,scenario",
    tu.selected_cases(tu.special_value_cases(_FLOAT_DTYPES), quick=[]),
)
def test_sparse_coo_tensor_unsafe_special_values(dtype, scenario):
    size = [16, 32]
    values = tu.make_special_input(dtype, scenario)
    indices = _coo_indices(size, values.numel())

    ref_out = torch.ops.aten._sparse_coo_tensor_unsafe(
        tu.to_reference(indices), tu.to_reference(values), size
    )
    res_out = flag_gems._sparse_coo_tensor_unsafe(indices, values, size)

    _assert_sparse_equal(res_out, ref_out)


@pytest.mark.sparse_coo_tensor_unsafe
@pytest.mark.parametrize("dtype", tu.selected_cases(_BACKWARD_DTYPES, quick=[]))
def test_sparse_coo_tensor_unsafe_backward(dtype):
    size = [20, 320, 15]
    indices = _coo_indices(size, _NNZ)
    values = tu.make_input(dtype, (_NNZ,), ["-1", "1"]).requires_grad_(True)
    ref_values = values.detach().clone().requires_grad_(True)
    # Distinct weights make each stored entry's gradient observable.
    weights = tu.make_input(dtype, size, ["0", "1"])

    ref_out = torch.ops.aten._sparse_coo_tensor_unsafe(
        tu.to_reference(indices), ref_values, size
    )
    res_out = flag_gems._sparse_coo_tensor_unsafe(indices, values, size)

    ref_grad = torch.autograd.grad((ref_out.to_dense() * weights).sum(), ref_values)[0]
    res_grad = torch.autograd.grad((res_out.to_dense() * weights).sum(), values)[0]

    tu.assert_result_close(res_grad, ref_grad)


@pytest.mark.sparse_coo_tensor_unsafe
@pytest.mark.parametrize(
    "indices_dtype,indices_shape",
    [(torch.int32, (2, 4)), (torch.int64, (2, 4, 1))],
)
def test_sparse_coo_tensor_unsafe_rejects_invalid_indices(indices_dtype, indices_shape):
    # Native contract: coordinates must be an int64 tensor of rank two.
    indices = torch.zeros(indices_shape, dtype=indices_dtype, device=flag_gems.device)
    values = tu.make_input(torch.float32, (4,), ["-1", "1"])

    with pytest.raises(RuntimeError):
        flag_gems._sparse_coo_tensor_unsafe(indices, values, [5, 6])


@pytest.mark.sparse_coo_tensor_unsafe
@pytest.mark.parametrize("size", [[5], [-5, 6], [5.5, 6]])
def test_sparse_coo_tensor_unsafe_rejects_invalid_size(size):
    # Native contract: the size list must hold one int per sparse + dense dim.
    indices = torch.zeros((2, 1), dtype=torch.int64, device=flag_gems.device)
    values = tu.make_input(torch.float32, (1,), ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._sparse_coo_tensor_unsafe(indices, values, size)


@pytest.mark.sparse_coo_tensor_unsafe
def test_sparse_coo_tensor_unsafe_rejects_nnz_mismatch():
    indices = torch.zeros((2, 1), dtype=torch.int64, device=flag_gems.device)
    values = tu.make_input(torch.float32, (3,), ["-1", "1"])

    with pytest.raises(RuntimeError):
        flag_gems._sparse_coo_tensor_unsafe(indices, values, [5, 6])
