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
from . import conftest as cfg
from . import test_utils as tu

# Exercise explicit/inferred size, size-only and size_out COO construction.
# Preserve the stored entries and coalesced flag; pass dtype/device explicitly.
_COO_DTYPES = tu.REQUIRED_DTYPES + [torch.float64, torch.int16, torch.bool]

# (size, indices) with explicit 2-D sizes.
_COO_2D_CASES = [
    ((2, 3), [[0, 1, 1], [2, 0, 2]]),
    ((4, 5), [[0, 1, 3, 0], [1, 2, 4, 0]]),
    ((6, 8), [[0, 1, 3, 0], [1, 2, 4, 0]]),
    ((5, 6), [[2, 4, 1], [3, 0, 5]]),
    ((3, 3), [[0, 2, 1, 2], [0, 1, 2, 0]]),
]

# Trailing dimensions beyond len(indices) are dense.
_COO_ND_CASES = [
    ((5,), [[0, 2, 4]]),
    ((3, 4, 5), [[0, 1, 2, 1], [1, 3, 0, 2]]),
    ((2, 3, 7), [[0, 1, 1], [2, 0, 2]]),
    ((2, 3, 4, 5), [[0, 1, 1], [2, 0, 2]]),
    ((2, 3, 4), [[0, 1, 1], [2, 0, 2], [1, 3, 0]]),
    ((3, 3, 3), [[0, 2, 1], [1, 0, 2], [2, 1, 0]]),
]

# (inferred_size, indices, dense_shape).
_COO_INFERRED_CASES = [
    ((2, 3), [[0, 1, 1], [2, 0, 2]], ()),
    ((4, 5), [[0, 1, 3, 0], [1, 2, 4, 0]], ()),
    ((5, 6), [[2, 4, 1], [3, 0, 5]], ()),
    ((2, 3, 4), [[0, 1, 1], [2, 0, 2]], (4,)),
]

# Size-only construction creates empty storage.
_COO_SIZE_ONLY_CASES = [
    ((5,),),
    ((2, 3),),
    ((4, 5, 6),),
]

# (size, sparse_dim), with no stored entries.
_COO_EMPTY_CASES = [
    ((2, 3), 2),
    ((4, 5, 6), 2),
    ((2, 3, 4, 5), 2),
    ((3, 4, 5), 3),
]

# (variant, size, indices, dense_shape); inferred cases use the expected size.
_COO_VALUE_CASES = [
    ("indices_size", (2, 3), [[0, 1, 1], [2, 0, 2]], ()),
    ("indices_size", (2, 3, 4, 5), [[0, 1, 1], [2, 0, 2]], (4, 5)),
    ("indices", (2, 3, 4), [[0, 1, 1], [2, 0, 2]], (4,)),
]

# (size, indices) for special-value scenarios.
_NAN_INF_CASES = [
    ((5,), [[0, 2, 4]]),
    ((2, 3), [[0, 1, 1], [2, 0, 2]]),
    ((4, 5, 6), [[0, 1, 1], [2, 0, 2]]),
]

_VALUE_RANGE_CASES = [
    (value_range, case, dtype)
    for case in _COO_VALUE_CASES
    for dtype in _COO_DTYPES
    for value_range in tu.selected_ranges()
]


def _reference_device():
    # Use CPU only when --ref cpu was requested.
    return "cpu" if cfg.TO_CPU else flag_gems.device


def _make_values(nnz, dense_shape, dtype, value_range=None):
    if value_range is None:
        value_range = ["-1", "1"]
    shape = (nnz,) + tuple(dense_shape)
    return tu.make_input(dtype, shape, value_range).to(flag_gems.device)


def _make_special_values(shape, dtype, scenario):
    base = tu.make_special_input(dtype, scenario)
    numel = math.prod(int(extent) for extent in shape)
    repeats = (numel + base.numel() - 1) // base.numel()
    return base.repeat(repeats)[:numel].reshape(shape)


def _make_out_buffer(size, dtype, device, nnz):
    # Non-empty buffers make size_out clear existing entries.
    if nnz == 0:
        return torch.ops.aten.sparse_coo_tensor(list(size), dtype=dtype, device=device)
    indices = torch.zeros((len(size), nnz), dtype=torch.long, device=device)
    values = tu.make_input(dtype, (nnz,), ["0", "1"]).to(device)
    return torch.ops.aten.sparse_coo_tensor(
        indices, values, list(size), dtype=dtype, device=device
    )


def _assert_coo_structure(
    res_out, ref_out, size, nnz, dtype, sparse_dim, dense_dim, is_coalesced=None
):
    assert res_out.layout == torch.sparse_coo
    assert tuple(res_out.shape) == tuple(size)
    assert res_out.dtype == dtype
    assert res_out.device.type == torch.device(flag_gems.device).type
    assert res_out.sparse_dim() == sparse_dim
    assert res_out.dense_dim() == dense_dim
    assert torch.ops.aten._nnz(res_out) == nnz
    assert tuple(torch.ops.aten._indices(res_out).shape) == (sparse_dim, nnz)
    assert tuple(torch.ops.aten._values(res_out).shape) == (nnz,) + tuple(
        size[sparse_dim:]
    )
    # Preserve the constructor's coalesced flag.
    assert res_out.is_coalesced() == ref_out.is_coalesced()
    if is_coalesced is not None:
        assert res_out.is_coalesced() == is_coalesced
    # The index tensors are exact integer data stored verbatim.
    utils.gems_assert_equal(
        torch.ops.aten._indices(res_out), torch.ops.aten._indices(ref_out)
    )


def _call_reference(indices, values, size, dtype):
    # Clone components; size=None selects the size-inferred overload.
    ref_indices = tu.to_reference(indices)
    ref_values = tu.to_reference(values)
    if size is None:
        return torch.ops.aten.sparse_coo_tensor(
            ref_indices, ref_values, dtype=dtype, device=ref_indices.device
        )
    return torch.ops.aten.sparse_coo_tensor(
        ref_indices, ref_values, list(size), dtype=dtype, device=ref_indices.device
    )


def _call_candidate(indices, values, size, dtype, **extra):
    # Match the reference overload: size=None selects size inference.
    if size is None:
        return flag_gems.sparse_coo_tensor(
            indices, values, dtype=dtype, device=indices.device, **extra
        )
    return flag_gems.sparse_coo_tensor(
        indices, values, list(size), dtype=dtype, device=indices.device, **extra
    )


def _assert_rejected(ref_call, candidate_call):
    with pytest.raises(RuntimeError):
        ref_call()
    with pytest.raises((NotImplementedError, RuntimeError, TypeError, ValueError)):
        candidate_call()


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("case", _COO_SIZE_ONLY_CASES)
@pytest.mark.parametrize("dtype", _COO_DTYPES)
def test_sparse_coo_tensor_size(case, dtype):
    (size,) = case
    ref_device = _reference_device()
    ref_out = torch.ops.aten.sparse_coo_tensor(
        list(size), dtype=dtype, device=ref_device
    )
    res_out = flag_gems.sparse_coo_tensor(
        list(size), dtype=dtype, device=flag_gems.device
    )

    _assert_coo_structure(res_out, ref_out, size, 0, dtype, len(size), 0)
    tu.assert_result_equal(res_out._values(), ref_out._values())


@pytest.mark.sparse_coo_tensor_size_out
@pytest.mark.parametrize("case", _COO_SIZE_ONLY_CASES)
@pytest.mark.parametrize("dtype", _COO_DTYPES)
def test_sparse_coo_tensor_size_out(case, dtype):
    (size,) = case
    ref_device = _reference_device()
    ref_out = _make_out_buffer(size, dtype, ref_device, nnz=2)
    out = _make_out_buffer(size, dtype, flag_gems.device, nnz=2)

    ref_ret = torch.ops.aten.sparse_coo_tensor.size_out(list(size), out=ref_out)
    res_ret = flag_gems.sparse_coo_tensor(list(size), out=out)

    assert res_ret is out
    _assert_coo_structure(res_ret, ref_ret, size, 0, dtype, len(size), 0)
    tu.assert_result_equal(res_ret._values(), ref_ret._values())
    # The returned tensor aliases the caller buffer (asserted above), so the
    # buffer itself already carries the reference structure.
    assert tuple(out.shape) == tuple(size)
    assert torch.ops.aten._nnz(out) == 0


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("case", _COO_2D_CASES)
@pytest.mark.parametrize("dtype", _COO_DTYPES)
def test_sparse_coo_tensor_indices_size(case, dtype):
    size, indices = case
    nnz = len(indices[0])
    indices_t = torch.tensor(indices, dtype=torch.long, device=flag_gems.device)
    values = _make_values(nnz, (), dtype)

    ref_out = _call_reference(indices_t, values, size, dtype)
    res_out = _call_candidate(indices_t, values, size, dtype)

    _assert_coo_structure(res_out, ref_out, size, nnz, dtype, 2, 0)
    tu.assert_result_equal(res_out._values(), ref_out._values())


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("case", _COO_ND_CASES)
@pytest.mark.parametrize("dtype", _COO_DTYPES)
def test_sparse_coo_tensor_indices_size_nd(case, dtype):
    size, indices = case
    sparse_dim = len(indices)
    dense_dim = len(size) - sparse_dim
    nnz = len(indices[0])
    indices_t = torch.tensor(indices, dtype=torch.long, device=flag_gems.device)
    values = _make_values(nnz, tuple(size[sparse_dim:]), dtype)

    ref_out = _call_reference(indices_t, values, size, dtype)
    res_out = _call_candidate(indices_t, values, size, dtype)

    _assert_coo_structure(res_out, ref_out, size, nnz, dtype, sparse_dim, dense_dim)
    tu.assert_result_equal(res_out._values(), ref_out._values())


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("case", _COO_INFERRED_CASES)
@pytest.mark.parametrize("dtype", _COO_DTYPES)
def test_sparse_coo_tensor_indices(case, dtype):
    size, indices, dense_shape = case
    sparse_dim = len(indices)
    dense_dim = len(dense_shape)
    nnz = len(indices[0])
    indices_t = torch.tensor(indices, dtype=torch.long, device=flag_gems.device)
    values = _make_values(nnz, dense_shape, dtype)

    ref_out = _call_reference(indices_t, values, None, dtype)
    res_out = _call_candidate(indices_t, values, None, dtype)

    _assert_coo_structure(res_out, ref_out, size, nnz, dtype, sparse_dim, dense_dim)
    tu.assert_result_equal(res_out._values(), ref_out._values())


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("case", _COO_EMPTY_CASES)
@pytest.mark.parametrize("dtype", _COO_DTYPES)
def test_sparse_coo_tensor_indices_size_empty(case, dtype):
    size, sparse_dim = case
    dense_shape = tuple(size[sparse_dim:])
    indices_t = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = _make_values(0, dense_shape, dtype)

    ref_out = _call_reference(indices_t, values, size, dtype)
    res_out = _call_candidate(indices_t, values, size, dtype)

    _assert_coo_structure(
        res_out, ref_out, size, 0, dtype, sparse_dim, len(dense_shape)
    )
    tu.assert_result_equal(res_out._values(), ref_out._values())


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("dtype", _COO_DTYPES)
def test_sparse_coo_tensor_indices_size_is_coalesced(dtype):
    size = (2, 3)
    indices = [[0, 1, 1], [2, 0, 2]]
    nnz = 3
    indices_t = torch.tensor(indices, dtype=torch.long, device=flag_gems.device)
    values = _make_values(nnz, (), dtype)
    ref_indices = tu.to_reference(indices_t)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten.sparse_coo_tensor(
        ref_indices,
        ref_values,
        list(size),
        dtype=dtype,
        device=ref_indices.device,
        is_coalesced=True,
    )
    res_out = _call_candidate(indices_t, values, size, dtype, is_coalesced=True)

    _assert_coo_structure(res_out, ref_out, size, nnz, dtype, 2, 0, is_coalesced=True)
    tu.assert_result_equal(res_out._values(), ref_out._values())


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("value_range,case,dtype", _VALUE_RANGE_CASES)
def test_sparse_coo_tensor_value_ranges(value_range, case, dtype):
    variant, size, indices, dense_shape = case
    sparse_dim = len(indices)
    dense_dim = len(dense_shape)
    nnz = len(indices[0])
    indices_t = torch.tensor(indices, dtype=torch.long, device=flag_gems.device)
    values = _make_values(nnz, dense_shape, dtype, value_range)

    ref_out = _call_reference(
        indices_t, values, size if variant == "indices_size" else None, dtype
    )
    res_out = _call_candidate(
        indices_t, values, size if variant == "indices_size" else None, dtype
    )

    _assert_coo_structure(res_out, ref_out, size, nnz, dtype, sparse_dim, dense_dim)
    tu.assert_result_equal(res_out._values(), ref_out._values())


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("case", _NAN_INF_CASES)
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_COO_DTYPES))
)
def test_sparse_coo_tensor_nan_inf(case, dtype, scenario):
    size, indices = case
    sparse_dim = len(indices)
    dense_shape = tuple(size[sparse_dim:])
    nnz = len(indices[0])
    indices_t = torch.tensor(indices, dtype=torch.long, device=flag_gems.device)
    values = _make_special_values((nnz,) + dense_shape, dtype, scenario)

    ref_out = _call_reference(indices_t, values, size, dtype)
    res_out = _call_candidate(indices_t, values, size, dtype)

    _assert_coo_structure(
        res_out, ref_out, size, nnz, dtype, sparse_dim, len(dense_shape)
    )
    tu.assert_result_equal(res_out._values(), ref_out._values())


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("dtype", _COO_DTYPES)
def test_sparse_coo_tensor_zero_extent(dtype):
    size = (0, 4)
    indices_t = torch.empty(2, 0, dtype=torch.long, device=flag_gems.device)
    values = _make_values(0, (), dtype)

    ref_out = _call_reference(indices_t, values, size, dtype)
    res_out = _call_candidate(indices_t, values, size, dtype)

    _assert_coo_structure(res_out, ref_out, size, 0, dtype, 2, 0)
    tu.assert_result_equal(res_out._values(), ref_out._values())


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("case", _COO_VALUE_CASES)
def test_sparse_coo_tensor_inputs_not_mutated(case):
    variant, size, indices, dense_shape = case
    nnz = len(indices[0])
    dtype = torch.float32
    indices_t = torch.tensor(indices, dtype=torch.long, device=flag_gems.device)
    values = _make_values(nnz, dense_shape, dtype)
    # Snapshot through the reference-device helper: under ``--ref cpu`` the
    # snapshots live on the CPU, which is the convention the accuracy helpers
    # expect for the reference operand.
    indices_before = tu.to_reference(indices_t)
    values_before = tu.to_reference(values)

    out = _call_candidate(
        indices_t, values, size if variant == "indices_size" else None, dtype
    )

    assert out is not indices_t
    utils.gems_assert_equal(indices_t, indices_before)
    tu.assert_result_equal(values, values_before)


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_indices_ndim():
    indices_t = torch.tensor([0, 1, 2], dtype=torch.long, device=flag_gems.device)
    values = _make_values(3, (), torch.float32)

    _assert_rejected(
        lambda: _call_reference(indices_t, values, [3], torch.float32),
        lambda: _call_candidate(indices_t, values, [3], torch.float32),
    )


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_size():
    indices_t = torch.tensor(
        [[0, 1], [2, 0]], dtype=torch.long, device=flag_gems.device
    )
    values = _make_values(2, (), torch.float32)

    _assert_rejected(
        lambda: _call_reference(indices_t, values, [-2, 3], torch.float32),
        lambda: _call_candidate(indices_t, values, [-2, 3], torch.float32),
    )


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_size_only_size():
    ref_device = _reference_device()

    _assert_rejected(
        lambda: torch.ops.aten.sparse_coo_tensor(
            [-2, 3], dtype=torch.float32, device=ref_device
        ),
        lambda: flag_gems.sparse_coo_tensor(
            [-2, 3], dtype=torch.float32, device=flag_gems.device
        ),
    )


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_non_integer_size():
    _assert_rejected(
        lambda: torch.ops.aten.sparse_coo_tensor(
            [2.5, 3], dtype=torch.float32, device=_reference_device()
        ),
        lambda: flag_gems.sparse_coo_tensor(
            [2.5, 3], dtype=torch.float32, device=flag_gems.device
        ),
    )


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_indices_dtype():
    indices_t = torch.tensor(
        [[0, 1], [2, 0]], dtype=torch.int32, device=flag_gems.device
    )
    values = _make_values(2, (), torch.float32)

    _assert_rejected(
        lambda: _call_reference(indices_t, values, [2, 3], torch.float32),
        lambda: _call_candidate(indices_t, values, [2, 3], torch.float32),
    )


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_indices_float():
    indices_t = torch.tensor(
        [[0, 1], [2, 0]], dtype=torch.long, device=flag_gems.device
    ).float()
    values = _make_values(2, (), torch.float32)

    _assert_rejected(
        lambda: _call_reference(indices_t, values, [2, 3], torch.float32),
        lambda: _call_candidate(indices_t, values, [2, 3], torch.float32),
    )


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_nnz_mismatch():
    indices_t = torch.tensor(
        [[0, 1], [2, 0]], dtype=torch.long, device=flag_gems.device
    )
    values = _make_values(3, (), torch.float32)

    _assert_rejected(
        lambda: _call_reference(indices_t, values, [2, 3], torch.float32),
        lambda: _call_candidate(indices_t, values, [2, 3], torch.float32),
    )


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_values_dense_dims():
    indices_t = torch.tensor(
        [[0, 1], [2, 0]], dtype=torch.long, device=flag_gems.device
    )
    values = _make_values(2, (), torch.float32)

    _assert_rejected(
        lambda: _call_reference(indices_t, values, [2, 3, 4], torch.float32),
        lambda: _call_candidate(indices_t, values, [2, 3, 4], torch.float32),
    )


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_indices_size_rank():
    indices_t = torch.tensor(
        [[0, 1], [2, 0]], dtype=torch.long, device=flag_gems.device
    )
    values = _make_values(2, (), torch.float32)

    _assert_rejected(
        lambda: _call_reference(indices_t, values, [2], torch.float32),
        lambda: _call_candidate(indices_t, values, [2], torch.float32),
    )


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_inferred_negative_index():
    indices_t = torch.tensor([[-1, 1]], dtype=torch.long, device=flag_gems.device)
    values = _make_values(2, (), torch.float32)

    _assert_rejected(
        lambda: _call_reference(indices_t, values, None, torch.float32),
        lambda: _call_candidate(indices_t, values, None, torch.float32),
    )


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_layout():
    indices_t = torch.tensor(
        [[0, 1], [2, 0]], dtype=torch.long, device=flag_gems.device
    )
    values = _make_values(2, (), torch.float32)
    ref_indices = tu.to_reference(indices_t)
    ref_values = tu.to_reference(values)

    _assert_rejected(
        lambda: torch.ops.aten.sparse_coo_tensor(
            ref_indices,
            ref_values,
            [2, 3],
            dtype=torch.float32,
            device=ref_indices.device,
            layout=torch.sparse_csr,
        ),
        lambda: _call_candidate(
            indices_t, values, [2, 3], torch.float32, layout=torch.sparse_csr
        ),
    )


@pytest.mark.sparse_coo_tensor_size_out
def test_sparse_coo_tensor_size_out_negative_shape():
    ref_out = torch.ops.aten.sparse_coo_tensor(
        [4, 5], dtype=torch.float32, device=_reference_device()
    )
    out = torch.ops.aten.sparse_coo_tensor(
        [4, 5], dtype=torch.float32, device=flag_gems.device
    )

    _assert_rejected(
        lambda: torch.ops.aten.sparse_coo_tensor.size_out([2, 3], out=ref_out),
        lambda: flag_gems.sparse_coo_tensor([2, 3], out=out),
    )
