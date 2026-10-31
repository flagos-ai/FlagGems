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

COO_UNSAFE_DTYPES = utils.FLOAT_DTYPES


def _make_components(nnz, dtype, seed=0, hybrid=False, unsafe_indices=False):
    """COO components: sorted/distinct by default, or unsorted with duplicates and
    out-of-range coordinates when ``unsafe_indices`` is set."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    if unsafe_indices:
        indices = torch.tensor([[0, 1, 1, 5], [0, 3, 3, 1]], dtype=torch.int64)
    elif nnz == 4:
        indices = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.int64)
    else:
        rows = torch.randint(0, 4, (nnz,), generator=g).to(torch.int64)
        cols = torch.randint(0, 4, (nnz,), generator=g).to(torch.int64)
        indices = torch.stack([rows, cols])
    if hybrid:
        values = torch.randn((nnz, 3), dtype=dtype, generator=g)
        size = (4, 4, 3)
    else:
        values = torch.randn((nnz,), dtype=dtype, generator=g)
        size = (4, 4)
    return indices.to(flag_gems.device), values.to(flag_gems.device), size


def _assert_sparse_equal(res, ref):
    assert res.layout == torch.sparse_coo
    assert ref.layout == torch.sparse_coo
    assert res.shape == ref.shape
    utils.gems_assert_equal(res._indices(), ref._indices())
    utils.gems_assert_equal(res._values(), ref._values())


@pytest.mark.sparse_coo_tensor_unsafe
@pytest.mark.parametrize("nnz", [1, 4])
@pytest.mark.parametrize("dtype", COO_UNSAFE_DTYPES)
def test_sparse_coo_tensor_unsafe(nnz, dtype):
    indices, values, size = _make_components(nnz, dtype)

    ref = torch.ops.aten._sparse_coo_tensor_unsafe(
        utils.to_reference(indices),
        utils.to_reference(values),
        size,
        device=utils.to_reference(values).device,
    )
    res = flag_gems.sparse_coo_tensor_unsafe(indices, values, size)

    _assert_sparse_equal(res, ref)
    # the payload aliases the input values storage
    assert res._values().data_ptr() == values.data_ptr()


@pytest.mark.sparse_coo_tensor_unsafe
@pytest.mark.parametrize("nnz", [1, 4])
@pytest.mark.parametrize("dtype", COO_UNSAFE_DTYPES)
def test_sparse_coo_tensor_unsafe_hybrid(nnz, dtype):
    """A values block carrying a dense dim assembles identically."""
    indices, values, size = _make_components(nnz, dtype, hybrid=True)

    ref = torch.ops.aten._sparse_coo_tensor_unsafe(
        utils.to_reference(indices),
        utils.to_reference(values),
        size,
        device=utils.to_reference(values).device,
    )
    res = flag_gems.sparse_coo_tensor_unsafe(indices, values, size)

    _assert_sparse_equal(res, ref)


@pytest.mark.sparse_coo_tensor_unsafe
@pytest.mark.parametrize("dtype", COO_UNSAFE_DTYPES)
def test_sparse_coo_tensor_unsafe_skips_index_validation(dtype):
    """The unsafe constructor accepts unsorted, duplicated and out-of-range
    indices the validated one rejects; the payload dtype wins over dtype=."""
    indices, values, size = _make_components(4, dtype, unsafe_indices=True)

    ref = torch.ops.aten._sparse_coo_tensor_unsafe(
        utils.to_reference(indices),
        utils.to_reference(values),
        size,
        dtype=dtype,
        device=utils.to_reference(values).device,
    )
    res = flag_gems.sparse_coo_tensor_unsafe(indices, values, size, dtype=dtype)

    _assert_sparse_equal(res, ref)
    # the native unsafe constructor ignores a dtype that differs from the payload
    assert res.dtype == values.dtype


@pytest.mark.sparse_coo_tensor_unsafe
def test_sparse_coo_tensor_unsafe_uncoalesced_flag():
    """Duplicate coordinates keep the uncoalesced flag, like the native op."""
    indices = torch.tensor([[0, 0], [0, 0]], dtype=torch.int64, device=flag_gems.device)
    values = torch.randn(2, device=flag_gems.device)
    ref = torch.ops.aten._sparse_coo_tensor_unsafe(
        utils.to_reference(indices),
        utils.to_reference(values),
        (4, 4),
        device=utils.to_reference(values).device,
    )
    res = flag_gems.sparse_coo_tensor_unsafe(indices, values, (4, 4))
    assert res.is_coalesced() == ref.is_coalesced() == False  # noqa: E712
    _assert_sparse_equal(res, ref)
