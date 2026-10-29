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


def _payload(size, sparse_dim, dtype, device, nnz=3, seed=0):
    """Build a deterministic (indices, values) pair for a COO descriptor."""
    g = torch.Generator().manual_seed(seed)
    base = torch.arange(nnz, dtype=torch.int64)
    indices = torch.stack([base % max(int(size[d]), 1) for d in range(sparse_dim)])
    dense_shape = tuple(size[sparse_dim:])
    values = torch.randn((nnz, *dense_shape), dtype=dtype, generator=g).contiguous()
    return indices.to(device), values.to(device)


def _native(
    sparse_dim, dense_dim, size, indices, values, dtype=None, is_coalesced=None
):
    return torch.ops.aten._sparse_coo_tensor_with_dims_and_tensors(
        sparse_dim,
        dense_dim,
        size,
        indices,
        values,
        dtype=dtype,
        layout=torch.sparse_coo,
        device=indices.device,
        is_coalesced=is_coalesced,
    )


def _assert_coo_close(res_out, ref_out):
    assert res_out.layout == torch.sparse_coo
    assert ref_out.layout == torch.sparse_coo
    assert res_out.shape == ref_out.shape
    assert res_out.sparse_dim() == ref_out.sparse_dim()
    assert res_out.dense_dim() == ref_out.dense_dim()
    assert res_out.dtype == ref_out.dtype
    assert res_out._nnz() == ref_out._nnz()

    res_dense = res_out.coalesce().to_dense()
    ref_dense = ref_out.coalesce().to_dense()
    if not utils.TO_CPU:
        ref_dense = ref_dense.to(flag_gems.device)
    utils.gems_assert_close(res_dense, ref_dense, ref_out.dtype)


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
@pytest.mark.parametrize(
    "sparse_dim,dense_dim,size",
    [
        (2, 0, [4, 4]),
        (1, 0, [8]),
        (3, 0, [2, 3, 4]),
        (2, 1, [4, 4, 7]),
        (1, 2, [5, 2, 3]),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_coo_tensor_with_dims_and_tensors(sparse_dim, dense_dim, size, dtype):
    """The assembled descriptor matches ATen for plain and hybrid shapes."""
    ref_indices, ref_values = _payload(size, sparse_dim, dtype, "cpu")
    indices, values = _payload(size, sparse_dim, dtype, flag_gems.device)
    ref_out = _native(sparse_dim, dense_dim, size, ref_indices, ref_values, dtype)
    res_out = flag_gems.sparse_coo_tensor_with_dims_and_tensors(
        sparse_dim,
        dense_dim,
        size,
        indices,
        values,
        dtype=dtype,
        layout=torch.sparse_coo,
    )

    _assert_coo_close(res_out, ref_out)


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
@pytest.mark.parametrize("nnz", [0, 1, 5])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_coo_tensor_with_dims_and_tensors_nnz(nnz, dtype):
    """Empty, single-entry and multi-entry payloads are all forwarded verbatim."""
    ref_indices, ref_values = _payload([4, 4], 2, dtype, "cpu", nnz=nnz)
    indices, values = _payload([4, 4], 2, dtype, flag_gems.device, nnz=nnz)
    ref_out = _native(2, 0, [4, 4], ref_indices, ref_values, dtype)
    res_out = flag_gems.sparse_coo_tensor_with_dims_and_tensors(
        2, 0, [4, 4], indices, values, dtype=dtype
    )

    assert res_out._nnz() == ref_out._nnz() == nnz
    _assert_coo_close(res_out, ref_out)


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
def test_sparse_coo_tensor_with_dims_and_tensors_aliases_payload():
    """The caller's index and value buffers are used directly, with no copy."""
    indices, values = _payload([4, 4], 2, torch.float32, flag_gems.device)
    res_out = flag_gems.sparse_coo_tensor_with_dims_and_tensors(
        2, 0, [4, 4], indices, values, is_coalesced=True
    )

    assert res_out.is_coalesced()
    assert res_out.indices().data_ptr() == indices.data_ptr()
    assert res_out.values().data_ptr() == values.data_ptr()


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
def test_sparse_coo_tensor_with_dims_and_tensors_default_uncoalesced():
    """A freshly assembled descriptor keeps the native uncoalesced default."""
    indices, values = _payload([4, 4], 2, torch.float32, flag_gems.device)
    res_out = flag_gems.sparse_coo_tensor_with_dims_and_tensors(
        2, 0, [4, 4], indices, values
    )
    ref_out = _native(2, 0, [4, 4], indices.cpu(), values.cpu())

    assert res_out.is_coalesced() == ref_out.is_coalesced() is False


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
@pytest.mark.parametrize("is_coalesced", [True, False])
def test_sparse_coo_tensor_with_dims_and_tensors_forwards_coalesced(is_coalesced):
    """is_coalesced is recorded verbatim without reordering the payload."""
    indices, values = _payload([4, 4], 2, torch.float32, flag_gems.device)
    res_out = flag_gems.sparse_coo_tensor_with_dims_and_tensors(
        2, 0, [4, 4], indices, values, is_coalesced=is_coalesced
    )

    assert res_out.is_coalesced() is is_coalesced


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_coo_tensor_with_dims_and_tensors_out(dtype):
    """The out form rewrites the caller's tensor and returns that same object."""
    ref_indices, ref_values = _payload([4, 4], 2, dtype, "cpu")
    indices, values = _payload([4, 4], 2, dtype, flag_gems.device)
    out = _native(2, 0, [4, 4], indices, values, dtype)
    ref_out = _native(2, 0, [4, 4], ref_indices, ref_values, dtype)

    res_out = flag_gems.sparse_coo_tensor_with_dims_and_tensors_out(
        2, 0, [4, 4], indices, values, out=out
    )

    assert res_out is out
    _assert_coo_close(res_out, ref_out)


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
def test_sparse_coo_tensor_with_dims_and_tensors_registration_names():
    implementations = (
        flag_gems.sparse_coo_tensor_with_dims_and_tensors,
        flag_gems.sparse_coo_tensor_with_dims_and_tensors_out,
    )
    names = [
        entry[0] for entry in flag_gems._FULL_CONFIG if entry[1] in implementations
    ]
    assert names == [
        "_sparse_coo_tensor_with_dims_and_tensors",
        "_sparse_coo_tensor_with_dims_and_tensors.out",
    ]


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
def test_sparse_coo_tensor_with_dims_and_tensors_rejects_bad_layout():
    """A non-COO layout request is rejected before assembling anything."""
    indices, values = _payload([4, 4], 2, torch.float32, flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.sparse_coo_tensor_with_dims_and_tensors(
            2, 0, [4, 4], indices, values, layout=torch.sparse_csr
        )
