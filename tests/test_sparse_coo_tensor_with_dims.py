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


def _native(sparse_dim, dense_dim, size, dtype, device):
    return torch.ops.aten._sparse_coo_tensor_with_dims(
        sparse_dim, dense_dim, size, dtype=dtype, layout=torch.sparse_coo, device=device
    )


def _assert_empty_coo_close(res_out, ref_out):
    assert res_out.layout == torch.sparse_coo
    assert ref_out.layout == torch.sparse_coo
    assert res_out.shape == ref_out.shape
    assert res_out.sparse_dim() == ref_out.sparse_dim()
    assert res_out.dense_dim() == ref_out.dense_dim()
    assert res_out.dtype == ref_out.dtype
    assert res_out._nnz() == ref_out._nnz() == 0
    assert res_out.is_coalesced() == ref_out.is_coalesced()
    res_dense = res_out.to_dense()
    ref_dense = ref_out.to_dense()
    if not utils.TO_CPU:
        ref_dense = ref_dense.to(flag_gems.device)
    utils.gems_assert_close(res_dense, ref_dense, ref_out.dtype)


@pytest.mark.sparse_coo_tensor_with_dims
@pytest.mark.parametrize(
    "sparse_dim,dense_dim,size",
    [
        (2, 0, [3, 4]),
        (1, 0, [7]),
        (3, 0, [2, 3, 4]),
        (2, 1, [3, 4, 7]),
        (1, 2, [5, 2, 3]),
        (2, 0, [1, 1]),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_coo_tensor_with_dims(sparse_dim, dense_dim, size, dtype):
    """The allocated descriptor matches ATen for plain and hybrid shapes."""
    ref_out = _native(sparse_dim, dense_dim, size, dtype, "cpu")
    res_out = flag_gems.sparse_coo_tensor_with_dims(
        sparse_dim, dense_dim, size, dtype=dtype, layout=torch.sparse_coo
    )

    _assert_empty_coo_close(res_out, ref_out)


@pytest.mark.sparse_coo_tensor_with_dims
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_sparse_coo_tensor_with_dims_dtypes(dtype):
    """Every supported dtype round-trips through the empty descriptor."""
    ref_out = _native(2, 0, [4, 5], dtype, "cpu")
    res_out = flag_gems.sparse_coo_tensor_with_dims(
        2, 0, [4, 5], dtype=dtype, layout=torch.sparse_coo
    )

    assert res_out.dtype == ref_out.dtype == dtype
    _assert_empty_coo_close(res_out, ref_out)


@pytest.mark.sparse_coo_tensor_with_dims
def test_sparse_coo_tensor_with_dims_default_dtype():
    """Omitting dtype falls back to the default dtype, as ATen does."""
    ref_out = torch.ops.aten._sparse_coo_tensor_with_dims(
        2, 0, [3, 3], layout=torch.sparse_coo
    )
    res_out = flag_gems.sparse_coo_tensor_with_dims(
        2, 0, [3, 3], layout=torch.sparse_coo
    )

    assert res_out.dtype == ref_out.dtype == torch.get_default_dtype()


@pytest.mark.sparse_coo_tensor_with_dims
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_coo_tensor_with_dims_out(dtype):
    """The out form rewrites the caller's tensor and returns that same object."""
    out = _native(2, 0, [4, 4], dtype, flag_gems.device)
    ref_out = _native(2, 0, [4, 4], dtype, "cpu")

    res_out = flag_gems.sparse_coo_tensor_with_dims_out(2, 0, [4, 4], out=out)

    assert res_out is out
    _assert_empty_coo_close(res_out, ref_out)


@pytest.mark.sparse_coo_tensor_with_dims
def test_sparse_coo_tensor_with_dims_rejects_bad_layout():
    """A non-COO layout request is rejected before allocation."""
    with pytest.raises(RuntimeError):
        flag_gems.sparse_coo_tensor_with_dims(2, 0, [3, 4], layout=torch.sparse_csr)


@pytest.mark.sparse_coo_tensor_with_dims
def test_sparse_coo_tensor_with_dims_rejects_size_mismatch():
    """size must carry sparse_dim + dense_dim extents, as ATen requires."""
    with pytest.raises(RuntimeError):
        _native(2, 1, [3, 4], torch.float32, "cpu")
