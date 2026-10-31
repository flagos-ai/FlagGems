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

WITH_DIMS_DTYPES = utils.FLOAT_DTYPES

INDEX_DTYPES = [torch.int64] + ([torch.int32] if utils.int64_is_supported else [])


def _expected_buffer_shapes(nnz, dense_dim, size, blocksize, layout):
    """The buffer geometry the constructor allocates for the layout."""
    blocked = layout in (torch.sparse_bsr, torch.sparse_bsc)
    n_comp = len(size) - dense_dim - (2 if blocked else 1)
    compressed_len = (nnz // (blocksize[1] if blocked else 1) or 0) + 1
    # the native geometry: one compressed-index slot per stored batch plus the
    # final boundary
    if blocked:
        compressed_len = n_comp + 1
    else:
        compressed_len = len(size) - dense_dim - 1 + 1
    compressed_len = (n_comp + 1) if blocked else (n_comp + 1)
    dense_tail = tuple(size[len(size) - dense_dim :]) if dense_dim else ()
    if blocked:
        values_shape = (nnz,) + tuple(blocksize) + dense_tail
    else:
        values_shape = (nnz,) + dense_tail
    return compressed_len, nnz, values_shape


@pytest.mark.sparse_compressed_tensor_with_dims
@pytest.mark.parametrize(
    "layout, blocksize",
    [
        (torch.sparse_csr, []),
        (torch.sparse_csc, []),
        (torch.sparse_bsr, [2, 2]),
        (torch.sparse_bsc, [2, 2]),
    ],
)
@pytest.mark.parametrize("dense_dim", [0, 1])
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", WITH_DIMS_DTYPES)
def test_sparse_compressed_tensor_with_dims(
    layout, blocksize, dense_dim, index_dtype, dtype
):
    """The shell geometry matches the native constructor buffer for buffer."""
    size = [4, 4] + ([2] if dense_dim else [])
    nnz = 4

    ref = torch.ops.aten._sparse_compressed_tensor_with_dims(
        nnz,
        dense_dim,
        size,
        blocksize,
        index_dtype,
        dtype=dtype,
        layout=layout,
        device=flag_gems.device,
    )
    res = flag_gems.sparse_compressed_tensor_with_dims(
        nnz,
        dense_dim,
        size,
        blocksize,
        index_dtype,
        dtype=dtype,
        layout=layout,
        device=flag_gems.device,
    )

    assert res.layout == layout == ref.layout
    assert res.shape == ref.shape
    assert res.dtype == ref.dtype == dtype
    if layout in (torch.sparse_csr, torch.sparse_bsr):
        res_c, ref_c = res.crow_indices(), ref.crow_indices()
        res_p, ref_p = res.col_indices(), ref.col_indices()
    else:
        res_c, ref_c = res.ccol_indices(), ref.ccol_indices()
        res_p, ref_p = res.row_indices(), ref.row_indices()
    assert res_c.shape == ref_c.shape
    assert res_c.dtype == ref_c.dtype == index_dtype
    assert res_p.shape == ref_p.shape
    assert res.values().shape == ref.values().shape


@pytest.mark.sparse_compressed_tensor_with_dims
def test_sparse_compressed_tensor_with_dims_zero_nnz():
    """nnz = 0 keeps the +1 compressed-index slot and empty plain/payload."""
    res = flag_gems.sparse_compressed_tensor_with_dims(
        0,
        0,
        [4, 4],
        [],
        torch.int64,
        dtype=torch.float32,
        layout=torch.sparse_csr,
        device=flag_gems.device,
    )
    assert res.crow_indices().shape == (5,)
    assert res.col_indices().shape == (0,)
    assert res.values().shape == (0,)
    assert res._nnz() == 0


@pytest.mark.sparse_compressed_tensor_with_dims
def test_sparse_compressed_tensor_with_dims_fill_matches_validated():
    """The shell, once its buffers are filled, equals the validated construction."""
    crow = torch.tensor([0, 2, 4, 4, 4], dtype=torch.int64, device=flag_gems.device)
    col = torch.tensor([0, 1, 0, 1], dtype=torch.int64, device=flag_gems.device)
    values = torch.randn(4, device=flag_gems.device)
    size = (4, 4)

    ref = torch.sparse_csr_tensor(crow, col, values, size, device=values.device)
    shell = flag_gems.sparse_compressed_tensor_with_dims(
        4,
        0,
        list(size),
        [],
        torch.int64,
        dtype=torch.float32,
        layout=torch.sparse_csr,
        device=flag_gems.device,
    )
    # the caller fills the shell buffers in place afterwards
    shell.crow_indices().copy_(crow)
    shell.col_indices().copy_(col)
    shell.values().copy_(values)

    assert torch.equal(shell.crow_indices(), ref.crow_indices())
    assert torch.equal(shell.col_indices(), ref.col_indices())
    assert torch.equal(shell.values(), ref.values())
