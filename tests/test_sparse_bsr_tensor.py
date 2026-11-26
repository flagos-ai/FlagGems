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


def _make_bsr(shape, dtype, density=0.3, blocksize=(16, 16)):
    """Build a block-compressed sparse row tensor on the device."""
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.rand(shape, device=flag_gems.device) < density
    return (x * mask).to_sparse_bsr(blocksize)


def _sparse_bsr_tensor_ref(crow_indices, col_indices, values, size=None):
    """CPU reference for ``sparse_bsr_tensor``."""
    if size is None:
        return torch.sparse_bsr_tensor(
            crow_indices.cpu(), col_indices.cpu(), values.cpu()
        )
    return torch.sparse_bsr_tensor(
        crow_indices.cpu(), col_indices.cpu(), values.cpu(), list(size)
    )


def _assert_bsr_close(res_out, ref_out, dtype):
    assert res_out.layout == torch.sparse_bsr
    assert ref_out.layout == torch.sparse_bsr
    assert res_out.shape == ref_out.shape

    res_dense = res_out.to_dense()
    ref_dense = ref_out.to_dense()
    if not utils.TO_CPU:
        ref_dense = ref_dense.to(flag_gems.device)
    utils.gems_assert_close(res_dense, ref_dense, dtype)


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("shape", [(256, 256), (512, 128), (1024, 1024)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_bsr_tensor_explicit_size(shape, dtype):
    """Reconstruct a BSR tensor from its components with an explicit size."""
    sp = _make_bsr(shape, dtype)
    crow_indices = sp.crow_indices()
    col_indices = sp.col_indices()
    values = sp.values()

    ref_out = _sparse_bsr_tensor_ref(crow_indices, col_indices, values, shape)
    res_out = flag_gems.sparse_bsr_tensor(crow_indices, col_indices, values, size=shape)

    _assert_bsr_close(res_out, ref_out, dtype)


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("shape", [(256, 256), (512, 128), (1024, 1024)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_bsr_tensor_infer_size(shape, dtype):
    """Infer the logical shape from the block grid (size omitted)."""
    sp = _make_bsr(shape, dtype)
    crow_indices = sp.crow_indices()
    col_indices = sp.col_indices()
    values = sp.values()

    ref_out = _sparse_bsr_tensor_ref(crow_indices, col_indices, values)
    res_out = flag_gems.sparse_bsr_tensor(crow_indices, col_indices, values)

    _assert_bsr_close(res_out, ref_out, dtype)


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize(
    "shape,blocksize", [((256, 256), (16, 16)), ((512, 128), (32, 32))]
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_bsr_tensor_blocksize(shape, blocksize, dtype):
    """Non-default block extents still round-trip through the constructor."""
    sp = _make_bsr(shape, dtype, blocksize=blocksize)
    crow_indices = sp.crow_indices()
    col_indices = sp.col_indices()
    values = sp.values()

    ref_out = _sparse_bsr_tensor_ref(crow_indices, col_indices, values, shape)
    res_out = flag_gems.sparse_bsr_tensor(crow_indices, col_indices, values, size=shape)

    _assert_bsr_close(res_out, ref_out, dtype)


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_bsr_tensor_rejects_wrong_layout(dtype):
    """An explicit non-BSR layout request is rejected, as in the native op."""
    sp = _make_bsr((256, 256), dtype)
    crow_indices = sp.crow_indices()
    col_indices = sp.col_indices()
    values = sp.values()

    with pytest.raises(RuntimeError):
        flag_gems.sparse_bsr_tensor(
            crow_indices,
            col_indices,
            values,
            size=(256, 256),
            layout=torch.sparse_csc,
        )
