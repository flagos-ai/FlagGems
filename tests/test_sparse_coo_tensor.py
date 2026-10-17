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


def _make_indices_values(shape, dtype, density=0.3):
    """Build a (indices, values) COO component pair on the device."""
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.rand(shape, device=flag_gems.device) < density
    sp = (x * mask).to_sparse().coalesce()
    return sp.indices(), sp.values()


def _sparse_coo_tensor_ref(indices, values, size):
    """CPU reference for ``sparse_coo_tensor``."""
    return torch.sparse_coo_tensor(indices.cpu(), values.cpu(), list(size))


def _assert_coo_close(res_out, ref_out, dtype):
    assert res_out.layout == torch.sparse_coo
    assert ref_out.layout == torch.sparse_coo
    assert res_out.shape == ref_out.shape

    res_dense = res_out.to_dense()
    ref_dense = ref_out.to_dense()
    if not utils.TO_CPU:
        ref_dense = ref_dense.to(flag_gems.device)
    utils.gems_assert_close(res_dense, ref_dense, dtype)


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("shape", [(256,), (20, 320, 15), (16, 128, 64, 60)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_coo_tensor_explicit_size(shape, dtype):
    """Reconstruct a COO tensor from its components with an explicit size."""
    indices, values = _make_indices_values(shape, dtype)

    ref_out = _sparse_coo_tensor_ref(indices, values, shape)
    res_out = flag_gems.sparse_coo_tensor(indices, values, shape)

    _assert_coo_close(res_out, ref_out, dtype)


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("shape", [(256,), (20, 320, 15)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_coo_tensor_infer_size(shape, dtype):
    """Dense extents are inferred from the stored coordinates (size omitted)."""
    indices, values = _make_indices_values(shape, dtype)

    ref_out = torch.sparse_coo_tensor(indices.cpu(), values.cpu())
    res_out = flag_gems.sparse_coo_tensor(indices, values)

    _assert_coo_close(res_out, ref_out, dtype)


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("shape", [(256, 256), (64, 128)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_coo_tensor_size_only(shape, dtype):
    """The size-only overload builds an all-zero COO tensor."""
    size = list(shape)
    ref_out = torch.sparse_coo_tensor(size, dtype=dtype, device=torch.device("cpu"))
    res_out = flag_gems.sparse_coo_tensor(size, dtype=dtype, device=flag_gems.device)

    _assert_coo_close(res_out, ref_out, dtype)


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_coo_tensor_rejects_wrong_layout(dtype):
    """An explicit non-COO layout request is rejected, as in the native op."""
    indices, values = _make_indices_values((64, 64), dtype)

    with pytest.raises(RuntimeError):
        flag_gems.sparse_coo_tensor(indices, values, (64, 64), layout=torch.sparse_csr)
