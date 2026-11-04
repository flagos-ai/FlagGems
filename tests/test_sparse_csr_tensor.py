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


def _make_csr(shape, dtype, density=0.3):
    """Build a compressed sparse row tensor on the device."""
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.rand(shape, device=flag_gems.device) < density
    return (x * mask).to_sparse(layout=torch.sparse_csr)


def _sparse_csr_tensor_ref(crow_indices, col_indices, values, size=None):
    """CPU reference for ``sparse_csr_tensor``."""
    if size is None:
        return torch.sparse_csr_tensor(
            crow_indices.cpu(), col_indices.cpu(), values.cpu()
        )
    return torch.sparse_csr_tensor(
        crow_indices.cpu(), col_indices.cpu(), values.cpu(), list(size)
    )


def _assert_csr_close(res_out, ref_out, dtype):
    assert res_out.layout == torch.sparse_csr
    assert ref_out.layout == torch.sparse_csr
    assert res_out.shape == ref_out.shape

    res_dense = res_out.to_dense()
    ref_dense = ref_out.to_dense()
    if not utils.TO_CPU:
        ref_dense = ref_dense.to(flag_gems.device)
    utils.gems_assert_close(res_dense, ref_dense, dtype)


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("shape", [(256, 256), (512, 128), (1024, 1024)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_csr_tensor_explicit_size(shape, dtype):
    """Reconstruct a CSR tensor from its components with an explicit size."""
    sp = _make_csr(shape, dtype)
    crow_indices = sp.crow_indices()
    col_indices = sp.col_indices()
    values = sp.values()

    ref_out = _sparse_csr_tensor_ref(crow_indices, col_indices, values, shape)
    res_out = flag_gems.sparse_csr_tensor(
        crow_indices, col_indices, values, size=shape, dtype=dtype
    )

    _assert_csr_close(res_out, ref_out, dtype)


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("shape", [(256, 256), (512, 128), (1024, 1024)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_csr_tensor_infer_size(shape, dtype):
    """Infer the dense extents from the component buffers (size omitted)."""
    sp = _make_csr(shape, dtype)
    crow_indices = sp.crow_indices()
    col_indices = sp.col_indices()
    values = sp.values()

    ref_out = _sparse_csr_tensor_ref(crow_indices, col_indices, values)
    res_out = flag_gems.sparse_csr_tensor(
        crow_indices, col_indices, values, dtype=dtype
    )

    _assert_csr_close(res_out, ref_out, dtype)


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_csr_tensor_infer_size_trailing_empty_columns(dtype):
    """Columns beyond the last stored index are dropped, as in the native op."""
    sp = _make_csr((256, 256), dtype, density=0.0)
    sp.values()  # all-empty matrix has a valid (trivial) CSR form
    dense = torch.zeros((256, 256), dtype=dtype, device=flag_gems.device)
    dense[:200, :192] = torch.randn(200, 192, dtype=dtype, device=flag_gems.device)
    sp = dense.to_sparse(layout=torch.sparse_csr)
    crow_indices = sp.crow_indices()
    col_indices = sp.col_indices()
    values = sp.values()

    ref_out = _sparse_csr_tensor_ref(crow_indices, col_indices, values)
    res_out = flag_gems.sparse_csr_tensor(
        crow_indices, col_indices, values, dtype=dtype
    )

    assert res_out.shape == ref_out.shape == (256, 192)
    _assert_csr_close(res_out, ref_out, dtype)


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_csr_tensor_aliases_values(dtype):
    """The result keeps the caller's payload buffer by reference."""
    sp = _make_csr((256, 256), dtype)
    crow_indices = sp.crow_indices()
    col_indices = sp.col_indices()
    values = sp.values()

    res_out = flag_gems.sparse_csr_tensor(
        crow_indices, col_indices, values, size=(256, 256), dtype=dtype
    )

    assert res_out.values().data_ptr() == values.data_ptr()


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_csr_tensor_rejects_dtype_mismatch(dtype):
    """A values/dtype mismatch is rejected, as the ATen contract requires."""
    sp = _make_csr((256, 256), dtype)
    crow_indices = sp.crow_indices()
    col_indices = sp.col_indices()
    values = sp.values()

    other = torch.bfloat16 if dtype is not torch.bfloat16 else torch.float32
    with pytest.raises(RuntimeError):
        flag_gems.sparse_csr_tensor(
            crow_indices, col_indices, values, size=(256, 256), dtype=other
        )


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_csr_tensor_autograd_flows_into_values(dtype):
    """Gradients reach the caller's values through the constructed tensor."""
    sp = _make_csr((128, 128), dtype)
    crow_indices = sp.crow_indices()
    col_indices = sp.col_indices()

    values = sp.values().detach().requires_grad_(True)
    res_out = flag_gems.sparse_csr_tensor(
        crow_indices, col_indices, values, size=(128, 128), dtype=dtype
    )
    (res_out.values() * 2).sum().backward()

    ref_values = sp.values().detach().requires_grad_(True)
    ref_out = torch.sparse_csr_tensor(
        crow_indices.cpu(),
        col_indices.cpu(),
        ref_values.cpu(),
        size=(128, 128),
        dtype=dtype,
    )
    ref_out.values().sum().backward()

    assert values.grad is not None
    expected = (2 * torch.ones_like(ref_values)).cpu()
    assert torch.allclose(values.grad.cpu(), expected)
