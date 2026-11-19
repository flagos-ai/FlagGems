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


def _make_csc(shape, dtype, density=0.3):
    """Build a compressed sparse column tensor on the device."""
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.rand(shape, device=flag_gems.device) < density
    return (x * mask).to_sparse(layout=torch.sparse_csc)


def _sparse_csc_tensor_ref(ccol_indices, row_indices, values, size=None):
    """CPU reference for ``sparse_csc_tensor``."""
    if size is None:
        return torch.sparse_csc_tensor(
            ccol_indices.cpu(), row_indices.cpu(), values.cpu()
        )
    return torch.sparse_csc_tensor(
        ccol_indices.cpu(), row_indices.cpu(), values.cpu(), list(size)
    )


def _assert_csc_close(res_out, ref_out, dtype):
    assert res_out.layout == torch.sparse_csc
    assert ref_out.layout == torch.sparse_csc
    assert res_out.shape == ref_out.shape

    res_dense = res_out.to_dense()
    ref_dense = ref_out.to_dense()
    if not utils.TO_CPU:
        ref_dense = ref_dense.to(flag_gems.device)
    utils.gems_assert_close(res_dense, ref_dense, dtype)


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("shape", [(256, 256), (512, 128), (1024, 1024)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_csc_tensor_explicit_size(shape, dtype):
    """Reconstruct a CSC tensor from its components with an explicit size."""
    sp = _make_csc(shape, dtype)
    ccol_indices = sp.ccol_indices()
    row_indices = sp.row_indices()
    values = sp.values()

    ref_out = _sparse_csc_tensor_ref(ccol_indices, row_indices, values, shape)
    res_out = flag_gems.sparse_csc_tensor(
        ccol_indices, row_indices, values, size=shape, dtype=dtype
    )

    _assert_csc_close(res_out, ref_out, dtype)


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("shape", [(256, 256), (512, 128), (1024, 1024)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_csc_tensor_infer_size(shape, dtype):
    """Infer the dense extents from the component buffers (size omitted)."""
    sp = _make_csc(shape, dtype)
    ccol_indices = sp.ccol_indices()
    row_indices = sp.row_indices()
    values = sp.values()

    ref_out = _sparse_csc_tensor_ref(ccol_indices, row_indices, values)
    res_out = flag_gems.sparse_csc_tensor(
        ccol_indices, row_indices, values, dtype=dtype
    )

    _assert_csc_close(res_out, ref_out, dtype)


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_csc_tensor_infer_size_trailing_empty_rows(dtype):
    """Rows beyond the last stored index are dropped, as in the native op."""
    dense = torch.zeros((256, 256), dtype=dtype, device=flag_gems.device)
    dense[:192, :200] = torch.randn(192, 200, dtype=dtype, device=flag_gems.device)
    sp = dense.to_sparse(layout=torch.sparse_csc)
    ccol_indices = sp.ccol_indices()
    row_indices = sp.row_indices()
    values = sp.values()

    ref_out = _sparse_csc_tensor_ref(ccol_indices, row_indices, values)
    res_out = flag_gems.sparse_csc_tensor(
        ccol_indices, row_indices, values, dtype=dtype
    )

    assert res_out.shape == ref_out.shape == (192, 256)
    _assert_csc_close(res_out, ref_out, dtype)


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_csc_tensor_aliases_values(dtype):
    """The result keeps the caller's payload buffer by reference."""
    sp = _make_csc((256, 256), dtype)
    ccol_indices = sp.ccol_indices()
    row_indices = sp.row_indices()
    values = sp.values()

    res_out = flag_gems.sparse_csc_tensor(
        ccol_indices, row_indices, values, size=(256, 256), dtype=dtype
    )

    assert res_out.values().data_ptr() == values.data_ptr()


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_csc_tensor_rejects_dtype_mismatch(dtype):
    """A values/dtype mismatch is rejected, as the ATen contract requires."""
    sp = _make_csc((256, 256), dtype)
    ccol_indices = sp.ccol_indices()
    row_indices = sp.row_indices()
    values = sp.values()

    other = torch.bfloat16 if dtype is not torch.bfloat16 else torch.float32
    with pytest.raises(RuntimeError):
        flag_gems.sparse_csc_tensor(
            ccol_indices, row_indices, values, size=(256, 256), dtype=other
        )


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_csc_tensor_roundtrip_to_csr(dtype):
    """The constructed CSC tensor converts to CSR with identical contents."""
    sp = _make_csc((256, 256), dtype)
    res_out = flag_gems.sparse_csc_tensor(
        sp.ccol_indices(),
        sp.row_indices(),
        sp.values(),
        size=(256, 256),
        dtype=dtype,
    )

    ref_out = _sparse_csc_tensor_ref(
        sp.ccol_indices(), sp.row_indices(), sp.values(), (256, 256)
    )

    res_csr = res_out.to_sparse_csr()
    ref_csr = ref_out.to_sparse_csr()
    assert res_csr.layout == torch.sparse_csr
    assert res_csr.shape == ref_csr.shape
    res_dense = res_csr.to_dense()
    ref_dense = ref_csr.to_dense()
    if not utils.TO_CPU:
        ref_dense = ref_dense.to(flag_gems.device)
    utils.gems_assert_close(res_dense, ref_dense, dtype)
