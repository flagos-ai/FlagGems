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

# ``dtype`` casting is the only path that enqueues the cast kernel, so cover
# both a widening and a narrowing-free conversion.
_DTYPE_CAST_PAIRS = [(torch.float16, torch.float32)]
if utils.bf16_is_supported:
    _DTYPE_CAST_PAIRS.append((torch.bfloat16, torch.float32))
if utils.fp64_is_supported:
    _DTYPE_CAST_PAIRS.append((torch.float32, torch.float64))


def _make_csc(shape, dtype, density=0.3):
    """Build a compressed sparse column tensor on the device."""
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.rand(shape, device=flag_gems.device) < density
    return (x * mask).to_sparse_csc()


def _sparse_csc_tensor_ref(ccol_indices, row_indices, values, size=None, dtype=None):
    """CPU reference for ``sparse_csc_tensor``."""
    if size is None:
        return torch.sparse_csc_tensor(
            ccol_indices.cpu(), row_indices.cpu(), values.cpu(), dtype=dtype
        )
    return torch.sparse_csc_tensor(
        ccol_indices.cpu(), row_indices.cpu(), values.cpu(), list(size), dtype=dtype
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
    res_out = flag_gems.sparse_csc_tensor(ccol_indices, row_indices, values, size=shape)

    _assert_csc_close(res_out, ref_out, dtype)


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("shape", [(256, 256), (512, 128), (1024, 1024)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_csc_tensor_infer_size(shape, dtype):
    """Infer the logical shape from the index buffers (size omitted)."""
    sp = _make_csc(shape, dtype)
    ccol_indices = sp.ccol_indices()
    row_indices = sp.row_indices()
    values = sp.values()

    ref_out = _sparse_csc_tensor_ref(ccol_indices, row_indices, values)
    res_out = flag_gems.sparse_csc_tensor(ccol_indices, row_indices, values)

    _assert_csc_close(res_out, ref_out, dtype)


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("shape", [(256, 256), (128, 512)])
@pytest.mark.parametrize("in_dtype,out_dtype", _DTYPE_CAST_PAIRS)
def test_sparse_csc_tensor_dtype_cast(shape, in_dtype, out_dtype):
    """A requested dtype different from values.dtype is applied to values."""
    sp = _make_csc(shape, in_dtype)
    ccol_indices = sp.ccol_indices()
    row_indices = sp.row_indices()
    values = sp.values()

    ref_out = _sparse_csc_tensor_ref(
        ccol_indices, row_indices, values, shape, dtype=out_dtype
    )
    res_out = flag_gems.sparse_csc_tensor(
        ccol_indices, row_indices, values, size=shape, dtype=out_dtype
    )

    assert res_out.dtype == out_dtype
    _assert_csc_close(res_out, ref_out, out_dtype)


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_csc_tensor_aliases_values(dtype):
    """The matching dtype/size path keeps the caller's value buffer."""
    sp = _make_csc((256, 256), dtype)
    ccol_indices = sp.ccol_indices()
    row_indices = sp.row_indices()
    values = sp.values()

    res_out = flag_gems.sparse_csc_tensor(
        ccol_indices, row_indices, values, size=(256, 256)
    )

    assert res_out.values().data_ptr() == values.data_ptr()


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_csc_tensor_rejects_wrong_layout(dtype):
    """An explicit non-CSC layout request is rejected, as in the native op."""
    sp = _make_csc((256, 256), dtype)
    ccol_indices = sp.ccol_indices()
    row_indices = sp.row_indices()
    values = sp.values()

    with pytest.raises(RuntimeError):
        flag_gems.sparse_csc_tensor(
            ccol_indices,
            row_indices,
            values,
            size=(256, 256),
            layout=torch.sparse_csr,
        )
