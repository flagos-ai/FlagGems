# Copyright 2026, The FlagOS Contributors.
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

pytestmark = [
    pytest.mark.sparse_csr_tensor,
]


def _make_csr(shape, dtype, density=0.3):
    """Build a compressed sparse row tensor on the device."""
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.rand(shape, device=flag_gems.device) < density
    return (x * mask).to_sparse_csr()


def _sparse_csr_tensor_ref(crow_indices, col_indices, values, size=None, **kwargs):
    """CPU reference for ``sparse_csr_tensor``."""
    args = [crow_indices.cpu(), col_indices.cpu(), values.cpu()]
    if size is not None:
        args.append(list(size))
    return torch.sparse_csr_tensor(*args, **kwargs)


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
    res_out = flag_gems.sparse_csr_tensor(crow_indices, col_indices, values, size=shape)

    _assert_csr_close(res_out, ref_out, dtype)


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("shape", [(256, 256), (512, 128), (1024, 1024)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_csr_tensor_infer_size(shape, dtype):
    """Infer the logical shape from the pointer/index extents (size omitted)."""
    sp = _make_csr(shape, dtype)
    crow_indices = sp.crow_indices()
    col_indices = sp.col_indices()
    values = sp.values()

    ref_out = _sparse_csr_tensor_ref(crow_indices, col_indices, values)
    res_out = flag_gems.sparse_csr_tensor(crow_indices, col_indices, values)

    _assert_csr_close(res_out, ref_out, dtype)


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("shape", [(256, 256), (512, 128)])
@pytest.mark.parametrize("target_dtype", utils.FLOAT_DTYPES)
def test_sparse_csr_tensor_dtype_cast(shape, target_dtype):
    """A requested dtype that differs from the payload is cast by the kernel."""
    sp = _make_csr(shape, torch.float32)
    crow_indices = sp.crow_indices()
    col_indices = sp.col_indices()
    values = sp.values()

    ref_out = _sparse_csr_tensor_ref(
        crow_indices, col_indices, values, shape, dtype=target_dtype
    )
    res_out = flag_gems.sparse_csr_tensor(
        crow_indices, col_indices, values, size=shape, dtype=target_dtype
    )

    assert res_out.dtype == target_dtype
    _assert_csr_close(res_out, ref_out, target_dtype)


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_csr_tensor_explicit_options(dtype):
    """Explicit layout/device kwargs equal to the natural defaults are accepted."""
    shape = (128, 64)
    sp = _make_csr(shape, dtype)
    crow_indices = sp.crow_indices()
    col_indices = sp.col_indices()
    values = sp.values()

    ref_out = _sparse_csr_tensor_ref(
        crow_indices,
        col_indices,
        values,
        shape,
        dtype=dtype,
        layout=torch.sparse_csr,
    )
    res_out = flag_gems.sparse_csr_tensor(
        crow_indices,
        col_indices,
        values,
        size=shape,
        dtype=dtype,
        layout=torch.sparse_csr,
        device=flag_gems.device,
        pin_memory=False,
    )

    assert res_out.device == values.device
    _assert_csr_close(res_out, ref_out, dtype)


@pytest.mark.sparse_csr_tensor
def test_sparse_csr_tensor_empty_with_cast():
    """An empty payload skips the cast kernel but still honours the dtype."""
    x = torch.empty((0, 4), dtype=torch.float32, device=flag_gems.device)
    sp = x.to_sparse_csr()

    res_out = flag_gems.sparse_csr_tensor(
        sp.crow_indices(),
        sp.col_indices(),
        sp.values(),
        size=(0, 4),
        dtype=torch.float16,
    )

    assert tuple(res_out.shape) == (0, 4)
    assert res_out.layout == torch.sparse_csr
    assert res_out.dtype == torch.float16


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_csr_tensor_rejects_wrong_layout(dtype):
    """An explicit non-CSR layout request is rejected, as in the native op."""
    sp = _make_csr((256, 256), dtype)
    crow_indices = sp.crow_indices()
    col_indices = sp.col_indices()
    values = sp.values()

    with pytest.raises(RuntimeError):
        flag_gems.sparse_csr_tensor(
            crow_indices,
            col_indices,
            values,
            size=(256, 256),
            layout=torch.sparse_csc,
        )
