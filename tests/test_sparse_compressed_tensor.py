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


def _make_compressed(shape, dtype, layout, blocksize=(16, 16)):
    """Build a compressed sparse tensor of the requested layout on the device."""
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.rand(shape, device=flag_gems.device) < 0.3
    if layout in (torch.sparse_bsr, torch.sparse_bsc):
        return (x * mask).to_sparse(layout=layout, blocksize=blocksize)
    return (x * mask).to_sparse(layout=layout)


def _compressed_components(sp):
    """Return the (compressed, plain, values) component buffers of ``sp``."""
    if sp.layout in (torch.sparse_csr, torch.sparse_bsr):
        return sp.crow_indices(), sp.col_indices(), sp.values()
    return sp.ccol_indices(), sp.row_indices(), sp.values()


def _sparse_compressed_tensor_ref(c_indices, p_indices, values, size, layout):
    """CPU reference for ``sparse_compressed_tensor``."""
    return torch.ops.aten.sparse_compressed_tensor(
        c_indices.cpu(),
        p_indices.cpu(),
        values.cpu(),
        list(size),
        dtype=values.dtype,
        layout=layout,
    )


def _assert_compressed_close(res_out, ref_out, dtype):
    assert res_out.layout == ref_out.layout
    assert res_out.shape == ref_out.shape

    res_dense = res_out.to_dense()
    ref_dense = ref_out.to_dense()
    if not utils.TO_CPU:
        ref_dense = ref_dense.to(flag_gems.device)
    utils.gems_assert_close(res_dense, ref_dense, dtype)


@pytest.mark.sparse_compressed_tensor
@pytest.mark.parametrize(
    "layout", [torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc]
)
@pytest.mark.parametrize("shape", [(256, 256), (512, 128), (1024, 1024)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_compressed_tensor_explicit_size(shape, dtype, layout):
    """Reconstruct a compressed sparse tensor from its components, all layouts."""
    blocksize = (16, 16) if layout in (torch.sparse_bsr, torch.sparse_bsc) else None
    sp = _make_compressed(shape, dtype, layout, blocksize or (16, 16))
    c_indices, p_indices, values = _compressed_components(sp)

    ref_out = _sparse_compressed_tensor_ref(c_indices, p_indices, values, shape, layout)
    res_out = flag_gems.sparse_compressed_tensor(
        c_indices, p_indices, values, size=shape, dtype=dtype, layout=layout
    )

    _assert_compressed_close(res_out, ref_out, dtype)


@pytest.mark.sparse_compressed_tensor
@pytest.mark.parametrize(
    "layout", [torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc]
)
@pytest.mark.parametrize("shape", [(256, 256), (512, 128)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_compressed_tensor_infer_size(shape, dtype, layout):
    """Dense extents are inferred from the component buffers (size omitted)."""
    blocksize = (16, 16) if layout in (torch.sparse_bsr, torch.sparse_bsc) else None
    sp = _make_compressed(shape, dtype, layout, blocksize or (16, 16))
    c_indices, p_indices, values = _compressed_components(sp)

    ref_out = torch.ops.aten.sparse_compressed_tensor(
        c_indices.cpu(),
        p_indices.cpu(),
        values.cpu(),
        dtype=values.dtype,
        layout=layout,
    )
    res_out = flag_gems.sparse_compressed_tensor(
        c_indices, p_indices, values, dtype=dtype, layout=layout
    )

    _assert_compressed_close(res_out, ref_out, dtype)


@pytest.mark.sparse_compressed_tensor
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_compressed_tensor_rejects_bad_layout(dtype):
    """A missing or non-compressed layout is rejected, as in the native op."""
    sp = _make_compressed((256, 256), dtype, torch.sparse_csr)
    c_indices, p_indices, values = _compressed_components(sp)

    with pytest.raises(RuntimeError):
        flag_gems.sparse_compressed_tensor(
            c_indices, p_indices, values, size=(256, 256), layout=torch.sparse_coo
        )
