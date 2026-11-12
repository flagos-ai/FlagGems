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


def _make_sparse_coo(shape, dtype, density=0.3):
    """Build a coalesced sparse COO tensor on the device."""
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.rand(shape, device=flag_gems.device) < density
    return (x * mask).to_sparse().coalesce()


def _sparse_broadcast_to_ref(x, size):
    """CPU reference for ``_sparse_broadcast_to``."""
    return torch.ops.aten._sparse_broadcast_to(x.cpu(), list(size))


def _assert_sparse_close(res_out, ref_out, dtype):
    assert res_out.shape == ref_out.shape
    assert res_out.layout == torch.sparse_coo
    assert ref_out.layout == torch.sparse_coo

    res_dense = res_out.to_dense()
    ref_dense = ref_out.to_dense()
    if not utils.TO_CPU:
        ref_dense = ref_dense.to(flag_gems.device)
    utils.gems_assert_close(res_dense, ref_dense, dtype)


@pytest.mark.sparse_broadcast_to
@pytest.mark.parametrize("shape", [(256,), (20, 320, 15), (16, 128, 64, 60)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_broadcast_to_identity(shape, dtype):
    """Target shape equals the source shape (no expansion)."""
    inp = _make_sparse_coo(shape, dtype)

    ref_out = _sparse_broadcast_to_ref(inp, shape)
    res_out = flag_gems.sparse_broadcast_to(inp, shape)

    _assert_sparse_close(res_out, ref_out, dtype)


@pytest.mark.sparse_broadcast_to
@pytest.mark.parametrize("shape", [(256,), (20, 320, 15), (16, 128, 64, 60)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_broadcast_to_prepend(shape, dtype):
    """Prepend a new sparse dimension of size 4."""
    inp = _make_sparse_coo(shape, dtype)
    target = (4,) + shape

    ref_out = _sparse_broadcast_to_ref(inp, target)
    res_out = flag_gems.sparse_broadcast_to(inp, target)

    _assert_sparse_close(res_out, ref_out, dtype)


@pytest.mark.sparse_broadcast_to
@pytest.mark.parametrize(
    "src_shape,target_shape", [((1,), (5,)), ((1, 3), (5, 3)), ((2, 1, 4), (2, 6, 4))]
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_broadcast_to_expand_sparse_dim(src_shape, target_shape, dtype):
    """Expand a size-1 sparse dimension to a larger size."""
    inp = _make_sparse_coo(src_shape, dtype)

    ref_out = _sparse_broadcast_to_ref(inp, target_shape)
    res_out = flag_gems.sparse_broadcast_to(inp, target_shape)

    _assert_sparse_close(res_out, ref_out, dtype)
