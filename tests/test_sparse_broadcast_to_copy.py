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


def _make_sparse_coo(shape, dtype, density=0.3, coalesced=True):
    """Build a sparse COO tensor on the device."""
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.rand(shape, device=flag_gems.device) < density
    out = (x * mask).to_sparse()
    return out.coalesce() if coalesced else out


def _sparse_broadcast_to_ref(x, size):
    """Reference: the broadcast semantics are shared with
    ``aten::_sparse_broadcast_to``; the copy variant has no native sparse
    kernel, so the broadcast oracle is the exact expected output."""
    return torch.ops.aten._sparse_broadcast_to(utils.to_reference(x), list(size))


def _assert_sparse_equal(res_out, ref_out):
    assert res_out.shape == ref_out.shape
    assert res_out.layout == torch.sparse_coo
    assert ref_out.layout == torch.sparse_coo
    assert res_out.is_coalesced() == ref_out.is_coalesced()
    assert res_out._nnz() == ref_out._nnz()
    utils.gems_assert_equal(res_out._indices(), ref_out._indices())
    utils.gems_assert_equal(res_out._values(), ref_out._values())


@pytest.mark.sparse_broadcast_to_copy
@pytest.mark.parametrize("shape", [(256,), (20, 320, 15), (16, 128, 64, 60)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_broadcast_to_copy_identity(shape, dtype):
    """Target shape equals the source shape (no expansion)."""
    inp = _make_sparse_coo(shape, dtype)

    ref_out = _sparse_broadcast_to_ref(inp, shape)
    res_out = flag_gems.sparse_broadcast_to_copy(inp, shape)

    _assert_sparse_equal(res_out, ref_out)


@pytest.mark.sparse_broadcast_to_copy
@pytest.mark.parametrize("shape", [(256,), (20, 320, 15), (16, 128, 64, 60)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_broadcast_to_copy_prepend(shape, dtype):
    """Prepend a new sparse dimension of size 4."""
    inp = _make_sparse_coo(shape, dtype)
    target = (4,) + shape

    ref_out = _sparse_broadcast_to_ref(inp, target)
    res_out = flag_gems.sparse_broadcast_to_copy(inp, target)

    _assert_sparse_equal(res_out, ref_out)


@pytest.mark.sparse_broadcast_to_copy
@pytest.mark.parametrize(
    "src_shape,target_shape", [((1,), (5,)), ((1, 3), (5, 3)), ((2, 1, 4), (2, 6, 4))]
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_broadcast_to_copy_expand_sparse_dim(src_shape, target_shape, dtype):
    """Expand a size-1 sparse dimension to a larger size."""
    inp = _make_sparse_coo(src_shape, dtype)

    ref_out = _sparse_broadcast_to_ref(inp, target_shape)
    res_out = flag_gems.sparse_broadcast_to_copy(inp, target_shape)

    _assert_sparse_equal(res_out, ref_out)


@pytest.mark.sparse_broadcast_to_copy
@pytest.mark.parametrize("shape", [(256,), (20, 320, 15)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_broadcast_to_copy_uncoalesced(shape, dtype):
    """An uncoalesced input keeps its stored order in the broadcast copy."""
    inp = _make_sparse_coo(shape, dtype, coalesced=False)

    ref_out = _sparse_broadcast_to_ref(inp, (4,) + shape)
    res_out = flag_gems.sparse_broadcast_to_copy(inp, (4,) + shape)

    _assert_sparse_equal(res_out, ref_out)


@pytest.mark.sparse_broadcast_to_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_broadcast_to_copy_materialises_fresh_storage(dtype):
    """The copy variant never aliases the input's indices or values."""
    inp = _make_sparse_coo((20, 320, 15), dtype)
    snapshot = (inp._indices().clone(), inp._values().clone())

    res_out = flag_gems.sparse_broadcast_to_copy(inp, (4,) + (20, 320, 15))

    assert res_out._values().data_ptr() != inp._values().data_ptr()
    assert res_out._indices().data_ptr() != inp._indices().data_ptr()
    # the source tensor is left untouched
    utils.gems_assert_equal(inp._indices(), utils.to_reference(snapshot[0]))
    utils.gems_assert_equal(inp._values(), utils.to_reference(snapshot[1]))


@pytest.mark.sparse_broadcast_to_copy
def test_sparse_broadcast_to_copy_out():
    """The ``out=`` form writes into the provided sparse COO tensor."""
    inp = _make_sparse_coo((20, 320, 15), torch.float32)
    target = (4, 20, 320, 15)

    ref_out = torch.ops.aten._sparse_broadcast_to(utils.to_reference(inp), list(target))
    template = torch.ops.aten._sparse_broadcast_to(
        utils.to_reference(inp), list(target)
    )
    out = torch.empty_like(template)
    res_out = flag_gems.sparse_broadcast_to_copy(inp, target, out=out)

    assert res_out is out
    _assert_sparse_equal(res_out, ref_out)
