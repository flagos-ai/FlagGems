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

pytestmark = [pytest.mark.sparse_resize_and_clear_]

# ATen has shipped two wordings for rejecting len(size) != sparse_dim + dense_dim.
# Both name the two dimension counts, so accept either instead of pinning one build.
_DIM_MISMATCH_MSG = (
    r"number of dimensions must be sparse_dim|len\(size\) == sparse_dim \+ dense_dim"
)

# (target size, sparse_dim, dense_dim). The target metadata covers growing,
# shrinking, changing the sparse/dense split, and turning a plain COO tensor
# into a hybrid one; the reference accepts all of them because clearing makes
# the tensor empty before the resize.
CLEAR_CASES = [
    ((4, 5), 2, 0),
    ((2, 3), 2, 0),
    ((4, 5), 1, 1),
    ((4, 5, 6), 2, 1),
]

EMPTY_CASES = [
    ((2, 3), 2, 0),
    ((4, 5), 1, 1),
    ((6, 7, 8), 2, 1),
    ((5,), 1, 0),
]


def _ref(inp, size, sparse_dim, dense_dim):
    return torch.ops.aten.sparse_resize_and_clear_.default(
        inp, size, sparse_dim, dense_dim
    )


def _make_coo(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    return x.to_sparse()


def _make_uncoalesced(shape, dtype):
    indices = torch.tensor([[0, 0, 1], [1, 1, 2]], device=flag_gems.device)
    values = torch.randn(3, dtype=dtype, device=flag_gems.device)
    out = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    out._coalesced_(False)
    return out


def _assert_same_empty_state(res, ref):
    """Compare the observable end state against the ATen reference."""
    assert res.shape == ref.shape
    assert res.layout == ref.layout == torch.sparse_coo
    assert res.sparse_dim() == ref.sparse_dim()
    assert res.dense_dim() == ref.dense_dim()
    assert res._nnz() == ref._nnz() == 0
    assert res.is_coalesced() == ref.is_coalesced()
    assert res.dtype == ref.dtype
    assert res.device.type == flag_gems.device
    assert tuple(res._indices().shape) == tuple(ref._indices().shape)
    assert tuple(res._values().shape) == tuple(ref._values().shape)
    assert res._indices().dtype == ref._indices().dtype
    assert res._values().dtype == ref._values().dtype
    assert res._indices().numel() == 0
    assert res._values().numel() == 0


@pytest.mark.sparse_resize_and_clear_
@pytest.mark.parametrize("size,sparse_dim,dense_dim", CLEAR_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_resize_and_clear_nonempty(size, sparse_dim, dense_dim, dtype):
    # No version-counter assertion: the counter is bumped by the dispatcher's
    # ADInplaceOrView key above the backend, so it is exercised by a real
    # torch.ops.aten call rather than by calling the op function directly.
    inp = _make_coo((2, 3), dtype)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _ref(ref_inp, size, sparse_dim, dense_dim)
    res_out = flag_gems.sparse_resize_and_clear_(inp, size, sparse_dim, dense_dim)

    assert ref_out is ref_inp
    assert res_out is inp, "in-place op must return the same object"
    _assert_same_empty_state(res_out, ref_out)


@pytest.mark.sparse_resize_and_clear_
@pytest.mark.parametrize("size,sparse_dim,dense_dim", EMPTY_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_resize_and_clear_empty(size, sparse_dim, dense_dim, dtype):
    inp = torch.sparse_coo_tensor(size=(2, 3), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _ref(ref_inp, size, sparse_dim, dense_dim)
    res_out = flag_gems.sparse_resize_and_clear_(inp, size, sparse_dim, dense_dim)

    assert res_out is inp
    _assert_same_empty_state(res_out, ref_out)


@pytest.mark.sparse_resize_and_clear_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_resize_and_clear_hybrid_input(dtype):
    # A non-empty hybrid tensor (sparse_dim=2, dense_dim=1) is cleared like a
    # plain COO one and may adopt a different size and dense extent.
    inp = torch.randn(2, 3, 4, dtype=dtype, device=flag_gems.device).to_sparse(2)
    assert (inp.sparse_dim(), inp.dense_dim()) == (2, 1)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _ref(ref_inp, (5, 6, 7), 2, 1)
    res_out = flag_gems.sparse_resize_and_clear_(inp, (5, 6, 7), 2, 1)

    assert res_out is inp
    _assert_same_empty_state(res_out, ref_out)


@pytest.mark.sparse_resize_and_clear_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_resize_and_clear_uncoalesced_input(dtype):
    inp = _make_uncoalesced((2, 3), dtype)
    assert not inp.is_coalesced()
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _ref(ref_inp, (4, 5), 2, 0)
    res_out = flag_gems.sparse_resize_and_clear_(inp, (4, 5), 2, 0)

    assert res_out is inp
    _assert_same_empty_state(res_out, ref_out)


@pytest.mark.sparse_resize_and_clear_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_resize_and_clear_keeps_old_buffers_intact(dtype):
    # The reference replaces the index/value storage with fresh empty buffers;
    # it does not zero the discarded ones, so aliases held before the call
    # still expose the old entries. The implementation must not clear them.
    inp = _make_coo((2, 3), dtype)
    res_values = inp._values().clone()
    res_indices = inp._indices().clone()
    alias_values = inp._values()
    alias_indices = inp._indices()

    ref_inp = utils.to_reference(inp.clone())
    ref_alias_values = ref_inp._values()
    ref_alias_indices = ref_inp._indices()

    _ref(ref_inp, (4, 5), 2, 0)
    flag_gems.sparse_resize_and_clear_(inp, (4, 5), 2, 0)

    # These two compare the device tensor against a snapshot of itself, so they
    # must not go through gems_assert_equal: that helper expects its second
    # argument to be the CPU reference and asserts so under --ref=cpu.
    assert torch.equal(alias_values, res_values)
    assert torch.equal(alias_indices, res_indices)
    utils.gems_assert_equal(alias_values, ref_alias_values)
    utils.gems_assert_equal(alias_indices, ref_alias_indices)


@pytest.mark.sparse_resize_and_clear_
@pytest.mark.parametrize("empty", [False, True])
def test_sparse_resize_and_clear_size_mismatch_raises(empty):
    if empty:
        inp = torch.sparse_coo_tensor(
            size=(2, 3), dtype=torch.float32, device=flag_gems.device
        )
    else:
        inp = _make_coo((2, 3), torch.float32)
    ref_inp = utils.to_reference(inp.clone())

    # The contract is "rejected because len(size) != sparse_dim + dense_dim",
    # not one exact sentence: ATen has shipped both of these wordings
    #   torch 2.8: "number of dimensions must be sparse_dim (2) + dense_dim (1), but got 2"
    #   later:     "'len(size) == sparse_dim + dense_dim' is not satisfied: ..."
    # Pinning one string makes the test fail on whichever build says the other.
    with pytest.raises(RuntimeError, match=_DIM_MISMATCH_MSG):
        _ref(ref_inp, (4, 5), 2, 1)
    with pytest.raises(RuntimeError, match=_DIM_MISMATCH_MSG):
        flag_gems.sparse_resize_and_clear_(inp, (4, 5), 2, 1)


@pytest.mark.sparse_resize_and_clear_
def test_sparse_resize_and_clear_non_coo_rejected():
    dense = torch.zeros(3, 4, device=flag_gems.device)
    csr = torch.randn(4, 4, device=flag_gems.device).to_sparse_csr()
    for inp in (dense, csr):
        with pytest.raises(NotImplementedError):
            _ref(utils.to_reference(inp), (4, 5), 2, 0)
        with pytest.raises(NotImplementedError):
            flag_gems.sparse_resize_and_clear_(inp, (4, 5), 2, 0)


@pytest.mark.sparse_resize_and_clear_
def test_sparse_resize_and_clear_leaf_requires_grad_rejected():
    inp = _make_coo((2, 3), torch.float32)
    inp.requires_grad_(True)
    ref_inp = utils.to_reference(inp.detach().clone())
    ref_inp.requires_grad_(True)
    assert ref_inp.is_leaf and inp.is_leaf

    with pytest.raises(RuntimeError, match="a leaf Variable that requires grad"):
        _ref(ref_inp, (4, 5), 2, 0)
    with pytest.raises(RuntimeError, match="a leaf Variable that requires grad"):
        flag_gems.sparse_resize_and_clear_(inp, (4, 5), 2, 0)
