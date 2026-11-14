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

"""Correctness tests for ``aten::indices``.

``indices(self)`` returns the ``(sparse_dim, nnz)`` coordinate tensor of a
**coalesced** sparse COO tensor as a zero-copy alias of the internal indices
storage. Unlike the private ``_indices``, the public accessor requires
coalescence: an uncoalesced input is rejected with the native
``RuntimeError`` (message mirrored byte-for-byte). Hybrid tensors keep only
their sparse coordinates, and the stored dtype is promoted to int64.

Only sparse COO layouts are accepted: CSR, CSC, BSR, BSC and strided inputs
are rejected with the native ``RuntimeError`` message, byte-identical. The
values are layout metadata, not computed data, so the suite is equally valid
under ``utils.TO_CPU`` (it passes with ``--ref=cpu``).

This file is shared with the private ``_indices`` accessor (its suite lives
in the same file under the ``underscore_indices`` marker); the module-level
``pytestmark`` below also carries this operator's bare marker so CI marker
derivation selects these cases from either entry point.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# CI derives the pytest marker from the implementation file name via the
# #6359 convention. The bare ``indices`` marker is this operator's own id and
# the private ``_indices`` suite shares this file under ``underscore_indices``;
# declare both so ``pytest -m <marker>`` selects the right cases either way.
pytestmark = [
    pytest.mark.indices,
    pytest.mark.underscore_indices,
]


def _make_coo(nnd, nnz, size, seed=0):
    """Build a coalesced sparse COO tensor with distinct, sorted coordinates."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    numel = 1
    for s in size[:nnd]:
        numel *= s
    flat = torch.randperm(numel, generator=gen)[:nnz]
    indices = torch.empty((nnd, nnz), dtype=torch.int64)
    work = flat.clone()
    for d in reversed(range(nnd)):
        indices[d] = work % size[d]
        work = work // size[d]
    perm = torch.arange(nnz)
    for d in reversed(range(nnd)):
        perm = perm[torch.argsort(indices[d][perm], stable=True)]
    indices = indices[:, perm].contiguous()
    values = torch.randn(nnz, generator=gen)
    # The public accessor requires a coalesced input; construction alone does
    # not set the coalesced flag even when the coordinates are already unique
    # and sorted.
    return torch.sparse_coo_tensor(
        indices.to(flag_gems.device), values.to(flag_gems.device), size
    ).coalesce()


# Sparse COO shapes as (sparse_dim, nnz, size): the accessor reads the
# (sparse_dim, nnz) coordinate buffer, so the case mix covers 1-D to 3-D
# layouts and small-to-large nnz.
COO_SHAPES = [
    (1, 8, (16,)),
    (2, 4, (2, 2)),
    (2, 8, (4, 4)),
    (2, 64, (16, 16)),
    (3, 32, (5, 7, 9)),
]


@pytest.mark.indices
@pytest.mark.parametrize("nnd, nnz, size", COO_SHAPES)
def test_indices(nnd, nnz, size):
    res_inp = _make_coo(nnd, nnz, size, seed=3)
    ref_inp = utils.to_reference(res_inp)

    ref_out = torch.ops.aten.indices(ref_inp)
    res_out = flag_gems.indices(res_inp)

    assert res_out.dtype == torch.int64
    assert res_out.shape == (nnd, nnz)
    utils.gems_assert_equal(res_out, ref_inp._indices())
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.indices
def test_indices_hybrid():
    # A hybrid COO tensor keeps its dense dimensions in values; the coordinate
    # table stays (sparse_dim, nnz).
    res_inp = torch.sparse_coo_tensor(
        torch.tensor([[1, 0], [0, 1]], device=flag_gems.device),
        torch.randn(2, 3, 4, device=flag_gems.device),
        (2, 2, 3, 4),
    ).coalesce()
    res_out = flag_gems.indices(res_inp)
    ref_out = torch.ops.aten.indices(utils.to_reference(res_inp))
    assert res_out.shape == (2, 2)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.indices
def test_indices_empty():
    res_inp = torch.sparse_coo_tensor(
        torch.zeros((2, 0), dtype=torch.int64, device=flag_gems.device),
        torch.zeros((0,), device=flag_gems.device),
        (4, 4),
    ).coalesce()
    res_out = flag_gems.indices(res_inp)
    assert res_out.shape == (2, 0)


@pytest.mark.indices
def test_indices_alias_and_mutation():
    # The schema ``Tensor(a) self -> Tensor(a)`` makes the result an aliasing
    # view: two reads share storage, and mutating the result mutates the
    # sparse tensor's stored indices. A copying implementation breaks both.
    res_inp = _make_coo(2, 8, (4, 4), seed=7)

    first = flag_gems.indices(res_inp)
    second = flag_gems.indices(res_inp)
    assert (
        first.data_ptr() == second.data_ptr()
    ), "indices must return an aliasing view, not a fresh copy"
    assert first.data_ptr() == res_inp._indices().data_ptr()

    first[0, 0] = 77
    assert (
        res_inp._indices()[0, 0].item() == 77
    ), "mutating the returned view must be visible in the sparse tensor"


@pytest.mark.indices
def test_indices_uncoalesced_rejected():
    # The public accessor requires coalescence; the message is mirrored
    # byte-for-byte from the native operator. Build the raw tensor without the
    # coalesce() call so its stored coordinates stay as constructed.
    gen = torch.Generator(device="cpu").manual_seed(5)
    flat = torch.randperm(16, generator=gen)[:8]
    indices = torch.stack([flat % 4, flat // 4]).contiguous()
    res_inp = torch.sparse_coo_tensor(
        indices.to(flag_gems.device),
        torch.randn(8, generator=gen).to(flag_gems.device),
        (4, 4),
    )
    assert not res_inp.is_coalesced()

    with pytest.raises(RuntimeError) as ref_err:
        torch.ops.aten.indices(utils.to_reference(res_inp))
    with pytest.raises(RuntimeError) as res_err:
        flag_gems.indices(res_inp)
    assert str(ref_err.value) == str(
        res_err.value
    ), f"message differs: {res_err.value!r} vs {ref_err.value!r}"


@pytest.mark.indices
@pytest.mark.parametrize("layout", ["csr", "csc"], ids=["csr", "csc"])
def test_indices_unsupported_layout_raises(layout):
    # Native ATen accepts only sparse COO; the rejection is a RuntimeError
    # with the native message, byte-identical.
    indptr = torch.tensor([0, 2, 4], dtype=torch.int64, device=flag_gems.device)
    indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=flag_gems.device)
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], device=flag_gems.device)
    if layout == "csr":
        inp = torch.sparse_csr_tensor(indptr, indices, values, size=(2, 4))
    else:
        inp = torch.sparse_csc_tensor(indptr, indices, values, size=(2, 4))
    with pytest.raises(RuntimeError) as ref_err:
        torch.ops.aten.indices(utils.to_reference(inp))
    with pytest.raises(RuntimeError) as res_err:
        flag_gems.indices(inp)
    assert str(ref_err.value) == str(res_err.value)


@pytest.mark.indices
def test_indices_dense_rejected():
    inp = torch.randn(2, 3, device=flag_gems.device)
    with pytest.raises(RuntimeError) as ref_err:
        torch.ops.aten.indices(utils.to_reference(inp))
    with pytest.raises(RuntimeError) as res_err:
        flag_gems.indices(inp)
    assert str(ref_err.value) == str(res_err.value)
