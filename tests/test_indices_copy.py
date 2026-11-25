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

# Sparse COO shapes for the indices copy: (n_sparse_dims, nnz, size). The op
# copies the (sparse_dim, nnz) index buffer, so nnz drives the workload.
SPARSE_INDICES_SHAPES = (
    [
        (1, 8, (16,)),
        (2, 4, (2, 2)),
        (2, 8, (4, 4)),
        (2, 16, (8, 8)),
        (3, 32, (5, 7, 9)),
    ]
    if utils.QUICK_MODE
    else [
        (1, 8, (16,)),
        (1, 64, (64,)),
        (2, 4, (2, 2)),
        (2, 8, (4, 4)),
        (2, 64, (16, 16)),
        (2, 1024, (32, 32)),
        (3, 32, (5, 7, 9)),
        (3, 128, (16, 16, 16)),
        (4, 64, (8, 8, 8, 8)),
    ]
)


def _make_coo(nnd, nnz, size, device, seed=0):
    """Build a sparse COO tensor with distinct, sorted coordinates."""
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
    return torch.sparse_coo_tensor(indices.to(device), values.to(device), size)


def _ref_device():
    """Device that must hold the reference tensor.

    Under ``--ref cpu`` the reference lives on the CPU; otherwise it stays on
    ``flag_gems.device`` so the default lane keeps its device-side comparison.
    """
    return "cpu" if utils.TO_CPU else flag_gems.device


@pytest.mark.indices_copy
@pytest.mark.parametrize("nnd, nnz, size", SPARSE_INDICES_SHAPES)
def test_indices_copy(nnd, nnz, size):
    # The native op has no sparse-COO CPU kernel, so the reference is the COO
    # accessor `_indices()`, taken from a second tensor built on the reference
    # device (CPU under --ref cpu) with the same seed and CPU generator.
    res_inp = _make_coo(nnd, nnz, size, flag_gems.device, seed=3)
    ref_inp = _make_coo(nnd, nnz, size, _ref_device(), seed=3)
    ref_out = ref_inp._indices().clone()

    res_out = flag_gems.indices_copy(res_inp)

    assert res_out.dtype == torch.int64
    assert res_out.shape == (nnd, nnz)
    # The copy must not alias the input storage.
    assert res_out.data_ptr() != res_inp._indices().data_ptr()
    utils.gems_assert_equal(res_out, ref_inp._indices())
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.indices_copy
def test_indices_copy_empty():
    # Only the output shape is checked (there is no reference tensor), so the
    # case is device-independent and runs in both lanes.
    res_inp = torch.sparse_coo_tensor(
        torch.zeros((2, 0), dtype=torch.int64, device=flag_gems.device),
        torch.zeros((0,), device=flag_gems.device),
        (4, 4),
    )
    res_out = flag_gems.indices_copy(res_inp)

    assert res_out.shape == (2, 0)


@pytest.mark.indices_copy_out
def test_indices_copy_out():
    # The reference is the COO accessor of a same-seed tensor built on the
    # reference device (CPU under --ref cpu).
    res_inp = _make_coo(2, 16, (8, 8), flag_gems.device, seed=5)
    ref_inp = _make_coo(2, 16, (8, 8), _ref_device(), seed=5)
    res_out = torch.empty((2, 16), dtype=torch.int64, device=flag_gems.device)

    res_ret = flag_gems.indices_copy_out(res_inp, out=res_out)

    assert res_ret is res_out
    utils.gems_assert_equal(res_out, ref_inp._indices())


@pytest.mark.underscore_indices_copy
@pytest.mark.parametrize("nnd, nnz, size", SPARSE_INDICES_SHAPES)
def test__indices_copy(nnd, nnz, size):
    # The native op has no sparse-COO CPU kernel, so the reference is the COO
    # accessor `_indices()`, taken from a second tensor built on the reference
    # device (CPU under --ref cpu) with the same seed and CPU generator.
    res_inp = _make_coo(nnd, nnz, size, flag_gems.device, seed=3)
    ref_inp = _make_coo(nnd, nnz, size, _ref_device(), seed=3)
    ref_out = ref_inp._indices().clone()

    res_out = flag_gems._indices_copy(res_inp)

    assert res_out.dtype == torch.int64
    assert res_out.shape == (nnd, nnz)
    # The copy must not alias the input storage.
    assert res_out.data_ptr() != res_inp._indices().data_ptr()
    utils.gems_assert_equal(res_out, ref_inp._indices())
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.underscore_indices_copy
def test__indices_copy_empty():
    # Only the output shape is checked (there is no reference tensor), so the
    # case is device-independent and runs in both lanes.
    res_inp = torch.sparse_coo_tensor(
        torch.zeros((2, 0), dtype=torch.int64, device=flag_gems.device),
        torch.zeros((0,), device=flag_gems.device),
        (4, 4),
    )
    res_out = flag_gems._indices_copy(res_inp)

    assert res_out.shape == (2, 0)


@pytest.mark.underscore_indices_copy_out
def test__indices_copy_out():
    # The reference is the COO accessor of a same-seed tensor built on the
    # reference device (CPU under --ref cpu).
    res_inp = _make_coo(2, 16, (8, 8), flag_gems.device, seed=5)
    ref_inp = _make_coo(2, 16, (8, 8), _ref_device(), seed=5)
    res_out = torch.empty((2, 16), dtype=torch.int64, device=flag_gems.device)

    res_ret = flag_gems._indices_copy_out(res_inp, out=res_out)

    assert res_ret is res_out
    utils.gems_assert_equal(res_out, ref_inp._indices())


@pytest.mark.underscore_indices_copy_out
def test__indices_copy_out_noncontiguous_out():
    res_inp = _make_coo(2, 16, (8, 8), flag_gems.device, seed=5)
    ref_inp = _make_coo(2, 16, (8, 8), _ref_device(), seed=5)
    out_base = torch.full((2, 32), -1, dtype=torch.int64, device=flag_gems.device)
    out = out_base[:, ::2]
    assert not out.is_contiguous()
    # The expectation is built on the reference device; the untouched strided
    # columns must stay at the fill value.
    expected = torch.full((2, 32), -1, dtype=torch.int64, device=ref_inp.device)
    expected[:, ::2] = ref_inp._indices()

    res_ret = flag_gems._indices_copy_out(res_inp, out=out)

    assert res_ret is out
    utils.gems_assert_equal(out_base, expected)


@pytest.mark.underscore_indices_copy
def test__indices_copy_noncoo_layout_raises_not_implemented():
    inp = torch.randn(4, 4, device=flag_gems.device)
    with pytest.raises(NotImplementedError):
        torch.ops.aten._indices_copy(utils.to_reference(inp))
    with pytest.raises(NotImplementedError):
        flag_gems._indices_copy(inp)
    out = torch.empty((2, 0), dtype=torch.int64, device=flag_gems.device)
    with pytest.raises(NotImplementedError):
        flag_gems._indices_copy_out(inp, out=out)
