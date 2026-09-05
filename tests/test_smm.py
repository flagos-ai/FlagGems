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

SMM_DENSITIES = [0.1, 0.3, 0.5]


# ``torch.smm`` (sspmm) is only implemented for float32 on the CUDA backend, so
# the GEMS implementation mirrors that restriction; there is no
# ``utils.FLOAT_DTYPES`` parametrization here.
SMM_DTYPES = [torch.float32]


# (M, K, N) shapes with varying sparsity-friendly dimensions.
SMM_SHAPES = (
    [(4, 8, 5)]
    if utils.QUICK_MODE
    else [
        (4, 8, 5),
        (7, 33, 11),
        (16, 64, 32),
        (64, 128, 64),
        (32, 256, 48),
    ]
)


def _make_sparse_coo(M, K, dtype, device, density=0.3, seed=0):
    """Build a coalesced sparse COO matrix of shape (M, K) with the given density."""
    torch.manual_seed(seed)
    dense = torch.randn(M, K, dtype=dtype, device=device)
    mask = (torch.rand(M, K, device=device) < density).to(torch.bool)
    dense = dense * mask
    sparse = dense.to_sparse().coalesce()
    return sparse


@pytest.mark.smm
@pytest.mark.parametrize("M, K, N", SMM_SHAPES)
@pytest.mark.parametrize("density", SMM_DENSITIES)
@pytest.mark.parametrize("dtype", SMM_DTYPES)
def test_smm(M, K, N, density, dtype):
    res_sparse = _make_sparse_coo(M, K, dtype, flag_gems.device, density=density)
    res_mat = torch.randn(K, N, dtype=dtype, device=flag_gems.device)

    # ``torch.smm`` has no native CUDA kernel (it decomposes to ``sspaddmm``
    # which is NYI on CUDA), so the reference must run on CPU. ``to_reference``
    # honors the ``TO_CPU`` flag and the trailing ``.cpu()`` forces CPU here
    # regardless, since CUDA has no ``torch.smm`` path.
    ref_sparse = utils.to_reference(res_sparse).cpu()
    ref_mat = utils.to_reference(res_mat).cpu()
    ref_out = torch.smm(ref_sparse, ref_mat)

    # GEMS on GPU: route through ``flag_gems.smm`` which dispatches to our kernel.
    res_out = flag_gems.smm(res_sparse, res_mat)

    # Compare the materialized dense results; sparse layouts/index ordering may
    # differ but the dense values must match.
    res_dense = utils.to_reference(res_out.to_dense()).cpu()
    ref_dense = ref_out.to_dense()
    utils.gems_assert_close(res_dense, ref_dense, dtype=dtype, reduce_dim=K)


@pytest.mark.smm
@pytest.mark.parametrize("M, K, N", SMM_SHAPES)
@pytest.mark.parametrize("density", SMM_DENSITIES)
def test_smm_sparse_pattern(M, K, N, density):
    """The output sparse pattern must match ``torch.smm``: rows appearing in the
    input contribute all output columns."""
    dtype = torch.float32
    res_sparse = _make_sparse_coo(M, K, dtype, flag_gems.device, density=density)
    res_mat = torch.randn(K, N, dtype=dtype, device=flag_gems.device)

    ref_sparse = utils.to_reference(res_sparse).cpu()
    ref_mat = utils.to_reference(res_mat).cpu()
    ref_out = torch.smm(ref_sparse, ref_mat)

    res_out = flag_gems.smm(res_sparse, res_mat)

    # Compare the set of (row, col) indices.
    ref_idx = ref_out.coalesce().indices()
    res_idx = utils.to_reference(res_out.coalesce().indices()).cpu()
    assert set(zip(ref_idx[0].tolist(), ref_idx[1].tolist())) == set(
        zip(res_idx[0].tolist(), res_idx[1].tolist())
    ), "smm output sparse pattern mismatch"


@pytest.mark.smm
def test_smm_edge_cases():
    """Edge cases: empty sparse input and all-rows-present input."""
    dtype = torch.float32
    M, K, N = 4, 3, 5

    # Empty sparse input -> empty output.
    empty_indices = torch.empty((2, 0), dtype=torch.int64, device=flag_gems.device)
    empty_values = torch.empty((0,), dtype=dtype, device=flag_gems.device)
    res_sparse = torch.sparse_coo_tensor(
        empty_indices, empty_values, size=(M, K)
    ).coalesce()
    res_mat = torch.randn(K, N, dtype=dtype, device=flag_gems.device)

    ref_sparse = utils.to_reference(res_sparse).cpu()
    ref_mat = utils.to_reference(res_mat).cpu()
    ref_out = torch.smm(ref_sparse, ref_mat)

    res_out = flag_gems.smm(res_sparse, res_mat)

    assert res_out._nnz() == 0
    res_dense = utils.to_reference(res_out.to_dense()).cpu()
    ref_dense = ref_out.to_dense()
    utils.gems_assert_close(res_dense, ref_dense, dtype=dtype, reduce_dim=K)

    # Fully-populated sparse input -> output contains all M*N entries.
    res_sparse = _make_sparse_coo(M, K, dtype, flag_gems.device, density=1.0, seed=1)
    res_mat = torch.randn(K, N, dtype=dtype, device=flag_gems.device)
    ref_sparse = utils.to_reference(res_sparse).cpu()
    ref_mat = utils.to_reference(res_mat).cpu()
    ref_out = torch.smm(ref_sparse, ref_mat)
    res_out = flag_gems.smm(res_sparse, res_mat)
    assert res_out._nnz() == M * N
    res_dense = utils.to_reference(res_out.to_dense()).cpu()
    ref_dense = ref_out.to_dense()
    utils.gems_assert_close(res_dense, ref_dense, dtype=dtype, reduce_dim=K)
