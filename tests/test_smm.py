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


def _ref_dense(sparse, mat):
    """``torch.smm`` reference, materialized dense. CPU-only: ``smm`` decomposes
    to ``sspaddmm``, which is NYI on CUDA."""
    return torch.smm(sparse.cpu(), mat.cpu()).to_dense()


def _assert_matches_ref(sparse, mat, dtype=torch.float32, reduce_dim=None):
    res = flag_gems.smm(sparse, mat)
    res_dense = utils.to_reference(res.to_dense()).cpu()
    utils.gems_assert_close(
        res_dense, _ref_dense(sparse, mat), dtype=dtype, reduce_dim=reduce_dim
    )
    return res


@pytest.mark.smm
def test_smm_uncoalesced_duplicate_indices():
    """An uncoalesced input with duplicate ``(row, col)`` entries must have those
    duplicates summed, matching ``coalesce()`` semantics."""
    dtype = torch.float32
    indices = torch.tensor(
        [[0, 0, 1, 1, 1, 3], [2, 2, 0, 3, 0, 1]], device=flag_gems.device
    )
    values = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, -1.5], dtype=dtype, device=flag_gems.device
    )
    sparse = torch.sparse_coo_tensor(indices, values, size=(4, 6))
    assert not sparse.is_coalesced()
    mat = torch.randn(6, 3, dtype=dtype, device=flag_gems.device)
    _assert_matches_ref(sparse, mat, dtype=dtype, reduce_dim=6)


@pytest.mark.smm
def test_smm_explicit_zero_values():
    """Explicitly stored zeros are part of the structure: they contribute no
    value but their rows still appear in the output pattern."""
    dtype = torch.float32
    indices = torch.tensor([[0, 1, 2], [1, 2, 3]], device=flag_gems.device)
    values = torch.tensor([0.0, 0.0, 3.0], dtype=dtype, device=flag_gems.device)
    sparse = torch.sparse_coo_tensor(indices, values, size=(4, 5)).coalesce()
    mat = torch.randn(5, 3, dtype=dtype, device=flag_gems.device)
    res = _assert_matches_ref(sparse, mat, dtype=dtype, reduce_dim=5)
    # Rows 0 and 1 are structurally present despite their zero values.
    res_rows = set(utils.to_reference(res.coalesce().indices()[0]).cpu().tolist())
    assert {0, 1, 2} <= res_rows


@pytest.mark.smm
@pytest.mark.parametrize("kind", ["stride2", "transposed", "offset"])
def test_smm_non_contiguous_mat(kind):
    """The dense operand may be a view; the kernel must honor its strides."""
    dtype = torch.float32
    K, N = 8, 4
    sparse = _make_sparse_coo(6, K, dtype, flag_gems.device, density=0.4)
    if kind == "stride2":
        mat = torch.randn(K, N * 2, dtype=dtype, device=flag_gems.device)[:, ::2]
    elif kind == "transposed":
        mat = torch.randn(N, K, dtype=dtype, device=flag_gems.device).t()
    else:
        base = torch.randn(K * N + 7, dtype=dtype, device=flag_gems.device)
        mat = base[5:].as_strided((K, N), (N, 1), 5)
    assert not mat.is_contiguous() or kind == "offset"
    _assert_matches_ref(sparse, mat, dtype=dtype, reduce_dim=K)


@pytest.mark.smm
@pytest.mark.parametrize("M, K, N", [(0, 8, 4), (4, 0, 4), (4, 8, 0), (0, 0, 0)])
def test_smm_zero_sized(M, K, N):
    """Zero-sized dimensions produce a correctly shaped empty result."""
    dtype = torch.float32
    indices = torch.empty((2, 0), dtype=torch.int64, device=flag_gems.device)
    values = torch.empty((0,), dtype=dtype, device=flag_gems.device)
    sparse = torch.sparse_coo_tensor(indices, values, size=(M, K)).coalesce()
    mat = torch.randn(K, N, dtype=dtype, device=flag_gems.device)
    res = flag_gems.smm(sparse, mat)
    assert res.shape == (M, N)
    res_dense = utils.to_reference(res.to_dense()).cpu()
    assert res_dense.shape == _ref_dense(sparse, mat).shape


@pytest.mark.smm
@pytest.mark.parametrize("density", [0.001, 0.005, 0.01])
def test_smm_low_density(density):
    """Genuinely sparse inputs: the point of the operator. These are large enough
    that a dense materialization of the sparse operand would dominate memory."""
    dtype = torch.float32
    M, K, N = 2048, 2048, 16
    torch.manual_seed(7)
    nnz = max(1, int(M * K * density))
    rows = torch.randint(0, M, (nnz,), device=flag_gems.device)
    cols = torch.randint(0, K, (nnz,), device=flag_gems.device)
    vals = torch.randn(nnz, dtype=dtype, device=flag_gems.device)
    sparse = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), vals, size=(M, K)
    ).coalesce()
    mat = torch.randn(K, N, dtype=dtype, device=flag_gems.device)
    _assert_matches_ref(sparse, mat, dtype=dtype, reduce_dim=K)


@pytest.mark.smm
def test_smm_scales_with_nnz_not_dense_size():
    """A sparse operand whose dense form cannot fit in memory must still work.

    ``(2**20, 2**20)`` dense fp32 would be 4 TiB; with 512 nonzeros the COO
    implementation only ever touches O(nnz + nrows * N).
    """
    dtype = torch.float32
    M = K = 1 << 20
    N, nnz = 4, 512
    torch.manual_seed(11)
    rows = torch.randint(0, M, (nnz,), device=flag_gems.device)
    cols = torch.randint(0, K, (nnz,), device=flag_gems.device)
    vals = torch.randn(nnz, dtype=dtype, device=flag_gems.device)
    sparse = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), vals, size=(M, K)
    ).coalesce()
    mat = torch.randn(K, N, dtype=dtype, device=flag_gems.device)
    res = flag_gems.smm(sparse, mat)
    assert res.shape == (M, N)
    # One structural row per distinct input row, each with all N columns.
    n_distinct = int(torch.unique(sparse.indices()[0]).numel())
    assert res._nnz() == n_distinct * N


@pytest.mark.smm
def test_smm_device_mismatch():
    """Mixed devices must raise rather than hand a foreign pointer to the kernel."""
    dtype = torch.float32
    sparse = _make_sparse_coo(6, 8, dtype, flag_gems.device, density=0.4)
    mat_cpu = torch.randn(8, 4, dtype=dtype)
    with pytest.raises(RuntimeError, match="same device"):
        flag_gems.smm(sparse, mat_cpu)

    # A second visible GPU catches the foreign-device case, which a CPU operand
    # would not: both tensors are accelerator-resident but on different devices.
    if sparse.device.type == "cuda" and torch.cuda.device_count() > 1:
        mat_other = torch.randn(8, 4, dtype=dtype, device="cuda:1")
        with pytest.raises(RuntimeError, match="same device"):
            flag_gems.smm(sparse, mat_other)


@pytest.mark.smm
def test_smm_invalid_inputs():
    """Shape and layout validation, matching ``torch.smm``'s own messages."""
    dtype = torch.float32
    dev = flag_gems.device
    sparse = _make_sparse_coo(6, 8, dtype, dev, density=0.4)

    with pytest.raises(RuntimeError, match="Expected dim 0 size 8, got 9"):
        flag_gems.smm(sparse, torch.randn(9, 4, dtype=dtype, device=dev))

    with pytest.raises(RuntimeError, match="can only be called on sparse tensors"):
        flag_gems.smm(
            torch.randn(6, 8, dtype=dtype, device=dev),
            torch.randn(8, 4, dtype=dtype, device=dev),
        )

    with pytest.raises(RuntimeError, match="doesn't have storage"):
        flag_gems.smm(sparse, torch.randn(8, 4, dtype=dtype, device=dev).to_sparse())

    sparse_3d = torch.randn(2, 6, 8, dtype=dtype, device=dev).to_sparse().coalesce()
    with pytest.raises(RuntimeError, match="matrices expected, got 3D tensor"):
        flag_gems.smm(sparse_3d, torch.randn(8, 4, dtype=dtype, device=dev))


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
