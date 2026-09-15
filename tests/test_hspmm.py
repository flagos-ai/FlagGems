import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def _make_sparse_coo(shape, nnz, dtype, device):
    """Create a random sparse COO tensor with specified nnz."""
    M, K = shape
    if nnz == 0:
        indices = torch.zeros((2, 0), dtype=torch.long, device=device)
        values = torch.zeros(0, dtype=dtype, device=device)
    else:
        # Generate random row and column indices
        row_indices = torch.randint(0, M, (nnz,), device=device)
        col_indices = torch.randint(0, K, (nnz,), device=device)
        indices = torch.stack([row_indices, col_indices], dim=0)
        values = torch.randn(nnz, dtype=dtype, device=device)

    sp = torch.sparse_coo_tensor(indices, values, shape, dtype=dtype, device=device)
    return sp.coalesce()


@pytest.mark.hspmm
@pytest.mark.parametrize("M", [1, 16, 64, 256])
@pytest.mark.parametrize("K", [1, 32, 128])
@pytest.mark.parametrize("N", [1, 8, 64])
@pytest.mark.parametrize("nnz", [0, 1, 10, 100])
# CUDA aten::hspmm only supports float32/float64 (fp16/bf16 not implemented)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hspmm_accuracy(M, K, N, nnz, dtype):
    """Test hspmm correctness against dense reference."""
    device = flag_gems.device

    # Limit nnz to matrix capacity
    actual_nnz = min(nnz, M * K)

    # Create sparse mat1 and dense mat2
    mat1 = _make_sparse_coo((M, K), actual_nnz, dtype, device)
    mat2 = torch.randn(K, N, dtype=dtype, device=device)

    # FlagGems result
    result = flag_gems.hspmm(mat1, mat2)
    result_dense = result.to_dense()

    # High-precision reference: upcast inputs to fp64 so the reference matmul
    # is close to the true value, isolating GPU float32 accumulation error.
    ref_mat1 = utils.to_reference(mat1.to_dense(), upcast=True)
    ref_mat2 = utils.to_reference(mat2, upcast=True)
    ref_dense = torch.mm(ref_mat1, ref_mat2)

    # Compare
    utils.gems_assert_close(result_dense, ref_dense, dtype, reduce_dim=K)


@pytest.mark.hspmm
# CUDA aten::hspmm only supports float32/float64 (fp16/bf16 not implemented)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hspmm_non_contiguous(dtype):
    """Test hspmm with non-contiguous mat2."""
    device = flag_gems.device
    M, K, N = 16, 32, 8
    nnz = 50

    mat1 = _make_sparse_coo((M, K), nnz, dtype, device)
    mat2 = torch.randn(N, K, dtype=dtype, device=device).t()  # Non-contiguous

    result = flag_gems.hspmm(mat1, mat2)

    ref_mat1 = utils.to_reference(mat1.to_dense(), upcast=True)
    ref_mat2 = utils.to_reference(mat2, upcast=True)
    ref_dense = torch.mm(ref_mat1, ref_mat2)

    utils.gems_assert_close(result.to_dense(), ref_dense, dtype, reduce_dim=K)


@pytest.mark.hspmm
# CUDA aten::hspmm only supports float32/float64 (fp16/bf16 not implemented)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hspmm_uncoalesced_input(dtype):
    """Test hspmm with uncoalesced sparse input."""
    device = flag_gems.device
    M, K, N = 8, 16, 4

    # Create uncoalesced sparse tensor (with duplicate indices)
    indices = torch.tensor([[0, 0, 2, 2, 4], [1, 1, 3, 5, 2]], device=device)
    values = torch.randn(5, dtype=dtype, device=device)
    mat1 = torch.sparse_coo_tensor(indices, values, (M, K), dtype=dtype, device=device)

    mat2 = torch.randn(K, N, dtype=dtype, device=device)

    result = flag_gems.hspmm(mat1, mat2)

    ref_mat1 = utils.to_reference(mat1.to_dense(), upcast=True)
    ref_mat2 = utils.to_reference(mat2, upcast=True)
    ref_dense = torch.mm(ref_mat1, ref_mat2)

    utils.gems_assert_close(result.to_dense(), ref_dense, dtype, reduce_dim=K)
