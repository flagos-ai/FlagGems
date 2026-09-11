import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.spdiags
@pytest.mark.parametrize("shape", [(3, 3), (4, 4), (5, 5), (10, 10)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_spdiags_single_diagonal(shape, dtype):
    """Test single diagonal (main diagonal)"""
    nrows, ncols = shape
    diag_len = min(nrows, ncols)

    res_diagonals = torch.randn((1, diag_len), dtype=dtype, device=flag_gems.device)
    res_offsets = torch.tensor([0], dtype=torch.int64, device=flag_gems.device)

    # Must use .cpu() here: torch._spdiags only exists on CPU (no CUDA kernel)
    ref_diagonals = res_diagonals.cpu()
    ref_offsets = res_offsets.cpu()

    ref_out = torch.ops.aten._spdiags(ref_diagonals, ref_offsets, list(shape))
    res_out = flag_gems._spdiags(res_diagonals, res_offsets, list(shape))

    # Compare sparse tensors
    assert res_out.layout == torch.sparse_coo
    assert ref_out.layout == torch.sparse_coo
    assert res_out.shape == ref_out.shape

    # Convert to dense for comparison. gems_assert_close requires res and ref
    # to share a device: in quick-cpu mode (TO_CPU) it moves res to CPU and
    # asserts ref is already there, otherwise both must stay on the device.
    ref_dense = ref_out.to_dense()
    if not utils.TO_CPU:
        ref_dense = ref_dense.to(flag_gems.device)
    res_dense = res_out.to_dense()
    utils.gems_assert_close(res_dense, ref_dense, dtype)


@pytest.mark.spdiags
@pytest.mark.parametrize("shape", [(4, 4), (5, 5), (10, 10)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_spdiags_multiple_diagonals(shape, dtype):
    """Test multiple diagonals with different offsets"""
    nrows, ncols = shape
    diag_len = min(nrows, ncols)

    # Test main diagonal, upper diagonal, and lower diagonal
    res_diagonals = torch.randn((3, diag_len), dtype=dtype, device=flag_gems.device)
    res_offsets = torch.tensor([0, 1, -1], dtype=torch.int64, device=flag_gems.device)

    # Reference uses CPU
    ref_diagonals = res_diagonals.cpu()
    ref_offsets = res_offsets.cpu()

    ref_out = torch.ops.aten._spdiags(ref_diagonals, ref_offsets, list(shape))
    res_out = flag_gems._spdiags(res_diagonals, res_offsets, list(shape))

    # Compare sparse tensors
    assert res_out.layout == torch.sparse_coo
    assert ref_out.layout == torch.sparse_coo
    assert res_out.shape == ref_out.shape

    # Convert to dense for comparison. gems_assert_close requires res and ref
    # to share a device: in quick-cpu mode (TO_CPU) it moves res to CPU and
    # asserts ref is already there, otherwise both must stay on the device.
    ref_dense = ref_out.to_dense()
    if not utils.TO_CPU:
        ref_dense = ref_dense.to(flag_gems.device)
    res_dense = res_out.to_dense()
    utils.gems_assert_close(res_dense, ref_dense, dtype)


@pytest.mark.spdiags
@pytest.mark.parametrize("offset", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_spdiags_various_offsets(offset, dtype):
    """Test various diagonal offsets"""
    # Small square matrix to test all offset directions within bounds
    shape = (5, 5)
    nrows, ncols = shape
    diag_len = min(nrows, ncols)

    res_diagonals = torch.randn((1, diag_len), dtype=dtype, device=flag_gems.device)
    res_offsets = torch.tensor([offset], dtype=torch.int64, device=flag_gems.device)

    # Reference uses CPU
    ref_diagonals = res_diagonals.cpu()
    ref_offsets = res_offsets.cpu()

    ref_out = torch.ops.aten._spdiags(ref_diagonals, ref_offsets, list(shape))
    res_out = flag_gems._spdiags(res_diagonals, res_offsets, list(shape))

    # Convert to dense for comparison. gems_assert_close requires res and ref
    # to share a device: in quick-cpu mode (TO_CPU) it moves res to CPU and
    # asserts ref is already there, otherwise both must stay on the device.
    ref_dense = ref_out.to_dense()
    if not utils.TO_CPU:
        ref_dense = ref_dense.to(flag_gems.device)
    res_dense = res_out.to_dense()
    utils.gems_assert_close(res_dense, ref_dense, dtype)


@pytest.mark.spdiags
@pytest.mark.parametrize("shape", [(3, 5), (5, 3), (10, 20), (20, 10)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_spdiags_non_square(shape, dtype):
    """Test non-square matrices"""
    nrows, ncols = shape
    diag_len = min(nrows, ncols)  # Offset-0 diagonal length is the smaller dimension

    res_diagonals = torch.randn((1, diag_len), dtype=dtype, device=flag_gems.device)
    res_offsets = torch.tensor([0], dtype=torch.int64, device=flag_gems.device)

    # Reference uses CPU
    ref_diagonals = res_diagonals.cpu()
    ref_offsets = res_offsets.cpu()

    ref_out = torch.ops.aten._spdiags(ref_diagonals, ref_offsets, list(shape))
    res_out = flag_gems._spdiags(res_diagonals, res_offsets, list(shape))

    # Convert to dense for comparison. gems_assert_close requires res and ref
    # to share a device: in quick-cpu mode (TO_CPU) it moves res to CPU and
    # asserts ref is already there, otherwise both must stay on the device.
    ref_dense = ref_out.to_dense()
    if not utils.TO_CPU:
        ref_dense = ref_dense.to(flag_gems.device)
    res_dense = res_out.to_dense()
    utils.gems_assert_close(res_dense, ref_dense, dtype)


@pytest.mark.spdiags
def test_spdiags_empty():
    """Test empty case"""
    # Small square matrix sufficient for empty diagonal list edge case
    shape = (3, 3)
    dtype = torch.float32

    res_diagonals = torch.randn((0, 3), dtype=dtype, device=flag_gems.device)
    res_offsets = torch.tensor([], dtype=torch.int64, device=flag_gems.device)

    # Reference uses CPU
    ref_diagonals = res_diagonals.cpu()
    ref_offsets = res_offsets.cpu()

    ref_out = torch.ops.aten._spdiags(ref_diagonals, ref_offsets, list(shape))
    res_out = flag_gems._spdiags(res_diagonals, res_offsets, list(shape))

    # Both should be empty sparse tensors
    assert res_out._nnz() == 0
    assert ref_out._nnz() == 0
    assert res_out.shape == ref_out.shape
