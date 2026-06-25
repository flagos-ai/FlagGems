import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Small shapes for linalg_svdvals tests to avoid SVD kernel compilation timeouts
SVD_SHAPES = [
    (3, 4),  # tall matrix
    (4, 3),  # wide matrix
    (4, 4),  # square matrix
    (5, 3),  # rectangular
    (3, 5),  # rectangular
    (6, 4),  # rectangular
]


@pytest.mark.linalg_svdvals
@pytest.mark.parametrize("M, N", SVD_SHAPES)
# Only float32 is supported for SVD on CUDA (PyTorch limitation)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_svdvals(M, N, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Skipping fp32 linalg_svdvals test on tsingmicro platform")

    A = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    ref_A = utils.to_reference(A, True)

    ref_out = torch.linalg.svdvals(ref_A)
    res_out = flag_gems.linalg_svdvals(A)

    utils.gems_assert_close(res_out, ref_out, dtype)

    # Verify dispatch via use_gems()
    with flag_gems.use_gems():
        gems_out = torch.ops.aten.linalg_svdvals(A)
    utils.gems_assert_close(gems_out, ref_out, dtype)


@pytest.mark.linalg_svdvals
@pytest.mark.parametrize("M, N", SVD_SHAPES)
# Only float32 is supported for SVD on CUDA (PyTorch limitation)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_svdvals_batch(M, N, dtype):
    """Test linalg_svdvals with batch dimensions"""
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Skipping fp32 linalg_svdvals test on tsingmicro platform")

    batch_size = 4
    A = torch.randn((batch_size, M, N), dtype=dtype, device=flag_gems.device)
    ref_A = utils.to_reference(A, True)

    ref_out = torch.linalg.svdvals(ref_A)
    res_out = flag_gems.linalg_svdvals(A)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_svdvals
@pytest.mark.parametrize("M, N", SVD_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_svdvals_non_contiguous(M, N, dtype):
    """Test linalg_svdvals with non-contiguous input"""
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Skipping fp32 linalg_svdvals test on tsingmicro platform")

    # Create a non-contiguous view by slicing from a larger transposed tensor
    big = torch.randn((N + 2, M + 2), dtype=dtype, device=flag_gems.device).T
    A = big[:M, :N]
    assert not A.is_contiguous(), "Expected non-contiguous input"
    ref_A = utils.to_reference(A, True)

    ref_out = torch.linalg.svdvals(ref_A)
    res_out = flag_gems.linalg_svdvals(A)

    utils.gems_assert_close(res_out, ref_out, dtype)
