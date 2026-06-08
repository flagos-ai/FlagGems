import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.sparse_semi_structured_addmm
@pytest.mark.parametrize("shape", [(64, 64), (128, 128), (256, 128)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_semi_structured_addmm(shape, dtype):
    from flag_gems.ops._sparse_semi_structured_addmm import (
        _sparse_semi_structured_addmm_ref,
    )

    M, N = shape
    K4 = 32  # K = 4 * K4

    # Create input tensors for sparse semi-structured addmm
    # input: (M, N), mat1: (M, 4*K4), mat1_meta: (M, K4), mat2: (4*K4, N)
    input_tensor = torch.randn(M, N, dtype=dtype, device=flag_gems.device)
    mat1 = torch.randn(M, 4 * K4, dtype=dtype, device=flag_gems.device)
    mat1_meta = torch.randint(0, 2, (M, K4), dtype=torch.bool, device=flag_gems.device)
    mat2 = torch.randn(4 * K4, N, dtype=dtype, device=flag_gems.device)

    # Reference implementation (on CPU via to_reference)
    ref_input = utils.to_reference(input_tensor)
    ref_mat1 = utils.to_reference(mat1)
    ref_meta = utils.to_reference(mat1_meta)
    ref_mat2 = utils.to_reference(mat2)
    ref_out = _sparse_semi_structured_addmm_ref(ref_input, ref_mat1, ref_meta, ref_mat2)

    # GEMS implementation
    with flag_gems.use_gems():
        res_out = flag_gems._sparse_semi_structured_addmm(
            input_tensor, mat1, mat1_meta, mat2
        )

    # Use more permissive tolerance for this operator due to numerical precision differences
    # between reference and Triton implementations
    if dtype in (torch.float16, torch.bfloat16):
        # float16/bfloat16 have limited precision, use higher atol
        utils.gems_assert_close(res_out, ref_out, dtype, atol=0.1)
    else:
        # float32, use moderate tolerance
        utils.gems_assert_close(res_out, ref_out, dtype, atol=0.02)


@pytest.mark.sparse_semi_structured_addmm
@pytest.mark.parametrize("shape", [(64, 64), (128, 128)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_semi_structured_addmm_with_alpha_beta(shape, dtype):
    from flag_gems.ops._sparse_semi_structured_addmm import (
        _sparse_semi_structured_addmm_ref,
    )

    M, N = shape
    K4 = 32  # K = 4 * K4

    # Test with alpha and beta
    alpha = 2.5
    beta = 0.5

    # Create input tensors
    input_tensor = torch.randn(M, N, dtype=dtype, device=flag_gems.device)
    mat1 = torch.randn(M, 4 * K4, dtype=dtype, device=flag_gems.device)
    mat1_meta = torch.randint(0, 2, (M, K4), dtype=torch.bool, device=flag_gems.device)
    mat2 = torch.randn(4 * K4, N, dtype=dtype, device=flag_gems.device)

    # Reference implementation (on CPU via to_reference)
    ref_input = utils.to_reference(input_tensor)
    ref_mat1 = utils.to_reference(mat1)
    ref_meta = utils.to_reference(mat1_meta)
    ref_mat2 = utils.to_reference(mat2)
    ref_out = _sparse_semi_structured_addmm_ref(
        ref_input, ref_mat1, ref_meta, ref_mat2, alpha=alpha, beta=beta
    )

    # GEMS implementation
    with flag_gems.use_gems():
        res_out = flag_gems._sparse_semi_structured_addmm(
            input_tensor, mat1, mat1_meta, mat2, alpha=alpha, beta=beta
        )

    # Use more permissive tolerance (higher for alpha/beta scaling)
    if dtype in (torch.float16, torch.bfloat16):
        utils.gems_assert_close(res_out, ref_out, dtype, atol=0.3)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype, atol=0.05)
