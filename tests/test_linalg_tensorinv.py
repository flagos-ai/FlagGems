import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Shapes for linalg_tensorinv tests
# Each shape must satisfy prod(shape[:ind]) == prod(shape[ind:])
TENSORINV_SHAPES_IND2 = [
    (4, 6, 8, 3),  # 4*6=24, 8*3=24 -> output (8, 3, 4, 6)
    (2, 8, 4, 4),  # 2*8=16, 4*4=16 -> output (4, 4, 2, 8)
    (3, 4, 6, 2),  # 3*4=12, 6*2=12 -> output (6, 2, 3, 4)
    (1, 1, 1, 1),  # 1*1=1, 1*1=1 -> output (1, 1, 1, 1)
]

TENSORINV_SHAPES_IND1 = [
    (2, 2),  # 2x2 matrix -> output (2, 2)
    (4, 4),  # 4x4 matrix -> output (4, 4)
    (8, 8),  # 8x8 matrix -> output (8, 8)
    (3, 3),  # 3x3 matrix -> output (3, 3)
]


@pytest.mark.linalg_tensorinv
@pytest.mark.parametrize("shape", TENSORINV_SHAPES_IND2)
# torch.linalg.tensorinv requires a float32 reference path for lower precision inputs.
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_linalg_tensorinv_ind2(shape, dtype):
    """Test linalg_tensorinv with ind=2"""
    # Generate a random invertible matrix by using A = L @ L.T + I where L is random
    # This ensures the matrix is positive definite and invertible
    ind = 2
    m = shape[0] * shape[1]
    n = shape[2] * shape[3]
    assert m == n, f"Shape {shape} invalid for ind={ind}"

    # Create a random matrix and ensure it's well-conditioned
    # For float16, we need to compute in float32 and then convert
    if dtype == torch.float16:
        # Create in float32 and convert to float16
        A = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
        A_flat = A.reshape(m, n)
        A_flat = (
            A_flat @ A_flat.T
            + torch.eye(m, dtype=torch.float32, device=flag_gems.device) * 0.1
        )
        A = A_flat.reshape(shape).to(torch.float16)
    else:
        A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        # Make it invertible by adding a large identity-like term
        A_flat = A.reshape(m, n)
        A_flat = (
            A_flat @ A_flat.T + torch.eye(m, dtype=dtype, device=flag_gems.device) * 0.1
        )
        A = A_flat.reshape(shape)

    ref_A = utils.to_reference(A)

    # Compute reference in float32 (since PyTorch's tensorinv doesn't support float16)
    if dtype == torch.float16:
        ref_A_fp32 = ref_A.to(torch.float32)
        ref_out = torch.linalg.tensorinv(ref_A_fp32, ind=ind).to(torch.float16)
    else:
        ref_out = torch.linalg.tensorinv(ref_A, ind=ind)

    with flag_gems.use_gems():
        res_out = torch.linalg.tensorinv(A, ind=ind)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_tensorinv
@pytest.mark.parametrize("shape", TENSORINV_SHAPES_IND1)
# torch.linalg.tensorinv requires a float32 reference path for lower precision inputs.
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_linalg_tensorinv_ind1(shape, dtype):
    """Test linalg_tensorinv with ind=1 (equivalent to matrix inverse)"""
    ind = 1
    m = shape[0]
    n = shape[1]
    assert m == n, f"Shape {shape} must be square for ind={ind}"

    # Create a random invertible matrix
    # For float16, we need to compute in float32 and then convert
    if dtype == torch.float16:
        A = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
        A = A @ A.T + torch.eye(m, dtype=torch.float32, device=flag_gems.device) * 0.1
        A = A.to(torch.float16)
    else:
        A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        # Make it invertible by adding a identity-like term
        A = A @ A.T + torch.eye(m, dtype=dtype, device=flag_gems.device) * 0.1

    ref_A = utils.to_reference(A)

    # Compute reference in float32 (since PyTorch's tensorinv doesn't support float16)
    if dtype == torch.float16:
        ref_A_fp32 = ref_A.to(torch.float32)
        ref_out = torch.linalg.tensorinv(ref_A_fp32, ind=ind).to(torch.float16)
    else:
        ref_out = torch.linalg.tensorinv(ref_A, ind=ind)

    with flag_gems.use_gems():
        res_out = torch.linalg.tensorinv(A, ind=ind)

    utils.gems_assert_close(res_out, ref_out, dtype)
