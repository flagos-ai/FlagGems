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

# Only float32 and float16 are covered: float32 exercises PyTorch's native
# tensorinv reference, while float16 exercises the manual fp32 reference path.
TENSORINV_DTYPES = [torch.float32, torch.float16]


@pytest.mark.linalg_tensorinv
@pytest.mark.parametrize("shape", TENSORINV_SHAPES_IND2)
# torch.linalg.tensorinv requires a float32 reference path for lower precision inputs.
@pytest.mark.parametrize("dtype", TENSORINV_DTYPES)
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
@pytest.mark.parametrize("dtype", TENSORINV_DTYPES)
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


@pytest.mark.linalg_tensorinv
# Non-SPD inputs exercise partial pivoting.
@pytest.mark.parametrize("shape", [(2, 2), (3, 3), (8, 8), (16, 16), (4, 6, 8, 3)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_tensorinv_non_spd(shape, dtype):
    """General (non symmetric / non positive-definite) invertible inputs."""
    if len(shape) == 2:
        ind = 1
    else:
        ind = 2
    m = 1
    for i in range(ind):
        m *= shape[i]
    n = 1
    for i in range(ind, len(shape)):
        n *= shape[i]
    assert m == n

    # Plain randn: general, non-SPD, invertible with high probability.  Scale
    # to keep cond(A) moderate so float32 reference comparison is meaningful.
    A = torch.randn(m, m, dtype=dtype, device=flag_gems.device) * 2.0
    A = A.reshape(shape)

    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.tensorinv(ref_A, ind=ind)

    with flag_gems.use_gems():
        res_out = torch.linalg.tensorinv(A, ind=ind)

    # Looser tolerance: general randn matrices can be moderately conditioned.
    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-2, reduce_dim=1)


@pytest.mark.linalg_tensorinv
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_linalg_tensorinv_zero_diagonal_pivot(dtype):
    """A permutation matrix: the diagonal is zero but a valid pivot exists
    off the diagonal, so partial pivoting must still produce the correct
    inverse.
    """
    A = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype, device=flag_gems.device)
    ind = 1

    ref_A = utils.to_reference(A)
    if dtype == torch.float16:
        ref_A_fp32 = ref_A.to(torch.float32)
        ref_out = torch.linalg.tensorinv(ref_A_fp32, ind=ind).to(torch.float16)
    else:
        ref_out = torch.linalg.tensorinv(ref_A, ind=ind)

    with flag_gems.use_gems():
        res_out = torch.linalg.tensorinv(A, ind=ind)

    assert not torch.isnan(res_out).any(), "tensorinv produced NaN on permutation"
    assert not torch.isinf(res_out).any(), "tensorinv produced Inf on permutation"
    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-2, reduce_dim=1)


@pytest.mark.linalg_tensorinv
def test_linalg_tensorinv_blocked_path():
    """Exercise the blocked (> _TENSORINV_BLOCK_MAX=64) dispatch path with a
    non-SPD matrix, which both covers the larger-size code route and the
    partial-pivoting logic in the blocked kernel.
    """
    dtype = torch.float32
    n = 80  # > 64 -> blocked kernel
    A = torch.randn(n, n, dtype=dtype, device=flag_gems.device) * 2.0
    A = A @ A.mT + 0.1 * torch.eye(n, dtype=dtype, device=flag_gems.device)
    ind = 1

    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.tensorinv(ref_A, ind=ind)

    with flag_gems.use_gems():
        res_out = torch.linalg.tensorinv(A, ind=ind)

    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-2, reduce_dim=1)


@pytest.mark.linalg_tensorinv
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_tensorinv_ind3(dtype):
    """Cover ind=3 (prod(shape[:3]) == prod(shape[3:])), exercising the
    matrix-size computation for ind values beyond {1, 2}.
    """
    shape = (2, 2, 2, 8)  # prod(shape[:3]) = 8 == prod(shape[3:]) = 8
    ind = 3
    m = 1
    for i in range(ind):
        m *= shape[i]

    A = torch.randn(m, m, dtype=dtype, device=flag_gems.device) * 2.0
    A = A.reshape(shape)

    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.tensorinv(ref_A, ind=ind)

    with flag_gems.use_gems():
        res_out = torch.linalg.tensorinv(A, ind=ind)

    assert tuple(res_out.shape) == shape[ind:] + shape[:ind]
    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-2, reduce_dim=1)


@pytest.mark.linalg_tensorinv
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_tensorinv_ind1_higher_dim(dtype):
    """ind=1 on a >2D input: only the first dim is the rows axis, the rest
    flatten to cols (prod must match). Covers the higher-dimensional reshape
    path.
    """
    shape = (8, 2, 2, 2)  # prod(shape[:1]) = 8 == prod(shape[1:]) = 8
    ind = 1
    m = shape[0]

    A = torch.randn(m, m, dtype=dtype, device=flag_gems.device) * 2.0
    A = A.reshape(shape)

    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.tensorinv(ref_A, ind=ind)

    with flag_gems.use_gems():
        res_out = torch.linalg.tensorinv(A, ind=ind)

    assert tuple(res_out.shape) == shape[ind:] + shape[:ind]
    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-2, reduce_dim=1)


@pytest.mark.linalg_tensorinv
@pytest.mark.parametrize(
    "A_cpu",
    [
        torch.zeros(4, 4),  # exact-zero pivot -> rank 0
        torch.tensor([[1.0, 2.0], [2.0, 4.0]]),  # exact-zero pivot -> rank 1
    ],
)
def test_linalg_tensorinv_singular(A_cpu):
    """A singular input with an exact-zero pivot must return inf/nan, not a
    finite wrong inverse. (With partial pivoting a zero pivot means the whole
    remaining column is zero.) Only structurally-exact zero pivots are
    asserted; tiny-but-nonzero pivots from round-off are a tolerance question
    outside this kernel's scope.
    """
    A = A_cpu.to(flag_gems.device)
    with flag_gems.use_gems():
        res_out = torch.linalg.tensorinv(A, ind=1)

    assert torch.isnan(res_out).any() or torch.isinf(res_out).any(), (
        "tensorinv on a singular (exact-zero-pivot) matrix must return "
        "inf/nan, not a finite wrong inverse"
    )
