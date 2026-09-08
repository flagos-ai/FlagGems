import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# lu_factor only supports float32/float64 in CUDA environment
LU_SHAPES = [
    (3, 3),
    (4, 4),
    (3, 4),
    (4, 3),
    (5, 5),
    (8, 8),
    (64, 64),
    (256, 128),
    (512, 512),
    (768, 768),
    (1024, 1024),
]
# torch.linalg.lu_factor only supports float32/float64; half/bfloat16 not supported
LU_DTYPES = [torch.float32, torch.float64]


@pytest.mark.lu_unpack
@pytest.mark.parametrize("shape", LU_SHAPES)
@pytest.mark.parametrize("dtype", LU_DTYPES)
def test_lu_unpack(shape, dtype):
    # Create a square or rectangular matrix and compute LU factorization
    m, n = shape
    k = min(m, n)
    A = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    ref_A = utils.to_reference(A)

    # Get LU factorization
    LU, pivots = torch.linalg.lu_factor(A)
    ref_LU, ref_pivots = torch.linalg.lu_factor(ref_A)

    # Unpack using FlagGems
    with flag_gems.use_gems():
        P, L, U = torch.ops.aten.lu_unpack(LU, pivots)

    # Unpack using reference
    ref_P, ref_L, ref_U = torch.ops.aten.lu_unpack(ref_LU, ref_pivots)

    # Compare all three outputs
    # LU factorization has accumulated numerical error proportional to k
    utils.gems_assert_close(P, ref_P, dtype)
    utils.gems_assert_close(L, ref_L, dtype, reduce_dim=k)
    utils.gems_assert_close(U, ref_U, dtype, reduce_dim=k)


@pytest.mark.lu_unpack
@pytest.mark.parametrize("shape", LU_SHAPES)
@pytest.mark.parametrize("dtype", LU_DTYPES)
def test_lu_unpack_unpack_data_false(shape, dtype):
    # Test with unpack_data=False
    m, n = shape
    A = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    ref_A = utils.to_reference(A)

    LU, pivots = torch.linalg.lu_factor(A)
    ref_LU, ref_pivots = torch.linalg.lu_factor(ref_A)

    with flag_gems.use_gems():
        P, L, U = torch.ops.aten.lu_unpack(LU, pivots, unpack_data=False)

    ref_P, ref_L, ref_U = torch.ops.aten.lu_unpack(
        ref_LU, ref_pivots, unpack_data=False
    )

    utils.gems_assert_close(P, ref_P, dtype)
    # L and U should be empty tensors
    assert L.numel() == 0
    assert U.numel() == 0


@pytest.mark.lu_unpack
@pytest.mark.parametrize("shape", LU_SHAPES)
@pytest.mark.parametrize("dtype", LU_DTYPES)
def test_lu_unpack_unpack_pivots_false(shape, dtype):
    # Test with unpack_pivots=False
    m, n = shape
    A = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    ref_A = utils.to_reference(A)

    LU, pivots = torch.linalg.lu_factor(A)
    ref_LU, ref_pivots = torch.linalg.lu_factor(ref_A)

    with flag_gems.use_gems():
        P, L, U = torch.ops.aten.lu_unpack(LU, pivots, unpack_pivots=False)

    ref_P, ref_L, ref_U = torch.ops.aten.lu_unpack(
        ref_LU, ref_pivots, unpack_pivots=False
    )

    # P should be empty
    assert P.numel() == 0
    utils.gems_assert_close(L, ref_L, dtype, reduce_dim=min(m, n))
    utils.gems_assert_close(U, ref_U, dtype, reduce_dim=min(m, n))


@pytest.mark.lu_unpack
@pytest.mark.parametrize("shape", LU_SHAPES)
@pytest.mark.parametrize("dtype", LU_DTYPES)
def test_lu_unpack_batched(shape, dtype):
    # Test with batched input
    batch_size = 4
    m, n = shape
    k = min(m, n)
    A = torch.randn(batch_size, m, n, dtype=dtype, device=flag_gems.device)
    ref_A = utils.to_reference(A)

    LU, pivots = torch.linalg.lu_factor(A)
    ref_LU, ref_pivots = torch.linalg.lu_factor(ref_A)

    with flag_gems.use_gems():
        P, L, U = torch.ops.aten.lu_unpack(LU, pivots)

    ref_P, ref_L, ref_U = torch.ops.aten.lu_unpack(ref_LU, ref_pivots)

    utils.gems_assert_close(P, ref_P, dtype)
    utils.gems_assert_close(L, ref_L, dtype, reduce_dim=k)
    utils.gems_assert_close(U, ref_U, dtype, reduce_dim=k)


@pytest.mark.lu_unpack
@pytest.mark.parametrize("shape", LU_SHAPES)
@pytest.mark.parametrize("dtype", LU_DTYPES)
def test_lu_unpack_out(shape, dtype):
    # Test lu_unpack.out with pre-allocated output tensors
    m, n = shape
    k = min(m, n)
    A = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    ref_A = utils.to_reference(A)

    LU, pivots = torch.linalg.lu_factor(A)
    ref_LU, ref_pivots = torch.linalg.lu_factor(ref_A)

    # Pre-allocate output tensors
    P_out = torch.empty(m, m, dtype=dtype, device=flag_gems.device)
    L_out = torch.empty(m, k, dtype=dtype, device=flag_gems.device)
    U_out = torch.empty(k, n, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        torch.ops.aten.lu_unpack(LU, pivots, P=P_out, L=L_out, U=U_out)

    ref_P, ref_L, ref_U = torch.ops.aten.lu_unpack(ref_LU, ref_pivots)

    utils.gems_assert_close(P_out, ref_P, dtype)
    utils.gems_assert_close(L_out, ref_L, dtype, reduce_dim=k)
    utils.gems_assert_close(U_out, ref_U, dtype, reduce_dim=k)
