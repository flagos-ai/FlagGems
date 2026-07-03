import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Test for linalg_ldl_solve
LDL_SOLVE_SHAPES = [
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (32, 7),
]


# CUDA ldl_factor_ex used to build LD supports float32/float64/complex64/complex128 here.
LDL_SOLVE_DTYPES = [torch.float32, torch.float64, torch.complex64]
if utils.fp64_is_supported:
    LDL_SOLVE_DTYPES.append(torch.complex128)


def _make_ldl_inputs(batch_shape, n, k, dtype, device):
    A = torch.randn(*batch_shape, n, n, dtype=dtype, device=device)
    A = A @ A.mT + torch.eye(n, dtype=dtype, device=device) * n
    B = torch.randn(*batch_shape, n, k, dtype=dtype, device=device)
    return A, B


def _assert_ldl_close(res_out, ref_out, dtype):
    if dtype == torch.complex128:
        torch.testing.assert_close(res_out, ref_out, atol=1e-7, rtol=1e-7)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_ldl_solve
@pytest.mark.parametrize("shape", LDL_SOLVE_SHAPES)
@pytest.mark.parametrize("dtype", LDL_SOLVE_DTYPES)
def test_linalg_ldl_solve(shape, dtype):
    n, k = shape
    A, B = _make_ldl_inputs((), n, k, dtype, flag_gems.device)

    # Compute LDL factorization
    LD, pivots, info = torch.linalg.ldl_factor_ex(A)
    assert info.eq(0).all()

    ref_A = utils.to_reference(A, upcast=dtype in (torch.float32, torch.complex64))
    ref_B = utils.to_reference(B, upcast=dtype in (torch.float32, torch.complex64))
    ref_LD, ref_pivots, ref_info = torch.linalg.ldl_factor_ex(ref_A)
    assert ref_info.eq(0).all()

    ref_out = torch.linalg.ldl_solve(ref_LD, ref_pivots, ref_B)
    with flag_gems.use_gems():
        res_out = torch.linalg.ldl_solve(LD, pivots, B)

    _assert_ldl_close(res_out, ref_out, dtype)


@pytest.mark.linalg_ldl_solve
@pytest.mark.parametrize("shape", LDL_SOLVE_SHAPES)
@pytest.mark.parametrize("dtype", LDL_SOLVE_DTYPES)
def test_linalg_ldl_solve_batched(shape, dtype):
    batch_size = 4
    n, k = shape
    A, B = _make_ldl_inputs((batch_size,), n, k, dtype, flag_gems.device)

    # Compute LDL factorization
    LD, pivots, info = torch.linalg.ldl_factor_ex(A)
    assert info.eq(0).all()

    ref_A = utils.to_reference(A, upcast=dtype in (torch.float32, torch.complex64))
    ref_B = utils.to_reference(B, upcast=dtype in (torch.float32, torch.complex64))
    ref_LD, ref_pivots, ref_info = torch.linalg.ldl_factor_ex(ref_A)
    assert ref_info.eq(0).all()

    ref_out = torch.linalg.ldl_solve(ref_LD, ref_pivots, ref_B)
    with flag_gems.use_gems():
        res_out = torch.linalg.ldl_solve(LD, pivots, B)

    _assert_ldl_close(res_out, ref_out, dtype)
