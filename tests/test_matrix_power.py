import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# n covers identity (0), single (1), and even/odd repeated squaring.
MATRIX_POWER_EXPONENTS = [0, 1, 2, 3, 5, 8]


# Negative powers require a matrix inverse, which torch only supports for
# float32/float64, so they are exercised separately.
MATRIX_POWER_NEG_EXPONENTS = [-1, -3]


# (batch_shape, matrix_dim). Covers unbatched, batched and stacked-batch cases.
MATRIX_POWER_SHAPES = [
    ((), 2),
    ((), 8),
    ((), 32),
    ((4,), 16),
    ((2, 3), 8),
]


def _make_input(shape, dim, dtype):
    # Build a well-conditioned matrix close to the identity so that repeated
    # squaring keeps element magnitudes bounded. Unit-variance gaussians would
    # blow up geometrically with n and lose all precision in low-precision dtypes.
    eye = torch.eye(dim, dtype=dtype, device=flag_gems.device)
    return torch.randn(shape, dtype=dtype, device=flag_gems.device) * 0.1 + eye


@pytest.mark.matrix_power
@pytest.mark.parametrize("batch_shape, dim", MATRIX_POWER_SHAPES)
@pytest.mark.parametrize("n", MATRIX_POWER_EXPONENTS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_matrix_power(batch_shape, dim, n, dtype):
    shape = (*batch_shape, dim, dim)
    res_inp = _make_input(shape, dim, dtype)
    # Compare against torch in the same dtype: matrix_power chains matmuls via
    # repeated squaring, so an upcast reference would accumulate far less
    # rounding than the low-precision gems path.
    ref_inp = utils.to_reference(res_inp)

    ref_out = torch.matrix_power(ref_inp, n)
    res_out = flag_gems.matrix_power(res_inp, n)

    # Accumulated error scales with the matrix dimension and the number of products.
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=dim * max(n, 1))


@pytest.mark.matrix_power
@pytest.mark.parametrize("batch_shape, dim", MATRIX_POWER_SHAPES)
@pytest.mark.parametrize("n", MATRIX_POWER_NEG_EXPONENTS)
# Negative powers require matrix inversion via lu_factor/lu_solve, which torch
# only supports for float32/float64 (bf16/fp16 inverse is not supported by torch).
@pytest.mark.parametrize("dtype", [torch.float32])
def test_matrix_power_negative(batch_shape, dim, n, dtype):
    shape = (*batch_shape, dim, dim)
    res_inp = _make_input(shape, dim, dtype)
    ref_inp = utils.to_reference(res_inp)

    ref_out = torch.matrix_power(ref_inp, n)
    res_out = flag_gems.matrix_power(res_inp, n)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=dim * abs(n))


@pytest.mark.matrix_power_out
@pytest.mark.parametrize("batch_shape, dim", MATRIX_POWER_SHAPES)
@pytest.mark.parametrize("n", MATRIX_POWER_EXPONENTS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_matrix_power_out(batch_shape, dim, n, dtype):
    shape = (*batch_shape, dim, dim)
    res_inp = _make_input(shape, dim, dtype)
    ref_inp = utils.to_reference(res_inp)

    ref_out = torch.empty_like(ref_inp)
    torch.matrix_power(ref_inp, n, out=ref_out)

    res_out = torch.empty_like(res_inp)
    flag_gems.matrix_power_out(res_inp, n, out=res_out)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=dim * max(n, 1))
