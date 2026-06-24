import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_shifted_chebyshev_polynomial_u
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# shifted_chebyshev_polynomial_u_cuda does not support Half/BFloat16
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_shifted_chebyshev_polynomial_u(shape, dtype):
    # x in [0, 1] for shifted Chebyshev polynomial
    x = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    n = torch.randint(0, 10, shape, dtype=torch.long, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_n = n.to(ref_x.device).to(ref_x.dtype)

    ref_out = torch.special.shifted_chebyshev_polynomial_u(ref_x, ref_n)
    with flag_gems.use_gems():
        res_out = torch.special.shifted_chebyshev_polynomial_u(x, n)

    # Use larger tolerance for float32 due to trigonometric function precision
    utils.gems_assert_close(res_out, ref_out, dtype, atol=5e-3)


@pytest.mark.special_shifted_chebyshev_polynomial_u
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# shifted_chebyshev_polynomial_u_cuda does not support Half/BFloat16
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_shifted_chebyshev_polynomial_u_scalar_n(shape, dtype):
    # Test with scalar n (same n for all elements)
    x = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    n = 3  # scalar

    ref_x = utils.to_reference(x, True)

    ref_out = torch.special.shifted_chebyshev_polynomial_u(ref_x, n)
    with flag_gems.use_gems():
        res_out = torch.special.shifted_chebyshev_polynomial_u(x, n)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_shifted_chebyshev_polynomial_u_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# shifted_chebyshev_polynomial_u_cuda does not support Half/BFloat16
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_shifted_chebyshev_polynomial_u_(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    n = torch.randint(0, 10, shape, dtype=torch.long, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_n = n.to(ref_inp.device).to(ref_inp.dtype)

    ref_out = torch.special.shifted_chebyshev_polynomial_u(ref_inp, ref_n)
    ref_inp.copy_(ref_out)

    with flag_gems.use_gems():
        res_out = torch.special.shifted_chebyshev_polynomial_u(inp, n, out=inp)

    utils.gems_assert_close(inp, ref_inp, dtype, atol=5e-3)
    utils.gems_assert_close(res_out, ref_inp, dtype, atol=5e-3)
