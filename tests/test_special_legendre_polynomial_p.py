import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_legendre_polynomial_p
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# special.legendre_polynomial_p only supports float32 in PyTorch
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_legendre_polynomial_p(shape, dtype):
    # Fixed polynomial degree for basic correctness test
    n = 3
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.special.legendre_polynomial_p(ref_inp, n)
    with flag_gems.use_gems():
        res_out = torch.special.legendre_polynomial_p(inp, n)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_legendre_polynomial_p
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# special.legendre_polynomial_p only supports float32 in PyTorch
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("n", [0, 1, 2, 5, 10])
def test_special_legendre_polynomial_p_various_n(shape, dtype, n):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.special.legendre_polynomial_p(ref_inp, n)
    with flag_gems.use_gems():
        res_out = torch.special.legendre_polynomial_p(inp, n)
    utils.gems_assert_close(res_out, ref_out, dtype)
