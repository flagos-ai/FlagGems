import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_hermite_polynomial_h
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# special.hermite_polynomial_h reference only supports float32 and float64
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_special_hermite_polynomial_h(shape, dtype):
    # Test with tensor n
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    n = torch.randint(0, 5, (1,), device=flag_gems.device).squeeze()

    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.special.hermite_polynomial_h(ref_inp, utils.to_reference(n, True))
    with flag_gems.use_gems():
        res_out = torch.special.hermite_polynomial_h(inp, n)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_hermite_polynomial_h
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# special.hermite_polynomial_h reference only supports float32 and float64
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_special_hermite_polynomial_h_scalar(shape, dtype):
    # Test with scalar n
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    n = 3  # scalar

    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.special.hermite_polynomial_h(ref_inp, n)
    with flag_gems.use_gems():
        res_out = torch.special.hermite_polynomial_h(inp, n)

    utils.gems_assert_close(res_out, ref_out, dtype)
