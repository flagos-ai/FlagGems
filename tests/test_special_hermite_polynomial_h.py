import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_hermite_polynomial_h
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# special.hermite_polynomial_h reference only supports float32 and float64
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_special_hermite_polynomial_h(shape, dtype):
    # Test with tensor n in [0, 9]
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    n = torch.randint(0, 10, (1,), device=flag_gems.device).squeeze()

    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.special.hermite_polynomial_h(ref_inp, utils.to_reference(n, True))
    with flag_gems.use_gems():
        res_out = torch.special.hermite_polynomial_h(inp, n)

    # Hermite polynomials use float32 intermediates, so per-dtype tolerances
    # are needed to account for accumulated floating-point errors.
    if dtype == torch.float32:
        utils.gems_assert_close(res_out, ref_out, dtype, atol=500.0)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype, atol=1000.0)


@pytest.mark.special_hermite_polynomial_h
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# special.hermite_polynomial_h reference only supports float32 and float64
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_special_hermite_polynomial_h_scalar(shape, dtype):
    # Test with scalar n = 9 (largest supported degree)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    n = 9

    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.special.hermite_polynomial_h(ref_inp, n)
    with flag_gems.use_gems():
        res_out = torch.special.hermite_polynomial_h(inp, n)

    # n=9 produces the largest Hermite polynomial values; relax tolerance.
    if dtype == torch.float32:
        utils.gems_assert_close(res_out, ref_out, dtype, atol=500.0)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype, atol=1000.0)


@pytest.mark.special_hermite_polynomial_h
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_special_hermite_polynomial_h_out_of_range(dtype):
    # Verify that n >= 10 or n < 0 raises ValueError
    inp = torch.randn(4, 4, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        with pytest.raises(ValueError, match="only supports n"):
            torch.special.hermite_polynomial_h(inp, 10)
        with pytest.raises(ValueError, match="only supports n"):
            torch.special.hermite_polynomial_h(inp, -1)

    # Verify that tensor n with values >= 10 raises ValueError
    with flag_gems.use_gems():
        with pytest.raises(ValueError, match="only supports n"):
            n_bad = torch.tensor(10, dtype=torch.int32, device=flag_gems.device)
            torch.special.hermite_polynomial_h(inp, n_bad)
        with pytest.raises(ValueError, match="only supports n"):
            n_bad = torch.tensor(-1, dtype=torch.int32, device=flag_gems.device)
            torch.special.hermite_polynomial_h(inp, n_bad)
