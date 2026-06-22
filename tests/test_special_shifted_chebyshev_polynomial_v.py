import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_shifted_chebyshev_polynomial_v
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# special.* ops only support float32; fp16/bf16 not implemented in PyTorch
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_shifted_chebyshev_polynomial_v(shape, dtype):
    # x: float tensor (input value)
    # n: int tensor (degree of polynomial)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    n = torch.randint(0, 10, shape, device="cpu").to(torch.int32)

    ref_x = utils.to_reference(x, True)
    ref_n = n.to(ref_x.device)

    ref_out = torch.special.shifted_chebyshev_polynomial_v(ref_x, ref_n)
    with flag_gems.use_gems():
        res_out = torch.special.shifted_chebyshev_polynomial_v(x, n.to(x.device))

    # Use a relaxed atol because the recurrence-based computation accumulates
    # float32 rounding errors for higher-degree polynomials.
    utils.gems_assert_close(res_out, ref_out, dtype, atol=5e-3)


@pytest.mark.special_shifted_chebyshev_polynomial_v
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# special.* ops only support float32; fp16/bf16 not implemented in PyTorch
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_shifted_chebyshev_polynomial_v_scalar_n(shape, dtype):
    # Test with scalar n
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    n = 3  # scalar degree

    ref_x = utils.to_reference(x, True)

    ref_out = torch.special.shifted_chebyshev_polynomial_v(ref_x, n)
    with flag_gems.use_gems():
        res_out = torch.special.shifted_chebyshev_polynomial_v(x, n)

    utils.gems_assert_close(res_out, ref_out, dtype, atol=5e-3)


@pytest.mark.special_shifted_chebyshev_polynomial_v
@pytest.mark.parametrize("shape", [(1024,), (20, 320, 15)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_shifted_chebyshev_polynomial_v_out_of_range_tensor(shape, dtype):
    """Verify that tensor n >= 16 returns 0.0 (kernel guards with arithmetic masking)."""
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    n = torch.full(shape, 16, device="cpu").to(torch.int32)

    with flag_gems.use_gems():
        res_out = torch.special.shifted_chebyshev_polynomial_v(x, n.to(x.device))

    expected = torch.zeros(res_out.shape, dtype=res_out.dtype)
    utils.gems_assert_close(res_out.cpu(), expected, dtype, atol=0.0)


@pytest.mark.special_shifted_chebyshev_polynomial_v
@pytest.mark.parametrize("bad_n", [16, -1, 100])
@pytest.mark.parametrize("shape", [(3,), (10, 5)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_shifted_chebyshev_polynomial_v_out_of_range_scalar(
    bad_n, shape, dtype
):
    """Verify that scalar n outside [0, 15] returns 0.0 (kernel guards with arithmetic masking)."""
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res_out = torch.special.shifted_chebyshev_polynomial_v(x, bad_n)

    expected = torch.zeros(res_out.shape, dtype=res_out.dtype)
    utils.gems_assert_close(res_out.cpu(), expected, dtype, atol=0.0)
