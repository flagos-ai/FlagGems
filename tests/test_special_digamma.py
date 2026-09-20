import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_digamma
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_digamma_large(shape, dtype):
    """Test x >= 1.0 (direct asymptotic path)."""
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 1.0
    ref_inp = utils.to_reference(inp)

    ref_out = torch.special.digamma(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.digamma(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_digamma
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_digamma_small_positive(shape, dtype):
    """Test x in (0.05, 0.45) (reflection formula path)."""
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 0.4 + 0.05
    ref_inp = utils.to_reference(inp)

    ref_out = torch.special.digamma(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.digamma(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_digamma
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_digamma_negative(shape, dtype):
    """Test negative values (reflection formula + cot path).

    Digamma has poles at non-positive integers. Near these poles, both the
    reference and kernel suffer from float32 catastrophic cancellation in
    pi*cot(pi*x). We restrict inputs to fractional parts in [0.1, 0.9] to
    avoid pole neighborhoods while still exercising the reflection path.
    """
    # Generate values in (-4.9, -0.1) with fractional part in [0.1, 0.9]
    # This avoids the neighborhood of poles at -1, -2, -3, -4
    base = torch.randint(0, 4, shape, device=flag_gems.device).float()
    frac = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 0.8 + 0.1
    inp = -(base + frac.float()).to(dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.special.digamma(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.digamma(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_digamma
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_digamma_mid(shape, dtype):
    """Test x in [0.5, 1.0) (recurrence path)."""
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 0.5 + 0.5
    ref_inp = utils.to_reference(inp)

    ref_out = torch.special.digamma(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.digamma(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)
