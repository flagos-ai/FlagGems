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
    """Test x in (0, 0.5) (reflection formula path)."""
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
    """Test negative values (reflection formula + cot path)."""
    # Avoid integers where digamma has poles
    inp = -(torch.rand(shape, dtype=dtype, device=flag_gems.device) * 4.5 + 0.25)
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
