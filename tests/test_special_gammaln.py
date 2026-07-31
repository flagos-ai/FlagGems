import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_gammaln
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_gammaln(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.special.gammaln(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.gammaln(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_gammaln_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_gammaln_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    out = torch.empty_like(inp)
    ref_out = torch.empty_like(ref_inp)

    torch.special.gammaln(ref_inp, out=ref_out)
    with flag_gems.use_gems():
        torch.special.gammaln(inp, out=out)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.special_gammaln
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_gammaln_edge_cases(dtype):
    # Known values:
    #   gammaln(1) = ln(Γ(1)) = ln(1) = 0
    #   gammaln(2) = ln(Γ(2)) = ln(1) = 0
    # Poles (should return inf):
    #   gammaln(0) = inf,  gammaln(-1) = inf,  gammaln(-2) = inf
    vals = [0.0, -1.0, -2.0, 1.0, 2.0]
    inp = torch.tensor(vals, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.special.gammaln(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.gammaln(inp)
    # For pole positions res and ref should both be inf; known values should be zero or near-zero.
    # Use equal_nan can't handle inf — rely on the default error formula.
    utils.gems_assert_close(res_out, ref_out, dtype)
