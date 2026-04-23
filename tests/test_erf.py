import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.erf
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_erf(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.erf(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.erf(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.erf_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_erf_(shape, dtype):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.erf_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.erf_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)
