import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.acosh
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_acosh(shape, dtype):
    # acosh domain is [1, inf), so generate input in [1, 2]
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(1.0, 2.0)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.acosh(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.acosh(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.acosh_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_acosh_(shape, dtype):
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(1.0, 2.0)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.acosh_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.acosh_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)
