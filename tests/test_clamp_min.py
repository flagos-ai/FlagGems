import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.clamp_min
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clamp_min(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mini = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_mini = utils.to_reference(mini)

    ref_out = torch.clamp_min(ref_inp, min=ref_mini)
    with flag_gems.use_gems():
        res_out = torch.clamp_min(inp, min=mini)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.clamp_min_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clamp_min_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mini = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())
    ref_mini = utils.to_reference(mini)

    ref_out = torch.clamp_min_(ref_inp, min=ref_mini)
    with flag_gems.use_gems():
        res_out = torch.clamp_min_(inp, min=mini)

    utils.gems_assert_equal(res_out, ref_out)
