import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

SPECIAL_SHAPES = utils.POINTWISE_SHAPES


@pytest.mark.bernoulli
@pytest.mark.parametrize("shape", SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_bernoulli(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = torch.bernoulli(inp)
    assert (res_out == 0).logical_or(res_out == 1).all()
    assert res_out.shape == shape


# bfloat16 excluded: deterministic boundary test only needs float16+float32
@pytest.mark.bernoulli
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_accuracy_bernoulli_deterministic(dtype):
    inp_zeros = torch.zeros(1024, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        out_zeros = torch.bernoulli(inp_zeros)
    assert (out_zeros == 0).all()

    inp_ones = torch.ones(1024, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        out_ones = torch.bernoulli(inp_ones)
    assert (out_ones == 1).all()
