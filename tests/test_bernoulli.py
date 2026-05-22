import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

SPECIAL_SHAPES = utils.POINTWISE_SHAPES


@pytest.mark.bernoulli
@pytest.mark.parametrize("shape", SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bernoulli(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = torch.bernoulli(inp)
    assert (res_out == 0).logical_or(res_out == 1).all()
    assert res_out.shape == shape


# bfloat16 excluded: deterministic boundary test only needs float16+float32
@pytest.mark.bernoulli
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_bernoulli_deterministic(dtype):
    inp_zeros = torch.zeros(1024, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        out_zeros = torch.bernoulli(inp_zeros)
    assert (out_zeros == 0).all()

    inp_ones = torch.ones(1024, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        out_ones = torch.bernoulli(inp_ones)
    assert (out_ones == 1).all()


@pytest.mark.bernoulli_
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bernoulli_(shape, dtype):
    x = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    p = 0.5
    with flag_gems.use_gems():
        x.bernoulli_(p)

    assert ((x == 0) | (x == 1)).all()

    mean = x.float().mean().item()
    assert abs(mean - p) < 0.1


@pytest.mark.bernoulli_
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("p", [0.0, 0.3, 0.7, 1.0])
def test_bernoulli_various_p(shape, dtype, p):
    x = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        x.bernoulli_(p)

    assert ((x == 0) | (x == 1)).all()

    if p == 0.0:
        assert (x == 0).all()
    elif p == 1.0:
        assert (x == 1).all()
    else:
        mean = x.float().mean().item()
        assert abs(mean - p) < 0.15
