import pytest
import torch

import flag_gems

from .accuracy_utils import DISTRIBUTION_SHAPES, FLOAT_DTYPES


@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_normal(shape, dtype):
    loc = torch.full(size=shape, fill_value=3.0, dtype=dtype, device="cuda")
    scale = torch.full(size=shape, fill_value=10.0, dtype=dtype, device="cuda")
    with flag_gems.use_gems():
        res_out = torch.distributions.normal.Normal(loc, scale).sample()
    mean = torch.mean(res_out)
    std = torch.std(res_out)
    assert torch.abs(mean - 3.0) < 0.1
    assert torch.abs(std - 10.0) < 0.1


@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_uniform(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device="cuda")
    with flag_gems.use_gems():
        x.uniform_(-3, 3)
    assert (x <= 3.0).all()
    assert (x >= -3.0).all()


@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_exponential_(shape, dtype):
    x = torch.empty(size=shape, dtype=dtype, device="cuda")
    with flag_gems.use_gems():
        x.exponential_()
    assert x.min() > 0
