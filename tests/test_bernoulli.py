import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.bernoulli_
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bernoulli_(shape, dtype):
    x = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    p = 0.5
    with flag_gems.use_gems():
        x.bernoulli_(p)

    # Check that all values are 0 or 1
    assert ((x == 0) | (x == 1)).all()

    # Check that the mean is approximately p (statistical test)
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

    # Check that all values are 0 or 1
    assert ((x == 0) | (x == 1)).all()

    # Check boundary cases
    if p == 0.0:
        assert (x == 0).all()
    elif p == 1.0:
        assert (x == 1).all()
    else:
        # Check that the mean is approximately p
        mean = x.float().mean().item()
        assert abs(mean - p) < 0.15
