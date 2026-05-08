import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.geometric_
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_geometric_(shape, dtype):
    p = 0.5
    x = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        x.geometric_(p)

    # Check that all values are positive integers (>= 1)
    assert (x >= 1).all()

    # Check that the mean is approximately 1/p (statistical test)
    mean = x.float().mean().item()
    assert abs(mean - 1.0 / p) < 0.2


@pytest.mark.geometric_
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_geometric_various_p(shape, dtype, p):
    x = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        x.geometric_(p)

    # Check that all values are positive integers (>= 1)
    assert (x >= 1).all()

    # Check that the mean is approximately 1/p
    mean = x.float().mean().item()
    assert abs(mean - 1.0 / p) < 0.3


@pytest.mark.geometric
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_geometric(shape, dtype):
    p = 0.5
    x = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        y = torch.ops.aten.geometric(x, p)

    # Check that the output is a new tensor
    assert y is not x

    # Check that all values are positive integers (>= 1)
    assert (y >= 1).all()

    # Check that the mean is approximately 1/p
    mean = y.float().mean().item()
    assert abs(mean - 1.0 / p) < 0.2
