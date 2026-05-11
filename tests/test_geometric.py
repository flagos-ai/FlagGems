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
    positive_mask = (x >= 1).float().to(dtype)
    utils.gems_assert_equal(positive_mask, torch.ones_like(x))

    # Check that the mean is approximately 1/p
    mean = x.float().mean()
    expected = torch.tensor(1.0 / p, dtype=torch.float32, device=flag_gems.device)
    utils.gems_assert_close(mean, expected, dtype=torch.float32, atol=0.2)


@pytest.mark.geometric_
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_geometric_various_p(shape, dtype, p):
    x = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        x.geometric_(p)

    # Check that all values are positive integers (>= 1)
    positive_mask = (x >= 1).float().to(dtype)
    utils.gems_assert_equal(positive_mask, torch.ones_like(x))

    # Check that the mean is approximately 1/p
    mean = x.float().mean()
    expected = torch.tensor(1.0 / p, dtype=torch.float32, device=flag_gems.device)
    utils.gems_assert_close(mean, expected, dtype=torch.float32, atol=0.3)


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
    positive_mask = (y >= 1).float().to(dtype)
    utils.gems_assert_equal(positive_mask, torch.ones_like(y))

    # Check that the mean is approximately 1/p
    mean = y.float().mean()
    expected = torch.tensor(1.0 / p, dtype=torch.float32, device=flag_gems.device)
    utils.gems_assert_close(mean, expected, dtype=torch.float32, atol=0.2)
