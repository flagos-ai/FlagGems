import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

OUTPLACE_SHAPES = [
    (64, 64),
    (100, 1, 100),
    (10000, 1),
    (100, 256, 100),
    (10000, 256),
    (20, 320, 15),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
]

OUTPLACE_MOMENT_SHAPES = [(256, 1024), (1024, 1024), (4096, 4096), (64, 512, 512)]


@pytest.mark.exponential
@pytest.mark.parametrize("shape", OUTPLACE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_exponential(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    x_ref = x.clone()
    y = flag_gems.exponential(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.min() > 0

    torch.testing.assert_close(x, x_ref)


@pytest.mark.exponential
@pytest.mark.parametrize("shape", OUTPLACE_MOMENT_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_exponential_moments(shape, dtype):
    x = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    lambd = 1.0
    mean_tol = 0.05
    var_tol = 0.05
    y = flag_gems.exponential(x, lambd=lambd)

    y_res = utils.to_reference(y)
    mean_res = torch.mean(y_res.to(torch.float32)).to(dtype)
    var_res = torch.var(y_res.to(torch.float32)).to(dtype)
    mean_ref = 1.0 / lambd
    var_ref = 1.0 / (lambd**2)

    assert torch.abs(mean_res - mean_ref) < mean_tol
    assert torch.abs(var_res - var_ref) < var_tol
