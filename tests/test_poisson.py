import pytest
import torch

import flag_gems

from .accuracy_utils import DISTRIBUTION_SHAPES, FLOAT_DTYPES, to_reference


@pytest.mark.poisson
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_poisson(shape, dtype):
    lam = 5.0
    inp = torch.full(size=shape, fill_value=lam, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res_out = torch.poisson(inp)

    ref_out = to_reference(res_out)
    mean = torch.mean(ref_out.to(torch.float32))
    var = torch.var(ref_out.to(torch.float32))

    assert torch.abs(mean - lam) < 0.3
    assert torch.abs(var - lam) < 0.5
    assert (res_out >= 0).all()


@pytest.mark.poisson
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_poisson_varying_rates(shape, dtype):
    inp = torch.rand(size=shape, dtype=dtype, device=flag_gems.device) * 10 + 1

    with flag_gems.use_gems():
        res_out = torch.poisson(inp)

    assert (res_out >= 0).all()
    assert torch.isfinite(res_out).all()


@pytest.mark.poisson
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_poisson_large_lambda(shape, dtype):
    lam = 50.0
    inp = torch.full(size=shape, fill_value=lam, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res_out = torch.poisson(inp)

    ref_out = to_reference(res_out)
    mean = torch.mean(ref_out.to(torch.float32))
    var = torch.var(ref_out.to(torch.float32))

    assert torch.abs(mean - lam) < 1.0
    assert torch.abs(var - lam) < 5.0
    assert (res_out >= 0).all()


@pytest.mark.poisson
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_poisson_zero_rate(dtype):
    shape = (1000,)
    inp = torch.zeros(size=shape, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res_out = torch.poisson(inp)

    assert (res_out == 0).all()
