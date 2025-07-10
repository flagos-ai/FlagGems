import numpy as np
import pytest
import scipy
import torch

import flag_gems

from .accuracy_utils import DISTRIBUTION_SHAPES, FLOAT_DTYPES

if flag_gems.vendor_name == "kunlunxin":
    pytestmark = pytest.mark.skip("Test Files for Operators Not Pending Testing")

# The Kolmogorov-Smirnov test (K-S test or KS test) is performed on the
# distribution operator. By having randomness, CI does not perform


@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_normal_pvalue(shape, dtype):
    loc = torch.full(size=shape, fill_value=3.0, dtype=dtype, device=flag_gems.device)
    scale = torch.full(
        size=shape, fill_value=10.0, dtype=dtype, device=flag_gems.device
    )
    with flag_gems.use_gems():
        res_out = torch.distributions.normal.Normal(loc, scale).sample()
    pvalue = scipy.stats.kstest(
        res_out.cpu().numpy().flatten(),
        lambda x: scipy.stats.norm.cdf(x, loc=3.0, scale=10.0),
    ).pvalue
    assert pvalue > 0.05


@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_accuracy_uniform_pvalue(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        x.uniform_(-3, 3)
    pvalue = scipy.stats.kstest(
        x.cpu().numpy().flatten(),
        lambda x: scipy.stats.uniform.cdf(x, loc=-3.0, scale=6.0),
    ).pvalue
    assert pvalue > 0.05


@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", (torch.float32,))
@pytest.mark.parametrize("lambd", (0.01, 0.5, 100.0))
def test_accuracy_exponential_pvalue(shape, dtype, lambd):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)
    x = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        x.exponential_(lambd=lambd)
    expo_cdf = lambda x: np.where(x < 0, 0, 1.0 - np.exp(-lambd * x))
    pvalue = scipy.stats.kstest(x.cpu().numpy().flatten(), expo_cdf).pvalue
    assert pvalue > 0.05


@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_accuracy_rand_pvalue(shape, dtype):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)
    with flag_gems.use_gems():
        res_out = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    pvalue = scipy.stats.kstest(
        res_out.cpu().numpy().flatten(), lambda x: scipy.stats.uniform.cdf(x)
    ).pvalue
    assert pvalue > 0.05


@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_accuracy_randn_pvalue(shape, dtype):
    with flag_gems.use_gems():
        res_out = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    pvalue = scipy.stats.kstest(
        res_out.cpu().numpy().flatten(), lambda x: scipy.stats.norm.cdf(x)
    ).pvalue
    assert pvalue > 0.05
