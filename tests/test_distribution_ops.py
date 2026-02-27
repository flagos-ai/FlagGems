import numpy as np
import pytest
import scipy
import torch

import flag_gems

from .accuracy_utils import DISTRIBUTION_SHAPES, FLOAT_DTYPES, to_reference

device = flag_gems.device


@pytest.mark.normal
@pytest.mark.parametrize("float", ["none", "mean", "std"])
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_normal(float, shape, dtype):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)
    if flag_gems.vendor_name in ["metax", "iluvatar", "kunlunxin"]:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    loc = (
        3.0
        if float == "mean"
        else torch.full(
            size=shape, fill_value=3.0, dtype=dtype, device=flag_gems.device
        )
    )
    scale = (
        10.0
        if float == "std"
        else torch.full(
            size=shape, fill_value=10.0, dtype=dtype, device=flag_gems.device
        )
    )
    with flag_gems.use_gems():
        res_out = torch.normal(loc, scale)
    ref_out = to_reference(res_out)
    mean = torch.mean(ref_out)
    std = torch.std(ref_out)
    assert torch.abs(mean - 3.0) < 0.1
    assert torch.abs(std - 10.0) < 0.1


@pytest.mark.inplace
@pytest.mark.normal_
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_normal_(shape, dtype):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)
    if flag_gems.vendor_name in ["metax", "iluvatar"]:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    loc = 3.0
    scale = 10.0
    res_out = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out.normal_(loc, scale)
    ref_out = to_reference(res_out)
    mean = torch.mean(ref_out)
    std = torch.std(ref_out)
    assert torch.abs(mean - 3.0) < 0.1
    assert torch.abs(std - 10.0) < 0.1


@pytest.mark.uniform_
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_uniform(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        x.uniform_(-3, 3)
    assert (x <= 3.0).all()
    assert (x >= -3.0).all()


@pytest.mark.exponential_
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_exponential_(shape, dtype):
    x = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        x.exponential_()
    assert x.min() > 0


@pytest.mark.exponential_
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_fast_exponential_(shape, dtype):
    x = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    lambd = 1.0
    mean_tol = 0.05
    var_tol = 0.05
    with flag_gems.use_gems():
        x.exponential_()
    x_res = to_reference(x)
    mean_res = torch.mean(x_res.to(torch.float32)).to(dtype)
    var_res = torch.var(x_res.to(torch.float32)).to(dtype)
    mean_ref = 1.0 / lambd
    var_ref = 1.0 / (lambd**2)
    assert torch.abs(mean_res - mean_ref) < mean_tol
    assert torch.abs(var_res - var_ref) < var_tol


@pytest.mark.poisson
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_poisson(shape, dtype):
    # Test that Poisson samples have correct mean and variance
    # For Poisson(lambda), mean = lambda and variance = lambda
    lam = 5.0
    inp = torch.full(size=shape, fill_value=lam, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res_out = torch.poisson(inp)

    ref_out = to_reference(res_out)
    mean = torch.mean(ref_out.to(torch.float32))
    var = torch.var(ref_out.to(torch.float32))

    # Statistical tolerance
    mean_tol = 0.3
    var_tol = 0.5

    assert torch.abs(mean - lam) < mean_tol, f"Mean {mean} differs from expected {lam}"
    assert torch.abs(var - lam) < var_tol, f"Variance {var} differs from expected {lam}"
    # Poisson samples should be non-negative integers
    assert (res_out >= 0).all(), "Poisson samples should be non-negative"


@pytest.mark.poisson
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_poisson_varying_rates(shape, dtype):
    # Test with varying lambda values
    # Create input tensor with different rate values
    inp = torch.rand(size=shape, dtype=dtype, device=flag_gems.device) * 10 + 1  # rates between 1 and 11

    with flag_gems.use_gems():
        res_out = torch.poisson(inp)

    # All outputs should be non-negative
    assert (res_out >= 0).all(), "Poisson samples should be non-negative"
    # Output should be finite
    assert torch.isfinite(res_out).all(), "Poisson samples should be finite"


@pytest.mark.poisson
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_poisson_large_lambda(shape, dtype):
    # Test with large lambda (uses normal approximation)
    lam = 50.0
    inp = torch.full(size=shape, fill_value=lam, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res_out = torch.poisson(inp)

    ref_out = to_reference(res_out)
    mean = torch.mean(ref_out.to(torch.float32))
    var = torch.var(ref_out.to(torch.float32))

    # For large lambda, tolerances can be slightly larger
    mean_tol = 1.0
    var_tol = 5.0

    assert torch.abs(mean - lam) < mean_tol, f"Mean {mean} differs from expected {lam}"
    assert torch.abs(var - lam) < var_tol, f"Variance {var} differs from expected {lam}"
    assert (res_out >= 0).all(), "Poisson samples should be non-negative"


@pytest.mark.poisson
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_poisson_zero_rate(dtype):
    # Test with zero rate - should always return 0
    shape = (1000,)
    inp = torch.zeros(size=shape, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res_out = torch.poisson(inp)

    assert (res_out == 0).all(), "Poisson(0) should always return 0"


@pytest.mark.multinomial
@pytest.mark.parametrize("shape", [(1024, 10)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("n_samples", [2048])
def test_accuracy_multinomial_with_replacement(shape, dtype, n_samples):
    # First use multinomial to generate a series of indices, then
    # use the index counts as the input probabilities (scaled)
    rand_indices = torch.multinomial(torch.rand(shape), n_samples, True).to(device)
    inp_counts = torch.nn.functional.one_hot(rand_indices).sum(1)
    with flag_gems.use_gems():
        out_indices = torch.multinomial(inp_counts.to(dtype=dtype), n_samples, True)
    out_counts = torch.nn.functional.one_hot(out_indices).sum(1)
    # Do a simple Chi-square test
    assert torch.equal(inp_counts.sum(-1), out_counts.sum(-1))
    chi2, pvalue = scipy.stats.chisquare(
        out_counts.tolist(), inp_counts.tolist(), axis=-1
    )
    assert np.sum(pvalue < 0.05) / len(pvalue) < 0.1
