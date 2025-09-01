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


@pytest.mark.skipif(flag_gems.vendor_name == "mthreads", reason="skip for updating")
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
