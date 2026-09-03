import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.sample_dirichlet
@pytest.mark.parametrize("shape", [(10, 5), (20, 10), (100, 3), (5, 20)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sample_dirichlet(shape, dtype):
    # _sample_dirichlet is stochastic, so it is validated by the distribution's
    # defining property (each K-dim sample is non-negative and sums to 1) rather
    # than an elementwise match against a reference draw.
    torch.manual_seed(42)
    alpha = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 5.0 + 0.5

    res_out = flag_gems._sample_dirichlet(alpha)

    assert res_out.shape == shape
    assert res_out.dtype == dtype
    assert (res_out >= 0).all()

    # Sum over the last dimension must be 1 for every sample.
    row_sums = res_out.sum(dim=-1)
    ref_sums = torch.ones_like(row_sums)
    utils.gems_assert_close(row_sums, ref_sums, dtype)


@pytest.mark.sample_dirichlet
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sample_dirichlet_uniform_alpha(dtype):
    """Uniform alpha (all ones) yields a symmetric Dirichlet with mean 1/K."""
    torch.manual_seed(42)
    # Large batch (1000) so the empirical mean concentrates on the true mean.
    shape = (1000, 5)
    alpha = torch.ones(shape, dtype=dtype, device=flag_gems.device)

    res_out = flag_gems._sample_dirichlet(alpha)

    mean_vals = res_out.float().mean(dim=0)
    expected_mean = torch.full_like(mean_vals, 1.0 / shape[-1])
    # Loose atol: Monte-Carlo estimate over a finite sample.
    utils.gems_assert_close(mean_vals, expected_mean, torch.float32, atol=0.05)


@pytest.mark.sample_dirichlet
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sample_dirichlet_skewed_alpha(dtype):
    """Skewed alpha concentrates probability mass on the largest component."""
    torch.manual_seed(42)
    # Large batch (500) to keep the ordering of component means stable.
    shape = (500, 3)
    alpha = torch.tensor(
        [[10.0, 1.0, 1.0]], dtype=dtype, device=flag_gems.device
    ).expand(shape)

    res_out = flag_gems._sample_dirichlet(alpha)

    mean_vals = res_out.float().mean(dim=0)
    assert mean_vals[0] > mean_vals[1]
    assert mean_vals[0] > mean_vals[2]


@pytest.mark.sample_dirichlet
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sample_dirichlet_1d(dtype):
    """1-D alpha produces a single normalized sample."""
    torch.manual_seed(42)
    alpha = torch.tensor([2.0, 3.0, 5.0], dtype=dtype, device=flag_gems.device)

    res_out = flag_gems._sample_dirichlet(alpha)

    assert res_out.shape == (3,)
    total = res_out.sum().reshape(1)
    ref_total = torch.ones_like(total)
    utils.gems_assert_close(total, ref_total, dtype)
