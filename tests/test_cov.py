import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# cov treats rows as variables and columns as observations, so the input is a
# 2D (variables x observations) matrix. These shapes cover a single-variable
# degenerate case, square matrices, and tall/wide matrices.
COV_SHAPES = [
    (1, 8),
    (2, 4),
    (4, 16),
    (8, 64),
    (16, 256),
    (32, 1024),
    (64, 128),
    (128, 512),
]


@pytest.mark.cov
@pytest.mark.parametrize("shape", COV_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cov(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, upcast=True)

    with flag_gems.use_gems():
        res_out = torch.cov(inp)

    ref_out = torch.cov(ref_inp).to(dtype)

    # The covariance matrix is NxN; use a reduce_dim tolerance sized to the
    # observation count, since the matrix product accumulates over n_cols.
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[1])


@pytest.mark.cov
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cov_1d(dtype):
    # A 1D input represents a single variable observed across the elements;
    # torch.cov returns a 0D scalar (the variance).
    inp = torch.randn(64, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res_out = torch.cov(inp)

    ref_out = torch.cov(utils.to_reference(inp, upcast=True)).to(dtype)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=64)


@pytest.mark.cov
@pytest.mark.parametrize("correction", [0, 1, 2])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cov_correction(correction, dtype):
    # correction controls the divisor (n_obs - correction); 0 is the biased
    # estimator, 1 is Bessel's correction (default), 2 exercises a larger shift.
    inp = torch.randn((8, 64), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, upcast=True)

    with flag_gems.use_gems():
        res_out = torch.cov(inp, correction=correction)

    ref_out = torch.cov(ref_inp, correction=correction).to(dtype)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=64)


@pytest.mark.cov
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cov_fweights(dtype):
    # Frequency weights repeat observations; they must be a non-negative
    # integer vector the length of the observation axis.
    n_cols = 128
    inp = torch.randn((8, n_cols), dtype=dtype, device=flag_gems.device)
    fweights = torch.randint(1, 5, (n_cols,), device=flag_gems.device)
    ref_inp = utils.to_reference(inp, upcast=True)
    ref_fweights = utils.to_reference(fweights)

    with flag_gems.use_gems():
        res_out = torch.cov(inp, fweights=fweights)

    ref_out = torch.cov(ref_inp, fweights=ref_fweights).to(dtype)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=n_cols)


@pytest.mark.cov
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cov_aweights(dtype):
    # Analytic weights are positive real observation weights and change the
    # divisor via the sum-of-squared-weights term.
    n_cols = 128
    inp = torch.randn((8, n_cols), dtype=dtype, device=flag_gems.device)
    aweights = torch.rand((n_cols,), device=flag_gems.device) + 0.1
    ref_inp = utils.to_reference(inp, upcast=True)
    ref_aweights = utils.to_reference(aweights)

    with flag_gems.use_gems():
        res_out = torch.cov(inp, aweights=aweights)

    ref_out = torch.cov(ref_inp, aweights=ref_aweights).to(dtype)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=n_cols)


@pytest.mark.cov
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cov_fweights_aweights(dtype):
    # Both weight kinds combine multiplicatively over the observation axis.
    n_cols = 128
    inp = torch.randn((8, n_cols), dtype=dtype, device=flag_gems.device)
    fweights = torch.randint(1, 4, (n_cols,), device=flag_gems.device)
    aweights = torch.rand((n_cols,), device=flag_gems.device) + 0.1
    ref_inp = utils.to_reference(inp, upcast=True)
    ref_fweights = utils.to_reference(fweights)
    ref_aweights = utils.to_reference(aweights)

    with flag_gems.use_gems():
        res_out = torch.cov(inp, fweights=fweights, aweights=aweights)

    ref_out = torch.cov(ref_inp, fweights=ref_fweights, aweights=ref_aweights).to(dtype)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=n_cols)


@pytest.mark.cov
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cov_non_contiguous(dtype):
    # Verify cov handles non-contiguous inputs by working on a strided slice of
    # a larger matrix.
    base = torch.randn(64, 256, dtype=dtype, device=flag_gems.device)
    inp = base[::2, ::2]
    ref_inp = utils.to_reference(inp, upcast=True)

    with flag_gems.use_gems():
        res_out = torch.cov(inp)

    ref_out = torch.cov(ref_inp).to(dtype)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=inp.shape[1])
