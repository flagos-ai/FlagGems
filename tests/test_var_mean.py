import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIMS_LIST = [1]
    CORRECTION = [1]
    KEEP_DIM = [True]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIMS_LIST = [0, 1, [0, 1], [1, 0]]
    CORRECTION = [0, 1]
    KEEP_DIM = [True, False]

random.seed(time.time() // 100)


@pytest.mark.var_mean
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIMS_LIST)
@pytest.mark.parametrize("correction", CORRECTION)
@pytest.mark.parametrize("keepdim", KEEP_DIM)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_var_mean(shape, dim, correction, keepdim, dtype):
    if shape[0] == 1:  # TODO: res is inf, while ref is nan
        shape = (2, 2)

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_var, ref_mean = torch.var_mean(
        ref_inp, dim, correction=correction, keepdim=keepdim
    )
    with flag_gems.use_gems():
        res_var, res_mean = torch.var_mean(
            inp, dim, correction=correction, keepdim=keepdim
        )

    utils.gems_assert_close(res_mean, ref_mean, dtype)
    utils.gems_assert_close(res_var, ref_var, dtype)


@pytest.mark.var_mean
@pytest.mark.parametrize(
    "shape, dim",
    [
        ((4, 8, 4096), 1),  # non-inner, multi-N-tile
        ((4, 4096, 8), 1),  # non-inner, multi-N-tile with small K
        ((8, 4096), 1),  # inner, multi-N-tile
        ((4096, 8), 0),  # non-inner, large M
        ((4096, 4096), 0),  # non-inner, multi-N-tile and multi-K-tile
    ],
)
@pytest.mark.parametrize("correction", [0, 1])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_var_mean_dim_multi_tile(shape, dim, correction, keepdim, dtype):
    # Larger shapes that exercise the multi-tile reduction loops in the inner
    # and non-inner kernels, which the small default shapes do not reach.
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_var, ref_mean = torch.var_mean(
        ref_inp, dim, correction=correction, keepdim=keepdim
    )
    with flag_gems.use_gems():
        res_var, res_mean = torch.var_mean(
            inp, dim, correction=correction, keepdim=keepdim
        )

    utils.gems_assert_close(res_var, ref_var, dtype)
    utils.gems_assert_close(res_mean, ref_mean, dtype)


@pytest.mark.var_mean
@pytest.mark.parametrize(
    "shape, dim, kind",
    [
        ((4, 0, 5), 1, "empty_reduce"),  # N == 0: both outputs NaN
        ((2, 0, 3), 1, "empty_reduce"),  # N == 0, keepdim variants below
        ((0, 5), 1, "empty_spectator"),  # M == 0: empty output
        ((5, 0), 0, "empty_spectator"),  # K == 0: empty output
        ((0, 4, 5), 1, "empty_spectator"),
        ((4, 5, 0), 1, "empty_spectator"),
    ],
)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_var_mean_dim_empty(shape, dim, kind, keepdim, dtype):
    # Exercise the empty-dimension branches of the single-dim path: an empty
    # reduction dim (N == 0) returns NaN for both outputs, an empty spectator
    # dim returns empty outputs. Both match torch.
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)

    ref_var, ref_mean = torch.var_mean(ref_inp, dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_var, res_mean = torch.var_mean(inp, dim, keepdim=keepdim)

    utils.gems_assert_equal(res_var, ref_var, equal_nan=True)
    utils.gems_assert_equal(res_mean, ref_mean, equal_nan=True)
