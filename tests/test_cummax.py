import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    CUMMAX_SHAPES = [(2, 32)]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    CUMMAX_SHAPES = utils.REDUCTION_SHAPES + [(2637,), (16, 1025, 255)]

random.seed(time.time() // 100)


@pytest.mark.cummax
@pytest.mark.skipif(
    utils.SkipVersion("triton", "<3.0"),
    reason="Skipping when associative_scan only support single tensor input.",
)
@pytest.mark.parametrize("shape", CUMMAX_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.INT_DTYPES)
def test_cummax(shape, dtype):
    dim = 1 if shape == utils.REDUCTION_SHAPES[-1] else -1
    if dtype in utils.INT_DTYPES:
        inp = torch.randint(-3, 3, shape, device=flag_gems.device).to(dtype)
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.cummax(ref_inp, dim=dim)

    with flag_gems.use_gems():
        res_out = torch.cummax(inp, dim=dim)

    utils.gems_assert_close(
        res_out.values, ref_out.values, dtype, reduce_dim=shape[dim]
    )
    utils.gems_assert_equal(res_out.indices, ref_out.indices)


@pytest.mark.cummax
@pytest.mark.parametrize("shape", CUMMAX_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("nan_ratio", [0.1, 0.3, 0.5])
def test_cummax_with_nan(shape, dtype, nan_ratio):
    """Test cummax with NaN values at different ratios"""
    dim = 1 if shape == utils.REDUCTION_SHAPES[-1] else -1

    # Create tensor with some NaN values
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    # Randomly set some values to NaN
    total_elements = inp.numel()
    nan_count = int(total_elements * nan_ratio)
    nan_indices = torch.randperm(total_elements)[:nan_count]
    flat_inp = inp.flatten()
    flat_inp[nan_indices] = float("nan")
    inp = flat_inp.view(shape)

    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.cummax(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.cummax(inp, dim=dim)

    utils.gems_assert_close(
        res_out.values, ref_out.values, dtype, reduce_dim=shape[dim], equal_nan=True
    )
    utils.gems_assert_equal(res_out.indices, ref_out.indices)
