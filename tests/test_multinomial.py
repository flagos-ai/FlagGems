import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

random.seed(time.time() // 100)

device = flag_gems.device


@pytest.mark.multinomial
@pytest.mark.parametrize("shape", utils.UT_SHAPES_1D + utils.UT_SHAPES_2D)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("n_samples", [1000])
def test_multinomial_with_replacement(shape, dtype, n_samples):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)

    if shape[-1] == 1:
        dist = torch.rand(size=shape, dtype=dtype, device=flag_gems.device)
        with flag_gems.use_gems():
            res_out = torch.multinomial(dist, n_samples, True)
        assert torch.all(res_out == 0)
    else:
        # Mask p% off of the categories and test the sampling results fall in the rest
        for p in (0.1, 0.5, 0.9):
            dist = torch.rand(size=shape, dtype=dtype, device=flag_gems.device)
            dist[torch.rand(shape) < p] = 0
            # Make sure there's at least one non-zero probability
            dist[..., -1] = 0.5

            with flag_gems.use_gems():
                res_out = torch.multinomial(dist, n_samples, True)

            res_dist = torch.gather(dist, -1, res_out)
            # assert torch.all(res_dist)
            assert torch.sum(res_dist == 0) / res_dist.numel() < 0.001


@pytest.mark.multinomial
@pytest.mark.parametrize("pool", utils.UT_SHAPES_2D)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_multinomial_without_replacement(pool, dtype):
    dist = torch.rand(size=pool, dtype=dtype, device=flag_gems.device)
    k = pool[-1]

    if k > 1:
        ns = [k // 2, k]
    else:
        ns = [1]

    for n in ns:
        with flag_gems.use_gems():
            out = torch.multinomial(dist, n, False)

        # Verifies uniqueness
        idx_cnt = torch.nn.functional.one_hot(out).sum(1)
        assert torch.all(idx_cnt <= 1)
