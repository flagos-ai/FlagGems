import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

ORGQR_SHAPES = [(4, 3), (8, 5), (16, 8), (32, 16), (64, 32), (128, 64)]
ORGQR_BATCH_SHAPES = [(2, 8, 5), (2, 3, 16, 8)]
ORGQR_DTYPES = [torch.float32, torch.float64]


def make_reflectors(shape, dtype):
    matrix = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    return torch.geqrf(matrix)


@pytest.mark.orgqr
@pytest.mark.parametrize("shape", ORGQR_SHAPES + ORGQR_BATCH_SHAPES)
@pytest.mark.parametrize("dtype", ORGQR_DTYPES)
def test_accuracy_orgqr(shape, dtype):
    input, tau = make_reflectors(shape, dtype)
    ref = torch.orgqr(utils.to_reference(input), utils.to_reference(tau))
    result = flag_gems.orgqr(input, tau)
    utils.gems_assert_close(result, ref, dtype)


@pytest.mark.orgqr_out
@pytest.mark.parametrize("shape", ORGQR_SHAPES)
@pytest.mark.parametrize("dtype", ORGQR_DTYPES)
def test_accuracy_orgqr_out(shape, dtype):
    input, tau = make_reflectors(shape, dtype)
    ref = torch.orgqr(utils.to_reference(input), utils.to_reference(tau))
    out = torch.empty(0, dtype=dtype, device=flag_gems.device)
    result = flag_gems.orgqr_out(input, tau, out=out)
    assert result is out
    assert out.shape == input.shape
    utils.gems_assert_close(out, ref, dtype)
