import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

PDIST_SHAPES = [
    (4, 8),
    (8, 16),
    (16, 32),
    (32, 64),
    (64, 128),
    (128, 256),
]


@pytest.mark.pdist
@pytest.mark.parametrize("shape", PDIST_SHAPES)
# pdist CUDA kernel only supports float32; Half/BFloat16 raise RuntimeError
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_pdist(shape, dtype):
    if shape[0] < 2:
        pytest.skip("pdist requires at least 2 rows")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    p = 2.0
    ref_out = torch.pdist(ref_inp, p=p)
    with flag_gems.use_gems():
        res_out = torch.pdist(inp, p=p)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.pdist
@pytest.mark.parametrize("shape", PDIST_SHAPES)
# pdist CUDA kernel only supports float32; Half/BFloat16 raise RuntimeError
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_pdist_p1(shape, dtype):
    if shape[0] < 2:
        pytest.skip("pdist requires at least 2 rows")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    p = 1.0
    ref_out = torch.pdist(ref_inp, p=p)
    with flag_gems.use_gems():
        res_out = torch.pdist(inp, p=p)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.pdist
@pytest.mark.parametrize("shape", PDIST_SHAPES)
# pdist CUDA kernel only supports float32; Half/BFloat16 raise RuntimeError
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_pdist_pinf(shape, dtype):
    if shape[0] < 2:
        pytest.skip("pdist requires at least 2 rows")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    p = float("inf")
    ref_out = torch.pdist(ref_inp, p=p)
    with flag_gems.use_gems():
        res_out = torch.pdist(inp, p=p)

    utils.gems_assert_close(res_out, ref_out, dtype)
