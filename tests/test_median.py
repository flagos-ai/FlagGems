import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.median
@pytest.mark.parametrize("shape", [(64, 64), (256, 256), (1024, 1024), (20, 320, 15)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("keepdim", [True, False])
def test_accuracy_median_dim(shape, dtype, dim, keepdim):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_values, _ = torch.median(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_values, _ = torch.median(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_close(res_values, ref_values, dtype)


@pytest.mark.median
@pytest.mark.parametrize(
    "shape",
    [(1, 1), (8, 8), (64, 64), (256, 256), (1024, 1024)],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_median_dim_various_sizes(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_v, _ = torch.median(ref_inp, dim=-1)
    with flag_gems.use_gems():
        res_v, _ = torch.median(inp, dim=-1)

    utils.gems_assert_close(res_v, ref_v, dtype)


@pytest.mark.median
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_median_dim_single_element(dtype):
    inp = torch.randn((5, 1, 8), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_v, _ = torch.median(ref_inp, dim=1)
    with flag_gems.use_gems():
        res_v, _ = torch.median(inp, dim=1)

    utils.gems_assert_close(res_v, ref_v, dtype)
