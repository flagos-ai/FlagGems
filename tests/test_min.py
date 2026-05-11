import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIM_LIST = [0]
    KEEPDIM = [True]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIM_LIST = [0, 1]
    KEEPDIM = [True, False]


@pytest.mark.min
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_min(shape, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.min(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.min(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.min
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_min_all_inf(shape, dtype):
    # ensure that padding value used in min is inf, not max value
    inp = torch.full(
        shape, fill_value=float("inf"), dtype=dtype, device=flag_gems.device
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.min(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.min(inp)

    utils.gems_assert_equal(res_out, ref_out)


# TODO: failed at (200, 40999, 3), while successed at this shape in mean_dim
@pytest.mark.min_dim
@pytest.mark.parametrize("shape", utils.REDUCTION_SMALL_SHAPES)
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_min_dim(shape, dim, keepdim, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = utils.to_reference(inp)

    ref_out_value, ref_out_index = torch.min(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out_value, res_out_index = torch.min(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out_index, ref_out_index)
    utils.gems_assert_equal(res_out_value, ref_out_value)
