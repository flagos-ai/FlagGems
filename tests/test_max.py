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


@pytest.mark.max
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_max(shape, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.max(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.max(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.max
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_max_all_neg_inf(shape, dtype):
    inp = torch.full(
        shape, fill_value=float("-inf"), dtype=dtype, device=flag_gems.device
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.max(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.max(inp)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)


@pytest.mark.max
@pytest.mark.skipif(
    flag_gems.vendor_name not in ["cambricon", "metax"],
    reason="Cambricon and Metax test only",
)
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES + [[1]])
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
def test_max_int(shape, dtype):
    inp = torch.randint(-1000, 1000, shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.max(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.max(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.max
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_max_uncontiguous(shape, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device="cpu")[::2, ::2].to(
            flag_gems.device
        )
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cpu")[
            ::2, ::2
        ].to(flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.max(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.max(inp)

    utils.gems_assert_equal(res_out, ref_out)


# TODO: failed at (200, 40999, 3), while successed at this shape in mean_dim
@pytest.mark.max_dim
@pytest.mark.parametrize("shape", utils.REDUCTION_SMALL_SHAPES)
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_max_dim(shape, dim, keepdim, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = utils.to_reference(inp)

    ref_out_value, ref_out_index = torch.max(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out_value, res_out_index = torch.max(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out_index, ref_out_index)
    utils.gems_assert_equal(res_out_value, ref_out_value)


@pytest.mark.max_dim
@pytest.mark.skipif(flag_gems.vendor_name == "aipu", reason="Big shape run slowly.")
@pytest.mark.parametrize("shape", [(4, 1048577, 4)])
@pytest.mark.parametrize("keepdim, dim", [(True, 1), (False, 1)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_max_dim_big_shape(shape, dim, keepdim, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = utils.to_reference(inp)

    ref_out_value, ref_out_index = torch.max(ref_inp, dim=dim, keepdim=keepdim)

    with flag_gems.use_gems():
        res_out_value, res_out_index = torch.max(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out_index, ref_out_index)
    utils.gems_assert_equal(res_out_value, ref_out_value)
