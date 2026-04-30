import math

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


@pytest.mark.median_dim
@pytest.mark.parametrize("shape", utils.REDUCTION_SMALL_SHAPES)
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_median_dim_float(shape, dim, keepdim, dtype):
    rank = len(shape)
    d = dim if dim >= 0 else dim + rank
    if d < 0 or d >= rank:
        pytest.skip("dim out of range for shape")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_v, ref_i = torch.median(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_v, res_i = torch.median(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_i, ref_i)
    utils.gems_assert_equal(res_v, ref_v)


@pytest.mark.median_dim
@pytest.mark.parametrize("shape", utils.REDUCTION_SMALL_SHAPES)
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
def test_median_dim_int(shape, dim, keepdim, dtype):
    rank = len(shape)
    d = dim if dim >= 0 else dim + rank
    if d < 0 or d >= rank:
        pytest.skip("dim out of range for shape")
    numel = math.prod(shape)
    inp = torch.arange(numel, dtype=dtype, device="cpu").reshape(shape).to(
        flag_gems.device
    )
    ref_inp = utils.to_reference(inp)

    ref_v, ref_i = torch.median(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_v, res_i = torch.median(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_i, ref_i)
    utils.gems_assert_equal(res_v, ref_v)


@pytest.mark.median_dim
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_median_dim_nan_mix(dtype):
    inp = torch.tensor([[1.0, 2.0], [3.0, float("nan")]], dtype=dtype, device=flag_gems.device)
    ref_inp = inp.cpu()
    ref_v, ref_i = torch.median(ref_inp, dim=0, keepdim=False)
    ref_v = ref_v.to(flag_gems.device)
    ref_i = ref_i.to(flag_gems.device)
    with flag_gems.use_gems():
        res_v, res_i = torch.median(inp, dim=0, keepdim=False)
    utils.gems_assert_equal(res_v, ref_v)
    utils.gems_assert_equal(res_i, ref_i)


@pytest.mark.median_dim
def test_median_dim_out_kw():
    inp = torch.randn((4, 7), dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ov = torch.empty((4,), dtype=inp.dtype, device=inp.device)
    oi = torch.empty((4,), dtype=torch.int64, device=inp.device)
    rov = torch.empty((4,), dtype=ref_inp.dtype)
    roi = torch.empty((4,), dtype=torch.int64)
    torch.median(ref_inp, dim=-1, keepdim=False, out=(rov, roi))
    with flag_gems.use_gems():
        torch.median(inp, dim=-1, keepdim=False, out=(ov, oi))
    utils.gems_assert_equal(ov, rov)
    utils.gems_assert_equal(oi, roi)


@pytest.mark.median_dim
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_median_dim_zero_scalar(dtype):
    inp = torch.randn((), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_v, ref_i = torch.median(ref_inp, dim=0, keepdim=False)
    with flag_gems.use_gems():
        res_v, res_i = torch.median(inp, dim=-1, keepdim=False)

    utils.gems_assert_equal(res_i, ref_i)
    utils.gems_assert_equal(res_v, ref_v)
