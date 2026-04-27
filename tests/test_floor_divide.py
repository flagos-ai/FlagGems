import random

import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg


def replace_zeros(inp):
    return torch.where(inp == 0, 1, inp)


# TODO: failed at large size, eg. (65536 * 2048,)
@pytest.mark.floor_divide
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_floor_divide_float(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.div(ref_inp1, ref_inp2, rounding_mode="floor")
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2, rounding_mode="floor")

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)


# TODO: failed at large size, eg. (65536 * 2048,)
@pytest.mark.floor_divide_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_floor_divide_float_(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone(), False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = ref_inp1.div_(ref_inp2, rounding_mode="floor")
    with flag_gems.use_gems():
        res_out = inp1.div_(inp2, rounding_mode="floor")

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.floor_divide
@pytest.mark.skipif(flag_gems.vendor_name == "aipu", reason="TODO")
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_floor_divide_int(shape, dtype):
    inp1 = torch.randint(
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        shape,
        dtype=dtype,
        device="cpu",
    ).to(flag_gems.device)
    inp2 = torch.randint(
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        shape,
        dtype=dtype,
        device="cpu",
    ).to(flag_gems.device)

    if cfg.TO_CPU:
        inp1 = replace_zeros(inp1)
        inp2 = replace_zeros(inp2)

    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = ref_inp1 // ref_inp2
    with flag_gems.use_gems():
        res_out = inp1 // inp2

    utils.gems_assert_equal(res_out, ref_out)

    for d in inp2.flatten()[:2]:
        ref_d = utils.to_reference(d, False)
        ref_out = ref_inp1 // ref_d
        with flag_gems.use_gems():
            res_out = inp1 // d
        utils.gems_assert_equal(res_out, ref_out)

        ref_out = ref_d // ref_inp1
        with flag_gems.use_gems():
            res_out = d // inp1
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.floor_divide_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_floor_divide_int_(shape, dtype):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)

    inp1 = torch.randint(
        torch.iinfo(dtype).min, torch.iinfo(dtype).max, shape, dtype=dtype, device="cpu"
    ).to(
        flag_gems.device,
    )
    inp2 = torch.randint(
        torch.iinfo(dtype).min, torch.iinfo(dtype).max, shape, dtype=dtype, device="cpu"
    ).to(
        flag_gems.device,
    )

    if cfg.TO_CPU:
        inp1 = replace_zeros(inp1)
        inp2 = replace_zeros(inp2)

    ref_inp1 = utils.to_reference(inp1.clone(), False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = ref_inp1.floor_divide_(ref_inp2)
    with flag_gems.use_gems():
        res_out = inp1.floor_divide_(inp2)

    utils.gems_assert_equal(res_out, ref_out)

    ref_inp1 = utils.to_reference(inp1.clone(), False)
    for d in inp2.flatten()[:2]:
        ref_d = utils.to_reference(d, False)
        ref_out = ref_inp1.floor_divide_(ref_d)
        with flag_gems.use_gems():
            res_out = inp1.floor_divide_(d)
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.floor_divide
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_floor_divide_scalar_scalar(dtype):
    if dtype == torch.float32:
        inp1 = float(np.float32(random.random() + 0.01))
        inp2 = float(np.float32(random.random() + 0.01))
    else:
        inp1 = random.randint(1, 100)
        inp2 = random.randint(1, 100)

    ref_out = torch.floor_divide(inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.floor_divide(inp1, inp2)

    if dtype == torch.int64:
        utils.gems_assert_equal(res_out, ref_out)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype)
