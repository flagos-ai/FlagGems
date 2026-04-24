import logging
import random

import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg


# Note: tl.math.div_rz only support float32, cast will cause diff
# with torch, so we only do float32 test for now.
@pytest.mark.trunc_divide
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_trunc_div(shape, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    inp1 = torch.randn(shape, dtype=dtype, device="cpu").to(flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device="cpu").to(flag_gems.device)

    upcast = False
    if flag_gems.vendor_name not in ["cambricon", "iluvatar", "kunlunxin"]:
        upcast = True

    ref_inp1 = utils.to_reference(inp1, upcast)
    ref_inp2 = utils.to_reference(inp2, upcast)

    ref_out = torch.div(ref_inp1, ref_inp2, rounding_mode="trunc")
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2, rounding_mode="trunc")

    if not cfg.TO_CPU:
        logging.debug(
            f"The maximum difference between torch and triton is "
            f"{torch.max(torch.abs(ref_out - res_out))}"
        )
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


# Note : tl.math.div_rz only support float32, cast will cause diff
# with torch, so we only do float32 test for now.
@pytest.mark.trunc_divide_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_trunc_divide_(shape, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    inp1 = torch.randn(shape, dtype=dtype, device="cpu").to(flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device="cpu").to(flag_gems.device)
    upcast = True
    if flag_gems.vendor_name in ("cambricon", "kunlunxin", "iluvatar"):
        upcast = False
    ref_inp1 = utils.to_reference(inp1, upcast)
    ref_inp2 = utils.to_reference(inp2, upcast)

    ref_out = ref_inp1.div_(ref_inp2, rounding_mode="trunc")
    with flag_gems.use_gems():
        res_out = inp1.div_(inp2, rounding_mode="trunc")

    if not cfg.TO_CPU:
        logging.debug(
            f"The maximum difference between torch and triton is "
            f"{torch.max(torch.abs(ref_out - res_out))}"
        )

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.trunc_divide
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_trunc_divide_scalar_scalar(dtype):
    if dtype == torch.float32:
        inp1 = float(np.float32(random.random() + 0.01))
        inp2 = float(np.float32(random.random() + 0.01))
    else:
        inp1 = random.randint(1, 100)
        inp2 = random.randint(1, 100)

    ref_out = torch.div(inp1, inp2, rounding_mode="trunc")
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2, rounding_mode="trunc")

    if dtype == torch.int64:
        utils.gems_assert_equal(res_out, ref_out)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.trunc_divide
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES + [torch.int64])
def test_trunc_divide_tensor_int(shape, dtype):
    # Regression test: integer types must be dispatched at Python layer to avoid
    # passing int tensors to div_rz which only supports floating point.
    inp1 = torch.randint(1, 100, shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randint(1, 100, shape, dtype=dtype, device=flag_gems.device)

    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.div(ref_inp1, ref_inp2, rounding_mode="trunc")
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2, rounding_mode="trunc")

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.trunc_divide
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES + [torch.int64])
def test_trunc_divide_tensor_scalar_int(shape, dtype):
    # Regression test: integer types must be dispatched at Python layer to avoid
    # passing int tensors to div_rz which only supports floating point.
    inp1 = torch.randint(1, 100, shape, dtype=dtype, device=flag_gems.device)
    scalar = random.randint(1, 10)
    ref_inp1 = utils.to_reference(inp1, False)

    ref_out = torch.div(ref_inp1, scalar, rounding_mode="trunc")
    with flag_gems.use_gems():
        res_out = torch.div(inp1, scalar, rounding_mode="trunc")

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.trunc_divide
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES + [torch.int64])
def test_trunc_div_scalar_tensor_int(shape, dtype):
    # Regression test: integer types must be dispatched at Python layer to avoid
    # passing int tensors to div_rz which only supports floating point.
    inp2 = torch.randint(1, 100, shape, dtype=dtype, device=flag_gems.device)
    scalar = random.randint(1, 100)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.div(scalar, ref_inp2, rounding_mode="trunc")
    with flag_gems.use_gems():
        res_out = torch.div(scalar, inp2, rounding_mode="trunc")

    utils.gems_assert_equal(res_out, ref_out)
