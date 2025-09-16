import logging

import pytest
import torch

import flag_gems

from .accuracy_utils import (
    FLOAT_DTYPES,
    POINTWISE_SHAPES,
    SCALARS,
    gems_assert_close,
    gems_assert_equal,
    to_reference,
)

if flag_gems.vendor_name == "kunlunxin":
    pytestmark = pytest.mark.skip("Test Files for Operators Not Pending Testing")


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("float_type", FLOAT_DTYPES)
def test_type_promotion_default(shape, alpha, float_type):
    inp1 = torch.randint(10, shape, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=float_type, device=flag_gems.device)
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)
    # arg0:int  arg1:float
    ref_out = torch.add(ref_inp1, ref_inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.add(inp1, inp2, alpha=alpha)
    gems_assert_close(res_out, ref_out, float_type)
    # arg0:float  arg1:int
    ref_out = torch.add(ref_inp2, ref_inp1, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.add(inp2, inp1, alpha=alpha)
    gems_assert_close(res_out, ref_out, float_type)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("float_type", FLOAT_DTYPES)
def test_type_promotion_no_opmath(shape, float_type):
    inp1 = torch.randint(10, shape, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=float_type, device=flag_gems.device)
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)
    # arg0:bool  arg1:int  arg2:float
    ref_out = torch.where(ref_inp1 > 0, ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.where(inp1 > 0, inp1, inp2)
    gems_assert_equal(res_out, ref_out)

    # arg0:bool  arg1:float  arg2:int
    ref_out = torch.where(ref_inp1 > 0, ref_inp2, ref_inp1)
    with flag_gems.use_gems():
        res_out = torch.where(inp1 > 0, inp2, inp1)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("float_type", FLOAT_DTYPES)
def test_type_promotion_int_to_float(shape, float_type):
    # arg0:float
    inp_float = torch.randn(shape, dtype=float_type, device=flag_gems.device)
    ref_inp = to_reference(inp_float, True)
    ref_out = torch.sin(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.sin(inp_float)
    gems_assert_close(res_out, ref_out, float_type)

    # arg0:int
    inp_int = torch.randint(10, shape, device=flag_gems.device)
    ref_inp_int = to_reference(inp_int, True)
    ref_out = torch.sin(ref_inp_int)
    with flag_gems.use_gems():
        res_out = torch.sin(inp_int)
    gems_assert_close(res_out, ref_out, torch.float32)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
def test_type_promotion_always_bool(shape):
    # arg0:int  arg0:int
    inp1 = torch.randint(0, 10, shape, device=flag_gems.device)
    inp2 = torch.randint(0, 10, shape, device=flag_gems.device)
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)
    ref_out = torch.eq(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.eq(inp1, inp2)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("float_type", FLOAT_DTYPES)
def test_type_promotion_complex_to_long(shape, float_type):
    # arg0:float
    inp = torch.randn(shape, dtype=float_type, device=flag_gems.device)
    ref_inp = to_reference(inp)
    ref_out = torch.abs(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.abs(inp)
    gems_assert_equal(res_out, ref_out)

    # arg0:int
    inp1 = torch.randint(0, 10, shape, device=flag_gems.device)
    ref_inp1 = to_reference(inp1)
    ref_out1 = torch.abs(ref_inp1)
    with flag_gems.use_gems():
        res_out1 = torch.abs(inp1)
    gems_assert_equal(res_out1, ref_out1)


@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("float_dtype", FLOAT_DTYPES)
def test_type_promotion_bool_to_long(shape, float_dtype):
    inp1 = torch.randn(shape, dtype=float_dtype, device=flag_gems.device)
    inp2 = torch.randint(0, 10, shape, device=flag_gems.device)
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)
    # arg0: float  arg1: int
    ref_out = torch.pow(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.pow(inp1, inp2)
    logging.debug(ref_out.dtype)
    logging.debug(res_out.dtype)
    gems_assert_close(res_out, ref_out, float_dtype, equal_nan=True)

    # arg0: int  arg1: float
    ref_out = torch.pow(ref_inp2, ref_inp1)
    with flag_gems.use_gems():
        res_out = torch.pow(inp2, inp1)
    logging.debug(ref_out.dtype)
    logging.debug(res_out.dtype)
    gems_assert_close(res_out, ref_out, float_dtype, equal_nan=True)
