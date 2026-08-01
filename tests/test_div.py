import random

import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


# div.Tensor with true_divide
@pytest.mark.true_divide
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_tensor_tensor(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.div(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


# div_.Tensor with true_divide_
@pytest.mark.div_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_tensor_tensor_(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone(), False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = ref_inp1.div_(ref_inp2)
    with flag_gems.use_gems():
        res_out = inp1.div_(inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


# div.Tensor with true_divide
@pytest.mark.div
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_tensor_scalar(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = utils.to_reference(inp1, False)

    ref_out = torch.div(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


# div_.Tensor with true_divide_
@pytest.mark.div_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_tensor_scalar_(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = utils.to_reference(inp1.clone(), False)

    ref_out = ref_inp1.div_(inp2)
    with flag_gems.use_gems():
        res_out = inp1.div_(inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


# div.Scalar with true_divide
@pytest.mark.div
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_scalar_tensor(shape, scalar, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.div(inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


# div.Scalar with true_divide
@pytest.mark.div
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_div_scalar_scalar(dtype):
    if dtype == torch.float32:
        inp1 = float(np.float32(random.random() + 0.01))
        inp2 = float(np.float32(random.random() + 0.01))
    else:
        inp1 = random.randint(1, 100)
        inp2 = random.randint(1, 100)

    ref_out = torch.mul(inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.mul(inp1, inp2)

    if dtype == torch.int64:
        utils.gems_assert_equal(res_out, ref_out)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype)


# div.Tensor
# Complex
@pytest.mark.div
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("complex_dtype", utils.COMPLEX_DTYPES)
def test_div_complex_complex(shape, complex_dtype):
    inp1 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)

    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.div(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, complex_dtype, equal_nan=True)


# div.Tensor
# Complex
@pytest.mark.div
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("complex_dtype", utils.COMPLEX_DTYPES)
def test_div_complex_float_tensor(shape, complex_dtype):
    inp1 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)

    if complex_dtype == torch.complex64:
        float_dtype = torch.float32
    elif complex_dtype == torch.complex32:
        float_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported complex_dtype: {complex_dtype}")

    inp2 = torch.randn(shape, dtype=float_dtype, device=flag_gems.device)

    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.div(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, complex_dtype, equal_nan=True)


# div.Tensor
# Complex
@pytest.mark.div
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("complex_dtype", utils.COMPLEX_DTYPES)
def test_div_complex_int_tensor(shape, complex_dtype):
    inp1 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)
    inp2 = torch.randint(1, 20, shape, device=flag_gems.device)

    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.div(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, complex_dtype, equal_nan=True)


@pytest.mark.div
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("complex_dtype", utils.COMPLEX_DTYPES)
def test_div_complex_int_scalar(shape, complex_dtype):
    inp1 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)
    inp2 = 3

    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = inp2

    ref_out = torch.div(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, complex_dtype, equal_nan=True)


# divide.out with true_divide_out
# aten::divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
@pytest.mark.true_divide_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_divide_out(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.divide(ref_inp1, ref_inp2, out=torch.empty_like(ref_inp1))
    out = torch.empty_like(inp1)
    with flag_gems.use_gems():
        res_out = torch.divide(inp1, inp2, out=out)

    assert res_out is out
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


# divide.out_mode with div_mode_out, rounding_mode=None reuses true division and
# supports all float dtypes.
# aten::divide.out_mode(Tensor self, Tensor other, *, str? rounding_mode,
#                       Tensor(a!) out) -> Tensor(a!)
@pytest.mark.div_mode_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_divide_out_mode_none(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.divide(
        ref_inp1, ref_inp2, rounding_mode=None, out=torch.empty_like(ref_inp1)
    )
    out = torch.empty_like(inp1)
    with flag_gems.use_gems():
        res_out = torch.divide(inp1, inp2, rounding_mode=None, out=out)

    assert res_out is out
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


# divide.out_mode with div_mode_out, trunc/floor reuse tl.math.div_rz which only
# supports float32 (see test_trunc_divide.py); casting would diff from torch.
# Upcast the reference to float64 as in test_trunc_divide.py so that the CPU
# reference and the float32 div_rz kernel agree at integer boundaries.
@pytest.mark.div_mode_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("rounding_mode", ["trunc", "floor"])
def test_divide_out_mode_rounding(shape, dtype, rounding_mode):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.divide(
        ref_inp1,
        ref_inp2,
        rounding_mode=rounding_mode,
        out=torch.empty_like(ref_inp1),
    )
    out = torch.empty_like(inp1)
    with flag_gems.use_gems():
        res_out = torch.divide(inp1, inp2, rounding_mode=rounding_mode, out=out)

    assert res_out is out
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
