import random

import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


# div.Tensor with true_divide
@pytest.mark.div_tensor
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
@pytest.mark.div_tensor_
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
@pytest.mark.div_tensor
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
@pytest.mark.div_tensor_
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
@pytest.mark.div_scalar
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
@pytest.mark.div_scalar
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_div_scalar_scalar(dtype):
    if dtype == torch.float32:
        inp1 = float(np.float32(random.random() + 0.01))
        inp2 = float(np.float32(random.random() + 0.01))
    else:
        inp1 = random.randint(1, 100)
        inp2 = random.randint(1, 100)

    ref_out = torch.div(inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    if dtype == torch.int64:
        utils.gems_assert_equal(res_out, ref_out)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype)


# div.Tensor
# Complex
@pytest.mark.div_tensor
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
@pytest.mark.div_tensor
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
@pytest.mark.div_tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("complex_dtype", utils.COMPLEX_DTYPES)
def test_div_tensor_int(shape, complex_dtype):
    inp1 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)
    inp2 = torch.randint(1, 20, shape, device=flag_gems.device)

    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.div(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, complex_dtype, equal_nan=True)


@pytest.mark.div_scalar
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


@pytest.mark.div_scalar
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_scalar(shape, scalar, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)

    ref_out = torch.div(ref_inp, scalar)
    with flag_gems.use_gems():
        res_out = torch.div(inp, scalar)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div_scalar_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_scalar_(shape, scalar, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), False)

    ref_out = ref_inp.div_(scalar)
    with flag_gems.use_gems():
        res_out = inp.div_(scalar)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div_scalar_mode
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("rounding_mode", ["trunc", "floor"])
def test_div_scalar_mode(shape, scalar, dtype, rounding_mode):
    if dtype in (torch.float16, torch.bfloat16):
        pytest.xfail(
            "Operator bug: trunc/floor div scalar kernels fail to compile with half-precision dtypes"
        )
    if rounding_mode == "trunc" and abs(scalar) < 0.01:
        pytest.xfail(
            "Operator bug: trunc mode has off-by-one precision issues with very small divisors"
        )
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)

    ref_out = torch.div(ref_inp, scalar, rounding_mode=rounding_mode)
    with flag_gems.use_gems():
        res_out = torch.div(inp, scalar, rounding_mode=rounding_mode)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div_scalar_mode_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_scalar_mode_none(shape, scalar, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), False)

    ref_out = ref_inp.div_(scalar, rounding_mode=None)
    with flag_gems.use_gems():
        res_out = inp.div_(scalar, rounding_mode=None)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div_scalar_mode_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", [-0.999, 100.001, -111.999])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_div_scalar_mode_trunc(shape, scalar, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = ref_inp.div_(scalar, rounding_mode="trunc")
    with flag_gems.use_gems():
        res_out = inp.div_(scalar, rounding_mode="trunc")

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div_scalar_mode_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", [-0.999, 100.001, -111.999])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_div_scalar_mode_floor(shape, scalar, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = ref_inp.div_(scalar, rounding_mode="floor")
    with flag_gems.use_gems():
        res_out = inp.div_(scalar, rounding_mode="floor")

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div_tensor_mode_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_tensor_mode_inplace_none(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone(), False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = ref_inp1.div_(ref_inp2, rounding_mode=None)
    with flag_gems.use_gems():
        res_out = inp1.div_(inp2, rounding_mode=None)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div_tensor_mode_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("rounding_mode", ["trunc", "floor"])
@pytest.mark.xfail(
    reason="Operator bug: trunc/floor div kernels fail Triton compilation for float dtypes"
)
def test_div_tensor_mode_inplace_trunc_floor(shape, dtype, rounding_mode):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone(), False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = ref_inp1.div_(ref_inp2, rounding_mode=rounding_mode)
    with flag_gems.use_gems():
        res_out = inp1.div_(inp2, rounding_mode=rounding_mode)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.div_tensor_mode_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_tensor_mode_scalar_inplace_none(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone(), False)

    ref_out = ref_inp1.div_(scalar, rounding_mode=None)
    with flag_gems.use_gems():
        res_out = inp1.div_(scalar, rounding_mode=None)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div_tensor_mode_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("rounding_mode", ["trunc", "floor"])
@pytest.mark.xfail(
    reason="Operator bug: trunc/floor div scalar kernels fail Triton compilation"
)
def test_div_tensor_mode_scalar_inplace_trunc_floor(
    shape, scalar, dtype, rounding_mode
):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone(), False)

    ref_out = ref_inp1.div_(scalar, rounding_mode=rounding_mode)
    with flag_gems.use_gems():
        res_out = inp1.div_(scalar, rounding_mode=rounding_mode)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.div_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_out_tensor_tensor(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    out = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.div(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        torch.div(inp1, inp2, out=out)

    utils.gems_assert_close(out, ref_out, dtype, equal_nan=True)


@pytest.mark.div_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_out_tensor_scalar(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    out = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, False)

    ref_out = torch.div(ref_inp1, scalar)
    with flag_gems.use_gems():
        torch.div(inp1, scalar, out=out)

    utils.gems_assert_close(out, ref_out, dtype, equal_nan=True)


@pytest.mark.div_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_out_scalar_tensor(shape, scalar, dtype):
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    out = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.div(scalar, ref_inp2)
    with flag_gems.use_gems():
        torch.div(scalar, inp2, out=out)

    utils.gems_assert_close(out, ref_out, dtype, equal_nan=True)
