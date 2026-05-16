import random

import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.sub
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("alpha", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sub_tensor_tensor(shape, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.sub(ref_inp1, ref_inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.sub(inp1, inp2, alpha=alpha)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.sub
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("alpha", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sub_tensor_scalar(shape, scalar, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = utils.to_reference(inp1, True)

    ref_out = torch.sub(ref_inp1, inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.sub(inp1, inp2, alpha=alpha)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.sub_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("alpha", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sub_tensor_tensor_(shape, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone(), True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = ref_inp1.sub_(ref_inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = inp1.sub_(inp2, alpha=alpha)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.sub_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("alpha", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sub_tensor_scalar_(shape, scalar, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = utils.to_reference(inp1.clone(), True)

    ref_out = ref_inp1.sub_(inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = inp1.sub_(inp2, alpha=alpha)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.sub
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("alpha", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sub_scalar_tensor(shape, scalar, alpha, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.sub(inp1, ref_inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.sub(inp1, inp2, alpha=alpha)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.sub
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_sub_scalar_scalar(dtype):
    if dtype == torch.float32:
        inp1 = float(np.float32(random.random()))
        inp2 = float(np.float32(random.random()))
        alpha = float(np.float32(random.random()))
    else:
        inp1 = random.randint(0, 100)
        inp2 = random.randint(0, 100)
        alpha = random.randint(0, 100)

    ref_out = torch.sub(inp1, inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.sub(inp1, inp2, alpha=alpha)

    if dtype == torch.int64:
        utils.gems_assert_equal(res_out, ref_out)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.sub
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Issues #3267: Ascend NPU does not support complex32 dtype",
)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("complex_dtype", utils.COMPLEX_DTYPES)
def test_sub_complex_complex(shape, complex_dtype):
    inp1 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)

    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.sub(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.sub(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, complex_dtype)


@pytest.mark.sub
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Issues #3267: Ascend NPU does not support complex32 dtype",
)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("complex_dtype", utils.COMPLEX_DTYPES)
def test_sub_complex_float_tensor(shape, complex_dtype):
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

    ref_out = torch.sub(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.sub(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, complex_dtype)


@pytest.mark.sub
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Issues #3267: Ascend NPU does not support complex32 dtype",
)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("complex_dtype", utils.COMPLEX_DTYPES)
def test_sub_complex_int_tensor(shape, complex_dtype):
    inp1 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)
    inp2 = torch.randint(10, 20, shape, device=flag_gems.device)

    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.sub(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.sub(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, complex_dtype)


@pytest.mark.sub
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Issues #3267: Ascend NPU does not support complex32 dtype",
)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("complex_dtype", utils.COMPLEX_DTYPES)
def test_sub_complex_int_scalar(shape, complex_dtype):
    inp1 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)
    inp2 = torch.randint(10, 20, shape, device=flag_gems.device)

    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = (
        utils.to_reference(inp2, True) if isinstance(inp2, torch.Tensor) else inp2
    )

    ref_out = torch.sub(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.sub(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, complex_dtype)
