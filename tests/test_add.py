import random

import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.add
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("alpha", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_add(shape, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.add(ref_inp1, ref_inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.add(inp1, inp2, alpha=alpha)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.add
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Issues #3267: Ascend NPU does not support complex32 dtype",
)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("complex_dtype", utils.COMPLEX_DTYPES)
@pytest.mark.parametrize(
    "other_type", ["complex", "float_tensor", "int_tensor", "int_scalar"]
)
def test_add_complex(shape, complex_dtype, other_type):
    inp1 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)

    if other_type == "complex":
        inp2 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)
    elif other_type == "float_tensor":
        if complex_dtype == torch.complex64:
            float_dtype = torch.float32
        elif complex_dtype == torch.complex32:
            float_dtype = torch.float16
        else:
            raise ValueError(f"Unsupported complex_dtype: {complex_dtype}")
        inp2 = torch.randn(shape, dtype=float_dtype, device=flag_gems.device)
    elif other_type == "int_tensor":
        inp2 = torch.randint(10, 20, shape, device=flag_gems.device)
    elif other_type == "int_scalar":
        inp2 = 3
    else:
        raise ValueError(f"Unknown other_type: {other_type}")

    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = (
        utils.to_reference(inp2, True) if isinstance(inp2, torch.Tensor) else inp2
    )

    ref_out = torch.add(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.add(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, complex_dtype)


@pytest.mark.add_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("alpha", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_add_(shape, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone(), True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = ref_inp1.add_(ref_inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = inp1.add_(inp2, alpha=alpha)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.add
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("alpha", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_add_tensor_scalar(shape, scalar, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = utils.to_reference(inp1, True)

    ref_out = torch.add(ref_inp1, inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.add(inp1, inp2, alpha=alpha)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.add_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("alpha", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_add_tensor_scalar_(shape, scalar, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = utils.to_reference(inp1.clone(), True)

    ref_out = ref_inp1.add_(inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = inp1.add_(inp2, alpha=alpha)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.add
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("alpha", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_add_scalar_tensor(shape, scalar, alpha, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.add(inp1, ref_inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.add(inp1, inp2, alpha=alpha)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.add
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_add_scalar_scalar(dtype):
    if dtype == torch.float32:
        inp1 = float(np.float32(random.random()))
        inp2 = float(np.float32(random.random()))
        alpha = float(np.float32(random.random()))
    else:
        inp1 = random.randint(0, 100)
        inp2 = random.randint(0, 100)
        alpha = random.randint(0, 100)

    ref_out = torch.add(inp1, inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.add(inp1, inp2, alpha=alpha)

    if dtype == torch.int64:
        utils.gems_assert_equal(res_out, ref_out)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype)
