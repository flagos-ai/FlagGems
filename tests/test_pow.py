import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.pow_tensor_tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_pow_tensor_tensor(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    if flag_gems.vendor_name == "kunlunxin" or flag_gems.vendor_name == "ascend":
        inp1 = inp1.uniform_(-1, 1)
        inp2 = inp2.uniform_(-1, 1)

    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.pow(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.pow(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.pow_tensor_tensor_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_pow_tensor_tensor_(shape, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Skiping fp32 pow test on tsingmicro platform")

    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    if flag_gems.vendor_name == "kunlunxin":
        inp1 = inp1.uniform_(-1, 1)
        inp2 = inp2.uniform_(-1, 1)

    ref_inp1 = utils.to_reference(inp1.clone(), True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = ref_inp1.pow_(ref_inp2)
    with flag_gems.use_gems():
        res_out = inp1.pow_(inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.pow_tensor_scalar
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "scalar",
    utils.SCALARS
    + ([1, 2, 3, 4, 5, 8] if flag_gems.vendor_name == "cambricon" else []),
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_pow_tensor_scalar(scalar, shape, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Skiping fp32 pow test on tsingmicro platform")

    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)

    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar

    if flag_gems.vendor_name == "kunlunxin" or flag_gems.vendor_name == "ascend":
        if scalar == -0.999:
            inp1 = inp1.uniform_(-1, 1)
        elif scalar == -111.999 and dtype == torch.float16:
            inp1 = inp1.uniform_(-1, 1)
        else:
            inp1 = inp1.uniform_(-0.1, 0.1)

    ref_inp1 = utils.to_reference(inp1, True)

    ref_out = torch.pow(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.pow(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.pow_tensor_scalar_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_pow_tensor_scalar_(scalar, shape, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Skiping fp32 pow test on tsingmicro platform")

    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)

    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar

    if flag_gems.vendor_name == "kunlunxin":
        if scalar == -0.999:
            inp1 = inp1.uniform_(-1, 1)
        elif scalar == -111.999 and dtype == torch.float16:
            inp1 = inp1.uniform_(-1, 1)
        else:
            inp1 = inp1.uniform_(-0.1, 0.1)

    ref_inp1 = utils.to_reference(inp1.clone(), True)

    ref_out = ref_inp1.pow_(inp2)
    with flag_gems.use_gems():
        res_out = inp1.pow_(inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.pow_scalar_tensor
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_pow_scalar_tensor(scalar, shape, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Skiping fp32 pow test on tsingmicro platform")

    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    if flag_gems.vendor_name == "kunlunxin" or flag_gems.vendor_name == "ascend":
        inp2 = inp2.uniform_(-1, 1)

    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.pow(inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.pow(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
