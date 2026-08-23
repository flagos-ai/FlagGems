import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.float_power_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_float_power_(shape, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Skipping fp32 float_power_ test on tsingmicro platform")

    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    if flag_gems.vendor_name == "kunlunxin":
        inp1 = inp1.uniform_(-1, 1)
        inp2 = inp2.uniform_(-1, 1)

    ref_inp1 = utils.to_reference(inp1.clone(), True)
    ref_inp2 = utils.to_reference(inp2, True)

    # PyTorch's float_power_ has issues with non-float64 types, so use workaround
    ref_out = torch.float_power(ref_inp1, ref_inp2)
    ref_inp1.copy_(ref_out)
    ref_out = ref_inp1

    with flag_gems.use_gems():
        res_out = torch.ops.aten.float_power_.Tensor(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.float_power_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_float_power_scalar(shape, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Skipping fp32 float_power_ test on tsingmicro platform")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    scalar = 2.0

    if flag_gems.vendor_name == "kunlunxin":
        inp = inp.uniform_(-1, 1)

    ref_inp = utils.to_reference(inp.clone(), True)

    # PyTorch's float_power_ has issues with non-float64 types, so use workaround
    ref_out = torch.float_power(ref_inp, scalar)
    ref_inp.copy_(ref_out)
    ref_out = ref_inp

    with flag_gems.use_gems():
        res_out = torch.ops.aten.float_power_.Scalar(inp, scalar)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
