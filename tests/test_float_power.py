import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.float_power
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_float_power_tensor_tensor(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    exponent = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_exp = utils.to_reference(exponent, True)

    ref_out = torch.float_power(ref_inp, ref_exp)
    with flag_gems.use_gems():
        res_out = torch.float_power(inp, exponent)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.float_power
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_float_power_scalar(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.float_power(ref_inp, 2.0)
    with flag_gems.use_gems():
        res_out = torch.float_power(inp, 2.0)

    utils.gems_assert_close(res_out, ref_out, dtype)
