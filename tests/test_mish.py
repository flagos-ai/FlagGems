import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.mish
@pytest.mark.parametrize("shape", [(8,), (16, 16), (32, 32, 32)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_mish(shape, dtype):
    """Test mish activation with various shapes and dtypes."""
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input)
    ref_output = torch.ops.aten.mish(ref_input)

    with flag_gems.use_gems():
        res_output = torch.ops.aten.mish(input)

    utils.gems_assert_close(res_output, ref_output, dtype)


@pytest.mark.mish_backward
@pytest.mark.parametrize("shape", [(8,), (16, 16), (32, 32, 32)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_mish_backward(shape, dtype):
    """Test mish_backward with various shapes and dtypes."""
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)

    ref_input = utils.to_reference(input)
    ref_output = torch.ops.aten.mish(ref_input)
    ref_grad = torch.randn_like(ref_output)
    ref_output.backward(ref_grad)

    with flag_gems.use_gems():
        res_output = torch.ops.aten.mish(input)
        res_grad = torch.randn_like(res_output)
        res_output.backward(res_grad)

    utils.gems_assert_close(input.grad, ref_input.grad, dtype)
