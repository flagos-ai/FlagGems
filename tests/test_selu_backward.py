import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# SELU constants from PyTorch
ALPHA = 1.6732632423543772848170429916717
SCALE = 1.0507009873554804934193349852946


@pytest.mark.selu_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_selu_backward(shape, dtype):
    """Test selu_backward with various shapes."""
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_grad_output = utils.to_reference(grad_output)

    ref_grad_input = torch.ops.aten.selu_backward(ref_grad_output, ref_x, ALPHA, SCALE)

    with flag_gems.use_gems():
        res_grad_input = torch.ops.aten.selu_backward(grad_output, x, ALPHA, SCALE)

    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)
