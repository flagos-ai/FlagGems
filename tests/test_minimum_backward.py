import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.minimum_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_minimum_backward(shape, dtype):
    """Test minimum_backward with various shapes."""
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_y = utils.to_reference(y)
    ref_grad = utils.to_reference(grad_output)

    ref_grad_input = torch.ops.aten.minimum_backward(ref_grad, ref_x, ref_y)

    with flag_gems.use_gems():
        res_grad_input = torch.ops.aten.minimum_backward(grad_output, x, y)

    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)
