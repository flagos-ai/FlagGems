import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.relu6_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_relu6_backward(shape, dtype):
    """Test relu6_backward with various shapes."""
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_grad_output = utils.to_reference(grad_output)

    ref_grad_input = torch.ops.aten.relu6_backward(ref_grad_output, ref_x)

    with flag_gems.use_gems():
        res_grad_input = torch.ops.aten.relu6_backward(grad_output, x)

    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)
