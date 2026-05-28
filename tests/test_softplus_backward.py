import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.softplus_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("beta", [1.0, 2.0])
@pytest.mark.parametrize("threshold", [20.0])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_softplus_backward(shape, dtype, beta, threshold):
    """Test softplus_backward with various shapes and parameters."""
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_grad_output = utils.to_reference(grad_output)

    ref_grad_input = torch.ops.aten.softplus_backward(ref_grad_output, ref_x, beta, threshold)

    with flag_gems.use_gems():
        res_grad_input = torch.ops.aten.softplus_backward(grad_output, x, beta, threshold)

    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)
