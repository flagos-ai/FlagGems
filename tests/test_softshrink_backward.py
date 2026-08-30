import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.softshrink_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("lambd", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_softshrink_backward(shape, dtype, lambd):
    """Test softshrink_backward with various shapes and lambda values."""
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_grad_output = utils.to_reference(grad_output)

    ref_grad_input = torch.ops.aten.softshrink_backward(ref_grad_output, ref_x, lambd)

    with flag_gems.use_gems():
        res_grad_input = torch.ops.aten.softshrink_backward(grad_output, x, lambd)

    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)
