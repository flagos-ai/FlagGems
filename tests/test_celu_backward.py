import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.celu_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_celu_backward(shape, dtype, alpha):
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_grad = utils.to_reference(grad_output)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.ops.aten.celu_backward(ref_grad, ref_inp, alpha)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.celu_backward(grad_output, inp, alpha)
    utils.gems_assert_close(res_out, ref_out, dtype)
