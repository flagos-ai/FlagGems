import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.rrelu_with_noise_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_rrelu_with_noise_backward(shape, dtype):
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    noise = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    lower, upper = 0.125, 1.0 / 3.0
    ref_grad = utils.to_reference(grad_output)
    ref_inp = utils.to_reference(inp)
    ref_noise = utils.to_reference(noise)
    ref_out = torch.ops.aten.rrelu_with_noise_backward(
        ref_grad, ref_inp, ref_noise, lower, upper, True, False
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten.rrelu_with_noise_backward(
            grad_output, inp, noise, lower, upper, True, False
        )
    utils.gems_assert_close(res_out, ref_out, dtype)
