import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

SPECIAL_VALUES = [float("-inf"), float("inf"), -300]


@pytest.mark.log_sigmoid
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log_sigmoid(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if len(shape) == 1:
        special_inputs = torch.tensor(
            SPECIAL_VALUES, dtype=dtype, device=flag_gems.device
        )
        inp = torch.cat((inp, special_inputs))
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.nn.functional.logsigmoid(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.logsigmoid(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log_sigmoid_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log_sigmoid_backward(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if len(shape) == 1:
        special_inputs = torch.tensor(
            SPECIAL_VALUES, dtype=dtype, device=flag_gems.device
        )
        inp = torch.cat((inp, special_inputs))
    res_inp = inp.detach().requires_grad_(True)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.nn.functional.logsigmoid(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.logsigmoid(res_inp)

    out_grad = torch.randn_like(res_out)
    ref_grad = utils.to_reference(out_grad, True)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)

    with flag_gems.use_gems():
        (res_in_grad,) = torch.autograd.grad(res_out, res_inp, out_grad)

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)
