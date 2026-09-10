import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.logit_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logit_backward(shape, dtype):
    # logit is defined for inputs in (0, 1), so generate self in that range
    res_inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    res_grad = torch.randn_like(res_inp)

    ref_inp = utils.to_reference(res_inp, True)
    ref_grad = utils.to_reference(res_grad, True)

    ref_out = torch.ops.aten.logit_backward(ref_grad, ref_inp)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.logit_backward(res_grad, res_inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.logit_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("eps", [1e-6, 1e-4, 0.5])
def test_logit_backward_eps(shape, dtype, eps):
    # logit is defined for inputs in (0, 1), so generate self in that range
    res_inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    res_grad = torch.randn_like(res_inp)

    ref_inp = utils.to_reference(res_inp, True)
    ref_grad = utils.to_reference(res_grad, True)

    ref_out = torch.ops.aten.logit_backward(ref_grad, ref_inp, eps)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.logit_backward(res_grad, res_inp, eps)

    utils.gems_assert_close(res_out, ref_out, dtype)
