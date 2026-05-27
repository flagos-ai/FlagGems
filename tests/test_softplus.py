import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.softplus
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_softplus(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    beta = torch.rand(1).item()
    threshold = torch.rand(1).item() * 40.0
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.nn.functional.softplus(ref_inp, beta=beta, threshold=threshold)

    with flag_gems.use_gems():
        res_out = torch.nn.functional.softplus(inp, beta=beta, threshold=threshold)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.softplus_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_softplus_backward(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(res_inp, True)

    beta = torch.rand(1).item()
    threshold = torch.rand(1).item() * 40.0

    ref_out = torch.nn.functional.softplus(ref_inp, beta=beta, threshold=threshold)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.softplus(res_inp, beta=beta, threshold=threshold)

    out_grad = torch.randn_like(res_out)
    ref_grad = utils.to_reference(out_grad, True)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)

    with flag_gems.use_gems():
        (res_in_grad,) = torch.autograd.grad(res_out, res_inp, out_grad)

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)
