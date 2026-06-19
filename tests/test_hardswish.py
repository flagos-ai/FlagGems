import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.hardswish_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_hardswish_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.hardswish_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.hardswish_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.hardswish_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_hardswish_backward(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.nn.functional.hardswish(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.hardswish(res_inp)

    out_grad = torch.randn_like(res_out)
    ref_grad = utils.to_reference(out_grad, True)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)

    with flag_gems.use_gems():
        (res_in_grad,) = torch.autograd.grad(res_out, res_inp, out_grad)

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)
