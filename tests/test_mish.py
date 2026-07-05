import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.mish
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_mish(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.mish(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.mish(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.mish_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_mish_backward(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_out = torch.randn_like(inp)

    ref_inp = utils.to_reference(inp, True)
    ref_grad_out = utils.to_reference(grad_out, True)

    ref_in_grad = torch.ops.aten.mish_backward(ref_grad_out, ref_inp)
    with flag_gems.use_gems():
        res_in_grad = torch.ops.aten.mish_backward(grad_out, inp)

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)
