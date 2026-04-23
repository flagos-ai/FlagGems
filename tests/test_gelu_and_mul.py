import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.gelu_and_mul
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_and_mul(shape, approximate, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.mul(
        torch.nn.functional.gelu(ref_inp1, approximate=approximate), ref_inp2
    )
    with flag_gems.use_gems():
        res_out = flag_gems.gelu_and_mul(inp1, inp2, approximate)

    out_grad = torch.randn_like(res_out)
    ref_grad = utils.to_reference(out_grad, True)

    ref_inp1_grad, ref_inp2_grad = torch.autograd.grad(
        ref_out, (ref_inp1, ref_inp2), ref_grad
    )

    res_inp1_grad, res_inp2_grad = torch.autograd.grad(res_out, (inp1, inp2), out_grad)

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(res_inp1_grad, ref_inp1_grad, dtype)
    utils.gems_assert_close(res_inp2_grad, ref_inp2_grad, dtype)
