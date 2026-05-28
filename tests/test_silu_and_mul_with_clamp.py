import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

SILU_AND_MUL_WITH_CLAMP_LIMITS = [3.0, 7.0]


@pytest.mark.silu_and_mul_with_clamp
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("limit", SILU_AND_MUL_WITH_CLAMP_LIMITS)
def test_silu_and_mul_with_clamp(shape, dtype, limit):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_gate = torch.clamp(ref_inp1, max=limit)
    ref_up = torch.clamp(ref_inp2, min=-limit, max=limit)
    ref_out = torch.mul(torch.nn.functional.silu(ref_gate), ref_up)
    with flag_gems.use_gems():
        res_out = flag_gems.silu_and_mul_with_clamp(inp1, inp2, limit)

    out_grad = torch.randn_like(res_out)
    ref_grad = utils.to_reference(out_grad, True)

    ref_inp1_grad, ref_inp2_grad = torch.autograd.grad(
        ref_out, (ref_inp1, ref_inp2), ref_grad
    )

    res_inp1_grad, res_inp2_grad = torch.autograd.grad(res_out, (inp1, inp2), out_grad)

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(res_inp1_grad, ref_inp1_grad, dtype)
    utils.gems_assert_close(res_inp2_grad, ref_inp2_grad, dtype)


@pytest.mark.silu_and_mul_with_clamp_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("limit", SILU_AND_MUL_WITH_CLAMP_LIMITS)
def test_silu_and_mul_with_clamp_out(shape, dtype, limit):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_gate = torch.clamp(ref_inp1, max=limit)
    ref_up = torch.clamp(ref_inp2, min=-limit, max=limit)
    ref_out = torch.mul(torch.nn.functional.silu(ref_gate), ref_up)

    out = torch.empty_like(inp1)
    with flag_gems.use_gems():
        ret = flag_gems.silu_and_mul_with_clamp_out(inp1, inp2, out, limit)

    assert ret is out
    utils.gems_assert_close(out, ref_out, dtype)
