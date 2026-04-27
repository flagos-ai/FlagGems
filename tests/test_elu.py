import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.elu
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_elu(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    alpha = torch.rand(1).item()

    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.nn.functional.elu(ref_inp, alpha)

    with flag_gems.use_gems():
        res_out = torch.nn.functional.elu(inp, alpha)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.elu_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_elu_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    alpha = torch.rand(1).item()

    res_inp = inp.clone().to(flag_gems.device)
    inp_clone = inp.clone()
    ref_inp = utils.to_reference(inp_clone, True)
    torch.nn.functional.elu_(ref_inp, alpha)

    with flag_gems.use_gems():
        torch.nn.functional.elu_(res_inp, alpha)

    utils.gems_assert_close(res_inp, ref_inp, dtype)


@pytest.mark.elu_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("is_result", [True, False])
def test_elu_backward(shape, dtype, is_result):
    alpha = torch.rand(1).item()
    scale = 1.0
    input_scale = 1.0

    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_grad_out = torch.randn_like(res_inp)

    if is_result:
        res_self_or_result = torch.ops.aten.elu(res_inp, alpha, scale, input_scale)
    else:
        res_self_or_result = res_inp

    ref_grad_out = utils.to_reference(res_grad_out, True)
    ref_self_or_result = utils.to_reference(res_self_or_result, True)

    ref_in_grad = torch.ops.aten.elu_backward(
        ref_grad_out, alpha, scale, input_scale, is_result, ref_self_or_result
    )

    with flag_gems.use_gems():
        res_in_grad = torch.ops.aten.elu_backward(
            res_grad_out, alpha, scale, input_scale, is_result, res_self_or_result
        )

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)
