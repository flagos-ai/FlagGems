import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg


@pytest.mark.leaky_relu
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_leaky_relu(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    negative_slope = 0.01
    ref_out = torch.nn.functional.leaky_relu(ref_inp, negative_slope=negative_slope)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.leaky_relu(res_inp, negative_slope=negative_slope)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.leaky_relu_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_leaky_relu_(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp.clone(), True)

    negative_slope = 0.01
    ref_out = torch.nn.functional.leaky_relu_(ref_inp, negative_slope=negative_slope)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.leaky_relu_(
            res_inp, negative_slope=negative_slope
        )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.leaky_relu_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_leaky_relu_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    negative_slope = 0.01

    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.empty(shape, dtype=ref_inp.dtype, device=ref_inp.device)
    torch.ops.aten.leaky_relu.out(ref_inp, negative_slope=negative_slope, out=ref_out)

    out = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        torch.ops.aten.leaky_relu.out(inp, negative_slope=negative_slope, out=out)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.leaky_relu_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_leaky_relu_backward(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(res_inp, True)

    negative_slope = 0.01
    ref_out = torch.nn.functional.leaky_relu(ref_inp, negative_slope=negative_slope)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.leaky_relu(res_inp, negative_slope=negative_slope)

    out_grad = torch.randn_like(res_out)
    ref_grad = utils.to_reference(out_grad, True)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)

    with flag_gems.use_gems():
        (res_in_grad,) = torch.autograd.grad(res_out, res_inp, out_grad)

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)
