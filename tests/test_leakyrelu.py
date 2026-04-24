import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.leaky_relu
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("negative_slope", [0.01, 0.1, 0.2])
def test_accuracy_leaky_relu(shape, dtype, negative_slope):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(x, True)
    ref_out = torch.nn.functional.leaky_relu(ref_inp, negative_slope=negative_slope)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.leaky_relu_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("negative_slope", [0.01, 0.2])
def test_accuracy_leaky_relu_(shape, dtype, negative_slope):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref = utils.to_reference(x.clone(), True)
    ref_out = torch.nn.functional.leaky_relu_(ref, negative_slope=negative_slope).to(
        dtype
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.leaky_relu_(x, negative_slope=negative_slope)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.leaky_relu
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_leaky_relu_non_contiguous(dtype):
    x = torch.randn((19, 17), dtype=dtype, device=flag_gems.device).t()
    ref_inp = utils.to_reference(x, True)
    ref_out = torch.nn.functional.leaky_relu(ref_inp, negative_slope=0.2)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.leaky_relu
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_leaky_relu_special_values(dtype):
    x = torch.tensor(
        [float("-inf"), -1.0, -0.0, 0.0, 1.0, float("inf"), float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(x, True)
    ref_out = torch.nn.functional.leaky_relu(ref_inp, negative_slope=0.2)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.leaky_relu
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_leaky_relu_empty(dtype):
    x = torch.empty((0, 3), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(x, True)
    ref_out = torch.nn.functional.leaky_relu(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.leaky_relu(x)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.leaky_relu_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("negative_slope", [0.01, 0.2])
def test_accuracy_leaky_relu_backward(shape, dtype, negative_slope):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad = torch.randn_like(x)
    ref_inp = utils.to_reference(x, True)
    ref_grad = utils.to_reference(grad, True)

    ref_out = torch.ops.aten.leaky_relu_backward(
        ref_grad, ref_inp, negative_slope, False
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten.leaky_relu_backward(grad, x, negative_slope, False)
    utils.gems_assert_close(res_out, ref_out, dtype)
