import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.leaky_relu
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_leaky_relu(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.nn.functional.leaky_relu(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.leaky_relu(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.leaky_relu_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_leaky_relu_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = torch.nn.functional.leaky_relu_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.leaky_relu_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.leaky_relu_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_leaky_relu_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty_like(ref_inp)
    torch.ops.aten.leaky_relu.out(ref_inp, out=ref_out)
    with flag_gems.use_gems():
        res_out = torch.empty_like(inp)
        torch.ops.aten.leaky_relu.out(inp, out=res_out)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.leaky_relu
@pytest.mark.parametrize("negative_slope", [0.0, 0.01, 0.1, 0.5, 1.0])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_leaky_relu_slope(negative_slope, dtype):
    inp = torch.randn((64, 64), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.nn.functional.leaky_relu(ref_inp, negative_slope=negative_slope)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.leaky_relu(inp, negative_slope=negative_slope)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.leaky_relu
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_leaky_relu_special_values(dtype):
    inp = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, float("inf"), float("-inf"), float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.nn.functional.leaky_relu(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.leaky_relu(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.leaky_relu
@pytest.mark.parametrize("shape", [(0,), (4, 0), (2, 0, 3)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_leaky_relu_empty(shape, dtype):
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.nn.functional.leaky_relu(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.leaky_relu(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.leaky_relu
@pytest.mark.parametrize("shape", [(17, 33), (5, 7, 9)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_leaky_relu_noncontiguous(shape, dtype):
    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = base.transpose(-1, -2)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.nn.functional.leaky_relu(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.leaky_relu(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)
