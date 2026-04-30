import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIM_LIST = [1]
    SHAPES = [(1, 256)]
    BACKWARD_SHAPES = [(1, 256)]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIM_LIST = [0, 1]
    SHAPES = [(1, 256), (4096, 256), (200, 2560, 3), (1, 0, 128, 512)]
    BACKWARD_SHAPES = [(1, 256), (4096, 256), (200, 2560, 3)]

random.seed(time.time() // 100)


# Issue 2852: This fails at (1, 2) (200, 40999, 3)
@pytest.mark.softmax
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("neg_inf", [True, False])
def test_softmax(shape, dtype, dim, neg_inf):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if neg_inf:
        inp = torch.where(inp < 0.0, float("-inf"), inp)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.nn.functional.softmax(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.softmax(inp, dim=dim)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.softmax_out
@pytest.mark.parametrize(
    "shape",
    [(1, 256)]
    if cfg.QUICK_MODE
    else [(1, 256), (4096, 256), (200, 2560, 3), (1, 0, 128, 512)],
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("neg_inf", [True, False])
def test_softmax_out(shape, dtype, dim, neg_inf):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if neg_inf:
        inp = torch.where(inp < 0.0, float("-inf"), inp)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty(shape, dtype=ref_inp.dtype, device=ref_inp.device)
    torch.ops.aten._softmax.out(ref_inp, dim, False, out=ref_out)

    res_out = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        torch.ops.aten._softmax.out(inp, dim, False, out=res_out)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.softmax_backward_out
@pytest.mark.parametrize(
    "shape", [(1, 256)] if cfg.QUICK_MODE else [(1, 256), (4096, 256), (200, 2560, 3)]
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("neg_inf", [True, False])
def test_softmax_backward_out(shape, dtype, dim, neg_inf):
    if shape[dim] == 1 and flag_gems.vendor_name == "kunlunxin":
        pytest.skip(
            "#2851 _softmax_backward_data short-circuits to zero when reduction dim "
            "is 1, while the Triton kernel computes normally with synthetic inputs, "
            "causing a mismatch that does not reflect a real correctness issue."
        )
    res_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if neg_inf:
        res_grad = torch.where(res_grad < 0.0, float("-inf"), res_grad)
    res_out = torch.randn_like(res_grad)

    ref_grad = utils.to_reference(res_grad, True)
    ref_out = utils.to_reference(res_out, True)

    ref_in_grad = torch.empty(shape, dtype=ref_grad.dtype, device=ref_grad.device)
    torch.ops.aten._softmax_backward_data.out(
        ref_grad, ref_out, dim, ref_grad.dtype, grad_input=ref_in_grad
    )

    res_in_grad = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        torch.ops.aten._softmax_backward_data.out(
            res_grad, res_out, dim, dtype, grad_input=res_in_grad
        )
    utils.gems_assert_close(
        res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim], equal_nan=True
    )


@pytest.mark.softmax_backward
@pytest.mark.parametrize("shape", BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("neg_inf", [True, False])
def test_softmax_backward(shape, dtype, dim, neg_inf):
    if shape[dim] == 1 and flag_gems.vendor_name == "kunlunxin":
        pytest.skip(
            "XPU _softmax_backward_data short-circuits to zero when reduction dim "
            "is 1, while the Triton kernel computes normally with synthetic inputs, "
            "causing a mismatch that does not reflect a real correctness issue."
        )

    res_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if neg_inf:
        res_grad = torch.where(res_grad < 0.0, float("-inf"), res_grad)
    res_out = torch.randn_like(res_grad)

    ref_grad = utils.to_reference(res_grad, True)
    ref_out = utils.to_reference(res_out, True)

    ref_in_grad = torch.ops.aten._softmax_backward_data(
        ref_grad, ref_out, dim, ref_grad.dtype
    )
    with flag_gems.use_gems():
        res_in_grad = torch.ops.aten._softmax_backward_data(
            res_grad, res_out, dim, dtype
        )

    utils.gems_assert_close(
        res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim], equal_nan=True
    )
