import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES

if flag_gems.vendor_name == "cambricon":
    DIM_LIST = [0, 1]
else:
    DIM_LIST = [1]


random.seed(time.time() // 100)


@pytest.mark.log_softmax
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", DIM_LIST)
def test_log_softmax(shape, dtype, dim):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.nn.functional.log_softmax(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.log_softmax(inp, dim=dim)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log_softmax_out
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [0, 1] if flag_gems.vendor_name == "cambricon" else [1])
def test_accuracy_log_softmax_out(shape, dtype, dim):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty(shape, dtype=ref_inp.dtype, device=ref_inp.device)
    torch.ops.aten._log_softmax.out(ref_inp, dim, False, out=ref_out)

    res_out = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        torch.ops.aten._log_softmax.out(inp, dim, False, out=res_out)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log_softmax_backward
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", DIM_LIST)
def test_log_softmax_backward(shape, dtype, dim):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)

    res_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_out = torch.randn_like(res_grad)

    ref_grad = utils.to_reference(res_grad, True)
    ref_out = utils.to_reference(res_out, True)

    ref_in_grad = torch.ops.aten._log_softmax_backward_data(
        ref_grad, ref_out, dim, ref_grad.dtype
    )
    with flag_gems.use_gems():
        res_in_grad = torch.ops.aten._log_softmax_backward_data(
            res_grad, res_out, dim, dtype
        )

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim])


@pytest.mark.log_softmax_backward_data_out
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [0, 1] if flag_gems.vendor_name == "cambricon" else [1])
def test_accuracy_log_softmax_backward_out(shape, dtype, dim):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)
    res_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_out = torch.randn_like(res_grad)
    ref_grad = utils.to_reference(res_grad, True)
    ref_out = utils.to_reference(res_out, True)

    ref_in_grad = torch.empty(shape, dtype=ref_grad.dtype, device=ref_grad.device)
    torch.ops.aten._log_softmax_backward_data.out(
        ref_grad, ref_out, dim, ref_grad.dtype, out=ref_in_grad
    )

    res_in_grad = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        torch.ops.aten._log_softmax_backward_data.out(
            res_grad, res_out, dim, dtype, out=res_in_grad
        )
    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim])
