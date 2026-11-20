# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    if flag_gems.vendor_name == "sunrise" and shape == (200, 40999, 3):
        pytest.skip("Issue #3836: Skip for big shape, '--ref cpu' too slow.")
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch._log_softmax(ref_inp, dim=dim, half_to_float=False)
    res_out = flag_gems.log_softmax(inp, dim=dim)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log_softmax_out
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [0, 1] if flag_gems.vendor_name == "cambricon" else [1])
def test_accuracy_log_softmax_out(shape, dtype, dim):
    if flag_gems.vendor_name == "sunrise" and shape == (200, 40999, 3):
        pytest.skip("Issue #3836: Skip for big shape, '--ref cpu' too slow.")
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty(shape, dtype=ref_inp.dtype, device=ref_inp.device)
    torch._log_softmax(ref_inp, dim, False, out=ref_out)

    res_out = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    flag_gems.log_softmax_out(inp, dim, out=res_out)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log_softmax_backward_data
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", DIM_LIST)
def test_log_softmax_backward_data(shape, dtype, dim):
    if flag_gems.vendor_name == "sunrise" and shape == (200, 40999, 3):
        pytest.skip("Issue #3836: Skip for big shape, '--ref cpu' too slow.")
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)

    res_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_out = torch.randn_like(res_grad)

    ref_grad = utils.to_reference(res_grad, True)
    ref_out = utils.to_reference(res_out, True)

    ref_in_grad = torch._log_softmax_backward_data(
        ref_grad, ref_out, dim, ref_grad.dtype
    )
    res_in_grad = flag_gems.log_softmax_backward(res_grad, res_out, dim, dtype)

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim])


@pytest.mark.log_softmax_backward_data_out
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [0, 1] if flag_gems.vendor_name == "cambricon" else [1])
def test_accuracy_log_softmax_backward_out(shape, dtype, dim):
    if flag_gems.vendor_name == "sunrise" and shape == (200, 40999, 3):
        pytest.skip("Issue #3836: Skip for big shape, '--ref cpu' too slow.")
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)
    res_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_out = torch.randn_like(res_grad)
    ref_grad = utils.to_reference(res_grad, True)
    ref_out = utils.to_reference(res_out, True)

    ref_in_grad = torch.empty(shape, dtype=ref_grad.dtype, device=ref_grad.device)
    torch._log_softmax_backward_data(
        ref_grad, ref_out, dim, ref_grad.dtype, out=ref_in_grad
    )

    res_in_grad = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    flag_gems.log_softmax_backward_out(res_grad, res_out, dim, dtype, out=res_in_grad)
    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim])

@pytest.mark.log_softmax_backward
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", DIM_LIST)
def test_log_softmax_backward(shape, dtype, dim):
    if flag_gems.vendor_name == "sunrise" and shape == (200, 40999, 3):
        pytest.skip("Issue #3836: Skip for big shape, '--ref cpu' too slow.")
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    grad_output = torch.randn_like(inp)
    ref_grad_output = utils.to_reference(grad_output, True)

    # The forward output is the saved tensor the backward consumes.
    output = torch._log_softmax(ref_inp, dim=dim, half_to_float=False)
    res_output = flag_gems.log_softmax(inp, dim=dim)

    ref_in_grad = torch._log_softmax_backward_data(
        ref_grad_output, output, dim, ref_grad_output.dtype
    )
    res_in_grad = flag_gems.log_softmax_backward(grad_output, res_output, dim, dtype)
    utils.gems_assert_close(res_output, output, dtype)
    # bfloat16 has a 1/128 mantissa, so a fixed 1e-4 atol sits below one ulp of
    # these gradients; widen it to the dtype's own resolution.
    utils.gems_assert_close(
        res_in_grad,
        ref_in_grad,
        dtype,
        reduce_dim=shape[dim],
        atol=max(1e-4, 2 * torch.finfo(dtype).eps),
    )


@pytest.mark.log_softmax_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_log_softmax_backward_autograd(dtype):
    """FlagGems owns ``aten::_log_softmax_backward_data``, so the gradient that
    ``torch._log_softmax`` autograd produces must match the native backward."""
    shape, dim = (16, 33), -1
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp.requires_grad_()
    ref_inp = utils.to_reference(inp.detach(), True).requires_grad_()
    grad_output = torch.randn_like(inp)

    (in_grad,) = torch.autograd.grad(
        torch._log_softmax(inp, dim, False), inp, grad_output
    )
    (ref_in_grad,) = torch.autograd.grad(
        torch._log_softmax(ref_inp, dim, False),
        ref_inp,
        utils.to_reference(grad_output),
    )
    utils.gems_assert_close(
        in_grad,
        ref_in_grad,
        dtype,
        reduce_dim=shape[dim],
        atol=max(1e-4, 2 * torch.finfo(dtype).eps),
    )


@pytest.mark.log_softmax_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_log_softmax_backward_non_last_dim(dtype):
    # A middle reduction dimension leaves inner > 1, so the row is strided.
    shape, dim = (7, 11, 5), 1
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    grad_output = torch.randn_like(inp)
    ref_grad_output = utils.to_reference(grad_output, True)

    output = torch._log_softmax(ref_inp, dim, False)
    res_output = flag_gems.log_softmax(inp, dim=dim)
    res_in_grad = flag_gems.log_softmax_backward(grad_output, res_output, dim, dtype)
    ref_in_grad = torch._log_softmax_backward_data(
        ref_grad_output, output, dim, ref_grad_output.dtype
    )
    # bfloat16 has a 1/128 mantissa, so a fixed 1e-4 atol sits below one ulp of
    # these gradients; widen it to the dtype's own resolution.
    utils.gems_assert_close(
        res_in_grad,
        ref_in_grad,
        dtype,
        reduce_dim=shape[dim],
        atol=max(1e-4, 2 * torch.finfo(dtype).eps),
    )


@pytest.mark.log_softmax_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_log_softmax_backward_expanded_grad(dtype):
    # The incoming gradient is a broadcast view; the backward must not require it
    # to be contiguous and must not write through it.
    shape, dim = (6, 13), -1
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    grad_output = torch.randn((shape[0], 1), dtype=dtype, device=flag_gems.device)
    grad_output = grad_output.expand(shape)
    ref_grad_output = utils.to_reference(grad_output, True)

    output = torch._log_softmax(ref_inp, dim, False)
    res_output = flag_gems.log_softmax(inp, dim=dim)
    res_in_grad = flag_gems.log_softmax_backward(grad_output, res_output, dim, dtype)
    ref_in_grad = torch._log_softmax_backward_data(
        ref_grad_output, output, dim, ref_grad_output.dtype
    )
    utils.gems_assert_close(
        res_in_grad,
        ref_in_grad,
        dtype,
        reduce_dim=shape[dim],
        atol=max(1e-4, 2 * torch.finfo(dtype).eps),
    )
