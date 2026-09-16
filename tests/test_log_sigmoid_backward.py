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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

SPECIAL_VALUES = [float("-inf"), float("inf"), -300, 0.0]


def _reference_log_sigmoid_backward(grad_output, inp, buffer):
    if flag_gems.vendor_name == "ascend":
        return grad_output * torch.sigmoid(-inp)
    return torch.ops.aten.log_sigmoid_backward(grad_output, inp, buffer)


def _reference_log_sigmoid_backward_out(grad_output, inp, buffer, *, grad_input):
    if flag_gems.vendor_name == "ascend":
        grad_input.copy_(grad_output * torch.sigmoid(-inp))
        return grad_input
    return torch.ops.aten.log_sigmoid_backward.grad_input(
        grad_output, inp, buffer, grad_input=grad_input
    )


@pytest.mark.log_sigmoid_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log_sigmoid_backward(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_grad = torch.randn_like(res_inp)
    if len(shape) == 1:
        special_inputs = torch.tensor(
            SPECIAL_VALUES, dtype=dtype, device=flag_gems.device
        )
        res_inp = torch.cat((res_inp, special_inputs))
        res_grad = torch.cat((res_grad, torch.randn_like(special_inputs)))

    ref_inp = utils.to_reference(res_inp, True)
    ref_grad = utils.to_reference(res_grad, True)
    ref_buffer = torch.exp(-torch.abs(ref_inp))
    ref_in_grad = _reference_log_sigmoid_backward(ref_grad, ref_inp, ref_buffer)

    # CUDA's native forward returns an empty buffer. The backward operator must
    # therefore not require buffer elements on device backends.
    res_buffer = torch.empty(0, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems(include=["log_sigmoid_backward"]):
        res_in_grad = torch.ops.aten.log_sigmoid_backward(res_grad, res_inp, res_buffer)

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.log_sigmoid_backward_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log_sigmoid_backward_out(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_grad = torch.randn_like(res_inp)
    if len(shape) == 1:
        special_inputs = torch.tensor(
            SPECIAL_VALUES, dtype=dtype, device=flag_gems.device
        )
        res_inp = torch.cat((res_inp, special_inputs))
        res_grad = torch.cat((res_grad, torch.randn_like(special_inputs)))

    ref_inp = utils.to_reference(res_inp, True)
    ref_grad = utils.to_reference(res_grad, True)
    ref_buffer = torch.exp(-torch.abs(ref_inp))
    ref_grad_input = torch.empty_like(ref_inp)
    _reference_log_sigmoid_backward_out(
        ref_grad, ref_inp, ref_buffer, grad_input=ref_grad_input
    )

    res_buffer = torch.empty(0, dtype=dtype, device=flag_gems.device)
    res_grad_input = torch.empty_like(res_inp)
    with flag_gems.use_gems(include=["log_sigmoid_backward_out"]):
        result = torch.ops.aten.log_sigmoid_backward.grad_input(
            res_grad, res_inp, res_buffer, grad_input=res_grad_input
        )

    assert result is res_grad_input
    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)


@pytest.mark.log_sigmoid_backward_out
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log_sigmoid_backward_out_noncontiguous(dtype):
    res_inp = torch.randn((4, 3), dtype=dtype, device=flag_gems.device).T
    res_grad = torch.randn_like(res_inp)
    res_buffer = torch.empty(0, dtype=dtype, device=flag_gems.device)
    res_grad_input = torch.empty_like(res_inp)

    ref_inp = utils.to_reference(res_inp, True)
    ref_grad = utils.to_reference(res_grad, True)
    ref_buffer = torch.exp(-torch.abs(ref_inp))
    ref_grad_input = torch.empty_like(ref_inp)
    _reference_log_sigmoid_backward_out(
        ref_grad, ref_inp, ref_buffer, grad_input=ref_grad_input
    )

    with flag_gems.use_gems(include=["log_sigmoid_backward_out"]):
        result = torch.ops.aten.log_sigmoid_backward.grad_input(
            res_grad, res_inp, res_buffer, grad_input=res_grad_input
        )

    assert result is res_grad_input
    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)


@pytest.mark.log_sigmoid_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log_sigmoid_backward_contiguous_buffer(dtype):
    res_inp = torch.randn((37, 41), dtype=dtype, device=flag_gems.device)
    res_grad = torch.randn_like(res_inp)
    res_buffer = torch.exp(-torch.abs(res_inp))

    ref_inp = utils.to_reference(res_inp, True)
    ref_grad = utils.to_reference(res_grad, True)
    ref_buffer = torch.exp(-torch.abs(ref_inp))
    ref_in_grad = _reference_log_sigmoid_backward(ref_grad, ref_inp, ref_buffer)

    with flag_gems.use_gems(include=["log_sigmoid_backward"]):
        res_in_grad = torch.ops.aten.log_sigmoid_backward(res_grad, res_inp, res_buffer)

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.log_sigmoid_backward_out
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log_sigmoid_backward_out_contiguous_buffer(dtype):
    res_inp = torch.randn((37, 41), dtype=dtype, device=flag_gems.device)
    res_grad = torch.randn_like(res_inp)
    res_buffer = torch.exp(-torch.abs(res_inp))
    res_grad_input = torch.empty_like(res_inp)

    ref_inp = utils.to_reference(res_inp, True)
    ref_grad = utils.to_reference(res_grad, True)
    ref_buffer = torch.exp(-torch.abs(ref_inp))
    ref_in_grad = _reference_log_sigmoid_backward(ref_grad, ref_inp, ref_buffer)

    with flag_gems.use_gems(include=["log_sigmoid_backward_out"]):
        result = torch.ops.aten.log_sigmoid_backward.grad_input(
            res_grad, res_inp, res_buffer, grad_input=res_grad_input
        )

    assert result is res_grad_input
    utils.gems_assert_close(res_grad_input, ref_in_grad, dtype)


@pytest.mark.log_sigmoid_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log_sigmoid_backward_via_autograd(dtype):
    res_inp = torch.randn(
        (4, 7), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    res_grad = torch.randn_like(res_inp)
    ref_inp = utils.to_reference(res_inp.detach(), True).requires_grad_()
    ref_grad = utils.to_reference(res_grad, True)

    if flag_gems.vendor_name == "ascend":
        ref_in_grad = ref_grad * torch.sigmoid(-ref_inp)
    else:
        ref_out, _ = torch.ops.aten.log_sigmoid_forward(ref_inp)
        ref_out.backward(ref_grad)
        ref_in_grad = ref_inp.grad

    with flag_gems.use_gems(include=["log_sigmoid_backward"]):
        res_out, _ = torch.ops.aten.log_sigmoid_forward(res_inp)
        res_out.backward(res_grad)

    utils.gems_assert_close(res_inp.grad, ref_in_grad, dtype)
