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


@pytest.mark.lerp_tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lerp(shape, dtype):
    torch.manual_seed(0)

    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    end = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    input.uniform_(-0.1, 0.1)
    end.uniform_(-0.1, 0.1)
    weight.uniform_(-0.1, 0.1)

    ref_input = utils.to_reference(input)
    ref_end = utils.to_reference(end)
    ref_weight = utils.to_reference(weight)

    ref_out = torch.lerp(ref_input, ref_end, weight=5.0)
    res_out = flag_gems.lerp_scalar(input, end, weight=5.0)
    utils.gems_assert_close(res_out, ref_out, dtype)

    ref_out = torch.lerp(ref_input, ref_end, weight=ref_weight)
    res_out = flag_gems.lerp_tensor(input, end, weight=weight)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.lerp_tensor_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lerp_(shape, dtype):
    torch.manual_seed(0)

    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    end = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    input.uniform_(-0.1, 0.1)
    end.uniform_(-0.1, 0.1)
    weight.uniform_(-0.1, 0.1)

    ref_input = utils.to_reference(input)
    ref_end = utils.to_reference(end)
    ref_weight = utils.to_reference(weight)

    ref_out = ref_input.clone().lerp_(ref_end, weight=5.0)
    res_out = flag_gems.lerp_scalar_(input.clone(), end, weight=5.0)
    utils.gems_assert_close(res_out, ref_out, dtype)

    ref_out = ref_input.clone().lerp_(ref_end, weight=ref_weight)
    res_out = flag_gems.lerp_tensor_(input.clone(), end, weight=weight)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.lerp_scalar
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lerp_scalar(shape, dtype):
    torch.manual_seed(0)

    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    end = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    input.uniform_(-0.1, 0.1)
    end.uniform_(-0.1, 0.1)

    ref_input = utils.to_reference(input)
    ref_end = utils.to_reference(end)

    ref_out = torch.lerp(ref_input, ref_end, weight=5.0)
    res_out = flag_gems.lerp_scalar(input, end, weight=5.0)
    utils.gems_assert_close(res_out, ref_out, dtype)

    ref_out = torch.lerp(ref_input, ref_end, weight=-2.0)
    res_out = flag_gems.lerp_scalar(input, end, weight=-2.0)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.lerp_scalar_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lerp_scalar_(shape, dtype):
    torch.manual_seed(0)

    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    end = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    input.uniform_(-0.1, 0.1)
    end.uniform_(-0.1, 0.1)

    ref_input = utils.to_reference(input)
    ref_end = utils.to_reference(end)

    ref_out = ref_input.clone().lerp_(ref_end, weight=0.2)
    res_out = flag_gems.lerp_scalar_(input.clone(), end, weight=0.2)
    utils.gems_assert_close(res_out, ref_out, dtype)

    ref_out = ref_input.clone().lerp_(ref_end, weight=0.8)
    res_out = flag_gems.lerp_scalar_(input.clone(), end, weight=0.8)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.lerp_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lerp_tensor_backward(shape, dtype):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    end = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    weight = torch.randn(
        shape, dtype=dtype, device=flag_gems.device, requires_grad=True
    )

    res_out = flag_gems.lerp_tensor(input, end, weight)
    ref_input = utils.to_reference(input, True)
    ref_end = utils.to_reference(end, True)
    ref_weight = utils.to_reference(weight, True)
    ref_out = torch.lerp(ref_input, ref_end, ref_weight)

    out_grad = torch.randn_like(res_out)
    ref_out_grad = utils.to_reference(out_grad, True)

    res_grads = torch.autograd.grad(res_out, (input, end, weight), out_grad)
    ref_grads = torch.autograd.grad(
        ref_out, (ref_input, ref_end, ref_weight), ref_out_grad
    )
    for res_grad, ref_grad in zip(res_grads, ref_grads):
        utils.gems_assert_close(res_grad, ref_grad, dtype)

    res_direct = flag_gems.lerp_backward(out_grad, input, end, weight)
    for res_grad, ref_grad in zip(res_direct, ref_grads):
        utils.gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.lerp_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lerp_scalar_backward(shape, dtype):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    end = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    weight = 0.3

    res_out = flag_gems.lerp_scalar(input, end, weight)
    ref_input = utils.to_reference(input, True)
    ref_end = utils.to_reference(end, True)
    ref_out = torch.lerp(ref_input, ref_end, weight)

    out_grad = torch.randn_like(res_out)
    ref_out_grad = utils.to_reference(out_grad, True)

    res_grads = torch.autograd.grad(res_out, (input, end), out_grad)
    ref_grads = torch.autograd.grad(ref_out, (ref_input, ref_end), ref_out_grad)
    for res_grad, ref_grad in zip(res_grads, ref_grads):
        utils.gems_assert_close(res_grad, ref_grad, dtype)

    res_direct = flag_gems.lerp_backward(out_grad, input, end, weight)
    assert res_direct[2] is None
    utils.gems_assert_close(res_direct[0], ref_grads[0], dtype)
    utils.gems_assert_close(res_direct[1], ref_grads[1], dtype)
