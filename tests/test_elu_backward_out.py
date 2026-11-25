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


@pytest.mark.elu_backward_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("is_result", [True, False])
def test_elu_backward_out(shape, dtype, is_result):
    alpha = 1.0
    scale = 1.0
    input_scale = 1.0

    res_self_or_result = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_grad_out = torch.randn_like(res_self_or_result)
    if is_result:
        res_self_or_result = torch.ops.aten.elu(
            res_self_or_result, alpha, scale, input_scale
        )

    ref_grad_out = utils.to_reference(res_grad_out, True)
    ref_self_or_result = utils.to_reference(res_self_or_result, True)

    res_grad_input = torch.empty_like(res_self_or_result)
    ref_grad_input = torch.empty_like(ref_self_or_result)

    with torch.no_grad():
        torch.ops.aten.elu_backward.grad_input(
            ref_grad_out,
            alpha,
            scale,
            input_scale,
            is_result,
            ref_self_or_result,
            grad_input=ref_grad_input,
        )
        res_out = flag_gems.elu_backward_out(
            res_grad_out,
            alpha,
            scale,
            input_scale,
            is_result,
            res_self_or_result,
            grad_input=res_grad_input,
        )

    assert res_out is res_grad_input
    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)


@pytest.mark.elu_backward_out
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_elu_backward_out_noncontiguous(dtype):
    alpha = 1.0
    scale = 1.0
    input_scale = 1.0

    res_inp = torch.randn((4, 3), dtype=dtype, device=flag_gems.device).T
    res_grad = torch.randn((4, 3), dtype=dtype, device=flag_gems.device).T

    ref_inp = utils.to_reference(res_inp, True)
    ref_grad = utils.to_reference(res_grad, True)
    ref_grad_input = torch.empty_like(ref_inp)
    with torch.no_grad():
        torch.ops.aten.elu_backward.grad_input(
            ref_grad,
            alpha,
            scale,
            input_scale,
            False,
            ref_inp,
            grad_input=ref_grad_input,
        )

    res_grad_input = torch.empty_like(res_inp)
    res_out = flag_gems.elu_backward_out(
        res_grad, alpha, scale, input_scale, False, res_inp, grad_input=res_grad_input
    )

    assert res_out is res_grad_input
    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)
