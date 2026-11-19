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


@pytest.mark.copysign_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_copysign_backward(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    other = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    out_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_other = utils.to_reference(other, True)
    ref_grad = utils.to_reference(out_grad, True)

    ref_out = torch.copysign(ref_inp, ref_other)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)

    res_in_grad = flag_gems.copysign_backward(out_grad, inp, other)

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.copysign_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_copysign_backward_signed_zeros(shape, dtype):
    # `other` must contribute its sign *bit*, so -0.0 counts as negative.
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    other = torch.tensor(-0.0, dtype=dtype, device=flag_gems.device)
    out_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_grad = utils.to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(
        torch.copysign(ref_inp, -0.0), ref_inp, ref_grad
    )
    res_in_grad = flag_gems.copysign_backward(out_grad, inp, other)

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.copysign_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("other_scalar", [-2.0, 0.5])
def test_copysign_backward_scalar(shape, dtype, other_scalar):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    out_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_grad = utils.to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(
        torch.copysign(ref_inp, other_scalar), ref_inp, ref_grad
    )
    res_in_grad = flag_gems.copysign_backward(out_grad, inp, other_scalar)

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)
