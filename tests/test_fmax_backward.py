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


@pytest.mark.fmax_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    "shapes",
    [
        ((4, 7), (4, 7)),
        ((2, 3, 7), (1, 3, 1)),
        ((3, 1), (1, 4)),
        ((), (2, 3)),
        ((0, 3), (1, 3)),
    ],
)
def test_fmax_backward(shapes, dtype):
    a_shape, b_shape = shapes
    a = torch.randn(a_shape, dtype=dtype, device=flag_gems.device)
    b = torch.randn(b_shape, dtype=dtype, device=flag_gems.device)
    # Include ties and both possible one-sided NaNs.
    a = a.clone()
    b = b.clone()
    if a.numel() and b.numel():
        a.view(-1)[0] = 1
        b.view(-1)[0] = 1
        a.view(-1)[-1] = float("nan")
        b.view(-1)[-1] = float("nan")
        if a.numel() > 2 and b.numel() > 2:
            a.view(-1)[1] = float("nan")
            b.view(-1)[1] = 2
            a.view(-1)[2] = 2
            b.view(-1)[2] = float("nan")
    a_ref = utils.to_reference(a).detach().requires_grad_()
    b_ref = utils.to_reference(b).detach().requires_grad_()
    a = a.detach().requires_grad_()
    b = b.detach().requires_grad_()
    grad = torch.randn(
        torch.broadcast_shapes(a_shape, b_shape), dtype=dtype, device=flag_gems.device
    )
    torch.fmax(a_ref, b_ref).backward(utils.to_reference(grad))
    flag_gems.fmax(a, b).backward(grad)
    utils.gems_assert_equal(a.grad, a_ref.grad)
    utils.gems_assert_equal(b.grad, b_ref.grad)


@pytest.mark.fmax_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("requires_grad", [(True, False), (False, True)])
def test_fmax_backward_one_input(dtype, requires_grad):
    a = torch.tensor(
        [1, 2, float("nan"), 4, float("nan")], dtype=dtype, device=flag_gems.device
    )
    b = torch.tensor(
        [1, float("nan"), 3, float("nan"), float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    a_ref = utils.to_reference(a).detach().requires_grad_(requires_grad[0])
    b_ref = utils.to_reference(b).detach().requires_grad_(requires_grad[1])
    a = a.detach().requires_grad_(requires_grad[0])
    b = b.detach().requires_grad_(requires_grad[1])
    grad = torch.tensor([2, 3, 4, 5, 6], dtype=dtype, device=flag_gems.device)
    torch.fmax(a_ref, b_ref).backward(utils.to_reference(grad))
    flag_gems.fmax(a, b).backward(grad)
    if requires_grad[0]:
        utils.gems_assert_equal(a.grad, a_ref.grad)
    if requires_grad[1]:
        utils.gems_assert_equal(b.grad, b_ref.grad)


@pytest.mark.fmax_backward
def test_fmax_backward_dispatch():
    a = torch.tensor(
        [1.0, float("nan"), 2.0], device=flag_gems.device, requires_grad=True
    )
    b = torch.tensor(
        [1.0, 3.0, float("nan")], device=flag_gems.device, requires_grad=True
    )
    grad = torch.tensor([2.0, 3.0, 4.0], device=flag_gems.device)
    a_ref = utils.to_reference(a).detach().requires_grad_()
    b_ref = utils.to_reference(b).detach().requires_grad_()
    torch.fmax(a_ref, b_ref).backward(utils.to_reference(grad))
    with flag_gems.use_gems():
        torch.fmax(a, b).backward(grad)
    utils.gems_assert_equal(a.grad, a_ref.grad)
    utils.gems_assert_equal(b.grad, b_ref.grad)
