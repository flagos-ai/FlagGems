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


@pytest.mark.lift
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lift(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.ops.aten.lift(ref_inp)
    res_out = flag_gems.lift(inp)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.lift_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lift_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.empty_like(ref_inp)
    torch.ops.aten.lift.out(ref_inp, out=ref_out)
    res_out = torch.empty_like(inp)
    flag_gems.lift_out(inp, out=res_out)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.lift_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lift_backward(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(inp, True)

    res_out = flag_gems.lift(inp)
    ref_out = torch.ops.aten.lift(ref_inp)

    out_grad = torch.randn_like(res_out)
    (res_dx,) = torch.autograd.grad(res_out, inp, out_grad)
    (ref_dx,) = torch.autograd.grad(
        ref_out, ref_inp, utils.to_reference(out_grad, True)
    )
    utils.gems_assert_close(res_dx, ref_dx, dtype)

    res_direct = flag_gems.lift_backward(out_grad)
    assert torch.equal(res_direct, out_grad)
