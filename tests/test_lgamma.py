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


@pytest.mark.lgamma
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lgamma(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.1
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.lgamma(ref_inp)
    res_out = flag_gems.lgamma(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.lgamma_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lgamma_(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.1
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.lgamma(ref_inp)
    res_out = inp.clone()
    flag_gems.lgamma_(res_out)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.lgamma_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lgamma_backward(shape, dtype):
    inp = (
        torch.rand(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
        + 0.1
    )
    ref_inp = utils.to_reference(inp, True)

    res_out = flag_gems.lgamma(inp)
    ref_out = torch.lgamma(ref_inp)

    out_grad = torch.randn_like(res_out)
    ref_out_grad = utils.to_reference(out_grad, True)
    (res_dx,) = torch.autograd.grad(res_out, inp, out_grad)
    (ref_dx,) = torch.autograd.grad(ref_out, ref_inp, ref_out_grad)
    utils.gems_assert_close(res_dx, ref_dx, dtype)

    res_direct = flag_gems.lgamma_backward(out_grad, inp)
    utils.gems_assert_close(res_direct, ref_dx, dtype)
