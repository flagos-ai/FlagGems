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


@pytest.mark.expm1
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_expm1(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.expm1(ref_inp)
    res_out = flag_gems.expm1(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.expm1_backward
@pytest.mark.parametrize("shape", [(), (0,), (257,), (19, 37)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_expm1_backward(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if x.ndim == 2:
        x = x.t()
    x.requires_grad_()
    ref_x = utils.to_reference(x.detach()).requires_grad_()
    grad = torch.randn_like(x, memory_format=torch.contiguous_format)

    out = flag_gems.expm1(x)
    ref_out = torch.expm1(ref_x)
    (dx,) = torch.autograd.grad(out, x, grad)
    (ref_dx,) = torch.autograd.grad(ref_out, ref_x, utils.to_reference(grad))

    utils.gems_assert_close(out, ref_out, dtype)
    utils.gems_assert_close(dx, ref_dx, dtype)


@pytest.mark.expm1_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_expm1_backward_expanded_grad(dtype):
    x = torch.randn((7, 13), dtype=dtype, device=flag_gems.device).requires_grad_()
    ref_x = utils.to_reference(x.detach()).requires_grad_()
    grad = torch.randn((7, 1), dtype=dtype, device=x.device).expand_as(x)

    (dx,) = torch.autograd.grad(flag_gems.expm1(x), x, grad)
    (ref_dx,) = torch.autograd.grad(torch.expm1(ref_x), ref_x, utils.to_reference(grad))
    utils.gems_assert_close(dx, ref_dx, dtype)


@pytest.mark.expm1_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_expm1_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = torch.expm1_(ref_inp)
    res_out = flag_gems.expm1_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.expm1_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_expm1_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty_like(ref_inp)
    torch.expm1(ref_inp, out=ref_out)
    res_out = torch.empty_like(inp)
    flag_gems.expm1_out(inp, out=res_out)

    utils.gems_assert_close(res_out, ref_out, dtype)
