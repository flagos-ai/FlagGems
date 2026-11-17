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


@pytest.mark.deg2rad
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_deg2rad(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.deg2rad(ref_inp)
    res_out = flag_gems.deg2rad(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.deg2rad_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_deg2rad_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_inp.deg2rad_()
    flag_gems.deg2rad_(inp)

    utils.gems_assert_close(inp, ref_inp, dtype)


@pytest.mark.deg2rad_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_deg2rad_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty_like(ref_inp)
    torch.deg2rad(ref_inp, out=ref_out)

    res_out = torch.empty_like(inp)
    flag_gems.deg2rad_out(inp, out=res_out)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.deg2rad_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.BYTE_DTYPES)
def test_deg2rad_out_byte(shape, dtype):
    # torch promotes integral deg2rad inputs to float32 before scaling.
    inp = torch.randint(1, 8, shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.empty(shape, dtype=torch.float32, device=ref_inp.device)
    torch.deg2rad(ref_inp, out=ref_out)

    res_out = torch.empty(shape, dtype=torch.float32, device=flag_gems.device)
    flag_gems.deg2rad_out(inp, out=res_out)

    utils.gems_assert_close(res_out, ref_out, torch.float32)


@pytest.mark.deg2rad_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_deg2rad_backward(shape, dtype):
    """The input gradient of ``deg2rad`` is the upstream gradient times pi/180.

    The multiply runs in float32 opmath for fp16/bf16 and the result is cast
    back to the input dtype, matching ATen's ``Deg2RadBackward0``.
    """
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_out = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp.clone(), True).requires_grad_(True)
    ref_grad = utils.to_reference(grad_out, True)
    ref_out = torch.deg2rad(ref_inp)
    (ref_grad_in,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)

    res_inp = inp.clone().requires_grad_(True)
    res_out = flag_gems.deg2rad(res_inp)
    assert res_out.grad_fn is not None
    assert "Deg2RadFunction" in type(res_out.grad_fn).__name__
    (res_grad,) = torch.autograd.grad(res_out, res_inp, grad_out)

    utils.gems_assert_close(res_grad, ref_grad_in, dtype)


@pytest.mark.deg2rad_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_deg2rad_backward_functional(dtype):
    """The explicit ``deg2rad_backward(grad_output, input)`` entry point."""
    x = torch.randn((32, 48), dtype=dtype, device=flag_gems.device)
    grad = torch.randn((32, 48), dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(x, True).requires_grad_(True)
    ref_grad = utils.to_reference(grad, True)
    (ref_grad_in,) = torch.autograd.grad(torch.deg2rad(ref_inp), ref_inp, ref_grad)

    res_grad = flag_gems.deg2rad_backward(grad, x)

    utils.gems_assert_close(res_grad, ref_grad_in, dtype)


@pytest.mark.deg2rad_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_deg2rad_backward_non_contiguous(dtype):
    """A transposed input and a strided upstream gradient."""
    base = torch.randn((48, 64), dtype=dtype, device=flag_gems.device)
    x = base.t()
    assert not x.is_contiguous()
    grad = torch.randn(x.shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(base, True).t().requires_grad_(True)
    ref_grad = utils.to_reference(grad, True)
    (ref_grad_in,) = torch.autograd.grad(torch.deg2rad(ref_inp), ref_inp, ref_grad)

    res_inp = base.clone().t().requires_grad_(True)
    res_out = flag_gems.deg2rad(res_inp)
    (res_grad,) = torch.autograd.grad(res_out, res_inp, grad)

    utils.gems_assert_close(res_grad, ref_grad_in, dtype)
