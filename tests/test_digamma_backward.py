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
from flag_gems.ops.digamma_backward import DigammaFunction

from . import accuracy_utils as utils


@pytest.mark.digamma_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_digamma_backward(shape, dtype):
    """The input gradient of ``digamma`` is grad_out * polygamma(1, x).

    polygamma(1, x) is computed in float32 opmath for fp16/bf16 and materialised
    in the input dtype before the multiply, matching ATen's
    ``DigammaBackward0``.
    """
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 1.0
    grad_out = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp.clone(), True).requires_grad_(True)
    ref_grad = utils.to_reference(grad_out, True)
    ref_out = torch.digamma(ref_inp)
    (ref_grad_in,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)

    res_inp = inp.clone().requires_grad_(True)
    res_out = DigammaFunction.apply(res_inp)
    assert res_out.grad_fn is not None
    assert "DigammaFunction" in type(res_out.grad_fn).__name__
    (res_grad,) = torch.autograd.grad(res_out, res_inp, grad_out)

    utils.gems_assert_close(res_grad, ref_grad_in, dtype)


@pytest.mark.digamma_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_digamma_backward_functional(dtype):
    """The explicit ``digamma_backward(grad_output, input)`` entry point.

    Inputs are kept in [1, 2): polygamma(1, x) has poles at the non-positive
    integers, and fp16/bf16 overflow to inf well before the fp64 reference does,
    so a diverging domain would compare two different (both "correct") results.
    """
    x = torch.rand((32, 48), dtype=dtype, device=flag_gems.device) + 1.0
    grad = torch.randn((32, 48), dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(x, True).requires_grad_(True)
    ref_grad = utils.to_reference(grad, True)
    (ref_grad_in,) = torch.autograd.grad(torch.digamma(ref_inp), ref_inp, ref_grad)

    res_grad = flag_gems.digamma_backward(grad, x)

    utils.gems_assert_close(res_grad, ref_grad_in, dtype)


@pytest.mark.digamma_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_digamma_backward_non_contiguous(dtype):
    """A transposed input and a strided upstream gradient."""
    base = torch.rand((64, 48), dtype=dtype, device=flag_gems.device) + 1.0
    x = base.t()
    assert not x.is_contiguous()
    grad = torch.randn(x.shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(x, True).requires_grad_(True)
    ref_grad = utils.to_reference(grad, True)
    (ref_grad_in,) = torch.autograd.grad(torch.digamma(ref_inp), ref_inp, ref_grad)

    res_inp = x.clone().requires_grad_(True)
    res_out = DigammaFunction.apply(res_inp)
    (res_grad,) = torch.autograd.grad(res_out, res_inp, grad)

    utils.gems_assert_close(res_grad, ref_grad_in, dtype)


@pytest.mark.digamma_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_digamma_backward_negative_input(dtype):
    """Negative inputs exercise the reflection branch, x in (-3, -0.5).

    The reference is native ATen autograd in the *same* dtype: near the poles at
    non-positive integers the fp16/bf16 result overflows to inf long before an
    fp64 reference would, so an upcast reference is not a meaningful oracle
    here. The Triton kernel reproduces native ATen bit-for-bit in this domain.
    """
    x = -(torch.rand((2, 3, 64), dtype=dtype, device=flag_gems.device) * 2.5 + 0.5)
    grad = torch.randn((2, 3, 64), dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(x, False).requires_grad_(True)
    ref_grad = utils.to_reference(grad, False)
    (ref_grad_in,) = torch.autograd.grad(torch.digamma(ref_inp), ref_inp, ref_grad)

    res_grad = flag_gems.digamma_backward(grad, x)

    utils.gems_assert_close(res_grad, ref_grad_in, dtype)


@pytest.mark.digamma_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_digamma_backward_igamma_precision(dtype):
    """fp16 boundary accuracy: polygamma(1, x) diverges as x -> 0, check in-range."""
    # x in [1, 2]: polygamma(1) is finite and smooth there.
    x = torch.rand((2, 3, 64), dtype=dtype, device=flag_gems.device) + 1.0
    grad = torch.randn((2, 3, 64), dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(x, True).requires_grad_(True)
    ref_grad = utils.to_reference(grad, True)
    (ref_grad_in,) = torch.autograd.grad(torch.digamma(ref_inp), ref_inp, ref_grad)

    res_grad = flag_gems.digamma_backward(grad, x)

    utils.gems_assert_close(res_grad, ref_grad_in, dtype)


@pytest.mark.digamma_backward
def test_digamma_backward_use_gems_entry():
    """Dispatch path: ``torch.digamma`` under ``only_enable`` runs the FlagGems
    forward, and pairing it with ``DigammaFunction`` drives the Triton backward.

    ``aten::digamma`` has no standalone backward schema to register against, so
    the registration surface here is the forward dispatch plus the explicit
    ``flag_gems.digamma_backward`` / ``DigammaFunction`` backward entry.
    """
    x = torch.rand((32, 48), dtype=torch.float32, device=flag_gems.device) + 1.0
    grad = torch.randn_like(x)

    ref_inp = utils.to_reference(x, True).requires_grad_(True)
    ref_grad = utils.to_reference(grad, True)
    (ref_grad_in,) = torch.autograd.grad(torch.digamma(ref_inp), ref_inp, ref_grad)

    # Capture the native forward before registering anything: once
    # ``only_enable`` runs, a second ``torch.digamma`` call would dispatch to
    # the Gems kernel too and compare it against itself.
    expected_fwd = torch.digamma(x)

    # A private Library scopes the registration to this test, matching the
    # teardown the ``use_gems()`` context manager used to provide. Tests must
    # not call ``use_gems()``; ``only_enable`` is the sanctioned registration
    # API and reaches the same registrar.
    lib = torch.library.Library("aten", "IMPL")
    try:
        flag_gems.only_enable(lib=lib, include=["digamma"])
        fwd = torch.digamma(x)
    finally:
        lib._destroy()

    res_inp = x.clone().requires_grad_(True)
    res_out = DigammaFunction.apply(res_inp)
    (res_grad,) = torch.autograd.grad(res_out, res_inp, grad)

    # Compare on the CPU explicitly. Under ``--ref cpu`` ``utils.to_cpu`` moves
    # only the result, so a reference tensor still on the device (here the
    # freshly computed forward) would trip the device check.
    utils.gems_assert_close(fwd.cpu(), expected_fwd.cpu(), torch.float32)
    utils.gems_assert_close(res_grad.cpu(), ref_grad_in.cpu(), torch.float32)
