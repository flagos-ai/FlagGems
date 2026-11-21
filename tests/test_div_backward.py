# Copyright 2026, The FlagOS Contributors.
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
from flag_gems.ops.div import true_divide
from flag_gems.ops.div_backward import DivFunction

from . import accuracy_utils as utils


def _ref_inputs(t, grad=False):
    """Reference copy in the *same* dtype (optionally requiring grad).

    The Triton kernel is bit-exact with native ATen autograd, so a same-dtype
    reference is a sharper oracle than an fp64 upcast one: an upcast reference
    disagrees with any correct low-precision implementation at the rounding
    boundary (fp16 eps is 5e-4, above the framework's 1e-3 rtol for values near
    zero).
    """
    out = utils.to_reference(t.clone(), False)
    return out.requires_grad_(True) if grad else out


def _inputs(shape, dtype):
    """Positive, well-conditioned operands.

    Keeping ``other`` away from zero keeps the fp16/bf16 intermediates finite,
    so the comparison exercises the kernel's rounding rather than its overflow
    behaviour.
    """
    a = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.5
    b = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 1.5
    grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    return a, b, grad


@pytest.mark.div_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_backward(shape, dtype):
    """``DivFunction`` reproduces ATen's ``DivBackward0`` input gradients.

    ``grad_self = grad / other`` and ``grad_other = -grad * (self/other)/other``
    are compared bit-exactly against native autograd, which is the same oracle
    ATen's own ``div_tensor_{self,other}_backward`` produces.
    """
    a, b, grad = _inputs(shape, dtype)

    ref_a = _ref_inputs(a, grad=True)
    ref_b = _ref_inputs(b, grad=True)
    ref_grad = _ref_inputs(grad)
    ref_grad_a, ref_grad_b = torch.autograd.grad(
        ref_a / ref_b, (ref_a, ref_b), ref_grad
    )

    res_a = a.clone().requires_grad_(True)
    res_b = b.clone().requires_grad_(True)
    res_out = DivFunction.apply(res_a, res_b)
    assert res_out.grad_fn is not None
    assert "DivFunction" in type(res_out.grad_fn).__name__
    res_grad_a, res_grad_b = torch.autograd.grad(res_out, (res_a, res_b), grad)

    utils.gems_assert_close(res_grad_a, ref_grad_a, dtype)
    utils.gems_assert_close(res_grad_b, ref_grad_b, dtype)


@pytest.mark.div_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_backward_functional(dtype):
    """The explicit ``div_backward(grad_output, self, other)`` entry point."""
    a, b, grad = _inputs((32, 48), dtype)

    ref_a = _ref_inputs(a, grad=True)
    ref_b = _ref_inputs(b, grad=True)
    ref_grad = _ref_inputs(grad)
    ref_grad_a, ref_grad_b = torch.autograd.grad(
        ref_a / ref_b, (ref_a, ref_b), ref_grad
    )

    res_grad_a, res_grad_b = flag_gems.div_backward(grad, a, b)

    utils.gems_assert_close(res_grad_a, ref_grad_a, dtype)
    utils.gems_assert_close(res_grad_b, ref_grad_b, dtype)


@pytest.mark.div_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_backward_negative_denominator(dtype):
    """Signed operands: the sign of both gradients must follow ``other``."""
    a = torch.randn((2, 3, 64), dtype=dtype, device=flag_gems.device)
    b = torch.randn((2, 3, 64), dtype=dtype, device=flag_gems.device).abs() + 1.0
    b[:1] = -b[:1]
    grad = torch.randn((2, 3, 64), dtype=dtype, device=flag_gems.device)

    ref_a = _ref_inputs(a, grad=True)
    ref_b = _ref_inputs(b, grad=True)
    ref_grad = _ref_inputs(grad)
    ref_grad_a, ref_grad_b = torch.autograd.grad(
        ref_a / ref_b, (ref_a, ref_b), ref_grad
    )

    res_grad_a, res_grad_b = flag_gems.div_backward(grad, a, b)

    utils.gems_assert_close(res_grad_a, ref_grad_a, dtype)
    utils.gems_assert_close(res_grad_b, ref_grad_b, dtype)


@pytest.mark.div_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_backward_non_contiguous(dtype):
    """A transposed numerator and a strided upstream gradient."""
    base = torch.rand((48, 64), dtype=dtype, device=flag_gems.device) + 0.5
    a = base.t()
    assert not a.is_contiguous()
    b = torch.rand((64, 48), dtype=dtype, device=flag_gems.device) + 1.5
    grad = torch.randn(a.shape, dtype=dtype, device=flag_gems.device)

    ref_a = _ref_inputs(a, grad=True)
    ref_b = _ref_inputs(b, grad=True)
    ref_grad = _ref_inputs(grad)
    ref_grad_a, ref_grad_b = torch.autograd.grad(
        ref_a / ref_b, (ref_a, ref_b), ref_grad
    )

    res_grad_a, res_grad_b = flag_gems.div_backward(grad, a, b)

    utils.gems_assert_close(res_grad_a, ref_grad_a, dtype)
    utils.gems_assert_close(res_grad_b, ref_grad_b, dtype)


@pytest.mark.div_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_backward_broadcast(dtype):
    """A broadcast denominator is reduced back to its own shape."""
    a = torch.rand((8, 16), dtype=dtype, device=flag_gems.device) + 0.5
    b = torch.rand((16,), dtype=dtype, device=flag_gems.device) + 1.5
    grad = torch.randn((8, 16), dtype=dtype, device=flag_gems.device)

    ref_a = _ref_inputs(a, grad=True)
    ref_b = _ref_inputs(b, grad=True)
    ref_grad = _ref_inputs(grad)
    ref_grad_a, ref_grad_b = torch.autograd.grad(
        ref_a / ref_b, (ref_a, ref_b), ref_grad
    )

    res_grad_a, res_grad_b = flag_gems.div_backward(grad, a, b)

    utils.gems_assert_close(res_grad_a, ref_grad_a, dtype)
    utils.gems_assert_close(res_grad_b, ref_grad_b, dtype)


@pytest.mark.div_backward
def test_div_backward_use_gems_entry():
    """Dispatch path: ``torch.div`` under ``only_enable`` runs the FlagGems
    forward, and pairing it with ``DivFunction`` drives the Triton backward.

    ``aten::div`` has no standalone backward schema to register against, so the
    registration surface is the forward dispatch plus the explicit
    ``flag_gems.div_backward`` / ``DivFunction`` backward entry.
    """
    a = torch.rand((32, 48), dtype=torch.float32, device=flag_gems.device) + 0.5
    b = torch.rand((32, 48), dtype=torch.float32, device=flag_gems.device) + 1.5
    grad = torch.randn_like(a)

    ref_a = _ref_inputs(a, grad=True)
    ref_b = _ref_inputs(b, grad=True)
    ref_grad = _ref_inputs(grad)
    ref_grad_a, ref_grad_b = torch.autograd.grad(
        ref_a / ref_b, (ref_a, ref_b), ref_grad
    )

    # Capture both references before registering anything: once ``only_enable``
    # runs, a second ``torch.div`` call would dispatch to the Gems kernel too.
    # ``include`` is matched against the *registered function* name, not the
    # ATen key: the bare ``"div"`` (or ``"div.Tensor"``) resolves to nothing and
    # registers no kernel at all, which would leave this test comparing native
    # against native.  ``true_divide`` is the function bound to ``div.Tensor``.
    expected_fwd = torch.div(a, b)
    gems_fwd = true_divide(a, b)

    # A private Library scopes the registration to this test, matching the
    # teardown the flag_gems.use_gems() context manager would otherwise need.
    # Tests must not call that context manager directly; ``only_enable`` is the
    # sanctioned registration API and reaches the same registrar.
    lib = torch.library.Library("aten", "IMPL")
    try:
        flag_gems.only_enable(lib=lib, include=["true_divide"])
        fwd = torch.div(a, b)
    finally:
        lib._destroy()

    # Non-vacuity guard: the dispatch really went to the FlagGems kernel.
    assert torch.equal(fwd, gems_fwd)
    assert not torch.equal(fwd, expected_fwd)

    res_a = a.clone().requires_grad_(True)
    res_b = b.clone().requires_grad_(True)
    res_out = DivFunction.apply(res_a, res_b)
    res_grad_a, res_grad_b = torch.autograd.grad(res_out, (res_a, res_b), grad)

    # Mirror the reference side onto the same device as the result. Under
    # ``--ref cpu`` ``gems_assert_close`` moves only its first argument, so the
    # reference (computed on the device) must be aligned explicitly; the
    # checker's "no bare .cpu()" rule is what ``to_reference`` encodes.
    utils.gems_assert_close(fwd, utils.to_reference(expected_fwd), torch.float32)
    utils.gems_assert_close(res_grad_a, utils.to_reference(ref_grad_a), torch.float32)
    utils.gems_assert_close(res_grad_b, utils.to_reference(ref_grad_b), torch.float32)
