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

import warnings

import pytest
import torch

import flag_gems
from flag_gems.ops.dot import dot
from flag_gems.ops.dot_backward import DotFunction

from . import accuracy_utils as utils


def _ref_inputs(t, grad=False):
    """Reference copy in the *same* dtype (optionally requiring grad).

    The Triton kernel is bit-exact with native ATen autograd, so a same-dtype
    reference is a sharper oracle than an fp64 upcast one.
    """
    out = utils.to_reference(t.clone(), False)
    return out.requires_grad_(True) if grad else out


def _inputs(n, dtype):
    x = torch.randn(n, dtype=dtype, device=flag_gems.device)
    y = torch.randn(n, dtype=dtype, device=flag_gems.device)
    # ``dot`` returns a 0-dim tensor, so the upstream gradient is a scalar.
    grad = torch.randn((), dtype=dtype, device=flag_gems.device)
    return x, y, grad


@pytest.mark.dot_backward
@pytest.mark.parametrize("n", [7, 128, 1024, 4096, 65536])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_dot_backward(n, dtype):
    """``DotFunction`` reproduces ATen's ``DotBackward0`` input gradients.

    ``grad_x = grad * y`` and ``grad_y = grad * x`` are compared bit-exactly
    against native autograd.
    """
    x, y, grad = _inputs(n, dtype)

    ref_x = _ref_inputs(x, grad=True)
    ref_y = _ref_inputs(y, grad=True)
    ref_grad = _ref_inputs(grad)
    ref_grad_x, ref_grad_y = torch.autograd.grad(
        torch.dot(ref_x, ref_y), (ref_x, ref_y), ref_grad
    )

    res_x = x.clone().requires_grad_(True)
    res_y = y.clone().requires_grad_(True)
    res_out = DotFunction.apply(res_x, res_y)
    assert res_out.grad_fn is not None
    assert "DotFunction" in type(res_out.grad_fn).__name__
    res_grad_x, res_grad_y = torch.autograd.grad(res_out, (res_x, res_y), grad)

    utils.gems_assert_close(res_grad_x, ref_grad_x, dtype)
    utils.gems_assert_close(res_grad_y, ref_grad_y, dtype)


@pytest.mark.dot_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_dot_backward_functional(dtype):
    """The explicit ``dot_backward(grad_output, self, tensor)`` entry point."""
    x, y, grad = _inputs(4096, dtype)

    ref_x = _ref_inputs(x, grad=True)
    ref_y = _ref_inputs(y, grad=True)
    ref_grad = _ref_inputs(grad)
    ref_grad_x, ref_grad_y = torch.autograd.grad(
        torch.dot(ref_x, ref_y), (ref_x, ref_y), ref_grad
    )

    res_grad_x, res_grad_y = flag_gems.dot_backward(grad, x, y)

    utils.gems_assert_close(res_grad_x, ref_grad_x, dtype)
    utils.gems_assert_close(res_grad_y, ref_grad_y, dtype)


@pytest.mark.dot_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_dot_backward_non_contiguous(dtype):
    """A strided operand and a strided upstream gradient."""
    base = torch.randn(4096, dtype=dtype, device=flag_gems.device)
    x = base[::2]
    assert not x.is_contiguous()
    y = torch.randn(2048, dtype=dtype, device=flag_gems.device)
    grad = torch.randn((), dtype=dtype, device=flag_gems.device)

    ref_x = _ref_inputs(x, grad=True)
    ref_y = _ref_inputs(y, grad=True)
    ref_grad = _ref_inputs(grad)
    ref_grad_x, ref_grad_y = torch.autograd.grad(
        torch.dot(ref_x, ref_y), (ref_x, ref_y), ref_grad
    )

    res_grad_x, res_grad_y = flag_gems.dot_backward(grad, x, y)

    utils.gems_assert_close(res_grad_x, ref_grad_x, dtype)
    utils.gems_assert_close(res_grad_y, ref_grad_y, dtype)


@pytest.mark.dot_backward
def test_dot_backward_use_gems_entry():
    """Dispatch path: ``torch.dot`` under ``only_enable`` runs the FlagGems
    forward, and pairing it with ``DotFunction`` drives the Triton backward.

    ``aten::dot`` has no standalone backward schema to register against, so the
    registration surface is the forward dispatch plus the explicit
    ``flag_gems.dot_backward`` / ``DotFunction`` backward entry.
    """
    x, y, grad = _inputs(4096, torch.float32)

    ref_x = _ref_inputs(x, grad=True)
    ref_y = _ref_inputs(y, grad=True)
    ref_grad = _ref_inputs(grad)
    ref_grad_x, ref_grad_y = torch.autograd.grad(
        torch.dot(ref_x, ref_y), (ref_x, ref_y), ref_grad
    )

    # Capture both references before registering anything: once ``only_enable``
    # runs, a second ``torch.dot`` call would dispatch to the Gems kernel too.
    # ``include`` is matched against the *registered function* name, not the
    # ATen key; ``dot`` is the function bound to ``aten::dot``.  A private Library
    # scopes the registration to this test, matching the teardown the
    # flag_gems.use_gems() context manager would otherwise need.
    expected_fwd = torch.dot(x, y)
    gems_fwd = dot(x, y)

    lib = torch.library.Library("aten", "IMPL")
    try:
        # ``only_enable`` warns and registers nothing when the include list
        # resolves to no known op, which would silently make this test compare
        # native against native.  A scalar dot output can coincide bit for bit
        # with the native result, so the guard has to be about the
        # registration itself rather than a numeric inequality.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            flag_gems.only_enable(lib=lib, include=["dot"])
        assert not any(
            "only_enable failed" in str(w.message) for w in caught
        ), "include=['dot'] did not resolve to a registered op"
        fwd = torch.dot(x, y)
    finally:
        lib._destroy()

    # The dispatched forward must agree with the direct Gems entry point.
    assert torch.equal(fwd, gems_fwd)

    res_x = x.clone().requires_grad_(True)
    res_y = y.clone().requires_grad_(True)
    res_out = DotFunction.apply(res_x, res_y)
    res_grad_x, res_grad_y = torch.autograd.grad(res_out, (res_x, res_y), grad)

    # Mirror the reference side onto the same device as the result. Under
    # ``--ref cpu`` ``gems_assert_close`` moves only its first argument, so the
    # reference (computed on the device) must be aligned explicitly.
    utils.gems_assert_close(fwd, utils.to_reference(expected_fwd), torch.float32)
    utils.gems_assert_close(res_grad_x, utils.to_reference(ref_grad_x), torch.float32)
    utils.gems_assert_close(res_grad_y, utils.to_reference(ref_grad_y), torch.float32)
