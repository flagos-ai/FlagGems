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
from flag_gems.ops.exp_backward import ExpFunction

from . import accuracy_utils as utils


def _exp_backward_reference(grad_output, x):
    """Gradient of ``exp``, reproducing ATen's per-op rounding order.

    ATen has no dedicated backward kernel for ``exp``: the autograd formula is
    ``grad * result.conj()``, where ``result`` is the forward output already
    rounded to the storage dtype. The reference below rounds at the same places
    (exp is written back in the storage dtype before the multiply, and the
    product is rounded again), so it is bit-exact against
    ``torch.autograd.grad(torch.exp(x), x)`` on the same device.

    Under ``--ref cpu`` the reference runs on CPU instead, where ``torch.exp``
    is not bit-identical to the CUDA ``expf``; callers use
    ``gems_assert_close`` in that mode and ``gems_assert_equal`` on device.
    """
    if x.numel() == 0:
        return torch.empty(x.shape, dtype=x.dtype, device=x.device)
    y = torch.exp(x)
    return grad_output * y


def _assert_backward(res_grad, ref_grad):
    """Bit-exact on device; tolerance-based when the reference is on CPU."""
    if utils.TO_CPU:
        utils.gems_assert_close(res_grad, ref_grad, res_grad.dtype, atol=1e-4)
    else:
        utils.gems_assert_equal(res_grad, ref_grad)


@pytest.mark.exp_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_exp_backward_bit_exact(shape, dtype):
    """Bit-exactness against the eager on-device autograd chain.

    The kernel rounds each intermediate (exp recompute, grad mul) to the
    storage dtype, the same places ATen's two-kernel chain does, so the
    comparison is exact rather than tolerance based.
    """
    torch.manual_seed(0)
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # Sample a wide input range so both fast-decaying (x << 0) and growing
    # (x >> 0) branches of exp are exercised.
    x = (torch.randn(shape, dtype=dtype, device=flag_gems.device) * 2).to(dtype)

    ref_grad = _exp_backward_reference(
        utils.to_reference(grad_output), utils.to_reference(x)
    )
    res_grad = flag_gems.exp_backward(grad_output, x)

    _assert_backward(res_grad, ref_grad)


@pytest.mark.exp_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_exp_backward_autograd(dtype):
    # ``aten::exp`` keeps ATen's own autograd formula node, so the gradient
    # has to be driven through the FlagGems autograd bridge explicitly;
    # otherwise this would exercise the native chain instead of the Triton
    # kernel.
    torch.manual_seed(0)
    inp = (
        (torch.randn((1024,), dtype=dtype, device=flag_gems.device) * 2)
        .to(dtype)
        .requires_grad_()
    )
    out = ExpFunction.apply(inp)
    assert out.requires_grad and out.grad_fn is not None

    grad_output = torch.randn_like(out)
    (res_grad,) = torch.autograd.grad(out, (inp,), (grad_output,))

    ref_grad = _exp_backward_reference(
        utils.to_reference(grad_output), utils.to_reference(inp)
    )
    _assert_backward(res_grad, ref_grad)


@pytest.mark.exp_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_exp_backward_non_contiguous(dtype):
    torch.manual_seed(0)
    base = (torch.randn((64, 64), dtype=dtype, device=flag_gems.device) * 2).to(dtype)
    x = base.t()
    assert not x.is_contiguous()
    grad_output = torch.randn_like(x)

    ref_grad = _exp_backward_reference(
        utils.to_reference(grad_output), utils.to_reference(x)
    )
    res_grad = flag_gems.exp_backward(grad_output, x)

    _assert_backward(res_grad, ref_grad)


@pytest.mark.exp_backward
def test_exp_backward_boundary():
    """Extreme inputs match native autograd: 0 for very negative, inf on overflow."""
    x = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, 88.0, -88.0, 100.0, -100.0, float("nan")],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    grad_output = torch.ones_like(x)

    ref_inp = utils.to_reference(x).requires_grad_(True)
    (ref_grad,) = torch.autograd.grad(
        torch.exp(ref_inp), (ref_inp,), (utils.to_reference(grad_output),)
    )
    res_grad = flag_gems.exp_backward(grad_output, x)

    # ``equal_nan=True`` treats matching NaNs as equal; inf==inf is handled by
    # the underlying assert_close. No clamping is done, matching ATen.
    utils.gems_assert_close(res_grad, ref_grad, res_grad.dtype, equal_nan=True)


@pytest.mark.exp_backward
def test_exp_backward_dispatch():
    """The registered forward and the bridge share one gradient.

    ``flag_gems.only_enable`` installs the same registration the public enable
    path uses, so the forward that runs here is the FlagGems ``exp`` kernel.
    Comparing its gradient against the fp64 autograd reference taken before
    the registration proves the bridge is reachable from a dispatched forward.
    """
    x = (torch.randn((32, 48), dtype=torch.float32, device=flag_gems.device) * 2).to(
        torch.float32
    )
    grad_output = torch.randn_like(x)

    ref_inp = utils.to_reference(x, True).requires_grad_(True)
    ref_grad = utils.to_reference(grad_output, True)
    (ref_grad_in,) = torch.autograd.grad(torch.exp(ref_inp), (ref_inp,), (ref_grad,))

    lib = torch.library.Library("aten", "IMPL")
    try:
        # Scoped registration: unlike the global ``flag_gems.use_gems()``
        # context manager, a private Library tears the override down in
        # ``finally`` and leaves the native ``torch.exp`` measured above
        # untouched elsewhere.
        flag_gems.only_enable(lib=lib, include=["exp"])
        inp = x.clone().requires_grad_(True)
        out = ExpFunction.apply(inp)
        (res_grad,) = torch.autograd.grad(out, (inp,), (grad_output,))
    finally:
        lib._destroy()

    utils.gems_assert_close(res_grad, ref_grad_in, torch.float32)
