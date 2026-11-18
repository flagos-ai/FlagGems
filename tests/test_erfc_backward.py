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

import math

import pytest
import torch

import flag_gems
from flag_gems.ops.erfc_backward import ErfcFunction

from . import accuracy_utils as utils

_M_2_SQRTPI = 2.0 / math.sqrt(math.pi)


def _erfc_backward_reference(grad_output, x):
    """Gradient of ``erfc``, reproducing ATen's per-op rounding order.

    ATen has no dedicated backward kernel for ``erfc``: the autograd formula is
    ``grad * -M_2_SQRTPI * exp(-(self.pow(2)))``. Every step of that formula is a
    separate elementwise kernel whose result is written back in the storage
    dtype (fp16/bf16 included, no float32 accumulator is kept across steps),
    so the reference below rounds at the same places and is bit-exact against
    ``torch.autograd.grad(torch.erfc(x), x)`` on the same device.

    Under ``--ref cpu`` the reference runs on CPU instead, where ``torch.exp``
    is not bit-identical to the CUDA ``expf``; callers use ``gems_assert_close``
    in that mode and the bit-exact ``gems_assert_equal`` on device.
    """
    if x.numel() == 0:
        return torch.empty(x.shape, dtype=x.dtype, device=x.device)
    # ATen special-cases the integer exponent 2 to a plain multiply, so ``x * x``
    # reproduces ``self.pow(2)`` exactly without going through ``aten::pow``.
    squared = x * x
    exponential = torch.exp(-squared)
    return ((-_M_2_SQRTPI) * exponential) * grad_output


def _assert_backward(res_grad, ref_grad):
    """Bit-exact on device; tolerance-based when the reference is on CPU."""
    if utils.TO_CPU:
        utils.gems_assert_close(res_grad, ref_grad, res_grad.dtype, atol=1e-4)
    else:
        utils.gems_assert_equal(res_grad, ref_grad)


@pytest.mark.erfc_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_erfc_backward_bit_exact(shape, dtype):
    """Bit-exactness against the eager on-device autograd chain.

    The kernel rounds each intermediate (square, exp, scalar mul, grad mul) to
    the storage dtype, the same places ATen's four-kernel chain does, so the
    comparison is exact rather than tolerance based.
    """
    torch.manual_seed(0)
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_grad = _erfc_backward_reference(
        utils.to_reference(grad_output), utils.to_reference(x)
    )
    res_grad = flag_gems.erfc_backward(grad_output, x)

    _assert_backward(res_grad, ref_grad)


@pytest.mark.erfc_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_erfc_backward_autograd(dtype):
    # ``aten::erfc`` keeps ATen's own autograd formula node, so the gradient has
    # to be driven through the FlagGems autograd bridge explicitly; otherwise
    # this would exercise the native chain instead of the Triton kernel.
    torch.manual_seed(0)
    inp = torch.randn((1024,), dtype=dtype, device=flag_gems.device, requires_grad=True)
    out = ErfcFunction.apply(inp)
    assert out.requires_grad and out.grad_fn is not None

    grad_output = torch.randn_like(out)
    (res_grad,) = torch.autograd.grad(out, (inp,), (grad_output,))

    ref_grad = _erfc_backward_reference(
        utils.to_reference(grad_output), utils.to_reference(inp)
    )
    _assert_backward(res_grad, ref_grad)


@pytest.mark.erfc_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_erfc_backward_non_contiguous(dtype):
    torch.manual_seed(0)
    base = torch.randn((64, 64), dtype=dtype, device=flag_gems.device)
    x = base.t()
    assert not x.is_contiguous()
    grad_output = torch.randn_like(x)

    ref_grad = _erfc_backward_reference(
        utils.to_reference(grad_output), utils.to_reference(x)
    )
    res_grad = flag_gems.erfc_backward(grad_output, x)

    _assert_backward(res_grad, ref_grad)


@pytest.mark.erfc_backward
def test_erfc_backward_dispatch():
    """The registered forward and the bridge share one gradient.

    ``flag_gems.only_enable`` installs the same registration the public enable
    path uses, so the forward that runs here is the FlagGems ``erfc`` kernel.
    Comparing its gradient against the fp64 autograd reference taken before the
    registration proves the bridge is reachable from a dispatched forward.
    """
    x = torch.randn((32, 48), dtype=torch.float32, device=flag_gems.device)
    grad_output = torch.randn_like(x)

    ref_inp = utils.to_reference(x, True).requires_grad_(True)
    ref_grad = utils.to_reference(grad_output, True)
    (ref_grad_in,) = torch.autograd.grad(torch.erfc(ref_inp), (ref_inp,), (ref_grad,))

    lib = torch.library.Library("aten", "IMPL")
    try:
        # Scoped registration: unlike the global ``flag_gems.use_gems()`` context
        # manager, a private Library tears the override down in ``finally`` and
        # leaves the native ``torch.erfc`` measured above untouched elsewhere.
        flag_gems.only_enable(lib=lib, include=["erfc"])
        inp = x.clone().requires_grad_(True)
        out = ErfcFunction.apply(inp)
        (res_grad,) = torch.autograd.grad(out, (inp,), (grad_output,))
    finally:
        lib._destroy()

    utils.gems_assert_close(res_grad, ref_grad_in, torch.float32)
