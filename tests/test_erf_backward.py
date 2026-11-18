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
from flag_gems.ops.erf_backward import ErfFunction

from . import accuracy_utils as utils

_M_2_SQRTPI = 2.0 / math.sqrt(math.pi)


def _erf_backward_reference(grad_output, x):
    """Gradient of ``erf``, reproducing ATen's per-op rounding order.

    ATen has no dedicated backward kernel for ``erf``: the autograd formula is
    ``M_2_SQRTPI * exp(-(self.pow(2))) * grad``. Every step of that formula is a
    separate elementwise kernel whose result is written back in the storage
    dtype (fp16/bf16 included), so the reference below rounds at the same
    places and is bit-exact against ``torch.autograd.grad(torch.erf(x), x)``.
    """
    if x.numel() == 0:
        return torch.empty(x.shape, dtype=x.dtype, device=x.device)
    # ATen special-cases the integer exponent 2 to a plain multiply, so ``x * x``
    # reproduces ``self.pow(2)`` exactly without going through ``aten::pow``.
    squared = x * x
    exponential = torch.exp(-squared)
    return (_M_2_SQRTPI * exponential) * grad_output


@pytest.mark.erf_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_erf_backward_bit_exact(shape, dtype):
    """Bit-exactness against the eager on-device autograd chain.

    The kernel rounds each intermediate (square, exp, scalar mul, grad mul) to
    the storage dtype, the same places ATen's four-kernel chain does, so the
    comparison is exact rather than tolerance based.
    """
    torch.manual_seed(0)
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_grad = _erf_backward_reference(grad_output, x)
    res_grad = flag_gems.erf_backward(grad_output, x)

    # The reference is built on ``flag_gems.device`` from the on-device inputs,
    # so move both sides to the CPU explicitly: in ``--ref=cpu`` mode
    # ``utils.to_cpu`` only relocates ``res`` and would otherwise leave ``ref``
    # on the accelerator.
    utils.gems_assert_equal(res_grad.cpu(), ref_grad.cpu())


@pytest.mark.erf_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_erf_backward_autograd(dtype):
    # ``aten::erf`` keeps ATen's own autograd formula node, so the gradient has
    # to be driven through the FlagGems autograd bridge explicitly; otherwise
    # this would exercise the native chain instead of the Triton kernel.
    torch.manual_seed(0)
    inp = torch.randn((1024,), dtype=dtype, device=flag_gems.device, requires_grad=True)
    out = ErfFunction.apply(inp)
    assert out.requires_grad and out.grad_fn is not None

    grad_output = torch.randn_like(out)
    (res_grad,) = torch.autograd.grad(out, (inp,), (grad_output,))

    ref_grad = _erf_backward_reference(grad_output, inp)
    utils.gems_assert_equal(res_grad.cpu(), ref_grad.cpu())


@pytest.mark.erf_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_erf_backward_non_contiguous(dtype):
    torch.manual_seed(0)
    base = torch.randn((64, 64), dtype=dtype, device=flag_gems.device)
    x = base.t()
    assert not x.is_contiguous()
    grad_output = torch.randn_like(x)

    ref_grad = _erf_backward_reference(grad_output, x)
    res_grad = flag_gems.erf_backward(grad_output, x)

    utils.gems_assert_equal(res_grad.cpu(), ref_grad.cpu())


@pytest.mark.erf_backward
def test_erf_backward_dispatch():
    """The registered forward and the bridge share one gradient.

    ``flag_gems.only_enable`` installs the same registration the public enable
    path uses, so the forward that runs here is the FlagGems ``erf`` kernel.
    Comparing its gradient against the fp64 autograd reference taken before the
    registration proves the bridge is reachable from a dispatched forward.
    """
    x = torch.randn((32, 48), dtype=torch.float32, device=flag_gems.device)
    grad_output = torch.randn_like(x)

    ref_inp = utils.to_reference(x, True).requires_grad_(True)
    ref_grad = utils.to_reference(grad_output, True)
    (ref_grad_in,) = torch.autograd.grad(torch.erf(ref_inp), (ref_inp,), (ref_grad,))

    lib = torch.library.Library("aten", "IMPL")
    try:
        # Scoped registration: unlike the global ``flag_gems.use_gems()`` context
        # manager, a private Library tears the override down in ``finally`` and
        # leaves the native ``torch.erf`` measured above untouched elsewhere.
        flag_gems.only_enable(lib=lib, include=["erf"])
        inp = x.clone().requires_grad_(True)
        out = ErfFunction.apply(inp)
        (res_grad,) = torch.autograd.grad(out, (inp,), (grad_output,))
    finally:
        lib._destroy()

    utils.gems_assert_close(res_grad, ref_grad_in, torch.float32)
