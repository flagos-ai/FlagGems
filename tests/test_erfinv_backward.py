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
from flag_gems.ops.erfinv_backward import ErfinvFunction

from . import accuracy_utils as utils

_SQRTPI_OVER_2 = math.sqrt(math.pi) / 2.0


def _erfinv_backward_reference(grad_output, x):
    """Gradient of ``erfinv``, reproducing ATen's per-op rounding order.

    ATen has no dedicated backward kernel for ``erfinv``: the autograd formula
    is ``grad * sqrt(pi)/2 * exp(self.erfinv().square())``. Every step of that
    formula is a separate elementwise kernel whose result is written back in
    the storage dtype (fp16/bf16 included, no float32 accumulator is kept
    across steps), so the reference below rounds at the same places and is
    bit-exact against ``torch.autograd.grad(torch.erfinv(x), x)`` on the same
    device.

    ``y * y`` is used instead of ``y.pow(2)`` so the golden does not depend on
    any dispatched ``aten::pow`` implementation.

    Under ``--ref cpu`` the reference runs on CPU instead, where ``torch.exp``
    is not bit-identical to the CUDA ``expf``; callers use
    ``gems_assert_close`` in that mode and ``gems_assert_equal`` on device.
    """
    if x.numel() == 0:
        return torch.empty(x.shape, dtype=x.dtype, device=x.device)
    y = torch.erfinv(x)
    squared = y * y
    exponential = torch.exp(squared)
    return (_SQRTPI_OVER_2 * exponential) * grad_output


def _assert_backward(res_grad, ref_grad):
    """Bit-exact on device; tolerance-based when the reference is on CPU."""
    if utils.TO_CPU:
        # The gradient of erfinv grows like exp(erfinv(x)^2) and is unbounded
        # near |x| = 1, so an absolute tolerance has to scale with the values;
        # a relative comparison against the CPU reference is the right gate.
        utils.gems_assert_close(res_grad, ref_grad, res_grad.dtype, atol=1e-4)
    else:
        utils.gems_assert_equal(res_grad, ref_grad)


@pytest.mark.erfinv_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_erfinv_backward_bit_exact(shape, dtype):
    """Bit-exactness against the eager on-device autograd chain.

    The kernel rounds each intermediate (erfinv, square, exp, scalar mul, grad
    mul) to the storage dtype, the same places ATen's five-kernel chain does,
    so the comparison is exact rather than tolerance based.
    """
    torch.manual_seed(0)
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # erfinv is only real-valued on (-1, 1); keep the samples strictly inside
    # so the comparison does not hinge on inf/nan agreement.
    x = (
        torch.empty(shape, dtype=dtype, device=flag_gems.device)
        .uniform_(-0.99, 0.99)
        .to(dtype)
    )

    ref_grad = _erfinv_backward_reference(
        utils.to_reference(grad_output), utils.to_reference(x)
    )
    res_grad = flag_gems.erfinv_backward(grad_output, x)

    _assert_backward(res_grad, ref_grad)


@pytest.mark.erfinv_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_erfinv_backward_autograd(dtype):
    # ``aten::erfinv`` keeps ATen's own autograd formula node, so the gradient
    # has to be driven through the FlagGems autograd bridge explicitly;
    # otherwise this would exercise the native chain instead of the Triton
    # kernel.
    torch.manual_seed(0)
    inp = (
        torch.empty((1024,), dtype=dtype, device=flag_gems.device)
        .uniform_(-0.99, 0.99)
        .to(dtype)
    ).requires_grad_()
    out = ErfinvFunction.apply(inp)
    assert out.requires_grad and out.grad_fn is not None

    grad_output = torch.randn_like(out)
    (res_grad,) = torch.autograd.grad(out, (inp,), (grad_output,))

    ref_grad = _erfinv_backward_reference(
        utils.to_reference(grad_output), utils.to_reference(inp)
    )
    _assert_backward(res_grad, ref_grad)


@pytest.mark.erfinv_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_erfinv_backward_non_contiguous(dtype):
    torch.manual_seed(0)
    base = (
        torch.empty((64, 64), dtype=dtype, device=flag_gems.device)
        .uniform_(-0.99, 0.99)
        .to(dtype)
    )
    x = base.t()
    assert not x.is_contiguous()
    grad_output = torch.randn_like(x)

    ref_grad = _erfinv_backward_reference(
        utils.to_reference(grad_output), utils.to_reference(x)
    )
    res_grad = flag_gems.erfinv_backward(grad_output, x)

    _assert_backward(res_grad, ref_grad)


@pytest.mark.erfinv_backward
def test_erfinv_backward_boundary():
    """Domain edges match native autograd: inf at +-1, nan outside.

    ``erfinv`` diverges as its argument approaches +-1, so the gradient is
    unbounded there. The kernel must not clamp or otherwise "fix" those
    values -- ATen returns inf (or nan outside the domain) and so must we.
    """
    x = torch.tensor(
        [1.0, -1.0, 1.5, -1.5, float("nan"), 0.0, -0.0, 0.9999, -0.9999],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    grad_output = torch.ones_like(x)

    ref_inp = utils.to_reference(x).requires_grad_(True)
    (ref_grad,) = torch.autograd.grad(
        torch.erfinv(ref_inp), (ref_inp,), (utils.to_reference(grad_output),)
    )
    res_grad = flag_gems.erfinv_backward(grad_output, x)

    res_cpu = res_grad.cpu()
    ref_cpu = ref_grad.cpu()
    both_nan = torch.isnan(res_cpu) & torch.isnan(ref_cpu)
    same_inf = torch.isinf(res_cpu) & torch.isinf(ref_cpu) & (res_cpu == ref_cpu)
    finite_close = torch.isclose(res_cpu, ref_cpu, rtol=1e-4, atol=0.0)
    assert (both_nan | same_inf | finite_close).all()


@pytest.mark.erfinv_backward
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_erfinv_backward_lowp_tail_bit_exact(dtype):
    """fp16/bf16 tails stay bit-exact where the gradient is most extreme.

    The kernel's ``LOWP`` path rounds every intermediate to the storage dtype.
    The harshest spot is the tail near |x| = 1: ``exp(erfinv(x)^2)`` overflows
    the low-precision range there, so the boundary between finite and inf (and
    the rounding of the largest finite values) is exactly what this test pins
    down against native autograd.
    """
    torch.manual_seed(0)
    # Dense sampling near both tails, plus a couple of mid-range controls.
    x = torch.tensor(
        [0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99995, 0.99999]
        + [-v for v in (0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99995, 0.99999)]
        + [0.0, 0.5, -0.5],
        dtype=dtype,
        device=flag_gems.device,
    )
    grad_output = torch.ones_like(x)

    ref_inp = x.detach().clone().requires_grad_(True)
    (ref_grad,) = torch.autograd.grad(torch.erfinv(ref_inp), (ref_inp,), (grad_output,))
    res_grad = flag_gems.erfinv_backward(grad_output, x)

    # Tails are the point of this test: finite elements must match bit-exactly
    # (including the largest finite values), and every non-finite element must
    # agree in both kind and sign.
    res_c, ref_c = res_grad.cpu(), ref_grad.cpu()
    utils.gems_assert_equal(res_c[torch.isfinite(ref_c)], ref_c[torch.isfinite(ref_c)])
    assert torch.equal(torch.isnan(res_c), torch.isnan(ref_c))
    assert torch.equal(
        torch.sign(res_c[torch.isinf(res_c)]), torch.sign(ref_c[torch.isinf(ref_c)])
    )


@pytest.mark.erfinv_backward
def test_erfinv_backward_dispatch():
    """The registered forward and the bridge share one gradient.

    ``flag_gems.only_enable`` installs the same registration the public enable
    path uses, so the forward that runs here is the FlagGems ``erfinv``
    kernel. Comparing its gradient against the fp64 autograd reference taken
    before the registration proves the bridge is reachable from a dispatched
    forward.
    """
    x = (
        torch.empty((32, 48), dtype=torch.float32, device=flag_gems.device)
        .uniform_(-0.99, 0.99)
        .to(torch.float32)
    )
    grad_output = torch.randn_like(x)

    ref_inp = utils.to_reference(x, True).requires_grad_(True)
    ref_grad = utils.to_reference(grad_output, True)
    (ref_grad_in,) = torch.autograd.grad(torch.erfinv(ref_inp), (ref_inp,), (ref_grad,))

    lib = torch.library.Library("aten", "IMPL")
    try:
        # Scoped registration: unlike the global ``flag_gems.use_gems()``
        # context manager, a private Library tears the override down in
        # ``finally`` and leaves the native ``torch.erfinv`` measured above
        # untouched elsewhere.
        flag_gems.only_enable(lib=lib, include=["erfinv"])
        inp = x.clone().requires_grad_(True)
        out = ErfinvFunction.apply(inp)
        (res_grad,) = torch.autograd.grad(out, (inp,), (grad_output,))
    finally:
        lib._destroy()

    utils.gems_assert_close(res_grad, ref_grad_in, torch.float32)
