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

import importlib

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

LOGCUMSUMEXP_SHAPES = (
    [(2, 32)]
    if utils.QUICK_MODE
    else utils.REDUCTION_SHAPES + [(2637,), (16, 1025, 255)]
)


@pytest.mark.logcumsumexp
@pytest.mark.parametrize("shape", LOGCUMSUMEXP_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logcumsumexp(shape, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    dim = 1 if shape == utils.REDUCTION_SHAPES[-1] else -1
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.logcumsumexp(ref_inp, dim=dim)
    res_out = flag_gems.logcumsumexp(inp, dim=dim)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])


@pytest.mark.logcumsumexp_out
@pytest.mark.parametrize("shape", LOGCUMSUMEXP_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logcumsumexp_out(shape, dtype):
    dim = 1 if shape == utils.REDUCTION_SHAPES[-1] else -1
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out_buf = torch.empty_like(ref_inp)
    ref_out = torch.ops.aten.logcumsumexp.out(ref_inp, dim, out=ref_out_buf)

    res_out_buf = torch.empty_like(inp)
    res_out = flag_gems.logcumsumexp_out(inp, dim, out=res_out_buf)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])


BACKWARD_CASES = (
    [((2, 32), -1)]
    if utils.QUICK_MODE
    else [
        ((2, 32), -1),
        ((64, 64), 1),
        ((4096, 256), 1),
        ((200, 1025, 3), 1),
        ((2637,), -1),
    ]
)


def _alternating_grad(shape, dim, dtype):
    """Upstream gradient with alternating signs along the scanned axis.

    The backward of ``logcumsumexp`` accumulates along ``dim``; an all-positive
    gradient cannot distinguish the correct reverse scan from a wrong
    ``exp(x - y) * reverse_cumsum(g)`` factorization, while alternating signs
    can.
    """
    grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    pattern = torch.ones(shape[dim], dtype=dtype, device=flag_gems.device)
    pattern[1::2] = -1.0
    view = [1] * len(shape)
    view[dim] = shape[dim]
    return grad * pattern.view(view)


def _native_backward_fp64(grad_output, inp, out, dim):
    """float64 evaluation of ``torch/csrc/autograd/FunctionsManual.cpp``.

    ``logcumsumexp_backward(grad, self, result, dim)`` splits the upstream
    gradient by sign, reverse-logcumsumexps both parts in log space and subtracts
    them, which is exactly ``sum_{j >= i} g_j * exp(x_i - y_j)``.
    """
    dim = dim % inp.dim()
    grad = grad_output.to(torch.float64)
    inp = inp.to(torch.float64)
    out = out.to(torch.float64)
    if grad.dim() == 0 or grad.numel() == 0:
        return grad.clone().cpu()
    grad_min = torch.scalar_tensor(
        torch.finfo(grad.dtype).min, dtype=grad.dtype, device=grad.device
    )
    log_abs_grad = grad.abs().log()
    log_grad_positive = torch.where(grad > 0, log_abs_grad, grad_min)
    log_grad_negative = torch.where(grad < 0, log_abs_grad, grad_min)

    def reverse_logcumsumexp(x):
        return torch.flip(torch.logcumsumexp(torch.flip(x, [dim]), dim), [dim])

    positive = (reverse_logcumsumexp(log_grad_positive - out) + inp).exp()
    negative = (reverse_logcumsumexp(log_grad_negative - out) + inp).exp()
    # ``gems_assert_close`` mirrors the repository reference convention, which is
    # a CPU comparison tensor.
    return (positive - negative).cpu()


@pytest.mark.logcumsumexp_backward
@pytest.mark.parametrize("shape, dim", BACKWARD_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logcumsumexp_backward(shape, dim, dtype):
    """Gradient of ``flag_gems.logcumsumexp`` against the native formula.

    The reference is evaluated in float64 from the *candidate's* forward result.
    That isolates the backward: comparing against a float64 forward instead would
    fold the forward's dtype rounding into the gradient metric, which for bf16 is
    an order of magnitude larger than the backward kernel's own error (the native
    bf16 backward chain itself exceeds the repository tolerance on 4% of the
    elements of a (64, 64) case). End-to-end float64 comparison is covered by
    ``test_logcumsumexp_backward_autograd_chain``.
    """
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_out = _alternating_grad(shape, dim, dtype)

    res_inp = inp.detach().requires_grad_(True)
    res_out = flag_gems.logcumsumexp(res_inp, dim)
    (res_grad,) = torch.autograd.grad(res_out, res_inp, grad_out)

    ref_inp = utils.to_reference(inp.detach(), True)
    ref_out = torch.logcumsumexp(ref_inp, dim)
    ref_grad = _native_backward_fp64(grad_out, inp, res_out, dim)

    assert res_out.grad_fn is not None
    assert "LogcumsumexpFunction" in type(res_out.grad_fn).__name__

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])
    utils.gems_assert_close(res_grad, ref_grad.to(dtype), dtype, reduce_dim=shape[dim])


@pytest.mark.logcumsumexp_backward
@pytest.mark.parametrize("dim", [1, -1])
def test_logcumsumexp_backward_autograd_chain(dim):
    """End-to-end float32 autograd chain against the float64 torch chain."""
    shape = (128, 257)
    dtype = torch.float32
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_out = _alternating_grad(shape, dim, dtype)

    ref_inp = utils.to_reference(inp.detach(), True).requires_grad_(True)
    ref_grad_out = utils.to_reference(grad_out, True)
    ref_out = torch.logcumsumexp(ref_inp, dim)
    (ref_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad_out)

    res_inp = inp.detach().requires_grad_(True)
    res_out = flag_gems.logcumsumexp(res_inp, dim)
    (res_grad,) = torch.autograd.grad(res_out, res_inp, grad_out)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])
    utils.gems_assert_close(res_grad, ref_grad.to(dtype), dtype, reduce_dim=shape[dim])


@pytest.mark.logcumsumexp_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logcumsumexp_backward_non_contiguous(dtype):
    base = torch.randn((48, 64), dtype=dtype, device=flag_gems.device)
    inp = base.t()
    dim = -1
    grad_out = _alternating_grad(inp.shape, dim, dtype)

    res_inp = inp.detach().requires_grad_(True)
    res_out = flag_gems.logcumsumexp(res_inp, dim)
    (res_grad,) = torch.autograd.grad(res_out, res_inp, grad_out)

    ref_out = torch.logcumsumexp(utils.to_reference(inp.detach(), True), dim)
    ref_grad = _native_backward_fp64(grad_out, inp, res_out, dim)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=inp.shape[dim])
    utils.gems_assert_close(
        res_grad, ref_grad.to(dtype), dtype, reduce_dim=inp.shape[dim]
    )


@pytest.mark.logcumsumexp_backward
@pytest.mark.parametrize("dim", [0, 1, -2])
def test_logcumsumexp_backward_negative_and_middle_dim(dim):
    shape = (4, 8, 16)
    dtype = torch.float32
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_out = _alternating_grad(shape, dim, dtype)

    res_inp = inp.detach().requires_grad_(True)
    res_out = flag_gems.logcumsumexp(res_inp, dim)
    (res_grad,) = torch.autograd.grad(res_out, res_inp, grad_out)

    ref_grad = _native_backward_fp64(grad_out, inp, res_out, dim)
    utils.gems_assert_close(res_grad, ref_grad.to(dtype), dtype, reduce_dim=shape[dim])


@pytest.mark.logcumsumexp_backward
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_logcumsumexp_backward_wide_range(dtype, monkeypatch):
    """Rows spanning ~2000 must stay finite.

    The native backward needs this: ``result`` can exceed the fp32 exponent
    range, so a whole-row ``exp(x - result)`` would overflow. The kernel's
    factored recurrence keeps every multiplier <= 1. The forward result is taken
    from torch so the test measures the backward, not the forward kernel's own
    range limits (tracked separately in artifacts/forward_range_limitation.md).
    """
    torch.manual_seed(7)
    shape = (8, 4096)
    dim = -1
    inp = (torch.rand(shape, dtype=dtype, device=flag_gems.device) - 0.5) * 2000.0
    grad_out = _alternating_grad(shape, dim, dtype)
    out = torch.logcumsumexp(inp, dim)

    # Inject the native forward result to isolate the wide-range backward
    # while still exercising the public FlagGems autograd entry.
    module = importlib.import_module("flag_gems.ops.logcumsumexp")
    monkeypatch.setattr(module, "logcumsumexp_wrapper", lambda *args: out)
    res_inp = inp.detach().requires_grad_(True)
    res_out = flag_gems.logcumsumexp(res_inp, dim)
    (res_grad,) = torch.autograd.grad(res_out, res_inp, grad_out)
    ref_grad = _native_backward_fp64(grad_out, inp, out, dim)

    assert torch.isfinite(res_grad).all()
    utils.gems_assert_close(res_grad, ref_grad.to(dtype), dtype, reduce_dim=shape[dim])


@pytest.mark.logcumsumexp_backward
@pytest.mark.parametrize("shape", [(0, 5), (5, 0), (3, 0, 7)])
def test_logcumsumexp_backward_zero_length(shape):
    dim = 1
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    grad_out = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)

    ref_inp = utils.to_reference(inp.detach(), True).requires_grad_(True)
    ref_grad_out = utils.to_reference(grad_out, True)
    ref_out = torch.logcumsumexp(ref_inp, dim)
    (ref_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad_out)

    res_inp = inp.detach().requires_grad_(True)
    res_out = flag_gems.logcumsumexp(res_inp, dim)
    (res_grad,) = torch.autograd.grad(res_out, res_inp, grad_out)

    assert res_grad.shape == inp.shape
    assert res_grad.dtype == inp.dtype
    assert res_grad.numel() == 0
    assert ref_grad.numel() == 0


@pytest.mark.logcumsumexp_backward
def test_logcumsumexp_backward_scalar_input():
    inp = torch.tensor(2.5, dtype=torch.float32, device=flag_gems.device)
    res_inp = inp.detach().requires_grad_(True)
    res_out = flag_gems.logcumsumexp(res_inp, 0)
    (res_grad,) = torch.autograd.grad(res_out, res_inp, torch.ones_like(res_out))

    ref_inp = utils.to_reference(inp.detach(), True).requires_grad_(True)
    ref_out = torch.logcumsumexp(ref_inp, 0)
    (ref_grad,) = torch.autograd.grad(ref_out, ref_inp, torch.ones_like(ref_out))

    assert res_out.detach().item() == pytest.approx(float(ref_out.detach().item()))
    assert res_grad.item() == pytest.approx(float(ref_grad.item()))


@pytest.mark.logcumsumexp_backward
def test_logcumsumexp_backward_use_gems_entry():
    """Compare the FlagGems autograd entry against native PyTorch."""
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    shape, dim, dtype = (64, 128), 1, torch.float32
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_out = _alternating_grad(shape, dim, dtype)

    ref_inp = utils.to_reference(inp.detach(), True).requires_grad_(True)
    ref_grad_out = utils.to_reference(grad_out, True)
    ref_out = torch.logcumsumexp(ref_inp, dim)
    (ref_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad_out)

    res_inp = inp.detach().requires_grad_(True)
    res_out = flag_gems.logcumsumexp(res_inp, dim)
    (res_grad,) = torch.autograd.grad(res_out, res_inp, grad_out)

    assert type(res_out.grad_fn).__name__ == "LogcumsumexpFunctionBackward"
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])
    utils.gems_assert_close(res_grad, ref_grad.to(dtype), dtype, reduce_dim=shape[dim])
