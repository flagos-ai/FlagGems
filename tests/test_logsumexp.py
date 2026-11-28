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

from .accuracy_utils import (
    FLOAT_DTYPES,
    REDUCTION_SHAPES,
    gems_assert_close,
    gems_log_logger,
    gems_log_prefix,
    to_reference,
)
from .conftest import QUICK_MODE

DIM_LIST = [0] if QUICK_MODE else [0, 1]

# The end-to-end gradient comparison inherits the rounding of the *forward*
# result: ATen computes ``grad * (input - result).exp()`` from the saved result,
# so a one-ulp difference between the two forward implementations moves the
# gradient by one backward relative to ``result``'s dtype resolution. The
# forward test already accepts ``atol=5e-3`` for the same reason, and
# ``test_logsumexp_backward_matches_aten_composite`` checks the backward kernel
# itself against the exact ATen expression.
# ``exp`` amplifies that shift, so the bound follows the mantissa of the saved
# result: one bf16 ulp near log(64) is already 6e-2. FP32 and FP16 stay tight.
GRAD_ATOL = {torch.float16: 5e-3, torch.bfloat16: 2e-2, torch.float32: 1e-4}
# Low-precision forward results can differ from the reference by a few ulps,
# and the backward multiplies that difference by ``exp(input - result)``, so the
# end-to-end gradient bound carries the same factor on top of the dtype
# resolution. FP32 has no such amplification here.
GRAD_RTOL_SCALE = {torch.float16: 4, torch.bfloat16: 4, torch.float32: 1}


def _grad_atol(dtype):
    return GRAD_ATOL.get(dtype, 1e-4)


def _assert_grad_close(res_grad, ref_grad, dtype):
    if res_grad.device != ref_grad.device:
        # ``--ref cpu`` computes the reference gradient on the host.
        res_grad, ref_grad = res_grad.cpu(), ref_grad.cpu()
    torch.testing.assert_close(
        res_grad,
        ref_grad,
        atol=_grad_atol(dtype),
        rtol=GRAD_RTOL_SCALE.get(dtype, 1) * flag_gems.testing.RESOLUTION[dtype],
    )


# (shape, dim, keepdim). Single axis, negative axis and multi-axis reductions;
# both ``keepdim`` values; 2D, 3D and 4D inputs. The shapes stay small because
# every parameter combination runs a native autograd graph; the benchmark file
# covers the large default shapes.
BACKWARD_CASES = (
    [((4, 8), 1, False), ((4, 8), 0, True)]
    if QUICK_MODE
    else [
        ((4, 8), 0, False),
        ((4, 8), 1, False),
        ((4, 8), -1, False),
        ((4, 8), 0, True),
        ((4, 8), 1, True),
        ((16, 32, 8), 1, False),
        ((16, 32, 8), -2, False),
        ((16, 32, 8), (0, 2), False),
        ((16, 32, 8), (0, 2), True),
        ((16, 32, 8), (-1, 0), True),
        ((16, 32, 8), [1, 2], False),
        ((2, 3, 4, 5), 2, False),
        ((2, 3, 4, 5), (1, 3), False),
        ((2, 3, 4, 5), (0, 1, 2), True),
        # Wide innermost rows take the autotuned inner path (N > 1024).
        ((2, 2048), 1, False),
        ((3, 4096), 1, True),
        # Many rows with a strided reduction take the per-row loop path
        # (M >= 512, K > 1).
        ((1024, 8, 4), 1, False),
        ((600, 3, 5), 1, True),
    ]
)


def _reduced_dims(dim, ndim):
    if isinstance(dim, (list, tuple)):
        return sorted({d % ndim for d in dim})
    return [dim % ndim]


@pytest.mark.logsumexp_backward
@pytest.mark.parametrize("shape,dim,keepdim", BACKWARD_CASES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_logsumexp_backward(shape, dtype, dim, keepdim):
    # Deterministic inputs: the gradient inherits the forward rounding, so a
    # fixed seed keeps this comparison reproducible run to run.
    torch.manual_seed(1234)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = to_reference(inp.detach().clone().requires_grad_(True))

    ref_out = torch.logsumexp(ref_inp, dim=dim, keepdim=keepdim)
    out_grad = torch.randn(ref_out.shape, dtype=ref_out.dtype, device=flag_gems.device)

    res_out = flag_gems.logsumexp(inp, dim=dim, keepdim=keepdim)
    assert type(res_out.grad_fn).__name__ == "_LogSumExpBackward"
    assert res_out.shape == ref_out.shape
    (ref_grad,) = torch.autograd.grad(ref_out, (ref_inp,), (to_reference(out_grad),))
    (res_grad,) = torch.autograd.grad(res_out, (inp,), (out_grad,))

    # Same forward bound as the pre-existing test: atol scales with the size of
    # the reduced axis, which is what the accumulation error scales with.
    reduce_size = 1
    for axis in _reduced_dims(dim, inp.ndim):
        reduce_size *= shape[axis]
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_size, atol=5e-3)
    assert res_grad.shape == inp.shape and res_grad.dtype == inp.dtype
    _assert_grad_close(res_grad, ref_grad, dtype)


@pytest.mark.logsumexp_backward
@pytest.mark.parametrize("shape,dim,keepdim", BACKWARD_CASES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_logsumexp_backward_matches_aten_composite(shape, dtype, dim, keepdim):
    """The backward kernel must reproduce ATen's composite expression.

    ATen derives the gradient as ``grad_out * (input - result).exp()``, rounding
    every step to the input dtype. Feeding the candidate its own saved keepdim
    result isolates the backward from the forward, so this comparison stays at
    one dtype resolution instead of the one-ulp slack the end-to-end test needs.
    """
    torch.manual_seed(4321)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    saved = flag_gems.logsumexp(inp, dim=dim, keepdim=True)
    out_shape = list(saved.shape)
    if not keepdim:
        for reduced in sorted(_reduced_dims(dim, inp.ndim), reverse=True):
            del out_shape[reduced]
    out_grad = torch.randn(out_shape, dtype=dtype, device=flag_gems.device)

    grad_inp = inp.detach().clone().requires_grad_(True)
    out_g = flag_gems.logsumexp(grad_inp, dim=dim, keepdim=keepdim)
    (res_grad,) = torch.autograd.grad(out_g, (grad_inp,), (out_grad,))

    expected = out_grad
    if not keepdim:
        for reduced in _reduced_dims(dim, inp.ndim):
            expected = expected.unsqueeze(reduced)
    expected = expected * (inp - saved).exp()

    torch.testing.assert_close(
        res_grad, expected, atol=1e-6, rtol=2 * flag_gems.testing.RESOLUTION[dtype]
    )


@pytest.mark.logsumexp_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("keepdim", [True, False])
def test_accuracy_logsumexp_backward_large_and_neg_inf(dtype, keepdim):
    """Large magnitudes plus an all ``-inf`` reduction slice.

    ATen computes ``grad * exp(input - result)``, so an all ``-inf`` slice
    produces ``nan`` (``exp(-inf + inf)``); the candidate must match that
    instead of inventing a zero gradient. The value check uses the candidate's
    own saved result because a single-ulp difference in ``result`` is amplified
    to ``exp(ulp)`` for magnitudes this large, in the native reference too.
    """
    shape = (32, 17)
    torch.manual_seed(2468)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 30.0
    inp[0, :] = float("-inf")

    saved = flag_gems.logsumexp(inp, dim=1, keepdim=True)
    out = flag_gems.logsumexp(inp, dim=1, keepdim=keepdim)
    ref_saved = to_reference(saved)
    ref_out = to_reference(out)
    out_grad = torch.randn(out.shape, dtype=dtype, device=flag_gems.device)

    grad_inp = inp.detach().clone().requires_grad_(True)
    out_g = flag_gems.logsumexp(grad_inp, dim=1, keepdim=keepdim)
    (res_grad,) = torch.autograd.grad(out_g, (grad_inp,), (out_grad,))

    expected = (out_grad if keepdim else out_grad.unsqueeze(1)) * (inp - saved).exp()
    assert bool(torch.isnan(expected[0]).all())
    assert bool(torch.isnan(res_grad[0]).all())
    # Two dtype resolutions: Triton's ``exp`` may round the last bit
    # differently from ATen's on a handful of elements.
    torch.testing.assert_close(
        res_grad,
        expected,
        atol=1e-6,
        rtol=2 * flag_gems.testing.RESOLUTION[dtype],
        equal_nan=True,
    )
    gems_assert_close(out, ref_out, dtype, reduce_dim=shape[1], atol=5e-3)
    assert bool(torch.isinf(ref_saved[0]).all())


@pytest.mark.logsumexp_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_logsumexp_backward_non_contiguous(dtype):
    """A transposed input must still produce the gradient of its own layout."""
    torch.manual_seed(1357)
    base = torch.randn(12, 16, 8, dtype=dtype, device=flag_gems.device)
    inp = base.transpose(0, 1).detach().requires_grad_(True)
    assert not inp.is_contiguous()
    ref_inp = to_reference(inp.detach().clone().requires_grad_(True))

    ref_out = torch.logsumexp(ref_inp, dim=1)
    out_grad = torch.randn(ref_out.shape, dtype=ref_out.dtype, device=flag_gems.device)
    res_out = flag_gems.logsumexp(inp, dim=1)
    assert res_out.shape == ref_out.shape

    (ref_grad,) = torch.autograd.grad(ref_out, (ref_inp,), (to_reference(out_grad),))
    (res_grad,) = torch.autograd.grad(res_out, (inp,), (out_grad,))

    assert res_grad.shape == inp.shape
    _assert_grad_close(res_grad, ref_grad, dtype)


@pytest.mark.logsumexp_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_logsumexp_backward_empty_reduction(dtype):
    """An empty reduction axis yields ``-inf`` and an empty gradient.

    ATen does not raise for ``dim`` of size zero: every output element is
    ``-inf`` and the returned gradient is an empty tensor.
    """
    inp = torch.randn((4, 0, 3), dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)
    ref_out = torch.logsumexp(ref_inp, dim=1)
    res_out = flag_gems.logsumexp(inp, dim=1)
    assert res_out.shape == ref_out.shape
    assert bool(torch.isinf(res_out).all()) and bool((res_out < 0).all())
    gems_assert_close(res_out, ref_out, dtype)

    grad_inp = inp.detach().clone().requires_grad_(True)
    out = flag_gems.logsumexp(grad_inp, dim=1)
    assert out.requires_grad
    (res_grad,) = torch.autograd.grad(out, (grad_inp,), (torch.ones_like(out),))
    assert res_grad.shape == grad_inp.shape and res_grad.numel() == 0


@pytest.mark.logsumexp_backward
def test_logsumexp_backward_no_grad_input():
    """Without a differentiable input the entry must not create a graph."""
    inp = torch.randn((8, 16), device=flag_gems.device)
    out = flag_gems.logsumexp(inp, dim=1)
    assert not out.requires_grad and out.grad_fn is None


@pytest.mark.logsumexp
@pytest.mark.skipif(
    flag_gems.logsumexp.__module__ != "flag_gems.ops.logsumexp",
    reason="covers the generic logsumexp implementation",
)
@pytest.mark.parametrize(
    "shape, dim",
    [
        ((0,), 0),
        ((0, 2), 1),
    ],
)
@pytest.mark.parametrize("keepdim", [False, True])
def test_logsumexp_empty_integer(shape, dim, keepdim, caplog):
    inp = torch.empty(shape, dtype=torch.int64, device=flag_gems.device)
    ref_out = torch.logsumexp(to_reference(inp), dim=dim, keepdim=keepdim)
    with caplog.at_level("DEBUG", logger=gems_log_logger(flag_gems.logsumexp)):
        res_out = flag_gems.logsumexp(inp, dim=dim, keepdim=keepdim)
    assert f"{gems_log_prefix(flag_gems.logsumexp)} LOGSUMEXP" in caplog.text
    torch.testing.assert_close(res_out, ref_out.to(res_out.device), rtol=0, atol=0)


@pytest.mark.logsumexp
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("keepdim", [True, False])
def test_accuracy_logsumexp(shape, dtype, dim, keepdim):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.logsumexp(ref_inp, dim=dim, keepdim=keepdim)
    res_out = flag_gems.logsumexp(inp, dim=dim, keepdim=keepdim)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim], atol=5e-3)


def _skip_if_aten_multidim_logsumexp_aborts():
    # The reference side of this case is a raw aten call on the device
    # (`torch.logsumexp(x, (0, 1))`, 9-D input). On this backend that aten
    # entry aborts the whole process (SIGABRT) for a multi-axis reduction, so
    # the vendor-neutral duplicate-dim assertion above runs on every vendor and
    # only the aten comparison is skipped here.
    if flag_gems.vendor_name == "enflame":
        pytest.skip("enflame: aten::logsumexp aborts for a multi-axis reduction")


@pytest.mark.logsumexp_backward
def test_logsumexp_backward_high_rank_and_duplicate_dims():
    x = torch.randn((2,) * 9, device=flag_gems.device, requires_grad=True)
    with pytest.raises(RuntimeError, match="appears multiple times"):
        flag_gems.logsumexp(x, (0, -9))
    _skip_if_aten_multidim_logsumexp_aborts()
    ref = torch.logsumexp(x, (0, 1))
    result = flag_gems.logsumexp(x, (0, 1))
    grad = torch.randn_like(ref)
    torch.testing.assert_close(
        torch.autograd.grad(result, x, grad)[0],
        torch.autograd.grad(ref, x, grad)[0],
    )
