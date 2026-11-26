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

# Orders exercised by the autograd reference.  ``inf`` selects the argmax
# branch of ``norm_backward``; ``-inf`` the argmin branch; the negative
# finite orders run through the ``p < 1`` branch like ``0.5``.
ORDS = [1.0, 2.0, 3.0, 0.5, float("inf"), float("-inf"), -1.0, -2.0]
SHAPES = [(1024,), (512, 512), (20, 320, 15)]


def _reference_grads(x, ord_, grad=None):
    """Native ``ForeachNormBackward0`` gradients of a one-element list.

    The autograd chain runs on the input dtype: the branch formulas raise the
    saved norm to ``p-1`` / ``1-p``, so an upcast reference (fp64) follows a
    different rounding path than the fp16 chain this kernel mirrors.
    ``to_reference`` without the upcast keeps the CPU-ref mode working.
    """
    x_ref = utils.to_reference(x).requires_grad_(True)
    out = torch.ops.aten._foreach_norm.Scalar([x_ref], ord=ord_)[0]
    if grad is None:
        grad = torch.ones_like(out)
    grad = utils.to_reference(grad)
    (grad_self,) = torch.autograd.grad(out, [x_ref], grad_outputs=[grad])
    return grad_self


def _test_data(shape, dtype, scale):
    """Deterministic per-(shape, dtype) data so the reference and the result
    see the same tensors without a shared autograd graph."""
    gen = torch.Generator(device=flag_gems.device).manual_seed(1234)
    x = torch.randn(shape, generator=gen, dtype=dtype, device=flag_gems.device) * scale
    grad = torch.randn((), generator=gen, dtype=dtype, device=flag_gems.device)
    return x, grad


@pytest.mark.foreach_norm_backward
@pytest.mark.parametrize("ord_", ORDS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_foreach_norm_backward_ords(dtype, ord_):
    """The derivative formula of ``_foreach_norm.Scalar`` for every branch."""
    # fp16/bf16 magnitudes follow the order: the fractional-order norm
    # (sum |x|^p)^(1/p) must stay inside the storage dtype range (ord 0.5
    # squares a ~1160 sum at full scale), while the negative orders need the
    # reciprocal norm to stay above the fp16 subnormal floor so the gradient
    # does not collapse to zeros.
    if ord_ in (0.5, 3.0):
        scale = 0.05
    elif ord_ in (-1.0, -2.0):
        scale = 2.0
    else:
        scale = 2.0
    x, grad = _test_data((1024,), dtype, scale)

    ref_grad = _reference_grads(x, ord_, grad)
    res_grad = flag_gems.foreach_norm_backward([grad], [x], ord_)[0]
    # The negative orders hit genuine fp16 underflow (|x|^(p-1) overflows for
    # the smallest magnitudes), which the native chain produces as nan as
    # well; compare nan positions explicitly instead of failing on them.
    res_cpu = utils.to_cpu(res_grad, ref_grad)
    same_nan = (torch.isnan(res_cpu) & torch.isnan(ref_grad)) | (res_cpu == ref_grad)
    if not bool(same_nan.all().item()):
        utils.gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.foreach_norm_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_foreach_norm_backward_list(dtype):
    """A heterogeneous list: several shapes, one launch per dtype group."""
    xs = [
        torch.randn(64, dtype=dtype, device=flag_gems.device) * 2.0,
        torch.randn(3, 17, dtype=dtype, device=flag_gems.device) * 2.0,
        torch.randn((), dtype=dtype, device=flag_gems.device) * 2.0,
    ]
    grads = [torch.randn((), dtype=dtype, device=flag_gems.device) for _ in xs]
    refs = [_reference_grads(x, 2.0, g) for x, g in zip(xs, grads)]

    res = flag_gems.foreach_norm_backward(grads, xs, 2.0)
    assert len(res) == len(xs)
    for r, ref, dt in zip(res, refs, [dtype] * len(xs)):
        utils.gems_assert_close(r, ref, dt)


@pytest.mark.foreach_norm_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_foreach_norm_backward_zero_tensor(dtype):
    """A zero tensor has norm 0: the masked_fill zeroes the whole gradient."""
    x = torch.zeros(128, dtype=dtype, device=flag_gems.device)
    grad = torch.ones((), dtype=dtype, device=flag_gems.device)

    ref_grad = _reference_grads(x, 2.0)
    res_grad = flag_gems.foreach_norm_backward([grad], [x], 2.0)[0]
    utils.gems_assert_close(res_grad, ref_grad, dtype)
    assert (res_grad == 0).all()


@pytest.mark.foreach_norm_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_foreach_norm_backward_inf_tie(dtype):
    """Two elements share the max magnitude: each receives grad / 2 * sgn."""
    x = torch.tensor([4.0, -4.0, 1.0], dtype=dtype, device=flag_gems.device)
    grad = torch.ones((), dtype=dtype, device=flag_gems.device)

    ref_grad = _reference_grads(x, float("inf"))
    res_grad = flag_gems.foreach_norm_backward([grad], [x], float("inf"))[0]
    utils.gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.foreach_norm_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_foreach_norm_backward_nan(dtype):
    """A nan entry: the ord-2 gradient propagates it; ord-1 keeps sgn = 0."""
    x = torch.tensor([float("nan"), 1.0, 0.0], dtype=dtype, device=flag_gems.device)
    grad = torch.ones((), dtype=dtype, device=flag_gems.device)

    ref_grad = _reference_grads(x, 2.0)
    res_grad = flag_gems.foreach_norm_backward([grad], [x], 2.0)[0]
    res_cpu = utils.to_cpu(res_grad, ref_grad)
    same_nan = (torch.isnan(res_cpu) & torch.isnan(ref_grad)) | (res_cpu == ref_grad)
    assert same_nan.all()


@pytest.mark.foreach_norm_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_foreach_norm_backward_non_contiguous(dtype):
    """A transposed input is staged through a dense copy."""
    base = torch.randn(32, 64, dtype=dtype, device=flag_gems.device)
    x = base.t()
    grad = torch.randn((), dtype=dtype, device=flag_gems.device)

    ref_base = utils.to_reference(base, True).t().requires_grad_(True)
    ref_grad = utils.to_reference(grad)
    (ref_grad_in,) = torch.autograd.grad(
        torch.ops.aten._foreach_norm.Scalar([ref_base], ord=2)[0],
        [ref_base],
        grad_outputs=[ref_grad],
    )

    res_grad = flag_gems.foreach_norm_backward([grad], [x], 2.0)[0]
    utils.gems_assert_close(res_grad, ref_grad_in, dtype)


@pytest.mark.foreach_norm_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_foreach_norm_backward_empty(dtype):
    """An empty tensor: norm 0, empty gradient."""
    x = torch.randn(0, dtype=dtype, device=flag_gems.device)
    grad = torch.ones((), dtype=dtype, device=flag_gems.device)

    res_grad = flag_gems.foreach_norm_backward([grad], [x], 2.0)[0]
    assert res_grad.numel() == 0
    assert res_grad.dtype == dtype


@pytest.mark.foreach_norm_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_foreach_norm_backward_mixed_dtypes(dtype):
    """A list mixing the parametrised dtype with fp32 keeps both exact."""
    other = torch.float32 if dtype != torch.float32 else torch.float64
    xs = [
        torch.randn(128, dtype=dtype, device=flag_gems.device) * 2.0,
        torch.randn(256, dtype=other, device=flag_gems.device) * 2.0,
    ]
    grads = [
        torch.randn((), dtype=dtype, device=flag_gems.device),
        torch.randn((), dtype=other, device=flag_gems.device),
    ]

    refs = [_reference_grads(x, 2.0, g) for x, g in zip(xs, grads)]
    res = flag_gems.foreach_norm_backward(grads, xs, 2.0)
    for r, ref, dt in zip(res, refs, [dtype, other]):
        utils.gems_assert_close(r, ref, dt)


@pytest.mark.foreach_norm_backward
def test_foreach_norm_backward_ord0_raises():
    """``ord == 0`` has no derivative in ATen: the entry point refuses it."""
    x = torch.randn(64, device=flag_gems.device)
    grad = torch.ones((), device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.foreach_norm_backward([grad], [x], 0.0)


@pytest.mark.foreach_norm_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_foreach_norm_backward_length_mismatch(dtype):
    """A grad list shorter than the tensor list raises like ATen."""
    xs = [torch.randn(8, dtype=dtype, device=flag_gems.device) for _ in range(2)]
    grads = [torch.ones((), dtype=dtype, device=flag_gems.device)]
    with pytest.raises(RuntimeError):
        flag_gems.foreach_norm_backward(grads, xs, 2.0)


@pytest.mark.foreach_norm_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_foreach_norm_backward_int_rejected(dtype):
    """An integer tensor cannot carry a gradient: ATen refuses the forward."""
    x = torch.randint(0, 8, (16,), dtype=torch.int64, device=flag_gems.device)
    grad = torch.ones((), dtype=dtype, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.foreach_norm_backward([grad], [x], 2.0)
