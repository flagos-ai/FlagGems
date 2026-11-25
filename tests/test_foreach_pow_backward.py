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
from flag_gems.ops.foreach_pow_backward import (
    ForeachPowFunction,
    ForeachPowScalarListFunction,
    foreach_pow_backward,
    foreach_pow_scalar_and_tensor_backward,
    foreach_pow_scalar_list_backward,
)

from . import accuracy_utils as utils


def _draw_lists(shapes, dtype, device):
    """One draw shared by both sides: positive bases/exponents keep every
    forward result finite, so the reference formulas stay comparable element
    for element."""
    self_list = [torch.rand(s, dtype=dtype, device=device) + 0.5 for s in shapes]
    exp_list = [torch.rand(s, dtype=dtype, device=device) * 1.5 + 0.25 for s in shapes]
    grad_list = [torch.randn(s, dtype=dtype, device=device) for s in shapes]
    ref_self = [
        utils.to_reference(t.detach().clone()).requires_grad_(True) for t in self_list
    ]
    ref_exp = [
        utils.to_reference(t.detach().clone()).requires_grad_(True) for t in exp_list
    ]
    ref_grad = [utils.to_reference(t.detach().clone()) for t in grad_list]
    return self_list, exp_list, grad_list, ref_self, ref_exp, ref_grad


@pytest.mark.foreach_pow_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_foreach_pow_backward_list(shape, dtype):
    """The explicit entry reproduces the ``ForeachPowBackward0`` pair."""
    self_list, exp_list, grad_list, ref_self, ref_exp, ref_grad = _draw_lists(
        [shape], dtype, flag_gems.device
    )

    ref_out = torch.ops.aten._foreach_pow.List(ref_self, ref_exp)
    grads = torch.autograd.grad(ref_out, ref_self + ref_exp, ref_grad + ref_grad)
    ref_gs, ref_ge = grads[:1], grads[1:]

    res_gs, res_ge = foreach_pow_backward(
        grad_list, self_list, exp_list, [ref_out[0].detach().to(flag_gems.device)]
    )
    utils.gems_assert_close(res_gs[0], ref_gs[0], dtype, equal_nan=True)
    utils.gems_assert_close(res_ge[0], ref_ge[0], dtype, equal_nan=True)
    assert res_gs[0].shape == ref_out[0].shape
    assert res_ge[0].shape == ref_out[0].shape


@pytest.mark.foreach_pow_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_foreach_pow_backward_list_multi(dtype):
    """Paired lists of three tensors, compared bit for bit."""
    shapes = [(32, 48), (7,), (5, 11)]
    self_list, exp_list, grad_list, ref_self, ref_exp, ref_grad = _draw_lists(
        shapes, dtype, flag_gems.device
    )

    ref_out = torch.ops.aten._foreach_pow.List(ref_self, ref_exp)
    grads = torch.autograd.grad(ref_out, ref_self + ref_exp, ref_grad + ref_grad)
    ref_gs, ref_ge = grads[:3], grads[3:]

    res_gs, res_ge = foreach_pow_backward(
        grad_list,
        self_list,
        exp_list,
        [r.detach().to(flag_gems.device) for r in ref_out],
    )
    for i in range(len(shapes)):
        utils.gems_assert_close(res_gs[i], ref_gs[i], dtype, equal_nan=True)
        utils.gems_assert_close(res_ge[i], ref_ge[i], dtype, equal_nan=True)


@pytest.mark.foreach_pow_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_foreach_pow_backward_scalar_list(dtype):
    """``.ScalarList``: one full-precision exponent per tensor, no exponent grad."""
    shapes = [(32, 48), (64,), (16, 8)]
    exponents = [0.5, 2.0, -1.5]
    self_list, _, grad_list, ref_self, _, ref_grad = _draw_lists(
        shapes, dtype, flag_gems.device
    )

    ref_out = torch.ops.aten._foreach_pow.ScalarList(ref_self, exponents)
    ref_gs = torch.autograd.grad(ref_out, ref_self, ref_grad)

    res_gs = foreach_pow_scalar_list_backward(grad_list, self_list, exponents)
    for i in range(len(shapes)):
        utils.gems_assert_close(res_gs[i], ref_gs[i], dtype, equal_nan=True)
        assert res_gs[i].shape == shapes[i]


@pytest.mark.foreach_pow_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_foreach_pow_backward_scalar_list_exp_zero(dtype):
    """A zero exponent zeroes the gradient of its tensor (exact zeros)."""
    exponents = [0.0, 3.0]
    shapes = [(16,), (16,)]
    self_list, _, grad_list, ref_self, _, ref_grad = _draw_lists(
        shapes, dtype, flag_gems.device
    )

    ref_out = torch.ops.aten._foreach_pow.ScalarList(ref_self, exponents)
    ref_gs = torch.autograd.grad(ref_out, ref_self, ref_grad)

    res_gs = foreach_pow_scalar_list_backward(grad_list, self_list, exponents)
    assert torch.all(res_gs[0] == 0).item()
    utils.gems_assert_close(res_gs[1], ref_gs[1], dtype)


@pytest.mark.foreach_pow_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_foreach_pow_backward_scalar_and_tensor(dtype):
    """``.ScalarAndTensor``: grad * (result * log(base)), right-associated."""
    shape = (32, 48)
    base = 2.0
    _, exp_list, grad_list, _, ref_exp, ref_grad = _draw_lists(
        [shape], dtype, flag_gems.device
    )

    ref_out = torch.ops.aten._foreach_pow.ScalarAndTensor(base, ref_exp)
    ref_ge = torch.autograd.grad(ref_out, ref_exp, ref_grad)

    res_ge = foreach_pow_scalar_and_tensor_backward(
        grad_list,
        [r.detach().to(flag_gems.device) for r in ref_out],
        exp_list,
        base,
    )
    utils.gems_assert_close(res_ge[0], ref_ge[0], dtype, equal_nan=True)
    assert res_ge[0].shape == shape


@pytest.mark.foreach_pow_backward
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_foreach_pow_backward_scalar_precision(dtype):
    """The scalar never rounds to the storage dtype (unlike foreach_maximum).

    A Python ``0.3`` exponent keeps full double precision inside ``pow`` and
    ``mul``: an fp16 tensor holding the fp16 value nearest 0.3 gets its
    gradient from 0.3, not from the rounded 0.300048828125.
    """
    shape = (64,)
    x = (torch.rand(shape, dtype=torch.float32, device=flag_gems.device) + 0.5).to(
        dtype
    )
    go = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    exp = 0.3

    ref_x = utils.to_reference(x.clone()).requires_grad_(True)
    out = torch.ops.aten._foreach_pow.ScalarList([ref_x], [exp])[0]
    (ref_grad,) = torch.autograd.grad(out, ref_x, utils.to_reference(go))

    res_grad = foreach_pow_scalar_list_backward([go], [x], [exp])[0]
    utils.gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.foreach_pow_backward
def test_foreach_pow_backward_nan_propagation():
    """Negative bases with fractional exponents propagate ATen's NaN/inf."""
    dtype = torch.float32
    self_list = [torch.tensor([-2.0, 0.0, 4.0], dtype=dtype, device=flag_gems.device)]
    exp_list = [torch.tensor([0.5, 2.0, 0.5], dtype=dtype, device=flag_gems.device)]
    grad_list = [torch.ones(3, dtype=dtype, device=flag_gems.device)]

    ref_self = [utils.to_reference(t).requires_grad_(True) for t in self_list]
    ref_exp = [utils.to_reference(t).requires_grad_(True) for t in exp_list]
    ref_grad = [utils.to_reference(t) for t in grad_list]
    ref_out = torch.ops.aten._foreach_pow.List(ref_self, ref_exp)
    grads = torch.autograd.grad(ref_out, ref_self + ref_exp, ref_grad + ref_grad)
    ref_gs, ref_ge = grads[:1], grads[1:]

    res_gs, res_ge = foreach_pow_backward(
        grad_list,
        self_list,
        exp_list,
        [r.detach().to(flag_gems.device) for r in ref_out],
    )
    utils.gems_assert_close(res_gs[0], ref_gs[0], dtype, equal_nan=True)
    utils.gems_assert_close(res_ge[0], ref_ge[0], dtype, equal_nan=True)
    # The forward at (-2)^0.5 is NaN and both gradients keep NaN there.
    assert torch.isnan(res_gs[0][0]) and torch.isnan(res_ge[0][0])
    # Zero base with a nonnegative exponent: the cond branch zeroes the
    # exponent gradient, and d(x^2)/dx = 2x zeroes the self gradient.
    assert res_gs[0][1].item() == 0.0 and res_ge[0][1].item() == 0.0
    assert res_gs[0][2].item() == pytest.approx(0.25, rel=1e-6)


@pytest.mark.foreach_pow_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_foreach_pow_backward_non_contiguous(dtype):
    """A transposed self with contiguous grads stages to dense and back."""
    shape = (48, 64)
    base = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.5
    x = base.t()
    exp = (torch.rand(shape, dtype=dtype, device=flag_gems.device) * 1.5 + 0.25).t()
    go = torch.randn(x.shape, dtype=dtype, device=flag_gems.device)

    # ``to_reference`` clones preserving layout, so the reference copies are
    # cloned *after* transposing to keep the same (64, 48) shape.
    ref_x = utils.to_reference(base.t()).requires_grad_(True)
    ref_e = utils.to_reference(exp).requires_grad_(True)
    ref_go = utils.to_reference(go)
    out = torch.ops.aten._foreach_pow.List([ref_x], [ref_e])
    ref_gs, ref_ge = torch.autograd.grad(out, [ref_x, ref_e], [ref_go, ref_go])

    res_gs, res_ge = foreach_pow_backward(
        [go], [x], [exp], [out[0].detach().to(flag_gems.device)]
    )
    utils.gems_assert_close(res_gs[0], ref_gs, dtype)
    utils.gems_assert_close(res_ge[0], ref_ge, dtype)
    assert res_gs[0].shape == x.shape


@pytest.mark.foreach_pow_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_foreach_pow_backward_empty(dtype):
    """Empty tensors produce empty gradients without launching."""
    self_list = [torch.empty(0, dtype=dtype, device=flag_gems.device)]
    exp_list = [torch.empty(0, dtype=dtype, device=flag_gems.device)]
    grad_list = [torch.empty(0, dtype=dtype, device=flag_gems.device)]
    res_gs, res_ge = foreach_pow_backward(grad_list, self_list, exp_list, grad_list)
    assert res_gs[0].shape == (0,)
    assert res_ge[0].shape == (0,)


@pytest.mark.foreach_pow_backward
@pytest.mark.parametrize(
    "shape", [(), (1,), (7,), (2, 3, 4)] if not utils.QUICK_MODE else [(2, 3, 4)]
)
def test_foreach_pow_backward_zero_dim(shape):
    """Zero-dimensional and small tensors take the same branch structure."""
    dtype = torch.float32
    self_list, exp_list, grad_list, ref_self, ref_exp, ref_grad = _draw_lists(
        [shape], dtype, flag_gems.device
    )
    ref_out = torch.ops.aten._foreach_pow.List(ref_self, ref_exp)
    ref_gs, ref_ge = torch.autograd.grad(
        ref_out, ref_self + ref_exp, ref_grad + ref_grad
    )
    res_gs, res_ge = foreach_pow_backward(
        grad_list, self_list, exp_list, [ref_out[0].detach().to(flag_gems.device)]
    )
    utils.gems_assert_close(res_gs[0], ref_gs, dtype)
    utils.gems_assert_close(res_ge[0], ref_ge, dtype)


@pytest.mark.foreach_pow_backward
def test_foreach_pow_backward_autograd_bridge():
    """``ForeachPowFunction`` pairs the FlagGems forward with the kernel."""
    dtype = torch.float32
    x = torch.rand((32, 48), dtype=dtype, device=flag_gems.device) + 0.5
    e = torch.rand((32, 48), dtype=dtype, device=flag_gems.device) * 1.5 + 0.25
    go = torch.randn((32, 48), dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x.clone()).requires_grad_(True)
    ref_e = utils.to_reference(e.clone()).requires_grad_(True)
    ref_go = utils.to_reference(go)
    ref_out = torch.ops.aten._foreach_pow.List([ref_x], [ref_e])[0]
    ref_gs, ref_ge = torch.autograd.grad(ref_out, [ref_x, ref_e], [ref_go, ref_go])

    res_x = x.clone().requires_grad_(True)
    res_e = e.clone().requires_grad_(True)
    res_out = ForeachPowFunction.apply(res_x, res_e)
    assert isinstance(res_out, tuple) and res_out[0].grad_fn is not None
    assert "ForeachPowFunction" in type(res_out[0].grad_fn).__name__
    res_out[0].backward(go)

    utils.gems_assert_close(res_x.grad, ref_gs, dtype)
    utils.gems_assert_close(res_e.grad, ref_ge, dtype)


@pytest.mark.foreach_pow_backward
def test_foreach_pow_backward_scalar_list_bridge():
    """``ForeachPowScalarListFunction`` returns a graph-connected tuple."""
    dtype = torch.float32
    x = torch.rand((32,), dtype=dtype, device=flag_gems.device) + 0.5
    go = torch.randn((32,), dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x.clone()).requires_grad_(True)
    ref_out = torch.ops.aten._foreach_pow.ScalarList([ref_x], [0.5])
    (ref_gs,) = torch.autograd.grad(ref_out, ref_x, utils.to_reference(go))

    res_x = x.clone().requires_grad_(True)
    res_out = ForeachPowScalarListFunction.apply(res_x, 0.5)
    assert isinstance(res_out, tuple) and res_out[0].grad_fn is not None
    res_out[0].backward(go)
    utils.gems_assert_close(res_x.grad, ref_gs, dtype)


@pytest.mark.foreach_pow_backward
def test_foreach_pow_backward_length_mismatch_rejected():
    """Mismatched list lengths fail the way ATen fails."""
    x = [torch.rand(4, device=flag_gems.device) + 0.5 for _ in range(3)]
    e = [torch.rand(4, device=flag_gems.device) + 0.5 for _ in range(2)]
    g = [torch.randn(4, device=flag_gems.device) for _ in range(3)]
    with pytest.raises(RuntimeError):
        foreach_pow_backward(g, x, e, g)


@pytest.mark.foreach_pow_backward
def test_foreach_pow_backward_rejects_non_float():
    """Integer tensors cannot carry a gradient; complex is refused by ATen."""
    x = torch.ones(2, dtype=torch.int32, device=flag_gems.device)
    e = torch.ones(2, dtype=torch.int32, device=flag_gems.device)
    g = torch.ones(2, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        foreach_pow_backward([g], [x], [e], [g])
