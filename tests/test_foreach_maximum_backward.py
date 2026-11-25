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
from flag_gems.ops.foreach_maximum_backward import (
    ForeachMaximumFunction,
    foreach_maximum_backward,
)

from . import accuracy_utils as utils


def _ref_grad(shape, dtype, scalar, upcast=False):
    """ATen autograd reference for ``_foreach_maximum.Scalar``.

    The Triton kernel is bit-exact with native ATen autograd, so a same-dtype
    reference is the sharpest oracle: the derivative formula never leaves the
    tensor dtype (``grad / 2`` is exact in every float dtype), and an upcast
    reference would disagree with any correct low-precision implementation at
    the tie boundary.
    """
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 2.0
    go = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x.clone(), upcast).requires_grad_(True)
    ref_go = utils.to_reference(go, upcast)
    out = torch.ops.aten._foreach_maximum.Scalar([ref_x], scalar)[0]
    (grad,) = torch.autograd.grad(out, ref_x, ref_go)
    return x, go, grad


@pytest.mark.foreach_maximum_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_foreach_maximum_backward(shape, dtype):
    """The explicit entry reproduces ATen's ``ForeachMaximumBackward0``.

    Positions below the scalar receive 0, the tie position receives half the
    upstream gradient, positions above receive it unchanged -- all compared
    bit-exactly against native autograd.
    """
    x, go, ref_grad = _ref_grad(shape, dtype, 0.5)
    res_grad = foreach_maximum_backward([go], [x], 0.5)
    utils.gems_assert_close(res_grad[0], ref_grad, dtype)
    assert res_grad[0].shape == x.shape
    assert res_grad[0].dtype == dtype


@pytest.mark.foreach_maximum_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_foreach_maximum_backward_functional(dtype):
    """The entry point accepts the ``.ScalarList`` form (one scalar per tensor)."""
    xs = [
        torch.randn((32, 48), dtype=dtype, device=flag_gems.device) * 2.0
        for _ in range(3)
    ]
    gos = [
        torch.randn((32, 48), dtype=dtype, device=flag_gems.device) for _ in range(3)
    ]
    scalars = [0.5, -1.0, 2.0]

    ref_grads = []
    for x, go, s in zip(xs, gos, scalars):
        ref_x = utils.to_reference(x.clone()).requires_grad_(True)
        ref_go = utils.to_reference(go)
        out = torch.ops.aten._foreach_maximum.Scalar([ref_x], s)[0]
        (grad,) = torch.autograd.grad(out, ref_x, ref_go)
        ref_grads.append(grad)

    res_grads = foreach_maximum_backward(gos, xs, scalars)
    for got, want, s in zip(res_grads, ref_grads, scalars):
        utils.gems_assert_close(got, want, dtype)
        assert got.shape == (32, 48)


@pytest.mark.foreach_maximum_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_foreach_maximum_backward_tie(dtype):
    """The tie position takes half the gradient; ATen compares in tensor dtype.

    A Python ``0.3`` against an fp16 tensor compares equal to the fp16 value
    nearest to 0.3 -- the kernel must cast the scalar to the tensor dtype, not
    compare in fp32, or that position would receive the full gradient.
    """
    if dtype is torch.bfloat16:
        val = torch.tensor(0.3, dtype=torch.bfloat16).item()
    else:
        val = (
            torch.tensor(0.3, dtype=torch.float16).item()
            if dtype is torch.float16
            else 0.3
        )
    x = torch.tensor([val], dtype=dtype, device=flag_gems.device)
    go = torch.ones(1, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x.clone()).requires_grad_(True)
    out = torch.ops.aten._foreach_maximum.Scalar([ref_x], 0.3)[0]
    (ref_grad,) = torch.autograd.grad(out, ref_x, utils.to_reference(go))

    res_grad = foreach_maximum_backward([go], [x], 0.3)
    utils.gems_assert_close(res_grad[0], ref_grad, dtype)
    # Tie: exactly half, bit-exactly.
    assert res_grad[0].item() == 0.5


@pytest.mark.foreach_maximum_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_foreach_maximum_backward_non_contiguous(dtype):
    """A transposed input with a contiguous upstream gradient."""
    base = torch.randn((48, 64), dtype=dtype, device=flag_gems.device) * 2.0
    x = base.t()
    go = torch.randn(x.shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(base).t().requires_grad_(True)
    ref_go = utils.to_reference(go)
    out = torch.ops.aten._foreach_maximum.Scalar([ref_x], 0.5)[0]
    (ref_grad,) = torch.autograd.grad(out, ref_x, ref_go)

    res_grad = foreach_maximum_backward([go], [x], 0.5)
    utils.gems_assert_close(res_grad[0], ref_grad, dtype)
    assert res_grad[0].shape == x.shape


@pytest.mark.foreach_maximum_backward
def test_foreach_maximum_backward_inf_nan():
    """Mask overwrites non-finite gradients below the scalar; ties pass through.

    ``masked_fill_`` runs after the ``where``, so an ``inf`` gradient at a
    masked position becomes 0, while a ``nan`` gradient at or above the tie is
    propagated unchanged.  A NaN scalar never compares equal or less, so its
    gradients all pass through.
    """
    go = torch.tensor(
        [float("inf"), float("nan"), 1.0, -1.0],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    x = torch.tensor([1.0, 5.0, 0.5, 2.0], dtype=torch.float32, device=flag_gems.device)

    ref_x = utils.to_reference(x.clone()).requires_grad_(True)
    ref_go = utils.to_reference(go)
    out = torch.ops.aten._foreach_maximum.Scalar([ref_x], 2.0)[0]
    (ref_grad,) = torch.autograd.grad(out, ref_x, ref_go)

    res_grad = foreach_maximum_backward([go], [x], 2.0)
    utils.gems_assert_close(res_grad[0], ref_grad, torch.float32, equal_nan=True)
    got = res_grad[0].tolist()
    # x < 2 at positions 0 and 2 -> masked (the inf is overwritten too);
    # x == 2 at position 3 -> halved; x > 2 at position 1 -> nan passes through.
    assert got[0] == 0.0 and got[2] == 0.0 and got[3] == -0.5
    assert torch.isnan(res_grad[0][1])


@pytest.mark.foreach_maximum_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_foreach_maximum_backward_empty(dtype):
    """An empty tensor produces an empty gradient without launching."""
    x = torch.empty(0, dtype=dtype, device=flag_gems.device)
    go = torch.empty(0, dtype=dtype, device=flag_gems.device)
    res_grad = foreach_maximum_backward([go], [x], 0.5)
    assert res_grad[0].shape == (0,)


@pytest.mark.foreach_maximum_backward
@pytest.mark.parametrize(
    "shape", [(), (1,), (7,), (2, 3, 4)] if not utils.QUICK_MODE else [(2, 3, 4)]
)
def test_foreach_maximum_backward_zero_dim(shape):
    """Zero-dimensional and small tensors take the same branch structure."""
    dtype = torch.float32
    x, go, ref_grad = _ref_grad(shape, dtype, 0.5)
    res_grad = foreach_maximum_backward([go], [x], 0.5)
    utils.gems_assert_close(res_grad[0], ref_grad, dtype)


@pytest.mark.foreach_maximum_backward
def test_foreach_maximum_backward_autograd_bridge():
    """``ForeachMaximumFunction`` pairs the FlagGems forward with the kernel."""
    dtype = torch.float32
    x = torch.randn((32, 48), dtype=dtype, device=flag_gems.device) * 2.0
    go = torch.randn((32, 48), dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x.clone()).requires_grad_(True)
    ref_go = utils.to_reference(go)
    ref_out = torch.ops.aten._foreach_maximum.Scalar([ref_x], 0.5)[0]
    (ref_grad,) = torch.autograd.grad(ref_out, ref_x, ref_go)

    res_x = x.clone().requires_grad_(True)
    res_out = ForeachMaximumFunction.apply(res_x, 0.5)
    assert res_out.grad_fn is not None
    assert "ForeachMaximumFunction" in type(res_out.grad_fn).__name__
    res_out.backward(go)

    utils.gems_assert_close(res_x.grad, ref_grad, dtype)


@pytest.mark.foreach_maximum_backward
def test_foreach_maximum_backward_length_mismatch_rejected():
    """A scalar list shorter than the tensor list fails the way ATen fails."""
    xs = [torch.randn(4, device=flag_gems.device) for _ in range(3)]
    with pytest.raises(RuntimeError):
        foreach_maximum_backward(
            [torch.randn(4, device=flag_gems.device) for _ in range(3)], xs, [0.5, 1.0]
        )


@pytest.mark.foreach_maximum_backward
def test_foreach_maximum_backward_rejects_non_float():
    """Integer tensors cannot carry a gradient; complex is refused by ATen."""
    x = torch.ones(2, dtype=torch.int32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        foreach_maximum_backward([torch.ones(2, device=flag_gems.device)], [x], 1.0)
