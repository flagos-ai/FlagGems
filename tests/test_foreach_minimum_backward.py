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

"""Correctness tests for ``foreach_minimum_backward``.

``torch._foreach_minimum`` with a scalar (or scalar-list) operand has no
``aten::_foreach_minimum_backward`` schema: ATen derives the input gradient
from the autograd formula ``where(self == s, grads / 2, grads)`` masked to
zero where ``self > s``. The Triton kernel reproduces that formula with ATen's
rounding boundaries (every step in the storage dtype), so on-device results
are bit-exact against ``torch.autograd.grad`` and the comparisons can be
equality-based rather than tolerance-based.

Both the ``.Scalar`` (one broadcast scalar) and ``.ScalarList`` (one scalar
per tensor) forms are covered, including exact ties -- the scalar is rounded
to the storage dtype before comparing, which is what makes a value such as
``fp16(0.35)`` a tie rather than an "above".
"""

import pytest
import torch

import flag_gems
from flag_gems.ops.foreach_minimum_backward import (
    ForeachMinimumFunction,
    foreach_minimum_backward,
)

from . import accuracy_utils as utils

# Scalars chosen to exercise every region: strictly-below, exactly-at (after
# the scalar is rounded to the storage dtype) and strictly-above elements, plus
# a value whose fp16/bf16 rounding creates ties that a full-precision compare
# would miss.
TIE_SCALARS = [0.35, 0.5, -1.0, 0.0, 2.0]


def _aten_reference(grad_output, self_t, scalar, scalars):
    """The derivatives.yaml formula, one output per tensor."""
    outs = []
    for i, (g, t) in enumerate(zip(grad_output, self_t)):
        s = scalar if scalar is not None else scalars[i]
        r = torch.where(t == s, g / 2, g)
        r = torch.where(t > s, torch.zeros_like(r), r)
        outs.append(r.to(t.dtype))
    return outs


def _aten_grads(grad_output, self_t, scalar, scalars):
    """Gradients from the native ATen autograd chain on the reference device.

    ``utils.to_reference`` moves the tensors to the reference device (CPU under
    ``--ref cpu``), where the native chain runs; under the default device
    reference this is the same device the kernel runs on, so the comparison is
    bit-exact on-device there.
    """
    ref_self = [utils.to_reference(t) for t in self_t]
    ref_grads_in = [utils.to_reference(g) for g in grad_output]
    leaves = [t.detach().clone().requires_grad_(True) for t in ref_self]
    if scalar is not None:
        out = torch.ops.aten._foreach_minimum.Scalar(leaves, scalar)
    else:
        out = torch.ops.aten._foreach_minimum.ScalarList(leaves, scalars)
    return torch.autograd.grad(out, leaves, ref_grads_in)


def _assert_lists_equal(res, ref):
    assert len(res) == len(ref)
    for got, want in zip(res, ref):
        assert got.dtype == want.dtype
        assert got.shape == want.shape
        utils.gems_assert_equal(got, want)


@pytest.mark.foreach_minimum_backward
@pytest.mark.parametrize("scalar", TIE_SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_foreach_minimum_backward_scalar(dtype, scalar):
    """The ``.Scalar`` form: one broadcast scalar, bit-exact vs ATen autograd."""
    torch.manual_seed(0)
    shapes = [(64, 64), (128,), (1,)]
    self_t = [
        torch.randn(shape, dtype=dtype, device=flag_gems.device) for shape in shapes
    ]
    grad_output = [torch.randn_like(t) for t in self_t]

    ref_grads = _aten_grads(grad_output, self_t, scalar, None)
    res_grads = foreach_minimum_backward(grad_output, self_t, scalar=scalar)
    _assert_lists_equal(res_grads, ref_grads)


@pytest.mark.foreach_minimum_backward
@pytest.mark.parametrize("scalar", TIE_SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_foreach_minimum_backward_scalar_list(dtype, scalar):
    """The ``.ScalarList`` form: one scalar per tensor, bit-exact vs ATen."""
    torch.manual_seed(0)
    shapes = [(64, 64), (128,), (1,)]
    self_t = [
        torch.randn(shape, dtype=dtype, device=flag_gems.device) for shape in shapes
    ]
    grad_output = [torch.randn_like(t) for t in self_t]
    scalars = [scalar + i for i in range(len(self_t))]

    ref_grads = _aten_grads(grad_output, self_t, None, scalars)
    res_grads = foreach_minimum_backward(grad_output, self_t, scalars=scalars)
    _assert_lists_equal(res_grads, ref_grads)


@pytest.mark.foreach_minimum_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_foreach_minimum_backward_three_region(dtype):
    """Below passes through, ties halve, above zeroes -- the exact subgradient."""
    s = 1.0
    self_t = [
        torch.tensor(
            [0.0, 0.5, 1.0, 2.0, -1.0, s], dtype=dtype, device=flag_gems.device
        )
    ]
    grad_output = [torch.ones_like(t) for t in self_t]

    res = foreach_minimum_backward(grad_output, self_t, scalar=s)[0]
    expected = utils.to_reference(
        torch.tensor([1.0, 1.0, 0.5, 0.0, 1.0, 0.5], dtype=dtype)
    )
    utils.gems_assert_equal(res, expected)


@pytest.mark.foreach_minimum_backward
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_foreach_minimum_backward_tie_rounding(dtype):
    """The scalar is rounded to the storage dtype before comparing.

    ``fp16(0.35)`` reads back as 0.35009765625; an element holding exactly that
    value must be a tie (grad / 2), which a full-precision compare would
    misclassify as "above" (grad 0).
    """
    torch.manual_seed(0)
    s = 0.35
    rounded = torch.tensor(s, dtype=dtype, device=flag_gems.device).item()
    self_t = [
        torch.tensor(
            [
                rounded,
                torch.nextafter(
                    torch.tensor(rounded, dtype=dtype), torch.tensor(2.0, dtype=dtype)
                ).item(),
                -1.0,
            ],
            dtype=dtype,
            device=flag_gems.device,
        )
    ]
    grad_output = [torch.ones_like(t) for t in self_t]

    ref_grads = _aten_grads(grad_output, self_t, s, None)
    res_grads = foreach_minimum_backward(grad_output, self_t, scalar=s)
    _assert_lists_equal(res_grads, ref_grads)
    # The first element must actually be a tie, or the test proves nothing.
    assert res_grads[0][0].item() == pytest.approx(0.5, abs=1e-3)


@pytest.mark.foreach_minimum_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_foreach_minimum_backward_nan_inf(dtype):
    """NaN passes the gradient through, +inf is zeroed, -inf passes through.

    Every comparison against NaN is false, so the autograd formula keeps the
    upstream gradient at a NaN element (the forward output is NaN there and
    ATen makes no special case). ``+inf > s`` zeroes; ``-inf < s`` passes.
    """
    self_t = [
        torch.tensor(
            [float("nan"), float("inf"), float("-inf"), -2.0, 3.0],
            dtype=dtype,
            device=flag_gems.device,
        )
    ]
    grad_output = [torch.ones_like(t) for t in self_t]
    s = 1.0

    ref_grads = _aten_grads(grad_output, self_t, s, None)
    res_grads = foreach_minimum_backward(grad_output, self_t, scalar=s)
    _assert_lists_equal(res_grads, ref_grads)

    res_cpu = res_grads[0].cpu()
    assert res_cpu[0].item() == 1.0  # NaN: comparisons false, grad passes
    assert res_cpu[1].item() == 0.0  # inf > s -> zeroed
    assert res_cpu[2].item() == 1.0  # -inf < s -> passes through


@pytest.mark.foreach_minimum_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_foreach_minimum_backward_non_contiguous(dtype):
    """Transposed inputs and strided gradients produce the same values."""
    torch.manual_seed(0)
    base = torch.randn((64, 48), dtype=dtype, device=flag_gems.device)
    self_t = [base.t(), base]
    assert not self_t[0].is_contiguous()
    grad_output = [torch.randn_like(t) for t in self_t]
    s = 0.25

    ref_grads = _aten_grads(grad_output, self_t, s, None)
    res_grads = foreach_minimum_backward(grad_output, self_t, scalar=s)
    _assert_lists_equal(res_grads, ref_grads)


@pytest.mark.foreach_minimum_backward
def test_foreach_minimum_backward_mixed_dtypes():
    """One list holding fp16 and fp32 tensors (ATen accepts mixed lists)."""
    torch.manual_seed(0)
    self_t = [
        torch.randn((128,), dtype=torch.float16, device=flag_gems.device),
        torch.randn((128,), dtype=torch.float32, device=flag_gems.device),
    ]
    grad_output = [torch.randn_like(t) for t in self_t]
    scalars = [0.5, 0.5]

    ref_grads = _aten_grads(grad_output, self_t, None, scalars)
    res_grads = foreach_minimum_backward(grad_output, self_t, scalars=scalars)
    _assert_lists_equal(res_grads, ref_grads)


@pytest.mark.foreach_minimum_backward
def test_foreach_minimum_backward_empty():
    """An empty tensor in the list passes through untouched."""
    self_t = [
        torch.empty((0,), dtype=torch.float32, device=flag_gems.device),
        torch.randn((16,), dtype=torch.float32, device=flag_gems.device),
    ]
    grad_output = [torch.randn_like(t) for t in self_t]

    ref_grads = _aten_grads(grad_output, self_t, 0.5, None)
    res_grads = foreach_minimum_backward(grad_output, self_t, scalar=0.5)
    _assert_lists_equal(res_grads, ref_grads)


@pytest.mark.foreach_minimum_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_foreach_minimum_backward_bridge(dtype):
    """The autograd bridge pairs the FlagGems forward with this backward."""
    torch.manual_seed(0)
    self_t = [
        torch.randn((256,), dtype=dtype, device=flag_gems.device).requires_grad_(True)
        for _ in range(4)
    ]
    for scalar, scalars in ((0.5, None), (None, [0.25, -1.0, 2.0, 0.5])):
        out = ForeachMinimumFunction.apply(*self_t, scalar, scalars)
        assert all(o.requires_grad for o in out)
        grad_output = [torch.randn_like(o) for o in out]
        res_grads = torch.autograd.grad(out, self_t, grad_output)

        ref_grads = _aten_grads(grad_output, self_t, scalar, scalars)
        _assert_lists_equal(list(res_grads), ref_grads)


@pytest.mark.foreach_minimum_backward
def test_foreach_minimum_backward_validation():
    """Argument validation: exactly one of scalar / scalars, equal lengths."""
    t = torch.randn((8,), device=flag_gems.device)
    g = torch.randn_like(t)
    with pytest.raises(TypeError):
        foreach_minimum_backward([g], [t])  # neither scalar nor scalars
    with pytest.raises(TypeError):
        foreach_minimum_backward([g], [t], scalar=0.5, scalars=[0.5])
    with pytest.raises(ValueError):
        foreach_minimum_backward([g, g], [t], scalar=0.5)
    with pytest.raises(ValueError):
        foreach_minimum_backward([g], [t], scalars=[0.5, 0.5])
