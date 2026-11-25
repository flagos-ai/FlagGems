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
from flag_gems.ops.foreach_div_backward import foreach_div_backward

from . import accuracy_utils as utils


def _lists(shapes, dtype, device, requires_grad=False):
    """Well-conditioned paired lists: ``other`` stays away from zero so the
    fp16/bf16 intermediates keep finite, which exercises the kernel's rounding
    rather than its overflow behaviour."""
    self_list = [torch.rand(s, dtype=dtype, device=device) + 0.5 for s in shapes]
    other_list = [torch.rand(s, dtype=dtype, device=device) + 1.5 for s in shapes]
    if requires_grad:
        self_list = [t.requires_grad_(True) for t in self_list]
        other_list = [t.requires_grad_(True) for t in other_list]
    return self_list, other_list


def _ref_lists(shapes, dtype, device, requires_grad=False):
    self_list, other_list = _lists(shapes, dtype, device)
    self_list = [utils.to_reference(t).requires_grad_(True) for t in self_list]
    other_list = [utils.to_reference(t).requires_grad_(True) for t in other_list]
    return self_list, other_list


def _reference(self_list, other_list, grad_list):
    """Native ATen autograd: the same oracle ForeachDivBackward0 drives.

    The inputs are interleaved ``(self0, other0, self1, other1, ...)``, so the
    gradient list comes back in the same order; even and odd slices rebuild
    the two per-position gradient lists.
    """
    out = torch._foreach_div(self_list, other_list)
    inputs = [t for pair in zip(self_list, other_list) for t in pair]
    grads = torch.autograd.grad(out, inputs, grad_list)
    return grads[::2], grads[1::2]


@pytest.mark.foreach_div_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    "shapes",
    [
        [(2, 19, 7)],
        [(64,)],
        [(8, 16)],
        [()],
        [(2, 3), (2, 3)],
    ],
)
def test_foreach_div_backward_lists(shapes, dtype):
    """Paired lists of varying length, shape and rank.

    A 0-dim position is covered alone: mixing a 0-dim tensor with shaped ones
    in one list trips an autograd engine quirk in this Torch build (the
    0-dim operand's gradient comes back at the *next* position's shape), so
    there is no stable ATen oracle to compare a mixed case against.
    """
    device = flag_gems.device
    # The operator always runs on its own device; only the reference moves
    # (``to_reference`` relocates to CPU under ``--ref cpu``).
    self_list, other_list = _lists(shapes, dtype, device, requires_grad=True)
    ref_self_in = [
        utils.to_reference(t.detach().clone()).requires_grad_(True) for t in self_list
    ]
    ref_other_in = [
        utils.to_reference(t.detach().clone()).requires_grad_(True) for t in other_list
    ]
    grad_list = [torch.randn(s, dtype=dtype, device=device) for s in shapes]
    ref_grads = [utils.to_reference(g.clone()) for g in grad_list]

    out = torch._foreach_div(ref_self_in, ref_other_in)
    inputs = [t for pair in zip(ref_self_in, ref_other_in) for t in pair]
    ref = torch.autograd.grad(out, inputs, ref_grads)
    ref_self, ref_other = ref[::2], ref[1::2]

    res_self, res_other = foreach_div_backward(
        grad_list, [t.detach() for t in self_list], [t.detach() for t in other_list]
    )
    for got, want in zip(res_self + res_other, ref_self + ref_other):
        utils.gems_assert_close(got, want, dtype)


@pytest.mark.foreach_div_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_foreach_div_backward_single(dtype):
    """The minimal one-position list, the optimizer's common shape."""
    device = flag_gems.device
    self_list, other_list = _lists([(2, 19, 7)], dtype, device, requires_grad=True)
    grad = torch.randn((2, 19, 7), dtype=dtype, device=device)

    ref_self_in = utils.to_reference(self_list[0].detach().clone()).requires_grad_(True)
    ref_other_in = utils.to_reference(other_list[0].detach().clone()).requires_grad_(
        True
    )
    ref_grad = utils.to_reference(grad.clone())
    ref_gs, ref_go = torch.autograd.grad(
        torch._foreach_div([ref_self_in], [ref_other_in]),
        [ref_self_in, ref_other_in],
        [ref_grad],
    )

    res_self, res_other = foreach_div_backward(
        [grad], [t.detach() for t in self_list], [t.detach() for t in other_list]
    )
    utils.gems_assert_close(res_self[0], ref_gs, dtype)
    utils.gems_assert_close(res_other[0], ref_go, dtype)


@pytest.mark.foreach_div_backward
def test_foreach_div_backward_bit_exact_fp32():
    """The fp32 chain is bit-exact: every division rounds to nearest."""
    device = flag_gems.device
    a = torch.randn((64, 64), device=device, requires_grad=True)
    b = torch.randn((64, 64), device=device, requires_grad=True)
    out = torch._foreach_div([a], [b])
    grad = torch.randn((64, 64), device=device)
    ref_gs, ref_go = torch.autograd.grad(out, [a, b], [grad])
    gs, go = foreach_div_backward([grad], [a.detach()], [b.detach()])
    assert torch.equal(gs[0], ref_gs)
    assert torch.equal(go[0], ref_go)


@pytest.mark.foreach_div_backward
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_foreach_div_backward_per_op_rounding(dtype):
    """fp16/bf16: each elementwise step materialises in the storage dtype, as
    ATen's per-op opmath evaluation does."""
    device = flag_gems.device
    a = torch.randn((4096,), device=device, dtype=dtype, requires_grad=True)
    b = torch.randn((4096,), device=device, dtype=dtype, requires_grad=True)
    out = torch._foreach_div([a], [b])
    grad = torch.randn((4096,), device=device, dtype=dtype)
    ref_gs, ref_go = torch.autograd.grad(out, [a, b], [grad])
    gs, go = foreach_div_backward([grad], [a.detach()], [b.detach()])
    assert torch.equal(gs[0], ref_gs)
    assert torch.equal(go[0], ref_go)


@pytest.mark.foreach_div_backward
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_foreach_div_backward_complex(dtype):
    """Complex lists: grad_self = grad / conj(other) and
    grad_other = -grad * ((self/other)/other).conj(), compared at the
    framework tolerance (the c10 division is matched to a few fp32 ulp)."""
    device = flag_gems.device
    a = torch.randn((64,), device=device, dtype=dtype, requires_grad=True)
    b = torch.randn((64,), device=device, dtype=dtype, requires_grad=True)
    out = torch._foreach_div([a], [b])
    grad = torch.randn((64,), device=device, dtype=dtype)
    ref_gs, ref_go = torch.autograd.grad(out, [a, b], [grad])
    gs, go = foreach_div_backward([grad], [a.detach()], [b.detach()])
    utils.gems_assert_close(gs[0], utils.to_reference(ref_gs), dtype)
    utils.gems_assert_close(go[0], utils.to_reference(ref_go), dtype)


@pytest.mark.foreach_div_backward
def test_foreach_div_backward_mixed_dtypes():
    """A list mixing fp32 and fp16 positions groups into two launches."""
    device = flag_gems.device
    a0 = torch.randn((64,), device=device, requires_grad=True)
    a1 = torch.randn((64,), device=device, dtype=torch.float16, requires_grad=True)
    b0 = torch.randn((64,), device=device, requires_grad=True)
    b1 = torch.randn((64,), device=device, dtype=torch.float16, requires_grad=True)
    out = torch._foreach_div([a0, a1], [b0, b1])
    g0 = torch.randn((64,), device=device)
    g1 = torch.randn((64,), device=device, dtype=torch.float16)
    ones0 = torch.ones((64,), device=device)
    ones1 = torch.ones((64,), device=device, dtype=torch.float16)
    ref = torch.autograd.grad(out, [a0, a1, b0, b1], [g0, g1, ones0, ones1])
    gs, go = foreach_div_backward(
        [g0, g1], [a0.detach(), a1.detach()], [b0.detach(), b1.detach()]
    )
    assert torch.equal(gs[0], ref[0])
    assert torch.equal(gs[1], ref[1])
    assert torch.equal(go[0], ref[2])
    assert torch.equal(go[1], ref[3])


@pytest.mark.foreach_div_backward
def test_foreach_div_backward_non_contiguous():
    """Transposed operands of matching shape are staged to dense form."""
    device = flag_gems.device
    a = (torch.rand((64, 48), device=device) + 0.5).t().requires_grad_(True)
    b = (torch.rand((64, 48), device=device) + 1.5).t().requires_grad_(True)
    assert not a.is_contiguous() and not b.is_contiguous()
    out = torch._foreach_div([a], [b])
    grad = torch.randn((48, 64), device=device)
    ref = torch.autograd.grad(out, [a, b], [grad])
    gs, go = foreach_div_backward([grad], [a.detach()], [b.detach()])
    assert torch.equal(gs[0], ref[0])
    assert torch.equal(go[0], ref[1])


@pytest.mark.foreach_div_backward
def test_foreach_div_backward_zero_denominator():
    """Zero denominators propagate inf/nan exactly as ATen does."""
    device = flag_gems.device
    a = torch.randn((16,), device=device, requires_grad=True)
    b = torch.zeros((16,), device=device, requires_grad=True)
    out = torch._foreach_div([a], [b])
    grad = torch.randn((16,), device=device)
    ref_gs, ref_go = torch.autograd.grad(out, [a, b], [grad])
    gs, go = foreach_div_backward([grad], [a.detach()], [b.detach()])
    assert torch.equal(torch.isinf(gs[0]), torch.isinf(ref_gs))
    assert torch.equal(torch.isnan(go[0]), torch.isnan(ref_go))


@pytest.mark.foreach_div_backward
def test_foreach_div_backward_sixteen():
    """A sixteen-position list, the shape the optimizer benchmark targets."""
    device = flag_gems.device
    n = 16
    self_list = [
        torch.randn((32, 32), device=device, requires_grad=True) for _ in range(n)
    ]
    other_list = [
        torch.randn((32, 32), device=device, requires_grad=True) for _ in range(n)
    ]
    out = torch._foreach_div(self_list, other_list)
    grads = [torch.randn((32, 32), device=device) for _ in range(n)]
    inputs = [t for pair in zip(self_list, other_list) for t in pair]
    ref = torch.autograd.grad(out, inputs, grads)
    gs, go = foreach_div_backward(
        grads, [t.detach() for t in self_list], [t.detach() for t in other_list]
    )
    for i in range(n):
        assert torch.equal(gs[i], ref[2 * i])
        assert torch.equal(go[i], ref[2 * i + 1])


@pytest.mark.foreach_div_backward
def test_foreach_div_backward_empty_and_errors():
    """Empty tensors pass through; list-length and dtype refusals match ATen."""
    device = flag_gems.device
    gs, go = foreach_div_backward(
        [torch.empty(0, device=device)],
        [torch.empty(0, device=device)],
        [torch.empty(0, device=device)],
    )
    assert gs[0].shape == (0,) and go[0].shape == (0,)

    with pytest.raises(RuntimeError):
        foreach_div_backward(
            [torch.randn(4, device=device)],
            [torch.randn(4, device=device), torch.randn(4, device=device)],
            [torch.randn(4, device=device)],
        )
    with pytest.raises(RuntimeError):
        foreach_div_backward(
            [torch.randn(4, device=device)],
            [torch.ones(4, device=device, dtype=torch.int32)],
            [torch.randn(4, device=device)],
        )


@pytest.mark.foreach_div_backward
def test_foreach_div_backward_dispatch_entry():
    """``flag_gems.foreach_div_backward`` is exported from the package root."""
    assert hasattr(flag_gems, "foreach_div_backward")
    assert flag_gems.foreach_div_backward is foreach_div_backward
