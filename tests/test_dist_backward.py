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

from . import accuracy_utils as utils

# One case per code path: equal-shape contiguous (Triton kernel), zero-size
# and the +/-inf global tie path (ties inside one block and across blocks).
DIST_BACKWARD_CASES = [
    ((64,), (64,), 2),
    ((8, 32), (8, 32), 1),
    ((8, 32), (8, 32), 3),
    ((8, 32), (8, 32), 0.5),
    ((8, 32), (8, 32), 4.0),
    ((8, 32), (8, 32), float("inf")),
    ((8, 32), (8, 32), float("-inf")),
    ((8, 32), (8, 32), -1.0),
    ((), (), 2),
]

P_LIST = [1, 1.5, 2, 3, 0.5, 4.0, float("inf"), float("-inf"), -1.0]


def _dist_backward_reference(grad, inp, other, p, dist):
    """float64 reference: the ATen autograd chain of ``torch.dist``."""
    ref_inp = utils.to_reference(inp, True).requires_grad_(True)
    ref_other = utils.to_reference(other, True).requires_grad_(True)
    ref_grad = utils.to_reference(grad, True)
    return torch.autograd.grad(
        torch.dist(ref_inp, ref_other, p), [ref_inp, ref_other], grad_outputs=ref_grad
    )


@pytest.mark.dist_backward
@pytest.mark.parametrize("shape, other_shape, p", DIST_BACKWARD_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_dist_backward(shape, other_shape, p, dtype):
    """Gradient of ``dist``: the signed p-norm derivative at both inputs."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other = torch.randn(other_shape, dtype=dtype, device=flag_gems.device)
    grad = torch.randn((), dtype=dtype, device=flag_gems.device)
    dist = torch.dist(inp, other, p).detach()

    ref_grads = _dist_backward_reference(grad, inp, other, p, dist)
    res_grads = flag_gems.dist_backward(grad, inp, other, p, dist)

    for ref, res in zip(ref_grads, res_grads):
        utils.gems_assert_close(res, ref.to(dtype), dtype)


@pytest.mark.dist_backward
@pytest.mark.parametrize("p", P_LIST)
def test_dist_backward_p_values(p, dtype=torch.float64):
    """Every p the forward supports has a defined gradient (p = 0 excepted)."""
    inp = torch.randn(64, dtype=dtype, device=flag_gems.device)
    other = torch.randn(64, dtype=dtype, device=flag_gems.device)
    grad = torch.randn((), dtype=dtype, device=flag_gems.device)
    dist = torch.dist(inp, other, p).detach()

    ref_grads = _dist_backward_reference(grad, inp, other, p, dist)
    res_grads = flag_gems.dist_backward(grad, inp, other, p, dist)

    for ref, res in zip(ref_grads, res_grads):
        utils.gems_assert_close(res, ref.to(dtype), dtype)


@pytest.mark.dist_backward
@pytest.mark.parametrize("p", [0, 1, 2, 3, 0.5, float("inf")])
def test_dist_backward_zero_diff(p, dtype=torch.float64):
    """Identical inputs: dist == 0 and the gradient collapses to zero."""
    inp = torch.randn(16, dtype=dtype, device=flag_gems.device)
    other = inp.clone()
    grad = torch.randn((), dtype=dtype, device=flag_gems.device)
    dist = torch.dist(inp, other, p).detach()

    if p == 0:
        with pytest.raises(RuntimeError):
            torch.autograd.grad(
                torch.dist(inp, other, p), [inp, other], grad_outputs=grad
            )
        return

    ref_grads = _dist_backward_reference(grad, inp, other, p, dist)
    res_grads = flag_gems.dist_backward(grad, inp, other, p, dist)

    for ref, res in zip(ref_grads, res_grads):
        utils.gems_assert_equal(res, ref.to(dtype))


@pytest.mark.dist_backward
def test_dist_backward_use_gems_entry(dtype=torch.float32):
    """``flag_gems.dist_backward`` agrees with the public ``torch.dist`` chain.

    ``aten::dist`` is a CompositeExplicitAutograd op, so the forward runs on
    the Triton kernel while ``DistBackward0`` stays above it. This pins both
    paths to the same gradients and exercises the explicit
    ``flag_gems.dist_backward`` entry.
    """
    inp = torch.randn((8, 32), dtype=dtype, device=flag_gems.device)
    other = torch.randn((8, 32), dtype=dtype, device=flag_gems.device)
    grad = torch.randn((), dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp.detach(), True).requires_grad_(True)
    ref_other = utils.to_reference(other.detach(), True).requires_grad_(True)
    ref_grad = utils.to_reference(grad, True)
    ref_self, ref_other_grad = torch.autograd.grad(
        torch.dist(ref_inp, ref_other, 2), [ref_inp, ref_other], grad_outputs=ref_grad
    )

    res_inp = inp.detach().requires_grad_(True)
    res_other = other.detach().requires_grad_(True)
    res_self, res_other_grad = torch.autograd.grad(
        torch.dist(res_inp, res_other, 2), [res_inp, res_other], grad_outputs=grad
    )
    dist = torch.dist(inp, other, 2).detach()
    entry_self, entry_other = flag_gems.dist_backward(grad, inp, other, 2, dist)

    for got, expected in (
        (res_self, ref_self),
        (res_other_grad, ref_other_grad),
        (entry_self, ref_self),
        (entry_other, ref_other_grad),
    ):
        utils.gems_assert_close(got, expected.to(dtype), dtype)


@pytest.mark.dist_backward
def test_dist_backward_not_differentiable_at_p_zero(dtype=torch.float64):
    """p = 0 is a counting norm: no gradient, matching the ATen error."""
    inp = torch.randn(16, dtype=dtype, device=flag_gems.device)
    other = torch.randn(16, dtype=dtype, device=flag_gems.device)
    grad = torch.randn((), dtype=dtype, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        torch.autograd.grad(torch.dist(inp, other, 0), [inp, other], grad_outputs=grad)
    with pytest.raises(NotImplementedError):
        flag_gems.dist_backward(grad, inp, other, 0, None)


@pytest.mark.dist_backward
@pytest.mark.parametrize("p", [float("inf"), float("-inf"), 2, 1])
def test_dist_backward_global_tie_across_blocks(p, dtype=torch.float64):
    """p = +/-inf ties spanning multiple 1024-element blocks share the
    upstream gradient by the GLOBAL count (0.5/0.5 per maximizer), matching
    ATen's norm_backward."""
    n = 2048
    inp = torch.randn(n, dtype=dtype, device=flag_gems.device)
    other = torch.zeros(n, dtype=dtype, device=flag_gems.device)
    # Two maxima at |diff| = 2.5 in different blocks (indices 100 < 1024,
    # 1500 >= 1024); everything else stays strictly smaller.
    inp[100] = 2.5
    inp[1500] = -2.5
    rest = torch.randn(n, dtype=dtype, device=flag_gems.device) * 0.1
    keep = torch.ones(n, dtype=torch.bool, device=flag_gems.device)
    keep[100] = False
    keep[1500] = False
    inp[keep] = rest[keep]
    grad = torch.randn((), dtype=dtype, device=flag_gems.device)

    dist = torch.dist(inp, other, p).detach()
    ref_grads = _dist_backward_reference(grad, inp, other, p, dist)
    res_grads = flag_gems.dist_backward(grad, inp, other, p, dist)

    for ref, res in zip(ref_grads, res_grads):
        utils.gems_assert_close(res, ref.to(dtype), dtype)


@pytest.mark.dist_backward
def test_dist_backward_broadcast_not_supported(dtype=torch.float64):
    """Broadcasting / non-contiguous inputs degrade to ATen with a clear
    NotImplementedError (same policy as the fbgemm_pack_* family)."""
    inp = torch.randn(8, 32, dtype=dtype, device=flag_gems.device)
    other = torch.randn(32, dtype=dtype, device=flag_gems.device)
    grad = torch.randn((), dtype=dtype, device=flag_gems.device)
    dist = torch.dist(inp, other, 2).detach()

    with pytest.raises(NotImplementedError):
        flag_gems.dist_backward(grad, inp, other, 2, dist)
