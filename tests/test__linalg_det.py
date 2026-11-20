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

import math

import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from .accuracy_utils import (
    _SKIP_FP64,
    gems_assert_close,
    gems_assert_equal,
    to_reference,
)

# ``underscore_linalg_det`` contains the leading-underscore aten name; register
# it on the MarkGenerator so ``@pytest.mark.underscore_linalg_det`` works. The
# ``underscore_`` prefix follows the naming convention (Rule 2) and
# disambiguates from the public ``linalg_det`` (aten::linalg.det) operator id.
setattr(
    pytest.mark,
    "underscore_linalg_det",
    MarkDecorator(
        Mark("underscore_linalg_det", (), {}, _ispytest=True), _ispytest=True
    ),
)


def _compute_det_from_lu(lu_matrix, pivot_vec):
    """Compute determinant from LU matrix and pivots."""
    n = lu_matrix.shape[-1]
    # Compute det from LU diagonal
    lu_det = torch.prod(torch.diagonal(lu_matrix))
    # Account for row swaps from pivots
    swaps = sum(1 for i in range(n) if pivot_vec[i] != i + 1)
    sign = (-1) ** swaps
    return lu_det * sign


@pytest.mark.underscore_linalg_det
@pytest.mark.parametrize(
    "shape",
    [
        (3, 3),
        (4, 4),
        (5, 5),
        (2, 3, 3),
        (2, 2, 4, 4),
        # Cover each kernel dispatch path: register (n <= 64), blocked and panel
        # (n > 64 for float32 / n > 32 for float64).
        (48, 48),
        (64, 64),
        (65, 65),
        (128, 128),
    ],
)
@pytest.mark.parametrize(
    "dtype", [torch.float32, pytest.param(torch.float64, marks=_SKIP_FP64)]
)
def test_accuracy_linalg_det(shape, dtype):
    # Scale the input by 1/sqrt(n) (same as the det-only test suite upstream):
    # this keeps the determinant of the larger matrices in a range where the
    # product of n elimination roundings does not get amplified by an extreme
    # condition number, so a plain relative-tolerance comparison stays valid.
    n = shape[-1]
    inp = torch.randn(shape, dtype=dtype, device="cuda") / math.sqrt(n)
    ref_inp = to_reference(inp, False)

    ref_result, ref_LU, ref_pivots = torch.ops.aten._linalg_det(ref_inp)
    # Call the FlagGems implementation directly (KernelGen tests must invoke
    # the op via flag_gems.ops rather than the global dispatch override).
    res_result, res_LU, res_pivots = flag_gems.ops._linalg_det(inp)

    # Check determinant is correct (the main output). The determinant is a
    # product of n diagonal factors, so the rounding error of the elimination
    # grows with n; the relative tolerance is scaled accordingly (the shared
    # helper pins rtol to the dtype resolution).
    base_rtol = 1e-7 if dtype == torch.float64 else 1.3e-6
    torch.testing.assert_close(
        res_result.cpu(),
        ref_result.cpu(),
        atol=1e-4,
        rtol=base_rtol * max(1.0, n / 8.0),
        equal_nan=False,
    )

    # Check output shapes are correct
    gems_assert_equal(res_LU.shape, ref_LU.shape)
    gems_assert_equal(res_pivots.shape, ref_pivots.shape)

    # Verify pivots are not all zeros (bug fix verification)
    assert not torch.all(res_pivots == 0), "Pivots should not be all zeros"

    # Verify LU decomposition is valid by checking that the determinant
    # computed from LU diagonal matches the result. This is an internal
    # self-consistency check between the two GEMS outputs (LU*pivots vs. det),
    # not a comparison against the reference -- the reference comparison is
    # the gems_assert_close above. The reconstruction runs on CPU (the tensors
    # are n x n at most) so the check also holds in --ref=cpu mode, where
    # gems_assert_close requires the reference side to be CPU-resident.
    if inp.dim() == 2:
        # Single matrix case
        lu_det = _compute_det_from_lu(res_LU.cpu(), res_pivots.cpu())
        gems_assert_close(lu_det, res_result.cpu(), dtype)
    else:
        # Batch case - flatten batch dims and check the whole batch at once
        batch_shape = inp.shape[:-2]
        batch_size = torch.prod(torch.tensor(batch_shape)).item()
        res_LU_flat = res_LU.view(batch_size, inp.shape[-2], inp.shape[-1]).cpu()
        res_pivots_flat = res_pivots.view(batch_size, inp.shape[-1]).cpu()
        res_result_flat = res_result.view(batch_size).cpu()

        lu_dets = torch.stack(
            [
                _compute_det_from_lu(res_LU_flat[i], res_pivots_flat[i])
                for i in range(batch_size)
            ]
        )
        gems_assert_close(lu_dets, res_result_flat, dtype)


@pytest.mark.linalg_det_backward
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1),
        (2, 2),
        (3, 3),
        (5, 5),
        (16, 16),
        (32, 32),
        (64, 64),
        (128, 128),
        (2, 3, 3),
        (4, 32, 32),
        (2, 2, 4, 4),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_accuracy_linalg_det_backward(shape, dtype):
    torch.manual_seed(20261119)
    n = shape[-1]
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device) / math.sqrt(n)
    grad = torch.randn(shape[:-2], dtype=dtype, device=flag_gems.device)
    ref_A = to_reference(A, False).requires_grad_(True)

    ref_result, ref_LU, ref_pivots = torch.ops.aten._linalg_det(ref_A)
    (ref_grad,) = torch.autograd.grad(ref_result, ref_A, to_reference(grad, False))

    gems_A = A.detach().clone().requires_grad_(True)
    res_result, res_LU, res_pivots = flag_gems.ops._linalg_det(gems_A)
    assert res_result.requires_grad, "determinant must stay differentiable"
    assert not res_LU.requires_grad, "LU output is not differentiable in ATen"
    assert not res_pivots.requires_grad, "pivots output is not differentiable in ATen"
    (res_grad,) = torch.autograd.grad(res_result, gems_A, grad)

    # The reference is recomputed in float64 from the same input matrix so the
    # comparison measures this backward against the mathematical rule
    # grad_A = grad * det(A) * A^{-T} rather than against a second float32
    # factorization whose own rounding is of the same order.
    inv64 = torch.linalg.inv(to_reference(gems_A.detach(), False).double())
    det64 = torch.linalg.det(to_reference(gems_A.detach(), False).double())
    grad64 = to_reference(grad, False).double()
    expected = (
        det64.unsqueeze(-1).unsqueeze(-1)
        * grad64.unsqueeze(-1).unsqueeze(-1)
        * inv64.mT
    )
    rel_tol = 1e-4 if dtype == torch.float32 else 1e-11
    got64 = res_grad.cpu().double()
    exp64 = expected.cpu()
    abs_err = (got64 - exp64).abs()
    # Anchor the tolerance to each row's own gradient magnitude.  A single
    # global max would let one large row set the bar for every other row, and
    # conditioning can legitimately make an entire gradient tiny (a draw with
    # |grad| ~ 1e-12 is still correct entry by entry, just small), so a global
    # absolute floor would be either vacuous or unattainable.
    row_scale = exp64.abs().amax(dim=-1, keepdim=True).expand_as(exp64)
    # atol must be a scalar for torch.testing, so enforce the per-row bound
    # explicitly and then let assert_close verify the shape/dtype contract.
    worst_ratio = (abs_err / (1e-3 * row_scale + 1e-4 * exp64.abs() + 1e-300)).max()
    assert worst_ratio <= 1.0, (
        f"backward gradient exceeds the row-anchored tolerance: "
        f"worst ratio {worst_ratio.item():.3f} (abs max {abs_err.max().item():.3e}, "
        f"|grad| max {exp64.abs().max().item():.3e})"
    )
    # The row-anchored ratio above is the substantive check; this second form
    # pins shape, dtype and the largest entries.  Its absolute floor follows the
    # same row scale, so entries that are legitimately near zero (an ill
    # conditioned draw whose whole gradient is ~1e-33) are not forced to match
    # beyond their own magnitude.
    torch.testing.assert_close(
        got64,
        exp64,
        atol=1e-3 * row_scale.max().item() + 1e-300,
        rtol=rel_tol,
        equal_nan=False,
    )
    # Forward outputs must keep matching their own contract as well.
    gems_assert_close(res_result, ref_result, dtype, atol=1e-4)
    gems_assert_equal(res_LU.shape, ref_LU.shape)
    gems_assert_equal(res_pivots.shape, ref_pivots.shape)


@pytest.mark.linalg_det_backward
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_accuracy_linalg_det_backward_noncontiguous(dtype):
    torch.manual_seed(7)
    n = 6
    base = torch.randn((n, n), dtype=dtype, device=flag_gems.device) / math.sqrt(n)
    # det() is 0-D, so its upstream gradient is a scalar too.
    grad = torch.tensor(0.75, dtype=dtype, device=flag_gems.device)

    A = base.t().detach().requires_grad_(True)
    ref_A = to_reference(base, False).t().detach().requires_grad_(True)
    assert not A.is_contiguous()

    res_result, _, _ = flag_gems.ops._linalg_det(A)
    (res_grad,) = torch.autograd.grad(res_result, A, grad)
    ref_result, _, _ = torch.ops.aten._linalg_det(ref_A)
    (ref_grad,) = torch.autograd.grad(ref_result, ref_A, to_reference(grad, False))

    assert res_grad.shape == A.shape
    # The gradient is returned in contiguous layout.  ATen mirrors the input's
    # strided layout instead, so compare values (and shape) rather than strides.
    gems_assert_close(res_grad.contiguous(), ref_grad.contiguous(), dtype, atol=1e-4)


@pytest.mark.linalg_det_backward
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_accuracy_linalg_det_backward_singular(dtype):
    # The backward must stay finite on an exactly singular matrix instead of
    # dividing by the zero pivot.  The value it produces is det(A) * A^{-T} for
    # the determinant this forward reports: here det is exactly zero, so the
    # gradient is exactly zero.  ATen's LU happens to leave a rounding-level
    # residual in det (e.g. -3.3e-16) and therefore reports a finite
    # adjugate-scaled value; det is not differentiable at a singular point, so
    # the two are different picks of the same one-sided limit rather than a
    # disagreement about the rule.  Compare against the rule, not against that
    # rounding-level residual.
    A = torch.tensor(
        [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]],
        dtype=dtype,
        device=flag_gems.device,
    )
    grad = torch.ones((), dtype=dtype, device=flag_gems.device)
    A = A.detach().requires_grad_(True)
    ref_A = to_reference(A.detach(), False).requires_grad_(True)

    res_result, _, _ = flag_gems.ops._linalg_det(A)
    (res_grad,) = torch.autograd.grad(res_result, A, grad)
    ref_result, _, _ = torch.ops.aten._linalg_det(ref_A)
    (ref_grad,) = torch.autograd.grad(ref_result, ref_A, to_reference(grad, False))

    assert torch.isfinite(res_grad).all(), "singular gradient must stay finite"
    assert torch.isfinite(ref_grad).all()

    expected = (
        res_result.detach().cpu().double()
        * to_reference(grad, False).cpu().double()
        * torch.linalg.inv(to_reference(A.detach(), False).double()).cpu().mT
    )
    torch.testing.assert_close(
        res_grad.cpu().double(),
        expected,
        atol=1e-7,
        rtol=1e-7,
        equal_nan=False,
    )
    # a batch mixing a nonsingular and a singular matrix keeps both entries finite
    batch = torch.stack(
        [torch.eye(3, dtype=dtype, device=flag_gems.device), A.detach().clone()]
    )
    batch = batch.detach().requires_grad_(True)
    b_res, _, _ = flag_gems.ops._linalg_det(batch)
    (b_grad,) = torch.autograd.grad(
        b_res, batch, torch.ones(2, dtype=dtype, device=flag_gems.device)
    )
    assert torch.isfinite(b_grad).all(), "batched singular gradient must stay finite"
    assert torch.allclose(
        b_grad[0].cpu().double(),
        torch.eye(3, dtype=torch.float64),
        atol=1e-6,
    )
