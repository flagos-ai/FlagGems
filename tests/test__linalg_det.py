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
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from .accuracy_utils import gems_assert_close, gems_assert_equal, to_reference

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
@pytest.mark.parametrize("shape", [(3, 3), (4, 4), (5, 5), (2, 3, 3), (2, 2, 4, 4)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_linalg_det(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp, False)

    ref_result, ref_LU, ref_pivots = torch.ops.aten._linalg_det(ref_inp)
    # Call the FlagGems implementation directly (KernelGen tests must invoke
    # the op via flag_gems.ops rather than the global dispatch override).
    res_result, res_LU, res_pivots = flag_gems.ops._linalg_det(inp)

    # Check determinant is correct (the main output)
    gems_assert_close(res_result, ref_result, dtype)

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
