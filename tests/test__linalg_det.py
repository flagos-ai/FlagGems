import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_close, gems_assert_equal, to_reference


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
    # computed from LU diagonal matches the result
    if inp.dim() == 2:
        # Single matrix case
        lu_det = _compute_det_from_lu(res_LU, res_pivots)
        assert (
            torch.abs(lu_det - res_result) < 1e-3
        ), f"LU determinant {lu_det} doesn't match result {res_result}"
    else:
        # Batch case - flatten batch dims and iterate
        batch_shape = inp.shape[:-2]
        batch_size = torch.prod(torch.tensor(batch_shape)).item()
        res_LU_flat = res_LU.view(batch_size, inp.shape[-2], inp.shape[-1])
        res_pivots_flat = res_pivots.view(batch_size, inp.shape[-1])
        res_result_flat = res_result.view(batch_size)

        for i in range(batch_size):
            lu_det = _compute_det_from_lu(res_LU_flat[i], res_pivots_flat[i])
            det_result = res_result_flat[i]
            assert (
                torch.abs(lu_det - det_result) < 1e-3
            ), f"Batch {i}: LU det {lu_det} doesn't match result {det_result}"
