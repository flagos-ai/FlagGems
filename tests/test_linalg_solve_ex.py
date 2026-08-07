import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Test shapes: vector B and matrix B, various sizes
SOLVE_SHAPES = [
    # (A_shape, B_shape)
    ((4, 4), (4,)),  # vector RHS
    ((8, 8), (8, 2)),  # matrix RHS
    ((16, 16), (16, 4)),
    ((32, 32), (32, 8)),
]

# PyTorch _linalg_solve_ex supports float32, float64, complex64, complex128 on CUDA
# but NPU typically only supports float32
SOLVE_DTYPES = [torch.float32]


@pytest.mark.linalg_solve_ex
@pytest.mark.parametrize("shapes", SOLVE_SHAPES)
@pytest.mark.parametrize("dtype", SOLVE_DTYPES)
def test_linalg_solve_ex(shapes, dtype):
    """Test _linalg_solve_ex with well-conditioned matrices."""
    a_shape, b_shape = shapes
    n = a_shape[-1]

    # Create a well-conditioned matrix A
    A = torch.randn(a_shape, dtype=dtype, device=flag_gems.device)
    # Add strong diagonal to ensure stability
    A = A + n * torch.eye(n, dtype=dtype, device=flag_gems.device)

    B = torch.randn(b_shape, dtype=dtype, device=flag_gems.device)

    ref_A = utils.to_reference(A)
    ref_B = utils.to_reference(B)

    ref_out = torch.ops.aten._linalg_solve_ex(ref_A, ref_B)

    with flag_gems.use_gems():
        res_out = torch.ops.aten._linalg_solve_ex(A, B)

    # Compare result (solution X)
    utils.gems_assert_close(res_out[0], ref_out[0], dtype)

    # Compare info (should be 0 for successful solve)
    utils.gems_assert_equal(res_out[3], ref_out[3])

    # LU and pivots are implementation-dependent, so we only verify
    # that the solution is correct (already done above)


@pytest.mark.linalg_solve_ex
@pytest.mark.parametrize("dtype", SOLVE_DTYPES)
def test_linalg_solve_ex_batched(dtype):
    """Test batched _linalg_solve_ex."""
    batch_size = 2
    n = 5
    k = 3

    A = torch.randn(batch_size, n, n, dtype=dtype, device=flag_gems.device)
    # Add strong diagonal for stability
    A = A + n * torch.eye(n, dtype=dtype, device=flag_gems.device).unsqueeze(0)

    B = torch.randn(batch_size, n, k, dtype=dtype, device=flag_gems.device)

    ref_A = utils.to_reference(A)
    ref_B = utils.to_reference(B)

    ref_out = torch.ops.aten._linalg_solve_ex(ref_A, ref_B)

    with flag_gems.use_gems():
        res_out = torch.ops.aten._linalg_solve_ex(A, B)

    utils.gems_assert_close(res_out[0], ref_out[0], dtype)
    utils.gems_assert_equal(res_out[3], ref_out[3])
