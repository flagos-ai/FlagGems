import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg


@pytest.mark.linalg_tensorsolve
# torch.linalg.tensorsolve does not support float16/bfloat16 on CUDA
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_tensorsolve_simple(dtype):
    # Simple case: 2D matrix equation
    # A is (n, n), B is (n,), X is (n,)
    if cfg.QUICK_MODE:
        n = 4
    else:
        n = 16

    # Create a diagonal matrix for simpler solving
    A = torch.eye(n, dtype=dtype, device=flag_gems.device)
    B = torch.randn(n, dtype=dtype, device=flag_gems.device)

    ref_A = utils.to_reference(A)
    ref_B = utils.to_reference(B)

    ref_out = torch.linalg.tensorsolve(ref_A, ref_B)
    with flag_gems.use_gems():
        res_out = torch.linalg.tensorsolve(A, B)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_tensorsolve
# torch.linalg.tensorsolve does not support float16/bfloat16 on CUDA
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_tensorsolve_3d(dtype):
    # Test with batch of independent matrix equations
    # For linalg.tensorsolve with B.shape (batch, n):
    # - We need to test each element in the batch separately
    if cfg.QUICK_MODE:
        n = 4
        batch = 2
    else:
        n = 8
        batch = 4

    # Create batch of diagonal matrices
    A = torch.eye(n, dtype=dtype, device=flag_gems.device)
    A = A.unsqueeze(0).expand(batch, -1, -1).contiguous()
    B = torch.randn(batch, n, dtype=dtype, device=flag_gems.device)

    ref_A = utils.to_reference(A)
    ref_B = utils.to_reference(B)

    # Test each batch element independently
    # tensorsolve expects: prod(A.shape[:B.ndim]) == prod(A.shape[B.ndim:])
    # For B.shape (batch, n), B.ndim = 2
    # This constraint doesn't match, so we test each element separately
    results_ref = []
    results_res = []
    for i in range(batch):
        ref_out_i = torch.linalg.tensorsolve(ref_A[i], ref_B[i])
        results_ref.append(ref_out_i)

        with flag_gems.use_gems():
            res_out_i = torch.linalg.tensorsolve(A[i], B[i])
        results_res.append(res_out_i)

    ref_out = torch.stack(results_ref)
    res_out = torch.stack(results_res)

    utils.gems_assert_close(res_out, ref_out, dtype)
