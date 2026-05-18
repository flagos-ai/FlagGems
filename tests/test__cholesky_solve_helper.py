import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

CHOLESKY_SOLVE_SHAPES = [(3, 3), (4, 4), (8, 8), (16, 16), (2, 4, 4), (3, 8, 8)]


def _make_cholesky_inputs(shape, dtype, device, upper=False):
    if len(shape) == 2:
        n = shape[0]
        batch_shape = ()
    else:
        n = shape[-1]
        batch_shape = shape[:-2]

    A = torch.randn(*batch_shape, n, n, dtype=dtype, device=device)
    A = A @ A.transpose(-2, -1) + n * torch.eye(n, dtype=dtype, device=device)
    L = torch.linalg.cholesky(A)
    if upper:
        L = L.transpose(-2, -1).contiguous()

    b = torch.randn(*batch_shape, n, 1, dtype=dtype, device=device)
    return b, L


@pytest.mark.cholesky_solve_helper
@pytest.mark.parametrize("shape", CHOLESKY_SOLVE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("upper", [False, True])
def test__cholesky_solve_helper(shape, dtype, upper):
    b, L = _make_cholesky_inputs(shape, dtype, flag_gems.device, upper=upper)
    ref_b = utils.to_reference(b)
    ref_L = utils.to_reference(L)

    ref_out = torch._cholesky_solve_helper(ref_b, ref_L, upper)
    with flag_gems.use_gems():
        res_out = torch._cholesky_solve_helper(b, L, upper)

    utils.gems_assert_close(res_out, ref_out, dtype)
