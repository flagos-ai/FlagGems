import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

DEVICE = flag_gems.device
VENDOR = flag_gems.vendor_name

if DEVICE == "cuda":
    _TEST_DTYPES = [torch.float32, torch.float64]
elif DEVICE == "npu":
    _TEST_DTYPES = [torch.float32]
else:
    _TEST_DTYPES = [torch.float32, torch.float64]

# pivot=False is only supported on CUDA
if DEVICE == "cuda":
    _PIVOT_VALUES = [True, False]
else:
    _PIVOT_VALUES = [True]


def _unpack_lu_no_pivot(lu):
    m, n = lu.shape[-2], lu.shape[-1]
    k = min(m, n)
    ll = lu[..., :, :k].tril()
    diag = torch.arange(k, device=lu.device)
    ll[..., diag, diag] = 1
    u = lu[..., :k, :].triu()
    return ll, u


def _make_input(shape, pivot, device, dtype):
    """Generate a test matrix suitable for the given pivot mode.

    For pivot=True, a random matrix is used (partial pivoting handles stability).
    For pivot=False, the matrix is constructed as L @ U where L has unit diagonal
    to guarantee a stable no-pivot LU factorization exists.
    """
    if pivot:
        return torch.randn(shape, dtype=dtype, device=device)

    # Construct A = L @ U where L is unit lower triangular and U is upper
    # triangular with a well-conditioned diagonal. Scale L's off-diagonal
    # elements to keep the triangular solve well-conditioned.
    *batch, m, n = shape
    k = min(m, n)
    scaling = k**-0.5
    L = (torch.randn(*batch, m, k, dtype=dtype, device=device) * scaling).tril()
    L.diagonal(dim1=-2, dim2=-1).fill_(1.0)
    U = torch.randn(*batch, k, n, dtype=dtype, device=device).triu()
    # Make U diagonally dominant for numerical stability
    U.diagonal(dim1=-2, dim2=-1).abs_().add_(1.0)
    return L @ U


@pytest.mark.linalg_lu_factor
@pytest.mark.parametrize(
    "shape",
    [
        (4, 4),
        (32, 32),
        (16, 32),
        (64, 32),
        (128, 16, 16),
        (128, 128),
        (128, 64),
        (64, 128),
        (256, 256),
        (512, 512),
    ],
)
@pytest.mark.parametrize("dtype", _TEST_DTYPES)
@pytest.mark.parametrize("pivot", [True, False])
def test_linalg_lu_factor(shape, dtype, pivot):
    inp = _make_input(shape, pivot, flag_gems.device, dtype)
    # inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_lu, ref_pivots = torch.linalg.lu_factor(ref_inp, pivot=pivot)
    with flag_gems.use_gems():
        res_lu, res_pivots = torch.linalg.lu_factor(inp, pivot=pivot)

    # batch_shape = inp.shape[:-2]
    # m, n = inp.shape[-2], inp.shape[-1]
    # k = min(m, n)

    utils.gems_assert_close(ref_lu, res_lu, dtype)

    # assert res_lu.shape == inp.shape
    # assert res_pivots.dtype == torch.int32
    # assert res_pivots.shape == (*batch_shape, k)
    # assert torch.all(res_pivots >= 1)
    # assert torch.all(res_pivots <= m)
    # utils.gems_assert_close(ref_lu, res_lu, dtype)
    # _tf32 = torch.backends.cuda.matmul.allow_tf32
    # torch.backends.cuda.matmul.allow_tf32 = False
    # try:
    #     # utils.gems_assert_close(ref_lu, res_lu, dtype)
    #     if pivot:
    #         res_p, res_l, res_u = torch.lu_unpack(res_lu, res_pivots)
    #         reconstructed = res_p @ res_l @ res_u
    #         utils.gems_assert_close(reconstructed, ref_inp, dtype, reduce_dim=k)
    #     else:
    #         res_l, res_u = _unpack_lu_no_pivot(res_lu)
    #         reconstructed = res_l @ res_u
    #         utils.gems_assert_close(reconstructed, ref_inp, dtype, reduce_dim=k)
    # finally:
    #     torch.backends.cuda.matmul.allow_tf32 = _tf32
