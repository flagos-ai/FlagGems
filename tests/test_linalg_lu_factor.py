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
    ref_inp = utils.to_reference(inp)

    ref_lu, ref_pivots = torch.linalg.lu_factor(ref_inp, pivot=pivot)
    with flag_gems.use_gems():
        res_lu, res_pivots = torch.linalg.lu_factor(inp, pivot=pivot)

    batch_shape = inp.shape[:-2]
    m, n = inp.shape[-2], inp.shape[-1]
    k = min(m, n)

    assert res_lu.shape == inp.shape
    assert res_pivots.dtype == torch.int32
    assert res_pivots.shape == (*batch_shape, k)
    assert torch.all(res_pivots >= 1)
    assert torch.all(res_pivots <= m)

    _tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        if pivot:
            res_p, res_l, res_u = torch.lu_unpack(res_lu, res_pivots)
            reconstructed = res_p @ res_l @ res_u
            utils.gems_assert_close(reconstructed, ref_inp, dtype, reduce_dim=k)
        else:
            res_l, res_u = _unpack_lu_no_pivot(res_lu)
            reconstructed = res_l @ res_u
            utils.gems_assert_close(reconstructed, ref_inp, dtype, reduce_dim=k)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = _tf32


@pytest.mark.linalg_lu_factor_out
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
@pytest.mark.parametrize("pivot", _PIVOT_VALUES)
def test_linalg_lu_factor_out(shape, dtype, pivot):
    if not pivot and flag_gems.device != "cuda":
        pytest.skip("pivot=False only supported on CUDA")

    inp = _make_input(shape, pivot, flag_gems.device, dtype)
    ref_inp = utils.to_reference(inp)

    batch_shape = inp.shape[:-2]
    m, n = inp.shape[-2], inp.shape[-1]
    k = min(m, n)

    ref_LU_out = torch.empty_like(ref_inp)
    ref_pivots_out = torch.empty(
        (*batch_shape, k), dtype=torch.int32, device=ref_inp.device
    )
    ref_LU, ref_pivots = torch.linalg.lu_factor(
        ref_inp, pivot=pivot, out=(ref_LU_out, ref_pivots_out)
    )

    res_LU_out = torch.empty_like(inp)
    res_pivots_out = torch.empty(
        (*batch_shape, k), dtype=torch.int32, device=inp.device
    )
    out = (res_LU_out, res_pivots_out)
    with flag_gems.use_gems():
        res_LU, res_pivots = flag_gems.linalg_lu_factor_out(inp, pivot=pivot, out=out)

    assert res_LU is res_LU_out
    assert res_pivots is res_pivots_out

    assert res_LU.shape == inp.shape
    assert res_pivots.dtype == torch.int32
    assert res_pivots.shape == (*batch_shape, k)
    assert torch.all(res_pivots >= 1)
    assert torch.all(res_pivots <= m)

    _tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        if pivot:
            res_p, res_l, res_u = torch.lu_unpack(res_LU, res_pivots)
            reconstructed = res_p @ res_l @ res_u
            utils.gems_assert_close(reconstructed, ref_inp, dtype, reduce_dim=k)
        else:
            res_l, res_u = _unpack_lu_no_pivot(res_LU)
            reconstructed = res_l @ res_u
            utils.gems_assert_close(reconstructed, ref_inp, dtype, reduce_dim=k)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = _tf32
