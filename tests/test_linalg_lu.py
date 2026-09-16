from collections import namedtuple

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

DEVICE = flag_gems.device
VENDOR = flag_gems.vendor_name

if VENDOR == "nvidia":
    _TEST_DTYPES = [torch.float32, torch.float64]
else:
    _TEST_DTYPES = [torch.float32]

# pivot=False is only supported on CUDA
if utils.TO_CPU:
    _PIVOT_VALUES = [True]
elif DEVICE == "cuda":
    _PIVOT_VALUES = [True, False]
else:
    _PIVOT_VALUES = [True]

LinalgLUResult = namedtuple("LinalgLUResult", ["P", "L", "U"])

# Core shapes: square, rectangular (m>n, m<n), batched, edge cases
_LU_SHAPES = [
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
]


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


# ---------------------------------------------------------------------------
# Manual Python reference implementation for Ascend
# (matches the pattern in test_linalg_lu_factor_ex.py)
# ---------------------------------------------------------------------------


def _swap_rows(lu, i, pivot_row):
    # In-place gather/scatter swap: only (*batch_shape, n) temporaries instead
    # of the full-matrix broadcast temporaries of the mask-based swap.  Keeping
    # lu in place also means the pivot values recorded in pivot_vals keep
    # pointing at the one live lu tensor instead of pinning a fresh 1GB copy
    # per iteration (which OOMs the device for batched shapes like
    # (1024, 512, 512)).  Note: torch.gather/scatter_ are used instead of
    # advanced indexing (lu[..., pivot_row, :]) because torch_npu expands
    # tensor indices against the batch dims, yielding (b, b, n) instead of
    # (b, n).
    batch_shape = lu.shape[:-2]
    n = lu.shape[-1]
    row_i = lu[..., i, :].clone()
    idx = pivot_row.view(*batch_shape, 1, 1).expand(*batch_shape, 1, n)
    row_p = lu.gather(dim=-2, index=idx).squeeze(-2)
    lu[..., i, :] = row_p
    lu.scatter_(dim=-2, index=idx, src=row_i.unsqueeze(-2))
    return lu


def _lu_factor_pivot(lu, m, n, k):
    """LU factorization with partial pivoting."""
    *batch_shape, _, _ = lu.shape
    device = lu.device
    pivots = torch.empty((*batch_shape, k), dtype=torch.int32, device=device)

    for i in range(k):
        col = lu[..., i:, i].abs()
        pivot_rel = torch.argmax(col, dim=-1)
        pivot_row = pivot_rel + i
        pivots[..., i] = (pivot_row + 1).to(torch.int32)

        lu = _swap_rows(lu, i, pivot_row)

        pivot_val = lu[..., i, i]
        lu[..., i + 1 :, i] = lu[..., i + 1 :, i] / pivot_val.unsqueeze(-1)

        if i + 1 < m and i + 1 < n:
            l_col = lu[..., i + 1 :, i].unsqueeze(-1)
            u_row = lu[..., i : i + 1, i + 1 :]
            lu[..., i + 1 :, i + 1 :] = lu[..., i + 1 :, i + 1 :] - l_col @ u_row

    return lu, pivots


def _lu_factor_no_pivot(lu, m, n, k):
    """LU factorization without pivoting."""
    *batch_shape, _, _ = lu.shape
    device = lu.device
    pivots = torch.empty((*batch_shape, k), dtype=torch.int32, device=device)

    for i in range(k):
        pivots[..., i] = i + 1
        pivot_val = lu[..., i, i]
        lu[..., i + 1 :, i] = lu[..., i + 1 :, i] / pivot_val.unsqueeze(-1)

        if i + 1 < m and i + 1 < n:
            l_col = lu[..., i + 1 :, i].unsqueeze(-1)
            u_row = lu[..., i : i + 1, i + 1 :]
            lu[..., i + 1 :, i + 1 :] = lu[..., i + 1 :, i + 1 :] - l_col @ u_row

    return lu, pivots


def _unpack_lu_manual(lu, pivots, m, n, k, pivot):
    """Manual unpack of (lu, pivots) into (P, L, U).

    P is built by applying the row swaps to the identity in REVERSE order
    (i = k-1 .. 0), which is what torch.linalg.lu does: its P equals the
    product S_0 @ S_1 @ ... @ S_{k-1} of the row-swap matrices, so the
    last pivot swap acts first on the identity.  (Applying the swaps
    forward yields the wrong matrix.)
    """
    device = lu.device
    dtype = lu.dtype
    batch_shape = lu.shape[:-2]

    ll = lu[..., :, :k].tril()
    diag = torch.arange(k, device=device)
    ll[..., diag, diag] = 1
    u = lu[..., :k, :].triu()

    if not pivot:
        return torch.empty(0, device=device, dtype=dtype), ll, u

    P = torch.zeros((*batch_shape, m, m), device=device, dtype=dtype)
    diag_m = torch.arange(m, device=device)
    P[..., diag_m, diag_m] = 1
    for i in range(k - 1, -1, -1):
        P = _swap_rows(P, i, (pivots[..., i] - 1).to(torch.int64))
    return P, ll, u


def ops_lu(input, *, pivot=True):
    """Manual Python reference for linalg_lu."""
    if input.dim() < 2:
        raise RuntimeError(
            "torch.linalg.lu: Expected input to have at least 2 dimensions"
        )
    if input.dtype not in (torch.float32, torch.float64):
        raise NotImplementedError("Only float32 and float64 are supported")
    m, n = input.shape[-2], input.shape[-1]
    if m == 0 or n == 0:
        raise NotImplementedError("Empty matrices are not supported")
    if pivot not in (True, False):
        raise TypeError(f"pivot must be a bool, got {type(pivot)}")

    input_contiguous = input.contiguous()
    m, n = input_contiguous.shape[-2], input_contiguous.shape[-1]
    k = min(m, n)
    lu = input_contiguous.clone()

    if pivot:
        lu, pivots = _lu_factor_pivot(lu, m, n, k)
    else:
        lu, pivots = _lu_factor_no_pivot(lu, m, n, k)

    P, L, U = _unpack_lu_manual(lu, pivots, m, n, k, pivot)
    return LinalgLUResult(P, L, U)


def _run_torch_ops_path(inp, pivot):
    """Vendor-aware wrapper: uses manual ops on Ascend, native on other vendors."""
    res = ops_lu(inp, pivot=pivot)
    return res.P, res.L, res.U


def _check_structure(res_out, device):
    """Structural checks on P, L, U."""
    batch_shape = res_out.L.shape[:-2]
    m, _ = res_out.L.shape[-2], res_out.U.shape[-1]
    k = res_out.L.shape[-1]
    diag = torch.arange(k, device=device)

    # L: unit diagonal, zero above the diagonal
    assert torch.all(res_out.L[..., diag, diag] == 1)
    assert torch.all(res_out.L.triu(diagonal=1) == 0)
    # U: zero below the diagonal
    assert torch.all(res_out.U.tril(diagonal=-1) == 0)

    if res_out.P.numel() == 0:
        return

    # P: permutation matrix — 0/1 entries, exactly one 1 per row and column
    assert res_out.P.shape == (*batch_shape, m, m)
    assert torch.all((res_out.P == 0) | (res_out.P == 1))
    assert torch.all(res_out.P.sum(dim=-1) == 1)
    assert torch.all(res_out.P.sum(dim=-2) == 1)


def _reconstruct(out, pivot):
    if pivot:
        return out.P @ out.L @ out.U
    return out.L @ out.U


@pytest.mark.linalg_lu
@pytest.mark.parametrize("shape", _LU_SHAPES)
@pytest.mark.parametrize("dtype", _TEST_DTYPES)
@pytest.mark.parametrize("pivot", _PIVOT_VALUES)
def test_linalg_lu(shape, dtype, pivot):
    inp = _make_input(shape, pivot, flag_gems.device, dtype)
    ref_inp = utils.to_reference(inp)

    if flag_gems.vendor_name != "ascend":
        ref_out = torch.linalg.lu(ref_inp, pivot=pivot)
    else:
        ref_P, ref_L, ref_U = _run_torch_ops_path(ref_inp, pivot=pivot)
        ref_out = LinalgLUResult(ref_P, ref_L, ref_U)
    with flag_gems.use_gems():
        res_out = torch.linalg.lu(inp, pivot=pivot)

    batch_shape = inp.shape[:-2]
    m, n = inp.shape[-2], inp.shape[-1]
    k = min(m, n)

    # Validate shapes and types
    assert res_out.L.shape == (*batch_shape, m, k)
    assert res_out.U.shape == (*batch_shape, k, n)
    assert res_out.P.dtype == dtype
    assert res_out.L.dtype == dtype
    assert res_out.U.dtype == dtype
    if pivot:
        assert res_out.P.shape == (*batch_shape, m, m)
        assert ref_out.P.shape == (*batch_shape, m, m)
    else:
        # P is an empty 1-D tensor when pivot=False
        assert res_out.P.numel() == 0
        assert res_out.P.shape == (0,)
        assert ref_out.P.numel() == 0

    _check_structure(res_out, inp.device)
    _check_structure(ref_out, ref_inp.device)

    # Validate accuracy by reconstructing P @ L @ U (or L @ U) and comparing
    # against the reference reconstruction (both should reproduce A).
    torch.backends.cuda.matmul.allow_tf32 = False
    reconstructed = _reconstruct(res_out, pivot)
    ref_reconstructed = _reconstruct(ref_out, pivot)
    utils.gems_assert_close(reconstructed, ref_reconstructed, dtype, reduce_dim=k)


@pytest.mark.linalg_lu_out
@pytest.mark.parametrize("shape", _LU_SHAPES)
@pytest.mark.parametrize("dtype", _TEST_DTYPES)
@pytest.mark.parametrize("pivot", _PIVOT_VALUES)
def test_linalg_lu_out(shape, dtype, pivot):
    """Test the out= parameter variant."""
    if not pivot and flag_gems.device != "cuda":
        pytest.skip("pivot=False only supported on CUDA")

    inp = _make_input(shape, pivot, flag_gems.device, dtype)
    ref_inp = utils.to_reference(inp)

    batch_shape = inp.shape[:-2]
    m, n = inp.shape[-2], inp.shape[-1]
    k = min(m, n)

    # Reference: use out= parameter (or manual reference on Ascend).
    # For pivot=False, P is pre-allocated empty to avoid the deprecation
    # warning torch emits when it resizes a non-empty out tensor to (0,).
    ref_P_out = (
        torch.empty((*batch_shape, m, m), dtype=dtype, device=ref_inp.device)
        if pivot
        else torch.empty(0, dtype=dtype, device=ref_inp.device)
    )
    ref_L_out = torch.empty((*batch_shape, m, k), dtype=dtype, device=ref_inp.device)
    ref_U_out = torch.empty((*batch_shape, k, n), dtype=dtype, device=ref_inp.device)
    if flag_gems.vendor_name != "ascend":
        ref_out = torch.linalg.lu(
            ref_inp, pivot=pivot, out=(ref_P_out, ref_L_out, ref_U_out)
        )
    else:
        ref_P, ref_L, ref_U = _run_torch_ops_path(ref_inp, pivot=pivot)
        ref_P_out.resize_(ref_P.shape)
        ref_P_out.copy_(ref_P)
        ref_L_out.copy_(ref_L)
        ref_U_out.copy_(ref_U)
        ref_out = LinalgLUResult(ref_P_out, ref_L_out, ref_U_out)

    # Gems: use out= parameter
    res_P_out = (
        torch.empty((*batch_shape, m, m), dtype=dtype, device=inp.device)
        if pivot
        else torch.empty(0, dtype=dtype, device=inp.device)
    )
    res_L_out = torch.empty((*batch_shape, m, k), dtype=dtype, device=inp.device)
    res_U_out = torch.empty((*batch_shape, k, n), dtype=dtype, device=inp.device)
    out = (res_P_out, res_L_out, res_U_out)
    with flag_gems.use_gems():
        res_out = torch.linalg.lu(inp, pivot=pivot, out=out)

    # Verify outputs are the same objects (in-place write)
    assert res_out.P is res_P_out
    assert res_out.L is res_L_out
    assert res_out.U is res_U_out

    # Verify shapes
    assert res_out.L.shape == (*batch_shape, m, k)
    assert res_out.U.shape == (*batch_shape, k, n)
    if pivot:
        assert res_out.P.shape == (*batch_shape, m, m)
    else:
        assert res_out.P.numel() == 0
        assert ref_out.P.numel() == 0

    _check_structure(res_out, inp.device)

    # Validate accuracy via reconstructed product P @ L @ U (or L @ U)
    torch.backends.cuda.matmul.allow_tf32 = False
    reconstructed = _reconstruct(res_out, pivot)
    ref_reconstructed = _reconstruct(ref_out, pivot)
    utils.gems_assert_close(reconstructed, ref_reconstructed, dtype, reduce_dim=k)
