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

_CHECK_ERRORS_VALUES = [False, True]

LinalgLUFactorExResult = namedtuple("LinalgLUFactorExResult", ["LU", "pivots", "info"])

# Core shapes: square, rectangular (m>n, m<n), batched, edge cases
_LU_FACTOR_EX_SHAPES = [
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
    # triangular with a well-conditioned diagonal.
    *batch, m, n = shape
    k = min(m, n)
    scaling = k**-0.5
    L = (torch.randn(*batch, m, k, dtype=dtype, device=device) * scaling).tril()
    L.diagonal(dim1=-2, dim2=-1).fill_(1.0)
    U = torch.randn(*batch, k, n, dtype=dtype, device=device).triu()
    U.diagonal(dim1=-2, dim2=-1).abs_().add_(1.0)
    return L @ U


def _make_singular_input(shape, device, dtype):
    """Generate a singular matrix by making the last row all zeros.

    An all-zero last row guarantees that after processing all preceding columns,
    the updated pivot at that row will be exactly zero regardless of row swaps.
    This reliably produces info = k (1-indexed position of the first zero pivot).
    """
    A = torch.randn(shape, dtype=dtype, device=device)
    # Set the last row to zero. Partial pivoting may swap rows, but when
    # the last row is all zeros, it will always end up with a zero pivot
    # at the final elimination step.
    A[..., -1, :] = 0.0
    return A


# ---------------------------------------------------------------------------
# Manual Python reference implementation for Ascend
# (matches the pattern in test_linalg_lu_factor.py)
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


def _lu_factor_pivot_ex(lu, m, n, k):
    """LU factorization with partial pivoting + info tracking."""
    *batch_shape, _, _ = lu.shape
    device = lu.device
    pivots = torch.empty((*batch_shape, k), dtype=torch.int32, device=device)
    # Collect pivots in a plain Python list (zero extra device ops per step)
    # and compute info once after the loop with vector ops.
    pivot_vals = []

    for i in range(k):
        col = lu[..., i:, i].abs()
        pivot_rel = torch.argmax(col, dim=-1)
        pivot_row = pivot_rel + i
        pivots[..., i] = (pivot_row + 1).to(torch.int32)

        lu = _swap_rows(lu, i, pivot_row)

        # .clone() so the recorded value does not pin lu's storage (a plain
        # view would keep the referenced lu storage alive; combined with a
        # rebinding lu this OOMs the device for batched shapes).
        pivot_val = lu[..., i, i].clone()
        pivot_vals.append(pivot_val)

        lu[..., i + 1 :, i] = lu[..., i + 1 :, i] / pivot_val.unsqueeze(-1)

        if i + 1 < m and i + 1 < n:
            l_col = lu[..., i + 1 :, i].unsqueeze(-1)
            u_row = lu[..., i : i + 1, i + 1 :]
            lu[..., i + 1 :, i + 1 :] = lu[..., i + 1 :, i + 1 :] - l_col @ u_row

    # First zero/NaN pivot position (1-indexed), 0 if none.
    diag = torch.stack(pivot_vals, dim=-1)  # (*batch_shape, k)
    bad_mask = (diag == 0) | torch.isnan(diag)
    has_bad = torch.any(bad_mask, dim=-1)
    first_bad = torch.argmax(bad_mask.to(torch.int32), dim=-1)
    info = torch.where(has_bad, first_bad + 1, 0).to(torch.int32)

    return lu, pivots, info


def _lu_factor_no_pivot_ex(lu, m, n, k):
    """LU factorization without pivoting + info tracking."""
    *batch_shape, _, _ = lu.shape
    device = lu.device
    pivots = torch.empty((*batch_shape, k), dtype=torch.int32, device=device)
    pivot_vals = []

    for i in range(k):
        pivots[..., i] = i + 1
        # .clone() so the recorded value does not pin lu's storage (a plain
        # view would keep the referenced lu storage alive; combined with a
        # rebinding lu this OOMs the device for batched shapes).
        pivot_val = lu[..., i, i].clone()
        pivot_vals.append(pivot_val)

        lu[..., i + 1 :, i] = lu[..., i + 1 :, i] / pivot_val.unsqueeze(-1)

        if i + 1 < m and i + 1 < n:
            l_col = lu[..., i + 1 :, i].unsqueeze(-1)
            u_row = lu[..., i : i + 1, i + 1 :]
            lu[..., i + 1 :, i + 1 :] = lu[..., i + 1 :, i + 1 :] - l_col @ u_row

    # First zero/NaN pivot position (1-indexed), 0 if none.
    diag = torch.stack(pivot_vals, dim=-1)  # (*batch_shape, k)
    bad_mask = (diag == 0) | torch.isnan(diag)
    has_bad = torch.any(bad_mask, dim=-1)
    first_bad = torch.argmax(bad_mask.to(torch.int32), dim=-1)
    info = torch.where(has_bad, first_bad + 1, 0).to(torch.int32)

    return lu, pivots, info


def ops_lu_factor_ex(input, *, pivot=True, check_errors=False):
    """Manual Python reference for linalg_lu_factor_ex."""
    if input.dim() < 2:
        raise RuntimeError(
            "torch.linalg.lu_factor_ex: Expected input to have at least 2 dimensions"
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
        lu, pivots, info = _lu_factor_pivot_ex(lu, m, n, k)
    else:
        lu, pivots, info = _lu_factor_no_pivot_ex(lu, m, n, k)

    return LinalgLUFactorExResult(lu, pivots, info)


def _run_torch_ops_path_ex(inp, pivot):
    """Vendor-aware wrapper: uses manual ops on Ascend, native on other vendors."""
    res = ops_lu_factor_ex(inp, pivot=pivot)
    return res.LU, res.pivots, res.info


@pytest.mark.linalg_lu_factor_ex
@pytest.mark.parametrize("shape", _LU_FACTOR_EX_SHAPES)
@pytest.mark.parametrize("dtype", _TEST_DTYPES)
@pytest.mark.parametrize("pivot", _PIVOT_VALUES)
def test_linalg_lu_factor_ex(shape, dtype, pivot):
    inp = _make_input(shape, pivot, flag_gems.device, dtype)
    ref_inp = utils.to_reference(inp)

    if flag_gems.vendor_name != "ascend":
        ref_out = torch.linalg.lu_factor_ex(ref_inp, pivot=pivot)
    else:
        ref_lu, ref_pivots, ref_info = _run_torch_ops_path_ex(ref_inp, pivot=pivot)
        ref_out = namedtuple("_RefResult", ["LU", "pivots", "info"])(
            ref_lu, ref_pivots, ref_info
        )
    with flag_gems.use_gems():
        res_out = torch.linalg.lu_factor_ex(inp, pivot=pivot)

    batch_shape = inp.shape[:-2]
    m, n = inp.shape[-2], inp.shape[-1]
    k = min(m, n)

    # Validate shapes and types
    assert res_out.LU.shape == inp.shape
    assert res_out.pivots.dtype == torch.int32
    assert res_out.pivots.shape == (*batch_shape, k)
    assert res_out.info.dtype == torch.int32
    assert res_out.info.shape == batch_shape
    assert torch.all(res_out.pivots >= 1)
    assert torch.all(res_out.pivots <= m)

    # info must match: for well-conditioned random matrices both should be 0;
    # if either implementation reports singularity, catch it here.
    utils.gems_assert_equal(res_out.info, ref_out.info)

    # Validate LU factorization accuracy by reconstructing P @ L @ U (or L @ U)
    torch.backends.cuda.matmul.allow_tf32 = False
    if pivot:
        res_p, res_l, res_u = torch.lu_unpack(res_out.LU, res_out.pivots)
        ref_p, ref_l, ref_u = torch.lu_unpack(ref_out.LU, ref_out.pivots)
        reconstructed = res_p @ res_l @ res_u
        ref_reconstructed = ref_p @ ref_l @ ref_u
    else:
        res_l, res_u = _unpack_lu_no_pivot(res_out.LU)
        ref_l, ref_u = _unpack_lu_no_pivot(ref_out.LU)
        reconstructed = res_l @ res_u
        ref_reconstructed = ref_l @ ref_u
    utils.gems_assert_close(reconstructed, ref_reconstructed, dtype, reduce_dim=k)


@pytest.mark.linalg_lu_factor_ex
@pytest.mark.parametrize("shape", _LU_FACTOR_EX_SHAPES)
@pytest.mark.parametrize("dtype", _TEST_DTYPES)
@pytest.mark.parametrize("pivot", _PIVOT_VALUES)
def test_linalg_lu_factor_ex_check_errors(shape, dtype, pivot):
    """Test that check_errors=True doesn't raise for well-conditioned matrices."""
    inp = _make_input(shape, pivot, flag_gems.device, dtype)
    ref_inp = utils.to_reference(inp)

    if flag_gems.vendor_name != "ascend":
        ref_out = torch.linalg.lu_factor_ex(ref_inp, pivot=pivot, check_errors=True)
    else:
        ref_lu, ref_pivots, ref_info = _run_torch_ops_path_ex(ref_inp, pivot=pivot)
        ref_out = namedtuple("_RefResult", ["LU", "pivots", "info"])(
            ref_lu, ref_pivots, ref_info
        )
    with flag_gems.use_gems():
        res_out = torch.linalg.lu_factor_ex(inp, pivot=pivot, check_errors=True)

    # Both should have info == 0 for well-conditioned input
    assert torch.all(res_out.info == 0)
    assert torch.all(ref_out.info == 0)
    utils.gems_assert_equal(res_out.info, ref_out.info)


@pytest.mark.linalg_lu_factor_ex
@pytest.mark.parametrize("shape", [(4, 4), (32, 32), (16, 16, 16), (64, 64)])
@pytest.mark.parametrize("dtype", _TEST_DTYPES)
def test_linalg_lu_factor_ex_singular(shape, dtype):
    """Test that singular matrices produce non-zero info."""
    inp = _make_singular_input(shape, flag_gems.device, dtype)
    ref_inp = utils.to_reference(inp)

    if flag_gems.vendor_name != "ascend":
        ref_out = torch.linalg.lu_factor_ex(ref_inp, pivot=True, check_errors=False)
    else:
        ref_lu, ref_pivots, ref_info = _run_torch_ops_path_ex(ref_inp, pivot=True)
        ref_out = namedtuple("_RefResult", ["LU", "pivots", "info"])(
            ref_lu, ref_pivots, ref_info
        )
    with flag_gems.use_gems():
        res_out = torch.linalg.lu_factor_ex(inp, pivot=True, check_errors=False)

    # The last diagonal element should be zero, so info should indicate the position
    utils.gems_assert_equal(res_out.info, ref_out.info)


@pytest.mark.linalg_lu_factor_ex
@pytest.mark.parametrize("shape", [(4, 4), (32, 32), (16, 16, 16)])
@pytest.mark.parametrize("dtype", _TEST_DTYPES)
def test_linalg_lu_factor_ex_check_errors_raises(shape, dtype):
    """Test that check_errors=True raises RuntimeError for singular matrices."""
    inp = _make_singular_input(shape, flag_gems.device, dtype)

    with flag_gems.use_gems():
        with pytest.raises(RuntimeError, match="lu_factor_ex"):
            torch.linalg.lu_factor_ex(inp, pivot=True, check_errors=True)


@pytest.mark.linalg_lu_factor_ex_out
@pytest.mark.parametrize("shape", _LU_FACTOR_EX_SHAPES)
@pytest.mark.parametrize("dtype", _TEST_DTYPES)
@pytest.mark.parametrize("pivot", _PIVOT_VALUES)
def test_linalg_lu_factor_ex_out(shape, dtype, pivot):
    """Test the out= parameter variant."""

    inp = _make_input(shape, pivot, flag_gems.device, dtype)
    ref_inp = utils.to_reference(inp)

    batch_shape = inp.shape[:-2]
    m, n = inp.shape[-2], inp.shape[-1]
    k = min(m, n)

    # Reference: use out= parameter (or manual reference on Ascend)
    ref_LU_out = torch.empty_like(ref_inp)
    ref_pivots_out = torch.empty(
        (*batch_shape, k), dtype=torch.int32, device=ref_inp.device
    )
    ref_info_out = torch.empty(batch_shape, dtype=torch.int32, device=ref_inp.device)
    if flag_gems.vendor_name != "ascend":
        ref_out = torch.linalg.lu_factor_ex(
            ref_inp, pivot=pivot, out=(ref_LU_out, ref_pivots_out, ref_info_out)
        )
    else:
        ref_lu, ref_pivots, ref_info = _run_torch_ops_path_ex(ref_inp, pivot=pivot)
        ref_LU_out.copy_(ref_lu)
        ref_pivots_out.copy_(ref_pivots)
        ref_info_out.copy_(ref_info)
        ref_out = namedtuple("_RefResult", ["LU", "pivots", "info"])(
            ref_LU_out, ref_pivots_out, ref_info_out
        )

    # Gems: use out= parameter
    res_LU_out = torch.empty_like(inp)
    res_pivots_out = torch.empty(
        (*batch_shape, k), dtype=torch.int32, device=inp.device
    )
    res_info_out = torch.empty(batch_shape, dtype=torch.int32, device=inp.device)
    out = (res_LU_out, res_pivots_out, res_info_out)
    with flag_gems.use_gems():
        res_out = torch.linalg.lu_factor_ex(inp, pivot=pivot, out=out)

    # Verify outputs are the same objects (in-place write)
    assert res_out.LU is res_LU_out
    assert res_out.pivots is res_pivots_out
    assert res_out.info is res_info_out

    # Verify shapes
    assert res_out.LU.shape == inp.shape
    assert res_out.pivots.dtype == torch.int32
    assert res_out.pivots.shape == (*batch_shape, k)
    assert res_out.info.shape == batch_shape

    # Verify accuracy via reconstructed product P @ L @ U (more robust than
    # comparing LU factors directly, which can differ due to pivot choices).
    torch.backends.cuda.matmul.allow_tf32 = False
    if pivot:
        res_p, res_l, res_u = torch.lu_unpack(res_out.LU, res_out.pivots)
        ref_p, ref_l, ref_u = torch.lu_unpack(ref_out.LU, ref_out.pivots)
        reconstructed = res_p @ res_l @ res_u
        ref_reconstructed = ref_p @ ref_l @ ref_u
    else:
        res_l, res_u = _unpack_lu_no_pivot(res_out.LU)
        ref_l, ref_u = _unpack_lu_no_pivot(ref_out.LU)
        reconstructed = res_l @ res_u
        ref_reconstructed = ref_l @ ref_u
    utils.gems_assert_close(reconstructed, ref_reconstructed, dtype, reduce_dim=k)
