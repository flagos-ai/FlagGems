import pytest
import torch

torch.backends.cuda.matmul.allow_tf32 = False

import flag_gems  # noqa: E402

from . import accuracy_utils as utils  # noqa: E402

IS_ASCEND = flag_gems.vendor_name == "ascend"

DTYPES = [
    torch.float32,
]
if flag_gems.runtime.device.support_fp64 and not IS_ASCEND:
    # On ascend fp64 is not reliably supported (torch_npu casts double to float).
    DTYPES.append(torch.float64)


def _make_triangular(shape, dtype, device, upper, unitriangular):
    n = shape[-1]
    if len(shape) == 2:
        A = torch.randn(shape, dtype=dtype, device=device)
    else:
        batch_shape = shape[:-2]
        A = torch.randn(batch_shape + (n, n), dtype=dtype, device=device)

    off_diag = 0.1
    if upper:
        A = A.triu(diagonal=1)
    else:
        A = A.tril(diagonal=-1)
    A.mul_(off_diag)

    eye = torch.eye(n, dtype=dtype, device=device)
    batch_dims = [1] * (A.ndim - 2)
    if batch_dims:
        eye = eye.view(*batch_dims, n, n)
    A.add_(eye)

    if unitriangular:
        A.diagonal(0, -2, -1).fill_(1.0)

    return A


def _solve_tri_small_ops(A, B, upper=False, left=True, unitriangular=False):
    """Block forward/backward substitution built from small torch ops
    (matmul + inverse), running on AI Core.

    On ascend, torch_npu's solve_triangular runs on AI_CPU and is not a
    meaningful AI Core reference; this small-op combination is used as both the
    functional reference and the benchmark baseline.  test_baseline_matches_torch
    below proves it agrees with torch.linalg.solve_triangular.
    """
    n = A.shape[-1]
    bs = 64
    X = B.clone()
    if not left:
        # X A = B  <=>  A^T X^T = B^T  (reduce to left-multiply, transpose back)
        return _solve_tri_small_ops(
            A.mT.contiguous(),
            B.mT.contiguous(),
            not upper,
            True,
            unitriangular,
        ).mT.contiguous()
    if upper:
        for i in range(n - 1, -1, -bs):
            i0 = max(0, i - bs + 1)
            i1 = i + 1
            Aii = A[..., i0:i1, i0:i1]
            rhs = B[..., i0:i1, :]
            if i1 < n:
                rhs = rhs - torch.matmul(A[..., i0:i1, i1:], X[..., i1:, :])
            X[..., i0:i1, :] = torch.matmul(torch.linalg.inv(Aii), rhs)
    else:
        for i in range(0, n, bs):
            i1 = min(i + bs, n)
            Aii = A[..., i:i1, i:i1]
            rhs = B[..., i:i1, :]
            if i > 0:
                rhs = rhs - torch.matmul(A[..., i:i1, :i], X[..., :i, :])
            X[..., i:i1, :] = torch.matmul(torch.linalg.inv(Aii), rhs)
    return X


def _ref_solve_tri(A, B, **kwargs):
    """Correctness reference.  On ascend use the small-op combination (AI Core)
    because torch_npu's solve_triangular runs on AI_CPU; elsewhere use the
    torch reference."""
    if IS_ASCEND:
        return _solve_tri_small_ops(A, B, **kwargs)
    ref_A = utils.to_reference(A)
    ref_B = utils.to_reference(B)
    return torch.linalg.solve_triangular(ref_A, ref_B, **kwargs)


@pytest.mark.linalg_solve_triangular
@pytest.mark.parametrize("n", [1, 4, 8, 16, 32, 64, 128, 256, 512])
@pytest.mark.parametrize("k", [1, 3, 16])
@pytest.mark.parametrize("dtype", DTYPES)
def test_lower_left(n, k, dtype):
    A = _make_triangular(
        (n, n), dtype, flag_gems.device, upper=False, unitriangular=False
    )
    B = torch.randn(n, k, dtype=dtype, device=flag_gems.device)

    ref_out = _ref_solve_tri(A, B, upper=False)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.linalg_solve_triangular(A, B, upper=False)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_solve_triangular
@pytest.mark.parametrize("n", [1, 4, 8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("k", [1, 3, 16])
@pytest.mark.parametrize("dtype", DTYPES)
def test_upper_left(n, k, dtype):
    A = _make_triangular(
        (n, n), dtype, flag_gems.device, upper=True, unitriangular=False
    )
    B = torch.randn(n, k, dtype=dtype, device=flag_gems.device)

    ref_out = _ref_solve_tri(A, B, upper=True)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.linalg_solve_triangular(A, B, upper=True)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_solve_triangular
@pytest.mark.parametrize("n", [4, 16, 64, 128])
@pytest.mark.parametrize("k", [1, 8])
@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_right(n, k, upper, dtype):
    A = _make_triangular(
        (k, k), dtype, flag_gems.device, upper=upper, unitriangular=False
    )
    B = torch.randn(n, k, dtype=dtype, device=flag_gems.device)

    ref_out = _ref_solve_tri(A, B, upper=upper, left=False)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.linalg_solve_triangular(A, B, upper=upper, left=False)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_solve_triangular
@pytest.mark.parametrize("n", [4, 16, 64, 128])
@pytest.mark.parametrize("k", [1, 8])
@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_unitriangular(n, k, upper, dtype):
    A = _make_triangular(
        (n, n), dtype, flag_gems.device, upper=upper, unitriangular=True
    )
    B = torch.randn(n, k, dtype=dtype, device=flag_gems.device)

    ref_out = _ref_solve_tri(A, B, upper=upper, unitriangular=True)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.linalg_solve_triangular(
            A, B, upper=upper, unitriangular=True
        )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_solve_triangular
@pytest.mark.parametrize("batch_shape", [(3,), (2, 4)])
@pytest.mark.parametrize("n", [8, 32])
@pytest.mark.parametrize("k", [1, 4])
@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_batched(batch_shape, n, k, upper, dtype):
    shape_A = batch_shape + (n, n)
    shape_B = batch_shape + (n, k)
    A = _make_triangular(
        shape_A, dtype, flag_gems.device, upper=upper, unitriangular=False
    )
    B = torch.randn(shape_B, dtype=dtype, device=flag_gems.device)

    ref_out = _ref_solve_tri(A, B, upper=upper)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.linalg_solve_triangular(A, B, upper=upper)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_solve_triangular_out
@pytest.mark.parametrize("n", [16, 64, 128])
@pytest.mark.parametrize("k", [1, 8])
@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_out_kwarg(n, k, upper, dtype):
    A = _make_triangular(
        (n, n), dtype, flag_gems.device, upper=upper, unitriangular=False
    )
    B = torch.randn(n, k, dtype=dtype, device=flag_gems.device)
    out = torch.empty_like(B)

    ref_out = _ref_solve_tri(A, B, upper=upper)

    with flag_gems.use_gems():
        res_out = torch.linalg.solve_triangular(A, B, upper=upper, out=out)

    assert res_out is out
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_solve_triangular_out
@pytest.mark.parametrize("n", [16, 64, 128])
@pytest.mark.parametrize("k", [1, 8])
@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_linalg_solve_triangular_out(n, k, upper, dtype):
    A = _make_triangular(
        (n, n), dtype, flag_gems.device, upper=upper, unitriangular=False
    )
    B = torch.randn(n, k, dtype=dtype, device=flag_gems.device)
    out = torch.empty_like(B)

    ref_out = _ref_solve_tri(A, B, upper=upper)

    with flag_gems.use_gems():
        res_out = torch.linalg.solve_triangular(A, B, upper=upper, out=out)

    assert res_out is out
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_solve_triangular
@pytest.mark.parametrize("n", [16, 64, 128, 256])
@pytest.mark.parametrize("k", [1, 8])
@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.skipif(
    not flag_gems.runtime.device.support_fp64, reason="fp64 is not supported."
)
def test_residual_f64(n, k, upper):
    """Residual check (float64 for precision)"""
    dtype = torch.float64
    A = _make_triangular(
        (n, n), dtype, flag_gems.device, upper=upper, unitriangular=False
    )
    B = torch.randn(n, k, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.linalg_solve_triangular(A, B, upper=upper)

    residual = (A @ res_out - B).abs().max().item()
    assert residual < 1e-6, f"Residual too large: {residual}"


@pytest.mark.linalg_solve_triangular
@pytest.mark.parametrize("dtype", [torch.float32])
def test_empty(dtype):
    A = torch.empty(0, 0, dtype=dtype, device=flag_gems.device)
    B = torch.empty(0, 0, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.linalg_solve_triangular(A, B, upper=False)

    assert res_out.shape == (0, 0)
    assert res_out.dtype == dtype


_LARGE_K = [1, 8]
if IS_ASCEND:
    # wide-RHS coverage: the ascend backend had k-dependent corruption bugs
    # (fixed in v0.2); keep these shapes covered on ascend.
    _LARGE_K += [64, 256, 512, 1024]


@pytest.mark.linalg_solve_triangular
@pytest.mark.parametrize("n", [64, 128, 256, 512, 1024])
@pytest.mark.parametrize("k", _LARGE_K)
@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_large_n_f64(n, k, upper, dtype):
    """Large matrix tests - covering all three kernel dispatch paths"""
    A = _make_triangular(
        (n, n), dtype, flag_gems.device, upper=upper, unitriangular=False
    )
    B = torch.randn(n, k, dtype=dtype, device=flag_gems.device)

    ref_out = _ref_solve_tri(A, B, upper=upper)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.linalg_solve_triangular(A, B, upper=upper)

    atol = 1e-4
    if n >= 1024 and dtype == torch.float32:
        # fp32 accumulated-precision physical limit (measured 2026-08-03, vs fp64 reference):
        # our error is on par with torch (ratio 0.45-0.99, residual usually slightly better),
        # n=1024 diff ~1.9-3.2e-4. Use a static tolerance of 1e-3 (3-5x margin) instead of
        # anchoring to the runtime torch GPU/CPU difference: in quick-cpu mode (--ref=cpu)
        # the reference is the CPU torch solve, so the dynamic anchor (GPU vs CPU) collapses to 0.
        atol = 1e-3

    utils.gems_assert_close(res_out, ref_out, dtype, atol=atol)


@pytest.mark.linalg_solve_triangular
@pytest.mark.parametrize("n", [8, 32, 128, 512, 600])
@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_no_tle_fallback(n, upper, dtype, monkeypatch):
    """Non-TLE fallback smoke tests: force HAS_TLE=False to exercise pure-Triton fallback kernels."""
    import importlib

    import flag_gems.ops  # noqa: F401

    solve_mod = importlib.import_module("flag_gems.ops.linalg_solve_triangular")

    monkeypatch.setattr(solve_mod, "HAS_TLE", False)

    A = _make_triangular(
        (n, n), dtype, flag_gems.device, upper=upper, unitriangular=False
    )
    B = torch.randn(n, n, dtype=dtype, device=flag_gems.device)

    ref_out = _ref_solve_tri(A, B, upper=upper)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.linalg_solve_triangular(A, B, upper=upper)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_solve_triangular
@pytest.mark.parametrize("n", [16, 64, 128, 256, 512])
@pytest.mark.parametrize("k", [1, 8, 64, 256])
@pytest.mark.parametrize("upper", [False, True])
def test_baseline_matches_torch(n, k, upper):
    """Validate the small-op baseline used for benchmarking.

    The small-op combination (block forward/backward substitution via matmul and
    inverse, running on AI Core) must agree with torch.linalg.solve_triangular,
    so it can serve as the benchmark baseline on platforms where the torch
    reference runs on a different execution unit (e.g. ascend AI_CPU).
    """
    dtype = torch.float32
    A = _make_triangular(
        (n, n), dtype, flag_gems.device, upper=upper, unitriangular=False
    )
    B = torch.randn(n, k, dtype=dtype, device=flag_gems.device)

    res_small = _solve_tri_small_ops(A, B, upper=upper)
    ref_A = utils.to_reference(A)
    ref_B = utils.to_reference(B)
    res_torch = torch.linalg.solve_triangular(ref_A, ref_B, upper=upper)

    utils.gems_assert_close(res_small, res_torch, dtype)
