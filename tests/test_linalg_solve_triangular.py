import pytest
import torch

torch.backends.cuda.matmul.allow_tf32 = False

import flag_gems  # noqa: E402
from flag_gems.utils import get_device_properties  # noqa: E402

from . import accuracy_utils as utils  # noqa: E402

IS_ASCEND = flag_gems.vendor_name == "ascend"
IS_THEAD = flag_gems.vendor_name == "thead"

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
    because torch_npu's solve_triangular runs on AI_CPU; on thead/PPU solve in
    fp64 on CPU because the device fp32 trsm there carries ~2-3e-3 error vs the
    fp64 truth at n=1024 (measured 2026-09-16, see repro_solve_tri_ppu.py),
    which dwarfs the kernel's own ~3-6e-4 error and spuriously fails the test.
    Elsewhere use the torch reference."""
    if IS_THEAD:
        # Solve in fp64 on CPU (LAPACK, accurate), then place the reference
        # where the comparison machinery expects it: on the device in normal
        # mode (assert_close checks device), on CPU in --ref=cpu quick mode
        # (to_cpu asserts the ref is already CPU).
        ref = torch.linalg.solve_triangular(
            A.double().cpu(), B.double().cpu(), **kwargs
        )
        return ref if utils.TO_CPU else ref.to(A.device)
    ref_A = utils.to_reference(A)
    ref_B = utils.to_reference(B)
    if IS_ASCEND:
        return _solve_tri_small_ops(ref_A, ref_B, **kwargs)
    return torch.linalg.solve_triangular(ref_A, ref_B, **kwargs)


def _grid_programs(batch_shape, k):
    """Programs the diagonal-block kernel launches for one block.

    One work item per (column slice, batch); the column slice is SLIDE_SIZE=64
    wide, matching the backend's SLIDE_SIZE.
    """
    batch = 1
    for s in batch_shape:
        batch *= s
    return ((k + 63) // 64) * batch


def _core_count():
    try:
        return int(get_device_properties().multi_processor_count)
    except Exception:  # noqa: BLE001
        return 0


# The diagonal-block kernel solves its rows serially and orders the
# store -> load read-after-write with tl.debug_barrier().  That ordering only
# holds while every program has its own core and runs to completion; a grid
# larger than the core count is scheduled in waves and a later row's load reads
# X before the earlier row's store has landed.  Measured on Ascend910B4 (40
# vector cores): grid 40 is bit-stable, grid 41 is not, and the bad runs differ
# from each other.  The backend folds the surplus work items into an in-kernel
# loop, so the grid stays at or below the core count; these shapes pin that,
# each reaching the grid along a different axis.
_OVER_CORE_CASES = [
    ((128,), 64, 4),  # grid 128, the batch axis alone exceeds the cores
    ((8,), 64, 512),  # grid 64, the column-slice axis does
    ((1,), 64, 4096),  # grid 64, reachable with no batch dimension at all
    ((64,), 128, 128),  # grid 128, multi-block, so the update loop runs too
    # The shape from the original report (a/b=[4,32,64,64], grid 128).  Kept
    # separately from ((128,), 64, 4) even though both reach grid 128: here k
    # equals SLIDE_SIZE, so every lane of the column slice is active, whereas
    # k=4 masks all but four of them.
    ((4, 32), 64, 64),
]

_CORES = _core_count()
_MAX_CASE_GRID = max(_grid_programs(b, k) for b, _, k in _OVER_CORE_CASES)


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

    res_out = flag_gems.linalg_solve_triangular(A, B, upper=False)

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

    res_out = flag_gems.linalg_solve_triangular(A, B, upper=True)

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

    res_out = flag_gems.linalg_solve_triangular(A, B, upper=upper, left=False)

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

    res_out = flag_gems.linalg_solve_triangular(A, B, upper=upper, unitriangular=True)

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

    res_out = flag_gems.linalg_solve_triangular(A, B, upper=upper)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_solve_triangular
@pytest.mark.parametrize("batch_shape", [(3,), (2, 4)])
@pytest.mark.parametrize("n", [128, 192])
@pytest.mark.parametrize("k", [4, 64])
@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_batched_multiblock(batch_shape, n, k, upper, dtype):
    """Batched with n > BLOCK_SIZE, so the per-batch update loop between
    diagonal blocks actually runs.

    test_batched above tops out at n=32, which is a single diagonal block, so
    the batched multi-block path -- the mm update and the in-place subtract for
    every batch -- was never exercised.  The batch is kept small here so the
    launch grid stays inside the core count and this stays a test of the
    multi-block path rather than of the grid cap.
    """
    shape_A = batch_shape + (n, n)
    shape_B = batch_shape + (n, k)
    A = _make_triangular(
        shape_A, dtype, flag_gems.device, upper=upper, unitriangular=False
    )
    B = torch.randn(shape_B, dtype=dtype, device=flag_gems.device)

    ref_out = _ref_solve_tri(A, B, upper=upper)

    res_out = flag_gems.linalg_solve_triangular(A, B, upper=upper)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_solve_triangular
@pytest.mark.parametrize("batch_shape,n,k", _OVER_CORE_CASES)
@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.skipif(
    _MAX_CASE_GRID <= _CORES,
    reason="device has more cores than the largest case grid; nothing to pin.",
)
def test_grid_above_core_count(batch_shape, n, k, upper):
    """The solve must stay exact and reproducible with a grid past the cores.

    Regression test for the diagonal-block kernel reading an X row whose store
    had not landed: it returned wrong values, differently on every run, once the
    launch grid exceeded the core count.  Both halves matter -- the residual
    catches a consistently wrong answer, the repeat check catches the
    run-dependent one, which a single pass could miss.
    """
    dtype = torch.float32
    A = _make_triangular(
        batch_shape + (n, n), dtype, flag_gems.device, upper=upper, unitriangular=False
    )
    B = torch.randn(batch_shape + (n, k), dtype=dtype, device=flag_gems.device)

    runs = [
        flag_gems.linalg_solve_triangular(A, B, upper=upper).cpu() for _ in range(4)
    ]
    for i, other in enumerate(runs[1:], start=2):
        assert torch.equal(
            runs[0], other
        ), f"solve_triangular is not deterministic: run {i} differs from run 1"

    X = runs[0]
    Ad = A.cpu().double()
    Bd = B.cpu().double()
    ref = torch.linalg.solve_triangular(Ad, Bd, upper=upper)
    assert torch.allclose(
        X.double(), ref, atol=1e-3, rtol=1e-3
    ), f"max |X - X_ref| = {(X.double() - ref).abs().max().item()}"

    residual = (Ad @ X.double() - Bd).abs().max().item()
    assert residual < 1e-3, f"residual too large: {residual}"


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

    res_out = flag_gems.linalg_solve_triangular_out(A, B, upper=upper, out=out)

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

    res_out = flag_gems.linalg_solve_triangular_out(A, B, upper=upper, out=out)

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

    res_out = flag_gems.linalg_solve_triangular(A, B, upper=upper)

    residual = (A @ res_out - B).abs().max().item()
    assert residual < 1e-6, f"Residual too large: {residual}"


@pytest.mark.linalg_solve_triangular
@pytest.mark.parametrize("dtype", [torch.float32])
def test_empty(dtype):
    A = torch.empty(0, 0, dtype=dtype, device=flag_gems.device)
    B = torch.empty(0, 0, dtype=dtype, device=flag_gems.device)

    res_out = flag_gems.linalg_solve_triangular(A, B, upper=False)

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

    res_out = flag_gems.linalg_solve_triangular(A, B, upper=upper)

    atol = 1e-4
    if n >= 1024 and dtype == torch.float32:
        # fp32 accumulated-precision physical limit (measured 2026-08-03, vs fp64 reference):
        # our error is on par with torch (ratio 0.45-0.99, residual usually slightly better),
        # n=1024 diff ~1.9-3.2e-4. Use a static tolerance instead of anchoring to the
        # runtime torch GPU/CPU difference: in quick-cpu mode (--ref=cpu) the reference is
        # the CPU solve, so the dynamic anchor (GPU vs CPU) collapses to 0. On the CPU
        # reference path extra fp32 rounding was measured up to ~1.1e-3 (2026-09-08),
        # hence 2e-3 with ~2x margin.
        atol = 2e-3

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

    res_out = flag_gems.linalg_solve_triangular(A, B, upper=upper)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_solve_triangular
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="Verify the correctness of the PyTorch combination implementation path, "
    "and perform verification only on the Ascend platform.",
)
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
