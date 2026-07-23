import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# gels fast path covers overdetermined (m >= n) float32. Shapes are (m, n).
_SHAPES = [(16, 4), (64, 8), (128, 16), (200, 8), (256, 32)]


@pytest.mark.linalg_lstsq
def test_linalg_lstsq_gems_path_active():
    # Guard: the accuracy tests below call torch.ops.aten.linalg_lstsq under
    # use_gems and compare to a torch reference. If the op is NOT registered,
    # that call silently runs torch's native cuSOLVER and every test passes as
    # torch-vs-torch, exercising nothing. This asserts the gems override is
    # actually installed so that can never hide again.
    with flag_gems.use_gems():
        keys = flag_gems.all_registered_keys()
    assert "linalg_lstsq" in keys, (
        "linalg_lstsq is not registered — the gems kernel is not being "
        "exercised; the other tests would validate torch against itself."
    )


def _ref_and_gems(A, b, dtype):
    """Reference via CPU gels; gems via the aten override under use_gems."""
    ref_A = utils.to_reference(A)
    ref_b = utils.to_reference(b)
    ref = torch.linalg.lstsq(ref_A, ref_b, driver="gels")

    with flag_gems.use_gems():
        res = torch.ops.aten.linalg_lstsq(A, b, driver="gels")
    return ref, res


@pytest.mark.linalg_lstsq
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_lstsq_vector(shape, dtype):
    m, n = shape
    A = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    b = torch.randn(m, dtype=dtype, device=flag_gems.device)

    ref, res = _ref_and_gems(A, b, dtype)
    utils.gems_assert_close(res[0], ref.solution, dtype)
    # residuals available since m > n
    utils.gems_assert_close(res[1], ref.residuals, dtype)


@pytest.mark.linalg_lstsq
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("nrhs", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_lstsq_matrix(shape, nrhs, dtype):
    m, n = shape
    A = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    b = torch.randn(m, nrhs, dtype=dtype, device=flag_gems.device)

    ref, res = _ref_and_gems(A, b, dtype)
    utils.gems_assert_close(res[0], ref.solution, dtype)
    utils.gems_assert_close(res[1], ref.residuals, dtype)


@pytest.mark.linalg_lstsq
@pytest.mark.parametrize("batch_shape", [(2,), (3,), (2, 3)])
@pytest.mark.parametrize("shape", [(64, 8), (128, 16)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_lstsq_batched(batch_shape, shape, dtype):
    m, n = shape
    A = torch.randn(*batch_shape, m, n, dtype=dtype, device=flag_gems.device)
    b = torch.randn(*batch_shape, m, 2, dtype=dtype, device=flag_gems.device)

    ref, res = _ref_and_gems(A, b, dtype)
    utils.gems_assert_close(res[0], ref.solution, dtype)
    utils.gems_assert_close(res[1], ref.residuals, dtype)


@pytest.mark.linalg_lstsq
# native while next_pow2(n)*next_pow2(m) <= 32768: incl. wide n and larger m
@pytest.mark.parametrize(
    "shape",
    [
        (8, 32),
        (16, 128),
        (4, 256),
        (16, 64),
        (16, 512),
        (64, 256),
        (16, 1024),
        # area == 32768, exactly at the single-tile budget ceiling: the kernel
        # holds two BLOCK_R x BLOCK_M tiles here, so this is the largest
        # allocation the wide path can ever request. Pins the ceiling so a
        # smaller-SRAM backend fails loudly in CI rather than for a user.
        (64, 512),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_lstsq_underdetermined(shape, dtype):
    # m < n -> native minimum-norm path (QR of A^T). residuals are empty here,
    # so only the solution is compared. Reference gels handles m<n on device.
    m, n = shape
    A = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    b = torch.randn(m, dtype=dtype, device=flag_gems.device)

    ref, res = _ref_and_gems(A, b, dtype)
    utils.gems_assert_close(res[0], ref.solution, dtype)


@pytest.mark.linalg_lstsq
@pytest.mark.parametrize(
    "shape,nrhs", [((64, 1024), 1), ((128, 512), 1), ((32, 2048), 1), ((64, 1024), 3)]
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_lstsq_underdetermined_blocked(shape, nrhs, dtype):
    # beyond the wide tile budget (>32768) -> blocked TSQR of A^T (no Q),
    # NATIVE min-norm, no fallback.
    m, n = shape
    A = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    b = (
        torch.randn(m, dtype=dtype, device=flag_gems.device)
        if nrhs == 1
        else torch.randn(m, nrhs, dtype=dtype, device=flag_gems.device)
    )
    ref, res = _ref_and_gems(A, b, dtype)
    utils.gems_assert_close(res[0], ref.solution, dtype)


@pytest.mark.linalg_lstsq
@pytest.mark.parametrize("dtype", [torch.float64])
def test_linalg_lstsq_underdetermined_blocked_fp64(dtype):
    # fp64 blocked wide: tile 512*64=32768 == fp64... use (48,512): >16384 budget
    m, n = 48, 512
    A = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    b = torch.randn(m, dtype=dtype, device=flag_gems.device)
    ref = torch.linalg.lstsq(A.cpu(), b.cpu(), driver="gelsd")
    # gelsd is CPU-only, so the reference is computed on CPU; to_reference then
    # places it per the active --ref mode (it must stay on CPU under --ref=cpu,
    # where gems_assert_close asserts the reference lives on CPU).
    ref_sol = utils.to_reference(ref.solution.to(flag_gems.device))
    with flag_gems.use_gems():
        res = torch.ops.aten.linalg_lstsq(A, b)
    utils.gems_assert_close(res[0], ref_sol, dtype)


@pytest.mark.linalg_lstsq
@pytest.mark.parametrize("nrhs", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_lstsq_underdetermined_matrix(nrhs, dtype):
    m, n = 16, 64
    A = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    b = torch.randn(m, nrhs, dtype=dtype, device=flag_gems.device)

    ref, res = _ref_and_gems(A, b, dtype)
    utils.gems_assert_close(res[0], ref.solution, dtype)


@pytest.mark.linalg_lstsq
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_lstsq_broadcast(dtype):
    # Only the MATRIX rhs broadcasts its batch dims against A's (torch requires
    # A.dim()-b.dim() in {0,1}, and the vector rhs is exact-match only).
    dev = flag_gems.device
    m, n = 64, 8

    # matrix rhs, equal ndim, broadcast batch: A (2,1) x b (1,3) -> (2,3), nrhs=2
    A = torch.randn(2, 1, m, n, dtype=dtype, device=dev)
    b = torch.randn(1, 3, m, 2, dtype=dtype, device=dev)
    ref, res = _ref_and_gems(A, b, dtype)
    utils.gems_assert_close(res[0], ref.solution, dtype)

    # matrix rhs, 3-D batch broadcast: A (2,1,4) x b (1,3,1) -> (2,3,4), nrhs=2
    A = torch.randn(2, 1, 4, m, n, dtype=dtype, device=dev)
    b = torch.randn(1, 3, 1, m, 2, dtype=dtype, device=dev)
    ref, res = _ref_and_gems(A, b, dtype)
    utils.gems_assert_close(res[0], ref.solution, dtype)

    # batched VECTOR rhs, exact batch match (no broadcast): A (2,3) x b (2,3,m)
    A = torch.randn(2, 3, m, n, dtype=dtype, device=dev)
    b = torch.randn(2, 3, m, dtype=dtype, device=dev)
    ref, res = _ref_and_gems(A, b, dtype)
    utils.gems_assert_close(res[0], ref.solution, dtype)


@pytest.mark.linalg_lstsq
@pytest.mark.parametrize("shape", [(512, 96), (2048, 120)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_lstsq_tall_monolithic_edge(shape, dtype):
    # NC <= 128: monolithic dynamic-loop path (the fast path near its ceiling).
    m, n = shape
    A = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    b = torch.randn(m, dtype=dtype, device=flag_gems.device)

    ref, res = _ref_and_gems(A, b, dtype)
    utils.gems_assert_close(res[0], ref.solution, dtype)


@pytest.mark.linalg_lstsq
@pytest.mark.parametrize(
    "shape,nrhs", [((512, 200), 1), ((2048, 300), 1), ((4096, 160), 2), ((256, 250), 1)]
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_lstsq_tall_blocked(shape, nrhs, dtype):
    # NC > 128: blocked TSQR path (no register spill), NATIVE — no fallback.
    # Covers multi-chunk (m>>block_m) and near-square. Residuals checked too.
    m, n = shape
    A = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    b = (
        torch.randn(m, dtype=dtype, device=flag_gems.device)
        if nrhs == 1
        else torch.randn(m, nrhs, dtype=dtype, device=flag_gems.device)
    )
    ref, res = _ref_and_gems(A, b, dtype)
    utils.gems_assert_close(res[0], ref.solution, dtype)
    # residuals are a length-m reduction of O(1) squares: scale atol by m (the
    # default 1e-4 atol + fp32 rtol is borderline-flaky at m=4096, both sides
    # accumulating in fp32 along different orders).
    utils.gems_assert_close(res[1], ref.residuals, dtype, reduce_dim=m)


@pytest.mark.linalg_lstsq
@pytest.mark.parametrize("dtype", [torch.float64])
def test_linalg_lstsq_tall_blocked_fp64(dtype):
    # fp64 blocked path (fp64 tall ALWAYS routes to blocked kernels: measured
    # 3.5-10x slower monolithic at every NC, SMEM exhaustion at NC>=129).
    m, n = 256, 80
    A = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    b = torch.randn(m, dtype=dtype, device=flag_gems.device)
    ref = torch.linalg.lstsq(A.cpu(), b.cpu(), driver="gelsd")
    # gelsd is CPU-only, so the reference is computed on CPU; to_reference then
    # places it per the active --ref mode (it must stay on CPU under --ref=cpu,
    # where gems_assert_close asserts the reference lives on CPU).
    ref_sol = utils.to_reference(ref.solution.to(flag_gems.device))
    with flag_gems.use_gems():
        res = torch.ops.aten.linalg_lstsq(A, b)
    utils.gems_assert_close(res[0], ref_sol, dtype)


@pytest.mark.linalg_lstsq
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_lstsq_rank_deficient(dtype):
    # A zero column makes r_ii exactly 0, which deterministically trips the rank
    # guard -> NaN in the affected solution rows. (A merely near-dependent column
    # is threshold-sensitive under fp32 and not a reliable guard test; gels is
    # documented as undefined on rank-deficient input either way.)
    m, n = 128, 6
    A = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    A[:, 3] = 0.0
    b = torch.randn(m, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res = torch.ops.aten.linalg_lstsq(A, b, driver="gels")
    assert torch.isnan(res[0]).any(), "rank-deficient A should yield NaN solution"


@pytest.mark.linalg_lstsq
@pytest.mark.parametrize("shape", [(64, 8), (16, 64)])  # tall and wide
@pytest.mark.parametrize("dtype", [torch.float64])
def test_linalg_lstsq_fp64(shape, dtype):
    # float64 is now a NATIVE path (not a fallback). gelsd is CPU-only, so the
    # reference is computed on CPU and moved to device (harness compares on-device).
    m, n = shape
    A = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    b = torch.randn(m, dtype=dtype, device=flag_gems.device)

    ref = torch.linalg.lstsq(A.cpu(), b.cpu(), driver="gelsd")
    # gelsd is CPU-only, so the reference is computed on CPU; to_reference then
    # places it per the active --ref mode (it must stay on CPU under --ref=cpu,
    # where gems_assert_close asserts the reference lives on CPU).
    ref_sol = utils.to_reference(ref.solution.to(flag_gems.device))
    with flag_gems.use_gems():
        res = torch.ops.aten.linalg_lstsq(A, b)
    utils.gems_assert_close(res[0], ref_sol, dtype)


@pytest.mark.linalg_lstsq
def test_linalg_lstsq_driver_rejected():
    # torch's CUDA gels backend rejects non-gels drivers; we must raise likewise
    # (not silently fall back and compute).
    A = torch.randn(64, 8, dtype=torch.float32, device=flag_gems.device)
    b = torch.randn(64, dtype=torch.float32, device=flag_gems.device)
    with flag_gems.use_gems():
        with pytest.raises(RuntimeError):
            torch.ops.aten.linalg_lstsq(A, b, driver="gelsd")


@pytest.mark.linalg_lstsq
@pytest.mark.parametrize(
    "batch,m,n,nrhs",
    [
        ((), 8, 4, 0),  # nrhs == 0, m > n: residuals shape (0,)
        ((), 8, 0, 1),  # n == 0, vector b: solution (0,), residuals (1,) ZEROS
        ((), 0, 4, 1),  # m == 0: solution (4,) ZEROS, residuals empty(0)
        ((), 0, 0, 1),  # m == n == 0: solution (0,), residuals empty(0)
        ((2, 3), 8, 0, 2),  # batched n == 0 matrix rhs: residuals (2, 3, 2) ZEROS
        ((2,), 0, 4, 2),  # batched m == 0: solution (2, 4, 2) zeros
    ],
)
def test_linalg_lstsq_degenerate(batch, m, n, nrhs):
    # degenerate dims (m/n/nrhs == 0) are handled NATIVELY (no kernel, no
    # fallback) and must match torch on solution VALUES and residuals shape AND
    # values, plus empty rank/sv. LAPACK gels quick-returns on any zero dim and
    # ZEROES its buffer, so both solution and residuals are all-zeros here.
    dev = flag_gems.device
    A = torch.randn(*batch, m, n, dtype=torch.float32, device=dev)
    vector = nrhs == 1 and not batch
    b = (
        torch.randn(*batch, m, dtype=torch.float32, device=dev)
        if vector
        else torch.randn(*batch, m, nrhs, dtype=torch.float32, device=dev)
    )
    try:
        ref = torch.linalg.lstsq(A.cpu(), b.cpu(), driver="gels")
    except RuntimeError:
        # torch itself rejects this shape -> we must reject it too
        with flag_gems.use_gems():
            with pytest.raises(RuntimeError):
                torch.ops.aten.linalg_lstsq(A, b, driver="gels")
        return
    with flag_gems.use_gems():
        res = torch.ops.aten.linalg_lstsq(A, b, driver="gels")
    assert res[0].device.type == A.device.type
    assert res[0].shape == ref.solution.shape
    torch.testing.assert_close(res[0].cpu(), ref.solution)  # zeros when m==0
    assert res[1].shape == ref.residuals.shape
    torch.testing.assert_close(res[1].cpu(), ref.residuals)  # zeros when n==0
    assert res[2].numel() == 0 and res[3].numel() == 0  # gels: empty


@pytest.mark.linalg_lstsq
def test_linalg_lstsq_complex_fallback():
    # complex is outside the native real-only path -> must fall back (not crash)
    # and still be correct. Manual compare: the harness dtype path is real-only.
    m, n = 64, 8
    A = torch.randn(m, n, dtype=torch.complex64, device=flag_gems.device)
    b = torch.randn(m, dtype=torch.complex64, device=flag_gems.device)

    ref = torch.linalg.lstsq(A.cpu(), b.cpu(), driver="gels").solution.to(
        flag_gems.device
    )
    with flag_gems.use_gems():
        res = torch.ops.aten.linalg_lstsq(A, b)
    assert torch.allclose(res[0], ref, atol=1e-4, rtol=1e-4)
