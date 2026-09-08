import functools

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# gels fast path covers overdetermined (m >= n) float32. Shapes are (m, n).
_SHAPES = [(16, 4), (64, 8), (128, 16), (200, 8), (256, 32)]


# Condition number of the WY cases is CONTROLLED, not sampled. The forward error
# of a least-squares solve scales with kappa(A), and a square Gaussian has
# kappa ~ n with a heavy tail (P(kappa > t*n) ~ 1/t) -- so a tolerance chosen
# against a random draw is a probabilistic bound, not a bound. Following
# tests/test_linalg_svdvals.py, build the matrix through a LOCAL generator and
# re-impose singular values spread over [1, KAPPA]: deterministic, kappa known
# and independent of n, and the global RNG stream is left untouched so the
# tests stay order-independent.
_WY_KAPPA = 10.0
# kappa * eps_fp32 ~ 1.2e-6 per operation; over an n-column QR the accumulated
# forward error is ~n * kappa * eps, i.e. ~1.2e-3 at n=1024. 5e-3 is ~4x that,
# and 10x tighter than the sampled-kappa bound it replaces.
_WY_ATOL = 5e-3


def _cond_bounded(shape, dtype, device, seed=42, kappa=_WY_KAPPA):
    """Matrix with a KNOWN condition number, built deterministically on CPU."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    A = torch.randn(*shape, generator=g, dtype=torch.float64, device="cpu")
    U, _, Vh = torch.linalg.svd(A, full_matrices=False)
    k = min(shape[-2], shape[-1])
    S = torch.linspace(kappa, 1.0, k, dtype=torch.float64, device="cpu")
    return ((U * S) @ Vh).to(dtype=dtype).to(device)


def _det_randn(shape, dtype, device, seed):
    """Deterministic tensor that does not consume the global RNG stream."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    t = torch.randn(*shape, generator=g, dtype=torch.float64, device="cpu")
    return t.to(dtype=dtype).to(device)


@functools.lru_cache(maxsize=None)
def _gems_supports(dtype):
    """Does the registered gems kernel implement `dtype` on this device?

    Deliberately NOT a hardware or dtype probe. Both mislead here: an Ascend
    910B reports a float64 tensor as genuine float64 -- element_size 8, storage
    8 bytes/element, and (1 + 2**-40) - 1 exact -- because those ops fall back
    to the CPU, yet the device has no float64 unit and the backend's kernel
    does not implement it. Checking `.dtype`, element_size or precision all
    answer True there and the tests then fail rather than skip.

    What these tests actually depend on is whether the KERNEL supports the
    dtype, and a backend states that by raising NotImplementedError. So ask it
    directly, with the smallest possible call.

    Only NotImplementedError counts as unsupported; any other exception returns
    True so the test still runs and reports the real defect. A skip must never
    be able to hide a genuine failure.
    """
    try:
        A = torch.randn(4, 2, dtype=dtype, device=flag_gems.device)
        b = torch.randn(4, dtype=dtype, device=flag_gems.device)
    except Exception:
        return False  # device cannot hold the dtype at all (complex here)
    try:
        with flag_gems.use_gems():
            torch.ops.aten.linalg_lstsq(A, b, driver="gels")
    except NotImplementedError:
        return False
    except Exception:
        return True
    return True


def _require_dtype(dtype):
    """Skip when the backend has no kernel for `dtype`, so a run stays red
    only for real defects. Called inside the test rather than as a skipif mark
    so nothing touches the device at collection time."""
    if not _gems_supports(dtype):
        pytest.skip(f"gems linalg_lstsq has no {dtype} kernel on this device")


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
    _require_dtype(torch.float64)
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
    _require_dtype(torch.float64)
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
@pytest.mark.parametrize(
    "shape,kind",
    [
        ((128, 6), "col"),  # single-tile QR
        ((4096, 200), "col"),  # blocked TSQR: NC>128, <=256, >=16 chunks
        ((256, 256), "col"),  # compact-WY (square)
        ((16, 64), "row"),  # underdetermined min-norm
    ],
    ids=["tall_single", "tall_blocked", "square_wy", "wide"],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_lstsq_rank_deficient(shape, kind, dtype):
    # Exactly-singular input, one shape per QR path. Each path has its own rank
    # guard in a different kernel, so covering only the single-tile case leaves
    # three guards untested.
    #
    # A zero column makes r_ii exactly 0 and trips the guard deterministically.
    # (A merely near-dependent column is threshold-sensitive under fp32 and not
    # a reliable guard test -- that case is covered contractually by
    # test_linalg_lstsq_near_singular below. gels is documented as undefined on
    # rank-deficient input either way, so only the guard, not the values, is
    # asserted here.)
    #
    # For m < n the deficiency has to be a zero ROW: the min-norm path QRs A^T,
    # where a zero row of A is a zero column, which is what makes R singular.
    m, n = shape
    A = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    if kind == "col":
        A[:, 3] = 0.0
    else:
        A[2, :] = 0.0
    b = torch.randn(m, dtype=dtype, device=flag_gems.device)

    # Deliberately NOT compared against torch here: on EXACTLY singular input
    # torch's own behaviour is backend-dependent, so there is no single
    # reference to match. CPU LAPACK ?gels reports the deficiency (info > 0) and
    # PyTorch raises _LinAlgError ("... does not have full rank (error code: 4)",
    # the 1-based index of the dead column); cuSOLVER's gels does not check and
    # returns NaN/garbage silently. We follow the CUDA behaviour -- NaN, per the
    # deficient-gels contract -- and assert that directly. The comparison with
    # torch lives in test_linalg_lstsq_near_singular, where the matrix is only
    # ill-conditioned and every backend agrees.
    with flag_gems.use_gems():
        res = torch.ops.aten.linalg_lstsq(A, b, driver="gels")

    assert res[0].shape == (n,), "vector RHS -> solution is squeezed to (n,)"
    assert torch.isnan(res[0]).any(), "rank-deficient A should yield NaN solution"
    # How FAR the NaN spreads is deliberately not asserted. It is not part of
    # the gels contract -- the result is undefined on rank-deficient input --
    # and it already differs by path: measured all-NaN on the tall and square
    # paths, where it reaches every row, so pinning "only the rows from the dead
    # pivot upward" would be pinning implementation behaviour, not a spec.


@pytest.mark.linalg_lstsq
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_lstsq_near_singular(dtype):
    # Near-singular (not exactly singular): whether the guard fires depends on
    # where r_ii lands relative to rcond * max|r_jj|, which is not stable under
    # fp32 -- so neither NaN nor a value comparison is assertable here.
    #
    # What IS part of torch's contract, and is asserted: gels does not raise on
    # an ill-conditioned matrix, and the returned tuple keeps its shapes and
    # dtypes. Silently raising, or degrading to a differently-shaped result,
    # would be an incompatibility that the exactly-singular test above cannot
    # catch.
    m, n = 128, 6
    A = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    eps = torch.randn(m, dtype=dtype, device=flag_gems.device) * 1e-6
    A[:, 4] = A[:, 1] + eps  # column 4 nearly duplicates column 1
    b = torch.randn(m, dtype=dtype, device=flag_gems.device)

    ref = torch.linalg.lstsq(A, b, driver="gels")
    with flag_gems.use_gems():
        res = torch.ops.aten.linalg_lstsq(A, b, driver="gels")

    assert res[0].shape == ref.solution.shape
    assert res[0].dtype == ref.solution.dtype
    assert res[1].shape == ref.residuals.shape
    assert res[2].shape == ref.rank.shape
    assert res[3].shape == ref.singular_values.shape


@pytest.mark.linalg_lstsq
@pytest.mark.parametrize("shape", [(64, 8), (16, 64)])  # tall and wide
@pytest.mark.parametrize("dtype", [torch.float64])
def test_linalg_lstsq_fp64(shape, dtype):
    _require_dtype(torch.float64)
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
    _require_dtype(torch.complex64)
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


@pytest.mark.linalg_lstsq
@pytest.mark.parametrize(
    "batch,shape",
    [
        ((), (256, 256)),  # square -> WY
        ((), (512, 512)),  # square, exercises the panel row-block boundary
        ((), (1024, 1024)),  # square, multi row-block
        ((), (4096, 512)),  # tall but few chunks -> WY, not blocked TSQR
        ((2,), (512, 512)),  # batched square
        ((), (300, 260)),  # non-power-of-2, P does not divide n
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_lstsq_square_wy(batch, shape, dtype):
    # Square-ish shapes route to the right-looking blocked QR (compact-WY):
    # TSQR needs block_m >= NC, so it collapses to ONE program here (2048x2048
    # took ~5.7 s); WY panels over columns instead and keeps the trailing
    # update gridded. The point of the case is the CODE PATH, not conditioning,
    # so kappa is bounded by construction (see _cond_bounded).
    m, n = shape
    dev = flag_gems.device
    A = _cond_bounded((*batch, m, n), dtype, dev, seed=42)
    b = _det_randn((*batch, m), dtype, dev, seed=43)
    ref = torch.linalg.lstsq(A, b, driver="gels")
    with flag_gems.use_gems():
        res = torch.ops.aten.linalg_lstsq(A, b, driver="gels")
    assert res[0].shape == ref.solution.shape
    utils.gems_assert_close(
        res[0], utils.to_reference(ref.solution), dtype, atol=_WY_ATOL
    )


@pytest.mark.linalg_lstsq
@pytest.mark.parametrize("dtype", [torch.float64])
def test_linalg_lstsq_square_wy_fp64(dtype):
    _require_dtype(torch.float64)
    # fp64 square: same path, and fp64 conditioning is no longer the limit.
    # kappa is bounded here too, so the 1e-6 bound below is a real bound rather
    # than one that happens to hold for the matrix that was drawn.
    m, n = 256, 256
    dev = flag_gems.device
    A = _cond_bounded((m, n), dtype, dev, seed=46)
    b = _det_randn((m,), dtype, dev, seed=47)
    ref = torch.linalg.lstsq(A, b, driver="gels")
    with flag_gems.use_gems():
        res = torch.ops.aten.linalg_lstsq(A, b, driver="gels")
    sc = ref.solution.abs().max().clamp_min(1.0)
    err = ((res[0] - ref.solution).abs().max() / sc).item()
    assert err < 1e-6, f"fp64 square lstsq relerr {err:.2e} too large"


@pytest.mark.linalg_lstsq
@pytest.mark.parametrize(
    "batch,shape",
    [
        ((), (512, 1024)),  # large m -> A^T QR routes to WY
        ((), (1024, 2048)),  # larger still
        ((), (300, 700)),  # non-power-of-2
        ((2,), (512, 1024)),  # batched
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_lstsq_underdetermined_wy(batch, shape, dtype):
    # Underdetermined with LARGE m. The min-norm path QRs A^T, whose short
    # dimension is m -- so a big m makes TSQR collapse to a single program the
    # same way square does on the tall path. These route to compact-WY instead.
    # Untestable before that path existed (they took seconds).
    m, n = shape
    dev = flag_gems.device
    A = _cond_bounded((*batch, m, n), dtype, dev, seed=44)
    b = _det_randn((*batch, m), dtype, dev, seed=45)
    ref = torch.linalg.lstsq(A, b, driver="gels")
    with flag_gems.use_gems():
        res = torch.ops.aten.linalg_lstsq(A, b, driver="gels")
    assert res[0].shape == ref.solution.shape
    utils.gems_assert_close(
        res[0], utils.to_reference(ref.solution), dtype, atol=_WY_ATOL
    )
    # min-norm is the defining property: verify feasibility A x ~= b directly,
    # since any x with A x = b solves the system but only one has least norm.
    feas = (torch.matmul(A, res[0].unsqueeze(-1)).squeeze(-1) - b).abs().max()
    assert feas.item() < 5e-2, f"A x != b (feasibility {feas.item():.2e})"
