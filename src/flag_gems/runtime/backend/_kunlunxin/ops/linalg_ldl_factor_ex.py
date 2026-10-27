"""Kunlunxin ldl_factor_ex (aten::linalg_ldl_factor_ex) vendor override.

Implements the LAPACK DSYTF2 (unblocked Bunch-Kaufman) LDL^T factorization
with 1x1/2x2 pivoting and row/column interchanges, matching the semantics of
the CPU reference (LAPACK ssytrf/dsytf2) and the ATen CUDA implementation:
  - UPLO='L' storage: the returned LD has L in the strict lower triangle,
    D (1x1 or 2x2 blocks) on the diagonal, and an all-zero upper triangle.
  - pivots are 1-based; a positive ipiv(k) = p means a 1x1 pivot with rows and
    columns k and p interchanged; two equal negative entries -(p) twice encode
    a 2x2 block.
  - info == 0 on success (nonzero index only for zero/NaN pivot columns).

The previous vendor implementation was an *unpivoted* LDL that always wrote
pivots = 1..N.  LAPACK's Bunch-Kaufman factorization swaps rows/columns on a
fraction (~10-15%) of the random symmetric-definite test inputs (e.g. pivots
[1,2,3,4,5,6,8,8]), producing a different LD layout and different pivots, so
the old code failed those cases (see harness/solution/linalg_ldl_factor_ex).

Implementation notes (Kunlunxin/XPU Triton backend constraints):
  - An in-kernel loop over the Bunch-Kaufman steps is not usable: a
    [N x N] 2-D tile carried through a loop fails at TritonXPUCoreTiling
    (``out of resource: uni_sram``), and reductions (``tt.reduce``) inside a
    ``while`` loop cannot be legalized.  The step loop therefore lives on the
    host: one kernel launch per K (the proven linalg_det/linalg_slogdet
    structure, ~19us/launch on this backend).
  - The step size (kstep = 1 or 2) is data dependent, so the host cannot know
    which K a batch is at.  The kernel reads PIV[K-1] itself: if the previous
    step was a 2x2 pivot (PIV[K-1] < 0) the launch is a no-op, which mirrors
    LAPACK advancing K by kstep.  This keeps every batch correct in a single
    launch per K.
  - The workspace is a flat (batch, rows, LDA) tile (power-of-two TOT =
    rows*LDA lanes) so that every load/store is a single full-width strided
    access; reductions use sum-extraction of scalar-ized columns instead of
    2-D tile reductions (which crash the XPU backend); masks are applied with
    ``tl.where`` because masked stores are not honored.
  - The interchange follows LAPACK DSYTF2 exactly: it is the *partial*
    Bunch-Kaufman interchange (tail column swap, cross terms, diagonal swap,
    and the kstep==2 extra A[K+1,K] <-> A[kp,K] swap), not a full P.A.P
    transposition; the already-factored rows/cols < kk are left untouched.
  - The kernel computes in float32 (Kunlunxin Triton does not support fp64)
    and the host restores the requested dtype.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

MAX_MATRIX_SIZE = 64

_ALPHA = (1.0 + 17.0**0.5) / 8.0


def _next_pow2(n):
    return 1 << (n - 1).bit_length()


def _plan(n):
    """(LDA, TOT) for the flat workspace: TOT must be a power of two."""
    rows = triton.next_power_of_2(n)
    if n >= 16 and n % 8 == 0 and n * n <= 4096:
        lda = n
    else:
        lda = max(64, rows)
    return lda, rows * lda


@libentry()
@triton.jit
def _dsytf2_step_kernel(
    W, LD, PIV, INFO, N, K, LDA: tl.constexpr, TOT: tl.constexpr, ALPHA
):
    """One Bunch-Kaufman step (K) of DSYTF2 (UPLO='L'), one program per matrix.

    W    - (batch, N, LDA) float32 flat workspace (lower triangle persists
           between launches; each launch reads and rewrites it).
    LD   - (batch, TOT) float32 output, lower-triangle factors, upper zeroed.
    PIV  - (batch, N) int32 1-based pivot array.
    INFO - (batch,) int32.
    """
    b = tl.program_id(0)
    base = b * TOT
    e = tl.arange(0, TOT)
    row = e // LDA
    col = e % LDA
    live = (row < N) & (col < N)
    T0 = tl.load(W + base + e)
    T = T0

    prevk = tl.maximum(K - 1, 0)
    is_prev2x2 = (K > 0) & (tl.load(PIV + b * N + prevk) < 0)

    NR = TOT // LDA
    rowc = tl.where(row < NR, row, 0)
    colc = tl.where(col < NR, col, 0)
    ridx = tl.arange(0, LDA)
    rcl = tl.where(ridx < NR, ridx, 0)

    colk = tl.load(W + base + rcl * LDA + K)
    cand = tl.where((ridx > K) & (ridx < N), tl.abs(colk), 0.0)
    colmax = tl.max(cand, axis=0)
    imax = tl.min(tl.where(cand == colmax, ridx, LDA), axis=0)
    dkk = tl.sum(tl.where(ridx == K, colk, 0.0), axis=0)
    absakk = tl.abs(dkk)
    zero_col = (absakk == 0.0) & (colmax == 0.0)
    nan_col = dkk != dkk
    skip = zero_col | nan_col

    row_imax = tl.load(W + base + imax * LDA + ridx)
    r1 = tl.max(
        tl.where((ridx >= K) & (ridx < imax) & (ridx < N), tl.abs(row_imax), 0.0),
        axis=0,
    )
    col_imax = tl.load(W + base + rcl * LDA + imax)
    r2 = tl.max(tl.where((ridx > imax) & (ridx < N), tl.abs(col_imax), 0.0), axis=0)
    rowmax = tl.maximum(r1, r2)
    dimax = tl.sum(tl.where(ridx == imax, row_imax, 0.0), axis=0)

    c1 = absakk >= ALPHA * colmax
    c2 = (absakk >= ALPHA * colmax * (colmax / rowmax)) & (~c1)
    kp = tl.where(c1 | c2, K, imax)
    kstep = tl.where((c1 | c2) | (tl.abs(dimax) >= ALPHA * rowmax), 1, 2)
    kp = tl.where(skip, K, kp)
    kstep = tl.where(skip, 1, kstep)
    kk = K + kstep - 1
    two = kstep == 2

    ckk_r = tl.load(W + base + rowc * LDA + kk)
    ckp_r = tl.load(W + base + rowc * LDA + kp)
    rkp_r = tl.load(W + base + kp * LDA + rowc)
    ckk_c = tl.load(W + base + colc * LDA + kk)
    ck_r = tl.load(W + base + rowc * LDA + K)
    ck1_r = tl.load(W + base + rowc * LDA + K + 1)
    colkk_l = tl.load(W + base + rcl * LDA + kk)
    colkp_l = tl.load(W + base + rcl * LDA + kp)
    d_kk = tl.sum(tl.where(ridx == kk, colkk_l, 0.0), axis=0)
    d_kp = tl.sum(tl.where(ridx == kp, colkp_l, 0.0), axis=0)

    G = T
    G = tl.where((col == kk) & (row > kp), ckp_r, G)
    G = tl.where((col == kp) & (row > kp), ckk_r, G)
    G = tl.where((col == kk) & (row > kk) & (row < kp), rkp_r, G)
    G = tl.where((row == kp) & (col > kk) & (col < kp), ckk_c, G)
    G = tl.where((row == kk) & (col == kk), d_kp, G)
    G = tl.where((row == kp) & (col == kp), d_kk, G)
    a_kpK = tl.load(W + base + kp * LDA + K)
    a_kkK = tl.load(W + base + kk * LDA + K)
    G = tl.where((row == kk) & (col == K) & two, a_kpK, G)
    G = tl.where((row == kp) & (col == K) & two, a_kkK, G)

    ckp_c = tl.load(W + base + colc * LDA + kp)
    ck_c = tl.load(W + base + colc * LDA + K)
    ck1_c = tl.load(W + base + colc * LDA + K + 1)
    rkp_c = tl.load(W + base + kp * LDA + colc)
    a_kp_kk = tl.load(W + base + kp * LDA + kk)

    x = tl.where(
        row > kp,
        tl.where(two, ck_r, ckp_r),
        tl.where(
            row == kp,
            tl.where(two, a_kkK, a_kpK),
            tl.where(
                row == kk,
                tl.where(two, a_kpK, d_kp),
                tl.where(row > kk, tl.where(two, ck_r, rkp_r), ck_r),
            ),
        ),
    )
    xc = tl.where(
        col > kp,
        tl.where(two, ck_c, ckp_c),
        tl.where(
            col == kp,
            tl.where(two, a_kkK, a_kpK),
            tl.where(
                col == kk,
                tl.where(two, a_kpK, d_kp),
                tl.where(col > kk, tl.where(two, ck_c, rkp_c), ck_c),
            ),
        ),
    )
    x1 = tl.where(
        row > kp,
        ckp_r,
        tl.where(
            row == kp,
            a_kp_kk,
            tl.where(row == kk, d_kp, tl.where(row > kk, ckp_r, ck1_r)),
        ),
    )
    x1c = tl.where(
        col > kp,
        ckp_c,
        tl.where(
            col == kp,
            a_kp_kk,
            tl.where(col == kk, d_kp, tl.where(col > kk, ckp_c, ck1_c)),
        ),
    )

    g_KK = tl.where(two, tl.load(W + base + K * LDA + K), d_kp)
    d21 = a_kpK
    d21s = tl.where(d21 == 0.0, 1.0, d21)

    act1 = (~skip) & (K < N - 1)
    act2 = (~skip) & (K < N - 2)
    d11 = 1.0 / tl.where(g_KK == 0.0, 1.0, g_KK)
    upd1 = live & (row > K) & (col > K)
    T1 = tl.where(upd1 & act1, G - d11 * x * xc, G)
    T1 = tl.where(live & (col == K) & (row > K) & act1, x * d11, T1)

    d11v = tl.sum(tl.where(ridx == kp, colkp_l, 0.0), axis=0)
    d11v = d11v / d21s
    d22 = g_KK / d21s
    t = 1.0 / (d11v * d22 - 1.0)
    d21u = t / d21s
    w1 = d21u * (d11v * x - x1)
    w2 = d21u * (d22 * x1 - x)
    w1c = d21u * (d11v * xc - x1c)
    w2c = d21u * (d22 * x1c - xc)
    upd2 = live & (row > K + 1) & (col > K + 1)
    T2 = tl.where(upd2 & act2, G - x * w1c - x1 * w2c, G)
    T2 = tl.where(live & (col == K) & (row > K + 1) & act2, w1, T2)
    T2 = tl.where(live & (col == K + 1) & (row > K + 1) & act2, w2, T2)

    T = tl.where(two, T2, T1)
    T = tl.where(is_prev2x2, T0, T)
    tl.store(W + base + e, T)
    tl.store(LD + base + e, tl.where(col > row, 0.0, T))

    kp_val = tl.where(kstep == 1, kp + 1, -(kp + 1))
    two2 = two & (K + 1 < N)
    idx2 = tl.where(two2, K + 1, K)
    oldpiv = tl.load(PIV + b * N + K)
    oldpiv2 = tl.load(PIV + b * N + idx2)
    tl.store(PIV + b * N + K, tl.where(is_prev2x2, oldpiv, kp_val))
    tl.store(
        PIV + b * N + idx2,
        tl.where(is_prev2x2, oldpiv2, tl.where(two2, -(kp + 1), kp_val)),
    )

    cur_info = tl.load(INFO + b)
    new_info = tl.where((skip & (cur_info == 0)), K + 1, cur_info)
    new_info = tl.where(is_prev2x2, cur_info, new_info)
    tl.store(INFO + b, new_info)


def _check_linalg_ldl_factor_ex(A, hermitian, check_errors):
    if A.ndim < 2:
        raise ValueError("linalg_ldl_factor_ex: A must be at least 2D")
    if A.shape[-2] != A.shape[-1]:
        raise ValueError("linalg_ldl_factor_ex: matrix must be square")
    if not isinstance(hermitian, bool):
        raise TypeError(f"hermitian must be a bool, got {type(hermitian)}")
    if not isinstance(check_errors, bool):
        raise TypeError(f"check_errors must be a bool, got {type(check_errors)}")
    if A.dtype not in (torch.float32, torch.float64):
        raise TypeError(
            "Kunlunxin linalg_ldl_factor_ex supports float32 and float64 only"
        )
    if A.shape[-1] > MAX_MATRIX_SIZE:
        raise ValueError(
            f"linalg_ldl_factor_ex: matrix size {A.shape[-1]} exceeds maximum "
            f"{MAX_MATRIX_SIZE}"
        )


def ldl_factor_ex(A, hermitian=False, check_errors=False):
    logger.debug("GEMS_KUNLUNXIN LINALG_LDL_FACTOR_EX")
    _check_linalg_ldl_factor_ex(A, hermitian, check_errors)
    if hermitian:
        A = A.transpose(-2, -1).conj()
    n = A.shape[-1]
    batch_count = A.numel() // (n * n)
    input_contiguous = A.contiguous().reshape(batch_count, n, n)
    work_input = input_contiguous.to(torch.float32)
    lda, tot = _plan(n)
    W = torch.zeros(batch_count, n, lda, dtype=torch.float32, device=A.device)
    W[:, :, :n] = work_input[:, :n, :]
    Wf = W.reshape(batch_count, tot)
    LD = torch.zeros(batch_count, tot, dtype=torch.float32, device=A.device)
    pivots = torch.empty(batch_count, n, dtype=torch.int32, device=A.device)
    info = torch.zeros(batch_count, dtype=torch.int32, device=A.device)

    for K in range(n):
        _dsytf2_step_kernel[(batch_count,)](
            Wf,
            LD,
            pivots,
            info,
            n,
            K,
            LDA=lda,
            TOT=tot,
            ALPHA=_ALPHA,
            num_warps=1,
        )
    LD_out = LD.view(batch_count, n, lda)[:, :n, :n].contiguous()
    LD_out = LD_out.reshape(A.shape).to(A.dtype)
    return LD_out, pivots.reshape(A.shape[:-1]), info.reshape(A.shape[:-2])
