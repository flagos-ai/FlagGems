import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.copy import copy_ as _gems_copy_
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

MAX_MATRIX_SIZE = 32

_ALPHA = (1.0 + 17.0**0.5) / 8.0


def _next_pow2(n):
    return 1 << (n - 1).bit_length()


def _plan(n):
    rows = triton.next_power_of_2(n)
    if rows == n:
        lda = n if (n >= 16 and n % 8 == 0 and n * n <= 4096) else max(64, rows)
    else:
        lda = rows
    return lda, rows * lda


@libentry()
@triton.jit
def _dsytf2_step_kernel(
    W, LD, PIV, INFO, NEXT, N, K, LDA: tl.constexpr, TOT: tl.constexpr, ALPHA
):
    """One Bunch-Kaufman step (K) of DSYTF2 (UPLO='L'), one program per matrix.

    W    - (batch, N, LDA) float32 flat workspace (lower triangle persists
           between launches; each launch reads and rewrites it).
    LD   - (batch, TOT) float32 output, lower-triangle factors, upper zeroed.
    PIV  - (batch, N) int32 1-based pivot array.
    INFO - (batch,) int32.
    NEXT - (batch,) int32 index of the next step that must actually run.
    """
    b = tl.program_id(0)
    base = b * TOT
    e = tl.arange(0, TOT)
    row = e // LDA
    col = e % LDA
    live = (row < N) & (col < N)
    T0 = tl.load(W + base + e)
    T = T0

    # A 2x2 pivot writes two negative entries (PIV[k] == PIV[k+1]), so testing
    # only PIV[K-1] < 0 would also skip the step *after* the block.  NEXT is
    # advanced by the step that actually ran (K += kstep), which is exactly
    # LAPACK's loop induction.
    # NEXT is not initialized on the host (that would cost one extra
    # kernel launch); step K=0 is always the first one to run.
    next_k = tl.where(K == 0, 0, tl.load(NEXT + b))
    active = K == next_k

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
            # rows between KK and KP: the cross interchange moved A(KP, row)
            # into A(row, KK).  Read the lower triangle (rkp_r); the upper one
            # is stale after an interchange on this store-always tile.
            tl.where(row == kk, d_kp, tl.where(row > kk, rkp_r, ck1_r)),
        ),
    )
    x1c = tl.where(
        col > kp,
        ckp_c,
        tl.where(
            col == kp,
            a_kp_kk,
            tl.where(col == kk, d_kp, tl.where(col > kk, rkp_c, ck1_c)),
        ),
    )

    g_KK = tl.where(two, tl.load(W + base + K * LDA + K), d_kp)
    d21 = a_kpK
    d21s = tl.where(d21 == 0.0, 1.0, d21)

    act1 = (~skip) & (K < N - 1) & active
    act2 = (~skip) & (K < N - 2) & active
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
    T = tl.where(active, T, T0)
    tl.store(W + base + e, T)
    tl.store(LD + base + e, tl.where(col > row, 0.0, T))

    kp_val = tl.where(kstep == 1, kp + 1, -(kp + 1))
    two2 = two & (K + 1 < N)
    idx2 = tl.where(two2, K + 1, K)
    oldpiv = tl.load(PIV + b * N + K)
    oldpiv2 = tl.load(PIV + b * N + idx2)
    tl.store(PIV + b * N + K, tl.where(active, kp_val, oldpiv))
    tl.store(
        PIV + b * N + idx2,
        tl.where(active, tl.where(two2, -(kp + 1), kp_val), oldpiv2),
    )

    cur_info = tl.load(INFO + b)
    new_info = tl.where((skip & (cur_info == 0)), K + 1, cur_info)
    new_info = tl.where(active, new_info, cur_info)
    tl.store(INFO + b, new_info)
    tl.store(NEXT + b, tl.where(active, K + kstep, next_k))


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
    n = A.shape[-1]
    batch_count = A.numel() // (n * n)
    input_contiguous = A.contiguous().reshape(batch_count, n, n)
    work_input = input_contiguous.to(torch.float32)
    lda, tot = _plan(n)
    rows = tot // lda
    # The flat workspace carries ``rows = next_pow2(n)`` rows (TOT must be a
    # power of two for tl.arange); allocating only ``n`` rows made the reshape
    # below fail for every non-power-of-two n (e.g. n=3/5/7/17/33).
    W = torch.zeros(batch_count, rows, lda, dtype=torch.float32, device=A.device)
    W[:, :n, :n] = work_input
    Wf = W.reshape(batch_count, tot)
    LD = torch.zeros(batch_count, tot, dtype=torch.float32, device=A.device)
    pivots = torch.empty(batch_count, n, dtype=torch.int32, device=A.device)
    info = torch.zeros(batch_count, dtype=torch.int32, device=A.device)
    next_k = torch.empty(batch_count, dtype=torch.int32, device=A.device)

    for K in range(n):
        _dsytf2_step_kernel[(batch_count,)](
            Wf,
            LD,
            pivots,
            info,
            next_k,
            n,
            K,
            LDA=lda,
            TOT=tot,
            ALPHA=_ALPHA,
            num_warps=1,
        )
    LD_out = LD.view(batch_count, rows, lda)[:, :n, :n]
    if not LD_out.is_contiguous():
        # n < 16 keeps the historical 64-column workspace, so the slice above
        # is strided; materialise it with the gems Triton copy (not an ATen
        # copy) instead of Tensor.contiguous().
        LD_packed = torch.empty(batch_count, n, n, dtype=torch.float32, device=A.device)
        _gems_copy_(LD_packed, LD_out)
        LD_out = LD_packed
    LD_out = LD_out.reshape(A.shape).to(A.dtype)
    return LD_out, pivots.reshape(A.shape[:-1]), info.reshape(A.shape[:-2])
