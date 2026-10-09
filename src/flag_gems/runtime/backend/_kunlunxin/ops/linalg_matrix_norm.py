import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.linalg_matrix_norm import _nuc_norm as _generic_nuc_norm
from flag_gems.ops.linalg_matrix_norm import _ord2_norm as _generic_ord2_norm
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..utils.tle_copy import tle_copy
from .copy import copy_ as _gems_copy_
from .resize import resize_ as _gems_resize_

logger = logging.getLogger(__name__)

_SUPPORTED_NUMERIC = {1, -1, 2, -2, float("inf"), -float("inf")}

_OP_SUMSQ = 0
_OP_SUMABS = 1
_OP_SUM = 2
_OP_MAX = 3
_OP_MIN = 4

_BLOCK_M = 64
_BLOCK_N = 128
# `_pad_col_kernel` runs one program per `_PAD_BLOCK` rows of a buffer sized in
# whole `_BLOCK_M` row blocks, which `_BLOCK_M` divides evenly.
_PAD_BLOCK = 64
_PROG_TARGET = 96
_SPLIT_MIN_ELEMS = 1 << 18


def _combine_op(op):
    """Second-stage operator for a two-stage reduction."""
    if op in (_OP_SUMSQ, _OP_SUMABS, _OP_SUM):
        return _OP_SUM
    return op


def _identity(op):
    if op in (_OP_SUMSQ, _OP_SUMABS, _OP_SUM):
        return 0.0
    if op == _OP_MAX:
        return float("-inf")
    return float("inf")


@libentry()
@triton.jit(do_not_specialize=["R", "C_PITCH", "NFULL", "TPC", "RP"])
def _row_reduce_kernel(
    X,
    Out,
    R,
    C_PITCH,
    NFULL,
    TPC,
    RP,
    OP: tl.constexpr,
    ROWS_ALIGNED: tl.constexpr,
    FINAL_SQRT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Row-wise reduction of ``X`` along its last axis, ``axis=1`` only.

    Every load is unmasked: the caller guarantees ``C_PITCH % BLOCK_N == 0``
    with identity padding past the real extent, and rows are clamped to
    ``R - 1`` instead of masked (re-reading the last row is harmless because
    the extra output slots are never consumed).  So no lane ever reads outside
    an allocation and ``other=`` -- the backend's single worst silent-error
    source -- is never needed.

    NOTE: do not add further runtime (non-``constexpr``) scalar parameters to
    this kernel.  Adding a single unused ``i32`` argument (an attempt at
    in-kernel column clamping) made ``(1024, 65536)`` ``fro`` go from 1.10 ms to
    34.5 ms and ``(64, 64)`` ``fro`` from 0.14 ms to 1.50 ms -- a ~15-30x
    regression across the board, measured on XPU 1 - even with the guarded
    branch compiled out.
    """
    pid_m = tl.program_id(0)
    chunk = tl.program_id(1)

    rows_raw = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if ROWS_ALIGNED:
        rows = rows_raw
    else:
        rows = tl.where(rows_raw < R, rows_raw, R - 1)

    ar = tl.arange(0, BLOCK_N)[None, :]
    base = X + rows[:, None] * C_PITCH

    if OP == 3:
        acc = tl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=tl.float32)
    elif OP == 4:
        acc = tl.full([BLOCK_M, BLOCK_N], float("inf"), dtype=tl.float32)
    else:
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    t_end = tl.minimum(chunk * TPC + TPC, NFULL)
    for t in range(chunk * TPC, t_end):
        x = tl.load(base + t * BLOCK_N + ar).to(tl.float32)
        if OP == 0:
            acc += x * x
        elif OP == 1:
            acc += tl.abs(x)
        elif OP == 2:
            acc += x
        elif OP == 3:
            acc = tl.maximum(acc, x)
        else:
            acc = tl.minimum(acc, x)

    if OP == 3:
        res = tl.max(acc, axis=1)
    elif OP == 4:
        res = tl.min(acc, axis=1)
    else:
        res = tl.sum(acc, axis=1)

    if FINAL_SQRT:
        res = tl.sqrt(res)

    tl.store(Out + chunk * RP + rows_raw, res)


@libentry()
@triton.jit(do_not_specialize=["R", "XPITCH", "PITCH"])
def _pad_col_kernel(X, PAD, R, XPITCH, PITCH, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(PAD + row * PITCH, tl.load(X + tl.minimum(row, R - 1) * XPITCH))


def _pad_col(x, pad, R, pitch, ncols):
    """Write ``x``'s single column into column 0 of ``pad`` (``R`` live rows)."""
    _pad_col_kernel[(pad.shape[0] // _PAD_BLOCK,)](
        x, pad, R, pitch, ncols, BLOCK=_PAD_BLOCK
    )


def _native_contiguous(t):
    """Materialise ``t`` contiguously through the vendor DMA engine.

    ``Tensor.contiguous()`` is itself a FlagGems-registered operator, so using
    it here would drag a Triton copy kernel into the timed region (and into
    every ``use_gems`` call site).  ``tle_copy`` drives the vendor SDNN copy
    engine directly (no ATen, no FlagGems kernel).
    """
    if t.is_contiguous():
        return t
    out = torch.empty(t.shape, dtype=t.dtype, device=t.device)
    tle_copy(t, out)
    return out


def _row_reduce(x, R, C, op, final_sqrt=False):
    dev = x.device
    BM, BN = _BLOCK_M, _BLOCK_N
    RP = triton.cdiv(R, BM) * BM
    nrow_blocks = RP // BM
    rows_aligned = R % BM == 0

    assert tuple(x.shape) == (R, C) and (C == 1 or x.stride(-1) == 1)
    pitch = x.stride(0) if R > 1 else C
    ncols = C
    if C % BN:
        ncols = triton.cdiv(C, BN) * BN
        # `torch.full` lays the identity down natively (1187GB/s), where a
        # kernel writing the whole buffer manages ~29GB/s, so only the `R` live
        # rows are written and sized in `_BLOCK_M` row blocks -- that keeps
        # `_pad_col_kernel`'s grid exact, and the reduction clamps to `R - 1`
        # rather than reading the rows past `R`.
        pad = torch.full((RP, ncols), _identity(op), dtype=x.dtype, device=dev)
        if C == 1:
            # The one destination `tle_copy` cannot express, and at one element
            # per row a scatter costs less than the strided copy that used to
            # stand in for it.
            _pad_col(x, pad, R, pitch, ncols)
        elif not tle_copy(x, pad[:R, :C]):
            # Every wider destination is expressible; if that ever stops being
            # true, fail loudly rather than reduce a buffer left all identity.
            raise NotImplementedError(
                f"tle_copy cannot express a {C}-column pad destination"
            )
        x = pad
        pitch = ncols
    nfull = ncols // BN

    nchunk = 1
    if nfull > 1 and nrow_blocks < _PROG_TARGET and R * ncols >= _SPLIT_MIN_ELEMS:
        nchunk = min(nfull, max(1, _PROG_TARGET // nrow_blocks), BN)
    tpc = triton.cdiv(nfull, nchunk) if nchunk > 1 else nfull
    if nchunk > 1:
        nchunk = triton.cdiv(nfull, tpc)

    with torch_device_fn.device(dev):
        if nchunk == 1:
            out = torch.empty(RP + BM, dtype=torch.float32, device=dev)
            _row_reduce_kernel[(nrow_blocks, 1)](
                x,
                out,
                R,
                pitch,
                nfull,
                tpc,
                RP,
                OP=op,
                ROWS_ALIGNED=rows_aligned,
                FINAL_SQRT=final_sqrt,
                BLOCK_M=BM,
                BLOCK_N=BN,
                buffer_size_limit=2048,
            )
            return out[:R]

        part = torch.empty(nchunk * RP + BM, dtype=torch.float32, device=dev)
        _row_reduce_kernel[(nrow_blocks, nchunk)](
            x,
            part,
            R,
            pitch,
            nfull,
            tpc,
            RP,
            OP=op,
            ROWS_ALIGNED=rows_aligned,
            FINAL_SQRT=False,
            BLOCK_M=BM,
            BLOCK_N=BN,
            buffer_size_limit=2048,
        )
        cop = _combine_op(op)
        pt = torch.full((R, BN), _identity(cop), dtype=torch.float32, device=dev)
        part_t = part[: nchunk * RP].reshape(nchunk, RP)[:, :R].transpose(0, 1)
        tle_copy(part_t, pt[:, :nchunk])
        out = torch.empty(RP + BM, dtype=torch.float32, device=dev)
        _row_reduce_kernel[(nrow_blocks, 1)](
            pt,
            out,
            R,
            BN,
            1,
            1,
            RP,
            OP=cop,
            ROWS_ALIGNED=rows_aligned,
            FINAL_SQRT=final_sqrt,
            BLOCK_M=BM,
            BLOCK_N=BN,
            buffer_size_limit=2048,
        )
        return out[:R]


def _batched_view(A, dim):
    """Move the two target dims last and return a contiguous ``(B, M, N)``."""
    d0, d1 = dim
    ndim = A.ndim
    remaining = [d for d in range(ndim) if d != d0 and d != d1]
    perm = remaining + [d0, d1]
    Ap = A if perm == list(range(ndim)) else A.permute(perm)
    B = 1
    for i in range(Ap.ndim - 2):
        B *= Ap.size(i)
    M, N = Ap.size(-2), Ap.size(-1)
    Ab = _native_contiguous(Ap)
    return Ab.reshape(B, M, N), B, M, N


def _reshape_result(res, A, dim, keepdim, out_dtype):
    d0, d1 = dim
    ndim = A.ndim
    if keepdim:
        shape = list(A.shape)
        shape[d0] = 1
        shape[d1] = 1
    else:
        shape = [A.size(i) for i in range(ndim) if i != d0 and i != d1]
    out = res.reshape(shape)
    if out.dtype != out_dtype:
        out = out.to(out_dtype)
    return out


def _store_out(res, out):
    if out is None:
        return res
    if out.dtype != res.dtype:
        raise RuntimeError(
            "linalg_matrix_norm expected out tensor dtype "
            f"{res.dtype} but got: {out.dtype}"
        )
    if out.device != res.device:
        raise RuntimeError(
            "linalg_matrix_norm: expected out tensor to be on the same device "
            f"as the result, but got {out.device} and {res.device}"
        )
    _gems_resize_(out, res.shape)
    _gems_copy_(out, res)
    return out


def _split_factor(B, L):
    """Row-split factor for a full-matrix reduction.

    A ``[B, L]`` reduction with ``B < BLOCK_M`` would make every program read
    ``BLOCK_M`` clamped copies of the same row.  ``fro`` sums the whole matrix,
    so the segment can be cut into ``S`` equal pieces first (exact, because
    ``S`` divides ``L``) and folded afterwards.
    """
    need = triton.cdiv(_BLOCK_M, B)
    if need <= 1:
        return 1
    for cand in (2, 4, 8, 16, 32, 64, 128):
        if cand >= need and L % cand == 0 and L // cand >= _BLOCK_N:
            return cand
    return 1


def _fro(Ab, B, M, N):
    L = M * N
    flat = Ab.reshape(B, L)
    S = _split_factor(B, L)
    if S > 1:
        part = _row_reduce(flat.reshape(B * S, L // S), B * S, L // S, _OP_SUMSQ)
        return _row_reduce(part.reshape(B, S), B, S, _OP_SUM, final_sqrt=True)
    return _row_reduce(flat, B, L, _OP_SUMSQ, final_sqrt=True)


@libentry()
@triton.jit(do_not_specialize=["B", "PITCH", "NFULL"])
def _pair_dot_kernel(
    X,
    Out,
    B,
    PITCH,
    NFULL,
    ROWS_ALIGNED: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Per-batch dot product of the two rows of ``X[b]``.  Unmasked."""
    pid = tl.program_id(0)
    b_raw = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    if ROWS_ALIGNED:
        b = b_raw
    else:
        b = tl.where(b_raw < B, b_raw, B - 1)
    ar = tl.arange(0, BLOCK_N)[None, :]
    base = X + b[:, None] * (2 * PITCH)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for t in range(NFULL):
        off = t * BLOCK_N + ar
        xa = tl.load(base + off).to(tl.float32)
        xb = tl.load(base + PITCH + off).to(tl.float32)
        acc += xa * xb
    tl.store(Out + b_raw, tl.sum(acc, axis=1))


@libentry()
@triton.jit(do_not_specialize=["B"])
def _rank2_sigma_kernel(
    AA,
    BB,
    AB,
    Out,
    B,
    MODE: tl.constexpr,
    ROWS_ALIGNED: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Closed-form singular values of a rank-2 Gram matrix [[aa, ab], [ab, bb]]."""
    pid = tl.program_id(0)
    idx_raw = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    if ROWS_ALIGNED:
        idx = idx_raw
    else:
        idx = tl.where(idx_raw < B, idx_raw, B - 1)
    aa = tl.load(AA + idx)
    bb = tl.load(BB + idx)
    ab = tl.load(AB + idx)
    diff = aa - bb
    root = tl.sqrt(diff * diff + 4.0 * ab * ab)
    l0 = tl.maximum(0.5 * (aa + bb + root), 0.0)
    det = tl.maximum(aa * bb - ab * ab, 0.0)
    l1 = tl.where(l0 > 1.0e-30, det / l0, 0.0)
    s0 = tl.sqrt(l0)
    s1 = tl.sqrt(l1)
    if MODE == 0:
        res = s0
    elif MODE == 1:
        res = s1
    else:
        res = s0 + s1
    tl.store(Out + idx_raw, res)


def _rank2_sigma_norm(Ab, B, M, N, mode):
    """``ord = 2 / -2 / 'nuc'`` for ``min(M, N) == 2``.

    The two singular values come from the 2x2 Gram matrix, so only three
    reductions are needed (``|u|^2``, ``|v|^2``, ``u.v``); the eigenvalues then
    close in form.  This replaces the generic ``_rank2_svals_kernel``, whose
    vectorised branch issues a *masked strided* store into a ``2 * batch``
    buffer - on this backend a store always touches 64 contiguous elements and
    ignores the mask, so it writes far past the allocation and raises
    ``KL_XID_KERNEL_EXCEPTION`` (observed on XPU 1: the driver had to
    ``m3 mode1 reset`` the card mid-run).
    """
    dev = Ab.device
    BM, BN = _BLOCK_M, _BLOCK_N
    K = max(M, N)
    W = _native_contiguous(Ab.transpose(-2, -1)) if M >= N else Ab
    pitch = K
    if K % BN:
        pitch = triton.cdiv(K, BN) * BN
        Wp = torch.zeros((B, 2, pitch), dtype=W.dtype, device=dev)
        tle_copy(W, Wp[:, :, :K])
        W = Wp

    aa = _row_reduce(W[:, 0, :], B, pitch, _OP_SUMSQ)
    bb = _row_reduce(W[:, 1, :], B, pitch, _OP_SUMSQ)

    BP = triton.cdiv(B, BM) * BM
    rows_aligned = B % BM == 0
    with torch_device_fn.device(dev):
        ab = torch.empty(BP + BM, dtype=torch.float32, device=dev)
        _pair_dot_kernel[(BP // BM,)](
            W,
            ab,
            B,
            pitch,
            pitch // BN,
            ROWS_ALIGNED=rows_aligned,
            BLOCK_M=BM,
            BLOCK_N=BN,
            buffer_size_limit=2048,
        )
        out = torch.empty(BP + BM, dtype=torch.float32, device=dev)
        _rank2_sigma_kernel[(BP // BM,)](
            aa,
            bb,
            ab,
            out,
            B,
            MODE=mode,
            ROWS_ALIGNED=rows_aligned,
            BLOCK_M=BM,
        )
    return out[:B]


_BD_L = tl.constexpr(512)
_BD_RPAD = tl.constexpr(2048)
_BD_C = tl.constexpr(64)
_BD_R = tl.constexpr(64)
_BD_CN = tl.constexpr(128)
_BD_STURM_ITERS = tl.constexpr(48)
_BD_MAX_K = 512
_BD_MAX_L = 2048


@triton.jit
def _ds_two_sum(a, b):
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return s, e


@triton.jit
def _ds_two_prod(a, b):
    p = a * b
    c = 4097.0
    a_hi = c * a - (c * a - a)
    a_lo = a - a_hi
    b_hi = c * b - (c * b - b)
    b_lo = b - b_hi
    e = (a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi + a_lo * b_lo
    return p, e


@triton.jit
def _ds_add2(ah, al, bh, bl):
    """DS add, also the ``tl.reduce`` combine function (4-arg 2-input)."""
    s = ah + bh
    bb = s - ah
    e = (ah - (s - bb)) + (bh - bb)
    e = e + (al + bl)
    s2 = s + e
    bb2 = s2 - s
    e2 = (s - (s2 - bb2)) + (e - bb2)
    return s2, e2


@triton.jit
def _ds_add(ah, al, bh, bl):
    s, e = _ds_two_sum(ah, bh)
    e = e + (al + bl)
    return _ds_two_sum(s, e)


@triton.jit
def _ds_sub(ah, al, bh, bl):
    return _ds_add(ah, al, -bh, -bl)


@triton.jit
def _ds_mul(ah, al, bh, bl):
    ph, pl = _ds_two_prod(ah, bh)
    t = pl + ah * bl + al * bh
    return _ds_two_sum(ph, t)


@triton.jit
def _ds_div(ah, al, bh, bl):
    q0 = ah / bh
    ph, pl = _ds_two_prod(bh, q0)
    pl = pl + bl * q0
    rh, rl = _ds_sub(ah, al, ph, pl)
    q1 = rh / bh
    qh, ql = _ds_two_sum(q0, q1)
    ph2, pl2 = _ds_two_prod(bh, qh)
    pl2 = pl2 + bl * qh + bh * ql
    rh2, rl2 = _ds_sub(ah, al, ph2, pl2)
    q2 = rh2 / bh
    return _ds_two_sum(qh, q2)


@triton.jit
def _ds_sqrt(ah, al):
    sh = tl.sqrt(ah)
    r = (ah - sh * sh) + al
    sl = 0.5 * r / sh
    return _ds_two_sum(sh, sl)


@libentry()
@triton.jit(do_not_specialize=["B", "J"])
def _bidiag_col_h_kernel(SC_H, SC_L, VH, VL, TH, TL, B, J):
    """Householder on column J: column has been copied to SC (B, 512).

    Fills V (B, 512) with v = x - beta * e_J (r < J entries are zero) and
    TH/TL (B,) with tau.  All reductions are 1-D two-input ``axis=0``
    reduces over the 512-lane contiguous scratch.
    """
    b = tl.program_id(0)
    r = tl.arange(0, _BD_L)
    keep = r >= J
    xh = tl.load(SC_H + b * _BD_L + r)
    xl = tl.load(SC_L + b * _BD_L + r)
    xh = tl.where(keep, xh, 0.0)
    xl = tl.where(keep, xl, 0.0)
    ph, pl = _ds_two_prod(xh, xh)
    sh, sl = tl.reduce((ph, pl), axis=0, combine_fn=_ds_add2)
    sigh, sigl = _ds_sqrt(sh, sl)
    x0 = tl.load(SC_H + b * _BD_L + J)
    act = x0 != 0.0
    beta_h = tl.where(x0 > 0.0, -sigh, sigh)
    beta_l = tl.where(x0 > 0.0, -sigl, sigl)
    vh = tl.where(r == J, xh - beta_h, xh)
    vl = tl.where(r == J, xl - beta_l, xl)
    ph, pl = _ds_two_prod(vh, vh)
    v2h, v2l = tl.reduce((ph, pl), axis=0, combine_fn=_ds_add2)
    tau_h, tau_l = _ds_div(2.0, 0.0, v2h, v2l)
    tau_h = tl.where(act, tau_h, 0.0)
    tau_l = tl.where(act, tau_l, 0.0)
    tl.store(VH + b * _BD_L + r, vh)
    tl.store(VL + b * _BD_L + r, vl)
    tl.store(TH + b, tau_h)
    tl.store(TL + b, tau_l)


@libentry()
@triton.jit(do_not_specialize=["B", "J", "R", "RP", "PROW"])
def _bidiag_row_h_kernel(WH, WL, UH, UL, TH, TL, B, J, R, RP, PROW):
    """Householder on row J (contiguous in the (B, 512, RP) layout).

    Fills U (B, 2048) with u = x - beta * e_{J+1} (c < J+1 and c >= R entries
    are zero) and TH/TL (B,) with tau.  The row is read as 1-D 512-lane
    contiguous blocks (the pointer is clamped to the row tail, the value is
    masked to [J+1, R) so the pads are never consumed).
    """
    b = tl.program_id(0)
    j1 = J + 1
    acc_h = 0.0
    acc_l = 0.0
    for c0 in range(0, _BD_RPAD, _BD_CN):
        c = c0 + tl.arange(0, _BD_CN)
        cc = tl.minimum(c, RP - 1)
        keep = (c >= j1) & (c < R)
        xh = tl.load(WH + b * PROW + J * RP + cc)
        xl = tl.load(WL + b * PROW + J * RP + cc)
        xh = tl.where(keep, xh, 0.0)
        xl = tl.where(keep, xl, 0.0)
        ph, pl = _ds_two_prod(xh, xh)
        sh, sl = tl.reduce((ph, pl), axis=0, combine_fn=_ds_add2)
        acc_h, acc_l = _ds_add2(acc_h, acc_l, sh, sl)
    sigh, sigl = _ds_sqrt(acc_h, acc_l)
    x0 = tl.load(WH + b * PROW + J * RP + j1)
    act = x0 != 0.0
    beta_h = tl.where(x0 > 0.0, -sigh, sigh)
    beta_l = tl.where(x0 > 0.0, -sigl, sigl)
    acc_h = 0.0
    acc_l = 0.0
    for c0 in range(0, _BD_RPAD, _BD_CN):
        c = c0 + tl.arange(0, _BD_CN)
        cc = tl.minimum(c, RP - 1)
        keep = (c >= j1) & (c < R)
        xh = tl.load(WH + b * PROW + J * RP + cc)
        xl = tl.load(WL + b * PROW + J * RP + cc)
        xh = tl.where(keep, xh, 0.0)
        xl = tl.where(keep, xl, 0.0)
        uh = tl.where(c == j1, xh - beta_h, xh)
        ul = tl.where(c == j1, xl - beta_l, xl)
        ph, pl = _ds_two_prod(uh, uh)
        sh, sl = tl.reduce((ph, pl), axis=0, combine_fn=_ds_add2)
        acc_h, acc_l = _ds_add2(acc_h, acc_l, sh, sl)
        tl.store(UH + b * _BD_RPAD + c0 + tl.arange(0, _BD_CN), uh)
        tl.store(UL + b * _BD_RPAD + c0 + tl.arange(0, _BD_CN), ul)
    tau_h, tau_l = _ds_div(2.0, 0.0, acc_h, acc_l)
    tau_h = tl.where(act, tau_h, 0.0)
    tau_l = tl.where(act, tau_l, 0.0)
    tl.store(TH + b, tau_h)
    tl.store(TL + b, tau_l)


@libentry()
@triton.jit(do_not_specialize=["B", "J", "RP", "PROW"])
def _bidiag_left_w_kernel(WH, WL, VH, VL, TH, TL, WHB, WLB, B, J, RP, PROW):
    """w = tau * (V^T W): w[c] = tau * sum_r V[r] * W[r, c], c-blocks of 64.

    The sum over r uses a sequential r-loop of 64-wide contiguous row loads:
    a 2-D tile would need an ``axis=0`` reduce (rejected) or a column-major
    tile (silently wrong), so this is the only correct form.  w[c] for c < J
    is zeroed so the rank-1 update is a no-op on the already-reduced columns.
    """
    b = tl.program_id(0)
    cb = tl.program_id(1)
    c = cb * _BD_C + tl.arange(0, _BD_C)
    tau_h = tl.load(TH + b)
    tau_l = tl.load(TL + b)
    acc_h = tl.zeros([_BD_C], dtype=tl.float32)
    acc_l = tl.zeros([_BD_C], dtype=tl.float32)
    for r in range(J, _BD_L):
        vh = tl.load(VH + b * _BD_L + r)
        vl = tl.load(VL + b * _BD_L + r)
        wh = tl.load(WH + b * PROW + r * RP + c)
        wl = tl.load(WL + b * PROW + r * RP + c)
        ph, pl = _ds_two_prod(vh, wh)
        pl = pl + vl * wh + vh * wl
        acc_h, acc_l = _ds_add2(acc_h, acc_l, ph, pl)
    dh, dl = _ds_mul(tau_h, tau_l, acc_h, acc_l)
    keep = c >= J
    dh = tl.where(keep, dh, 0.0)
    dl = tl.where(keep, dl, 0.0)
    tl.store(WHB + b * _BD_RPAD + c, dh)
    tl.store(WLB + b * _BD_RPAD + c, dl)


@libentry()
@triton.jit(do_not_specialize=["B", "J", "R", "RP"])
def _bidiag_right_w_kernel(WH, WL, UH, UL, TH, TL, WRB, WRL, B, J, R, RP):
    """w = tau * (W U): w[r] = tau * sum_c W[r, c] * U[c], one program per row.

    The column loop is over 64-wide 1-D contiguous chunks (a 1-D load of
    width >= 128 inside a dynamic loop fails to lower on this backend and a
    2-D tile's ``u`` can only be loaded as a column-broadcast, which the
    tiling pass rejects), so the reduction is a 2-input ``axis=0`` DS reduce
    to a scalar per chunk.  The ``u`` tail is pointer-clamped and
    value-masked to [J+1, R) (the pads are never consumed).
    """
    b = tl.program_id(0)
    r = tl.program_id(1)
    tau_h = tl.load(TH + b)
    tau_l = tl.load(TL + b)
    acc_h = 0.0
    acc_l = 0.0
    for c0 in range(0, RP, _BD_C):
        c = c0 + tl.arange(0, _BD_C)
        cc = tl.minimum(c, RP - 1)
        keep = (c >= J + 1) & (c < R)
        uh = tl.load(UH + b * _BD_RPAD + c)
        ul = tl.load(UL + b * _BD_RPAD + c)
        uh = tl.where(keep, uh, 0.0)
        ul = tl.where(keep, ul, 0.0)
        wh = tl.load(WH + b * _BD_L * RP + r * RP + cc)
        wl = tl.load(WL + b * _BD_L * RP + r * RP + cc)
        ph, pl = _ds_two_prod(wh, uh)
        pl = pl + wl * uh + wh * ul
        sh, sl = tl.reduce((ph, pl), axis=0, combine_fn=_ds_add2)
        acc_h, acc_l = _ds_add2(acc_h, acc_l, sh, sl)
    dh, dl = _ds_mul(tau_h, tau_l, acc_h, acc_l)
    tl.store(WRB + b * _BD_L + r, dh)
    tl.store(WRL + b * _BD_L + r, dl)


@libentry()
@triton.jit(do_not_specialize=["B", "RP", "PROW"])
def _bidiag_update_left_kernel(WH, WL, VH, VL, TH, TL, WHB, WLB, B, RP, PROW):
    """W -= tau * V (V^T W): one row ``r`` and one 64-wide c-block per program.

    v is zero for r < J and w[b, c] is zero for c < J, so the full grid
    (r, cb) is a no-op outside the trailing submatrix (no masks, no clamps).
    """
    b = tl.program_id(0)
    r = tl.program_id(1)
    cb = tl.program_id(2)
    vh = tl.load(VH + b * _BD_L + r)
    vl = tl.load(VL + b * _BD_L + r)
    th, tl_ = vh, vl
    c = cb * _BD_C + tl.arange(0, _BD_C)
    wh = tl.load(WHB + b * _BD_RPAD + c)
    wl = tl.load(WLB + b * _BD_RPAD + c)
    ptr = WH + b * PROW + r * RP + c
    ah = tl.load(ptr)
    al = tl.load(WL + b * PROW + r * RP + c)
    ph, pl = _ds_two_prod(th, wh)
    pl = pl + tl_ * wh + th * wl
    nh, nl = _ds_sub(ah, al, ph, pl)
    tl.store(ptr, nh)
    tl.store(WL + b * PROW + r * RP + c, nl)


@libentry()
@triton.jit(do_not_specialize=["B", "RP", "PROW"])
def _bidiag_update_right_kernel(WH, WL, UH, UL, TH, TL, WRB, WRL, B, RP, PROW):
    """W -= (W U) U^T: one row ``r`` and one 64-wide c-block per program.

    w[b, r] is zero for r >= K and u[b, c] is zero for c < J+1 or c >= R, so
    the full grid is a no-op outside the trailing submatrix.
    """
    b = tl.program_id(0)
    r = tl.program_id(1)
    cb = tl.program_id(2)
    wh = tl.load(WRB + b * _BD_L + r)
    wl = tl.load(WRL + b * _BD_L + r)
    th, tl_ = wh, wl
    c = cb * _BD_C + tl.arange(0, _BD_C)
    uh = tl.load(UH + b * _BD_RPAD + c)
    ul = tl.load(UL + b * _BD_RPAD + c)
    ptr = WH + b * PROW + r * RP + c
    ah = tl.load(ptr)
    al = tl.load(WL + b * PROW + r * RP + c)
    ph, pl = _ds_two_prod(th, uh)
    pl = pl + tl_ * uh + th * ul
    nh, nl = _ds_sub(ah, al, ph, pl)
    tl.store(ptr, nh)
    tl.store(WL + b * PROW + r * RP + c, nl)


@libentry()
@triton.jit(do_not_specialize=["B"])
def _bidiag_tridiag_kernel(DH, DL, EH, EL, TD, TL, TE, EL2, B):
    """T = B B^T as DS tridiagonal: td = d^2 + e^2, te = e * d_{i+1}."""
    b = tl.program_id(0)
    i = tl.arange(0, _BD_L)
    dh = tl.load(DH + b * _BD_L + i)
    eh = tl.load(EH + b * _BD_L + i)
    p1h, p1l = _ds_two_prod(dh, dh)
    p2h, p2l = _ds_two_prod(eh, eh)
    th, t = _ds_add(p1h, p1l, p2h, p2l)
    ip1 = tl.minimum(i + 1, _BD_L - 1)
    d1h = tl.load(DH + b * _BD_L + ip1)
    d1l = tl.load(DL + b * _BD_L + ip1)
    d1h = tl.where(i + 1 < _BD_L, d1h, 0.0)
    d1l = tl.where(i + 1 < _BD_L, d1l, 0.0)
    uh, ul = _ds_two_prod(eh, d1h)
    tl.store(TD + b * _BD_L + i, th)
    tl.store(TL + b * _BD_L + i, t)
    tl.store(TE + b * _BD_L + i, uh)
    tl.store(EL2 + b * _BD_L + i, ul)


@libentry()
@triton.jit(do_not_specialize=["B", "K"])
def _sturm_eig_kernel(TD, TL, TE, EL, OUT, B, K):
    """Bisect the tridiagonal T's eigenvalues with a DS Sturm sequence.

    One program per (batch, 64-lane block); lane ``j`` tracks the (j+1)-th
    smallest eigenvalue (``tgt = j + 1``, ``take = count >= tgt``).  The
    Sturm recurrence q_{i+1} = (td_{i+1} - lam) - te_i^2 / q_i runs in DS; a
    zero quotient is replaced by -1e-30 so the sign count is well defined.
    Output OUT[b, j] is the ascending (j+1)-th smallest lambda; the caller
    takes the square roots and the desc/asc order.

    The lane width is 64 (not 512/128): on this backend a 1-D vector of
    width >= 128 inside a dynamic loop fails to lower (TritonXPUUnrollControl
    runs out of unified SRAM), while 64-wide 1-D vectors and 2-D [64, 128]
    tiles are both fine.
    """
    b = tl.program_id(0)
    pb = tl.program_id(1)
    j = pb * _BD_C + tl.arange(0, _BD_C)
    tgt = (j + 1).to(tl.float32)
    hi = tl.zeros([_BD_C], dtype=tl.float32)
    for i in range(K):
        td = tl.load(TD + b * _BD_L + i)
        te = tl.load(TE + b * _BD_L + i)
        te0 = tl.load(TE + b * _BD_L + tl.maximum(i - 1, 0))
        s = tl.abs(td) + tl.abs(te) + tl.abs(te0)
        hi = tl.maximum(hi, s)
    hi = hi * 1.000000001
    lo = tl.zeros([_BD_C], dtype=tl.float32)
    his = hi
    for _ in range(_BD_STURM_ITERS):
        mid = 0.5 * (lo + his)
        td0 = tl.load(TD + b * _BD_L)
        tl0 = tl.load(TL + b * _BD_L)
        qh, ql = _ds_sub(td0, tl0, mid, 0.0)
        z = (qh == 0.0) & (ql == 0.0)
        qh = tl.where(z, -1.0e-30, qh)
        cnt = ((qh < 0.0) | ((qh == 0.0) & (ql < 0.0))).to(tl.float32)
        for i in range(1, K):
            tdi = tl.load(TD + b * _BD_L + i)
            tli = tl.load(TL + b * _BD_L + i)
            ph, pl = _ds_sub(tdi, tli, mid, 0.0)
            th, t = _ds_mul(ph, pl, qh, ql)
            ei = tl.load(TE + b * _BD_L + i - 1)
            eli = tl.load(EL + b * _BD_L + i - 1)
            bh, bl = _ds_two_prod(ei, ei)
            bl = bl + 2.0 * ei * eli
            nh, nl = _ds_sub(th, t, bh, bl)
            qh, ql = _ds_div(nh, nl, qh, ql)
            z = (qh == 0.0) & (ql == 0.0)
            qh = tl.where(z, -1.0e-30, qh)
            cnt = cnt + ((qh < 0.0) | ((qh == 0.0) & (ql < 0.0))).to(tl.float32)
        take = cnt >= tgt
        his = tl.where(take, mid, his)
        lo = tl.where(take, lo, mid)
    lam = 0.5 * (lo + his)
    # T = B B^T is PSD, but the Sturm bisection can return lambdas slightly
    # below zero (pure rounding); clamp in-kernel so the caller never touches
    # an ATen sqrt/clamp (the sigma = sqrt(lambda) is all the caller consumes).
    tl.store(OUT + b * _BD_L + j, tl.sqrt(tl.maximum(lam, 0.0)))


def _svd_bidiag_sturm(Ab, B, M, N, mode):
    """``ord = 2 / -2 / 'nuc'`` for ``min(M, N) >= 3``.

    ``Ab`` is a contiguous ``(B, M, N)`` fp32 view.  The pipeline is

    1. W = A^T (k x rows, row-major, zero-padded to (B, 512, RP));
    2. two-sided Householder bidiagonalisation (DS arithmetic, W = Bh + Bl),
       one column-Householder + one row-Householder per step j;
    3. extract (d, e) = (B[i, i], B[i, i + 1]) and form the DS tridiagonal
       T = B B^T;
    4. DS Sturm bisection of T's eigenvalues; sigma = sqrt(lambda).

    Returns a ``(B,)`` fp32 tensor (sigma_max / sigma_min / nuclear norm for
    ``mode`` 0 / 1 / 2).
    """
    dev = Ab.device
    K = min(M, N)
    R = max(M, N)
    RP = int(max(2 * _BD_C, triton.next_power_of_2(R)))
    PROW = int(_BD_L) * RP
    Wh = torch.zeros((B, _BD_L, RP), dtype=torch.float32, device=dev)
    Wl = torch.zeros_like(Wh)
    if M >= N:
        Wt = _native_contiguous(Ab.transpose(-2, -1))
    else:
        Wt = Ab
    tle_copy(Wt, Wh[:, :K, :R])

    sc_h = torch.empty((B, _BD_L), dtype=torch.float32, device=dev)
    sc_l = torch.empty_like(sc_h)
    v_h = torch.empty_like(sc_h)
    v_l = torch.empty_like(sc_h)
    u_h = torch.empty((B, _BD_RPAD), dtype=torch.float32, device=dev)
    u_l = torch.empty_like(u_h)
    wh_b = torch.empty((B, _BD_RPAD), dtype=torch.float32, device=dev)
    wl_b = torch.empty_like(wh_b)
    wr_b = torch.empty((B, _BD_L), dtype=torch.float32, device=dev)
    wr_l = torch.empty_like(wr_b)
    th = torch.empty((B,), dtype=torch.float32, device=dev)
    tl_ = torch.empty_like(th)
    rh = torch.empty_like(th)
    rl = torch.empty_like(th)

    with torch_device_fn.device(dev):
        for j in range(min(K, R - 1)):
            tle_copy(Wh[:, :, j], sc_h)
            tle_copy(Wl[:, :, j], sc_l)
            _bidiag_col_h_kernel[(B,)](sc_h, sc_l, v_h, v_l, th, tl_, B, j)
            _bidiag_left_w_kernel[(B, int(RP // _BD_C))](
                Wh, Wl, v_h, v_l, th, tl_, wh_b, wl_b, B, j, RP, PROW
            )
            _bidiag_update_left_kernel[(B, int(_BD_L), int(RP // _BD_C))](
                Wh, Wl, v_h, v_l, th, tl_, wh_b, wl_b, B, RP, PROW
            )
            _bidiag_row_h_kernel[(B,)](Wh, Wl, u_h, u_l, rh, rl, B, j, int(R), RP, PROW)
            _bidiag_right_w_kernel[(B, int(_BD_L))](
                Wh, Wl, u_h, u_l, rh, rl, wr_b, wr_l, B, j, int(R), RP
            )
            _bidiag_update_right_kernel[(B, int(_BD_L), int(RP // _BD_C))](
                Wh, Wl, u_h, u_l, rh, rl, wr_b, wr_l, B, RP, PROW
            )
        dh = torch.zeros((B, _BD_L), dtype=torch.float32, device=dev)
        dl = torch.zeros_like(dh)
        eh = torch.zeros_like(dh)
        el = torch.zeros_like(dh)
        nd = min(_BD_L, RP)
        tle_copy(Wh.diagonal(0, 1, 2), dh[:, :nd])
        tle_copy(Wl.diagonal(0, 1, 2), dl[:, :nd])
        ne = min(_BD_L, RP - 1)
        if ne > 0:
            tle_copy(Wh.diagonal(1, 1, 2), eh[:, :ne])
            tle_copy(Wl.diagonal(1, 1, 2), el[:, :ne])
        td = torch.empty((B, _BD_L), dtype=torch.float32, device=dev)
        tdl = torch.empty_like(td)
        te = torch.empty_like(td)
        tel = torch.empty_like(td)
        _bidiag_tridiag_kernel[(B,)](dh, dl, eh, el, td, tdl, te, tel, B)
        sig = torch.empty((B, _BD_L), dtype=torch.float32, device=dev)
        _sturm_eig_kernel[(B, int(_BD_L // _BD_C))](td, tdl, te, tel, sig, B, int(K))
        if mode == 0:
            return sig[:, K - 1 : K].reshape(B)
        if mode == 1:
            return sig[:, 0:1].reshape(B)
        return sig[:, :K].sum(dim=1)


def _absmax_norm(Ab, B, M, N, is_min, along_rows):
    """|A| row/column sums followed by a max (or min) over the survivors.

    ``along_rows=True``  -> ord = +/-inf (sum over N, then max/min over M)
    ``along_rows=False`` -> ord = +/-1   (sum over M, then max/min over N)
    """
    if along_rows:
        base, R, C = Ab, B * M, N
    else:
        base = _native_contiguous(Ab.transpose(-2, -1))
        R, C = B * N, M
    sums = _row_reduce(base.reshape(R, C), R, C, _OP_SUMABS)
    inner = R // B
    return _row_reduce(sums.reshape(B, inner), B, inner, _OP_MIN if is_min else _OP_MAX)


def linalg_matrix_norm(
    A, ord="fro", dim=(-2, -1), keepdim=False, dtype=None, *, out=None
):
    """Matrix norm over ``dim``.

    ``out=None`` returns a freshly-allocated result; when ``out`` is given the
    result is written into it and the aliased tensor is returned (same contract
    as ``linalg_matrix_norm.out`` / the generic implementation).
    """
    logger.debug("GEMS_KUNLUNXIN LINALG_MATRIX_NORM")

    if A.ndim < 2:
        raise RuntimeError(
            f"linalg_matrix_norm: A must be at least 2-D, got shape {A.shape}"
        )
    dim = list(dim)
    if len(dim) != 2:
        raise RuntimeError(f"linalg_matrix_norm: dim must be a 2-tuple, got {dim}")
    dim = [d % A.ndim for d in dim]
    if dim[0] == dim[1]:
        raise RuntimeError(
            f"linalg_matrix_norm: dims must be different, got ({dim[0]}, {dim[1]})"
        )

    is_str = isinstance(ord, str)
    if is_str and ord not in ("fro", "nuc"):
        raise RuntimeError(
            f"linalg_matrix_norm: Order '{ord}' not supported. Use 'fro' or 'nuc'."
        )
    ord_val = None
    if not is_str:
        ord_val = float(ord)
        if ord_val not in _SUPPORTED_NUMERIC:
            raise RuntimeError(
                f"linalg_matrix_norm: Order {ord} not supported. "
                "Use 1, -1, 2, -2, inf, -inf."
            )

    is_svd = (is_str and ord == "nuc") or (ord_val is not None and abs(ord_val) == 2.0)
    if is_svd:
        if A.dtype in (torch.float16, torch.bfloat16):
            A = A.float()
        k = min(A.size(dim[0]), A.size(dim[1]))
        if k > 2 and (k > _BD_MAX_K or max(A.size(dim[0]), A.size(dim[1])) > _BD_MAX_L):
            if is_str:
                return _store_out(
                    _generic_nuc_norm(A, dim=dim, keepdim=keepdim, dtype=dtype), out
                )
            return _store_out(_generic_ord2_norm(A, ord_val, dim, keepdim, dtype), out)
        out_dtype = dtype if dtype is not None else A.dtype
        if dtype is not None:
            A = A.to(dtype)
        Ab, B, M, N = _batched_view(A, dim)
        if k <= 1:
            res = _fro(Ab, B, M, N)
        elif k == 2:
            mode = 2 if is_str else (0 if ord_val > 0 else 1)
            res = _rank2_sigma_norm(Ab, B, M, N, mode)
        else:
            mode = 2 if is_str else (0 if ord_val > 0 else 1)
            res = _svd_bidiag_sturm(Ab, B, M, N, mode)
        return _store_out(_reshape_result(res, A, dim, keepdim, out_dtype), out)

    out_dtype = dtype if dtype is not None else A.dtype
    Ab, B, M, N = _batched_view(A, dim)

    if is_str:
        res = _fro(Ab, B, M, N)
    else:
        res = _absmax_norm(Ab, B, M, N, ord_val < 0, math.isinf(ord_val))

    return _store_out(_reshape_result(res, A, dim, keepdim, out_dtype), out)


def linalg_matrix_norm_out(
    A, ord="fro", dim=(-2, -1), keepdim=False, dtype=None, *, out=None
):
    """``linalg_matrix_norm.out`` overload for the kunlunxin backend.

    Without this override the ``.out`` overload falls through to the generic
    ``flag_gems.ops.linalg_matrix_norm.linalg_matrix_norm_out``, whose Triton
    reduce kernel does an ``axis=0`` 2-D reduction that fails to lower on this
    backend (``axis must not be 0 for 2D+ shapes`` / uni_sram OOR).  The vendor
    ``linalg_matrix_norm`` already handles ``out=`` end to end (via
    ``_store_out``), so the out variant just requires a pre-allocated ``out``
    and delegates to it.
    """
    logger.debug("GEMS_KUNLUNXIN LINALG_MATRIX_NORM_OUT")
    if out is None:
        raise TypeError("linalg_matrix_norm(): out must be provided for out variant")
    return linalg_matrix_norm(
        A, ord=ord, dim=dim, keepdim=keepdim, dtype=dtype, out=out
    )
