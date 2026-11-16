import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

from ..utils.tle_copy import tle_copy

logger = logging.getLogger(__name__)

KS_SLICE = 64


@libentry()
@triton.jit
def _trsm_slice_xpu_kernel(
    A_ptr,
    B_ptr,
    Ah_ptr,
    Al_ptr,
    Bh_ptr,
    Bl_ptr,
    N,
    K,
    rsa,
    csb,
    F64: tl.constexpr,
    UNIT: tl.constexpr,
    KS: tl.constexpr,
    UPPER: tl.constexpr,
    NS: tl.constexpr,
):
    """One triangular solve of an RHS column-slice: X = A^-1 B.

    grid = (batch * NS,): program ``pid`` owns batch ``pid // NS`` and the RHS
    column slice ``pid % NS`` (``KS`` columns).  Column slices are independent
    solves, so the only cross-program traffic is the read-only A; each program
    reads back exactly the X lanes it wrote.  Batch/slice bases are derived
    from the existing strides (``N * rsa`` / ``N * csb``) on purpose: adding a
    runtime scalar argument to an XPU kernel costs 15-30x.

    Rows are solved serially (the triangular dependency chain); the dot product
    over the already-solved window is a serial scalar-j chain on K-vectors.
    TritonXPU cannot batch this reduction: a 2D tile inside the row loop fails
    `tt.addptr` verification, `tl.static_range` over tiles fails the XPU layout
    check, and a 1D `tl.sum` inside a runtime loop hits
    "failed to legalize operation 'tt.reduce'" (all measured 2026-08-30).

    ``UPPER`` selects the sweep direction in-kernel (backward substitution over
    reversed row/column indices) so the host never has to materialise a flipped
    copy of A/B: the gems-registered ``flip`` kernel raises
    KL_XID_KERNEL_EXCEPTION for a 512x512 fp32 last-dim flip (measured).
    """
    pid = tl.program_id(0)
    bidx = pid // NS
    sidx = pid % NS
    abase = bidx * N * rsa
    bbase = bidx * N * csb + sidx * KS
    cc = tl.arange(0, KS)
    cm = cc < K
    for t in range(N):
        row = N - 1 - t if UPPER else t
        if F64:
            acc_h = tl.load(Bh_ptr + bbase + row * csb + cc, mask=cm, other=0.0)
            acc_l = tl.load(Bl_ptr + bbase + row * csb + cc, mask=cm, other=0.0)
        else:
            acc = tl.load(B_ptr + bbase + row * csb + cc, mask=cm, other=0.0)
        for u in range(t):
            j = N - 1 - u if UPPER else u
            if F64:
                ah = tl.load(Ah_ptr + abase + row * rsa + j)
                al = tl.load(Al_ptr + abase + row * rsa + j)
                xh = tl.load(Bh_ptr + bbase + j * csb + cc, mask=cm, other=0.0)
                xl = tl.load(Bl_ptr + bbase + j * csb + cc, mask=cm, other=0.0)
                sp = 8193.0
                ca = ah * sp
                ab = ca - ah
                ahh = ca - ab
                ahl = ah - ahh
                cx = xh * sp
                xb = cx - xh
                xhh = cx - xb
                xhl = xh - xhh
                ph = ah * xh
                pl = (
                    (ahh * xhh - ph)
                    + (ahh * xhl + ahl * xhh)
                    + (ahl * xhl + ah * xl + al * xh + al * xl)
                )
                s = acc_h - ph
                acc_l = (acc_h - s) - ph + acc_l + pl
                acc_h = s
            else:
                a = tl.load(A_ptr + abase + row * rsa + j)
                x = tl.load(B_ptr + bbase + j * csb + cc, mask=cm, other=0.0)
                acc = acc - a * x
        if F64:
            out_h = acc_h
            out_l = acc_l
            if not UNIT:
                dh = tl.load(Ah_ptr + abase + row * rsa + row)
                dl = tl.load(Al_ptr + abase + row * rsa + row)
                q1 = out_h / dh
                ph = q1 * dh
                pl = tl.fma(q1, dh, -ph) + q1 * dl
                r = out_h - ph
                q2 = r / dh
                out_h = q1 + q2
                out_l = (q1 - out_h) + q2
            tl.store(Bh_ptr + bbase + row * csb + cc, out_h, mask=cm)
            tl.store(Bl_ptr + bbase + row * csb + cc, out_l, mask=cm)
        else:
            xv = acc
            if not UNIT:
                d = tl.load(A_ptr + abase + row * rsa + row)
                xv = xv * (1.0 / d)
            tl.store(B_ptr + bbase + row * csb + cc, xv, mask=cm)


def _expand_fp64_inputs(A, B):
    """Split fp64 tensors into (hi, lo) fp32 pairs (value = hi + lo)."""
    A_hi = A.float()
    A_lo = (A - A_hi.double()).float()
    B_hi = B.float()
    B_lo = (B - B_hi.double()).float()
    return A_hi, A_lo, B_hi, B_lo


@libentry()
@triton.jit
def _trsm_update_xpu_kernel(
    A_ptr,
    B_ptr,
    N,
    K,
    R0,
    rsa,
    csb,
    KS: tl.constexpr,
    NS: tl.constexpr,
    BR: tl.constexpr,
    UPPER: tl.constexpr,
):
    """Phase 1 of the blocked sweep: subtract the already-solved rows [0, R0).

    grid = (batch * BR * NS,).  Every (row-in-block, column-slice) pair is an
    independent program, so this phase exposes BR times more parallelism than
    the substitution sweep, which is what lifts the shapes whose RHS is too
    narrow to fill the device with column slices alone.
    """
    pid = tl.program_id(0)
    sidx = pid % NS
    rl = (pid // NS) % BR
    bidx = pid // (NS * BR)
    abase = bidx * N * rsa
    bbase = bidx * N * csb + sidx * KS
    cc = tl.arange(0, KS)
    cm = cc < K
    t = R0 + rl
    row = N - 1 - t if UPPER else t
    acc = tl.load(B_ptr + bbase + row * csb + cc, mask=cm, other=0.0)
    arow = A_ptr + abase + row * rsa
    for u in range(R0):
        j = N - 1 - u if UPPER else u
        a = tl.load(arow + j)
        x = tl.load(B_ptr + bbase + j * csb + cc, mask=cm, other=0.0)
        acc = acc - a * x
    tl.store(B_ptr + bbase + row * csb + cc, acc, mask=cm)


@libentry()
@triton.jit
def _trsm_diag_xpu_kernel(
    A_ptr,
    B_ptr,
    N,
    K,
    R0,
    rsa,
    csb,
    KS: tl.constexpr,
    NS: tl.constexpr,
    BR: tl.constexpr,
    UPPER: tl.constexpr,
    UNIT: tl.constexpr,
):
    """Phase 2 of the blocked sweep: serial solve of the BR x BR diagonal block.

    grid = (batch * NS,).  The RHS rows of the block were already updated by
    phase 1 (previous launch => globally visible), so only the intra-block
    dependency chain remains.
    """
    pid = tl.program_id(0)
    sidx = pid % NS
    bidx = pid // NS
    abase = bidx * N * rsa
    bbase = bidx * N * csb + sidx * KS
    cc = tl.arange(0, KS)
    cm = cc < K
    for q in range(BR):
        t = R0 + q
        row = N - 1 - t if UPPER else t
        acc = tl.load(B_ptr + bbase + row * csb + cc, mask=cm, other=0.0)
        arow = A_ptr + abase + row * rsa
        for u in range(R0, t):
            j = N - 1 - u if UPPER else u
            a = tl.load(arow + j)
            x = tl.load(B_ptr + bbase + j * csb + cc, mask=cm, other=0.0)
            acc = acc - a * x
        if not UNIT:
            d = tl.load(arow + row)
            acc = acc * (1.0 / d)
        tl.store(B_ptr + bbase + row * csb + cc, acc, mask=cm)


@libentry()
@triton.jit
def _trsm_m_square_kernel(
    A_ptr,
    M_ptr,
    N,
    R0,
    rsa,
    msa,
    msa_b,
    BSB: tl.constexpr,
    NST: tl.constexpr,
    UPPER: tl.constexpr,
    UNIT: tl.constexpr,
    PAD: tl.constexpr,
):
    """M[k] = (D^-1 A_blk - I)^(2^k) for k = 0..NST-1 (M[0] = strict triangle)."""
    pid = tl.program_id(0)
    abase = pid * N * rsa
    mbase = pid * msa_b
    rows = tl.arange(0, BSB)
    t = R0 + rows
    if PAD:
        rm = t < N
        a_blk = tl.load(
            A_ptr + abase + t[:, None] * rsa + t[None, :],
            mask=rm[:, None] & rm[None, :],
            other=0.0,
        )
        diag = tl.load(A_ptr + abase + t * rsa + t, mask=rm, other=1.0)
    else:
        a_blk = tl.load(A_ptr + abase + t[:, None] * rsa + t[None, :])
        diag = tl.load(A_ptr + abase + t * rsa + t)
    if not UNIT:
        d_inv = (1.0 / diag)[:, None]
    else:
        d_inv = tl.full((BSB, 1), 1.0, dtype=a_blk.dtype)
    if UPPER:
        Mk = tl.where(rows[:, None] < rows[None, :], a_blk * d_inv, 0.0)
    else:
        Mk = tl.where(rows[:, None] > rows[None, :], a_blk * d_inv, 0.0)
    for k in tl.static_range(NST):
        tl.store(M_ptr + mbase + k * msa + rows[:, None] * BSB + rows[None, :], Mk)
        if k < NST - 1:
            Mk = tl.dot(Mk, Mk, input_precision="ieee")


@libentry()
@triton.jit
def _trsm_m_apply_kernel(
    A_ptr,
    M_ptr,
    B_ptr,
    N,
    R0,
    rsa,
    csb,
    msa,
    msa_b,
    NS: tl.constexpr,
    KS: tl.constexpr,
    BSB: tl.constexpr,
    UPPER: tl.constexpr,
    UNIT: tl.constexpr,
    NF: tl.constexpr,
    K0: tl.constexpr,
    PAD: tl.constexpr,
):
    """x = (I-M)(I+M^2)...(I+M^(2^(NF-1))) (D^-1 B_blk), factors K0..K0+NF-1.

    The factor for k == 0 is (I - M), all others are (I + M^(2^k)); with
    K0 > 0 the kernel applies the tail factors only (``q`` enters already
    scaled by D^-1 since the K0 == 0 launch stored its result back to B).
    """
    pid = tl.program_id(0)
    sidx = pid % NS
    bidx = pid // NS
    abase = bidx * N * rsa
    bbase = bidx * N * csb + sidx * KS
    mbase = bidx * msa_b
    rows = tl.arange(0, BSB)
    cn = tl.arange(0, KS)
    t = R0 + rows
    if PAD:
        rm = t < N
        b = tl.load(
            B_ptr + bbase + t[:, None] * csb + cn[None, :], mask=rm[:, None], other=0.0
        )
    else:
        b = tl.load(B_ptr + bbase + t[:, None] * csb + cn[None, :])
    if K0 == 0:
        if not UNIT:
            if PAD:
                diag = tl.load(A_ptr + abase + t * rsa + t, mask=rm, other=1.0)
            else:
                diag = tl.load(A_ptr + abase + t * rsa + t)
            d_inv = (1.0 / diag)[:, None]
        else:
            d_inv = tl.full((BSB, 1), 1.0, dtype=B_ptr.dtype.element_ty)
        q = b * d_inv
    else:
        q = b
    for k in tl.static_range(K0, K0 + NF):
        Mk = tl.load(M_ptr + mbase + k * msa + rows[:, None] * BSB + rows[None, :])
        d = tl.where(k == 0, -1.0, 1.0)
        q = q + d * tl.dot(Mk, q, input_precision="ieee")
    if PAD:
        tl.store(B_ptr + bbase + t[:, None] * csb + cn[None, :], q, mask=rm[:, None])
    else:
        tl.store(B_ptr + bbase + t[:, None] * csb + cn[None, :], q)


@libentry()
@triton.jit
def _trsm_window_dot_kernel(
    A_ptr,
    B_ptr,
    N,
    J0,
    rsa,
    csb,
    WN: tl.constexpr,
    NS: tl.constexpr,
    KS: tl.constexpr,
    BSB: tl.constexpr,
    UPPER: tl.constexpr,
    PAD: tl.constexpr,
):
    """B[block J0+1+c] -= A[block J0+1+c, block J0] @ X[block J0], c = 0..WN-1.

    One CTA per (window row, k-slice) pair; the writes go to disjoint
    (block, slice) pairs so there is no read-modify-write race.
    """
    pid = tl.program_id(0)
    sidx = pid % NS
    c = (pid // NS) % WN
    bidx = pid // (NS * WN)
    abase = bidx * N * rsa
    bbase = bidx * N * csb + sidx * KS
    cn = tl.arange(0, KS)
    if UPPER:
        mrow = N - (J0 + c + 2) * BSB + tl.arange(0, BSB)
        mcol = N - (J0 + 1) * BSB + tl.arange(0, BSB)
    else:
        mrow = (J0 + 1 + c) * BSB + tl.arange(0, BSB)
        mcol = J0 * BSB + tl.arange(0, BSB)
    if PAD:
        rmr = (mrow >= 0) & (mrow < N)
        rmc = (mcol >= 0) & (mcol < N)
        a = tl.load(
            A_ptr + abase + mrow[:, None] * rsa + mcol[None, :],
            mask=rmr[:, None] & rmc[None, :],
            other=0.0,
        )
        x = tl.load(
            B_ptr + bbase + mcol[:, None] * csb + cn[None, :],
            mask=rmc[:, None],
            other=0.0,
        )
        b = tl.load(
            B_ptr + bbase + mrow[:, None] * csb + cn[None, :],
            mask=rmr[:, None],
            other=0.0,
        )
    else:
        a = tl.load(A_ptr + abase + mrow[:, None] * rsa + mcol[None, :])
        x = tl.load(B_ptr + bbase + mcol[:, None] * csb + cn[None, :])
        b = tl.load(B_ptr + bbase + mrow[:, None] * csb + cn[None, :])
    acc = tl.dot(a, x, input_precision="ieee")
    tl.store(B_ptr + bbase + mrow[:, None] * csb + cn[None, :], b - acc)


def _solve_tri_serial(A_view, B_view, unitriangular, upper, n, k, batch, orig_shape):
    """Two-phase serial substitution (fp32): the pre-dot-sweep production path.

    Kept as the fallback for shapes the dot sweep cannot handle (see
    _solve_tri_dot) and as reference.
    """
    kpad = ((k + KS_SLICE - 1) // KS_SLICE) * KS_SLICE
    nslices = kpad // KS_SLICE
    rsa = A_view.stride(1)
    if kpad == k:
        Bp = B_view.clone()
    else:
        Bp = torch.zeros((batch, n, kpad), dtype=A_view.dtype, device=A_view.device)
        if not tle_copy(B_view, Bp[:, :, :k]):
            torch.ops.aten._copy_from(B_view, Bp[:, :, :k], False)
    br = _pick_block_rows(n)
    if br:
        for r0 in range(0, n, br):
            if r0:
                _trsm_update_xpu_kernel[(batch * br * nslices,)](
                    A_view,
                    Bp,
                    n,
                    kpad,
                    r0,
                    rsa,
                    kpad,
                    KS=KS_SLICE,
                    NS=nslices,
                    BR=br,
                    UPPER=bool(upper),
                    num_warps=4,
                )
            _trsm_diag_xpu_kernel[(batch * nslices,)](
                A_view,
                Bp,
                n,
                kpad,
                r0,
                rsa,
                kpad,
                KS=KS_SLICE,
                NS=nslices,
                BR=br,
                UPPER=bool(upper),
                UNIT=bool(unitriangular),
                num_warps=4,
            )
    else:
        _trsm_slice_xpu_kernel[(batch * nslices,)](
            A_view,
            Bp,
            A_view,
            A_view,
            Bp,
            Bp,
            n,
            kpad,
            rsa,
            kpad,
            F64=False,
            UNIT=bool(unitriangular),
            KS=KS_SLICE,
            UPPER=bool(upper),
            NS=nslices,
            num_warps=4,
        )
    return Bp[:, :, :k].reshape(orig_shape)


def _solve_tri_dot(A_view, B_view, unitriangular, upper, n, k, batch, orig_shape):
    """Blocked Neumann-sweep solve (fp32, n >= 2).

    ``bs`` (and hence the number of stored M powers) is 32 for n <= 32 and 64
    otherwise; the RHS slice width is 32 for k <= 32 (16 for k <= 16) so a
    16-column RHS never triggers a host-side column padding.

    Every block except possibly the last (in sweep order) is full-size, and
    for exact n multiples of ``bs`` every block is full-size.  A ragged
    UPPER sweep would need a tile shifted to negative row offsets, and XPU
    masked loads do not protect against negative offsets (they read adjacent
    memory - measured), so such shapes fall back to _solve_tri_serial.
    """
    bs = 32 if n <= 32 else 64
    if n % bs != 0 and (n + bs - 1) // bs > 1:
        return _solve_tri_serial(
            A_view, B_view, unitriangular, upper, n, k, batch, orig_shape
        )
    nst = 5 if bs == 32 else 6
    ks = 64 if k > 32 else (32 if k > 16 else 16)
    kpad = ((k + ks - 1) // ks) * ks
    nslices = kpad // ks
    device = A_view.device
    dtype = A_view.dtype
    if kpad == k:
        Bp = B_view.clone()
    else:
        Bp = torch.zeros((batch, n, kpad), dtype=dtype, device=device)
        if not tle_copy(B_view, Bp[:, :, :k]):
            torch.ops.aten._copy_from(B_view, Bp[:, :, :k], False)
    rsa = A_view.stride(1)
    csb = kpad
    nb = (n + bs - 1) // bs
    pad = n % bs != 0
    Ms = torch.empty((batch, 6, bs, bs), dtype=dtype, device=device)
    msa = bs * bs
    msa_b = 6 * msa
    for j in range(nb):
        R0 = (n - (j + 1) * bs) if upper else j * bs
        if R0 < 0:
            R0 = 0
        _trsm_m_square_kernel[(batch,)](
            A_view,
            Ms,
            n,
            R0,
            rsa,
            msa,
            msa_b,
            BSB=bs,
            NST=nst,
            UPPER=bool(upper),
            UNIT=bool(unitriangular),
            PAD=bool(pad),
            num_warps=8,
        )
        if nst <= 5:
            _trsm_m_apply_kernel[(batch * nslices,)](
                A_view,
                Ms,
                Bp,
                n,
                R0,
                rsa,
                csb,
                msa,
                msa_b,
                NS=nslices,
                KS=ks,
                BSB=bs,
                UPPER=bool(upper),
                UNIT=bool(unitriangular),
                NF=nst,
                K0=0,
                PAD=bool(pad),
                num_warps=8,
            )
        else:
            _trsm_m_apply_kernel[(batch * nslices,)](
                A_view,
                Ms,
                Bp,
                n,
                R0,
                rsa,
                csb,
                msa,
                msa_b,
                NS=nslices,
                KS=ks,
                BSB=bs,
                UPPER=bool(upper),
                UNIT=bool(unitriangular),
                NF=3,
                K0=0,
                PAD=bool(pad),
                num_warps=8,
            )
            _trsm_m_apply_kernel[(batch * nslices,)](
                A_view,
                Ms,
                Bp,
                n,
                R0,
                rsa,
                csb,
                msa,
                msa_b,
                NS=nslices,
                KS=ks,
                BSB=bs,
                UPPER=bool(upper),
                UNIT=bool(unitriangular),
                NF=3,
                K0=3,
                PAD=bool(pad),
                num_warps=8,
            )
        if j + 1 < nb:
            _trsm_window_dot_kernel[(batch * (nb - 1 - j) * nslices,)](
                A_view,
                Bp,
                n,
                j,
                rsa,
                csb,
                WN=nb - 1 - j,
                NS=nslices,
                KS=ks,
                BSB=bs,
                UPPER=bool(upper),
                PAD=bool(pad),
                num_warps=8,
            )
    return Bp[:, :, :k].reshape(orig_shape)


def _pick_block_rows(n):
    """Row-block size for the two-phase sweep, or 0 to keep the single sweep.

    Measured on XPU 6 (fp32, KS=64, torch-reference floor in brackets):
      n=16  single 0.175 ms  vs BR=8 0.209  -> single wins (launch bound)
      n=32  single 0.460     vs BR=8 0.411  -> ~tie, keep single (fewer launches)
      n=64  single 1.663     vs BR=16 1.056
      n=128 single 6.548     vs BR=32 4.467
      n=256 single 26.34     vs BR=32 23.86
      n=512 single 94.44     vs BR=32 45.21
    BR must divide n exactly: a ragged tail block would need the block length as
    a runtime kernel argument, and extra runtime scalars are a 15-30x cliff on
    this backend.
    """
    if n < 64:
        return 0
    cap = 16 if n < 128 else 32
    for br in (32, 16, 8):
        if br <= cap and n % br == 0:
            return br
    return 0


def _solve_tri(A, B, unitriangular, upper):
    """Solve A X = B with A triangular (A, B contiguous)."""
    n, k = A.shape[-1], B.shape[-1]
    if n == 1:
        if not unitriangular:
            inv = (1.0 / A[..., 0, 0]).reshape(-1)
            X = B.reshape(-1, k) * inv[:, None]
            X = X.reshape(B.shape)
        else:
            X = B.clone()
        return X
    is_fp64 = A.dtype == torch.float64
    orig_shape = B.shape

    batch = 1
    for d in A.shape[:-2]:
        batch *= d
    A_view = A.reshape(batch, n, n)
    B_view = B.reshape(batch, n, k)

    rsa = A_view.stride(1)
    kpad = ((k + KS_SLICE - 1) // KS_SLICE) * KS_SLICE
    nslices = kpad // KS_SLICE
    grid = (batch * nslices,)
    if is_fp64:
        Ah, Al, Bh0, Bl0 = _expand_fp64_inputs(A_view, B_view)
        Bh = torch.zeros((batch, n, kpad), dtype=torch.float32, device=A.device)
        Bl = torch.zeros((batch, n, kpad), dtype=torch.float32, device=A.device)
        if not tle_copy(Bh0, Bh[:, :, :k]):
            torch.ops.aten._copy_from(Bh0, Bh[:, :, :k], False)
        if not tle_copy(Bl0, Bl[:, :, :k]):
            torch.ops.aten._copy_from(Bl0, Bl[:, :, :k], False)
        _trsm_slice_xpu_kernel[grid](
            A_view,
            B_view,
            Ah,
            Al,
            Bh,
            Bl,
            n,
            kpad,
            rsa,
            kpad,
            F64=True,
            UNIT=bool(unitriangular),
            KS=KS_SLICE,
            UPPER=bool(upper),
            NS=nslices,
            num_warps=4,
        )
        X = (Bh[:, :, :k].to(torch.float64) + Bl[:, :, :k].to(torch.float64)).reshape(
            orig_shape
        )
    else:
        if kpad == k:
            Bp = B_view.clone()
        else:
            Bp = torch.zeros((batch, n, kpad), dtype=A.dtype, device=A.device)
            if not tle_copy(B_view, Bp[:, :, :k]):
                torch.ops.aten._copy_from(B_view, Bp[:, :, :k], False)
        if n % 16 == 0 and n >= 64:
            bs = 16
            nb = n // bs
            for j in range(nb):
                if j:
                    _trsm_window_dot_kernel[(batch * (nb - j) * nslices,)](
                        A_view,
                        Bp,
                        n,
                        j - 1,
                        rsa,
                        kpad,
                        WN=nb - j,
                        NS=nslices,
                        KS=KS_SLICE,
                        BSB=bs,
                        UPPER=bool(upper),
                        PAD=False,
                        num_warps=8,
                    )
                _trsm_diag_xpu_kernel[(batch * nslices,)](
                    A_view,
                    Bp,
                    n,
                    kpad,
                    j * bs,
                    rsa,
                    kpad,
                    KS=KS_SLICE,
                    NS=nslices,
                    BR=bs,
                    UPPER=bool(upper),
                    UNIT=bool(unitriangular),
                    num_warps=4,
                )
        else:
            br = _pick_block_rows(n)
            if br:
                for r0 in range(0, n, br):
                    if r0:
                        _trsm_update_xpu_kernel[(batch * br * nslices,)](
                            A_view,
                            Bp,
                            n,
                            kpad,
                            r0,
                            rsa,
                            kpad,
                            KS=KS_SLICE,
                            NS=nslices,
                            BR=br,
                            UPPER=bool(upper),
                            num_warps=4,
                        )
                    _trsm_diag_xpu_kernel[(batch * nslices,)](
                        A_view,
                        Bp,
                        n,
                        kpad,
                        r0,
                        rsa,
                        kpad,
                        KS=KS_SLICE,
                        NS=nslices,
                        BR=br,
                        UPPER=bool(upper),
                        UNIT=bool(unitriangular),
                        num_warps=4,
                    )
            else:
                _trsm_slice_xpu_kernel[grid](
                    A_view,
                    Bp,
                    A_view,
                    A_view,
                    Bp,
                    Bp,
                    n,
                    kpad,
                    rsa,
                    kpad,
                    F64=False,
                    UNIT=bool(unitriangular),
                    KS=KS_SLICE,
                    UPPER=bool(upper),
                    NS=nslices,
                    num_warps=4,
                )
        X = Bp[:, :, :k].reshape(orig_shape)
    return X


def linalg_solve_triangular(A, B, *, upper, left=True, unitriangular=False, out=None):
    """Solve A X = B (left) or X A = B (right) with triangular A on XPU."""
    logger.debug("GEMS_KUNLUNXIN LINALG_SOLVE_TRIANGULAR")
    if A.dtype not in (torch.float32, torch.float64):
        raise ValueError("linalg_solve_triangular only supports float32 and float64")
    if B.dtype != A.dtype:
        raise ValueError("A and B must have the same dtype")
    if A.ndim < 2 or B.ndim < 2:
        raise ValueError("A and B must be at least 2D")
    if A.shape[-1] != A.shape[-2]:
        raise ValueError("A must be a square matrix")

    if A.numel() == 0 or B.numel() == 0:
        if out is not None:
            if not tle_copy(B, out):
                torch.ops.aten._copy_from(B, out, False)
            return out
        return B.clone()

    if not left:
        if A.shape[-1] != B.shape[-1]:
            raise ValueError("Shape mismatch for XA=B")
        result = linalg_solve_triangular(
            A.mT.contiguous(),
            B.mT.contiguous(),
            upper=not upper,
            left=True,
            unitriangular=unitriangular,
        )
        result = result.mT.contiguous()
        if out is not None:
            if not tle_copy(result, out):
                torch.ops.aten._copy_from(result, out, False)
            return out
        return result

    A = A.contiguous()
    B = B.contiguous()

    batch_shape = torch.broadcast_shapes(A.shape[:-2], B.shape[:-2])
    if B.shape[:-2] != batch_shape:
        B = B.expand(batch_shape + B.shape[-2:]).contiguous()

    if upper:
        X = _solve_tri(A, B, unitriangular, True)
    else:
        X = _solve_tri(A, B, unitriangular, False)

    if out is not None:
        if not tle_copy(X, out):
            torch.ops.aten._copy_from(X, out, False)
        return out
    return X


def linalg_solve_triangular_out(
    A, B, *, upper, left=True, unitriangular=False, out=None
):
    return linalg_solve_triangular(
        A, B, upper=upper, left=left, unitriangular=unitriangular, out=out
    )
