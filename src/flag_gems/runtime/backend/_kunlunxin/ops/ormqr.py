# Kunlunxin(XPU) backend override for ormqr.
#
# Why a vendor file at all: the generic implementation is not compilable on the
# XPU/FlagTree backend. Empirically (probes on P800, 2026-08-20):
#   1. tl.sum(..., axis=0) on 2D tiles -> hard compile error
#      ("axis must not be 0 for 2D+ shapes").
#   2. tl.trans / transposed (N,M)-tile loads -> ClusterLayoutAttr rank assert.
#   3. Any masked 2D load feeding tl.dot -> TritonXPUVectorize crash.
#   4. tl.dot with fp64 unsupported; scalar loads inside runtime loops are
#      silently mis-compiled (every row loads the first value).
#   5. 1D vectors wider than 64 lanes derived from strided pointer arithmetic
#      silently return wrong values; more than one static `tl.reduce` chunk per
#      iteration, or static unrolls beyond 64 lanes, hit backend pass crashes
#      ("Failed to legalize tt.reduce" / TritonXPUMemoryCache).
#
# Design (probe-verified exact on P800): every reflector application uses 1D
# vectors of width 64, a runtime row loop and a SINGLE static reduce per kernel
# launch:
#   - BR == 64 spans      : fused kernel (w per row + reflect in one pass).
#   - wider spans         : per 64-lane chunk: one w-partial launch + one
#                           update launch. w partials go to a fp32 scratch
#                           whose rows are 64-aligned (2*WCOL wide: a hi/lo
#                           double-double pair per chunk) so the update kernel
#                           reads exact fp64 w values with one reduce per group.
# All dot/update arithmetic accumulates in fp64 in-register (tl.sum on fp64 is
# exact on this backend); only the final store casts to the storage dtype
# (platform downgrades fp64 tensor allocations, so intermediates live in fp32
# memory).
# Both reflector sides share the same canonical trailing-columns form:
#   - right mode: C @ (I - tau v v^T)
#   - left  mode: (I - tau v v^T) @ C  ==  C^T @ (I - tau v v^T) applied by the
#     same kernels on the transpose of a row-padded C.
# v[0] == 1 is implicit in the packed reflector format; the storage diagonal
# (which holds the reflector norm otherwise) is overwritten with 1.0 on a
# private copy of the input. All out-of-range accesses are zero-padded
# host-side. The Householder application order (left/right x transpose index
# orders) is unchanged from the generic wrapper.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Max safe 1D vector width on the XPU backend (probe-verified).
_XPU_VEC = 64

# ---------------------------------------------------------------------------
# Fast "sweep" path (2026-08-30, XPU 7).
#
# Both reflector sides are row-independent once the work matrix is laid out
# along the reflector direction:
#   right mode: C <- C H(i) ... : every ROW of C evolves on its own,
#               reflector direction = N.
#   left  mode: C <- H(i) ... C : every COLUMN of C evolves on its own,
#               reflector direction = M  (H is symmetric, so no transposed
#               reduction is ever needed - the data dependency is
#               single-direction in both modes).
# So one program can own one work row, keep it in registers for the WHOLE
# reflector sequence and reduce with a 1D -> scalar tl.sum. On TritonXPU a
# runtime-bound loop may only contain a *1D* reduction (2D axis=0/axis=1
# chains and loop-carried 2D tiles all fail to lower), which is exactly what
# this shape gives. Total launches: 4, independent of k.
#
# Probe-verified compile envelope of _ormqr_sweep_kernel (XPU 7, 2026-08-30) -
# NOT monotonic in the tile width, so only verified widths are whitelisted
# (same envelope as _kunlunxin/ops/linalg_householder_product.py):
#   fp32 accumulator: 64 OK, 128 OK, 256 FAIL, 512 FAIL, 1024 OK, 2048 OK,
#                     4096 FAIL   ('uni_sram' <- TritonXPUUnrollControl)
#   fp64 accumulator: 64 OK, 128 OK, >=256 FAIL ("'arith.mulf' op requires the
#                     same type for all operands and results")
# Reflector directions longer than the widest whitelisted width fall back to
# the per-reflector kernels below.
_SWEEP_WIDTHS = (64, 128, 1024, 2048)
_SWEEP_ACC64_MAX = 128


@libentry()
@triton.jit
def _ormqr_init_vu_kernel(
    IN_ptr,
    T_ptr,
    V_ptr,
    U_ptr,
    S_IB: tl.constexpr,
    S_IR: tl.constexpr,
    S_IK: tl.constexpr,
    S_TB: tl.constexpr,
    S_TK: tl.constexpr,
    IN_ROWS: tl.constexpr,
    K: tl.constexpr,
    LP: tl.constexpr,
):
    """Unpack the packed reflectors: V[b, i, r] = v_i[r], U = tau_i * v_i.

    v_i[r] = 0 (r < i), 1 (r == i), input[r, i] (r > i); lanes r >= IN_ROWS are
    zero so the dot product and the rank-1 update ignore the padding.  All
    strides are constexpr (the geqrf output is column major, and folding a
    non-unit stride into the *scalar* term of the address instead of the
    per-lane term silently transposes the read), and tau is folded into U so
    that the sweep kernel needs no scalar argument at all.
    Grid: (batch, k).
    """
    bid = tl.program_id(0)
    i = tl.program_id(1)
    r = tl.arange(0, LP)
    rc = tl.minimum(r, IN_ROWS - 1)
    val = tl.load(IN_ptr + bid * S_IB + rc * S_IR + i * S_IK)
    v = tl.where(r > i, val, 0.0)
    v = tl.where(r == i, 1.0, v)
    v = tl.where(r < IN_ROWS, v, 0.0)
    tau = tl.load(T_ptr + bid * S_TB + i * S_TK)
    base = (bid * K + i) * LP
    tl.store(V_ptr + base + r, v)
    tl.store(U_ptr + base + r, tau * v)


@libentry()
@triton.jit
def _ormqr_pack_kernel(
    X_ptr,
    C_ptr,
    S_CB: tl.constexpr,
    S_CM: tl.constexpr,
    S_CN: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    LEFT: tl.constexpr,
    LP: tl.constexpr,
):
    """X[b, row, :] <- one row (right mode) / one column (left mode) of C.

    The destination row is LP wide, 64-aligned and written with a single
    unmasked store; lanes past the reflector direction are zeroed so they can
    never contaminate the dot product.  Grid: (batch, ROWS).
    """
    bid = tl.program_id(0)
    row = tl.program_id(1)
    r = tl.arange(0, LP)
    if LEFT:
        rc = tl.minimum(r, M - 1)
        val = tl.load(C_ptr + bid * S_CB + rc * S_CM + row * S_CN)
        val = tl.where(r < M, val, 0.0)
    else:
        rc = tl.minimum(r, N - 1)
        val = tl.load(C_ptr + bid * S_CB + row * S_CM + rc * S_CN)
        val = tl.where(r < N, val, 0.0)
    tl.store(X_ptr + (bid * ROWS + row) * LP + r, val)


@libentry()
@triton.jit
def _ormqr_sweep_kernel(
    X_ptr,
    V_ptr,
    U_ptr,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    LP: tl.constexpr,
    REV: tl.constexpr,
    ACC64: tl.constexpr,
):
    """Apply the whole reflector sequence to one work row, in registers.

    x <- x - tau_i * (x . v_i) * v_i for every i, in the caller's order; the
    only reduction is 1D -> scalar, which is the single reduction form a
    runtime-bound loop accepts on this backend.  Grid: (batch, ROWS).
    """
    bid = tl.program_id(0)
    row = tl.program_id(1)
    r = tl.arange(0, LP)
    xbase = (bid * ROWS + row) * LP
    vbase = bid * K * LP
    if ACC64:
        x = tl.load(X_ptr + xbase + r).to(tl.float64)
        for t in range(0, K):
            if REV:
                i = K - 1 - t
            else:
                i = t
            v = tl.load(V_ptr + vbase + i * LP + r).to(tl.float64)
            u = tl.load(U_ptr + vbase + i * LP + r).to(tl.float64)
            x = x - tl.sum(x * v) * u
        tl.store(X_ptr + xbase + r, x.to(tl.float32))
    else:
        x = tl.load(X_ptr + xbase + r)
        for t in range(0, K):
            if REV:
                i = K - 1 - t
            else:
                i = t
            v = tl.load(V_ptr + vbase + i * LP + r)
            u = tl.load(U_ptr + vbase + i * LP + r)
            x = x - tl.sum(x * v) * u
        tl.store(X_ptr + xbase + r, x)


@libentry()
@triton.jit
def _ormqr_out_kernel(
    OUT_ptr,
    X_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    TOTAL: tl.constexpr,
    LEFT: tl.constexpr,
    LP: tl.constexpr,
):
    """Gather the flat contiguous result in 64-element unmasked chunks.

    The transpose of the left-mode work matrix is done on the LOAD side (a
    clamped, unmasked discrete gather); the store is a single affine 64-wide
    store into an over-allocated buffer, so no masked or narrow store exists.
    Grid: (ceil(TOTAL / 64),).
    """
    pid = tl.program_id(0)
    f = pid * 64 + tl.arange(0, 64)
    fc = tl.minimum(f, TOTAL - 1)
    b = fc // (M * N)
    rem = fc % (M * N)
    m = rem // N
    n = rem % N
    if LEFT:
        xoff = (b * N + n) * LP + m
    else:
        xoff = (b * M + m) * LP + n
    tl.store(OUT_ptr + f, tl.load(X_ptr + xoff))


def _sweep_width(length):
    """Smallest probe-verified sweep width that covers `length` (or None)."""
    for w in _SWEEP_WIDTHS:
        if w >= length:
            return w
    return None


def _flat_batch_stride(t, ndim_batch):
    """Element stride between consecutive flattened batch entries, or None."""
    if ndim_batch <= 0:
        return 0
    shape = t.shape[:ndim_batch]
    stride = t.stride()[:ndim_batch]
    expect = stride[-1]
    for i in range(ndim_batch - 1, 0, -1):
        expect = expect * shape[i]
        if stride[i - 1] != expect:
            return None
    return stride[-1]


def _ormqr_sweep(input, tau, other, left, transpose):
    """Sweep-path ormqr; returns None when this shape/layout is not covered."""
    if input.dtype != torch.float32 or other.dtype != torch.float32:
        return None
    if tau.dtype != torch.float32:
        return None
    if other.dim() < 2 or input.dim() < 2 or tau.dim() < 1:
        return None
    M, N = other.shape[-2], other.shape[-1]
    k = tau.shape[-1]
    nb = other.dim() - 2
    if input.dim() - 2 != nb or tau.dim() - 1 != nb:
        return None
    if other.shape[:nb] != input.shape[:nb] or other.shape[:nb] != tau.shape[:nb]:
        return None
    B = 1
    for s in other.shape[:nb]:
        B *= s
    if k == 0 or M == 0 or N == 0 or B == 0:
        return None
    length = M if left else N
    LP = _sweep_width(length)
    if LP is None:
        return None
    s_cb = _flat_batch_stride(other, nb)
    s_ib = _flat_batch_stride(input, nb)
    s_tb = _flat_batch_stride(tau, nb)
    if s_cb is None or s_ib is None or s_tb is None:
        return None
    rows = N if left else M
    in_rows = input.shape[-2]
    dev = other.device
    # every element of V / U / X / OUT is written by the kernels below, so the
    # buffers are deliberately uninitialised (no gems `zeros` launch).
    V = torch.empty(B * k * LP, dtype=torch.float32, device=dev)
    U = torch.empty(B * k * LP, dtype=torch.float32, device=dev)
    X = torch.empty(B * rows * LP, dtype=torch.float32, device=dev)
    _ormqr_init_vu_kernel[(B, k)](
        input,
        tau,
        V,
        U,
        S_IB=s_ib,
        S_IR=input.stride(-2),
        S_IK=input.stride(-1),
        S_TB=s_tb,
        S_TK=tau.stride(-1),
        IN_ROWS=in_rows,
        K=k,
        LP=LP,
    )
    _ormqr_pack_kernel[(B, rows)](
        X,
        other,
        S_CB=s_cb,
        S_CM=other.stride(-2),
        S_CN=other.stride(-1),
        M=M,
        N=N,
        ROWS=rows,
        LEFT=left,
        LP=LP,
    )
    # reflector order, identical to the per-reflector path below
    rev = (not transpose) if left else transpose
    _ormqr_sweep_kernel[(B, rows)](
        X,
        V,
        U,
        ROWS=rows,
        K=k,
        LP=LP,
        REV=rev,
        ACC64=LP <= _SWEEP_ACC64_MAX,
    )
    total = B * M * N
    npad = ((total + 63) // 64) * 64
    OUT = torch.empty(npad, dtype=torch.float32, device=dev)
    _ormqr_out_kernel[(npad // 64,)](
        OUT,
        X,
        M=M,
        N=N,
        TOTAL=total,
        LEFT=left,
        LP=LP,
    )
    return OUT[:total].view(*other.shape)


@libentry()
@triton.jit
def _kunlunxin_reflect_fused_kernel(
    C_ptr,
    V_ptr,
    T_ptr,
    M,
    i,
    s_cb,
    s_cm,
    s_cn,
    s_vb,
    s_vm,
    s_vk,
    s_tb,
    NCHUNK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DT: tl.constexpr,
):
    """Apply one reflector (I - tau v v^T) to the trailing columns i.. of all
    rows (single 64-wide span; NCHUNK is always 1 here):
        w_m = C[m, i:] @ v ; C[m, i:] -= tau * w_m * v
    Grid: (batch,); runtime row loop.
    The dot/update accumulate in fp64 (probe-verified: fp64 tl.sum compiles and
    is exact on this backend) so the stored result is exact-fp32-rounded; only
    the final store casts back to the storage dtype.
    """
    bid = tl.program_id(0)
    tau = tl.load(T_ptr + bid * s_tb + i).to(tl.float64)
    col = tl.arange(0, BLOCK_N)
    v = tl.load(V_ptr + bid * s_vb + (i + col) * s_vm + i * s_vk)
    v64 = v.to(tl.float64)
    for m in range(0, M):
        crow64 = tl.load(C_ptr + bid * s_cb + m * s_cm + (i + col) * s_cn).to(
            tl.float64
        )
        w = tl.sum(crow64 * v64, axis=0, keep_dims=True)
        upd = crow64 - tau * tl.sum(w, axis=0) * v64
        tl.store(
            C_ptr + bid * s_cb + m * s_cm + (i + col) * s_cn,
            upd.to(tl.float32),
        )


@libentry()
@triton.jit
def _kunlunxin_w_one_kernel(
    C_ptr,
    V_ptr,
    W_ptr,
    M,
    i,
    s_cb,
    s_cm,
    s_cn,
    s_vb,
    s_vm,
    s_vk,
    s_wb,
    WCOL: tl.constexpr,
    CHUNK0: tl.constexpr,
    NCHUNK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DT: tl.constexpr,
):
    """w partial for one 64-lane chunk: w_m = sum_j C[m, i+j]*v[j], stored at
    W[row, CHUNK0] as a double-double (hi, lo) fp32 pair so the update kernel
    recovers the fp64 partial exactly. W rows are 2*WCOL wide (WCOL =
    64*WCH, hi plane then lo plane); lanes >= NCHUNK are never written and
    stay zero. One static chunk, runtime row loop. Grid: (batch,).
    """
    bid = tl.program_id(0)
    col = CHUNK0 * BLOCK_N + tl.arange(0, BLOCK_N)
    v = tl.load(V_ptr + bid * s_vb + (i + col) * s_vm + i * s_vk)
    v64 = v.to(tl.float64)
    for m in range(0, M):
        crow64 = tl.load(C_ptr + bid * s_cb + m * s_cm + (i + col) * s_cn).to(
            tl.float64
        )
        w = tl.sum(crow64 * v64, axis=0, keep_dims=True)
        wc = tl.sum(w, axis=0)  # fp64 scalar
        wh = wc.to(tl.float32)
        wl = (wc - wh.to(tl.float64)).to(tl.float32)
        base = bid * s_wb + m * (WCOL * 2)
        tl.store(W_ptr + base + CHUNK0, wh)
        tl.store(W_ptr + base + WCOL + CHUNK0, wl)


@libentry()
@triton.jit
def _kunlunxin_update_one_kernel(
    C_ptr,
    V_ptr,
    W_ptr,
    T_ptr,
    M,
    i,
    s_cb,
    s_cm,
    s_cn,
    s_vb,
    s_vm,
    s_vk,
    s_wb,
    s_tb,
    WCOL: tl.constexpr,
    CHUNK0: tl.constexpr,
    NCHUNK: tl.constexpr,
    WCH: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DT: tl.constexpr,
):
    """Apply one reflector column-chunk after all w parts are stored:
    C[m, i:] -= tau * (sum_c w[m, c]) * v.  The w row is folded back in
    64-lane groups (WCH = ceil(NCHUNK/64)); Grid: (batch,); runtime row loop.
    W rows are 2*WCOL = 2*64*WCH wide: hi plane then lo plane (double-double
    per chunk); every 64-lane group read lies inside a plane (unwritten pad
    lanes stay zero). A single tl.reduce per wg iteration sums hi+lo, and the
    whole accumulation is fp64.
    """
    bid = tl.program_id(0)
    tau = tl.load(T_ptr + bid * s_tb + i).to(tl.float64)
    col = CHUNK0 * BLOCK_N + tl.arange(0, BLOCK_N)
    v = tl.load(V_ptr + bid * s_vb + (i + col) * s_vm + i * s_vk)
    v64 = v.to(tl.float64)
    for m in range(0, M):
        wsum = tl.zeros([1], dtype=tl.float64)
        for wg in tl.static_range(0, WCH):
            base = bid * s_wb + m * (WCOL * 2)
            hi = tl.load(W_ptr + base + wg * 64 + tl.arange(0, 64))
            lo = tl.load(W_ptr + base + WCOL + wg * 64 + tl.arange(0, 64))
            wsum += tl.sum(
                hi.to(tl.float64) + lo.to(tl.float64), axis=0, keep_dims=True
            )
        wsc = tl.sum(wsum, axis=0)
        crow64 = tl.load(C_ptr + bid * s_cb + m * s_cm + (i + col) * s_cn).to(
            tl.float64
        )
        upd = crow64 - tau * wsc * v64
        tl.store(
            C_ptr + bid * s_cb + m * s_cm + (i + col) * s_cn,
            upd.to(tl.float32),
        )


def _set_diag(v_tensor, k):
    """Write 1.0 on the diagonal of the packed reflector buffer (v[0] == 1)."""
    B_, R_, Cc_ = v_tensor.shape
    n = min(int(k), R_, Cc_)
    if n <= 0:
        return v_tensor
    bb = torch.arange(B_, device=v_tensor.device).view(B_, 1)
    ii = torch.arange(n, device=v_tensor.device).view(1, n)
    v_tensor[bb, ii, ii] = 1.0
    return v_tensor


def _pad_len(length, block):
    """Zero-pad a length up to the next multiple of `block`."""
    return ((int(length) + block - 1) // block) * block


def _make_w_scratch(B, rows, Wn, dtype, device):
    """W scratch: rows of 2*Wc fp32 lanes (hi plane + lo plane, 64-aligned)."""
    Wc = max(1, (int(Wn) + 63) // 64) * 64
    return torch.zeros(int(B), int(rows), 2 * int(Wc), dtype=dtype, device=device)


def _apply_one(
    C, V, T, W, rows, i, s_cb, s_cm, s_cn, s_vb, s_vm, s_vk, s_wb, s_tb, Nchunk, DT
):
    """Apply one reflector: fused for single-64 spans; otherwise one w-partial
    launch per 64-lane chunk followed by one update launch per chunk."""
    Bn = W.shape[0]
    if Nchunk == 1:
        _kunlunxin_reflect_fused_kernel[(Bn,)](
            C,
            V,
            T,
            rows,
            i,
            s_cb,
            s_cm,
            s_cn,
            s_vb,
            s_vm,
            s_vk,
            s_tb,
            NCHUNK=1,
            BLOCK_N=_XPU_VEC,
            DT=DT,
        )
        return
    for c in range(Nchunk):
        _kunlunxin_w_one_kernel[(Bn,)](
            C,
            V,
            W,
            rows,
            i,
            s_cb,
            s_cm,
            s_cn,
            s_vb,
            s_vm,
            s_vk,
            s_wb,
            WCOL=W.shape[2] // 2,
            CHUNK0=c,
            NCHUNK=Nchunk,
            BLOCK_N=_XPU_VEC,
            DT=DT,
        )
    for c in range(Nchunk):
        _kunlunxin_update_one_kernel[(Bn,)](
            C,
            V,
            W,
            T,
            rows,
            i,
            s_cb,
            s_cm,
            s_cn,
            s_vb,
            s_vm,
            s_vk,
            s_wb,
            s_tb,
            WCOL=W.shape[2] // 2,
            CHUNK0=c,
            NCHUNK=Nchunk,
            WCH=(Nchunk + 63) // 64,
            BLOCK_N=_XPU_VEC,
            DT=DT,
        )


def ormqr(input, tau, other, left=True, transpose=False):
    """Multiply a general matrix by the Householder-reflector product Q."""
    logger.debug("GEMS KUNLUNXIN ORMQR")
    assert input.dtype in (
        torch.float32,
        torch.float64,
    ), f"ormqr only supports float32 and float64, got {input.dtype}"
    DT = tl.float32 if input.dtype == torch.float32 else tl.float64

    fast = _ormqr_sweep(input, tau, other, bool(left), bool(transpose))
    if fast is not None:
        return fast

    C = other.clone().contiguous()
    input_c = input.contiguous()
    tau_c = tau.contiguous()

    if C.dim() > 2:
        batch_shape = C.shape[:-2]
        M, N = C.shape[-2], C.shape[-1]
        k = tau_c.shape[-1]
        B = 1
        for s in batch_shape:
            B *= s
        C_flat = C.reshape(B, M, N)
        input_flat = input_c.reshape(B, *input_c.shape[-2:])
        tau_flat = tau_c.reshape(B, k)
    else:
        M, N = C.shape
        k = tau_c.shape[0]
        B = 1
        C_flat = C.unsqueeze(0)
        input_flat = input_c.unsqueeze(0)
        tau_flat = tau_c.unsqueeze(0)

    if k == 0:
        return C

    two_d = C.dim() == 2

    if left:
        # (I - tau v v^T) @ C == C^T @ (I - tau v v^T): run the canonical
        # kernels on the transpose of a row-padded C. Reflector i covers
        # rows [i, M) of C, so the reflector vector has length M.
        BR = _pad_len(int(M), _XPU_VEC)
        Nchunk = BR // _XPU_VEC
        V_pad_rows = M + 2 * BR
        V_work = torch.zeros(
            B,
            V_pad_rows,
            input_flat.shape[-1],
            dtype=input.dtype,
            device=input.device,
        )
        torch.ops.aten._copy_from(
            input_flat, V_work[:, : input_flat.shape[-2], :], False
        )
        V_work = _set_diag(V_work, k)
        C_pad = torch.zeros(B, M + 2 * BR, N, dtype=C.dtype, device=C.device)
        torch.ops.aten._copy_from(C_flat, C_pad[:, :M, :], False)
        C_t = C_pad.transpose(1, 2)  # (B, N, M + 2*BR)
        s_cb, s_cm, s_cn = C_t.stride(0), C_t.stride(1), C_t.stride(2)
        s_vb, s_vm, s_vk = V_work.stride(0), V_work.stride(1), V_work.stride(2)
        s_tb = tau_flat.stride(0)
        indices = range(k) if transpose else range(k - 1, -1, -1)
        Wscr = _make_w_scratch(B, N, Nchunk, C.dtype, C.device)
        s_wb = Wscr.stride(0)
        for i in indices:
            _apply_one(
                C_t,
                V_work,
                tau_flat,
                Wscr,
                N,
                i,
                s_cb,
                s_cm,
                s_cn,
                s_vb,
                s_vm,
                s_vk,
                s_wb,
                s_tb,
                Nchunk,
                DT,
            )
        res = C_pad[:, :M, :]
        return res.squeeze(0) if two_d else res
    else:
        # C @ (I - tau v v^T): reflector i covers trailing columns [i, N).
        BR = _pad_len(int(N), _XPU_VEC)
        Nchunk = BR // _XPU_VEC
        N_pad = N + 2 * BR
        C_pad = torch.zeros(B, M, N_pad, dtype=C.dtype, device=C.device)
        torch.ops.aten._copy_from(C_flat, C_pad[:, :, :N], False)
        V_rows = input_flat.shape[-2]
        V_pad_rows = N + 2 * BR
        V_work = torch.zeros(
            B,
            max(V_pad_rows, V_rows),
            input_flat.shape[-1],
            dtype=input.dtype,
            device=input.device,
        )
        torch.ops.aten._copy_from(input_flat, V_work[:, :V_rows, :], False)
        V_work = _set_diag(V_work, k)
        s_cb, s_cm, s_cn = C_pad.stride(0), C_pad.stride(1), C_pad.stride(2)
        s_vb, s_vm, s_vk = V_work.stride(0), V_work.stride(1), V_work.stride(2)
        s_tb = tau_flat.stride(0)
        indices = range(k - 1, -1, -1) if transpose else range(k)
        Wscr = _make_w_scratch(B, M, Nchunk, C.dtype, C.device)
        s_wb = Wscr.stride(0)
        for i in indices:
            _apply_one(
                C_pad,
                V_work,
                tau_flat,
                Wscr,
                M,
                i,
                s_cb,
                s_cm,
                s_cn,
                s_vb,
                s_vm,
                s_vk,
                s_wb,
                s_tb,
                Nchunk,
                DT,
            )
        res2 = C_pad[:, :, :N]
        return res2.squeeze(0) if two_d else res2
