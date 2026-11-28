"""Batched matrix determinant via LU factorization with partial pivoting.

Two paths, selected by matrix order, both keeping all arithmetic in device
kernels defined here:

* ``n <= _REG_MAX`` (128): one register-resident kernel. Each program holds
  BLOCK_M matrices as a BLOCK_M x BLOCK_N x BLOCK_N tile, performs
  in-register row swaps and keeps a running pivot product plus a swap parity.
  Nothing goes to scratch memory, so no block synchronisation is needed --
  important because tl.debug_barrier gives no cross-warp ordering on this
  backend (the same structure with scratch memory is exact at num_warps=1 and
  garbage at 2/4/8).

  The naive step needs seven cross-warp reductions (column extraction, pivot
  max, pivot index, two row extractions, two column-k scalars). Measured
  per-step cost is nearly independent of BLOCK_N, i.e. reduction latency, so
  they are fused into three multi-output reductions: the column extraction,
  one 4-output reduce returning max|col k| / its first row / the signed pivot
  / col k at row k, and one 2-output reduce returning both exchanging rows.
  This is the best measured configuration for every order up to 128.

* ``n > _REG_MAX``: blocked LU over a contiguous scratch copy. A panel
  factorisation kernel (one program per matrix, a single warp so the
  load-after-store chain inside a panel stays ordered) alternates with a
  trailing-update kernel parallel over (row-tile, col-tile) programs, with
  kernel-launch boundaries supplying the ordering. Above order 128 the
  register kernel cannot use more than 16 warps (device warp size is 64 and
  Triton caps a block at 1024 threads), and this blocked path measured
  1.6-1.9x faster there.

The input is read with its real strides and its batch index is decomposed in
the kernel from collapsed (size, stride) pairs, so no host-side copy,
reshape or dtype conversion of the input is needed.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Largest order handled by the register-resident kernel.
_REG_MAX = 128
# Slots available for the collapsed batch-dimension decomposition.
_ND = 6
# Blocked (n > _REG_MAX) path geometry. The panel phase runs in one warp, so
# its cost is dominated by the number of dependent global round trips, which is
# ceil(extent/BLOCK) per phase. A wide BLOCK keeps that near one per phase; the
# resulting BLOCK x PANEL update tile must stay register resident (2048
# elements = 64 per lane at one warp), and PANEL must stay >= 16 for tl.dot.
_PANEL = 16
_PBLOCK = 128
_PBLOCK1 = 256
_TBLOCK = 64
# Per-device warp cap discovered when a wide launch is rejected.
_WARP_CAP = {}


@triton.jit
def _base_off(m, h0, h1, h2, h3, h4, h5, t0, t1, t2, t3, t4, t5, NDIM: tl.constexpr):
    """Flat element offset of matrix ``m`` from collapsed batch sizes/strides."""
    r = m
    base = tl.zeros_like(m)
    if NDIM >= 1:
        base += (r % h0) * t0
        r = r // h0
    if NDIM >= 2:
        base += (r % h1) * t1
        r = r // h1
    if NDIM >= 3:
        base += (r % h2) * t2
        r = r // h2
    if NDIM >= 4:
        base += (r % h3) * t3
        r = r // h3
    if NDIM >= 5:
        base += (r % h4) * t4
        r = r // h4
    if NDIM >= 6:
        base += (r % h5) * t5
        r = r // h5
    return base


@triton.jit
def _pivot_comb(m1, i1, p1, s1, m2, i2, p2, s2):
    """Keep the first (smallest-row) maximum of |col k| together with its
    signed value, and accumulate the col-k entry that sits on row k."""
    use2 = (m2 > m1) | ((m2 == m1) & (i2 < i1))
    return (
        tl.where(use2, m2, m1),
        tl.where(use2, i2, i1),
        tl.where(use2, p2, p1),
        s1 + s2,
    )


@triton.jit
def _rows_comb(a1, b1, a2, b2):
    return a1 + a2, b1 + b2


@triton.jit
def _det_kernel(
    A,
    out,
    B,
    srow,
    scol,
    n,
    h0,
    h1,
    h2,
    h3,
    h4,
    h5,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    NDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_N)
    rowv = tl.broadcast_to(rows.to(tl.float32)[None, :], (BLOCK_M, BLOCK_N))

    m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mmask = m < B
    base = _base_off(
        tl.where(mmask, m, 0), h0, h1, h2, h3, h4, h5, t0, t1, t2, t3, t4, t5, NDIM
    )

    off = base[:, None, None] + rows[None, :, None] * srow + cols[None, None, :] * scol
    lmask = mmask[:, None, None] & (rows[None, :, None] < n) & (cols[None, None, :] < n)
    work = tl.load(A + off, mask=lmask, other=0.0)

    swap = tl.zeros((BLOCK_M,), dtype=tl.int32)
    prod = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)

    for k in range(0, n):
        # ---- one reduce: column k of every matrix ----
        colk = tl.sum(tl.where(cols[None, None, :] == k, work, 0.0), axis=2)

        # ---- one 4-output reduce: pivot value, its row, col k at row k ----
        absk = tl.where((rows[None, :] < k) | (rows[None, :] >= n), -1.0, tl.abs(colk))
        ksel = tl.where(rows[None, :] == k, colk, 0.0)
        _pmax, prowf, pivot, ck_k = tl.reduce((absk, rowv, colk, ksel), 1, _pivot_comb)
        prow = prowf.to(tl.int32)

        # ---- one 2-output reduce: the two rows that exchange places ----
        rowk, rowp = tl.reduce(
            (
                tl.where(rows[None, :, None] == k, work, 0.0),
                tl.where(rows[None, :, None] == prow[:, None, None], work, 0.0),
            ),
            1,
            _rows_comb,
        )
        work = tl.where(rows[None, :, None] == k, rowp[:, None, :], work)
        work = tl.where(
            rows[None, :, None] == prow[:, None, None], rowk[:, None, :], work
        )
        swap += (prow != k).to(tl.int32)

        # pivot is the (k, k) entry after the swap; ck_k is the entry the swap
        # moved down to row prow
        prod = prod * pivot
        safe = tl.where(pivot == 0.0, 1.0, pivot)
        colnew = tl.where(
            rows[None, :] == k,
            pivot[:, None],
            tl.where(rows[None, :] == prow[:, None], ck_k[:, None], colk),
        )
        # the multipliers are already zero outside the row range, so zeroing
        # the low columns of the U row is enough: two fused tile ops instead
        # of a separate 3-D select around the update
        mult = tl.where(rows[None, :] > k, colnew / safe[:, None], 0.0)
        urow = tl.where(cols[None, :] > k, rowp, 0.0)
        work = work - mult[:, :, None] * urow[:, None, :]

    det = tl.where(swap % 2 == 0, prod, -prod)
    tl.store(out + m, det.to(out.dtype.element_ty), mask=mmask)


@triton.jit
def _fill_ones_kernel(out, B, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out + off, 1.0, mask=off < B)


@triton.jit
def _gather_kernel(
    A,
    LU,
    srow,
    scol,
    n,
    h0,
    h1,
    h2,
    h3,
    h4,
    h5,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    NDIM: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pm = tl.program_id(0)
    pr = tl.program_id(1)
    mv = tl.zeros((1,), dtype=tl.int32) + pm
    offv = _base_off(mv, h0, h1, h2, h3, h4, h5, t0, t1, t2, t3, t4, t5, NDIM)

    r = pr * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = r < n
    mbase = pm * n * n
    for c0 in range(0, n, BLOCK_C):
        c = c0 + tl.arange(0, BLOCK_C)
        tmask = rmask[:, None] & (c[None, :] < n)
        v = tl.load(
            A + offv[:, None] + r[:, None] * srow + c[None, :] * scol,
            mask=tmask,
            other=0.0,
        )
        tl.store(LU + mbase + r[:, None] * n + c[None, :], v, mask=tmask)


@triton.jit
def _panel_kernel(
    LU,
    prod_p,
    sgn_p,
    N,
    k0,
    PANEL: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK1: tl.constexpr,
):
    """Factorise steps [k0, k0+PANEL) of one matrix inside a single warp.

    BLOCK1 sizes the width-N phases (pivot search, row swap) so that each of
    them costs a single dependent global round trip; BLOCK sizes the 2-D
    panel-update and L-strip tiles, whose BLOCK*PANEL element count has to stay
    around 2048 to remain register resident at one warp.
    """
    pid = tl.program_id(0)
    base = pid * N * N

    prod = tl.where(k0 == 0, 1.0, tl.load(prod_p + pid))
    sgn = tl.where(k0 == 0, 1.0, tl.load(sgn_p + pid))
    kend = tl.minimum(k0 + PANEL, N)

    for k in range(k0, kend):
        # partial pivot search over rows [k, N)
        best_val = tl.full((), -1.0, dtype=tl.float32)
        best_row = tl.zeros((), dtype=tl.int32) + k
        for i0 in range(k, N, BLOCK1):
            r = i0 + tl.arange(0, BLOCK1)
            col = tl.load(LU + base + r * N + k, mask=r < N, other=0.0)
            absc = tl.where((r >= k) & (r < N), tl.abs(col), -1.0)
            tmax = tl.max(absc, axis=0)
            trow = tl.min(tl.where(absc == tmax, r, N), axis=0)
            better = tmax > best_val
            best_row = tl.where(better, trow, best_row)
            best_val = tl.where(better, tmax, best_val)

        # swap rows k <-> best_row over the not-yet-eliminated columns
        for j0 in range(k0, N, BLOCK1):
            c = j0 + tl.arange(0, BLOCK1)
            cmask = c < N
            rk = tl.load(LU + base + k * N + c, mask=cmask, other=0.0)
            rp = tl.load(LU + base + best_row * N + c, mask=cmask, other=0.0)
            tl.store(LU + base + k * N + c, rp, mask=cmask)
            tl.store(LU + base + best_row * N + c, rk, mask=cmask)
        sgn = sgn * tl.where(best_row != k, -1.0, 1.0)

        tl.debug_barrier()

        pivot = tl.load(LU + base + k * N + k)
        prod = prod * pivot
        safe = tl.where(pivot == 0.0, 1.0, pivot)

        # scale column k and apply this step's rank-1 update inside the panel
        pcols = k + 1 + tl.arange(0, PANEL)
        pcmask = pcols < kend
        urow = tl.load(LU + base + k * N + pcols, mask=pcmask, other=0.0)
        for i0 in range(k + 1, N, BLOCK):
            r = i0 + tl.arange(0, BLOCK)
            rmask = r < N
            col = tl.load(LU + base + r * N + k, mask=rmask, other=0.0)
            lcol = tl.where(rmask, col / safe, 0.0)
            tl.store(LU + base + r * N + k, lcol, mask=rmask)
            tmask = rmask[:, None] & pcmask[None, :]
            tile = tl.load(
                LU + base + r[:, None] * N + pcols[None, :], mask=tmask, other=0.0
            )
            tile = tile - lcol[:, None] * urow[None, :]
            tl.store(LU + base + r[:, None] * N + pcols[None, :], tile, mask=tmask)

        tl.debug_barrier()

    # Panel rows over the columns the trailing-update kernel does not cover.
    # This is the forward substitution U[i, c] = LU[i, c] - sum_{j<i} L[i, j]
    # U[j, c] over i in [k0, kend). It is done from registers: the whole
    # PANEL x BLOCK column chunk and the small PANEL x PANEL L block are loaded
    # once, so the phase costs a handful of dependent round trips per panel
    # instead of one per (row, chunk) pair.
    prows = k0 + tl.arange(0, PANEL)
    prmask = prows < kend
    rowi = prows - k0
    off = rowi[:, None]
    lblk = tl.load(
        LU + base + prows[:, None] * N + prows[None, :],
        mask=prmask[:, None] & prmask[None, :],
        other=0.0,
    )
    for j0 in range(kend, N, BLOCK):
        cols = j0 + tl.arange(0, BLOCK)
        cmask = cols < N
        tmask = prmask[:, None] & cmask[None, :]
        tile = tl.load(
            LU + base + prows[:, None] * N + cols[None, :], mask=tmask, other=0.0
        )
        for c in range(k0, kend):
            ci = c - k0
            urow = tl.sum(tl.where(off == ci, tile, 0.0), axis=0)
            lcol = tl.sum(tl.where(rowi[None, :] == ci, lblk, 0.0), axis=1)
            lcol = tl.where(prows > c, lcol, 0.0)
            tile = tile - lcol[:, None] * urow[None, :]
        tl.store(LU + base + prows[:, None] * N + cols[None, :], tile, mask=tmask)

    tl.store(prod_p + pid, prod)
    tl.store(sgn_p + pid, sgn)


@triton.jit
def _trail_kernel(LU, N, k0, PANEL: tl.constexpr, BLOCK: tl.constexpr):
    """LU[rows, cols] -= L[rows, panel] @ U[panel, cols] for one tile."""
    pid = tl.program_id(0)
    base = pid * N * N
    kend = tl.minimum(k0 + PANEL, N)
    r = kend + tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    c = kend + tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    p = k0 + tl.arange(0, PANEL)

    rmask = r < N
    cmask = c < N
    pmask = p < kend
    # every row here lies below the whole panel, so the stored panel column is
    # the multiplier itself; the unit diagonal of L is never materialised
    lt = tl.load(
        LU + base + r[:, None] * N + p[None, :],
        mask=rmask[:, None] & pmask[None, :],
        other=0.0,
    )
    ut = tl.load(
        LU + base + p[:, None] * N + c[None, :],
        mask=pmask[:, None] & cmask[None, :],
        other=0.0,
    )
    tmask = rmask[:, None] & cmask[None, :]
    tile = tl.load(LU + base + r[:, None] * N + c[None, :], mask=tmask, other=0.0)
    tile = tile - tl.dot(lt, ut, input_precision="ieee")
    tl.store(LU + base + r[:, None] * N + c[None, :], tile, mask=tmask)


@triton.jit
def _final_kernel(out, prod_p, sgn_p, B, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < B
    p = tl.load(prod_p + off, mask=mask, other=0.0)
    s = tl.load(sgn_p + off, mask=mask, other=1.0)
    tl.store(out + off, (p * s).to(out.dtype.element_ty), mask=mask)


def _launch_reg(A, out_flat, B, srow, scol, n, hs, ts, nd, bm, nw, BN):
    """Launch the register kernel, dropping the warp count if the device
    rejects it (warp size is 64 here and Triton caps a block at 1024 threads,
    so a wide launch is not accepted everywhere; the outcome is cached per
    device as plain launch metadata)."""
    grid = (triton.cdiv(B, bm),)
    dev = 0 if A.device.index is None else int(A.device.index)
    cap = _WARP_CAP.get(dev)
    if cap is not None and nw > cap:
        nw = cap
    while True:
        try:
            _det_kernel[grid](
                A,
                out_flat,
                B,
                srow,
                scol,
                n,
                hs[0],
                hs[1],
                hs[2],
                hs[3],
                hs[4],
                hs[5],
                ts[0],
                ts[1],
                ts[2],
                ts[3],
                ts[4],
                ts[5],
                NDIM=nd,
                BLOCK_M=bm,
                BLOCK_N=BN,
                num_warps=nw,
            )
            return
        except Exception as exc:  # noqa: BLE001
            if nw > 1 and "out of resource" in str(exc):
                nw = max(1, nw // 2)
                _WARP_CAP[dev] = nw
                continue
            raise


def _collapse(shape, strides):
    """Collapse (size, stride) batch dims, innermost first."""
    dims = [(int(s), int(t)) for s, t in zip(shape, strides) if int(s) != 1]
    merged = []
    for s, t in dims:
        if merged and merged[-1][1] == t * s:
            ps, pt = merged.pop()
            merged.append((ps * s, t))
        else:
            merged.append((s, t))
    merged.reverse()
    return merged


def det(self):
    logger.debug("GEMS_ILUVATAR DET")
    A = self
    if A.dim() < 2:
        raise RuntimeError(
            "linalg.det: The input tensor A must have at least 2 dimensions."
        )
    m_, n = A.shape[-2], A.shape[-1]
    if m_ != n:
        raise RuntimeError(
            "linalg.det: A must be batches of square matrices, "
            f"but they are {m_} by {n} matrices"
        )
    if A.dtype not in (torch.float32, torch.float64):
        raise RuntimeError(
            f"linalg.det: Low precision dtypes not supported. Got {A.dtype}"
        )

    batch_shape = A.shape[:-2]
    B = 1
    for d in batch_shape:
        B *= int(d)

    if B == 0:
        return torch.empty(batch_shape, dtype=A.dtype, device=A.device)

    out = torch.empty(batch_shape, dtype=A.dtype, device=A.device)
    out_flat = out.view(-1)

    if n == 0:
        bs = 256
        _fill_ones_kernel[(triton.cdiv(B, bs),)](out_flat, B, BLOCK=bs)
        return out

    dims = _collapse(batch_shape, A.stride()[: len(batch_shape)])
    if len(dims) > _ND:
        raise RuntimeError("linalg.det: too many batch dimensions")
    hs = [1] * _ND
    ts = [0] * _ND
    for i, (s, t) in enumerate(dims):
        hs[i] = s
        ts[i] = t
    srow = int(A.stride(-2))
    scol = int(A.stride(-1))

    if n <= _REG_MAX:
        BN = max(8, triton.next_power_of_2(n))
        # keep at least ~16 programs so small batches still spread over the
        # SMs, and never let the register tile exceed ~8192 elements
        bm = min(min(32, max(1, 8192 // (BN * BN))), max(1, B // 16))
        if BN <= 64:
            nw = max(1, min(8, (bm * BN * BN) // 512))
        else:
            bm = 1
            nw = 16
        _launch_reg(A, out_flat, B, srow, scol, n, hs, ts, len(dims), bm, nw, BN)
        return out

    LU = torch.empty((B, n, n), dtype=A.dtype, device=A.device)
    _gather_kernel[(B, triton.cdiv(n, _PBLOCK))](
        A,
        LU,
        srow,
        scol,
        n,
        hs[0],
        hs[1],
        hs[2],
        hs[3],
        hs[4],
        hs[5],
        ts[0],
        ts[1],
        ts[2],
        ts[3],
        ts[4],
        ts[5],
        NDIM=len(dims),
        BLOCK_R=_PBLOCK,
        BLOCK_C=_PBLOCK,
        num_warps=8,
    )

    prod = torch.empty(B, dtype=torch.float32, device=A.device)
    sgn = torch.empty(B, dtype=torch.float32, device=A.device)
    for k0 in range(0, n, _PANEL):
        kend = min(k0 + _PANEL, n)
        _panel_kernel[(B,)](
            LU,
            prod,
            sgn,
            n,
            k0,
            PANEL=_PANEL,
            BLOCK=_PBLOCK,
            BLOCK1=_PBLOCK1,
            num_warps=1,
        )
        nend = n - kend
        if nend > 0:
            nt = triton.cdiv(nend, _TBLOCK)
            _trail_kernel[(B, nt, nt)](
                LU, n, k0, PANEL=_PANEL, BLOCK=_TBLOCK, num_warps=4
            )

    fb = 256
    _final_kernel[(triton.cdiv(B, fb),)](out_flat, prod, sgn, B, BLOCK=fb)
    return out
