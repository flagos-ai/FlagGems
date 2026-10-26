"""Kunlunxin (XPU) linalg_lstsq.

The general implementation keeps a whole Householder QR tile in registers and
drives it with a device function that mixes three reduction shapes
(``tl.sum(.., axis=1)`` on a 2-D tile, ``tl.sum(.., axis=0)`` on a 2-D tile and
a 1-D -> scalar reduce).  None of its six kernels compiles on TritonXPU: every
launch dies in ``make_ttxir`` with ``OutOfResources: uni_sram`` even for a
16x4 input, so the whole operator is unusable on this backend.

This backend-local version keeps the same algorithm (Householder QR of the
augmented matrix for m >= n, minimum-norm QR of A^T for m < n, Q never formed
on the tall path) but expresses it entirely with primitives that this compiler
lowers correctly.  Measured on XPU 3 (probes in /tmp/lstsq_probe*.py) and XPU 5
(harness perf_ir/prof_linalg_lstsq_*):

* ``tl.sum(.., axis=0)`` on a 2-D tile is rejected outright ("axis must not be
  0 for 2D+ shapes"), and a 1-D -> scalar reduce in the same kernel as a 2-D
  ``axis=1`` reduce crashes CoreTiling.  => every reduction here is a per-row
  1-D ``tl.sum`` (one program per row); no kernel mixes the two shapes.
* a dynamic loop wrapped around a 2-D reduction always OOMs uni_sram.  => the
  Householder sweep is driven from the host, one launch per step.
* a square tile feeding a 2-D reduce OOMs uni_sram.  => tile rows are always far
  smaller than tile columns.
* broadcasting a 1-D vector along the LAST axis of a tile that is then STORED
  silently produces wrong values once that axis exceeds 128 elements.  => the
  rank-1 update is one program per row: a scalar coefficient times a 1-D row.
* a stride-0 duplicated-address 2-D load lowers to a per-lane gathered access
  (that is how the old ``_dot_kernel`` carried u, and it dominated the profile).
  => the projection ``u . W[c]`` is reduced from the row the update program
  already holds, so neither a gather nor a second pass over the matrix exists.
* ``tl.where`` inside a reduction operand, and a masked load feeding one, are
  both miscompiled.  => v is zeroed above the diagonal in ``_pivot_kernel``
  before it is ever used as a reduce operand.
* masked tail loads are unreliable, so every buffer is zero-padded to the tile
  shape and every load/store covers the whole tile unmasked.

Each sweep step is therefore two launches (``_pivot_kernel`` +
``_row_apply_kernel``), back-substitution is two, and the min-norm Q-apply is
one; the operator is launch-bound on small batch and reduction-bound on large
square shapes.

Zero padding is exact, not approximate: extra zero ROWS of the augmented
matrix leave R unchanged, and extra zero COLUMNS stay zero under every
reflector.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = (torch.float32, torch.float64)

_MAX_ROW = 8192

_LANES = 64

_MIN_ROW = 64
_MIN_COL = 64

# Rows per program in the fused reduce-and-update kernels.  One row per program
# leaves too little work per program for the 12-CTA grid on the batched
# small-width cases; R rows reuse the reflector vector and amortise the reduce
# tree.  Row counts handed to these kernels are multiples of _LANES, so R
# divides them exactly and no tail masking is needed.  Wide rows already give a
# program plenty of work, so they take the larger factor (fewer waves, more
# reflector reuse); the narrow back-substitution / Q-apply rows keep it small so
# their grids still cover the CTAs.
_ROWS_WIDE = 8
_ROWS = 4


def _rows(k, mp):
    r = _ROWS_WIDE if mp >= 1024 else _ROWS
    while r > 1 and k % r:
        r //= 2
    return r


def _p2(x):
    return 1 << (max(1, int(x)) - 1).bit_length()


def _ceil_to(x, m):
    return ((int(x) + m - 1) // m) * m


@libentry()
@triton.jit
def _pivot_kernel(
    W,
    V,
    VS,
    BETAS,
    ALPHA,
    BETA,
    DIAG,
    RMAX,
    J,
    NP,
    NCP: tl.constexpr,
    MP: tl.constexpr,
    KEEP: tl.constexpr,
):
    """Reflector J in ONE launch: v <- column J (zeroed above the diagonal),
    ||x||^2 by a 1-D reduce, the scalars, and (optionally) the saved reflector.

    Folding the norm here removes the whole ``_dot_kernel`` pass the old design
    ran over every trailing row just to read ``S[J]``: ||x||^2 is a reduction of
    the pivot row the program already holds.  Every access is a 1-D affine row,
    which is the "one program per row" reduce shape this backend lowers
    correctly; a 2-D axis=0 reduce is rejected outright.

    v is stored with its pivot lane already shifted (``x_J - alpha``), i.e. in
    the form the rank-1 update consumes, so the old separate scalar kernel that
    patched ``V[J]`` after the fact is gone too.
    """
    b = tl.program_id(0)
    r = tl.arange(0, MP)
    base = b * (NCP * MP) + J * MP
    x = tl.load(W + base + r)
    xj = tl.load(W + base + J)
    v = tl.where(r >= J, x, 0.0)
    nrm2 = tl.sum(v * v)
    alpha = -tl.where(xj >= 0.0, 1.0, -1.0) * tl.sqrt(tl.maximum(nrm2, 0.0))
    den = nrm2 - alpha * xj
    beta = tl.where(den > 0.0, 1.0 / tl.where(den > 0.0, den, 1.0), 0.0)
    u = tl.where(r == J, xj - alpha, v)
    tl.store(V + b * MP + r, u)
    tl.store(ALPHA + b, alpha)
    tl.store(BETA + b, beta)
    tl.store(DIAG + b * NP + J, alpha)
    prev = tl.load(RMAX + b * 2 + (J % 2))
    tl.store(RMAX + b * 2 + ((J + 1) % 2), tl.maximum(prev, tl.abs(alpha)))
    if KEEP:
        tl.store(VS + b * (NP * MP) + J * MP + r, u)
        tl.store(BETAS + b * NP + J, beta)


@libentry()
@triton.jit
def _row_apply_kernel(
    W,
    U,
    BETA,
    C0,
    J,
    NCP: tl.constexpr,
    MP: tl.constexpr,
    R: tl.constexpr,
):
    """One pass over the trailing matrix: W[c] <- (I - beta u u^T) W[c].

    ``s = u . W[c]`` is reduced in registers from the row this program already
    holds, so the old ``_dot_kernel`` + ``_wvec_kernel`` + ``_upd_kernel`` chain
    (three launches, and a stride-0 duplicated-address tile for u that lowered
    to a per-lane gather) collapses into a single load + store of each row.
    ``u`` already carries the shifted pivot, so ``beta * (u . W[c])`` equals the
    old ``beta * (S[c] - alpha * W[c, J])`` term exactly.

    Rows below J are finished rows of the factor; they must not change, and
    their coefficient is forced to 0 (the same no-op the old WV did).
    """
    b = tl.program_id(0)
    cbase = C0 + tl.program_id(1) * R
    r = tl.arange(0, MP)
    u = tl.load(U + b * MP + r)
    be = tl.load(BETA + b)
    for i in tl.range(0, R):
        c = cbase + i
        off = b * (NCP * MP) + c * MP + r
        row = tl.load(W + off)
        s = tl.sum(row * u)
        coef = tl.where(c >= J, be * s, 0.0)
        tl.store(W + off, row - coef * u)


@libentry()
@triton.jit
def _qapply_kernel(
    Z,
    VS,
    BETAS,
    J,
    NP,
    KP: tl.constexpr,
    MP: tl.constexpr,
    R: tl.constexpr,
):
    """Z[k] <- Z[k] - (betas[J] * (Z[k] . v_J)) v_J for the min-norm Q-apply.

    Same fused reduce-and-subtract as ``_row_apply_kernel``; folds the old
    ``_rowcpy_kernel`` + ``_dot_kernel`` + ``_scale_kernel`` + ``_upd_kernel``
    chain (four launches per reflector) into one.
    """
    b = tl.program_id(0)
    kbase = tl.program_id(1) * R
    r = tl.arange(0, MP)
    u = tl.load(VS + b * (NP * MP) + J * MP + r)
    be = tl.load(BETAS + b * NP + J)
    for i in tl.range(0, R):
        k = kbase + i
        off = b * (KP * MP) + k * MP + r
        row = tl.load(Z + off)
        s = tl.sum(row * u)
        tl.store(Z + off, row - (be * s) * u)


@libentry()
@triton.jit
def _upd_kernel(
    W,
    WV,
    V,
    C0,
    NCP: tl.constexpr,
    MP: tl.constexpr,
):
    """M <- H M, i.e. W[c, r] -= WV[c] * v[r], one program per row c.

    The rank-1 operands are carried by a SCALAR (WV[c]) and a 1-D vector (v[:])
    instead of the previous pair of stride-0 duplicated-address 2-D tiles. On
    this backend a stride-0 2-D load lowers to a per-lane gathered access, so
    the old form ran at the ~4.3 ns/elem gather wall even though the store was
    affine; measured 71-91% of the whole operator inside this kernel. Scaling a
    1-D row by a scalar keeps both the load and the store affine (~0.5 ns/elem)
    and does not broadcast a 1-D vector along the last axis of a stored tile,
    which is the shape this compiler miscompiles for >128 elements.
    """
    b = tl.program_id(0)
    c = C0 + tl.program_id(1)
    r = tl.arange(0, MP)
    off = b * (NCP * MP) + c * MP + r
    wv = tl.load(WV + b * NCP + c)
    v = tl.load(V + b * MP + r)
    tl.store(W + off, tl.load(W + off) - wv * v)


@libentry()
@triton.jit
def _solve_upd_kernel(
    RHS,
    W,
    XI,
    ROW,
    NCP: tl.constexpr,
    MP: tl.constexpr,
    NW: tl.constexpr,
    KP: tl.constexpr,
    R: tl.constexpr,
):
    """RHS[k] <- RHS[k] - XI[k] * W[ROW, :] in one launch.

    W[ROW, :] is row ROW of the factorised matrix, i.e. row ROW of R, so the old
    ``_rowcpy_kernel`` staging copy of it (one launch per back-substitution
    step) is gone; the row is read straight out of W with a 1-D affine load.
    """
    b = tl.program_id(0)
    kbase = tl.program_id(1) * R
    r = tl.arange(0, NW)
    u = tl.load(W + b * (NCP * MP) + ROW * MP + r)
    for i in tl.range(0, R):
        k = kbase + i
        off = b * (KP * NW) + k * NW + r
        row = tl.load(RHS + off)
        coef = tl.load(XI + b * KP + k)
        tl.store(RHS + off, row - coef * u)


@libentry()
@triton.jit
def _colcpy_kernel(SRC, DST, SBATCH, J, LD: tl.constexpr, NW: tl.constexpr):
    """DST[b, :NW] <- SRC[b, :NW, J]. A strided gather, which loads correctly
    here; it is a strided STORE that is not honoured."""
    b = tl.program_id(0)
    i = tl.arange(0, NW)
    tl.store(DST + b * NW + i, tl.load(SRC + b * SBATCH + i * LD + J))


@libentry()
@triton.jit
def _solve_step_kernel(
    RHS,
    XI,
    XS,
    DIAG,
    RMAX,
    RCOND,
    I,
    NSTEP,
    NW: tl.constexpr,
    KP: tl.constexpr,
):
    """XI[k] = XS[I, k] = RHS[k, I] / r_II, NaN on a negligible pivot.

    NaN (not 0) is deliberate: 0 is not the minimum-norm solution, so it would
    look plausible while being wrong.
    """
    b = tl.program_id(0)
    k = tl.arange(0, KP)
    r = tl.load(RHS + b * (KP * NW) + k * NW + I)
    rii = tl.load(DIAG + b * NW + I)
    tol = RCOND * tl.load(RMAX + b * 2 + (NSTEP % 2))
    bad = tl.abs(rii) <= tol
    xi = tl.where(bad, float("nan"), r / tl.where(bad, 1.0, rii))
    tl.store(XI + b * KP + k, xi)
    tl.store(XS + b * (NW * KP) + I * KP + k, xi)


@libentry()
@triton.jit
def _resprep_kernel(
    W,
    T,
    N,
    M,
    NCP: tl.constexpr,
    MP: tl.constexpr,
    KP: tl.constexpr,
):
    """Row n..m-1 of Q^T B, masked into its own buffer for the residual norm."""
    b = tl.program_id(0)
    k = tl.arange(0, KP)
    i = tl.arange(0, MP)
    v = tl.load(W + b * (NCP * MP) + (N + k[:, None]) * MP + i[None, :])
    v = tl.where((i[None, :] >= N) & (i[None, :] < M), v, 0.0)
    tl.store(T + b * (KP * MP) + k[:, None] * MP + i[None, :], v)


@libentry()
@triton.jit
def _res_kernel(T, RES, KP: tl.constexpr, MP: tl.constexpr):
    b = tl.program_id(0)
    k = tl.arange(0, KP)
    i = tl.arange(0, MP)
    t = tl.load(T + b * (KP * MP) + k[:, None] * MP + i[None, :])
    tl.store(RES + b * KP + k, tl.sum(t * t, axis=1))


def _qr_sweep(W, NCP, MP, NP, nsteps, batch, dt, dev, keep_reflectors):
    """Householder QR of the matrix whose transpose is W. Returns (DIAG, RMAX).

    When keep_reflectors is set, also returns (VS, BETAS) for the Q-apply of
    the minimum-norm path.

    Each sweep step is two launches: ``_pivot_kernel`` (reflector + norm +
    scalars) and ``_row_apply_kernel`` (the rank-1 update, which also performs
    the projection the old ``_dot_kernel`` pass computed separately).
    """
    BC = _LANES
    V = torch.zeros((batch, MP), dtype=dt, device=dev)
    ALPHA = torch.zeros((batch,), dtype=dt, device=dev)
    BETA = torch.zeros((batch,), dtype=dt, device=dev)
    DIAG = torch.zeros((batch, NP), dtype=dt, device=dev)
    RMAX = torch.zeros((batch, 2), dtype=dt, device=dev)
    if keep_reflectors:
        VS = torch.zeros((batch, NP, MP), dtype=dt, device=dev)
        BETAS = torch.zeros((batch, NP), dtype=dt, device=dev)
    else:
        # Unused on the tall path (KEEP is a constexpr False, so the stores are
        # dead code); a 1-element allocation avoids zero-filling an NP x MP
        # buffer that would never be read.
        VS = torch.empty(1, dtype=dt, device=dev)
        BETAS = torch.empty(1, dtype=dt, device=dev)

    for j in range(nsteps):
        c0 = (j // BC) * BC
        _pivot_kernel[(batch,)](
            W,
            V,
            VS,
            BETAS,
            ALPHA,
            BETA,
            DIAG,
            RMAX,
            j,
            NP,
            NCP=NCP,
            MP=MP,
            KEEP=keep_reflectors,
        )
        rw = _rows(NCP - c0, MP)
        _row_apply_kernel[(batch, (NCP - c0) // rw)](
            W, V, BETA, c0, j, NCP=NCP, MP=MP, R=rw
        )
    if not keep_reflectors:
        VS, BETAS = None, None
    return DIAG, RMAX, VS, BETAS


def _lstsq_tall(A, B, rcond):
    """A: (batch, m, n) with m >= n; B: (batch, m, nrhs). Both contiguous."""
    batch, m, n = A.shape
    nrhs = B.shape[-1]
    dt, dev = A.dtype, A.device

    KP = max(_LANES, _p2(nrhs))
    MP = max(_MIN_ROW, _p2(m))
    NP = max(_MIN_COL, _p2(n))
    NCP = max(NP, _ceil_to(n + KP, _LANES))

    W = torch.zeros((batch, NCP, MP), dtype=dt, device=dev)
    W[:, :n, :m] = A.transpose(-1, -2)
    W[:, n : n + nrhs, :m] = B.transpose(-1, -2)

    with torch_device_fn.device(dev):
        DIAG, RMAX, _, _ = _qr_sweep(W, NCP, MP, NP, n, batch, dt, dev, False)

        RHS = W[:, n : n + KP, :NP].contiguous()
        XS = torch.zeros((batch, NP, KP), dtype=dt, device=dev)
        XI = torch.zeros((batch, KP), dtype=dt, device=dev)
        for t in range(n):
            i = n - 1 - t
            _solve_step_kernel[(batch,)](
                RHS, XI, XS, DIAG, RMAX, rcond, i, n, NW=NP, KP=KP
            )
            rs = _rows(KP, NP)
            _solve_upd_kernel[(batch, KP // rs)](
                RHS, W, XI, i, NCP=NCP, MP=MP, NW=NP, KP=KP, R=rs
            )

        RES = torch.zeros((batch, KP), dtype=dt, device=dev)
        if m > n:
            T = torch.zeros((batch, KP, MP), dtype=dt, device=dev)
            _resprep_kernel[(batch,)](W, T, n, m, NCP=NCP, MP=MP, KP=KP)
            _res_kernel[(batch,)](T, RES, KP=KP, MP=MP)

    return XS[:, :n, :nrhs].contiguous(), RES[:, :nrhs].contiguous()


def _lstsq_wide(A, B, rcond):
    """A: (batch, m, n) with m < n; B: (batch, m, nrhs). Minimum-norm solution.

    A^T = Q R  ->  R^T y = b  ->  x = Q y. The working matrix is the transpose
    of A^T, i.e. A itself: rows index the columns of A^T (m of them), columns
    index its rows (n of them).
    """
    batch, m, n = A.shape
    nrhs = B.shape[-1]
    dt, dev = A.dtype, A.device

    KP = max(_LANES, _p2(nrhs))
    MP = max(_MIN_ROW, _p2(n))
    NRP = max(_MIN_COL, _p2(m))
    MIP = NRP

    W = torch.zeros((batch, NRP, MP), dtype=dt, device=dev)
    W[:, :m, :n] = A

    with torch_device_fn.device(dev):
        DIAG, RMAX, VS, BETAS = _qr_sweep(W, NRP, MP, NRP, m, batch, dt, dev, True)

        RHS = torch.zeros((batch, KP, MIP), dtype=dt, device=dev)
        RHS[:, :nrhs, :m] = B.transpose(-1, -2)
        YS = torch.zeros((batch, MIP, KP), dtype=dt, device=dev)
        YI = torch.zeros((batch, KP), dtype=dt, device=dev)
        COL = torch.zeros((batch, MIP), dtype=dt, device=dev)
        for c in range(m):
            _solve_step_kernel[(batch,)](
                RHS, YI, YS, DIAG, RMAX, rcond, c, m, NW=MIP, KP=KP
            )
            _colcpy_kernel[(batch,)](W, COL, NRP * MP, c, LD=MP, NW=MIP)
            _upd_kernel[(batch, KP)](RHS, YI, COL, 0, NCP=KP, MP=MIP)

        Z = torch.zeros((batch, KP, MP), dtype=dt, device=dev)
        Z[:, :nrhs, :m] = YS[:, :m, :nrhs].transpose(-1, -2)
        for t in range(m):
            j = m - 1 - t
            rs = _rows(KP, MP)
            _qapply_kernel[(batch, KP // rs)](Z, VS, BETAS, j, NRP, KP=KP, MP=MP, R=rs)

    return Z[:, :nrhs, :n].transpose(-1, -2).contiguous()


def _fallback(A, b, rcond, driver):
    """Reference path for inputs outside the native scope (a padded row wider
    than the validated reduction tile, or a malformed batch/RHS shape). Routed
    through CPU so it does not re-enter this override."""
    res = torch.linalg.lstsq(A.cpu(), b.cpu(), rcond=rcond, driver=driver)
    dev = A.device
    return (
        res.solution.to(dev),
        res.residuals.to(dev),
        res.rank.to(dev),
        res.singular_values.to(dev),
    )


def _empty_rank_sv(A):
    return (
        torch.empty(0, dtype=torch.int64, device=A.device),
        torch.empty(0, dtype=A.dtype, device=A.device),
    )


def linalg_lstsq(A, b, rcond=None, driver=None):
    logger.debug("GEMS_KUNLUNXIN LINALG_LSTSQ")

    if driver not in (None, "gels"):
        raise RuntimeError(
            "torch.linalg.lstsq: `driver` other than `gels` is not supported on CUDA"
        )

    if A.dtype not in _SUPPORTED_DTYPES or A.is_complex():
        raise NotImplementedError(
            f"linalg_lstsq: {A.dtype} is not supported on this device"
        )
    if A.dim() < 2 or b.dim() < 1:
        return _fallback(A, b, rcond, driver)

    m, n = A.shape[-2], A.shape[-1]

    dim_diff = A.dim() - b.dim()
    if dim_diff == 1 and tuple(b.shape) == tuple(A.shape[:-1]):
        vector_rhs, b2 = True, b.unsqueeze(-1)
    elif dim_diff == 0:
        vector_rhs, b2 = False, b
    else:
        return _fallback(A, b, rcond, driver)
    if b2.shape[-2] != m:
        return _fallback(A, b, rcond, driver)
    nrhs = b2.shape[-1]

    try:
        batch_shape = torch.broadcast_shapes(A.shape[:-2], b2.shape[:-2])
    except RuntimeError:
        return _fallback(A, b, rcond, driver)

    if m == 0 or n == 0 or nrhs == 0:
        solution = torch.zeros((*batch_shape, n, nrhs), dtype=A.dtype, device=A.device)
        if vector_rhs:
            solution = solution.squeeze(-1)
        if m > n:
            residuals = torch.zeros(
                (*batch_shape, nrhs), dtype=A.dtype, device=A.device
            )
        else:
            residuals = torch.empty(0, dtype=A.dtype, device=A.device)
        rank, singular_values = _empty_rank_sv(A)
        return solution, residuals, rank, singular_values

    if max(_p2(m), _p2(n)) > _MAX_ROW:
        return _fallback(A, b, rcond, driver)

    Af = A.expand(*batch_shape, m, n).reshape(-1, m, n).contiguous()
    Bf = b2.expand(*batch_shape, m, nrhs).reshape(-1, m, nrhs).contiguous()
    if rcond is None:
        rcond = torch.finfo(A.dtype).eps * max(m, n)

    if m < n:
        X = _lstsq_wide(Af, Bf, rcond)
        RES = None
    else:
        X, RES = _lstsq_tall(Af, Bf, rcond)

    solution = X.reshape(*batch_shape, n, nrhs)
    if vector_rhs:
        solution = solution.squeeze(-1)

    if m > n:
        residuals = RES.reshape(*batch_shape, nrhs)
    else:
        residuals = torch.empty(0, dtype=A.dtype, device=A.device)

    rank, singular_values = _empty_rank_sv(A)
    return solution, residuals, rank, singular_values
