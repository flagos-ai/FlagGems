# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
lowers correctly.  Measured on XPU 3 (probes in /tmp/lstsq_probe*.py):

* ``tl.sum(.., axis=0)`` on a 2-D tile is rejected outright ("axis must not be
  0 for 2D+ shapes"), and a 1-D -> scalar reduce in the same kernel as a 2-D
  ``axis=1`` reduce crashes CoreTiling.  => every reduction here is a single
  2-D ``axis=1`` ``tl.sum``; scalars are carried through global memory.
* a dynamic loop wrapped around a 2-D reduction always OOMs uni_sram.  => the
  reduction tile spans the full padded row (validated to 4096 wide), and the
  Householder sweep is driven from the host, one launch per step.
* a square tile feeding a 2-D reduce OOMs uni_sram.  => tile rows (BC, KP) are
  always far smaller than tile columns (MP).
* broadcasting a 1-D vector along the LAST axis of a tile that is then STORED
  silently produces wrong values once that axis exceeds 128 elements.  => the
  rank-1 update loads both operands as stride-0 duplicated-address 2-D tiles.
* ``tl.where`` inside a reduction operand, and a masked load feeding one, are
  both miscompiled.  => anything that needs masking is pre-zeroed into its own
  scratch buffer by a separate elementwise kernel.
* reading address X of a buffer while storing to address Y != X of the SAME
  buffer inside one kernel returns stale data.  => read and write buffers are
  kept disjoint in every kernel.
* masked tail loads are unreliable, so every buffer is zero-padded to the tile
  shape and every load/store covers the whole tile unmasked.

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

# Largest padded row length a single reduction tile may span. 8192 is the
# widest validated on XPU 3; beyond it the reduce is untested, so those shapes
# take the reference path rather than risk a silent miscompile.
_MAX_ROW = 8192

# EVERY vector store on this backend writes exactly 64 contiguous elements,
# whatever length was asked for (measured: a 1-, 8- or 32-lane store all touch
# 64 slots; only a 0-d store touches one). So every tile row, every per-batch
# vector and every RHS block is a multiple of 64 wide, and the c-tile is exactly
# 64 rows so that neighbouring programs cannot overwrite each other's results.
_LANES = 64

# Floors on the padded tile extents. A 32-element-wide tile that is both read
# and written in one kernel comes back wrong on this backend (validated:
# 32 bad, 64 and 128 good), so every working buffer is padded to at least this.
_MIN_ROW = 64
_MIN_COL = 64


def _p2(x):
    return 1 << (max(1, int(x)) - 1).bit_length()


# ---------------------------------------------------------------------------
# One Householder step of the QR of the working matrix W.
#
# W is the TRANSPOSE of the matrix being factored: W[c, r] = M[r, c], shape
# (batch, NCP, MP).  Row c of W is column c of M, so "the pivot column" is a
# contiguous row of W and every reduction runs over the LAST axis.
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def _mk_v_kernel(
    W,
    V,
    XJ,
    J,
    NCP: tl.constexpr,
    MP: tl.constexpr,
):
    """v <- pivot column of M, zeroed above the diagonal; xj <- v[J].

    The zeroing has to happen here, in its own elementwise kernel writing its
    own buffer: masking it inside the dot product below is miscompiled.
    """
    b = tl.program_id(0)
    r = tl.arange(0, MP)
    base = b * (NCP * MP) + J * MP
    x = tl.load(W + base + r)
    tl.store(V + b * MP + r, tl.where(r >= J, x, 0.0))
    tl.store(XJ + b, tl.load(W + base + J))


@libentry()
@triton.jit
def _dot_kernel(
    W,
    V,
    S,
    C0,
    NCP: tl.constexpr,
    MP: tl.constexpr,
    BC: tl.constexpr,
):
    """S[c] = x . M[:, c]  for the c-block starting at C0.

    The single 2-D axis=1 tl.sum in the whole factorisation. v arrives as a
    stride-0 duplicated-address tile so no 1-D broadcast reaches the reduce.
    """
    b = tl.program_id(0)
    c = C0 + tl.program_id(1) * BC + tl.arange(0, BC)
    r = tl.arange(0, MP)
    t = tl.load(W + b * (NCP * MP) + c[:, None] * MP + r[None, :])
    vt = tl.load(V + b * MP + r[None, :] + c[:, None] * 0)
    tl.store(S + b * NCP + c, tl.sum(t * vt, axis=1))


@libentry()
@triton.jit
def _scal_kernel(
    W,
    V,
    S,
    XJ,
    ALPHA,
    BETA,
    DIAG,
    RMAX,
    J,
    NP,
    NCP: tl.constexpr,
    MP: tl.constexpr,
):
    """Reflector scalars for step J.

    alpha = -sign(x_J) ||x||, beta = 1 / (||x||^2 - alpha x_J) = 2 / v^T v.
    ||x||^2 is S[J], already reduced. RMAX keeps the running max |r_ii| for the
    rank guard and is ping-ponged between two slots so no address is both read
    and written in one launch. Only scalars are written here: a masked store is
    NOT honoured on this backend (it writes the whole tile), so row J of the
    factor is left to the generic rank-1 update, which produces it exactly.
    """
    b = tl.program_id(0)
    nrm2 = tl.load(S + b * NCP + J)
    xj = tl.load(XJ + b)
    alpha = -tl.where(xj >= 0.0, 1.0, -1.0) * tl.sqrt(tl.maximum(nrm2, 0.0))
    den = nrm2 - alpha * xj
    beta = tl.where(den > 0.0, 1.0 / tl.where(den > 0.0, den, 1.0), 0.0)
    tl.store(ALPHA + b, alpha)
    tl.store(BETA + b, beta)
    tl.store(DIAG + b * NP + J, alpha)
    prev = tl.load(RMAX + b * 2 + (J % 2))
    tl.store(RMAX + b * 2 + ((J + 1) % 2), tl.maximum(prev, tl.abs(alpha)))
    # v[J] = x_J - alpha completes the reflector
    tl.store(V + b * MP + J, xj - alpha)


@libentry()
@triton.jit
def _wvec_kernel(
    W,
    S,
    ALPHA,
    BETA,
    WV,
    J,
    C0,
    NCP: tl.constexpr,
    MP: tl.constexpr,
    BC: tl.constexpr,
):
    """WV[c] = beta * v . M[:, c], forced to exactly 0 for the finished c < J.

    v . M[:,c] = x . M[:,c] - alpha * M[J,c] = S[c] - alpha * W[c, J], so the
    reflector's own alpha term never needs a second reduction. Column J of W is
    read here rather than in the update kernel: reading one address of W while
    storing to another returns stale data on this backend.

    c == J is kept: there WV[J] == beta * (||x||^2 - alpha x_J) == 1, so the
    generic update turns row J into (alpha, 0, .., 0), which is exactly the
    finished row of the factor.
    """
    b = tl.program_id(0)
    c = C0 + tl.program_id(1) * BC + tl.arange(0, BC)
    a = tl.load(ALPHA + b)
    be = tl.load(BETA + b)
    s = tl.load(S + b * NCP + c)
    wcj = tl.load(W + b * (NCP * MP) + c * MP + J)
    tl.store(WV + b * NCP + c, tl.where(c >= J, be * (s - a * wcj), 0.0))


@libentry()
@triton.jit
def _upd_kernel(
    W,
    WV,
    V,
    C0,
    NCP: tl.constexpr,
    MP: tl.constexpr,
    BC: tl.constexpr,
):
    """M <- H M, i.e. W[c, r] -= WV[c] * v[r].

    Both rank-1 operands are stride-0 duplicated-address 2-D loads. Written as
    1-D broadcasts (``WV[c][:, None] * V[r][None, :]``) this is silently wrong
    for MP > 128 on this backend.
    """
    b = tl.program_id(0)
    c = C0 + tl.program_id(1) * BC + tl.arange(0, BC)
    r = tl.arange(0, MP)
    off = b * (NCP * MP) + c[:, None] * MP + r[None, :]
    wt = tl.load(WV + b * NCP + c[:, None] + r[None, :] * 0)
    vt = tl.load(V + b * MP + r[None, :] + c[:, None] * 0)
    tl.store(W + off, tl.load(W + off) - wt * vt)


@libentry()
@triton.jit
def _vsave_kernel(
    V,
    BETA,
    VS,
    BETAS,
    J,
    NP,
    MP: tl.constexpr,
):
    """Keep reflector J: the minimum-norm path has to APPLY Q afterwards."""
    b = tl.program_id(0)
    r = tl.arange(0, MP)
    tl.store(VS + b * (NP * MP) + J * MP + r, tl.load(V + b * MP + r))
    tl.store(BETAS + b * NP + J, tl.load(BETA + b))


@libentry()
@triton.jit
def _rowcpy_kernel(SRC, DST, SBATCH, J, LD: tl.constexpr, NW: tl.constexpr):
    """DST[b, :NW] <- SRC[b, J, :NW].

    Row J is materialised as its own per-batch vector so that every reduce can
    reach it through the ONE stride-0 form that lowers correctly here, namely
    ``base + b*NW + r[None, :] + c[:, None]*0``. Folding the row offset J*LD
    into a stride-0 2-D load instead ("scalar*stride + vector") is silently
    miscomputed on this backend -- verified against a host dump.
    """
    b = tl.program_id(0)
    i = tl.arange(0, NW)
    tl.store(DST + b * NW + i, tl.load(SRC + b * SBATCH + J * LD + i))


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
def _scale_kernel(SRC, FAC, DST, J, NP, KP: tl.constexpr):
    """DST[k] <- FAC[J] * SRC[k], the rank-1 coefficient of the Q-apply."""
    b = tl.program_id(0)
    k = tl.arange(0, KP)
    tl.store(DST + b * KP + k, tl.load(FAC + b * NP + J) * tl.load(SRC + b * KP + k))


# ---------------------------------------------------------------------------
# Triangular solve, shared by both paths.
#
# COLUMN-oriented, not the usual dot-product form: after each unknown is found
# the remaining right-hand side is updated by a rank-1 axpy.  That is not a
# style choice -- the dot-product form has to write the new unknown into a
# STRIDED slot of the solution matrix, and a strided store on this backend
# writes its value to the right address AND garbage to unrelated allocations
# (verified: it corrupted a neighbouring tensor 2.5 KB away).  Column-oriented,
# every store is contiguous and every reduce disappears.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# host drivers
# ---------------------------------------------------------------------------
def _qr_sweep(W, NCP, MP, NP, nsteps, batch, dt, dev, keep_reflectors):
    """Householder QR of the matrix whose transpose is W. Returns (DIAG, RMAX).

    When keep_reflectors is set, also returns (VS, BETAS) for the Q-apply of
    the minimum-norm path.
    """
    BC = _LANES
    V = torch.zeros((batch, MP), dtype=dt, device=dev)
    XJ = torch.zeros((batch,), dtype=dt, device=dev)
    S = torch.zeros((batch, NCP), dtype=dt, device=dev)
    WV = torch.zeros((batch, NCP), dtype=dt, device=dev)
    ALPHA = torch.zeros((batch,), dtype=dt, device=dev)
    BETA = torch.zeros((batch,), dtype=dt, device=dev)
    DIAG = torch.zeros((batch, NP), dtype=dt, device=dev)
    RMAX = torch.zeros((batch, 2), dtype=dt, device=dev)
    VS = BETAS = None
    if keep_reflectors:
        VS = torch.zeros((batch, NP, MP), dtype=dt, device=dev)
        BETAS = torch.zeros((batch, NP), dtype=dt, device=dev)

    for j in range(nsteps):
        c0 = (j // BC) * BC
        nb = (NCP - c0) // BC
        _mk_v_kernel[(batch,)](W, V, XJ, j, NCP=NCP, MP=MP)
        _dot_kernel[(batch, nb)](W, V, S, c0, NCP=NCP, MP=MP, BC=BC)
        _scal_kernel[(batch,)](
            W, V, S, XJ, ALPHA, BETA, DIAG, RMAX, j, NP, NCP=NCP, MP=MP
        )
        if keep_reflectors:
            _vsave_kernel[(batch,)](V, BETA, VS, BETAS, j, NP, MP=MP)
        _wvec_kernel[(batch, nb)](
            W, S, ALPHA, BETA, WV, j, c0, NCP=NCP, MP=MP, BC=BC
        )
        _upd_kernel[(batch, nb)](W, WV, V, c0, NCP=NCP, MP=MP, BC=BC)
    return DIAG, RMAX, VS, BETAS


def _lstsq_tall(A, B, rcond):
    """A: (batch, m, n) with m >= n; B: (batch, m, nrhs). Both contiguous."""
    batch, m, n = A.shape
    nrhs = B.shape[-1]
    dt, dev = A.dtype, A.device

    KP = max(_LANES, _p2(nrhs))
    MP = max(_MIN_ROW, _p2(m))
    NP = max(_MIN_COL, _p2(n))
    NCP = max(NP, _p2(n + KP))

    W = torch.zeros((batch, NCP, MP), dtype=dt, device=dev)
    W[:, :n, :m] = A.transpose(-1, -2)
    W[:, n : n + nrhs, :m] = B.transpose(-1, -2)

    with torch_device_fn.device(dev):
        DIAG, RMAX, _, _ = _qr_sweep(W, NCP, MP, NP, n, batch, dt, dev, False)

        # RHS[k, i] starts as C[i, k] = W[n+k, i], which is a plain slice.
        RHS = W[:, n : n + KP, :NP].contiguous()
        XS = torch.zeros((batch, NP, KP), dtype=dt, device=dev)
        XI = torch.zeros((batch, KP), dtype=dt, device=dev)
        RROW = torch.zeros((batch, NP), dtype=dt, device=dev)
        for t in range(n):
            i = n - 1 - t
            _solve_step_kernel[(batch,)](
                RHS, XI, XS, DIAG, RMAX, rcond, i, n, NW=NP, KP=KP
            )
            # R[j, i] == W[i, j]: the update coefficient is a ROW of W.
            _rowcpy_kernel[(batch,)](W, RROW, NCP * MP, i, LD=MP, NW=NP)
            _upd_kernel[(batch, 1)](RHS, XI, RROW, 0, NCP=KP, MP=NP, BC=KP)

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
    MP = max(_MIN_ROW, _p2(n))  # padded row length = rows of A^T
    NRP = max(_MIN_COL, _p2(m))  # padded rows of W = columns of A^T
    MIP = NRP

    W = torch.zeros((batch, NRP, MP), dtype=dt, device=dev)
    W[:, :m, :n] = A

    with torch_device_fn.device(dev):
        DIAG, RMAX, VS, BETAS = _qr_sweep(
            W, NRP, MP, NRP, m, batch, dt, dev, True
        )

        # forward substitution R^T y = b: the coefficient matrix is L[c, i] =
        # W[c, i], so the update coefficient is a COLUMN of W.
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
            _upd_kernel[(batch, 1)](RHS, YI, COL, 0, NCP=KP, MP=MIP, BC=KP)

        # x = Q y = H_0 .. H_{m-1} [y; 0]
        Z = torch.zeros((batch, KP, MP), dtype=dt, device=dev)
        Z[:, :nrhs, :m] = YS[:, :m, :nrhs].transpose(-1, -2)
        DOT = torch.zeros((batch, KP), dtype=dt, device=dev)
        COEF = torch.zeros((batch, KP), dtype=dt, device=dev)
        RROW = torch.zeros((batch, MP), dtype=dt, device=dev)
        for t in range(m):
            j = m - 1 - t
            _rowcpy_kernel[(batch,)](VS, RROW, NRP * MP, j, LD=MP, NW=MP)
            _dot_kernel[(batch, 1)](Z, RROW, DOT, 0, NCP=KP, MP=MP, BC=KP)
            _scale_kernel[(batch,)](DOT, BETAS, COEF, j, NRP, KP=KP)
            _upd_kernel[(batch, 1)](Z, COEF, RROW, 0, NCP=KP, MP=MP, BC=KP)

    return Z[:, :nrhs, :n].transpose(-1, -2).contiguous()


def _fallback(A, b, rcond, driver):
    """Reference path for inputs outside the native scope (complex, or a padded
    row wider than the validated reduction tile). Routed through CPU so it does
    not re-enter this override."""
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

    # torch's CUDA gels backend rejects any other driver; raise likewise rather
    # than silently computing a result torch would refuse.
    if driver not in (None, "gels"):
        raise RuntimeError(
            "torch.linalg.lstsq: `driver` other than `gels` is not supported on CUDA"
        )

    if A.dtype not in _SUPPORTED_DTYPES or A.is_complex() or A.dim() < 2 or b.dim() < 1:
        return _fallback(A, b, rcond, driver)

    m, n = A.shape[-2], A.shape[-1]

    # RHS classification, matching torch.linalg.lstsq exactly: a VECTOR rhs has
    # one fewer dim than A and must match A.shape[:-1] exactly; a MATRIX rhs has
    # the same ndim and broadcasts its batch dims. Anything else torch rejects.
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

    # degenerate dims are shape-determined; LAPACK ?gels quick-returns on any
    # zero dim and zeroes its buffer, so both solution and residuals are zeros.
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

    # torch returns residuals only when m > n; note it squeezes the SOLUTION for
    # a vector b but keeps residuals at shape (*, nrhs).
    if m > n:
        residuals = RES.reshape(*batch_shape, nrhs)
    else:
        residuals = torch.empty(0, dtype=A.dtype, device=A.device)

    rank, singular_values = _empty_rank_sv(A)
    return solution, residuals, rank, singular_values
