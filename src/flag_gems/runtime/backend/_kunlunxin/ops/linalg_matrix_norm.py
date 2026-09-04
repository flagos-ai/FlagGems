# Copyright 2026 FlagOS Contributors
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
"""Kunlunxin (XPU) ``linalg_matrix_norm``.

The generic ``flag_gems/ops/linalg_matrix_norm.py`` is unusable on this
backend for the five non-SVD orders:

* ``ord = +/-1`` reduces a 2-D tile with ``tl.sum(..., axis=0)``.  TritonXPU
  rejects that outright ("axis must not be 0 for 2D+ shapes, consider
  manually transpose"), so every ``+/-1`` case is a hard compile failure.
* ``ord = +/-inf`` and ``ord = 'fro'`` combine ``tl.atomic_add`` /
  ``tl.atomic_max`` / ``tl.atomic_min`` fan-in with masked tail tiles that
  carry ``other=0.0``.  Both are known silent-miscompute sources on this XPU,
  which is why those orders come out numerically wrong instead of failing
  loudly.
* the generic file also binds ``flag_gems.ops.max/min/sqrt/sum`` at *import*
  time, so the Kunlunxin overrides of those four operators can never be
  substituted by ``SpecOpRegistrar``; the generic (non-XPU-safe) reductions
  are what actually run.

This override reimplements the five non-SVD orders on a single XPU-safe
primitive: a row-wise reduction over a contiguous ``[R, C]`` buffer that

* only ever reduces along ``axis=1`` (a ``dim=-2`` reduction is materialised
  by a native transposing ``aten::_copy_from``, never by ``axis=0``),
* never issues a masked or ``other=``-carrying load - the row index is
  clamped and the ragged column tail is copied into a separate
  identity-filled tile,
* uses ``BLOCK_M = 64`` so the backend's "every vector store touches exactly
  64 contiguous elements" behaviour lands exactly on the slice the program
  owns (no cross-program clobber, no masked store),
* uses an inner tile width of 128 (>= 64 to dodge the narrow-tile lowering
  bug, != 32/64 to dodge the NOC wedge, 64 x 128 = 8192 elements to satisfy
  the 2-D tile minimum),
* accumulates in fp32 and uses no atomics at all.

The SVD-based orders (``2``, ``-2``, ``'nuc'``) still go through the generic
Triton helpers: they need an fp64 Gram/eigen solve, and this platform has no
fp64 compute path, so there is nothing to gain from a vendor rewrite yet.
Delegating keeps their behaviour bit-identical to the current baseline.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.linalg_matrix_norm import _nuc_norm as _generic_nuc_norm
from flag_gems.ops.linalg_matrix_norm import _ord2_norm as _generic_ord2_norm
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_SUPPORTED_NUMERIC = {1, -1, 2, -2, float("inf"), -float("inf")}

# Reduction kinds.
_OP_SUMSQ = 0  # sum(x*x)
_OP_SUMABS = 1  # sum(|x|)
_OP_SUM = 2  # sum(x)
_OP_MAX = 3  # max(x)
_OP_MIN = 4  # min(x)

# 64 rows per program: the backend writes exactly 64 contiguous elements per
# vector store regardless of the requested length, so a 64-row block makes the
# store land exactly on the slice this program owns.
_BLOCK_M = 64
# 128-wide inner tile: >= 64 (narrow tiles miscompile), not 32/64 (NOC wedge),
# and 64 * 128 == 8192 elements (2-D tile minimum on this backend).
_BLOCK_N = 128
# Rough program-count target used to decide how far to split the reduction
# axis when there are not enough rows to fill the device.
_PROG_TARGET = 96
# Splitting the reduction axis costs an extra kernel launch plus a transposing
# native copy; below this element count the launch overhead dominates.
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
    X,  # data, logical [R, C_PITCH]
    Out,  # partials / result, logical [NCHUNK, RP]
    R,
    C_PITCH,  # row stride of X
    NFULL,  # number of BLOCK_N tiles along the reduced axis
    TPC,  # tiles handled by one chunk
    RP,  # padded row count, also the Out pitch per chunk
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


def _native_contiguous(t):
    """Materialise ``t`` contiguously through the vendor strided-copy engine.

    ``Tensor.contiguous()`` is itself a FlagGems-registered operator, so using
    it here would drag a Triton copy kernel into the timed region (and into
    every ``use_gems`` call site).  ``aten::_copy_from`` is never overridden by
    FlagGems and goes straight to the vendor engine.
    """
    if t.is_contiguous():
        return t
    out = torch.empty(t.shape, dtype=t.dtype, device=t.device)
    torch.ops.aten._copy_from(t, out, False)
    return out


def _row_reduce(x, R, C, op, final_sqrt=False):
    """Reduce a 2-D ``[R, C]`` view along ``C``.

    ``x`` must have unit stride in its last dimension; the row stride is
    honoured as-is, so strided row selections (e.g. one row out of every pair)
    can be reduced without a copy.  Returns an fp32 tensor view of length
    ``R``.

    When ``C`` is not a multiple of ``BLOCK_N`` the rows are re-materialised
    into an identity-padded buffer first, so the kernel itself never needs a
    mask: a conditional tail tile inside the kernel makes TritonXPU emit an
    ``scf.if`` yielding a tensor, which fails to lower (``triton_xpu.vvaddf op
    requires the same type for all operands and results``), and clamping the
    column index instead costs an extra runtime kernel argument, which is worth
    a 15-30x slowdown here (see ``_row_reduce_kernel``).  ``C`` is additionally
    split across a second grid axis when there are not enough rows to fill the
    device; the per-chunk partials are transposed with a native strided copy so
    that the follow-up fold is again an ``axis=1`` reduction.
    """
    dev = x.device
    BM, BN = _BLOCK_M, _BLOCK_N
    RP = triton.cdiv(R, BM) * BM
    nrow_blocks = RP // BM
    rows_aligned = R % BM == 0

    # A size-1 reduction axis carries an arbitrary innermost stride; it always
    # takes the padding branch below, which copies through the native engine.
    assert tuple(x.shape) == (R, C) and (C == 1 or x.stride(-1) == 1)
    pitch = x.stride(0) if R > 1 else C
    ncols = C
    if C % BN:
        ncols = triton.cdiv(C, BN) * BN
        pad = torch.full((R, ncols), _identity(op), dtype=x.dtype, device=dev)
        torch.ops.aten._copy_from(x, pad[:, :C], False)
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
        torch.ops.aten._copy_from(
            part[: nchunk * RP].reshape(nchunk, RP)[:, :R].transpose(0, 1),
            pt[:, :nchunk],
            False,
        )
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
    X,  # [B, 2, PITCH], zero padded past the real extent
    Out,  # [BP]
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
    MODE: tl.constexpr,  # 0 = sigma_max, 1 = sigma_min, 2 = sigma_max + sigma_min
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
    # Normalise to (B, 2, K): the two vectors must be the contiguous rows.
    W = _native_contiguous(Ab.transpose(-2, -1)) if M >= N else Ab
    pitch = K
    if K % BN:
        pitch = triton.cdiv(K, BN) * BN
        Wp = torch.zeros((B, 2, pitch), dtype=W.dtype, device=dev)
        torch.ops.aten._copy_from(W, Wp[:, :, :K], False)
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


def _absmax_norm(Ab, B, M, N, is_min, along_rows):
    """|A| row/column sums followed by a max (or min) over the survivors.

    ``along_rows=True``  -> ord = +/-inf (sum over N, then max/min over M)
    ``along_rows=False`` -> ord = +/-1   (sum over M, then max/min over N)
    """
    if along_rows:
        base, R, C = Ab, B * M, N
    else:
        # A dim=-2 reduction: TritonXPU cannot reduce a 2-D tile along axis 0,
        # so transpose through the native strided-copy engine and keep every
        # kernel reduction on axis=1.
        base = _native_contiguous(Ab.transpose(-2, -1))
        R, C = B * N, M
    sums = _row_reduce(base.reshape(R, C), R, C, _OP_SUMABS)
    inner = R // B
    return _row_reduce(sums.reshape(B, inner), B, inner, _OP_MIN if is_min else _OP_MAX)


def linalg_matrix_norm(A, ord="fro", dim=(-2, -1), keepdim=False, dtype=None):
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

    # --- SVD-based orders --------------------------------------------------
    # k <= 2 has a closed form and is handled here (the generic rank-2 kernel
    # corrupts memory on this backend, see _rank2_sigma_norm).  k >= 3 needs an
    # fp64 Gram/eigen solve to stay inside the fp32 test tolerance and this
    # platform has no fp64 compute path, so those shapes keep running the
    # generic Triton path unchanged.
    is_svd = (is_str and ord == "nuc") or (ord_val is not None and abs(ord_val) == 2.0)
    if is_svd:
        if A.dtype in (torch.float16, torch.bfloat16):
            A = A.float()
        k = min(A.size(dim[0]), A.size(dim[1]))
        if k > 2:
            if is_str:
                return _generic_nuc_norm(A, dim=dim, keepdim=keepdim, dtype=dtype)
            return _generic_ord2_norm(A, ord_val, dim, keepdim, dtype)
        out_dtype = dtype if dtype is not None else A.dtype
        if dtype is not None:
            A = A.to(dtype)
        Ab, B, M, N = _batched_view(A, dim)
        if k == 1:
            # A single singular value: sigma_0 == ||A||_F, so ord 2 / -2 / nuc
            # all collapse to the Frobenius norm.
            res = _fro(Ab, B, M, N)
        else:
            mode = 2 if is_str else (0 if ord_val > 0 else 1)
            res = _rank2_sigma_norm(Ab, B, M, N, mode)
        return _reshape_result(res, A, dim, keepdim, out_dtype)

    out_dtype = dtype if dtype is not None else A.dtype
    Ab, B, M, N = _batched_view(A, dim)

    if is_str:  # "fro"
        res = _fro(Ab, B, M, N)
    else:
        res = _absmax_norm(Ab, B, M, N, ord_val < 0, math.isinf(ord_val))

    return _reshape_result(res, A, dim, keepdim, out_dtype)
