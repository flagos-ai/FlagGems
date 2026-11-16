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
"""Kunlunxin (XPU3) override for ``aten::geqrf`` (packed Householder QR).

Why a vendor override is needed
--------------------------------
The generic ``src/flag_gems/ops/geqrf.py`` does not compile on this backend:

* its small path reduces 2-D tiles with ``tl.sum(..., axis=1)`` / ``axis=0``,
  and every 2-D (axis) reduction is rejected by the XPU legalizer
  (``UNREACHABLE ... tensorShape.size() is not more than iterCount.size()``);
* both paths also hit ``arith.addi/cmpi requires the same type`` legalisation
  failures on plain 2-D masked tile code.

This override keeps the same *algorithm* (column-major working buffer, blocked
Householder with panel factorisation + panel application) but only uses
constructs that the XPU backend actually lowers correctly.  The backend
constraints that shaped it (all re-verified on P800 / XPU3, evidence in
``harness/perf_ir/`` and ``harness/run_logs/geqrf_batch13/``) are:

1. no 2-D reductions at all -> everything is a 1-D (BLOCK_M,) vector;
2. ``tl.trans`` is silently a no-op (a load/trans/store round-trip returns the
   input unchanged), so the pack kernel is written as
   *gather-load from A / contiguous store into the column-major buffer*;
3. a ``tl.sum`` inside a *dynamic* loop that is itself nested inside a static
   unroll silently miscompiles (wrong lanes get updated), so the panel
   factorisation is fully unrolled (no dynamic loops) and the intra-panel
   reflectors are read back from the working buffer;
4. loop-carried 1-D vectors only legalise at 64/128/1024 lanes (128 < BLOCK_M <
   1024 fails to compile; and at 64/128 lanes the panel factorisation
   miscompiles), so BLOCK_M is pinned to ``max(1024, next_pow2(m))``;
5. 2-D masked loads/stores are unreliable, so all tiles are made in-bounds by
   padding the working buffer to ``NPAD`` columns x ``BLOCK_M`` rows (zero
   filled) and the pack kernel clamps + zeroes instead of masking.

Layout: ``w`` is ``(B, NPAD, BLOCK_M)`` -- column ``c`` of the logical matrix is
``w[b, c, :]`` (contiguous), so a reflector column and a trailing column are
both contiguous 1-D loads.  The returned ``a`` is the transposed, strided view
``w[..., :n, :m].transpose(-1, -2)`` (LAPACK-style column-major output, the
same layout ``torch.geqrf`` returns on CUDA/XPU), so no extra copy is paid.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Pack tile (rows of the working buffer handled per program).
_TILE = 64
# Blocked-panel width (constexpr for the -factor / -apply kernels).
_PANEL = 8
# 1-D reduction width actually used; see note 4 in the module docstring.
_MIN_BLOCK_M = 1024


@libentry()
@triton.jit
def _geqrf_pack_aligned(
    A_ptr,
    w_ptr,
    sAb,
    sAm,
    sAn,
    sWb,
    BM: tl.constexpr,
    BT: tl.constexpr,
    NT: tl.constexpr,
):
    # Affine (no clamp, no mask) pack for m % BT == 0 and n % BT == 0:
    # gather A[:, cols] (rows contiguous in the last dim -> one affine strided
    # load) and store it contiguously into the column-major buffer.
    b = tl.program_id(0)
    pt = tl.program_id(1)
    tc = pt % NT
    tr = pt // NT
    rows = tr * BT + tl.arange(0, BT)
    cols = tc * BT + tl.arange(0, BT)
    val = tl.load(A_ptr + b * sAb + cols[:, None] * sAn + rows[None, :] * sAm)
    tl.store(w_ptr + b * sWb + cols[:, None] * BM + rows[None, :], val)


@libentry()
@triton.jit
def _geqrf_pack_safe(
    A_ptr,
    w_ptr,
    m,
    n,
    sAb,
    sAm,
    sAn,
    sWb,
    BM: tl.constexpr,
    BT: tl.constexpr,
):
    # General pack: clamp the indices (keeps every access in bounds, so no
    # masked load/store is needed) and zero the lanes outside the matrix.
    b = tl.program_id(0)
    pt = tl.program_id(1)
    nt = tl.cdiv(n, BT)
    tc = pt % nt
    tr = pt // nt
    rows = tr * BT + tl.arange(0, BT)
    cols = tc * BT + tl.arange(0, BT)
    rs = tl.minimum(rows, m - 1)
    cs = tl.minimum(cols, n - 1)
    val = tl.load(A_ptr + b * sAb + cs[:, None] * sAn + rs[None, :] * sAm)
    val = tl.where((cols[:, None] < n) & (rows[None, :] < m), val, 0.0)
    tl.store(w_ptr + b * sWb + cols[:, None] * BM + rows[None, :], val)


@libentry()
@triton.jit
def _geqrf_factor(
    w_ptr,
    tau_ptr,
    BM: tl.constexpr,
    NB: tl.constexpr,
    k,
    j0,
    jend,
    sWb,
):
    # Factor panel columns [j0, jend) of the column-major buffer, one program
    # per batch element.  Fully unrolled on purpose (see note 3): each column
    # absorbs the panel reflectors stored by the previous unrolled steps, is
    # factorised with the LAPACK dlarfg recipe and packed in place.
    b = tl.program_id(0)
    base = b * sWb
    rows = tl.arange(0, BM)
    for t in tl.static_range(0, NB):
        c = j0 + t
        col = tl.load(w_ptr + base + c * BM + rows)
        col0 = col  # untouched copy for the "past the panel" columns
        for i in range(0, t):
            r = j0 + i
            ref = tl.load(w_ptr + base + r * BM + rows)
            v = tl.where(rows == r, 1.0, tl.where(rows > r, ref, 0.0))
            tau_r = tl.load(tau_ptr + b * k + r)
            dot = tl.sum(v * col)
            col = col - tau_r * dot * v
        alpha = tl.sum(tl.where(rows == c, col, 0.0))
        x2 = tl.sum(tl.where(rows > c, col * col, 0.0))
        normx = tl.sqrt(alpha * alpha + x2)
        safe = x2 > 0.0
        sgn = tl.where(alpha >= 0.0, 1.0, -1.0)
        beta = tl.where(safe, -sgn * normx, alpha)
        vb0 = tl.where(safe, alpha - beta, 1.0)
        tau_c = tl.where(safe, (beta - alpha) / beta, 0.0)
        vtail = tl.where(rows > c, col / vb0, 0.0)
        keep = c < jend
        out = tl.where(rows == c, beta, tl.where(rows > c, vtail, col))
        out = tl.where(keep, out, col0)
        tl.store(w_ptr + base + c * BM + rows, out)
        tl.store(tau_ptr + b * k + c, tl.where(keep, tau_c, 0.0))


@libentry()
@triton.jit
def _geqrf_apply(
    w_ptr,
    tau_ptr,
    BM: tl.constexpr,
    NB: tl.constexpr,
    n,
    k,
    j0,
    jend,
    sWb,
):
    # Apply the panel reflectors [j0, jend) to every trailing column
    # jend..n-1, one program per (batch, column).
    pid = tl.program_id(0)
    ncol = n - jend
    b = pid // ncol
    cc = pid - b * ncol
    c = jend + cc
    rows = tl.arange(0, BM)
    base = b * sWb
    col = tl.load(w_ptr + base + c * BM + rows)
    for i in tl.static_range(0, NB):
        r = j0 + i
        ref = tl.load(w_ptr + base + r * BM + rows)
        v = tl.where(rows == r, 1.0, tl.where(rows > r, ref, 0.0))
        tau_r = tl.load(tau_ptr + b * k + r)
        tau_r = tl.where(r < jend, tau_r, 0.0)
        dot = tl.sum(v * col)
        col = col - tau_r * dot * v
    tl.store(w_ptr + base + c * BM + rows, col)


def geqrf(A):
    """Packed Householder QR (LAPACK GEQRF) with XPU-safe kernels.

    Args:
        A: (..., m, n) float32/float64 tensor.

    Returns:
        (a, tau) with the packed reflectors below the diagonal of the first
        k = min(m, n) columns of ``a`` and ``tau`` of shape (..., k).
    """
    logger.debug("GEMS GEQRF (kunlunxin)")
    if A.dim() < 2:
        raise RuntimeError("geqrf: input must have at least 2 dimensions")
    if A.dtype not in (torch.float32, torch.float64):
        raise NotImplementedError(
            "FlagGems geqrf currently supports float32 and float64 inputs; "
            f"got dtype={A.dtype}"
        )

    shape = A.shape
    m, n = shape[-2], shape[-1]
    k = m if m < n else n
    B = 1
    for d in shape[:-2]:
        B *= d
    tau = torch.empty((*shape[:-2], k), dtype=A.dtype, device=A.device)
    if B == 0 or k == 0:
        return torch.empty_like(A), tau
    sAb = A.stride(-3) if A.dim() > 2 else m * n
    sAm = A.stride(-2)
    sAn = A.stride(-1)

    nb = _PANEL
    if nb > k:
        nb = max(2, triton.next_power_of_2(k))
    BM = max(_MIN_BLOCK_M, triton.next_power_of_2(m))
    # Reflector columns up to j0+nb-1 are loaded (their tau is zeroed past the
    # panel), so the buffer needs k+nb columns; the pad must be finite.
    NPAD = triton.cdiv(n + nb, _TILE) * _TILE
    # Keep the leading dims so the transposed view below already has the
    # requested (*, m, n) shape (no reshape / copy is needed).
    w = torch.zeros((*shape[:-2], NPAD, BM), dtype=A.dtype, device=A.device)
    sWb = NPAD * BM

    nt = triton.cdiv(m, _TILE) * triton.cdiv(n, _TILE)
    if m % _TILE == 0 and n % _TILE == 0:
        _geqrf_pack_aligned[(B, nt)](
            A, w, sAb, sAm, sAn, sWb, BM, _TILE, n // _TILE, num_warps=4
        )
    else:
        _geqrf_pack_safe[(B, nt)](
            A, w, m, n, sAb, sAm, sAn, sWb, BM, _TILE, num_warps=4
        )

    j = 0
    while j < k:
        jend = j + nb if j + nb < k else k
        _geqrf_factor[(B,)](w, tau, BM, nb, k, j, jend, sWb, num_warps=1)
        if jend < n:
            _geqrf_apply[(B * (n - jend),)](
                w, tau, BM, nb, n, k, j, jend, sWb, num_warps=1
            )
        j = jend

    a = w[..., :n, :m].transpose(-1, -2)
    return a, tau
