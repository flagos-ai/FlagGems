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

"""Triton implementation of linalg_ldl_factor_ex (Bunch-Kaufman LDL^T).

Semantics reproduced from the target reference (torch.linalg.ldl_factor_ex,
rocSOLVER sytrf, unblocked LAPACK DSYTF2 form), verified by numerical probes:

* only the lower triangle of the input is read; the matrix is mirrored in
  registers,
* alpha = (1 + sqrt(17)) / 8, decision order
  max(|akk|, colmax) == 0 -> 1x1 (info = k+1 first time)
  |akk| >= alpha*colmax   -> 1x1 at k
  |akk| >= alpha*colmax*(colmax/rowmax) -> 1x1 at k
  |A[r,r]| >= alpha*rowmax -> 1x1 at r (swap k <-> r)
  else                     -> 2x2 block (k, k+1) with swap k+1 <-> r,
  where r is the first index attaining colmax (LAPACK IDAMAX tie-break),
* interchanges are symmetric permutations restricted to indices >= k (the
  already stored multipliers of columns < k are NOT permuted),
* ipiv is 1-based; a 2x2 block stores -(r+1) twice,
* the strict upper triangle of LD is zero,
* the reference output layout: LD column-major, pivots/info int32.

One CTA per matrix; the symmetric tile stays in registers.  The profiled
per-step cost of the earlier version was invariant in n (about 1.7us at n=4
and n=32), i.e. the serial chain of dependent cross-lane reductions dominates,
so each step now uses as few chained reductions as possible: one fused
(colmax, first-index, akk) reduction, one fused (rowmax, A[r,r]) reduction,
and rank updates whose masks are implied by the already masked vectors.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# (1 + sqrt(17)) / 8, LAPACK Bunch-Kaufman constant
_ALPHA = tl.constexpr(0.6403882032022076)


@triton.jit
def _maxidx_sum(m1, i1, s1, m2, i2, s2):
    """(max magnitude, smallest index attaining it, sum of the extras)."""
    gt = m1 > m2
    eq = m1 == m2
    return (
        tl.maximum(m1, m2),
        tl.where(gt, i1, tl.where(eq, tl.minimum(i1, i2), i2)),
        s1 + s2,
    )


@triton.jit
def _max_sum(m1, s1, m2, s2):
    return tl.maximum(m1, m2), s1 + s2


@triton.jit
def _ldl_kernel(
    A_ptr,
    LD_ptr,
    PIV_ptr,
    INFO_ptr,
    n,
    sA,
    sAr,
    sAc,
    sLB,
    sLr,
    sLc,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    r = tl.arange(0, BLOCK_N)
    c = tl.arange(0, BLOCK_N)
    inb = (r[:, None] < n) & (c[None, :] < n)

    # lower triangle of A, mirrored into a full symmetric register tile
    ap = A_ptr + pid * sA
    a = tl.load(
        ap + r[:, None] * sAr + c[None, :] * sAc,
        mask=inb & (r[:, None] >= c[None, :]),
        other=0.0,
    )
    aT = tl.load(
        ap + c[None, :] * sAr + r[:, None] * sAc,
        mask=inb & (r[:, None] < c[None, :]),
        other=0.0,
    )
    T = a + aT

    info = 0
    k = 0
    while k < n:
        # ---- current column k and all Bunch-Kaufman quantities ------------
        colk = tl.sum(tl.where(c[None, :] == k, T, 0.0), axis=1)
        cand = (r > k) & (r < n)
        colmax, imax, dkk = tl.reduce(
            (
                tl.where(cand, tl.abs(colk), 0.0),
                tl.where(cand, r, BLOCK_N),
                tl.where(r == k, colk, 0.0),
            ),
            0,
            _maxidx_sum,
        )
        absakk = tl.abs(dkk)

        kstep = 1
        kp = k
        dkk_new = dkk * 0.0
        if tl.maximum(absakk, colmax) == 0.0:
            if info == 0:
                info = k + 1
        else:
            if absakk < _ALPHA * colmax:
                row = tl.sum(tl.where(r[:, None] == imax, T, 0.0), axis=0)
                candr = (c >= k) & (c < n) & (c != imax)
                rowmax, drr = tl.reduce(
                    (tl.where(candr, tl.abs(row), 0.0), tl.where(c == imax, row, 0.0)),
                    0,
                    _max_sum,
                )
                dkk_new = drr
                if absakk < _ALPHA * colmax * (colmax / rowmax):
                    kp = imax
                    if tl.abs(drr) < _ALPHA * rowmax:
                        kstep = 2

        # ---- record pivot -------------------------------------------------
        piv = tl.where(kstep == 1, kp + 1, -(kp + 1))
        tl.store(PIV_ptr + pid * n + k, piv)
        if kstep == 2:
            tl.store(PIV_ptr + pid * n + k + 1, piv)

        # ---- symmetric interchange (indices >= k only) --------------------
        p = k + kstep - 1
        if p != kp:
            rp = tl.sum(tl.where(r[:, None] == p, T, 0.0), axis=0)
            rq = tl.sum(tl.where(r[:, None] == kp, T, 0.0), axis=0)
            csel = c[None, :] >= k
            T = tl.where((r[:, None] == p) & csel, rq[None, :], T)
            T = tl.where((r[:, None] == kp) & csel, rp[None, :], T)
            cp = tl.sum(tl.where(c[None, :] == p, T, 0.0), axis=1)
            cq = tl.sum(tl.where(c[None, :] == kp, T, 0.0), axis=1)
            rsel = r[:, None] >= k
            T = tl.where((c[None, :] == p) & rsel, cq[:, None], T)
            T = tl.where((c[None, :] == kp) & rsel, cp[:, None], T)
            # A 1x1 interchange (p == k) already wrote cq into column k and the
            # new diagonal (k, k) is the pre-swap A[kp, kp] = drr, so neither
            # needs another cross-lane extraction.  A 2x2 interchange writes cq
            # into column k+1 instead, so column k must still be re-read.
            if kstep == 2:
                colk = tl.sum(tl.where(c[None, :] == k, T, 0.0), axis=1)
            else:
                colk = cq
            dkk = dkk_new

        # ---- eliminate ----------------------------------------------------
        if kstep == 1:
            r1 = tl.where(dkk == 0.0, 0.0, 1.0 / dkk)
            w = tl.where((r > k) & (r < n), colk, 0.0)
            # w vanishes outside the trailing block, so no mask is needed
            T = T - w[:, None] * w[None, :] * r1
            T = tl.where((r[:, None] > k) & (c[None, :] == k), (w * r1)[:, None], T)
        else:
            colk1 = tl.sum(tl.where(c[None, :] == k + 1, T, 0.0), axis=1)
            d11 = tl.sum(tl.where(r == k, colk, 0.0), axis=0)
            d21 = tl.sum(tl.where(r == k + 1, colk, 0.0), axis=0)
            d22 = tl.sum(tl.where(r == k + 1, colk1, 0.0), axis=0)
            ws = tl.where((r > k + 1) & (r < n), colk, 0.0)
            wt = tl.where((r > k + 1) & (r < n), colk1, 0.0)
            q21 = tl.where(d21 == 0.0, 1.0, d21)
            e11 = d22 / q21
            e22 = d11 / q21
            g = (1.0 / (e11 * e22 - 1.0)) / q21
            wk = g * (e11 * ws - wt)
            wkp1 = g * (e22 * wt - ws)
            T = T - ws[:, None] * wk[None, :] - wt[:, None] * wkp1[None, :]
            T = tl.where((r[:, None] > k + 1) & (c[None, :] == k), wk[:, None], T)
            T = tl.where((r[:, None] > k + 1) & (c[None, :] == k + 1), wkp1[:, None], T)

        k += kstep

    out = tl.where(r[:, None] >= c[None, :], T, 0.0)
    ld_ptr = LD_ptr + pid * sLB + r[:, None] * sLr + c[None, :] * sLc
    tl.store(ld_ptr, out, mask=inb)
    tl.store(INFO_ptr + pid, info)


def _num_warps(block_n):
    if block_n <= 16:
        return 1
    if block_n <= 32:
        return 2
    if block_n <= 64:
        return 4
    if block_n <= 128:
        return 8
    return 16


def _ld_strides(batch_shape, n):
    # reference layout: LD is column-major inside each matrix block
    strides = [0] * (len(batch_shape) + 2)
    strides[-2] = 1
    strides[-1] = n
    acc = n * n
    for i in range(len(batch_shape) - 1, -1, -1):
        strides[i] = acc
        acc *= batch_shape[i]
    return tuple(strides)


_LAUNCH = {}


def _launch_meta(n):
    m = _LAUNCH.get(n)
    if m is None:
        bn = max(2, triton.next_power_of_2(n))
        m = (bn, _num_warps(bn))
        _LAUNCH[n] = m
    return m


def ldl_factor_ex(self, hermitian=False, check_errors=False, **_kwargs):
    logger.debug("GEMS_HYGON LDL_FACTOR_EX")
    A = self
    shape = A.shape
    n = shape[-1]
    batch_shape = shape[:-2]
    numel_batch = 1
    for d in batch_shape:
        numel_batch *= d

    out_shape = batch_shape + (n, n)
    ld_strides = _ld_strides(batch_shape, n)
    if numel_batch == 0 or n == 0:
        LD = torch.empty_strided(out_shape, ld_strides, dtype=A.dtype, device=A.device)
        piv = torch.empty(batch_shape + (n,), dtype=torch.int32, device=A.device)
        info = torch.empty(batch_shape, dtype=torch.int32, device=A.device)
        return LD, piv, info

    LD = torch.empty_strided(out_shape, ld_strides, dtype=A.dtype, device=A.device)
    piv = torch.empty(batch_shape + (n,), dtype=torch.int32, device=A.device)
    info = torch.empty(batch_shape, dtype=torch.int32, device=A.device)

    sA = A.stride(-3) if len(shape) > 2 else 0
    bn, nw = _launch_meta(n)
    _ldl_kernel[(numel_batch,)](
        A,
        LD,
        piv,
        info,
        n,
        sA,
        A.stride(-2),
        A.stride(-1),
        n * n,
        LD.stride(-2),
        LD.stride(-1),
        BLOCK_N=bn,
        num_warps=nw,
        num_stages=1,
    )
    return LD, piv, info
