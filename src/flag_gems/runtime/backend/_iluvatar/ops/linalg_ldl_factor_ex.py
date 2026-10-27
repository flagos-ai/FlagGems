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

import torch
import triton
import triton.language as tl

# Bunch-Kaufman pivot threshold constant used by torch's ldl_factor_ex
# (LAPACK sytrf AGL criterion): ALPHA = (1 + sqrt(17)) / 32.
_ALPHA = (1.0 + 4.123105625617661) / 32.0


@triton.jit
def _ldl_1x1(act, LD, k, d, colk, info, n, NP2: tl.constexpr):
    """Apply one 1x1 pivot elimination at column k (d = act[k,k], colk = column k)."""
    r = tl.arange(0, NP2)
    c = tl.arange(0, NP2)
    factor = tl.where((r > k) & (r < n), colk / d, 0.0)
    colvec = tl.where(r == k, d, tl.where((r > k) & (r < n), factor, 0.0))
    LD = tl.where(c == k, colvec[:, None], LD)
    upd = factor[:, None] * factor[None, :] * d
    # factor is exactly zero for rows <= k (and for padding rows >= n), so upd is
    # exactly zero outside the trailing block (r > k, c > k). The masked update
    # act = where(trail, act - upd, act) is therefore a no-op there; the plain
    # subtract is bit-identical and skips the full-tile select.
    act = act - upd
    info = tl.where((d == 0) & (k + 1 < n), k + 1, info)
    return act, LD, info


@triton.jit
def _ldl_2x2(act, LD, k, rmax, info, n, NP2: tl.constexpr):
    """Apply one 2x2 Bunch-Kaufman pivot elimination at columns k, k+1."""
    r = tl.arange(0, NP2)
    c = tl.arange(0, NP2)
    # Symmetric interchange of rows/cols (k+1) <-> rmax (no-op when equal).
    row_a = tl.sum(tl.where(r[:, None] == k + 1, act, 0.0), axis=0)
    row_b = tl.sum(tl.where(r[:, None] == rmax, act, 0.0), axis=0)
    act = tl.where(
        r[:, None] == k + 1,
        row_b[None, :],
        tl.where(r[:, None] == rmax, row_a[None, :], act),
    )
    col_a = tl.sum(tl.where(c[None, :] == k + 1, act, 0.0), axis=1)
    col_b = tl.sum(tl.where(c[None, :] == rmax, act, 0.0), axis=1)
    act = tl.where(
        c[None, :] == k + 1,
        col_b[:, None],
        tl.where(c[None, :] == rmax, col_a[:, None], act),
    )

    colk = tl.sum(tl.where(c[None, :] == k, act, 0.0), axis=1)
    colk1 = tl.sum(tl.where(c[None, :] == k + 1, act, 0.0), axis=1)
    a = tl.sum(tl.where(r == k, colk, 0.0))
    b = tl.sum(tl.where(r == k + 1, colk, 0.0))
    cc = tl.sum(tl.where(r == k + 1, colk1, 0.0))
    det = a * cc - b * b
    L0 = tl.where((r > k + 1) & (r < n), (cc * colk - b * colk1) / det, 0.0)
    L1 = tl.where((r > k + 1) & (r < n), (-b * colk + a * colk1) / det, 0.0)
    T0 = a * L0 + b * L1
    T1 = b * L0 + cc * L1
    upd = T0[:, None] * L0[None, :] + T1[:, None] * L1[None, :]
    # L0/L1 are zero for rows <= k+1 (and padding), so upd is exactly zero
    # outside the trailing block; plain subtract is bit-identical (see _ldl_1x1).
    act = act - upd

    colvec0 = tl.where(
        r == k, a, tl.where(r == k + 1, b, tl.where((r > k + 1) & (r < n), L0, 0.0))
    )
    colvec1 = tl.where(r == k + 1, cc, tl.where((r > k + 1) & (r < n), L1, 0.0))
    LD = tl.where(c == k, colvec0[:, None], LD)
    LD = tl.where(c == k + 1, colvec1[:, None], LD)
    info = tl.where(((a == 0) | (cc == 0)) & (k + 2 < n), k + 1, info)
    return act, LD, info


@triton.jit
def _ldl_kernel(A_ptr, LD_ptr, PIV_ptr, INFO_ptr, n, ALPHA, NP2: tl.constexpr):
    pid = tl.program_id(0)
    r = tl.arange(0, NP2)
    c = tl.arange(0, NP2)
    base = A_ptr + pid.to(tl.int64) * n * n
    offs = r[:, None] * n + c[None, :]
    valid = (r[:, None] < n) & (c[None, :] < n)
    act = tl.load(base + offs, mask=valid, other=0.0)
    LD = tl.zeros((NP2, NP2), dtype=act.dtype)
    piv_r = tl.zeros((NP2,), dtype=tl.int32)
    info = 0
    k = 0
    pi = 0
    while k < n:
        colk = tl.sum(tl.where(c[None, :] == k, act, 0.0), axis=1)
        dk = tl.sum(tl.where(r == k, colk, 0.0))
        sub = tl.where((r > k) & (r < n), tl.abs(colk), 0.0)
        max1 = tl.max(sub, axis=0)
        adk = tl.abs(dk)
        # m = max(max1, max2) >= max1, so t = ALPHA*max1^2/m <= ALPHA*max1.
        # Therefore adk >= ALPHA*max1 already guarantees the 1x1-at-k branch
        # (ktest) without computing rmax / max2 / diagr. This fast path is the
        # common case for the (symmetric positive definite) workload family.
        fast = (max1 == 0) | (adk >= ALPHA * max1)
        if fast:
            act, LD, info = _ldl_1x1(act, LD, k, dk, colk, info, n, NP2)
            piv_r = tl.where(r == pi, k + 1, piv_r)
            k = k + 1
            pi = pi + 1
        else:
            imax = tl.argmax(sub, axis=0)
            # sub is indexed by row id directly, so argmax is already the row index.
            rmax = imax.to(tl.int32)
            # act stays symmetric, so row rmax == column rmax (bit-identical on
            # the ktest path); read the column to save one masked 2D reduction.
            colr = tl.sum(tl.where(c[None, :] == rmax, act, 0.0), axis=1)
            diagr = tl.sum(tl.where(r == rmax, colr, 0.0))
            sub2 = tl.where((r >= k) & (r < n) & (r != rmax), tl.abs(colr), 0.0)
            max2 = tl.max(sub2, axis=0)
            m = tl.maximum(max1, max2)
            t = tl.where(m > 0, ALPHA * max1 * max1 / m, 0.0)
            ktest = (max1 == 0) | (adk >= t)
            rtest = (max1 > 0) & (~ktest) & (tl.abs(diagr) >= ALPHA * m)
            if ktest:
                act, LD, info = _ldl_1x1(act, LD, k, dk, colk, info, n, NP2)
                piv_r = tl.where(r == pi, k + 1, piv_r)
                k = k + 1
                pi = pi + 1
            elif rtest:
                # Symmetric interchange of rows/cols k <-> rmax, then 1x1 at k.
                row_k = tl.sum(tl.where(r[:, None] == k, act, 0.0), axis=0)
                row_r = tl.sum(tl.where(r[:, None] == rmax, act, 0.0), axis=0)
                act = tl.where(
                    r[:, None] == k,
                    row_r[None, :],
                    tl.where(r[:, None] == rmax, row_k[None, :], act),
                )
                col_k = tl.sum(tl.where(c[None, :] == k, act, 0.0), axis=1)
                col_r = tl.sum(tl.where(c[None, :] == rmax, act, 0.0), axis=1)
                act = tl.where(
                    c[None, :] == k,
                    col_r[:, None],
                    tl.where(c[None, :] == rmax, col_k[:, None], act),
                )
                colk2 = tl.sum(tl.where(c[None, :] == k, act, 0.0), axis=1)
                d2 = tl.sum(tl.where(r == k, colk2, 0.0))
                act, LD, info = _ldl_1x1(act, LD, k, d2, colk2, info, n, NP2)
                piv_r = tl.where(r == pi, rmax + 1, piv_r)
                k = k + 1
                pi = pi + 1
            else:
                act, LD, info = _ldl_2x2(act, LD, k, rmax, info, n, NP2)
                piv_r = tl.where(r == pi, -(rmax + 1), piv_r)
                piv_r = tl.where(r == pi + 1, -(rmax + 1), piv_r)
                k = k + 2
                pi = pi + 2
    tl.store(LD_ptr + pid.to(tl.int64) * n * n + offs, LD, mask=valid)
    tl.store(PIV_ptr + pid.to(tl.int64) * n + r, piv_r, mask=r < n)
    tl.store(INFO_ptr + pid, info)


def ldl_factor_ex(self, *, hermitian=False, check_errors=False):
    n = self.shape[-1]
    batch = self.numel() // (n * n)
    LD = torch.empty_like(self)
    piv = torch.empty(self.shape[:-2] + (n,), dtype=torch.int32, device=self.device)
    info = torch.empty(self.shape[:-2], dtype=torch.int32, device=self.device)
    NP2 = max(2, triton.next_power_of_2(n))
    # Measured on BI-V150: 2 warps win for the 8x8 tile (1 elem/thread layout),
    # 1 warp wins for 16/32 tiles (cross-warp reduction barriers dominate).
    nw = 2 if NP2 == 8 else 1
    _ldl_kernel[(batch,)](self, LD, piv, info, n, _ALPHA, NP2=NP2, num_warps=nw)
    return torch.return_types.linalg_ldl_factor_ex((LD, piv, info))
