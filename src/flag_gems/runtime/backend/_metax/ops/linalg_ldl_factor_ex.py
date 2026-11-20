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

"""LDL^H factorization with Bunch-Kaufman pivoting (LAPACK xSYTF2/xHETF2 semantics).

One Triton program owns one padded n x n symmetric working tile in registers and
performs the whole sequential factorization for that batch matrix, reproducing
LAPACK's compact LD storage, pivot rule and info reporting.

Per-column critical path: the working column and the pivot-candidate row are the
only tile-wide reductions; interchange vectors are reconstructed from them by
1-D permutations instead of extra tile reductions.  A dynamic fast path handles
the LAPACK "no interchange, 1x1 pivot" step (what every step of a positive
definite matrix takes) with a single tile reduction and no interchange work; the
full Bunch-Kaufman path runs only when a pivot is actually needed.
"""

import math

import torch
import triton
import triton.language as tl

# (1 + sqrt(17)) / 8, rounded the way LAPACK rounds it in the working precision.
_ALPHA32 = float(
    torch.tensor((1.0 + math.sqrt(17.0)) / 8.0, dtype=torch.float32).item()
)
_ALPHA64 = (1.0 + math.sqrt(17.0)) / 8.0

_BN_MAX = 64


@triton.jit
def _ldl_body(
    A,
    LD,
    PIV,
    INFO,
    n,
    off,
    a_rs,
    a_cs,
    ld_rs,
    ld_cs,
    ALPHA: tl.constexpr,
    BN: tl.constexpr,
    NC: tl.constexpr,
):
    rows = tl.arange(0, BN)[:, None]
    cols = tl.arange(0, BN)[None, :]
    i1 = tl.arange(0, BN)
    inb = (rows < NC) & (cols < NC)

    a_ptr = A + off + rows * a_rs + cols * a_cs
    low = tl.load(a_ptr, mask=inb & (rows >= cols), other=0.0)
    W = tl.where(rows >= cols, low, tl.trans(low))
    W = tl.where(inb, W, 0.0)

    k = 0
    info_val = 0
    # 1x1 steps produce ipiv(k)=k+1, so the identity default needs no per-step write
    piv_vec = i1 + 1
    # L columns are scaled by 1/D(k,k) for output only; nothing later reads them, so
    # the scaling is deferred to one final pass instead of a per-step tile select.
    scale_vec = tl.full((BN,), 1.0, W.dtype)

    for _ in range(0, n):
        active = k < n

        # --- working column k (tile stays symmetric, so row k == column k) ---
        # Extracted as the k-th column and reduced along axis 1: that axis is the
        # one the blocked layout splits across fewest threads, so the butterfly
        # stage count is halved versus reducing along the row axis.
        row_k = tl.sum(tl.where(cols == k, W, 0.0), axis=1)
        dkk0 = tl.sum(tl.where(i1 == k, row_k, 0.0), axis=0)
        absakk = tl.abs(dkk0)
        gt = (i1 > k) & (i1 < NC)
        av = tl.abs(row_k)
        colmax = tl.max(tl.where(gt, av, 0.0), axis=0)

        if (absakk >= ALPHA * colmax) & active:
            # LAPACK 1x1 pivot, no interchange: no pivot-related tile work at all
            iszero = absakk == 0.0
            r1 = tl.where(iszero, 0.0, 1.0 / dkk0)
            info_val = tl.where(iszero & (info_val == 0), k + 1, info_val)
            scale_vec = tl.where(i1 == k, r1, scale_vec)
            reg1 = (rows > k) & (cols > k) & inb
            W = tl.where(reg1, W - (r1 * (row_k[:, None] * row_k[None, :])), W)
            k += 1
        else:
            imax = tl.min(tl.where(gt & (av == colmax), i1, BN), axis=0)
            row_i = tl.sum(tl.where(cols == imax, W, 0.0), axis=1)
            diag_i = tl.sum(tl.where(i1 == imax, row_i, 0.0), axis=0)
            rmask = (i1 >= k) & (i1 < NC) & (i1 != imax)
            rowmax = tl.max(tl.where(rmask, tl.abs(row_i), 0.0), axis=0)

            zero_col = tl.maximum(absakk, colmax) == 0.0
            a2 = absakk >= ALPHA * colmax * (colmax / rowmax)
            a3 = tl.abs(diag_i) >= ALPHA * rowmax
            kstep = tl.where(a2 | a3, 1, 2)
            kp = tl.where(a2, k, imax)
            kstep = tl.where(zero_col, 1, kstep)
            kp = tl.where(zero_col, k, kp)

            piv_vec = tl.where(
                i1 == k, tl.where(kstep == 1, kp + 1, -(kp + 1)), piv_vec
            )
            piv_vec = tl.where((i1 == k + 1) & (kstep == 2), -(kp + 1), piv_vec)
            info_val = tl.where(zero_col & active & (info_val == 0), k + 1, info_val)

            if kstep == 1:
                # 1x1 pivot block: interchange kk=k with kp over [k:n,k:n]
                rkp = tl.where(kp == k, row_k, row_i)
                a_kk = tl.sum(tl.where(i1 == kp, rkp, 0.0), axis=0)
                a_b = tl.sum(tl.where(i1 == k, rkp, 0.0), axis=0)
                r_k = tl.where(i1 == k, a_kk, tl.where(i1 == kp, a_b, rkp))
                c_kk = tl.sum(tl.where(i1 == kp, row_k, 0.0), axis=0)
                c_b = tl.sum(tl.where(i1 == k, row_k, 0.0), axis=0)
                col_new = tl.where(i1 == k, c_kk, tl.where(i1 == kp, c_b, row_k))

                sw = (kp != k) & active
                reg = (rows >= k) & (cols >= k) & inb
                W = tl.where(reg & (rows == k), rkp[None, :], W)
                W = tl.where(sw & reg & (rows == kp), row_k[None, :], W)
                W = tl.where(reg & (cols == k), r_k[:, None], W)
                W = tl.where(sw & reg & (cols == kp), col_new[:, None], W)

                dkk = tl.sum(tl.where(i1 == k, r_k, 0.0), axis=0)
                r1 = tl.where(zero_col, 0.0, 1.0 / dkk)
                scale_vec = tl.where(i1 == k, r1, scale_vec)
                reg1 = (rows > k) & (cols > k) & inb
                W = tl.where(reg1, W - (r1 * (r_k[:, None] * r_k[None, :])), W)
            else:
                # 2x2 pivot block: interchange kk=k+1 with kp=imax over [kk:n,kk:n]
                kk = k + 1
                row_k1 = tl.sum(tl.where(cols == kk, W, 0.0), axis=1)
                a1v = tl.sum(tl.where(i1 == kp, row_k1, 0.0), axis=0)
                a2v = tl.sum(tl.where(i1 == kk, row_k1, 0.0), axis=0)
                A_vec = tl.where(i1 == kk, a1v, tl.where(i1 == kp, a2v, row_k1))
                b1v = tl.sum(tl.where(i1 == kp, row_i, 0.0), axis=0)
                b2v = tl.sum(tl.where(i1 == kk, row_i, 0.0), axis=0)
                B_vec = tl.where(i1 == kk, b1v, tl.where(i1 == kp, b2v, row_i))

                reg = (rows >= kk) & (cols >= kk) & inb
                W = tl.where(reg & (rows == kk), row_i[None, :], W)
                W = tl.where(reg & (rows == kp), row_k1[None, :], W)
                W = tl.where(reg & (cols == kk), B_vec[:, None], W)
                W = tl.where(reg & (cols == kp), A_vec[:, None], W)

                # LAPACK's explicit D21 slot interchange (k+1,k) <-> (kp,k)
                va = tl.sum(tl.where(i1 == k, row_k1, 0.0), axis=0)
                vb = tl.sum(tl.where(i1 == kp, row_k, 0.0), axis=0)
                swd = (kp != kk) & active
                W = tl.where(swd & inb & (rows == kk) & (cols == k), vb, W)
                W = tl.where(swd & inb & (rows == k) & (cols == kk), vb, W)
                W = tl.where(swd & inb & (rows == kp) & (cols == k), va, W)
                W = tl.where(swd & inb & (rows == k) & (cols == kp), va, W)

                # post-interchange working columns k and k+1
                d1v = tl.sum(tl.where(i1 == kp, row_k, 0.0), axis=0)
                d2v = tl.sum(tl.where(i1 == kk, row_k, 0.0), axis=0)
                r_k = tl.where(i1 == kk, d1v, tl.where(i1 == kp, d2v, row_k))
                r_k1 = B_vec

                d11 = tl.sum(tl.where(i1 == k, r_k, 0.0), axis=0)
                d21 = tl.where(swd, vb, va)
                d22 = tl.sum(tl.where(i1 == kk, r_k1, 0.0), axis=0)

                D11 = d22 / d21
                D22 = d11 / d21
                T = 1.0 / (D11 * D22 - 1.0)
                D21p = T / d21
                wk = D21p * (D11 * r_k - r_k1)
                wkp1 = D21p * (D22 * r_k1 - r_k)

                reg2 = (rows > kk) & (cols > kk) & inb
                upd = r_k[:, None] * wk[None, :] + r_k1[:, None] * wkp1[None, :]
                W = tl.where(reg2, W - upd, W)
                W = tl.where(((rows > kk) & (cols == k)) & inb, wk[:, None], W)
                W = tl.where(((rows == k) & (cols > kk)) & inb, wk[None, :], W)
                W = tl.where(((rows > kk) & (cols == kk)) & inb, wkp1[:, None], W)
                W = tl.where(((rows == kk) & (cols > kk)) & inb, wkp1[None, :], W)

            k = tl.where(active, k + kstep, k)

    one = tl.arange(0, 1)
    tl.store(INFO + one, info_val + tl.zeros((1,), tl.int32))
    tl.store(PIV + i1, piv_vec, mask=i1 < NC)
    lscale = tl.where(rows == cols, 1.0, scale_vec[None, :])
    out = tl.where(rows >= cols, W * lscale, 0.0)
    tl.store(LD + rows * ld_rs + cols * ld_cs, out, mask=inb)


@triton.jit
def _ldl_2d_kernel(
    A,
    LD,
    PIV,
    INFO,
    n,
    NC: tl.constexpr,
    ALPHA: tl.constexpr,
    BN: tl.constexpr,
):
    _ldl_body(A, LD, PIV, INFO, n, 0, NC, 1, NC, 1, ALPHA, BN, NC)


@triton.jit
def _ldl_batch_kernel(
    A,
    LD,
    PIV,
    INFO,
    n,
    a_rs,
    a_cs,
    bd0,
    bd1,
    bd2,
    bd3,
    bs0,
    bs1,
    bs2,
    bs3,
    ALPHA: tl.constexpr,
    BN: tl.constexpr,
    NC: tl.constexpr,
    NBD: tl.constexpr,
):
    pid = tl.program_id(0)
    off = 0
    r = pid
    if NBD >= 1:
        off += (r % bd0) * bs0
        r = r // bd0
    if NBD >= 2:
        off += (r % bd1) * bs1
        r = r // bd1
    if NBD >= 3:
        off += (r % bd2) * bs2
        r = r // bd2
    if NBD >= 4:
        off += (r % bd3) * bs3
    _ldl_body(
        A,
        LD + pid * NC * NC,
        PIV + pid * NC,
        INFO + pid,
        n,
        off,
        a_rs,
        a_cs,
        NC,
        1,
        ALPHA,
        BN,
        NC,
    )


def _next_pow2(v):
    p = 1
    while p < v:
        p *= 2
    return p


def ldl_factor_ex(self, *args, hermitian=False, check_errors=False, **kwargs):
    A = self
    if args:
        hermitian = bool(args[0])  # noqa: F841 - accepted for ATen signature parity
        if len(args) > 1:
            check_errors = bool(args[1])

    if not isinstance(A, torch.Tensor):
        raise TypeError("linalg_ldl_factor_ex: input must be a Tensor")
    shape = A.shape
    nbd = len(shape) - 2
    if nbd < 0:
        raise ValueError("linalg_ldl_factor_ex: input must have at least 2 dimensions")
    n = shape[-1]
    if n != shape[-2]:
        raise ValueError(
            "linalg_ldl_factor_ex: input must be batches of square matrices"
        )
    dt = A.dtype
    if dt is torch.float32:
        alpha = _ALPHA32
    elif dt is torch.float64:
        alpha = _ALPHA64
    else:
        raise NotImplementedError("linalg_ldl_factor_ex: unsupported dtype " + str(dt))
    if n > _BN_MAX:
        raise NotImplementedError(
            "linalg_ldl_factor_ex: matrix size %d exceeds the supported maximum %d"
            % (n, _BN_MAX)
        )
    BN = (
        4 if n <= 4 else (8 if n <= 8 else (16 if n <= 16 else (32 if n <= 32 else 64)))
    )
    # One warp keeps every reduction warp-local: with 2 warps the blocked layout
    # splits the reduction axis (warpsPerCTA=[2,1]) and each step pays cross-warp
    # shared-memory exchange plus barriers.
    num_warps = 1 if n <= 32 else 8

    # Row-major single matrix: the whole per-call host path below is written so the
    # outputs are the objects the reference returns (no view/reshape work), because
    # for small n the per-call host floor dominates the measured latency.
    if nbd == 0 and A.stride(0) == n and A.stride(1) == 1:
        dev = A.device
        LD = torch.empty((n, n), dtype=dt, device=dev)
        aux = torch.empty(n + 1, dtype=torch.int32, device=dev)
        piv = aux[:n]
        info = aux[n]
        _ldl_2d_kernel[(1,)](A, LD, piv, info, n, n, alpha, BN, num_warps=num_warps)
        if check_errors:
            if info.item() > 0:
                raise RuntimeError("linalg_ldl_factor_ex: the factorization failed")
        return LD, piv, info

    batch_shape = shape[:-2]
    B = 1
    for d in batch_shape:
        B *= d
    if B == 0 or n == 0:
        LD = torch.zeros(shape, dtype=dt, device=A.device)
        aux = torch.empty(B * n + 1, dtype=torch.int32, device=A.device)
        return (
            LD,
            aux[: B * n].view(B, n).reshape(*batch_shape, n),
            aux[B * n :].view(batch_shape),
        )
    if nbd > 4:
        raise NotImplementedError(
            "linalg_ldl_factor_ex: at most 4 batch dimensions are supported"
        )

    LD_out = torch.empty((B, n, n), dtype=dt, device=A.device)
    aux = torch.empty(B * n + 1, dtype=torch.int32, device=A.device)
    piv_flat = aux[: B * n]
    info_flat = aux[B * n :]

    ds = [1, 1, 1, 1]
    ss = [0, 0, 0, 0]
    for i in range(nbd):
        ds[i] = int(batch_shape[nbd - 1 - i])
        ss[i] = int(A.stride(nbd - 1 - i))
    _ldl_batch_kernel[(B,)](
        A,
        LD_out,
        piv_flat,
        info_flat,
        n,
        A.stride(-2),
        A.stride(-1),
        ds[0],
        ds[1],
        ds[2],
        ds[3],
        ss[0],
        ss[1],
        ss[2],
        ss[3],
        ALPHA=alpha,
        BN=BN,
        NC=n,
        NBD=nbd,
        num_warps=num_warps,
    )

    if check_errors:
        if max(info_flat.tolist()) > 0:
            raise RuntimeError("linalg_ldl_factor_ex: the factorization failed")

    return (
        LD_out.reshape(shape),
        piv_flat.view(B, n).reshape(*batch_shape, n),
        info_flat.view(batch_shape),
    )
