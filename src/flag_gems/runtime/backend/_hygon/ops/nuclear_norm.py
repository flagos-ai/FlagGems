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

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _nuclear_norm_kernel(
    A,
    OUT,
    WORK,
    stride_row,
    stride_col,
    bs0,
    bs1,
    bs2,
    bs3,
    bs4,
    bs5,
    st0,
    st1,
    st2,
    st3,
    st4,
    st5,
    NB: tl.constexpr,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    TALL: tl.constexpr,
    ROUND: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
    NEED_MASK: tl.constexpr,
    MAX_SWEEPS: tl.constexpr,
    TOL: tl.constexpr,
):
    """Sum of singular values per matrix via one-sided cyclic Jacobi.

    One program per matrix. The logical working matrix is [ROWS, K] (K columns of
    length ROWS) staged in scratch so rotations can rewrite columns. Each Jacobi
    step rotates a whole perfect matching of disjoint column pairs at once; the
    pair tiles are [BLOCK_P, BLOCK_R] so the unit-stride row axis is innermost and
    scratch offsets stay in int32.
    """
    pid = tl.program_id(0).to(tl.int64)

    rem = pid
    off = tl.zeros((), dtype=tl.int64)
    i5 = rem % bs5
    rem = rem // bs5
    off += i5 * st5
    i4 = rem % bs4
    rem = rem // bs4
    off += i4 * st4
    i3 = rem % bs3
    rem = rem // bs3
    off += i3 * st3
    i2 = rem % bs2
    rem = rem // bs2
    off += i2 * st2
    i1 = rem % bs1
    rem = rem // bs1
    off += i1 * st1
    i0 = rem % bs0
    off += i0 * st0

    a_base = A + off
    r = tl.arange(0, BLOCK_R)
    rmask = r < ROWS
    c0 = tl.arange(0, BLOCK_C)
    w_base = WORK + pid * (K * ROWS)
    row_off = r[None, :]

    # stage the logical [ROWS, K] matrix into scratch (vectorized column chunks)
    cbase = 0
    while cbase < K:
        cols = cbase + c0
        cmask = cols < K
        cm64 = cols.to(tl.int64)
        if TALL:
            tile = tl.load(
                a_base + cm64[:, None] * stride_col + row_off * stride_row,
                mask=cmask[:, None] & rmask[None, :],
                other=0.0,
            )
        else:
            tile = tl.load(
                a_base + cm64[:, None] * stride_row + row_off * stride_col,
                mask=cmask[:, None] & rmask[None, :],
                other=0.0,
            )
        tl.store(
            w_base + cols[:, None] * ROWS + row_off,
            tile,
            mask=cmask[:, None] & rmask[None, :],
        )
        cbase += BLOCK_C

    # cyclic sweeps of parallel (disjoint-pair) Jacobi rotations
    sweep = 0
    mx = tl.full((), 1.0, dtype=tl.float32)
    while (sweep < MAX_SWEEPS) & (mx > TOL):
        mx = tl.full((), 0.0, dtype=tl.float32)
        step = 0
        while step < ROUND - 1:
            pbase = 0
            while pbase < HALF:
                jj = pbase + tl.arange(0, BLOCK_P)
                ring = ROUND - 1
                pos_p = jj
                pos_q = ROUND - 1 - jj
                pi = tl.where(pos_p == 0, 0, ((pos_p + ring - step - 1) % ring) + 1)
                qi = tl.where(pos_q == 0, 0, ((pos_q + ring - step - 1) % ring) + 1)
                swap = pi > qi
                p = tl.where(swap, qi, pi)
                q = tl.where(swap, pi, qi)
                if NEED_MASK:
                    valid = (jj < HALF) & (p < K) & (q < K)
                    vm = valid[:, None] & rmask[None, :]
                    ap = tl.load(
                        w_base + p[:, None] * ROWS + row_off, mask=vm, other=0.0
                    )
                    aq = tl.load(
                        w_base + q[:, None] * ROWS + row_off, mask=vm, other=0.0
                    )
                else:
                    ap = tl.load(
                        w_base + p[:, None] * ROWS + row_off,
                        mask=rmask[None, :],
                        other=0.0,
                    )
                    aq = tl.load(
                        w_base + q[:, None] * ROWS + row_off,
                        mask=rmask[None, :],
                        other=0.0,
                    )

                alpha = tl.sum(ap * ap, axis=1)
                beta = tl.sum(aq * aq, axis=1)
                gamma = tl.sum(ap * aq, axis=1)
                # fp32-only correlation: |gamma| * rsqrt(alpha*beta). No division
                # and no subnormal literal, so nothing is promoted to fp64.
                prod = alpha * beta
                corr = tl.where(prod > 0.0, tl.abs(gamma) * tl.rsqrt(prod), 0.0).to(
                    tl.float32
                )
                mx = tl.maximum(mx, tl.max(corr, axis=0))

                active = corr > TOL
                safe_gamma = tl.where(active, gamma, 1.0)
                tau = (beta - alpha) / (safe_gamma + safe_gamma)
                sign_tau = tl.where(tau >= 0.0, 1.0, -1.0)
                t = sign_tau / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
                c = tl.rsqrt(1.0 + t * t)
                s_rot = t * c
                c = tl.where(active, c, 1.0)
                s_rot = tl.where(active, s_rot, 0.0)

                new_ap = c[:, None] * ap - s_rot[:, None] * aq
                new_aq = s_rot[:, None] * ap + c[:, None] * aq
                if NEED_MASK:
                    tl.store(w_base + p[:, None] * ROWS + row_off, new_ap, mask=vm)
                    tl.store(w_base + q[:, None] * ROWS + row_off, new_aq, mask=vm)
                else:
                    tl.store(
                        w_base + p[:, None] * ROWS + row_off,
                        new_ap,
                        mask=rmask[None, :],
                    )
                    tl.store(
                        w_base + q[:, None] * ROWS + row_off,
                        new_aq,
                        mask=rmask[None, :],
                    )
                pbase += BLOCK_P
            step += 1
        sweep += 1

    # sum of the (now orthogonal) column norms = sum of singular values
    total = tl.zeros((), dtype=tl.float32)
    jbase = 0
    while jbase < K:
        cols = jbase + c0
        cmask = cols < K
        tile = tl.load(
            w_base + cols[:, None] * ROWS + row_off,
            mask=cmask[:, None] & rmask[None, :],
            other=0.0,
        )
        norms = tl.sqrt(tl.maximum(tl.sum(tile * tile, axis=1), 0.0))
        total += tl.sum(tl.where(cmask, norms, 0.0), axis=0)
        jbase += BLOCK_C

    tl.store(OUT + pid, total)


@triton.jit
def _stage_kernel(
    A,
    WORK,
    stride_row,
    stride_col,
    bs0,
    bs1,
    bs2,
    bs3,
    bs4,
    bs5,
    st0,
    st1,
    st2,
    st3,
    st4,
    st5,
    NB: tl.constexpr,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    TALL: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Copy the logical [ROWS, K] matrix into contiguous column-major scratch."""
    pid = tl.program_id(0).to(tl.int64)

    rem = pid
    off = tl.zeros((), dtype=tl.int64)
    i5 = rem % bs5
    rem = rem // bs5
    off += i5 * st5
    i4 = rem % bs4
    rem = rem // bs4
    off += i4 * st4
    i3 = rem % bs3
    rem = rem // bs3
    off += i3 * st3
    i2 = rem % bs2
    rem = rem // bs2
    off += i2 * st2
    i1 = rem % bs1
    rem = rem // bs1
    off += i1 * st1
    i0 = rem % bs0
    off += i0 * st0

    a_base = A + off
    r = tl.arange(0, BLOCK_R)
    rmask = r < ROWS
    c0 = tl.arange(0, BLOCK_C)
    w_base = WORK + pid * (K * ROWS)
    row_off = r[None, :]

    cbase = 0
    while cbase < K:
        cols = cbase + c0
        cmask = cols < K
        cm64 = cols.to(tl.int64)
        if TALL:
            tile = tl.load(
                a_base + cm64[:, None] * stride_col + row_off * stride_row,
                mask=cmask[:, None] & rmask[None, :],
                other=0.0,
            )
        else:
            tile = tl.load(
                a_base + cm64[:, None] * stride_row + row_off * stride_col,
                mask=cmask[:, None] & rmask[None, :],
                other=0.0,
            )
        tl.store(
            w_base + cols[:, None] * ROWS + row_off,
            tile,
            mask=cmask[:, None] & rmask[None, :],
        )
        cbase += BLOCK_C


@triton.jit
def _pair_step_kernel(
    WORK,
    step,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    ROUND: tl.constexpr,
    BLOCK_R: tl.constexpr,
    TOL: tl.constexpr,
):
    """Rotate one column pair of one matrix; grid = (n_mats, ROUND//2).

    One launch per Jacobi step spreads the matching over many CUs; every pair in a
    step is disjoint, so the launches are the only synchronization.
    """
    pid = tl.program_id(0).to(tl.int64)
    pair = tl.program_id(1)
    ring = ROUND - 1
    pos_p = pair
    pos_q = ROUND - 1 - pair
    pi = tl.where(pos_p == 0, 0, ((pos_p + ring - step - 1) % ring) + 1)
    qi = tl.where(pos_q == 0, 0, ((pos_q + ring - step - 1) % ring) + 1)
    swap = pi > qi
    p = tl.where(swap, qi, pi)
    q = tl.where(swap, pi, qi)

    r = tl.arange(0, BLOCK_R)
    rmask = r < ROWS
    w_base = WORK + pid * (K * ROWS)
    ap = tl.load(w_base + p * ROWS + r, mask=rmask, other=0.0)
    aq = tl.load(w_base + q * ROWS + r, mask=rmask, other=0.0)
    alpha = tl.sum(ap * ap)
    beta = tl.sum(aq * aq)
    gamma = tl.sum(ap * aq)
    prod = alpha * beta
    corr = tl.where(prod > 0.0, tl.abs(gamma) * tl.rsqrt(prod), 0.0).to(tl.float32)
    active = corr > TOL
    safe_gamma = tl.where(active, gamma, 1.0)
    tau = (beta - alpha) / (safe_gamma + safe_gamma)
    sign_tau = tl.where(tau >= 0.0, 1.0, -1.0)
    t = sign_tau / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
    c = tl.rsqrt(1.0 + t * t)
    s_rot = t * c
    c = tl.where(active, c, 1.0)
    s_rot = tl.where(active, s_rot, 0.0)
    tl.store(w_base + p * ROWS + r, c * ap - s_rot * aq, mask=rmask)
    tl.store(w_base + q * ROWS + r, s_rot * ap + c * aq, mask=rmask)


@triton.jit
def _final_kernel(
    WORK,
    OUT,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Sum of the working matrix's column norms = sum of singular values."""
    pid = tl.program_id(0).to(tl.int64)
    r = tl.arange(0, BLOCK_R)
    rmask = r < ROWS
    c0 = tl.arange(0, BLOCK_C)
    w_base = WORK + pid * (K * ROWS)
    row_off = r[None, :]

    total = tl.zeros((), dtype=tl.float32)
    jbase = 0
    while jbase < K:
        cols = jbase + c0
        cmask = cols < K
        tile = tl.load(
            w_base + cols[:, None] * ROWS + row_off,
            mask=cmask[:, None] & rmask[None, :],
            other=0.0,
        )
        norms = tl.sqrt(tl.maximum(tl.sum(tile * tile, axis=1), 0.0))
        total += tl.sum(tl.where(cmask, norms, 0.0), axis=0)
        jbase += BLOCK_C
    tl.store(OUT + pid, total)


def nuclear_norm(self, keepdim=False):
    logger.debug("GEMS_HYGON NUCLEAR_NORM")
    A = self
    if A.dim() < 2:
        raise RuntimeError("nuclear_norm: Expected a tensor with 2 or more dimensions")
    if A.dtype != torch.float32:
        raise AssertionError(f"Expected float32, got {A.dtype}")

    m = int(A.shape[-2])
    n = int(A.shape[-1])
    batch_shape = tuple(int(d) for d in A.shape[:-2])
    k = m if m < n else n
    rows = n if m < n else m
    tall = m >= n

    n_mats = 1
    for d in batch_shape:
        n_mats *= d

    out = torch.empty(
        batch_shape + ((1, 1) if keepdim else ()), dtype=A.dtype, device=A.device
    )
    if n_mats == 0:
        return out

    work = torch.empty(max(1, n_mats * k * rows), dtype=torch.float32, device=A.device)

    sizes = [1, 1, 1, 1, 1, 1]
    strides = [0, 0, 0, 0, 0, 0]
    pad = 6 - len(batch_shape)
    for i, d in enumerate(batch_shape):
        sizes[pad + i] = d
        strides[pad + i] = int(A.stride(i))

    round_size = k if (k % 2 == 0 and k > 0) else k + 1
    half = round_size // 2
    block_r = max(1, triton.next_power_of_2(max(rows, 1)))
    cap = 4096 // block_r
    if cap < 1:
        cap = 1
    block_p = max(1, min(triton.next_power_of_2(max(half, 1)), cap))
    block_c = max(1, min(triton.next_power_of_2(max(k, 1)), cap))
    elems = block_r * max(block_p, block_c)
    need_mask = (k % 2 != 0) or (block_p > half)
    # NOTE: 8 warps miscompiles on this backend for several narrow tiles
    # ([32,64] at k=64 ~4% low, masked [32,128] at 100x60 ~10% low, [4,1024] at
    # 8x1024 ~10% low), while [32,128] at k=128 and [16,256] at k=200/256 were
    # verified accurate at 8 warps (<= 1.8e-4). Allow 8 warps only for exactly
    # those wide, unmasked tiles; everything else stays at <= 4 warps.
    if (not need_mask) and block_p >= 16 and 128 <= block_r <= 256 and elems >= 2048:
        num_warps = 8
    elif elems <= 256:
        num_warps = 1
    elif elems <= 512:
        num_warps = 2
    else:
        num_warps = 4

    if k >= 340 and k % 2 == 0:
        # Large square-ish matrices: spread every Jacobi step over many CUs.
        sweeps = 4
        _stage_kernel[(n_mats,)](
            A,
            work,
            int(A.stride(-2)),
            int(A.stride(-1)),
            sizes[0],
            sizes[1],
            sizes[2],
            sizes[3],
            sizes[4],
            sizes[5],
            strides[0],
            strides[1],
            strides[2],
            strides[3],
            strides[4],
            strides[5],
            NB=len(batch_shape),
            K=k,
            ROWS=rows,
            TALL=tall,
            BLOCK_R=block_r,
            BLOCK_C=block_c,
            num_warps=num_warps,
        )
        pair_warps = 2 if block_r >= 256 else 1
        for _ in range(sweeps):
            for st in range(round_size - 1):
                _pair_step_kernel[(n_mats, half)](
                    work,
                    st,
                    K=k,
                    ROWS=rows,
                    ROUND=round_size,
                    BLOCK_R=block_r,
                    TOL=1e-3,
                    num_warps=pair_warps,
                )
        _final_kernel[(n_mats,)](
            work,
            out,
            K=k,
            ROWS=rows,
            BLOCK_R=block_r,
            BLOCK_C=block_c,
            num_warps=num_warps,
        )
        return out

    _nuclear_norm_kernel[(n_mats,)](
        A,
        out,
        work,
        int(A.stride(-2)),
        int(A.stride(-1)),
        sizes[0],
        sizes[1],
        sizes[2],
        sizes[3],
        sizes[4],
        sizes[5],
        strides[0],
        strides[1],
        strides[2],
        strides[3],
        strides[4],
        strides[5],
        NB=len(batch_shape),
        K=k,
        ROWS=rows,
        TALL=tall,
        ROUND=round_size,
        HALF=half,
        BLOCK_R=block_r,
        BLOCK_P=block_p,
        BLOCK_C=block_c,
        NEED_MASK=need_mask,
        MAX_SWEEPS=(4 if k > 0 else 0),
        TOL=1e-3,
        num_warps=num_warps,
    )
    return out
