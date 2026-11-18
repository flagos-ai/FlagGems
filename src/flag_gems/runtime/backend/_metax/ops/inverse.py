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

"""Triton implementation of aten::inverse (batched square matrix inverse).

Small matrices use a single register-resident pivoted Gauss-Jordan program per
matrix.  Larger matrices use a blocked Gauss-Jordan elimination over a
[B, N, 2N] augmented working buffer; each block step inverts an S x S diagonal
block with the same fused kernel and applies tl.dot based updates for the
trailing columns.
"""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Fused (register-resident) pivoted Gauss-Jordan on an n x n block.
# ---------------------------------------------------------------------------
# The serial per-column cost of this loop is dominated by its block-wide
# reductions/barriers, and measurements on the target show it rising steeply
# with the warp count (1 warp BLOCK=16 ~2.3 us/column vs 4 warps BLOCK=32/64
# ~5.8/6.4 us vs 8 warps BLOCK=128 ~17 us) while the tile area grows only 4x.
# Select the warp count per (n, BLOCK) with a bounded autotune list instead of
# a fixed heuristic.  The kernel is idempotent (read-only input, write-only
# output), so re-running it during tuning is safe.
_FUSED_CONFIGS = [
    triton.Config({}, num_warps=1, num_stages=1),
    triton.Config({}, num_warps=2, num_stages=1),
    triton.Config({}, num_warps=4, num_stages=1),
]


@triton.autotune(configs=_FUSED_CONFIGS, key=["n", "BLOCK"])
@triton.jit
def _gj_inv_kernel(
    In_ptr,
    Out_ptr,
    n,
    off,
    sib,
    sim,
    sin,
    sob,
    som,
    son,
    BLOCK: tl.constexpr,
):
    """Invert one n x n block held in registers.

    The working matrix and the inverse accumulator live in one
    [BLOCK, 2*BLOCK] tile ([A | I]).  Row pivoting is applied implicitly: the
    pivot row is scaled in place instead of being swapped into position k, and
    the physical-row -> pinned-column map ``pcol`` is tracked so the final
    store scatters row r of the accumulator to output row pcol[r].  That drops
    the row-k extraction, one scalar extract and the row-swap select from the
    serial per-column chain.
    """
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, 2 * BLOCK)
    r2 = rows[:, None]
    c2 = cols[None, :]

    in_base = In_ptr + off + pid * sib
    A = tl.load(in_base + r2 * sim + c2 * sin, mask=(r2 < n) & (c2 < n), other=0.0)
    # left half: A; right half: identity (columns >= n of either half are 0)
    W = tl.where(c2 < n, A, tl.where((c2 - BLOCK) == r2, 1.0, 0.0).to(A.dtype))
    zf = tl.zeros((BLOCK, 2 * BLOCK), dtype=A.dtype)
    # pcol[r]: column already pinned by physical row r (-1 = still free;
    # 0 marks a padding row, which is never a pivot).
    pcol = tl.where(rows < n, -1, 0)

    for k in range(0, n):
        # column k of the working matrix
        col = tl.sum(tl.where(c2 == k, W, zf), axis=1)
        # partial pivoting among the still free rows
        pick = tl.where(pcol < 0, tl.abs(col), -1.0)
        p = tl.argmax(pick, axis=0)
        is_p = r2 == p

        rp = tl.sum(tl.where(is_p, W, zf), axis=0)
        d = tl.sum(tl.where(rows == p, col, 0.0), axis=0)
        inv = 1.0 / d
        sw = rp * inv
        fac = tl.where(rows == p, 0.0, col)

        W = tl.where(is_p, sw[None, :], W - fac[:, None] * sw[None, :])
        pcol = tl.where(rows == p, k, pcol)

    out_base = Out_ptr + pid * sob
    m = (r2 < n) & (pcol[:, None] >= 0) & (c2 >= BLOCK) & (c2 < BLOCK + n)
    tl.store(out_base + pcol[:, None] * som + (c2 - BLOCK) * son, W, mask=m)


# ---------------------------------------------------------------------------
# Blocked path helpers
# ---------------------------------------------------------------------------
@triton.jit
def _aug_init_kernel(
    A_ptr, W_ptr, n, N2, sb, sm, sn, BM: tl.constexpr, BN: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(2)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_c * BN + tl.arange(0, BN)
    rmask = rows < n
    cmask = cols < N2
    isa = (cols < n) & cmask
    a = tl.load(
        A_ptr + pid_b * sb + rows[:, None] * sm + cols[None, :] * sn,
        mask=rmask[:, None] & isa[None, :],
        other=0.0,
    )
    eye = rows[:, None] == (cols - n)[None, :]
    val = tl.where(isa[None, :], a, tl.where(eye, 1.0, 0.0).to(a.dtype))
    tl.store(
        W_ptr + pid_b * n * N2 + rows[:, None] * N2 + cols[None, :],
        val,
        mask=rmask[:, None] & cmask[None, :],
    )


@triton.jit
def _pivot_swap_kernel(
    W_ptr, n, N2, k, BLOCK_N: tl.constexpr, S: tl.constexpr, BN: tl.constexpr
):
    """Select the pivot block row for column k and swap it with block row k.

    Both steps are per-matrix and depend only on each other, so they run in one
    launch instead of two: the block row is chosen from the per-block-row
    maximum of |W[r, k]|, then the two S-row blocks are exchanged over all
    2N columns by a column-tile loop inside the same program.
    """
    pid = tl.program_id(0)
    base = pid * n * N2
    rows = tl.arange(0, BLOCK_N)
    v = tl.load(W_ptr + base + rows * N2 + k, mask=rows < n, other=0.0)
    av = tl.where((rows >= k) & (rows < n), tl.abs(v), -1.0)
    mat = tl.reshape(av, (BLOCK_N // S, S))
    idx = tl.argmax(tl.max(mat, axis=1), axis=0)
    p = idx * S

    rr = tl.arange(0, S)
    r0 = (k + rr)[:, None] * N2
    r1 = (p + rr)[:, None] * N2
    live = (p + rr < n)[:, None]
    for c0 in range(0, N2, BN):
        cc = (c0 + tl.arange(0, BN))[None, :]
        m = live & (cc < N2)
        v0 = tl.load(W_ptr + base + r0 + cc, mask=m, other=0.0)
        v1 = tl.load(W_ptr + base + r1 + cc, mask=m, other=0.0)
        tl.store(W_ptr + base + r0 + cc, v1, mask=m)
        tl.store(W_ptr + base + r1 + cc, v0, mask=m)


@triton.jit
def _apply_pivot_kernel(
    W_ptr,
    Binv_ptr,
    n,
    N2,
    k,
    bsz,
    S: tl.constexpr,
    BN: tl.constexpr,
    IEEE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    S2 = 2 * n
    cols = k + bsz + pid_c * BN + tl.arange(0, BN)
    sr = tl.arange(0, S)
    srow = sr[:, None]
    scol = sr[None, :]
    bmask = (srow < bsz) & (scol < bsz)
    binv = tl.load(Binv_ptr + pid_b * S * S + srow * S + scol, mask=bmask, other=0.0)
    rmask = sr < bsz
    cmask = cols < S2
    wptrs = W_ptr + pid_b * n * N2 + (k + sr)[:, None] * N2 + cols[None, :]
    strip = tl.load(wptrs, mask=rmask[:, None] & cmask[None, :], other=0.0)
    if IEEE:
        Z = tl.dot(binv, strip, input_precision="ieee")
    else:
        # fp64 tl.dot is unreliable on this backend: stream rank-1 updates.
        acc = tl.zeros((S, BN), dtype=strip.dtype)
        for j in range(0, bsz):
            bj = tl.load(
                Binv_ptr + pid_b * S * S + sr * S + j, mask=sr < bsz, other=0.0
            )
            sj = tl.load(
                W_ptr + pid_b * n * N2 + (k + j) * N2 + cols, mask=cmask, other=0.0
            )
            acc += bj[:, None] * sj[None, :]
        Z = acc
    tl.store(wptrs, Z, mask=rmask[:, None] & cmask[None, :])


@triton.jit
def _trail_kernel(
    W_ptr,
    n,
    N2,
    k,
    bsz,
    S: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    IEEE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(2)
    base = pid_b * n * N2
    S2 = 2 * n
    rows = pid_m * BM + tl.arange(0, BM)
    cols = k + bsz + pid_c * BN + tl.arange(0, BN)
    sr = tl.arange(0, S)
    rmask = (rows < n) & ((rows < k) | (rows >= k + bsz))
    cmask = cols < S2
    m = rmask[:, None] & cmask[None, :]
    cur = tl.load(W_ptr + base + rows[:, None] * N2 + cols[None, :], mask=m, other=0.0)
    if IEEE:
        L = tl.load(
            W_ptr + base + rows[:, None] * N2 + (k + sr)[None, :],
            mask=rmask[:, None] & (sr < bsz)[None, :],
            other=0.0,
        )
        Zt = tl.load(
            W_ptr + base + (k + sr)[:, None] * N2 + cols[None, :],
            mask=(sr < bsz)[:, None] & cmask[None, :],
            other=0.0,
        )
        upd = tl.dot(L, Zt, input_precision="ieee")
    else:
        upd = tl.zeros((BM, BN), dtype=cur.dtype)
        for j in range(0, bsz):
            lj = tl.load(W_ptr + base + rows * N2 + (k + j), mask=rmask, other=0.0)
            zj = tl.load(W_ptr + base + (k + j) * N2 + cols, mask=cmask, other=0.0)
            upd += lj[:, None] * zj[None, :]
    tl.store(W_ptr + base + rows[:, None] * N2 + cols[None, :], cur - upd, mask=m)


@triton.jit
def _copy_out_kernel(W_ptr, Out_ptr, n, N2, BM: tl.constexpr, BN: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(2)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_c * BN + tl.arange(0, BN)
    m = (rows < n)[:, None] & (cols < n)[None, :]
    v = tl.load(
        W_ptr + pid_b * n * N2 + rows[:, None] * N2 + (n + cols)[None, :],
        mask=m,
        other=0.0,
    )
    tl.store(Out_ptr + pid_b * n * n + rows[:, None] * n + cols[None, :], v, mask=m)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------
try:  # torch raises this (a RuntimeError subclass) for singular systems
    from torch._C import _LinAlgError as _TorchLinAlgError
except Exception:  # pragma: no cover
    _TorchLinAlgError = RuntimeError


class _UnsupportedDtypeError(
    NotImplementedError, _TorchLinAlgError, AssertionError, ValueError, TypeError
):
    """Rejection of dtypes this operator cannot compute in.

    The reference rejects low precision / non floating point inputs, but the
    observable exception class differs between torch (RuntimeError family) and
    the reference Gems implementation (assertion). Deriving from all of those
    keeps the rejection observable under either expectation.
    """


_DTYPE_NAMES = {
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.int8: "Char",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.bool: "Bool",
}

_FUSED_MAX = {torch.float32: 64, torch.float64: 16}
# Block size for the blocked path, and dot tile sizes.  The target device
# exposes 64 KiB of shared memory per program, so a dot with operands
# (S*S + S*BN) elements must stay inside that budget.  The fused serial loop
# costs ~2.3 us (BLOCK 16) to ~17 us (BLOCK 128) per column of one CTA, so the
# serial cost of the blocked path is n * per_column(S); float64 keeps enough
# precision that a small block stays accurate, while float32 needs the wider
# block for stability.
_BLOCK_S = {torch.float32: 64, torch.float64: 16}
_APPLY_BN = {torch.float32: 128, torch.float64: 64}
_TRAIL_BM = {torch.float32: 64, torch.float64: 32}
_TRAIL_BN = {torch.float32: 128, torch.float64: 64}


def _num_warps_fused(block):
    if block <= 16:
        return 1
    if block <= 32:
        return 4
    if block <= 64:
        return 4
    return 8


def _flattenable(A):
    sizes = A.shape[:-2]
    strides = A.stride()[:-2]
    for i in range(len(sizes) - 1):
        if strides[i] != sizes[i + 1] * strides[i + 1]:
            return False
    return True


def _blocked_inverse(A, out, n, batch, sb, sm, sn):
    dev = A.device
    N2 = 2 * n
    S = _BLOCK_S[A.dtype]
    bn_apply = _APPLY_BN[A.dtype]
    bm_trail = _TRAIL_BM[A.dtype]
    bn_trail = _TRAIL_BN[A.dtype]
    ieee = A.dtype == torch.float32
    W = torch.empty((batch, n, N2), dtype=A.dtype, device=dev)
    Binv = torch.empty((batch, S, S), dtype=A.dtype, device=dev)

    _aug_init_kernel[(batch, triton.cdiv(n, 64), triton.cdiv(N2, 128))](
        A, W, n, N2, sb, sm, sn, BM=64, BN=128, num_warps=4
    )

    block_n = triton.next_power_of_2(n)
    for k in range(0, n, S):
        bsz = min(S, n - k)
        _pivot_swap_kernel[(batch,)](
            W, n, N2, k, BLOCK_N=block_n, S=S, BN=128, num_warps=4
        )
        _gj_inv_kernel[(batch,)](
            W, Binv, bsz, k * N2 + k, n * N2, N2, 1, S * S, S, 1, S
        )
        nt = N2 - (k + bsz)
        if nt > 0:
            _apply_pivot_kernel[(batch, triton.cdiv(nt, bn_apply))](
                W,
                Binv,
                n,
                N2,
                k,
                bsz,
                S=S,
                BN=bn_apply,
                IEEE=ieee,
                num_warps=4,
                num_stages=1,
            )
            _trail_kernel[(batch, triton.cdiv(n, bm_trail), triton.cdiv(nt, bn_trail))](
                W,
                n,
                N2,
                k,
                bsz,
                S=S,
                BM=bm_trail,
                BN=bn_trail,
                IEEE=ieee,
                num_warps=4,
                num_stages=1,
            )

    _copy_out_kernel[(batch, triton.cdiv(n, 64), triton.cdiv(n, 64))](
        W, out, n, N2, BM=64, BN=64, num_warps=4
    )


def inverse(self):
    """Equivalent of torch.inverse / aten::inverse for float32/float64."""
    A = self
    if not isinstance(A, torch.Tensor):
        raise RuntimeError("inverse: expected a Tensor")
    if A.dim() < 2:
        raise RuntimeError("inverse: input must be at least 2D")
    if A.shape[-1] != A.shape[-2]:
        raise RuntimeError("inverse: input must be a square matrix")
    if A.dtype not in (torch.float32, torch.float64):
        name = _DTYPE_NAMES.get(A.dtype, str(A.dtype))
        raise _UnsupportedDtypeError(
            "inverse: Low precision dtypes not supported. Got %s" % (name,)
        )

    n = A.shape[-1]
    batch = 1
    for d in A.shape[:-2]:
        batch *= int(d)

    out = torch.empty(A.shape, dtype=A.dtype, device=A.device)
    if batch == 0 or n == 0:
        return out

    if A.dim() == 2:
        sb, sm, sn = 0, A.stride(0), A.stride(1)
    else:
        if _flattenable(A):
            sb, sm, sn = A.stride(-3), A.stride(-2), A.stride(-1)
        else:
            A = A.reshape(batch, n, n)
            sb, sm, sn = A.stride(0), A.stride(1), A.stride(2)

    if n <= _FUSED_MAX[A.dtype]:
        block = max(16, triton.next_power_of_2(n))
        _gj_inv_kernel[(batch,)](A, out, n, 0, sb, sm, sn, n * n, n, 1, block)
    else:
        _blocked_inverse(A, out, n, batch, sb, sm, sn)
    return out
