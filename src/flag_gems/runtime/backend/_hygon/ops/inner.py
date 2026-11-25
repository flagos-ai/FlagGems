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

"""Triton implementation of ``torch.inner`` (generalized inner product).

``inner(A, B)`` contracts the last dimension of both operands and keeps the
leading dimensions per side:

    out[a_0, ..., a_{r-2}, b_0, ..., b_{s-2}] =
        sum_k A[a_0, ..., a_{r-2}, k] * B[b_0, ..., b_{s-2}, k]

which is ``A2d @ B2d.T`` with ``A2d:(M, K)`` and ``B2d:(N, K)``.

Dispatch (all numerical work stays in the device kernels below):
  * any 0-D operand       -> broadcast scalar multiply (``mul`` semantics)
  * ``N == 1`` / ``M == 1`` -> tiled K-reduction (dot product / matvec / mat-mat
                             with a single row or column; ``tl.dot`` would
                             waste a whole 16-wide tile on the degenerate axis)
  * dot-capable dtypes    -> grouped tiled GEMM with fp32 accumulation
  * other dtypes (fp64)   -> one program per output element, fp64 accumulate
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_DOT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _roff(idx, LR: tl.constexpr, rs, d1, t1, d2, t2, d3, t3):
    """Storage offset of ``idx`` rows along the leading dimensions.

    ``LR == 0`` means the leading dimensions collapse into one stride ``rs``.
    Otherwise the row index is decomposed over the (reversed) leading shape so
    that the last leading dimension varies fastest.
    """
    if LR == 0:
        return idx.to(tl.int64) * rs
    off = idx.to(tl.int64) * 0
    rem = idx
    if LR >= 2:
        off += (rem % d1).to(tl.int64) * t1
        rem = rem // d1
    if LR >= 3:
        off += (rem % d2).to(tl.int64) * t2
        rem = rem // d2
    if LR >= 4:
        off += (rem % d3).to(tl.int64) * t3
        rem = rem // d3
    return off


@triton.jit
def _inner_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    DT,
    a_rs,
    a_ks,
    b_rs,
    b_ks,
    a_d1,
    a_t1,
    a_d2,
    a_t2,
    a_d3,
    a_t3,
    b_d1,
    b_t1,
    b_d2,
    b_t2,
    b_d3,
    b_t3,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    A_LR: tl.constexpr,
    B_LR: tl.constexpr,
    HAS_MASK: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_rows = _roff(offs_m, A_LR, a_rs, a_d1, a_t1, a_d2, a_t2, a_d3, a_t3)
    b_rows = _roff(offs_n, B_LR, b_rs, b_d1, b_t1, b_d2, b_t2, b_d3, b_t3)

    a_ptrs = a_ptr + a_rows[:, None] + (offs_k.to(tl.int64) * a_ks)[None, :]
    b_ptrs = b_ptr + b_rows[None, :] + (offs_k.to(tl.int64) * b_ks)[:, None]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        if HAS_MASK:
            kmask = (k0 + offs_k) < K
            a = tl.load(a_ptrs, mask=(offs_m < M)[:, None] & kmask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=kmask[:, None] & (offs_n < N)[None, :], other=0.0)
        else:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        if IS_FP32:
            acc = tl.dot(a, b, acc, input_precision="ieee")
        else:
            acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * a_ks
        b_ptrs += BLOCK_K * b_ks

    c_ptrs = c_ptr + offs_m.to(tl.int64)[:, None] * N + offs_n[None, :]
    if HAS_MASK:
        tl.store(
            c_ptrs,
            acc.to(c_ptr.dtype.element_ty),
            mask=(offs_m < M)[:, None] & (offs_n < N)[None, :],
        )
    else:
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))


@triton.jit
def _inner_dot_kernel(
    a_ptr, b_ptr, c_ptr, K, a_ks, b_ks, BLOCK_K: tl.constexpr, MASK_K: tl.constexpr
):
    """Pure 1-D contraction: one program contracts the whole K extent.

    A single program issues every element of both operands in one wide
    iteration (or a few), which maximizes memory-level parallelism; the
    per-lane partial products are then tree-reduced in fp32.
    """
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_k.to(tl.int64) * a_ks
    b_ptrs = b_ptr + offs_k.to(tl.int64) * b_ks
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        if MASK_K:
            kmask = (k0 + offs_k) < K
            a = tl.load(a_ptrs, mask=kmask, other=0.0).to(tl.float32)
            b = tl.load(b_ptrs, mask=kmask, other=0.0).to(tl.float32)
        else:
            a = tl.load(a_ptrs).to(tl.float32)
            b = tl.load(b_ptrs).to(tl.float32)
        acc += a * b
        a_ptrs += BLOCK_K * a_ks
        b_ptrs += BLOCK_K * b_ks
    tl.store(c_ptr, tl.sum(acc, axis=0).to(c_ptr.dtype.element_ty))


@triton.jit
def _inner_reduce_kernel(
    p_ptr,
    q_ptr,
    c_ptr,
    R,
    K,
    p_rs,
    p_ks,
    q_ks,
    p_d1,
    p_t1,
    p_d2,
    p_t2,
    p_d3,
    p_t3,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
    P_LR: tl.constexpr,
    MASK_R: tl.constexpr,
    MASK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    p_rows = _roff(rows, P_LR, p_rs, p_d1, p_t1, p_d2, p_t2, p_d3, p_t3)
    offs_k = tl.arange(0, BLOCK_K)
    p_ptrs = p_ptr + p_rows[:, None] + (offs_k.to(tl.int64) * p_ks)[None, :]
    q_ptrs = q_ptr + offs_k.to(tl.int64) * q_ks

    acc = tl.zeros((BLOCK_R,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        if MASK_K:
            kmask = (k0 + offs_k) < K
            q = tl.load(q_ptrs, mask=kmask, other=0.0).to(tl.float32)
            if MASK_R:
                p = tl.load(
                    p_ptrs, mask=(rows < R)[:, None] & kmask[None, :], other=0.0
                ).to(tl.float32)
            else:
                p = tl.load(p_ptrs, mask=kmask[None, :], other=0.0).to(tl.float32)
        else:
            q = tl.load(q_ptrs).to(tl.float32)
            if MASK_R:
                p = tl.load(p_ptrs, mask=(rows < R)[:, None], other=0.0).to(tl.float32)
            else:
                p = tl.load(p_ptrs).to(tl.float32)
        acc += tl.sum(p * q[None, :], axis=1)
        p_ptrs += BLOCK_K * p_ks
        q_ptrs += BLOCK_K * q_ks

    if MASK_R:
        tl.store(c_ptr + rows, acc.to(c_ptr.dtype.element_ty), mask=rows < R)
    else:
        tl.store(c_ptr + rows, acc.to(c_ptr.dtype.element_ty))


@triton.jit
def _inner_elem_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    a_rs,
    a_ks,
    b_rs,
    b_ks,
    a_d1,
    a_t1,
    a_d2,
    a_t2,
    a_d3,
    a_t3,
    b_d1,
    b_t1,
    b_d2,
    b_t2,
    b_d3,
    b_t3,
    BLOCK_K: tl.constexpr,
    A_LR: tl.constexpr,
    B_LR: tl.constexpr,
):
    """One program per output element; used for dtypes without ``tl.dot``."""
    pid = tl.program_id(0)
    m = pid // N
    n = pid % N
    offs_k = tl.arange(0, BLOCK_K)
    a_base = a_ptr + _roff(m, A_LR, a_rs, a_d1, a_t1, a_d2, a_t2, a_d3, a_t3)
    b_base = b_ptr + _roff(n, B_LR, b_rs, b_d1, b_t1, b_d2, b_t2, b_d3, b_t3)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float64)
    for k0 in range(0, K, BLOCK_K):
        cur = k0 + offs_k
        mask = cur < K
        a = tl.load(a_base + cur.to(tl.int64) * a_ks, mask=mask, other=0.0).to(
            tl.float64
        )
        b = tl.load(b_base + cur.to(tl.int64) * b_ks, mask=mask, other=0.0).to(
            tl.float64
        )
        acc += a * b
    tl.store(c_ptr + pid, tl.sum(acc, axis=0).to(c_ptr.dtype.element_ty))


@triton.jit
def _mul_scalar_flat_kernel(x_ptr, s_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    dt = c_ptr.dtype.element_ty
    s = tl.load(s_ptr).to(dt)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(dt)
    tl.store(c_ptr + offs, x * s, mask=mask)


@triton.jit
def _mul_scalar_strided_kernel(
    x_ptr,
    s_ptr,
    c_ptr,
    n,
    d0,
    h0,
    t0,
    d1,
    h1,
    t1,
    d2,
    h2,
    t2,
    d3,
    h3,
    t3,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    o64 = offs.to(tl.int64)
    off = o64 * 0
    if NDIM >= 1:
        off += ((o64 // d0) % h0) * t0
    if NDIM >= 2:
        off += ((o64 // d1) % h1) * t1
    if NDIM >= 3:
        off += ((o64 // d2) % h2) * t2
    if NDIM >= 4:
        off += ((o64 // d3) % h3) * t3
    dt = c_ptr.dtype.element_ty
    s = tl.load(s_ptr).to(dt)
    x = tl.load(x_ptr + off, mask=mask, other=0.0).to(dt)
    tl.store(c_ptr + offs, x * s, mask=mask)


def _row_layout(t, k):
    """Describe the leading-dimension addressing of one operand.

    Returns ``(row_stride, lead_rank, d1, t1, d2, t2, d3, t3)`` where the
    ``(d, t)`` pairs are the leading shapes/strides in reverse order.
    ``lead_rank == 0`` means one row is addressed with the single stride.
    """
    strides = t.stride()
    nlead = t.dim() - 1
    if nlead <= 1:
        return (strides[0] if nlead == 1 else 0, 0, 0, 0, 0, 0, 0, 0)
    if k:
        try:
            t.view(-1, k)
        except Exception:  # noqa: BLE001 - the view is a pure metadata probe
            pass
        else:
            return (strides[-2], 0, 0, 0, 0, 0, 0, 0)
    shape = t.shape
    if nlead > 3:
        raise NotImplementedError(
            "inner: non-contiguous operands with more than 4 dimensions are "
            "not supported"
        )
    vals = [nlead]
    for i in range(nlead - 1, -1, -1):
        vals.extend((shape[i], strides[i]))
    while len(vals) < 7:
        vals.extend((1, 0))
    return tuple(vals)


def _gemm_cfg(M, N, K, dtype):
    """Tile/warp/stage heuristic for the small (launch-bound) products."""
    work = M * N * K
    if dtype is torch.float32:
        if work >= 2_000_000_000:
            return 128, 64, 32, 8, 8, 2
        return 64, 64, 32, 8, 4, 3
    if work >= 2_000_000_000:
        return 128, 128, 64, 8, 8, 2
    return 64, 64, 64, 8, 4, 3


# Bounded search space for the large, tile-aligned products. The smallest entry
# is the shallow fp32 pipeline, the 128x128 entries deepen the K-tile or the
# stage count (more operand prefetch at the same occupancy), and the last two
# change the warp count that maps waves onto the same tile.
_BIG_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
        num_warps=8,
        num_stages=2,
    ),
]

_inner_gemm_big = triton.autotune(configs=_BIG_CONFIGS, key=["M", "N", "K", "DT"])(
    _inner_gemm_kernel
)


def inner(self, other):
    logger.debug("GEMS_HYGON INNER")
    a_dim = self.dim()
    b_dim = other.dim()

    if a_dim == 0 or b_dim == 0:
        # torch.inner degenerates to a broadcast multiply when either operand
        # has no contraction axis.
        if a_dim == 0:
            t, s = other, self
        else:
            t, s = self, other
        out = torch.empty(
            t.shape, dtype=torch.promote_types(t.dtype, s.dtype), device=t.device
        )
        n = out.numel()
        if n == 0:
            return out
        if t.is_contiguous():
            _mul_scalar_flat_kernel[(triton.cdiv(n, 1024),)](
                t, s, out, n, BLOCK=1024, num_warps=4
            )
            return out
        nd = t.dim()
        if nd > 4:
            raise NotImplementedError(
                "inner: unsupported rank for the 0-dim broadcast path"
            )
        shp = t.shape
        strd = t.stride()
        dens = [1] * nd
        den = 1
        for i in range(nd - 1, -1, -1):
            dens[i] = den
            den *= shp[i]
        args = []
        for i in range(4):
            if i < nd:
                args.extend((dens[i], shp[i], strd[i]))
            else:
                args.extend((1, 1, 0))
        _mul_scalar_strided_kernel[(triton.cdiv(n, 1024),)](
            t, s, out, n, *args, NDIM=nd, BLOCK=1024, num_warps=4
        )
        return out

    a_shape = self.shape
    b_shape = other.shape
    if self.dtype != other.dtype:
        raise RuntimeError("both inputs should have same dtype")
    if a_shape[-1] != b_shape[-1]:
        raise RuntimeError(
            "inner() the last dimension must match on both input tensors but "
            f"got shapes {list(a_shape)} and {list(b_shape)}"
        )

    k = a_shape[-1]
    M = 1
    for d in a_shape[:-1]:
        M *= d
    N = 1
    for d in b_shape[:-1]:
        N *= d

    out = torch.empty(a_shape[:-1] + b_shape[:-1], dtype=self.dtype, device=self.device)
    if M == 0 or N == 0:
        return out

    a_rs, a_lr, *a_ex = _row_layout(self, k)
    a_ks = self.stride(-1)
    b_rs, b_lr, *b_ex = _row_layout(other, k)
    b_ks = other.stride(-1)

    if self.dtype not in _DOT_DTYPES:
        _inner_elem_kernel[(M * N,)](
            self,
            other,
            out,
            M,
            N,
            k,
            a_rs,
            a_ks,
            b_rs,
            b_ks,
            *a_ex,
            *b_ex,
            BLOCK_K=128,
            A_LR=a_lr,
            B_LR=b_lr,
            num_warps=4,
        )
        return out

    if N == 1 or M == 1:
        # One operand contributes a single row: reduce over K without a tile.
        if N == 1:
            p, q = self, other
            p_rs, p_lr, p_ex = a_rs, a_lr, a_ex
            p_ks, q_ks = a_ks, b_ks
            R = M
        else:
            p, q = other, self
            p_rs, p_lr, p_ex = b_rs, b_lr, b_ex
            p_ks, q_ks = b_ks, a_ks
            R = N
        block_r = min(16, triton.next_power_of_2(R))
        if R == 1:
            block_k = min(16384, triton.next_power_of_2(max(k, 16)))
            warps = 16 if block_k >= 8192 else (8 if block_k >= 2048 else 4)
            _inner_dot_kernel[(1,)](
                p,
                q,
                out,
                k,
                p_ks,
                q_ks,
                BLOCK_K=block_k,
                MASK_K=bool(k % block_k),
                num_warps=warps,
            )
            return out
        block_k = min(2048, triton.next_power_of_2(max(k, 16)))
        grid = (triton.cdiv(R, block_r),)
        _inner_reduce_kernel[grid](
            p,
            q,
            out,
            R,
            k,
            p_rs,
            p_ks,
            q_ks,
            *p_ex,
            BLOCK_R=block_r,
            BLOCK_K=block_k,
            P_LR=p_lr,
            MASK_R=bool(R % block_r),
            MASK_K=bool(k % block_k),
            num_warps=4,
        )
        return out

    work = M * N * k
    if work >= 2_000_000_000 and M % 128 == 0 and N % 128 == 0 and k % 64 == 0:
        # Large, tile-aligned product: let a bounded configuration search pick
        # the K-tile depth / stage count / warp count for this dtype and shape.
        dt_code = (
            0
            if self.dtype is torch.float16
            else (1 if self.dtype is torch.bfloat16 else 2)
        )

        def _big_grid(meta):
            return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

        _inner_gemm_big[_big_grid](
            self,
            other,
            out,
            M,
            N,
            k,
            dt_code,
            a_rs,
            a_ks,
            b_rs,
            b_ks,
            *a_ex,
            *b_ex,
            A_LR=a_lr,
            B_LR=b_lr,
            HAS_MASK=False,
            IS_FP32=self.dtype is torch.float32,
        )
        return out

    bm, bn, bk, gm, warps, stages = _gemm_cfg(M, N, k, self.dtype)
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    has_mask = bool(M % bm or N % bn or k % bk)
    _inner_gemm_kernel[grid](
        self,
        other,
        out,
        M,
        N,
        k,
        3,
        a_rs,
        a_ks,
        b_rs,
        b_ks,
        *a_ex,
        *b_ex,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        GROUP_M=gm,
        A_LR=a_lr,
        B_LR=b_lr,
        HAS_MASK=has_mask,
        IS_FP32=self.dtype is torch.float32,
        num_warps=warps,
        num_stages=stages,
    )
    return out
