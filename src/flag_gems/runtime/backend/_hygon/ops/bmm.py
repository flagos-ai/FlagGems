# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Small batched reductions without padding every batch to a GEMM tile."""

import torch
import triton
import triton.language as tl

from flag_gems.ops.bmm import bmm as _generic_bmm
from flag_gems.ops.bmm import bmm_out as _generic_bmm_out
from flag_gems.runtime import torch_device_fn


@triton.jit
def _single_row_bmm(
    A,
    B,
    C,
    N: tl.constexpr,
    K: tl.constexpr,
    AS: tl.constexpr,
    AK: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    NC: tl.constexpr,
    KW: tl.constexpr,
):
    batch = tl.program_id(0)
    n = tl.arange(0, NC)
    k = tl.arange(0, KW)
    # Reuse one A vector across the columns without repeating batch div/mod.
    a = tl.load(A + batch * AS + k * AK, k < K, other=0)
    b = tl.load(
        B + batch * BS + k[:, None] * BK + n[None, :] * BN,
        (k[:, None] < K) & (n[None, :] < N),
        other=0,
    )
    value = tl.sum(a[:, None] * b, 0)
    tl.store(C + batch * N + n, value, n < N)


@triton.jit
def _short_k_rows(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AS: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(1)
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    b0 = tl.load(B + batch * BS + n * BN, n < N, other=0)
    b1 = tl.load(B + batch * BS + BK + n * BN, n < N, other=0)
    b2 = tl.load(B + batch * BS + 2 * BK + n * BN, n < N, other=0)
    for m in tl.static_range(M):
        a0 = tl.load(A + batch * AS + m * AM)
        a1 = tl.load(A + batch * AS + m * AM + AK)
        a2 = tl.load(A + batch * AS + m * AM + 2 * AK)
        value = tl.fma(a2, b2, tl.fma(a1, b1, a0 * b0))
        tl.store(C + batch * M * N + m * N + n, value, n < N)


@triton.jit
def _short_k_matrix(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AS: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    BM: tl.constexpr,
    NC: tl.constexpr,
    KW: tl.constexpr,
):
    batch = tl.program_id(2)
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * NC + tl.arange(0, NC)
    k = tl.arange(0, KW)
    a = tl.load(
        A + batch * AS + m[:, None] * AM + k[None, :] * AK,
        (m[:, None] < M) & (k[None, :] < K),
        other=0,
    )
    b = tl.load(
        B + batch * BS + k[:, None] * BK + n[None, :] * BN,
        (k[:, None] < K) & (n[None, :] < N),
        other=0,
    )
    value = tl.dot(a, b, allow_tf32=False)
    tl.store(
        C + batch * M * N + m[:, None] * N + n[None, :],
        value,
        (m[:, None] < M) & (n[None, :] < N),
    )


@triton.jit
def _small_bmm(
    A,
    B,
    C,
    TOTAL: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AS: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    CS: tl.constexpr,
    CM: tl.constexpr,
    CN: tl.constexpr,
    BLOCK: tl.constexpr,
    REDUCE: tl.constexpr,
):
    out = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    batch = out // (M * N)
    row = out // N % M
    col = out % N
    valid = out < TOTAL
    ap = batch * AS + row * AM
    bp = batch * BS + col * BN
    if REDUCE:
        k = tl.arange(0, triton.next_power_of_2(K))
        if BK == 1:
            a = tl.load(
                A + ap[:, None] + k[None, :] * AK,
                valid[:, None] & (k[None, :] < K),
                other=0,
            )
            b = tl.load(
                B + bp[:, None] + k[None, :] * BK,
                valid[:, None] & (k[None, :] < K),
                other=0,
            )
            if A.dtype.element_ty != tl.float64:
                a, b = a.to(tl.float32), b.to(tl.float32)
            value = tl.sum(a * b, 1)
        else:
            a = tl.load(
                A + ap[None, :] + k[:, None] * AK,
                valid[None, :] & (k[:, None] < K),
                other=0,
            )
            b = tl.load(
                B + bp[None, :] + k[:, None] * BK,
                valid[None, :] & (k[:, None] < K),
                other=0,
            )
            if A.dtype.element_ty != tl.float64:
                a, b = a.to(tl.float32), b.to(tl.float32)
            value = tl.sum(a * b, 0)
    else:
        dtype: tl.constexpr = (
            tl.float64 if A.dtype.element_ty == tl.float64 else tl.float32
        )
        value = tl.full((BLOCK,), 0, dtype)
        for k in tl.static_range(K):
            a = tl.load(A + ap + k * AK, valid, other=0).to(dtype)
            b = tl.load(B + bp + k * BK, valid, other=0).to(dtype)
            value = tl.fma(a, b, value)
    tl.store(C + batch * CS + row * CM + col * CN, value, valid)


def _use_reduction(a, b):
    m, k = a.shape[1:]
    n = b.shape[2]
    return (
        a.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        and k <= 128
        and (
            m <= 9
            or n == 1
            or (k <= 16 and (a.dtype == torch.float64 or m * n <= 4096))
        )
    )


def _use_short_matrix(a, b):
    m, k = a.shape[1:]
    n = b.shape[2]
    return a.dtype == torch.float32 and (
        (
            16 <= m <= 256
            and ((k == 16 and 16 <= n <= 256) or (16 < k <= 128 and 16 <= n <= 32))
        )
        or (1 <= k <= 8 and 64 <= m <= 512 and 64 <= n <= 512)
    )


def _launch(a, b, out):
    batch, m, k = a.shape
    n = b.shape[2]
    total = batch * m * n
    if total:
        if (
            a.dtype == torch.float32
            and m == 1
            and 32 <= k <= 64
            and 32 <= n <= 64
            and b.stride(2) == 1
            and out.is_contiguous()
        ):
            with torch_device_fn.device(a.device):
                _single_row_bmm[(batch,)](
                    a,
                    b,
                    out,
                    n,
                    k,
                    a.stride(0),
                    a.stride(2),
                    *b.stride(),
                    64,
                    triton.next_power_of_2(k),
                    num_warps=2,
                    num_stages=1,
                )
            return out
        if (
            a.dtype == torch.float64
            and k == 3
            and m <= 32
            and n >= 128
            and out.is_contiguous()
        ):
            with torch_device_fn.device(a.device):
                _short_k_rows[(triton.cdiv(n, 128), batch)](
                    a,
                    b,
                    out,
                    m,
                    n,
                    k,
                    *a.stride(),
                    *b.stride(),
                    128,
                    num_warps=1,
                    num_stages=1,
                )
            return out
        if _use_short_matrix(a, b) and out.is_contiguous():
            tile = 64 if k <= 16 else 32
            with torch_device_fn.device(a.device):
                _short_k_matrix[(triton.cdiv(m, tile), triton.cdiv(n, tile), batch)](
                    a,
                    b,
                    out,
                    m,
                    n,
                    k,
                    *a.stride(),
                    *b.stride(),
                    tile,
                    tile,
                    max(16, triton.next_power_of_2(k)),
                    num_warps=8 if k < 16 else 4,
                    num_stages=1,
                )
            return out
        reduce = k >= 16
        block = 64 if reduce else 256
        if a.dtype == torch.float64 and k <= 3 and total >= 262144:
            block = 512
        with torch_device_fn.device(a.device):
            _small_bmm[(triton.cdiv(total, block),)](
                a,
                b,
                out,
                total,
                m,
                n,
                k,
                *a.stride(),
                *b.stride(),
                *out.stride(),
                BLOCK=block,
                REDUCE=reduce,
                num_warps=4,
            )
    return out


def bmm(a, b):
    if not (_use_reduction(a, b) or _use_short_matrix(a, b)):
        return _generic_bmm(a, b)
    assert (
        a.shape[0] == b.shape[0] and a.shape[2] == b.shape[1]
    ), "Incompatible dimensions"
    out = torch.empty(
        (a.shape[0], a.shape[1], b.shape[2]), device=a.device, dtype=a.dtype
    )
    return _launch(a, b, out)


def bmm_out(a, b, out):
    if not (_use_reduction(a, b) or (_use_short_matrix(a, b) and out.is_contiguous())):
        return _generic_bmm_out(a, b, out)
    assert out.shape == (
        a.shape[0],
        a.shape[1],
        b.shape[2],
    ), "Incompatible output shape"
    assert (
        a.shape[0] == b.shape[0] and a.shape[2] == b.shape[1]
    ), "Incompatible dimensions"
    return _launch(a, b, out)
