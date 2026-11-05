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
def _lc_row_kernel(
    A_ptr,
    U_ptr,
    N,
    batch_stride,
    row_stride,
    col_stride,
    i,
    CBLK: tl.constexpr,
):
    """Upper-factor Cholesky, row ``i`` of every matrix; one launch per row.

    Lanes = columns (full width CBLK = next_pow2(n) <= 32, mask js < N is
    constant per launch); the k-loop reads only rows produced by *previous
    launches*, so the write -> read dependency is ordered by the kernel
    boundary. A single-launch row-serial version is not safe on this
    backend: reading back the vector row just stored inside the loop is
    unordered and produced process-dependent garbage (observed 1e-1..1e0
    max errors for n = 32 on some processes). Output buffer is
    zero-initialized; upper values are written directly and the lower
    triangle stays zero.
    """
    b = tl.program_id(0)
    base = b * batch_stride
    js = tl.arange(0, CBLK)
    jmask = js < N
    dtype = A_ptr.dtype.element_ty
    acc = tl.zeros((CBLK,), dtype=dtype)
    sq = tl.zeros((), dtype=dtype)
    for k in range(i):
        fk = tl.load(U_ptr + base + k * row_stride + i * col_stride)
        pk = tl.load(
            U_ptr + base + k * row_stride + js * col_stride,
            mask=jmask,
            other=0.0,
        )
        acc += fk * pk
        sq += fk * fk
    a_row = tl.load(
        A_ptr + base + i * row_stride + js * col_stride,
        mask=jmask,
        other=0.0,
    )
    d = tl.sqrt(tl.load(A_ptr + base + i * row_stride + i * col_stride) - sq)
    inv = 1.0 / d
    val = tl.where(js == i, d, tl.where(js > i, (a_row - acc) * inv, 0.0))
    tl.store(
        U_ptr + base + i * row_stride + js * col_stride,
        val,
        mask=jmask,
    )


@triton.jit
def _lc_prep_kernel(W_ptr, A_ptr, L_ptr, LT_ptr, N, r: tl.constexpr):
    """W[:,r] = A[:,rb] - sum_{k<r} L[:,kb] (L[rb,kb])^T  (32x32 dot acc)."""
    b = tl.program_id(0)
    rows = tl.arange(0, 32)
    cols = tl.arange(0, 32)
    acc = -tl.load(
        A_ptr + b * N * N + r * 32 * N + r * 32 + rows[:, None] * N + cols[None, :]
    )
    for k in range(r):
        l1 = tl.load(
            L_ptr + b * N * N + r * 32 * N + k * 32 + rows[:, None] * N + cols[None, :]
        )
        l2 = tl.load(
            LT_ptr + b * N * N + k * 32 * N + r * 32 + rows[:, None] * N + cols[None, :]
        )
        acc = tl.dot(l1, l2, acc, input_precision="ieee")
    tl.store(
        W_ptr + b * 32768 + r * 1024 + rows[:, None] * 32 + cols[None, :],
        -acc,
    )


@triton.jit
def _lc_chol_kernel(W_ptr, L_ptr, N, r: tl.constexpr):
    """Serial 32x32 Cholesky of W[:,r] into the diagonal block L[:,r]."""
    b = tl.program_id(0)
    w0 = b * 32768 + r * 1024
    l0 = b * N * N + r * 32 * N + r * 32
    dtype = W_ptr.dtype.element_ty
    for i in range(32):
        for jj in range(i):
            s2 = tl.zeros((), dtype=dtype)
            for k in range(jj):
                s2 += tl.load(L_ptr + l0 + i * N + k) * tl.load(L_ptr + l0 + jj * N + k)
            val = (tl.load(W_ptr + w0 + i * 32 + jj) - s2) / tl.load(
                L_ptr + l0 + jj * N + jj
            )
            tl.store(L_ptr + l0 + i * N + jj, val)
        s4 = tl.zeros((), dtype=dtype)
        for k in range(i):
            v = tl.load(L_ptr + l0 + i * N + k)
            s4 += v * v
        d = tl.sqrt(tl.load(W_ptr + w0 + i * 32 + i) - s4)
        tl.store(L_ptr + l0 + i * N + i, d)


@triton.jit
def _lc_inv_kernel(L_ptr, X_ptr, N, i, r: tl.constexpr):
    """Row ``i`` of X[:,r] = (L[:,r])^{-1} (lower); one launch per row.

    Lanes = columns. Row ``i`` needs rows ``k < i``; reading them back
    inside a single program is unordered on this backend (the earlier
    vector form produced max errors from 3e-7 to 1e0 across repeated
    launches/processes and made the whole n > 32 path emit NaN). One launch
    per row makes every dependency cross a kernel boundary, which is
    ordered, while keeping the 32-lane vector work.
    """
    b = tl.program_id(0)
    l0 = b * N * N + r * 32 * N + r * 32
    x0 = b * 32768 + r * 1024
    J = tl.arange(0, 32)
    acc = tl.where(J == i, 1.0, 0.0)
    for k in range(i):
        acc -= tl.load(L_ptr + l0 + i * N + k) * tl.load(X_ptr + x0 + k * 32 + J)
    d = tl.load(L_ptr + l0 + i * N + i)
    tl.store(X_ptr + x0 + i * 32 + J, acc / d)


@triton.jit
def _lc_off_kernel(
    A_ptr,
    L_ptr,
    LT_ptr,
    XT_ptr,
    N,
    m: tl.constexpr,
    r: tl.constexpr,
):
    """L[:,m][:,r] = (A[m,r] - sum_{k<r} L[m,k] (L[r,k])^T) @ (L_rr^{-1})^T."""
    b = tl.program_id(0)
    rows = tl.arange(0, 32)
    cols = tl.arange(0, 32)
    acc = -tl.load(
        A_ptr + b * N * N + m * 32 * N + r * 32 + rows[:, None] * N + cols[None, :]
    )
    for k in range(r):
        l1 = tl.load(
            L_ptr + b * N * N + m * 32 * N + k * 32 + rows[:, None] * N + cols[None, :]
        )
        l2 = tl.load(
            LT_ptr + b * N * N + k * 32 * N + r * 32 + rows[:, None] * N + cols[None, :]
        )
        acc = tl.dot(l1, l2, acc, input_precision="ieee")
    xt = tl.load(XT_ptr + b * 32768 + r * 1024 + rows[:, None] * 32 + cols[None, :])
    out = tl.dot(-acc, xt, input_precision="ieee")
    tl.store(
        L_ptr + b * N * N + m * 32 * N + r * 32 + rows[:, None] * N + cols[None, :], out
    )


def _chol_blocked(work, batch, n):
    """Blocked lower Cholesky of a (batch, n, n) work matrix (n % 32 == 0)."""
    nb = n // 32
    L = torch.zeros_like(work)
    W = torch.zeros((batch, 32, 32, 32), dtype=work.dtype, device=work.device)
    X = torch.zeros_like(W)
    XT = torch.zeros_like(W)
    LT = torch.zeros_like(L)
    for r in range(nb):
        _lc_prep_kernel[(batch,)](W, work, L, LT, n, r=r, num_warps=4, num_stages=1)
        _lc_chol_kernel[(batch,)](W, L, n, r=r, num_warps=1, num_stages=1)
        for i in range(32):
            _lc_inv_kernel[(batch,)](L, X, n, i, r=r, num_warps=1, num_stages=1)
        XT[:, r, :, :] = X[:, r, :, :].transpose(-2, -1).contiguous()
        for m in range(r + 1, nb):
            _lc_off_kernel[(batch,)](
                work, L, LT, XT, n, m=m, r=r, num_warps=4, num_stages=1
            )
        LT = L.transpose(-2, -1).contiguous()
    return L


def linalg_cholesky(A, upper=False):
    """Cholesky decomposition of a symmetric positive-definite matrix."""
    logger.debug("GEMS_KUNLUNXIN LINALG_CHOLESKY")
    assert A.dtype in (
        torch.float32,
        torch.float64,
    ), "linalg_cholesky only supports float32 and float64"

    if A.numel() == 0:
        return A

    shape = A.shape
    if len(shape) < 2:
        raise ValueError("A must be at least 2D")
    n = shape[-1]
    if n != shape[-2]:
        raise ValueError("A must be a square matrix")

    batch = 1
    for d in shape[:-2]:
        batch *= d

    A64 = A.reshape(batch, n, n)
    if not A64.is_contiguous():
        A64 = A64.contiguous()

    if n <= 32:
        U = torch.zeros((batch, n, n), dtype=A.dtype, device=A.device)
        cblk = triton.next_power_of_2(n)
        for i in range(n):
            _lc_row_kernel[(batch,)](
                A64,
                U,
                n,
                A64.stride(0),
                A64.stride(1),
                A64.stride(2),
                i,
                cblk,
                num_warps=1,
                num_stages=1,
            )
        if upper:
            return U.reshape(shape)
        return U.transpose(-2, -1).reshape(shape)

    nn = ((n + 31) // 32) * 32
    if nn != n:
        work = torch.zeros((batch, nn, nn), dtype=A64.dtype, device=A64.device)
        work[:, :n, :n] = A64
    else:
        work = A64
    L = _chol_blocked(work, batch, nn)
    L = L[:, :n, :n]
    if upper:
        return L.transpose(-2, -1).reshape(shape)
    return L.reshape(shape)
