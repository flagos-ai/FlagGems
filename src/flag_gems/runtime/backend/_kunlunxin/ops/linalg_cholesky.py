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

"""Kunlunxin backend override for linalg_cholesky.

The generic implementation walks the whole matrix with scalar loads in
triply nested loops (O(N^3) scalar ops in a single program): ~10 ms at N=32,
~0.31 s at N=128 and ~2 s at N=256 on TritonXPU.

This override keeps the chips that TritonXPU lowers reliably (per the
cholesky family experience on this backend):

  - n <= 32: one row-serial kernel over the whole matrix with a
    constant-mask 32-lane(<=) vector (verified deterministic across many
    fresh processes for n = 2/4/8/16/32);
  - n > 32: a blocked, mask-free 32x32 pipeline: diagonal 32-blocks via a
    scalar serial kernel, Schur/off-diagonal solves via tl.dot with
    input_precision="ieee" and fully unmasked 32x32 operands, inverse
    diagonal blocks via a 32-lane serial kernel, transposed working copies
    done with native strided copies (no tl.gather: illegal on this backend).

Known backend limitation (documented): 64-wide / partial-false masked
vectors inside row-serial Cholesky loops and multi-dot pipelines at N=256
have shown process-dependent miscompilation on this backend; kernels here
stay at <=32 lanes for the serial parts and the blocked path is only
exercised by benchmark shapes (the accuracy suite covers n <= 32).

dtype: on this backend fp64 requests never reach the kernels (the vendor
silently downcasts to fp32), the kernels are dtype-generic.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _lc_small_kernel(
    A_ptr,
    U_ptr,
    N,
    batch_stride,
    row_stride,
    col_stride,
    CBLK: tl.constexpr,
):
    """Upper-factor Cholesky, one program per matrix, rows serial.

    Lanes = columns (full width CBLK = next_pow2(n) <= 32, mask js < N is
    constant per launch); the k-loop bound min(i, N) keeps the accumulated
    sums exact. Output buffer is zero-initialized; upper values are written
    directly and the lower triangle stays zero.
    """
    b = tl.program_id(0)
    base = b * batch_stride
    js = tl.arange(0, CBLK)
    jmask = js < N
    dtype = A_ptr.dtype.element_ty
    for i in range(N):
        k_lim = tl.minimum(i, N)
        acc = tl.zeros((CBLK,), dtype=dtype)
        sq = tl.zeros((), dtype=dtype)
        for k in range(0, k_lim):
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
    I = tl.arange(0, 32)
    J = tl.arange(0, 32)
    w0 = b * 32768 + r * 1024
    acc = -tl.load(A_ptr + b * N * N + r * 32 * N + r * 32 + I[:, None] * N + J[None, :])
    for k in range(r):
        l1 = tl.load(L_ptr + b * N * N + r * 32 * N + k * 32 + I[:, None] * N + J[None, :])
        l2 = tl.load(LT_ptr + b * N * N + k * 32 * N + r * 32 + I[:, None] * N + J[None, :])
        acc = tl.dot(l1, l2, acc, input_precision="ieee")
    tl.store(
        W_ptr + b * 32768 + r * 1024 + I[:, None] * 32 + J[None, :],
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
            val = (tl.load(W_ptr + w0 + i * 32 + jj) - s2) / tl.load(L_ptr + l0 + jj * N + jj)
            tl.store(L_ptr + l0 + i * N + jj, val)
        s4 = tl.zeros((), dtype=dtype)
        for k in range(i):
            v = tl.load(L_ptr + l0 + i * N + k)
            s4 += v * v
        d = tl.sqrt(tl.load(W_ptr + w0 + i * 32 + i) - s4)
        tl.store(L_ptr + l0 + i * N + i, d)


@triton.jit
def _lc_inv_kernel(L_ptr, X_ptr, N, r: tl.constexpr):
    """X[:,r] = (L[:,r])^{-1} (lower); 32-lane vectors, mask-free."""
    b = tl.program_id(0)
    l0 = b * N * N + r * 32 * N + r * 32
    x0 = b * 32768 + r * 1024
    J = tl.arange(0, 32)
    dtype = L_ptr.dtype.element_ty
    for i in range(32):
        acc = tl.where(J == i, 1.0, 0.0)
        for k in range(i):
            acc -= tl.load(L_ptr + l0 + i * N + k) * tl.load(X_ptr + x0 + k * 32 + J)
        d = tl.load(L_ptr + l0 + i * N + i)
        tl.store(X_ptr + x0 + i * 32 + J, acc / d)


@triton.jit
def _lc_off_kernel(
    A_ptr, L_ptr, LT_ptr, XT_ptr, N, m: tl.constexpr, r: tl.constexpr,
):
    """L[:,m][:,r] = (A[m,r] - sum_{k<r} L[m,k] (L[r,k])^T) @ (L_rr^{-1})^T."""
    b = tl.program_id(0)
    I = tl.arange(0, 32)
    J = tl.arange(0, 32)
    acc = -tl.load(A_ptr + b * N * N + m * 32 * N + r * 32 + I[:, None] * N + J[None, :])
    for k in range(r):
        l1 = tl.load(L_ptr + b * N * N + m * 32 * N + k * 32 + I[:, None] * N + J[None, :])
        l2 = tl.load(LT_ptr + b * N * N + k * 32 * N + r * 32 + I[:, None] * N + J[None, :])
        acc = tl.dot(l1, l2, acc, input_precision="ieee")
    xt = tl.load(XT_ptr + b * 32768 + r * 1024 + I[:, None] * 32 + J[None, :])
    out = tl.dot(-acc, xt, input_precision="ieee")
    tl.store(L_ptr + b * N * N + m * 32 * N + r * 32 + I[:, None] * N + J[None, :], out)


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
        _lc_inv_kernel[(batch,)](L, X, n, r=r, num_warps=1, num_stages=1)
        XT[:, r, :, :] = X[:, r, :, :].transpose(-2, -1).contiguous()
        for m in range(r + 1, nb):
            _lc_off_kernel[(batch,)](
                work, L, LT, XT, n, m=m, r=r, num_warps=4, num_stages=1
            )
        LT = L.transpose(-2, -1).contiguous()
    return L


def linalg_cholesky(A, upper=False):
    """Cholesky decomposition of a symmetric positive-definite matrix."""
    logger.debug("KUNLUNXIN LINALG_CHOLESKY")
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
        _lc_small_kernel[(batch,)](
            A64,
            U,
            n,
            A64.stride(0),
            A64.stride(1),
            A64.stride(2),
            triton.next_power_of_2(n),
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