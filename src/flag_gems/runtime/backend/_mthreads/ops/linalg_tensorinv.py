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

"""Triton implementation of torch.linalg.tensorinv.

The operator computes the multiplicative inverse of the tensordot of the
leading ``ind`` axes against the remaining axes:

    M = A.reshape(prod(A.shape[:ind]), prod(A.shape[ind:]))   # (N, N)
    X = M^{-1}
    out = X.reshape(A.shape[ind:] + A.shape[:ind])

There is exactly one dense N x N inverse per call (no batch axis).  For the
small N produced by the tensorinv test shapes the whole augmented tile
[M | I] fits in registers, so a single program performs Gauss-Jordan
elimination with partial pivoting entirely in registers and writes M^{-1}
straight into the (rest, first) flat output order.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.linalg_tensorinv import linalg_tensorinv as default_linalg_tensorinv
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64, so only fp32/fp16 are
# specialized; every other dtype falls back to the generic implementation.
_SUPPORTED_DTYPES = {torch.float16, torch.float32}

# N <= _REGISTER_MAX_N keeps the whole (BLOCK_N, 2 * BLOCK_N) augmented tile in
# registers.  Larger N falls back to a blocked global-memory Gauss-Jordan.
_REGISTER_MAX_N = 64
_LARGE_BLOCK = 64


def _tensorinv_check(A, ind):
    """Validate ``ind`` / shape compatibility, mirroring torch's errors."""
    if ind <= 0:
        raise RuntimeError(
            "linalg.tensorinv: Expected a strictly positive integer for "
            f"'ind', but got {ind}"
        )
    if A.dim() < 2:
        raise RuntimeError(
            f"linalg.tensorinv: Expected input to be at least 2D, got {A.dim()}D"
        )
    if ind > A.dim():
        raise RuntimeError(
            "linalg.tensorinv: Expected 0 <= ind <= input.dim(), "
            f"got ind={ind} for input.dim()={A.dim()}"
        )
    m = 1
    for i in range(ind):
        m *= A.shape[i]
    n = 1
    for i in range(ind, A.dim()):
        n *= A.shape[i]
    if m != n:
        raise RuntimeError(
            "linalg.tensorinv: Expected self to satisfy the requirement "
            "prod(self.shape[ind:]) == prod(self.shape[:ind]), "
            f"but got {n} != {m}"
        )
    return m


@triton.jit
def _tensorinv_register_kernel(A, OUT, N, BLOCK_N: tl.constexpr, ACC: tl.constexpr):
    """Fused Gauss-Jordan on the augmented tile [M | I] held in registers.

    The tile is (BLOCK_N, 2 * BLOCK_N): the left half holds M (masked to N x N),
    the right half holds the identity.  After the loop the right half is M^{-1}
    in (row = rest axes, col = first axes) order, i.e. exactly row-major flat
    order of the requested output, so the store needs no transpose.
    """
    rows = tl.arange(0, BLOCK_N)
    c2 = tl.arange(0, 2 * BLOCK_N)
    rmask = rows < N
    a_mask = (rows[:, None] < N) & (c2[None, :] < N)
    eye_mask = (c2[None, :] >= BLOCK_N) & (c2[None, :] - BLOCK_N == rows[:, None])

    one = tl.full((), 1.0, ACC)
    zero = tl.full((), 0.0, ACC)

    a_val = tl.load(A + rows[:, None] * N + c2[None, :], mask=a_mask, other=0.0).to(ACC)
    aug = tl.where(a_mask, a_val, tl.where(eye_mask & rmask[:, None], one, zero))

    for k in range(0, N):
        # --- column k (whole tile reduction; only the left half matches k) ---
        colk = tl.sum(tl.where(c2[None, :] == k, aug, zero), axis=1)

        # --- partial pivoting over rows k..N-1 ---
        abs_col = tl.where((rows >= k) & rmask, tl.abs(colk), -1.0)
        pmax = tl.max(abs_col, axis=0)
        piv = tl.min(tl.where(abs_col == pmax, rows, BLOCK_N), axis=0)

        # --- pivot row / current row exchange (whole augmented row) ---
        rowk = tl.sum(tl.where(rows[:, None] == k, aug, zero), axis=0)
        rowp = tl.sum(tl.where(rows[:, None] == piv, aug, zero), axis=0)
        pk = tl.sum(tl.where(rows == piv, colk, zero), axis=0)
        p0 = tl.sum(tl.where(rows == k, colk, zero), axis=0)
        rown = rowp / pk

        swapped = tl.where(
            rows[:, None] == k,
            rowp[None, :],
            tl.where(rows[:, None] == piv, rowk[None, :], aug),
        )
        # normalise the pivot row (row k after the exchange)
        aug = tl.where(rows[:, None] == k, rown[None, :], swapped)

        # --- eliminate column k from every other row via a rank-1 update ---
        fact = tl.where(rows == k, zero, tl.where(rows == piv, p0, colk))
        aug = aug - fact[:, None] * rown[None, :]

    oc = tl.where(c2[None, :] >= BLOCK_N, c2[None, :] - BLOCK_N, 0)
    s_mask = (rows[:, None] < N) & (c2[None, :] >= BLOCK_N) & (oc < N)
    tl.store(OUT + rows[:, None] * N + oc, aug.to(OUT.dtype.element_ty), mask=s_mask)


@triton.jit
def _tensorinv_init_kernel(A, AW, BW, N, BLOCK: tl.constexpr, ACC: tl.constexpr):
    """Scatter M and the identity into two contiguous fp scratch tiles."""
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rmask = rows < N
    one = tl.full((), 1.0, ACC)
    zero = tl.full((), 0.0, ACC)
    for j0 in range(0, N, BLOCK):
        cols = j0 + tl.arange(0, BLOCK)
        m = rmask[:, None] & (cols[None, :] < N)
        av = tl.load(A + rows[:, None] * N + cols[None, :], mask=m, other=0.0).to(ACC)
        tl.store(AW + rows[:, None] * N + cols[None, :], av, mask=m)
        eye = tl.where(cols[None, :] == rows[:, None], one, zero)
        tl.store(BW + rows[:, None] * N + cols[None, :], eye, mask=m)


@triton.jit
def _tensorinv_blocked_kernel(AW, BW, N, BLOCK: tl.constexpr, ACC: tl.constexpr):
    """Blocked Gauss-Jordan over global scratch for N > _REGISTER_MAX_N."""
    zero = tl.full((), 0.0, ACC)
    for k in range(0, N):
        # --- pivot search over rows k..N-1 ---
        best_val = tl.full((), -1.0, tl.float32)
        best_row = tl.full((), 0, tl.int32)
        for i0 in range(k, N, BLOCK):
            rows = i0 + tl.arange(0, BLOCK)
            rm = rows < N
            col = tl.load(AW + rows * N + k, mask=rm, other=0.0).to(tl.float32)
            av = tl.where(rm & (rows >= k), tl.abs(col), -1.0)
            tv = tl.max(av, axis=0)
            tr = tl.min(tl.where(av == tv, rows, N), axis=0)
            take = tv > best_val
            best_row = tl.where(take, tr, best_row)
            best_val = tl.where(take, tv, best_val)

        # --- exchange rows k and best_row in both tiles ---
        for j0 in range(0, N, BLOCK):
            cols = j0 + tl.arange(0, BLOCK)
            cm = cols < N
            rk_a = tl.load(AW + k * N + cols, mask=cm, other=0.0)
            rp_a = tl.load(AW + best_row * N + cols, mask=cm, other=0.0)
            tl.store(AW + k * N + cols, rp_a, mask=cm)
            tl.store(AW + best_row * N + cols, rk_a, mask=cm)
            rk_b = tl.load(BW + k * N + cols, mask=cm, other=0.0)
            rp_b = tl.load(BW + best_row * N + cols, mask=cm, other=0.0)
            tl.store(BW + k * N + cols, rp_b, mask=cm)
            tl.store(BW + best_row * N + cols, rk_b, mask=cm)

        tl.debug_barrier()

        # --- normalise the pivot row ---
        pivot = tl.load(AW + k * N + k).to(ACC)
        for j0 in range(0, N, BLOCK):
            cols = j0 + tl.arange(0, BLOCK)
            cm = cols < N
            a_row = tl.load(AW + k * N + cols, mask=cm, other=0.0)
            b_row = tl.load(BW + k * N + cols, mask=cm, other=0.0)
            tl.store(AW + k * N + cols, a_row / pivot, mask=cm)
            tl.store(BW + k * N + cols, b_row / pivot, mask=cm)

        tl.debug_barrier()

        # --- eliminate column k from all other rows ---
        for i0 in range(0, N, BLOCK):
            rows = i0 + tl.arange(0, BLOCK)
            rm = rows < N
            factor = tl.load(AW + rows * N + k, mask=rm, other=0.0)
            factor = tl.where(rows != k, factor, zero)
            for j0 in range(0, N, BLOCK):
                cols = j0 + tl.arange(0, BLOCK)
                cm = cols < N
                tm = rm[:, None] & cm[None, :]
                ua = tl.load(AW + k * N + cols, mask=cm, other=0.0)
                ub = tl.load(BW + k * N + cols, mask=cm, other=0.0)
                ta = tl.load(AW + rows[:, None] * N + cols[None, :], mask=tm, other=0.0)
                tb = tl.load(BW + rows[:, None] * N + cols[None, :], mask=tm, other=0.0)
                tl.store(
                    AW + rows[:, None] * N + cols[None, :],
                    ta - factor[:, None] * ua[None, :],
                    mask=tm,
                )
                tl.store(
                    BW + rows[:, None] * N + cols[None, :],
                    tb - factor[:, None] * ub[None, :],
                    mask=tm,
                )

        tl.debug_barrier()


def _register_num_warps(block_n):
    if block_n <= 8:
        return 1
    if block_n <= 32:
        return 2
    return 4


def _use_triton_kernel(A, ind):
    """Whether ``A`` / ``ind`` can be handled by the MUSA kernels."""
    if not isinstance(A, torch.Tensor):
        return False
    if A.device.type != "musa" or A.dtype not in _SUPPORTED_DTYPES:
        return False
    if A.is_complex() or A.numel() == 0:
        return False
    if ind <= 0 or A.dim() < 2 or ind > A.dim():
        # Torch raises for these; let the generic implementation own the error.
        return False
    m = 1
    for i in range(ind):
        m *= A.shape[i]
    n = 1
    for i in range(ind, A.dim()):
        n *= A.shape[i]
    return m == n


def linalg_tensorinv(A, ind=2, *, out=None):
    logger.debug("GEMS_MTHREADS LINALG_TENSORINV")

    if not _use_triton_kernel(A, ind):
        return default_linalg_tensorinv(A, ind, out=out)

    n = _tensorinv_check(A, ind)
    out_shape = A.shape[ind:] + A.shape[:ind]

    A_c = A if A.is_contiguous() else A.contiguous()
    if out is not None:
        dst = out.view(-1)
    else:
        dst = torch.empty(n * n, dtype=A.dtype, device=A.device)

    acc_dtype = torch.float32
    ACC = tl.float32

    with torch_device_fn.device(A.device):
        if n <= _REGISTER_MAX_N:
            block_n = max(2, triton.next_power_of_2(n))
            _tensorinv_register_kernel[(1,)](
                A_c,
                dst,
                n,
                BLOCK_N=block_n,
                ACC=ACC,
                num_warps=_register_num_warps(block_n),
            )
        else:
            aw = torch.empty(n * n, dtype=acc_dtype, device=A.device)
            bw = torch.empty(n * n, dtype=acc_dtype, device=A.device)
            block = _LARGE_BLOCK
            _tensorinv_init_kernel[(triton.cdiv(n, block),)](
                A_c, aw, bw, n, BLOCK=block, ACC=ACC, num_warps=4
            )
            _tensorinv_blocked_kernel[(1,)](
                aw, bw, n, BLOCK=block, ACC=ACC, num_warps=4
            )
            # copy the inverse out of the scratch tile into the flat output order
            _tensorinv_init_kernel[(triton.cdiv(n, block),)](
                bw, dst, aw, n, BLOCK=block, ACC=ACC, num_warps=4
            )

    if out is not None:
        return out
    return dst.view(out_shape)
