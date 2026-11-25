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

"""Householder product (orgqr) implemented with a single Triton kernel.

Q = H(0) @ H(1) @ ... @ H(k-1),  H(i) = I - tau[i] * v_i v_i^T
with v_i[j] = 0 for j < i, v_i[i] = 1, v_i[j] = A[j, i] for j > i.

The kernel keeps the full output column panel X (BLOCK_M x BLOCK_N) in
registers: X starts as the identity block E and is updated in place by every
reflector, sweeping i from k-1 down to 0.  Only the strict lower triangle of A
is read (the diagonal of A is R and never used).  Every column panel is written
to `out` exactly once.

Each reflector step costs a cross-thread reduction (w = v^T X), two broadcast
layout conversions and a strided gather of A[:, i], all on the loop-carried
chain, and only a handful of CTAs are resident to hide it.  A bounded autotune
over column-panel width, CTA width and compiler loop-unroll factor explores
narrow single-warp panels (conversions become intra-warp) and unrolled steps
(the scheduler can then overlap the next gather with the current update).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.linalg_householder_product import (
    linalg_householder_product as default_linalg_householder_product,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64, so only fp32 is specialized.
_SUPPORTED_DTYPES = {torch.float32}


def _make_configs():
    # (BLOCK_N, num_warps, loop-unroll factor, num_stages-or-None)
    pairs = [
        (1, 1, 1, None),
        (2, 1, 1, None),
        (4, 1, 1, None),
        (8, 1, 1, None),
        (8, 2, 1, None),
        (16, 2, 1, None),
        (16, 4, 1, None),
        (32, 4, 1, None),
        (4, 1, 4, None),
        (8, 1, 4, None),
        (16, 2, 4, None),
        (32, 4, 4, None),
        (1, 1, 4, None),
        (4, 2, 1, None),
        (8, 2, 4, None),
        (1, 1, 8, None),
        (2, 1, 8, None),
        (4, 1, 8, None),
        (4, 1, 1, 2),
        (8, 1, 1, 2),
    ]
    return [
        triton.Config({"BLOCK_N": bn, "UNROLL": u}, num_warps=nw, num_stages=ns)
        for bn, nw, u, ns in pairs
    ]


@triton.autotune(
    configs=_make_configs(),
    key=["m", "n", "k", "stride_am", "stride_an"],
)
@triton.jit
def _householder_product_kernel(
    A,
    tau,
    Q,
    m,
    n,
    k,
    stride_ab,
    stride_am,
    stride_an,
    stride_qb,
    stride_qm,
    stride_qn,
    stride_tb,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    UNROLL: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_b = tl.program_id(1)

    c0 = pid_c * BLOCK_N
    rows = tl.arange(0, BLOCK_M)
    cols = c0 + tl.arange(0, BLOCK_N)

    row_mask = rows < m
    col_mask = cols < n
    mask2d = row_mask[:, None] & col_mask[None, :]

    a_base = A + pid_b.to(tl.int64) * stride_ab
    tau_base = tau + pid_b.to(tl.int64) * stride_tb
    q_ptrs = (
        Q
        + pid_b.to(tl.int64) * stride_qb
        + rows[:, None] * stride_qm
        + cols[None, :] * stride_qn
    )

    # X = E (m x n identity block)
    ident = rows[:, None] == cols[None, :]
    X = tl.where(ident, 1.0, 0.0).to(A.dtype.element_ty)

    # Columns >= c0 are untouched by reflectors i > c0 + BLOCK_N - 1, so the
    # loop may start there (exact identity, not an approximation).
    i_max = tl.minimum(k - 1, c0 + BLOCK_N - 1)
    for t in tl.range(0, i_max + 1, loop_unroll_factor=UNROLL):
        i = i_max - t
        tau_i = tl.load(tau_base + i)
        a_col = tl.load(
            a_base + rows * stride_am + i * stride_an, mask=row_mask, other=0.0
        )
        v = tl.where(rows == i, 1.0, tl.where(rows > i, a_col, 0.0)).to(
            A.dtype.element_ty
        )
        w = tl.sum(v[:, None] * X, axis=0)
        X = X - tau_i * (v[:, None] * w[None, :])

    tl.store(q_ptrs, X, mask=mask2d)


def _use_triton_kernel(A, tau):
    """Whether the pair can be handled by the MUSA kernel."""
    if not isinstance(A, torch.Tensor) or not isinstance(tau, torch.Tensor):
        return False
    if A.device.type != "musa" or tau.device.type != "musa":
        return False
    if A.dtype not in _SUPPORTED_DTYPES or tau.dtype != A.dtype:
        return False
    if A.is_complex() or A.dim() < 2 or A.numel() == 0:
        return False
    return True


def linalg_householder_product(A, tau):
    logger.debug("GEMS_MTHREADS LINALG_HOUSEHOLDER_PRODUCT")

    if not _use_triton_kernel(A, tau):
        return default_linalg_householder_product(A, tau)

    shape = A.shape
    m = shape[-2]
    n = shape[-1]
    k = tau.shape[-1]

    batch = 1
    for d in shape[:-2]:
        batch *= d

    if A.is_contiguous():
        A_work = A.reshape(batch, m, n)
    else:
        A_work = A.contiguous().reshape(batch, m, n)
    if tau.is_contiguous():
        tau_work = tau.reshape(batch, k)
    else:
        tau_work = tau.contiguous().reshape(batch, k)

    Q = torch.empty((batch, m, n), dtype=A.dtype, device=A.device)

    if batch == 0 or m == 0 or n == 0:
        return Q.reshape(shape)

    block_m = triton.next_power_of_2(m)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_N"]), batch)  # noqa: E731

    with torch_device_fn.device(A.device):
        _householder_product_kernel[grid](
            A_work,
            tau_work,
            Q,
            m,
            n,
            k,
            A_work.stride(0),
            A_work.stride(1),
            A_work.stride(2),
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            tau_work.stride(0),
            BLOCK_M=block_m,
        )

    return Q.reshape(shape)
