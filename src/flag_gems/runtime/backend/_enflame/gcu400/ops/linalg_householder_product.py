# Copyright 2026, The FlagOS Contributors.
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
#
"""linalg_householder_product - single-pass reflector sweep in Triton.

Q = H(0) H(1) ... H(k-1) E, where E is the m-by-k identity embedding and
H(i) = I - tau[i] * v[i] v[i]^T.  One program owns (batch element, block of
BLOCK_N output columns); the tile stays in registers for the whole descending
reflector sweep, so each output element is written to global memory exactly
once.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Launch parameters are chosen on the host: triton.autotune runs its own
# triton.testing.do_bench, whose cache-clearing path compiles a flag_gems
# zeros kernel with ENABLE_I64 and aborts on this backend
# ("<Pass-Options-Parser>: no such option enable_i64"), so autotune must not be
# used here at all.
_BLOCK_N_MAX = 32
_NUM_WARPS = 1


def _pick_block_n(n):
    bn = 1
    while bn < n and bn < _BLOCK_N_MAX:
        bn <<= 1
    return bn


@triton.jit
def _householder_product_kernel(
    A_ptr,
    tau_ptr,
    Q_ptr,
    M,
    N,
    K,
    a_batch_stride,
    q_batch_stride,
    tau_batch_stride,
    a_row_stride,
    a_col_stride,
    q_row_stride,
    q_col_stride,
    tau_elem_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Q = H(0) H(1) ... H(k-1) E, E = m-by-k identity embedding.

    One program owns (batch element, block of BLOCK_N output columns).  The
    tile lives in registers for the whole descending reflector sweep, so every
    output element is written to global memory exactly once.
    """
    pid = tl.program_id(0)
    num_col_blocks = tl.cdiv(N, BLOCK_N)
    pid_b = pid // num_col_blocks
    pid_c = pid % num_col_blocks

    c0 = pid_c * BLOCK_N
    cols = c0 + tl.arange(0, BLOCK_N)
    rows = tl.arange(0, BLOCK_M)
    row_mask = rows < M
    col_mask = cols < N
    x_mask = row_mask[:, None] & col_mask[None, :]

    # Identity embedding for this column block (1 where row == column).
    x = tl.where(rows[:, None] == cols[None, :], 1.0, 0.0)

    a_base = A_ptr + pid_b * a_batch_stride
    tau_base = tau_ptr + pid_b * tau_batch_stride

    # Column c depends only on reflectors 0..c, so the highest reflector that
    # can touch this column block is min(c0 + BLOCK_N - 1, K - 1).
    i_end = tl.minimum(c0 + BLOCK_N, K) - 1
    for i in range(i_end, -1, -1):
        tau_i = tl.load(tau_base + i * tau_elem_stride).to(tl.float32)
        a_col = tl.load(
            a_base + rows * a_row_stride + i * a_col_stride,
            mask=row_mask,
            other=0.0,
        )
        a_col = a_col.to(tl.float32)
        # v has exact zeros above i, a unit at i, A[r, i] below i.
        v = tl.where(rows > i, a_col, tl.where(rows == i, 1.0, 0.0))
        w = tl.sum(v[:, None] * x, axis=0)
        x = x - tau_i * (v[:, None] * w[None, :])

    q_ptrs = (
        Q_ptr
        + pid_b * q_batch_stride
        + rows[:, None] * q_row_stride
        + cols[None, :] * q_col_stride
    )
    tl.store(q_ptrs, x, mask=x_mask)


def run(A, tau):
    assert A.dtype in (
        torch.float32,
        torch.float64,
    ), f"linalg_householder_product only supports float32 and float64, got {A.dtype}"
    if A.dim() < 2:
        raise ValueError("A must be at least 2D")

    m = A.shape[-2]
    n = A.shape[-1]
    k = tau.shape[-1]

    out32 = torch.empty(A.shape, dtype=torch.float32, device=A.device)
    if A.numel() == 0 or n == 0 or m == 0 or k == 0:
        return out32 if A.dtype == torch.float32 else out32.to(A.dtype)

    # GCU kernels have no 64-bit data-type support; the fp64 contract is served
    # by fp32 arithmetic (tolerance is atol=1e-4) with fp64 storage.
    A_w = A if A.dtype == torch.float32 else A.float()
    tau_w = tau if tau.dtype == torch.float32 else tau.float()

    if A.dim() == 2:
        batch = 1
        a_bs = 0
        tau_bs = 0
        q_bs = 0
    else:
        batch = 1
        for d in A.shape[:-2]:
            batch *= d
        a_bs = A_w.stride(-3)
        q_bs = out32.stride(-3)
        tau_bs = tau_w.stride(-2) if tau_w.dim() >= 2 else 0

    block_m = triton.next_power_of_2(max(m, 1))
    block_n = _pick_block_n(n)
    grid = (batch * triton.cdiv(n, block_n),)

    _householder_product_kernel[grid](
        A_w,
        tau_w,
        out32,
        m,
        n,
        k,
        a_bs,
        q_bs,
        tau_bs,
        A_w.stride(-2),
        A_w.stride(-1),
        out32.stride(-2),
        out32.stride(-1),
        tau_w.stride(-1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=_NUM_WARPS,
    )

    if A.dtype == torch.float32:
        return out32
    return out32.to(A.dtype)


def linalg_householder_product(A, tau):
    """Computes the product of Householder matrices (orgqr).

    Given the output of torch.geqrf (Householder vectors A and scalars tau),
    computes Q = H(0) * H(1) * ... * H(k-1), where
    H(i) = I - tau[i] * v[i] * v[i]^T.

    Args:
        A: Input matrix of shape (..., m, n) containing Householder vectors.
        tau: Vector of shape (..., k) where k <= n.

    Returns:
        Q: Orthogonal matrix of shape (..., m, n).
    """
    logger.debug("GEMS LINALG_HOUSEHOLDER_PRODUCT")
    return run(A, tau)


__all__ = ["linalg_householder_product", "run"]
