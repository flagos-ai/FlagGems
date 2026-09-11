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

from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# Block size for blocked right-looking Cholesky decomposition.
BLOCK = 32

# Tile sizes for parallel SYRK kernel.
TILE_M = 32
TILE_N = 32
TRSM_TILE_M = 8

# Cache for lower-triangular masks to avoid per-call CPU-to-GPU copies.
_tril_mask_cache = {}


def _get_tril_mask(n, dtype, device):
    """Return a cached lower-triangular mask of shape (n, n)."""
    key = (n, dtype, device)
    if key not in _tril_mask_cache:
        _tril_mask_cache[key] = torch.tril(
            torch.ones(n, n, device="cpu", dtype=dtype)
        ).to(device)
    return _tril_mask_cache[key]


# ---------------------------------------------------------------------------
# Sequential kernel (float64 fallback and N ≤ 32 fast path)
# ---------------------------------------------------------------------------
@triton.jit
def cholesky_sequential_kernel(A, L, N, batch_stride, stride_a, stride_l):
    """Sequential Cholesky decomposition -- one program per matrix."""
    pid = tle.program_id(0)
    a_offset = pid * batch_stride
    l_offset = pid * batch_stride

    for i in range(N):
        for j in range(i + 1):
            sum_val = tl.zeros((), dtype=A.dtype.element_ty)

            if j > 0:
                for k in range(j):
                    sum_val = sum_val + tl.load(
                        L + l_offset + i * stride_l + k
                    ) * tl.load(L + l_offset + j * stride_l + k)

            if j == i:
                a_diag = tl.load(A + a_offset + i * stride_a + i)
                tl.store(
                    L + l_offset + i * stride_l + i,
                    tl.sqrt(a_diag - sum_val),
                )
            else:
                a_val = tl.load(A + a_offset + i * stride_a + j)
                l_diag = tl.load(L + l_offset + j * stride_l + j)
                tl.store(
                    L + l_offset + i * stride_l + j,
                    (a_val - sum_val) / l_diag,
                )


# ---------------------------------------------------------------------------
# Blocked right-looking Cholesky kernels (float32 when N > BLOCK)
# ---------------------------------------------------------------------------
@triton.jit
def _cholesky_diag_kernel(L, block_start, block_size, N, stride_ln, batch_stride):
    """Factor the diagonal block L[start:start+sz, start:start+sz] in-place.

    Grid: (batch_size,)
    """
    pid = tle.program_id(0)
    l_offset = pid * batch_stride

    for i in range(block_size):
        for j in range(i + 1):
            acc = tl.zeros((), dtype=L.dtype.element_ty)
            if j > 0:
                for k in range(j):
                    acc = acc + tl.load(
                        L + l_offset + (block_start + i) * stride_ln + (block_start + k)
                    ) * tl.load(
                        L + l_offset + (block_start + j) * stride_ln + (block_start + k)
                    )

            if i == j:
                diag_val = tl.load(
                    L + l_offset + (block_start + i) * stride_ln + (block_start + i)
                )
                tl.store(
                    L + l_offset + (block_start + i) * stride_ln + (block_start + i),
                    tl.sqrt(diag_val - acc),
                )
            else:
                a_val = tl.load(
                    L + l_offset + (block_start + i) * stride_ln + (block_start + j)
                )
                l_diag = tl.load(
                    L + l_offset + (block_start + j) * stride_ln + (block_start + j)
                )
                tl.store(
                    L + l_offset + (block_start + i) * stride_ln + (block_start + j),
                    (a_val - acc) / l_diag,
                )


@triton.jit
def _trsm_kernel(
    L,
    diag_start,
    diag_size,
    panel_row_start,
    n_panel_rows,
    N,
    stride_ln,
    batch_stride,
    TILE_M: tl.constexpr,
):
    """Forward substitution: panel = panel @ inv(L_diag)^T.

    Grid: (batch_size, ceil(n_panel_rows / TILE_M))
    """
    pid_b = tle.program_id(0)
    pid_tile = tle.program_id(1)
    l_offset = pid_b * batch_stride

    for r in range(TILE_M):
        row_idx = panel_row_start + pid_tile * TILE_M + r
        if row_idx < panel_row_start + n_panel_rows:
            for k in range(diag_size):
                acc = tl.load(L + l_offset + row_idx * stride_ln + (diag_start + k))
                for p in range(k):
                    acc = acc - tl.load(
                        L + l_offset + (diag_start + k) * stride_ln + (diag_start + p)
                    ) * tl.load(L + l_offset + row_idx * stride_ln + (diag_start + p))
                l_kk = tl.load(
                    L + l_offset + (diag_start + k) * stride_ln + (diag_start + k)
                )
                tl.store(
                    L + l_offset + row_idx * stride_ln + (diag_start + k),
                    acc / l_kk,
                )


@triton.jit
def _syrk_kernel(
    L,
    panel_row_start,
    panel_col_start,
    diag_size,
    submatrix_start,
    n_submatrix,
    N,
    stride_ln,
    batch_stride,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    MAX_K: tl.constexpr,
):
    """Symmetric rank-k update: C -= A @ A^T using tl.dot.

    Zeros the upper triangle inline so a separate tril pass is not needed
    for the trailing submatrix.

    Grid: (batch_size, ceil(n_submatrix/TILE_M), ceil(n_submatrix/TILE_N))
    """
    pid_b = tle.program_id(0)
    pid_m = tle.program_id(1)
    pid_n = tle.program_id(2)

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, MAX_K)

    mask_m = offs_m < n_submatrix
    mask_n = offs_n < n_submatrix

    l_offset = pid_b * batch_stride

    # a[m,k] = L[panel_row_start + m, panel_col_start + k]
    a_ptrs = (
        L
        + l_offset
        + (panel_row_start + offs_m[:, None]) * stride_ln
        + (panel_col_start + offs_k[None, :])
    )

    # b[k,n] = L[panel_row_start + n, panel_col_start + k]
    b_ptrs = (
        L
        + l_offset
        + (panel_row_start + offs_n[None, :]) * stride_ln
        + (panel_col_start + offs_k[:, None])
    )

    # c[m,n] = L[submatrix_start + m, submatrix_start + n]
    c_ptrs = (
        L
        + l_offset
        + (submatrix_start + offs_m[:, None]) * stride_ln
        + (submatrix_start + offs_n[None, :])
    )

    mask_a = mask_m[:, None] & (offs_k[None, :] < diag_size)
    mask_b = (offs_k[:, None] < diag_size) & mask_n[None, :]

    a = tl.load(a_ptrs, mask=mask_a)
    b = tl.load(b_ptrs, mask=mask_b)
    accum = tl.dot(a, b)

    # Zero everything above the diagonal; update the lower triangle.
    is_lower = (pid_m * TILE_M + offs_m[:, None]) >= (pid_n * TILE_N + offs_n[None, :])
    mask_c = mask_m[:, None] & mask_n[None, :]

    c_val = tl.load(c_ptrs, mask=mask_c)
    tl.store(c_ptrs, tl.where(is_lower & mask_c, c_val - accum, 0.0), mask=mask_c)


# ---------------------------------------------------------------------------
# Host function
# ---------------------------------------------------------------------------
def linalg_cholesky(A, upper=False):
    """Cholesky decomposition optimized for Iluvatar BI-V150.

    Uses a blocked right-looking algorithm for float32 matrices larger than
    32x32, and a sequential kernel for small matrices and float64 (which
    does not rely on tl.dot, unavailable for fp64 on this backend).

    Like the generic implementation, the input is symmetrized and a small
    diagonal perturbation is added for numerical stability.
    """
    logger.debug("GEMS_ILUVATAR LINALG_CHOLESKY")
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
    m = shape[-2]

    if m != n:
        raise ValueError("A must be a square matrix")

    # Compute batch size
    if len(shape) == 2:
        batch_size = 1
    else:
        batch_dims = shape[:-2]
        batch_size = 1
        for dim in batch_dims:
            batch_size *= dim

    # Symmetrize: A = (A + A^T) / 2
    if len(shape) == 2:
        A_sym = (A + A.t().conj()) / 2
    else:
        A_view = A.reshape(-1, n, n)
        A_sym_view = (A_view + A_view.transpose(1, 2).conj()) / 2
        A_sym = A_sym_view.reshape(shape)

    # Add a small identity to improve numerical conditioning
    eps_val = 1e-5
    if len(shape) == 2:
        A_sym = A_sym + torch.eye(n, dtype=A.dtype, device=A.device) * eps_val
    else:
        eye = torch.eye(n, dtype=A.dtype, device=A.device)
        for _ in range(len(shape) - 2):
            eye = eye.unsqueeze(0)
        A_sym = A_sym + eye.expand(shape[:-2] + (n, n)) * eps_val

    # ---- Sequential path: N <= BLOCK or float64 ----
    if n <= BLOCK or A.dtype == torch.float64:
        L = torch.empty_like(A_sym)

        if len(shape) > 2:
            A_kernel = A_sym.reshape(-1, n, n)
            L_kernel = L.reshape(-1, n, n)
            stride_a = A_kernel.stride(1)
            stride_l = L_kernel.stride(1)
            batch_stride_val = A_kernel.stride(0)
        else:
            A_kernel = A_sym
            L_kernel = L
            stride_a = A_sym.stride(0)
            stride_l = L.stride(0)
            batch_stride_val = stride_a * n

        with torch.no_grad():
            cholesky_sequential_kernel[(batch_size,)](
                A_kernel, L_kernel, n, batch_stride_val, stride_a, stride_l
            )

        if len(shape) > 2:
            L = L.reshape(shape)

        # Use CPU-side mask to bypass backend tril kernel.
        mask = _get_tril_mask(n, L.dtype, L.device)
        L = L * mask
        if upper:
            L = L.transpose(-2, -1).conj()
        return L

    # ---- float32 blocked path (N > BLOCK) ----
    # Copy input into contiguous 3-D buffer (batch_size, n, n).
    if len(shape) > 2:
        A_kernel = A_sym.reshape(-1, n, n)
    else:
        A_kernel = A_sym.unsqueeze(0)
    L = A_kernel.clone(memory_format=torch.contiguous_format)
    stride_ln = L.stride(1)
    batch_stride_val = L.stride(0)

    with torch.no_grad():
        for j in range(0, n, BLOCK):
            block_size = min(BLOCK, n - j)
            n_rows = n - j - block_size

            # 1. Factor the diagonal block in-place.
            _cholesky_diag_kernel[(batch_size,)](
                L, j, block_size, n, stride_ln, batch_stride_val
            )

            if n_rows > 0:
                panel_row_start = j + block_size

                # 2. Triangular solve on the panel.
                n_tiles_m = triton.cdiv(n_rows, TRSM_TILE_M)
                _trsm_kernel[(batch_size, n_tiles_m)](
                    L,
                    j,
                    block_size,
                    panel_row_start,
                    n_rows,
                    n,
                    stride_ln,
                    batch_stride_val,
                    TILE_M=TRSM_TILE_M,
                )

                # 3. Symmetric rank-k update (zeros upper triangle).
                n_tiles_m = triton.cdiv(n_rows, TILE_M)
                n_tiles_n = triton.cdiv(n_rows, TILE_N)
                _syrk_kernel[(batch_size, n_tiles_m, n_tiles_n)](
                    L,
                    panel_row_start,
                    j,
                    block_size,
                    panel_row_start,
                    n_rows,
                    n,
                    stride_ln,
                    batch_stride_val,
                    TILE_M=TILE_M,
                    TILE_N=TILE_N,
                    MAX_K=BLOCK,
                )

    L = L.reshape(shape)

    # Zero upper triangle of the last diagonal block (not covered by syrk).
    mask = _get_tril_mask(n, L.dtype, L.device)
    L = L * mask

    if upper:
        L = L.transpose(-2, -1).conj()

    return L
