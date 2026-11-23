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

import logging
import warnings

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils.triton_lang_extension import program_id

logger = logging.getLogger(__name__)


def _check_cholesky_solve_out(B: torch.Tensor, out: torch.Tensor) -> None:
    """Match the device and safe-cast checks of aten::cholesky_solve.out."""
    if out.device != B.device:
        raise RuntimeError(
            "cholesky_solve: Expected result and input tensors to be on the "
            f"same device, but got result on {out.device} and input on {B.device}"
        )
    if not torch.can_cast(B.dtype, out.dtype):
        raise RuntimeError(
            "cholesky_solve: Expected result to be safely castable from "
            f"{B.dtype} dtype, but got result with dtype {out.dtype}"
        )


def _can_write_cholesky_solve_out_direct(
    B: torch.Tensor, L: torch.Tensor, out: torch.Tensor
) -> bool:
    """Whether the solve kernels can safely use ``out`` as their X buffer."""
    if B.layout != torch.strided or L.layout != torch.strided:
        return False
    if B.ndim < 2 or L.ndim < 2:
        return False
    if B.numel() == 0 or L.numel() == 0:
        return False
    if B.shape[:-2] != L.shape[:-2]:
        # Broadcasted solves may produce a result shape different from B.
        return False
    if out.shape != B.shape or out.dtype != B.dtype or out.device != B.device:
        return False
    if not out.is_contiguous() or out.is_conj() or out.is_neg():
        return False
    if torch._C._is_alias_of(out, B) or torch._C._is_alias_of(out, L):
        # A solve reads B and L while writing X, so overlapping storage needs
        # the existing temporary-and-copy fallback.
        return False
    return True


def _copy_cholesky_solve_out(result: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Resize and copy a temporary solve result into an out tensor."""
    if tuple(out.shape) != tuple(result.shape):
        if out.numel() != 0:
            warnings.warn(
                "An output with one or more elements was resized since it had "
                f"shape {list(out.shape)}, which does not match the required "
                f"output shape {list(result.shape)}. This behavior is deprecated, "
                "and in a future PyTorch release outputs will not be resized "
                "unless they have zero elements. You can explicitly reuse an out "
                "tensor t by resizing it, inplace, to zero elements with "
                "t.resize_(0).",
                UserWarning,
                stacklevel=3,
            )
        out.resize_(result.shape)
    out.copy_(result)
    return out


CHOLESKY_SOLVE_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_RHS": block_rhs}, num_warps=1, num_stages=1)
    for block_rhs in (1, 2, 4, 8, 16, 32)
]


@libentry()
@triton.autotune(
    configs=CHOLESKY_SOLVE_AUTOTUNE_CONFIGS, key=["N", "nrhs", "dtype_flag", "upper"]
)
@triton.jit
def cholesky_solve_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    N: tl.constexpr,
    nrhs: tl.constexpr,
    batch_stride_L,
    batch_stride_B,
    stride_L,
    stride_B,
    dtype_flag: tl.constexpr,
    upper: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
):
    """Cholesky solve kernel.

    Solves LL^T * X = B or U^T U * X = B for X, given the lower- or
    upper-triangular Cholesky factor and the right-hand side B. Each program
    computes one RHS tile for one matrix in the batch.

    Algorithm:
      lower=False path: L * Y = B, then L^T * X = Y.
      upper=True path: U^T * Y = B, then U * X = Y.
    """
    batch_pid = program_id(0)
    rhs_tile_pid = program_id(1)

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    cols = rhs_tile_pid * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    cols_mask = cols < nrhs

    # Phase 1: Forward substitution: solve L * Y = B.
    for i in range(N):
        sum_val = tl.load(B_ptr + B_base + i * stride_B + cols, mask=cols_mask)
        for j in range(i):
            if upper:
                L_val = tl.load(L_ptr + L_base + j * stride_L + i)
            else:
                L_val = tl.load(L_ptr + L_base + i * stride_L + j)
            Y_val = tl.load(X_ptr + B_base + j * stride_B + cols, mask=cols_mask)
            sum_val = sum_val - L_val * Y_val
        diag = tl.load(L_ptr + L_base + i * stride_L + i)
        # Fast reciprocal with Newton refinement
        inv_diag = 1.0 / diag
        inv_diag = inv_diag * (2.0 - diag * inv_diag)
        if dtype_flag == 1:
            inv_diag = inv_diag * (2.0 - diag * inv_diag)
        tl.store(
            X_ptr + B_base + i * stride_B + cols, sum_val * inv_diag, mask=cols_mask
        )

    # Phase 2: Backward substitution: solve L^T * X = Y.
    for i in range(N - 1, -1, -1):
        sum_val = tl.load(X_ptr + B_base + i * stride_B + cols, mask=cols_mask)
        for j in range(i + 1, N):
            if upper:
                L_val = tl.load(L_ptr + L_base + i * stride_L + j)
            else:
                L_val = tl.load(L_ptr + L_base + j * stride_L + i)
            Xj_val = tl.load(X_ptr + B_base + j * stride_B + cols, mask=cols_mask)
            sum_val = sum_val - L_val * Xj_val
        diag = tl.load(L_ptr + L_base + i * stride_L + i)
        inv_diag = 1.0 / diag
        inv_diag = inv_diag * (2.0 - diag * inv_diag)
        if dtype_flag == 1:
            inv_diag = inv_diag * (2.0 - diag * inv_diag)
        tl.store(
            X_ptr + B_base + i * stride_B + cols, sum_val * inv_diag, mask=cols_mask
        )


@libentry()
@triton.jit
def cholesky_solve_complex_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    N: tl.constexpr,
    nrhs: tl.constexpr,
    batch_stride_L,
    batch_stride_B,
    stride_L_row,
    stride_L_col,
    stride_B_row,
    stride_B_col,
    BLOCK_RHS: tl.constexpr,
    upper: tl.constexpr,
    storage_conj: tl.constexpr,
):
    """Complex Cholesky solve over interleaved real/imaginary storage.

    torch.view_as_real exposes each complex scalar as two adjacent real
    values. Keeping the arithmetic split avoids relying on native complex
    support in Triton and works for both complex64 and complex128 pointers.
    """
    batch_pid = program_id(0)
    rhs_tile_pid = program_id(1)

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    cols = rhs_tile_pid * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    cols_mask = cols < nrhs

    # Phase 1: solve L * Y = B, or U^H * Y = B for an upper factor.
    for i in range(N):
        b_offset = B_base + i * stride_B_row + cols * stride_B_col
        sum_real = tl.load(B_ptr + b_offset, mask=cols_mask, other=0.0)
        sum_imag = tl.load(B_ptr + b_offset + 1, mask=cols_mask, other=0.0)
        for j in range(i):
            if upper:
                l_offset = L_base + j * stride_L_row + i * stride_L_col
            else:
                l_offset = L_base + i * stride_L_row + j * stride_L_col
            l_real = tl.load(L_ptr + l_offset)
            l_imag = tl.load(L_ptr + l_offset + 1)
            if storage_conj != upper:
                l_imag = -l_imag

            y_offset = B_base + j * stride_B_row + cols * stride_B_col
            y_real = tl.load(X_ptr + y_offset, mask=cols_mask, other=0.0)
            y_imag = tl.load(X_ptr + y_offset + 1, mask=cols_mask, other=0.0)
            sum_real -= l_real * y_real - l_imag * y_imag
            sum_imag -= l_real * y_imag + l_imag * y_real

        diag_offset = L_base + i * stride_L_row + i * stride_L_col
        diag_real = tl.load(L_ptr + diag_offset)
        diag_imag = tl.load(L_ptr + diag_offset + 1)
        if storage_conj != upper:
            diag_imag = -diag_imag
        denominator = diag_real * diag_real + diag_imag * diag_imag
        y_real = (sum_real * diag_real + sum_imag * diag_imag) / denominator
        y_imag = (sum_imag * diag_real - sum_real * diag_imag) / denominator
        tl.store(X_ptr + b_offset, y_real, mask=cols_mask)
        tl.store(X_ptr + b_offset + 1, y_imag, mask=cols_mask)

    # Phase 2: solve L^H * X = Y, or U * X = Y for an upper factor.
    for i in range(N - 1, -1, -1):
        x_offset = B_base + i * stride_B_row + cols * stride_B_col
        sum_real = tl.load(X_ptr + x_offset, mask=cols_mask, other=0.0)
        sum_imag = tl.load(X_ptr + x_offset + 1, mask=cols_mask, other=0.0)
        for j in range(i + 1, N):
            if upper:
                l_offset = L_base + i * stride_L_row + j * stride_L_col
            else:
                l_offset = L_base + j * stride_L_row + i * stride_L_col
            l_real = tl.load(L_ptr + l_offset)
            l_imag = tl.load(L_ptr + l_offset + 1)
            if storage_conj == upper:
                l_imag = -l_imag

            xj_offset = B_base + j * stride_B_row + cols * stride_B_col
            xj_real = tl.load(X_ptr + xj_offset, mask=cols_mask, other=0.0)
            xj_imag = tl.load(X_ptr + xj_offset + 1, mask=cols_mask, other=0.0)
            sum_real -= l_real * xj_real - l_imag * xj_imag
            sum_imag -= l_real * xj_imag + l_imag * xj_real

        diag_offset = L_base + i * stride_L_row + i * stride_L_col
        diag_real = tl.load(L_ptr + diag_offset)
        diag_imag = tl.load(L_ptr + diag_offset + 1)
        if storage_conj == upper:
            diag_imag = -diag_imag
        denominator = diag_real * diag_real + diag_imag * diag_imag
        out_real = (sum_real * diag_real + sum_imag * diag_imag) / denominator
        out_imag = (sum_imag * diag_real - sum_real * diag_imag) / denominator
        tl.store(X_ptr + x_offset, out_real, mask=cols_mask)
        tl.store(X_ptr + x_offset + 1, out_imag, mask=cols_mask)


def _can_use_blocked_path(N, nrhs):
    return N >= 64 and N % 32 == 0 and nrhs >= 4


def _can_use_blocked_single_rhs_path(N, nrhs):
    # N <= 32 single-RHS solves use the register-resident small gather
    # kernel instead; the blocked kernels require N % BLOCK_K == 0.
    return nrhs == 1 and N >= 64 and N % 32 == 0


def _can_use_small_gather_path(N, nrhs):
    return (N <= 32 and nrhs <= 8) or (32 < N < 64 and nrhs == 1)


@libentry()
@triton.jit
def cholesky_solve_single_rhs_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    N: tl.constexpr,
    batch_stride_L,
    batch_stride_B,
    stride_L,
    stride_B,
    dtype_flag: tl.constexpr,
    upper: tl.constexpr,
):
    """Specialized Cholesky solve kernel for nrhs == 1.

    This path avoids RHS tile vectors and tail masks used by the general
    multi-RHS kernel. Each program solves one single-RHS system for one batch.
    """
    batch_pid = program_id(0)

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B

    # Phase 1: solve L * Y = B or U^T * Y = B.
    for i in range(N):
        sum_val = tl.load(B_ptr + B_base + i * stride_B)
        for j in range(i):
            if upper:
                L_val = tl.load(L_ptr + L_base + j * stride_L + i)
            else:
                L_val = tl.load(L_ptr + L_base + i * stride_L + j)
            Y_val = tl.load(X_ptr + B_base + j * stride_B)
            sum_val = sum_val - L_val * Y_val
        diag = tl.load(L_ptr + L_base + i * stride_L + i)
        inv_diag = 1.0 / diag
        inv_diag = inv_diag * (2.0 - diag * inv_diag)
        if dtype_flag == 1:
            inv_diag = inv_diag * (2.0 - diag * inv_diag)
        tl.store(X_ptr + B_base + i * stride_B, sum_val * inv_diag)

    # Phase 2: solve L^T * X = Y or U * X = Y.
    for i in range(N - 1, -1, -1):
        sum_val = tl.load(X_ptr + B_base + i * stride_B)
        for j in range(i + 1, N):
            if upper:
                L_val = tl.load(L_ptr + L_base + i * stride_L + j)
            else:
                L_val = tl.load(L_ptr + L_base + j * stride_L + i)
            Xj_val = tl.load(X_ptr + B_base + j * stride_B)
            sum_val = sum_val - L_val * Xj_val
        diag = tl.load(L_ptr + L_base + i * stride_L + i)
        inv_diag = 1.0 / diag
        inv_diag = inv_diag * (2.0 - diag * inv_diag)
        if dtype_flag == 1:
            inv_diag = inv_diag * (2.0 - diag * inv_diag)
        tl.store(X_ptr + B_base + i * stride_B, sum_val * inv_diag)


# ---------------------------------------------------------------------------
# Portable kernels (no tl.gather).
#
# Triton backends without a tl.gather lowering cannot run the optimized
# kernels above. The portable variants keep the same blocked TRSM structure
# but move the serial per-row substitution out of the solve kernels: a
# precompute kernel inverts every BLOCK_K diagonal block (fully parallel
# across blocks and batches, using a masked reduce instead of tl.gather),
# and the solve kernels then apply each diagonal block with a single
# tl.dot/matvec. Small systems use a register-resident kernel whose row
# extraction is a masked reduce instead of tl.gather.
# ---------------------------------------------------------------------------


@libentry()
@triton.jit
def cholesky_solve_invert_blocks_portable_kernel(
    L_ptr,
    T_ptr,
    Tt_ptr,
    N: tl.constexpr,
    batch_stride_L,
    stride_L,
    BLOCK_K: tl.constexpr,
    upper: tl.constexpr,
    dtype_flag: tl.constexpr,
):
    """Invert every BLOCK_K diagonal block of the effective lower factor.

    Each diagonal block is D(I + M) with unit lower (I + M). Stores
    T = (I + M)^-1 and its transpose, so the solve kernels replace the
    serial per-row substitution chain with one tl.dot/matvec per block:
    forward x = T @ (y * inv_diag), backward x = inv_diag * (T^T @ y).
    The row extraction uses a masked reduce instead of tl.gather, which
    keeps the kernel portable; it is fully parallel across blocks and
    batches, so the extra reduce cost is hidden.
    """
    batch_pid = program_id(0)
    block_pid = program_id(1)
    L_base = batch_pid * batch_stride_L
    k = block_pid * BLOCK_K
    rows = tl.arange(0, BLOCK_K)

    diag = tl.load(L_ptr + L_base + (k + rows) * stride_L + (k + rows))
    inv_diag = 1.0 / diag
    inv_diag = inv_diag * (2.0 - diag * inv_diag)
    if dtype_flag == 1:
        inv_diag = inv_diag * (2.0 - diag * inv_diag)

    t = tl.where(rows[:, None] == rows[None, :], 1.0, 0.0).to(L_ptr.dtype.element_ty)
    for i in range(BLOCK_K):
        if upper:
            # The effective lower factor is U^T: L_eff[r, i] = U[i, r].
            col = tl.load(
                L_ptr + L_base + (k + i) * stride_L + (k + rows),
                mask=rows > i,
                other=0.0,
            )
        else:
            col = tl.load(
                L_ptr + L_base + (k + rows) * stride_L + (k + i),
                mask=rows > i,
                other=0.0,
            )
        norm = col * inv_diag
        pivot = tl.sum(tl.where(rows[:, None] == i, t, 0.0), axis=0)
        t = t - norm[:, None] * pivot[None, :]

    t_base = batch_pid * N * BLOCK_K + k * BLOCK_K
    t_off = t_base + rows[:, None] * BLOCK_K + rows[None, :]
    tl.store(T_ptr + t_off, t)
    tl.store(Tt_ptr + t_off, tl.trans(t))


@libentry()
@triton.jit
def cholesky_solve_blocked_lower_portable_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    T_ptr,
    Tt_ptr,
    N: tl.constexpr,
    nrhs: tl.constexpr,
    batch_stride_L,
    batch_stride_B,
    stride_L,
    stride_B,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
    dtype_flag: tl.constexpr,
    USE_SUM: tl.constexpr,
):
    """Portable blocked lower-factor solve: diagonal blocks via precomputed
    inverses instead of the tl.gather substitution chain. Narrow RHS tiles use
    explicit reductions; wider tiles retain tl.dot."""
    batch_pid = program_id(0)
    rhs_tile_pid = program_id(1)

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    T_base = batch_pid * N * BLOCK_K
    rhs_cols = rhs_tile_pid * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    rhs_mask = rhs_cols < nrhs
    k_offsets = tl.arange(0, BLOCK_K)
    m_offsets = tl.arange(0, BLOCK_M)

    # Forward blocked TRSM: L * Y = B.
    for k in range(0, N, BLOCK_K):
        rows_k = k + k_offsets
        if k > 0:
            tl.debug_barrier()
        y_off = B_base + rows_k[:, None] * stride_B + rhs_cols[None, :]
        if k == 0:
            y_block = tl.load(B_ptr + y_off, mask=rhs_mask[None, :], other=0.0)
        else:
            y_block = tl.load(X_ptr + y_off, mask=rhs_mask[None, :], other=0.0)

        diag_block = tl.load(L_ptr + L_base + rows_k * stride_L + rows_k)
        inv_diag = 1.0 / diag_block
        inv_diag = inv_diag * (2.0 - diag_block * inv_diag)
        if dtype_flag == 1:
            inv_diag = inv_diag * (2.0 - diag_block * inv_diag)

        t_off = T_base + k * BLOCK_K + k_offsets[:, None] * BLOCK_K + k_offsets[None, :]
        t_block = tl.load(T_ptr + t_off)
        sv = y_block * inv_diag[:, None]
        if USE_SUM:
            y_block = tl.sum(t_block[:, :, None] * sv[None, :, :], axis=1)
        else:
            y_block = tl.dot(t_block, sv, input_precision="ieee")

        tl.store(X_ptr + y_off, y_block, mask=rhs_mask[None, :])

        for m in range(k + BLOCK_K, N, BLOCK_M):
            rows_m = m + m_offsets
            L_tile = tl.load(
                L_ptr + L_base + rows_m[:, None] * stride_L + rows_k[None, :]
            )
            tail_off = B_base + rows_m[:, None] * stride_B + rhs_cols[None, :]
            if k == 0:
                tail = tl.load(B_ptr + tail_off, mask=rhs_mask[None, :], other=0.0)
            else:
                tail = tl.load(X_ptr + tail_off, mask=rhs_mask[None, :], other=0.0)
            if USE_SUM:
                update = tl.sum(L_tile[:, :, None] * y_block[None, :, :], axis=1)
            else:
                update = tl.dot(L_tile, y_block, input_precision="ieee")
            tail = tail - update
            tl.store(X_ptr + tail_off, tail, mask=rhs_mask[None, :])

    # Backward blocked TRSM: L^T * X = Y.
    for k in range(N - BLOCK_K, -1, -BLOCK_K):
        rows_k = k + k_offsets
        tl.debug_barrier()
        x_off = B_base + rows_k[:, None] * stride_B + rhs_cols[None, :]
        x_block = tl.load(X_ptr + x_off, mask=rhs_mask[None, :], other=0.0)

        diag_block = tl.load(L_ptr + L_base + rows_k * stride_L + rows_k)
        inv_diag = 1.0 / diag_block
        inv_diag = inv_diag * (2.0 - diag_block * inv_diag)
        if dtype_flag == 1:
            inv_diag = inv_diag * (2.0 - diag_block * inv_diag)

        t_off = T_base + k * BLOCK_K + k_offsets[:, None] * BLOCK_K + k_offsets[None, :]
        tt_block = tl.load(Tt_ptr + t_off)
        if USE_SUM:
            x_block = tl.sum(tt_block[:, :, None] * x_block[None, :, :], axis=1)
        else:
            x_block = tl.dot(tt_block, x_block, input_precision="ieee")
        x_block = x_block * inv_diag[:, None]

        tl.store(X_ptr + x_off, x_block, mask=rhs_mask[None, :])

        for m in range(0, k, BLOCK_M):
            rows_m = m + m_offsets
            L_tile = tl.load(
                L_ptr + L_base + rows_k[None, :] * stride_L + rows_m[:, None]
            )
            head_off = B_base + rows_m[:, None] * stride_B + rhs_cols[None, :]
            head = tl.load(X_ptr + head_off, mask=rhs_mask[None, :], other=0.0)
            if USE_SUM:
                update = tl.sum(L_tile[:, :, None] * x_block[None, :, :], axis=1)
            else:
                update = tl.dot(L_tile, x_block, input_precision="ieee")
            head = head - update
            tl.store(X_ptr + head_off, head, mask=rhs_mask[None, :])


@libentry()
@triton.jit
def cholesky_solve_blocked_upper_portable_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    T_ptr,
    Tt_ptr,
    N: tl.constexpr,
    nrhs: tl.constexpr,
    batch_stride_L,
    batch_stride_B,
    stride_L,
    stride_B,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
    dtype_flag: tl.constexpr,
    USE_SUM: tl.constexpr,
):
    """Portable blocked upper-factor solve. Mirrors the lower kernel; the
    precomputed inverses belong to the effective lower factor U^T."""
    batch_pid = program_id(0)
    rhs_tile_pid = program_id(1)

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    T_base = batch_pid * N * BLOCK_K
    rhs_cols = rhs_tile_pid * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    rhs_mask = rhs_cols < nrhs
    k_offsets = tl.arange(0, BLOCK_K)
    m_offsets = tl.arange(0, BLOCK_M)

    # Forward blocked TRSM: U^T * Y = B.
    for k in range(0, N, BLOCK_K):
        rows_k = k + k_offsets
        if k > 0:
            tl.debug_barrier()
        y_off = B_base + rows_k[:, None] * stride_B + rhs_cols[None, :]
        if k == 0:
            y_block = tl.load(B_ptr + y_off, mask=rhs_mask[None, :], other=0.0)
        else:
            y_block = tl.load(X_ptr + y_off, mask=rhs_mask[None, :], other=0.0)

        diag_block = tl.load(L_ptr + L_base + rows_k * stride_L + rows_k)
        inv_diag = 1.0 / diag_block
        inv_diag = inv_diag * (2.0 - diag_block * inv_diag)
        if dtype_flag == 1:
            inv_diag = inv_diag * (2.0 - diag_block * inv_diag)

        t_off = T_base + k * BLOCK_K + k_offsets[:, None] * BLOCK_K + k_offsets[None, :]
        t_block = tl.load(T_ptr + t_off)
        sv = y_block * inv_diag[:, None]
        if USE_SUM:
            y_block = tl.sum(t_block[:, :, None] * sv[None, :, :], axis=1)
        else:
            y_block = tl.dot(t_block, sv, input_precision="ieee")

        tl.store(X_ptr + y_off, y_block, mask=rhs_mask[None, :])

        for m in range(k + BLOCK_K, N, BLOCK_M):
            rows_m = m + m_offsets
            U_tile = tl.trans(
                tl.load(L_ptr + L_base + rows_k[:, None] * stride_L + rows_m[None, :])
            )
            tail_off = B_base + rows_m[:, None] * stride_B + rhs_cols[None, :]
            if k == 0:
                tail = tl.load(B_ptr + tail_off, mask=rhs_mask[None, :], other=0.0)
            else:
                tail = tl.load(X_ptr + tail_off, mask=rhs_mask[None, :], other=0.0)
            if USE_SUM:
                update = tl.sum(U_tile[:, :, None] * y_block[None, :, :], axis=1)
            else:
                update = tl.dot(U_tile, y_block, input_precision="ieee")
            tail = tail - update
            tl.store(X_ptr + tail_off, tail, mask=rhs_mask[None, :])

    # Backward blocked TRSM: U * X = Y.
    for k in range(N - BLOCK_K, -1, -BLOCK_K):
        rows_k = k + k_offsets
        tl.debug_barrier()
        x_off = B_base + rows_k[:, None] * stride_B + rhs_cols[None, :]
        x_block = tl.load(X_ptr + x_off, mask=rhs_mask[None, :], other=0.0)

        diag_block = tl.load(L_ptr + L_base + rows_k * stride_L + rows_k)
        inv_diag = 1.0 / diag_block
        inv_diag = inv_diag * (2.0 - diag_block * inv_diag)
        if dtype_flag == 1:
            inv_diag = inv_diag * (2.0 - diag_block * inv_diag)

        t_off = T_base + k * BLOCK_K + k_offsets[:, None] * BLOCK_K + k_offsets[None, :]
        tt_block = tl.load(Tt_ptr + t_off)
        if USE_SUM:
            x_block = tl.sum(tt_block[:, :, None] * x_block[None, :, :], axis=1)
        else:
            x_block = tl.dot(tt_block, x_block, input_precision="ieee")
        x_block = x_block * inv_diag[:, None]

        tl.store(X_ptr + x_off, x_block, mask=rhs_mask[None, :])

        for m in range(0, k, BLOCK_M):
            rows_m = m + m_offsets
            U_tile = tl.load(
                L_ptr + L_base + rows_m[:, None] * stride_L + rows_k[None, :]
            )
            head_off = B_base + rows_m[:, None] * stride_B + rhs_cols[None, :]
            head = tl.load(X_ptr + head_off, mask=rhs_mask[None, :], other=0.0)
            if USE_SUM:
                update = tl.sum(U_tile[:, :, None] * x_block[None, :, :], axis=1)
            else:
                update = tl.dot(U_tile, x_block, input_precision="ieee")
            head = head - update
            tl.store(X_ptr + head_off, head, mask=rhs_mask[None, :])


@libentry()
@triton.jit
def cholesky_solve_blocked_lower_portable_fp64_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    T_ptr,
    Tt_ptr,
    N: tl.constexpr,
    nrhs: tl.constexpr,
    batch_stride_L,
    batch_stride_B,
    stride_L,
    stride_B,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
):
    """fp64 portable blocked lower-factor solve.

    tl.dot has no correct fp64 lowering on backends that need the portable
    path, so every product is an explicit broadcast-multiply + tl.sum.
    """
    batch_pid = program_id(0)
    rhs_tile_pid = program_id(1)

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    T_base = batch_pid * N * BLOCK_K
    rhs_cols = rhs_tile_pid * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    rhs_mask = rhs_cols < nrhs
    k_offsets = tl.arange(0, BLOCK_K)
    m_offsets = tl.arange(0, BLOCK_M)

    # Forward blocked TRSM: L * Y = B.
    for k in range(0, N, BLOCK_K):
        rows_k = k + k_offsets
        if k > 0:
            tl.debug_barrier()
        y_off = B_base + rows_k[:, None] * stride_B + rhs_cols[None, :]
        if k == 0:
            y_block = tl.load(B_ptr + y_off, mask=rhs_mask[None, :], other=0.0)
        else:
            y_block = tl.load(X_ptr + y_off, mask=rhs_mask[None, :], other=0.0)

        diag_block = tl.load(L_ptr + L_base + rows_k * stride_L + rows_k)
        inv_diag = 1.0 / diag_block
        inv_diag = inv_diag * (2.0 - diag_block * inv_diag)
        inv_diag = inv_diag * (2.0 - diag_block * inv_diag)

        t_off = T_base + k * BLOCK_K + k_offsets[:, None] * BLOCK_K + k_offsets[None, :]
        t_block = tl.load(T_ptr + t_off)
        sv = y_block * inv_diag[:, None]
        y_block = tl.sum(t_block[:, :, None] * sv[None, :, :], axis=1)

        tl.store(X_ptr + y_off, y_block, mask=rhs_mask[None, :])

        for m in range(k + BLOCK_K, N, BLOCK_M):
            rows_m = m + m_offsets
            rows_m_mask = rows_m < N
            L_tile = tl.load(
                L_ptr + L_base + rows_m[:, None] * stride_L + rows_k[None, :],
                mask=rows_m_mask[:, None],
                other=0.0,
            )
            tail_off = B_base + rows_m[:, None] * stride_B + rhs_cols[None, :]
            tail_mask = rows_m_mask[:, None] & rhs_mask[None, :]
            if k == 0:
                tail = tl.load(B_ptr + tail_off, mask=tail_mask, other=0.0)
            else:
                tail = tl.load(X_ptr + tail_off, mask=tail_mask, other=0.0)
            tail = tail - tl.sum(L_tile[:, :, None] * y_block[None, :, :], axis=1)
            tl.store(X_ptr + tail_off, tail, mask=tail_mask)

    # Backward blocked TRSM: L^T * X = Y.
    for k in range(N - BLOCK_K, -1, -BLOCK_K):
        rows_k = k + k_offsets
        tl.debug_barrier()
        x_off = B_base + rows_k[:, None] * stride_B + rhs_cols[None, :]
        x_block = tl.load(X_ptr + x_off, mask=rhs_mask[None, :], other=0.0)

        diag_block = tl.load(L_ptr + L_base + rows_k * stride_L + rows_k)
        inv_diag = 1.0 / diag_block
        inv_diag = inv_diag * (2.0 - diag_block * inv_diag)
        inv_diag = inv_diag * (2.0 - diag_block * inv_diag)

        t_off = T_base + k * BLOCK_K + k_offsets[:, None] * BLOCK_K + k_offsets[None, :]
        tt_block = tl.load(Tt_ptr + t_off)
        x_block = tl.sum(tt_block[:, :, None] * x_block[None, :, :], axis=1)
        x_block = x_block * inv_diag[:, None]

        tl.store(X_ptr + x_off, x_block, mask=rhs_mask[None, :])

        for m in range(0, k, BLOCK_M):
            rows_m = m + m_offsets
            rows_m_mask = rows_m < k
            L_tile = tl.load(
                L_ptr + L_base + rows_k[None, :] * stride_L + rows_m[:, None],
                mask=rows_m_mask[:, None],
                other=0.0,
            )
            head_off = B_base + rows_m[:, None] * stride_B + rhs_cols[None, :]
            head_mask = rows_m_mask[:, None] & rhs_mask[None, :]
            head = tl.load(X_ptr + head_off, mask=head_mask, other=0.0)
            head = head - tl.sum(L_tile[:, :, None] * x_block[None, :, :], axis=1)
            tl.store(X_ptr + head_off, head, mask=head_mask)


@libentry()
@triton.jit
def cholesky_solve_blocked_upper_portable_fp64_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    T_ptr,
    Tt_ptr,
    N: tl.constexpr,
    nrhs: tl.constexpr,
    batch_stride_L,
    batch_stride_B,
    stride_L,
    stride_B,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
):
    """fp64 portable blocked upper-factor solve (see the lower variant)."""
    batch_pid = program_id(0)
    rhs_tile_pid = program_id(1)

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    T_base = batch_pid * N * BLOCK_K
    rhs_cols = rhs_tile_pid * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    rhs_mask = rhs_cols < nrhs
    k_offsets = tl.arange(0, BLOCK_K)
    m_offsets = tl.arange(0, BLOCK_M)

    # Forward blocked TRSM: U^T * Y = B.
    for k in range(0, N, BLOCK_K):
        rows_k = k + k_offsets
        if k > 0:
            tl.debug_barrier()
        y_off = B_base + rows_k[:, None] * stride_B + rhs_cols[None, :]
        if k == 0:
            y_block = tl.load(B_ptr + y_off, mask=rhs_mask[None, :], other=0.0)
        else:
            y_block = tl.load(X_ptr + y_off, mask=rhs_mask[None, :], other=0.0)

        diag_block = tl.load(L_ptr + L_base + rows_k * stride_L + rows_k)
        inv_diag = 1.0 / diag_block
        inv_diag = inv_diag * (2.0 - diag_block * inv_diag)
        inv_diag = inv_diag * (2.0 - diag_block * inv_diag)

        t_off = T_base + k * BLOCK_K + k_offsets[:, None] * BLOCK_K + k_offsets[None, :]
        t_block = tl.load(T_ptr + t_off)
        sv = y_block * inv_diag[:, None]
        y_block = tl.sum(t_block[:, :, None] * sv[None, :, :], axis=1)

        tl.store(X_ptr + y_off, y_block, mask=rhs_mask[None, :])

        for m in range(k + BLOCK_K, N, BLOCK_M):
            rows_m = m + m_offsets
            rows_m_mask = rows_m < N
            U_tile = tl.trans(
                tl.load(
                    L_ptr + L_base + rows_k[:, None] * stride_L + rows_m[None, :],
                    mask=rows_m_mask[None, :],
                    other=0.0,
                )
            )
            tail_off = B_base + rows_m[:, None] * stride_B + rhs_cols[None, :]
            tail_mask = rows_m_mask[:, None] & rhs_mask[None, :]
            if k == 0:
                tail = tl.load(B_ptr + tail_off, mask=tail_mask, other=0.0)
            else:
                tail = tl.load(X_ptr + tail_off, mask=tail_mask, other=0.0)
            tail = tail - tl.sum(U_tile[:, :, None] * y_block[None, :, :], axis=1)
            tl.store(X_ptr + tail_off, tail, mask=tail_mask)

    # Backward blocked TRSM: U * X = Y.
    for k in range(N - BLOCK_K, -1, -BLOCK_K):
        rows_k = k + k_offsets
        tl.debug_barrier()
        x_off = B_base + rows_k[:, None] * stride_B + rhs_cols[None, :]
        x_block = tl.load(X_ptr + x_off, mask=rhs_mask[None, :], other=0.0)

        diag_block = tl.load(L_ptr + L_base + rows_k * stride_L + rows_k)
        inv_diag = 1.0 / diag_block
        inv_diag = inv_diag * (2.0 - diag_block * inv_diag)
        inv_diag = inv_diag * (2.0 - diag_block * inv_diag)

        t_off = T_base + k * BLOCK_K + k_offsets[:, None] * BLOCK_K + k_offsets[None, :]
        tt_block = tl.load(Tt_ptr + t_off)
        x_block = tl.sum(tt_block[:, :, None] * x_block[None, :, :], axis=1)
        x_block = x_block * inv_diag[:, None]

        tl.store(X_ptr + x_off, x_block, mask=rhs_mask[None, :])

        for m in range(0, k, BLOCK_M):
            rows_m = m + m_offsets
            rows_m_mask = rows_m < k
            U_tile = tl.load(
                L_ptr + L_base + rows_m[:, None] * stride_L + rows_k[None, :],
                mask=rows_m_mask[:, None],
                other=0.0,
            )
            head_off = B_base + rows_m[:, None] * stride_B + rhs_cols[None, :]
            head_mask = rows_m_mask[:, None] & rhs_mask[None, :]
            head = tl.load(X_ptr + head_off, mask=head_mask, other=0.0)
            head = head - tl.sum(U_tile[:, :, None] * x_block[None, :, :], axis=1)
            tl.store(X_ptr + head_off, head, mask=head_mask)


@libentry()
@triton.jit
def cholesky_solve_single_rhs_blocked_lower_portable_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    T_ptr,
    Tt_ptr,
    N: tl.constexpr,
    batch_stride_L,
    batch_stride_B,
    stride_L,
    stride_B,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    dtype_flag: tl.constexpr,
):
    """Portable blocked lower-factor single-RHS solve.

    Same precomputed-inverse structure as the multi-RHS portable kernel,
    with tl.sum matvecs in place of tl.dot (BLOCK_RHS == 1).
    """
    batch_pid = program_id(0)

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    T_base = batch_pid * N * BLOCK_K
    k_offsets = tl.arange(0, BLOCK_K)
    m_offsets = tl.arange(0, BLOCK_M)

    # Forward blocked TRSV: L * Y = B.
    for k in range(0, N, BLOCK_K):
        rows_k = k + k_offsets
        if k > 0:
            tl.debug_barrier()
        if k == 0:
            y_block = tl.load(B_ptr + B_base + rows_k * stride_B)
        else:
            y_block = tl.load(X_ptr + B_base + rows_k * stride_B)

        diag_block = tl.load(L_ptr + L_base + rows_k * stride_L + rows_k)
        inv_diag = 1.0 / diag_block
        inv_diag = inv_diag * (2.0 - diag_block * inv_diag)
        if dtype_flag == 1:
            inv_diag = inv_diag * (2.0 - diag_block * inv_diag)

        t_off = T_base + k * BLOCK_K + k_offsets[:, None] * BLOCK_K + k_offsets[None, :]
        t_block = tl.load(T_ptr + t_off)
        w = tl.sum(t_block * (y_block * inv_diag)[None, :], axis=1)

        tl.store(X_ptr + B_base + rows_k * stride_B, w)

        for m in range(k + BLOCK_K, N, BLOCK_M):
            rows_m = m + m_offsets
            rows_m_mask = rows_m < N
            L_tile = tl.load(
                L_ptr + L_base + rows_m[:, None] * stride_L + rows_k[None, :],
                mask=rows_m_mask[:, None],
                other=0.0,
            )
            if k == 0:
                tail = tl.load(
                    B_ptr + B_base + rows_m * stride_B, mask=rows_m_mask, other=0.0
                )
            else:
                tail = tl.load(
                    X_ptr + B_base + rows_m * stride_B, mask=rows_m_mask, other=0.0
                )
            tail = tail - tl.sum(L_tile * w[None, :], axis=1)
            tl.store(X_ptr + B_base + rows_m * stride_B, tail, mask=rows_m_mask)

    # Backward blocked TRSV: L^T * X = Y.
    for k in range(N - BLOCK_K, -1, -BLOCK_K):
        rows_k = k + k_offsets
        tl.debug_barrier()
        x_block = tl.load(X_ptr + B_base + rows_k * stride_B)

        diag_block = tl.load(L_ptr + L_base + rows_k * stride_L + rows_k)
        inv_diag = 1.0 / diag_block
        inv_diag = inv_diag * (2.0 - diag_block * inv_diag)
        if dtype_flag == 1:
            inv_diag = inv_diag * (2.0 - diag_block * inv_diag)

        t_off = T_base + k * BLOCK_K + k_offsets[:, None] * BLOCK_K + k_offsets[None, :]
        tt_block = tl.load(Tt_ptr + t_off)
        w = tl.sum(tt_block * x_block[None, :], axis=1) * inv_diag

        tl.store(X_ptr + B_base + rows_k * stride_B, w)

        for m in range(0, k, BLOCK_M):
            rows_m = m + m_offsets
            rows_m_mask = rows_m < k
            L_tile = tl.load(
                L_ptr + L_base + rows_k[None, :] * stride_L + rows_m[:, None],
                mask=rows_m_mask[:, None],
                other=0.0,
            )
            head = tl.load(
                X_ptr + B_base + rows_m * stride_B, mask=rows_m_mask, other=0.0
            )
            head = head - tl.sum(L_tile * w[None, :], axis=1)
            tl.store(X_ptr + B_base + rows_m * stride_B, head, mask=rows_m_mask)


@libentry()
@triton.jit
def cholesky_solve_single_rhs_blocked_upper_portable_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    T_ptr,
    Tt_ptr,
    N: tl.constexpr,
    batch_stride_L,
    batch_stride_B,
    stride_L,
    stride_B,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    dtype_flag: tl.constexpr,
):
    """Portable blocked upper-factor single-RHS solve (see lower variant)."""
    batch_pid = program_id(0)

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    T_base = batch_pid * N * BLOCK_K
    k_offsets = tl.arange(0, BLOCK_K)
    m_offsets = tl.arange(0, BLOCK_M)

    # Forward blocked TRSV: U^T * Y = B.
    for k in range(0, N, BLOCK_K):
        rows_k = k + k_offsets
        if k > 0:
            tl.debug_barrier()
        if k == 0:
            y_block = tl.load(B_ptr + B_base + rows_k * stride_B)
        else:
            y_block = tl.load(X_ptr + B_base + rows_k * stride_B)

        diag_block = tl.load(L_ptr + L_base + rows_k * stride_L + rows_k)
        inv_diag = 1.0 / diag_block
        inv_diag = inv_diag * (2.0 - diag_block * inv_diag)
        if dtype_flag == 1:
            inv_diag = inv_diag * (2.0 - diag_block * inv_diag)

        t_off = T_base + k * BLOCK_K + k_offsets[:, None] * BLOCK_K + k_offsets[None, :]
        t_block = tl.load(T_ptr + t_off)
        w = tl.sum(t_block * (y_block * inv_diag)[None, :], axis=1)

        tl.store(X_ptr + B_base + rows_k * stride_B, w)

        for m in range(k + BLOCK_K, N, BLOCK_M):
            rows_m = m + m_offsets
            rows_m_mask = rows_m < N
            U_tile = tl.load(
                L_ptr + L_base + rows_k[:, None] * stride_L + rows_m[None, :],
                mask=rows_m_mask[None, :],
                other=0.0,
            )
            if k == 0:
                tail = tl.load(
                    B_ptr + B_base + rows_m * stride_B, mask=rows_m_mask, other=0.0
                )
            else:
                tail = tl.load(
                    X_ptr + B_base + rows_m * stride_B, mask=rows_m_mask, other=0.0
                )
            tail = tail - tl.sum(U_tile * w[:, None], axis=0)
            tl.store(X_ptr + B_base + rows_m * stride_B, tail, mask=rows_m_mask)

    # Backward blocked TRSV: U * X = Y.
    for k in range(N - BLOCK_K, -1, -BLOCK_K):
        rows_k = k + k_offsets
        tl.debug_barrier()
        x_block = tl.load(X_ptr + B_base + rows_k * stride_B)

        diag_block = tl.load(L_ptr + L_base + rows_k * stride_L + rows_k)
        inv_diag = 1.0 / diag_block
        inv_diag = inv_diag * (2.0 - diag_block * inv_diag)
        if dtype_flag == 1:
            inv_diag = inv_diag * (2.0 - diag_block * inv_diag)

        t_off = T_base + k * BLOCK_K + k_offsets[:, None] * BLOCK_K + k_offsets[None, :]
        tt_block = tl.load(Tt_ptr + t_off)
        w = tl.sum(tt_block * x_block[None, :], axis=1) * inv_diag

        tl.store(X_ptr + B_base + rows_k * stride_B, w)

        for m in range(0, k, BLOCK_M):
            rows_m = m + m_offsets
            rows_m_mask = rows_m < k
            U_tile = tl.load(
                L_ptr + L_base + rows_m[:, None] * stride_L + rows_k[None, :],
                mask=rows_m_mask[:, None],
                other=0.0,
            )
            head = tl.load(
                X_ptr + B_base + rows_m * stride_B, mask=rows_m_mask, other=0.0
            )
            head = head - tl.sum(U_tile * w[None, :], axis=1)
            tl.store(X_ptr + B_base + rows_m * stride_B, head, mask=rows_m_mask)


@libentry()
@triton.jit
def cholesky_solve_small_portable_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    N: tl.constexpr,
    nrhs: tl.constexpr,
    batch_stride_L,
    batch_stride_B,
    stride_L,
    stride_B,
    BLOCK_N: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
    dtype_flag: tl.constexpr,
    upper: tl.constexpr,
):
    """Portable small-N register-resident Cholesky solve.

    Mirrors cholesky_solve_small_gather_kernel but extracts each pivot row
    with a masked reduce instead of tl.gather. One warp keeps the reduce
    intra-warp, so the extra cost over a warp shuffle stays small.
    """
    batch_pid = program_id(0)

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    rows = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_RHS)
    cols_mask = cols < nrhs
    rows_mask = rows < N

    b = tl.load(
        B_ptr + B_base + rows[:, None] * stride_B + cols[None, :],
        mask=rows_mask[:, None] & cols_mask[None, :],
        other=0.0,
    )
    diag = tl.load(
        L_ptr + L_base + rows * stride_L + rows,
        mask=rows_mask,
        other=1.0,
    )
    inv_diag = 1.0 / diag
    inv_diag = inv_diag * (2.0 - diag * inv_diag)
    if dtype_flag == 1:
        inv_diag = inv_diag * (2.0 - diag * inv_diag)

    # Phase 1: solve L * Y = B or U^T * Y = B.
    w = b * inv_diag[:, None]
    for i in range(N):
        if upper:
            col_vals = tl.load(
                L_ptr + L_base + i * stride_L + rows,
                mask=(rows > i) & rows_mask,
                other=0.0,
            )
        else:
            col_vals = tl.load(
                L_ptr + L_base + rows * stride_L + i,
                mask=(rows > i) & rows_mask,
                other=0.0,
            )
        w_i = tl.sum(tl.where(rows[:, None] == i, w, 0.0), axis=0)
        w = w - (col_vals * inv_diag)[:, None] * w_i[None, :]

    # Phase 2: solve L^T * X = Y or U * X = Y.
    w = w * inv_diag[:, None]
    for i in range(N - 1, -1, -1):
        if upper:
            col_vals = tl.load(
                L_ptr + L_base + rows * stride_L + i,
                mask=rows < i,
                other=0.0,
            )
        else:
            col_vals = tl.load(
                L_ptr + L_base + i * stride_L + rows,
                mask=rows < i,
                other=0.0,
            )
        w_i = tl.sum(tl.where(rows[:, None] == i, w, 0.0), axis=0)
        w = w - (col_vals * inv_diag)[:, None] * w_i[None, :]

    tl.store(
        X_ptr + B_base + rows[:, None] * stride_B + cols[None, :],
        w,
        mask=rows_mask[:, None] & cols_mask[None, :],
    )


def _get_single_rhs_blocked_launch_config(dtype, N):
    """Return H20 winners for the blocked single-RHS gather kernels.

    Measured on the upper-orientation kernel (the zero-copy dispatch sends
    the column-major factors produced by torch.linalg.cholesky there), with
    the cross-warp barriers in place, by sweeping the raw JIT function --
    libentry caches per constexpr key only, so num_warps/num_stages sweeps
    through the wrapped kernel silently reuse the first binary. fp32 is
    latency-bound (one warp) until the N=256 panel updates reward a second
    warp; fp64's wider elements make four warps pay off at every size.
    """
    if dtype == torch.float64:
        return {
            "BLOCK_K": 32,
            "BLOCK_M": 32,
            "num_warps": 4,
            "num_stages": 1,
        }
    if N >= 256:
        return {
            "BLOCK_K": 32,
            "BLOCK_M": 128,
            "num_warps": 2,
            "num_stages": 1,
        }
    if N >= 128:
        return {
            "BLOCK_K": 32,
            "BLOCK_M": 32,
            "num_warps": 1,
            "num_stages": 1,
        }
    return {
        "BLOCK_K": 32,
        "BLOCK_M": 64,
        "num_warps": 1,
        "num_stages": 1,
    }


def _get_portable_single_rhs_blocked_launch_config(dtype, N):
    """Return configs for the portable blocked single-RHS solve kernels.

    Mirrors _get_single_rhs_blocked_launch_config; the diagonal-block solve
    is a precomputed-inverse matvec, so the warp count only affects the
    panel updates.
    """
    return _get_single_rhs_blocked_launch_config(dtype, N)


def _get_portable_blocked_launch_config(dtype, N, nrhs):
    """Return tile/warp configs for the portable blocked multi-RHS kernels.

    Measured on MetaX MC550. fp32 keeps narrow BLOCK_RHS=4 tiles for nrhs < 16
    and uses explicit tl.sum products: MetaX Triton rejects tl.dot operands
    whose non-batch dimensions are smaller than 16. Wider RHS tiles use
    BLOCK_RHS=16 and retain the faster tl.dot path. fp64 always uses explicit
    tl.sum products (no correct fp64 tl.dot lowering), where BLOCK_RHS=2 wins.
    """
    if dtype == torch.float64:
        return {
            "BLOCK_K": 16,
            "BLOCK_M": 32 if N >= 128 else 16,
            "BLOCK_RHS": 2,
            "num_warps": 4 if N >= 256 else 8,
            "num_stages": 1,
        }
    return {
        "BLOCK_K": 32,
        "BLOCK_M": 32,
        "BLOCK_RHS": 4 if nrhs < 16 else 16,
        "num_warps": 4,
        "num_stages": 1,
    }


def _get_portable_complex_blocked_launch_config(dtype, N, nrhs):
    """Return tile/warp configs for the portable blocked complex kernels.

    Measured on MetaX MC550. complex64 uses explicit tl.sum products with
    BLOCK_RHS=4 for nrhs < 16, avoiding MetaX Triton's minimum tl.dot width;
    wider inputs retain tl.dot with BLOCK_RHS=16. complex128 always uses
    tl.sum products with BLOCK_RHS=2, mirroring the fp64 choice.
    """
    if dtype == torch.complex128:
        return {
            "BLOCK_K": 16,
            "BLOCK_M": 32 if N >= 128 else 16,
            "BLOCK_RHS": 2,
            "num_warps": 4 if N >= 256 else 8,
            "num_stages": 1,
        }
    return {
        "BLOCK_K": 32,
        "BLOCK_M": 64 if N >= 256 else 32,
        "BLOCK_RHS": 4 if nrhs < 16 else 16,
        "num_warps": 4,
        "num_stages": 1,
    }


def _get_complex_single_rhs_launch_config(dtype, N):
    """Return configs for complex blocked single-RHS solves (gather path).

    All dtypes use 32-row diagonal blocks. This mirrors the measured real
    single-RHS winners and halves complex128's serial block/barrier count.
    N=128 keeps the measured 32-row winners. At N=256, larger panels reduce
    the number of serial Python-unrolled panel groups: 128 rows for complex64
    and 64 rows for the wider complex128 elements. complex128 at N >= 256
    takes the inverse-matvec path instead (see the dispatcher).
    """
    if dtype == torch.complex128:
        return {
            "BLOCK_K": 32,
            "BLOCK_M": 64 if N >= 256 else 32,
            "num_warps": 4,
            "num_stages": 1,
        }
    if N >= 256:
        block_m = 128
    elif N == 128:
        block_m = 32
    else:
        block_m = 64
    return {
        "BLOCK_K": 32,
        "BLOCK_M": block_m,
        "num_warps": 4 if N >= 256 else 2,
        "num_stages": 1,
    }


# ---------------------------------------------------------------------------
# Portable complex kernels (no tl.gather). Same precomputed-inverse design as
# the real portable kernels above: one parallel kernel inverts every BLOCK_K
# diagonal block of the effective lower factor, and the solve kernels apply
# each block with a single complex matmul/matvec. complex128 uses explicit
# tl.sum products because fp64 tl.dot has no correct lowering on the backends
# that need this path.
# ---------------------------------------------------------------------------


@libentry()
@triton.jit
def cholesky_solve_complex_invert_blocks_portable_kernel(
    L_ptr,
    T_ptr,
    Tt_ptr,
    N: tl.constexpr,
    batch_stride_L,
    stride_L_row,
    stride_L_col,
    BLOCK_K: tl.constexpr,
    upper: tl.constexpr,
    storage_conj: tl.constexpr,
):
    """Invert every BLOCK_K diagonal block of the effective lower factor.

    Stores T = (I + M)^-1 and its plain transpose in interleaved real/imag
    scratch ([batch, N, BLOCK_K, 2]); the backward solve conjugates Tt on
    load to form T^H. Row extraction uses a masked reduce instead of
    tl.gather so the kernel stays portable.
    """
    batch_pid = program_id(0)
    block_pid = program_id(1)
    L_base = batch_pid * batch_stride_L
    k = block_pid * BLOCK_K
    rows = tl.arange(0, BLOCK_K)

    diag_offset = L_base + (k + rows) * stride_L_row + (k + rows) * stride_L_col
    diag_real = tl.load(L_ptr + diag_offset)
    # A valid complex Cholesky factor has a positive real diagonal.
    inv_diag = 1.0 / diag_real

    t_real = tl.where(rows[:, None] == rows[None, :], 1.0, 0.0).to(
        L_ptr.dtype.element_ty
    )
    t_imag = tl.zeros([BLOCK_K, BLOCK_K], dtype=L_ptr.dtype.element_ty)

    for i in range(BLOCK_K):
        if upper:
            factor_offset = L_base + (k + i) * stride_L_row + (k + rows) * stride_L_col
        else:
            factor_offset = L_base + (k + rows) * stride_L_row + (k + i) * stride_L_col
        factor_real = tl.load(L_ptr + factor_offset, mask=rows > i, other=0.0)
        factor_imag = tl.load(L_ptr + factor_offset + 1, mask=rows > i, other=0.0)
        if storage_conj != upper:
            factor_imag = -factor_imag
        norm_real = factor_real * inv_diag
        norm_imag = factor_imag * inv_diag
        row_sel = rows[:, None] == i
        pivot_real = tl.sum(tl.where(row_sel, t_real, 0.0), axis=0)
        pivot_imag = tl.sum(tl.where(row_sel, t_imag, 0.0), axis=0)
        t_real -= norm_real[:, None] * pivot_real - norm_imag[:, None] * pivot_imag
        t_imag -= norm_real[:, None] * pivot_imag + norm_imag[:, None] * pivot_real

    t_base = batch_pid * N * BLOCK_K * 2 + k * BLOCK_K * 2
    t_off = t_base + rows[:, None] * (BLOCK_K * 2) + rows[None, :] * 2
    tl.store(T_ptr + t_off, t_real)
    tl.store(T_ptr + t_off + 1, t_imag)
    tt_off = t_base + rows[None, :] * (BLOCK_K * 2) + rows[:, None] * 2
    tl.store(Tt_ptr + tt_off, t_real)
    tl.store(Tt_ptr + tt_off + 1, t_imag)


@libentry()
@triton.jit
def cholesky_solve_complex_blocked_portable_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    T_ptr,
    Tt_ptr,
    N: tl.constexpr,
    nrhs: tl.constexpr,
    batch_stride_L,
    batch_stride_B,
    stride_L_row,
    stride_L_col,
    stride_B_row,
    stride_B_col,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
    upper: tl.constexpr,
    storage_conj: tl.constexpr,
    USE_SUM: tl.constexpr,
):
    """Portable blocked complex TRSM with precomputed diagonal inverses.

    Mirrors cholesky_solve_complex_blocked_kernel, but each diagonal block is
    applied with one complex matmul against the precomputed (I + M)^-1
    instead of the serial per-row tl.gather chain.
    """
    batch_pid = program_id(0)
    rhs_tile_pid = program_id(1)
    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    T_base = batch_pid * N * BLOCK_K * 2

    k_offsets = tl.arange(0, BLOCK_K)
    m_offsets = tl.arange(0, BLOCK_M)
    rhs_cols = rhs_tile_pid * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    rhs_mask = rhs_cols < nrhs

    # Forward blocked TRSM: L * Y = B, or U^H * Y = B.
    for k in range(0, N, BLOCK_K):
        rows_k = k + k_offsets
        if k > 0:
            tl.debug_barrier()

        y_offset = (
            B_base + rows_k[:, None] * stride_B_row + rhs_cols[None, :] * stride_B_col
        )
        if k == 0:
            y_real = tl.load(B_ptr + y_offset, mask=rhs_mask[None, :], other=0.0)
            y_imag = tl.load(B_ptr + y_offset + 1, mask=rhs_mask[None, :], other=0.0)
        else:
            y_real = tl.load(X_ptr + y_offset, mask=rhs_mask[None, :], other=0.0)
            y_imag = tl.load(X_ptr + y_offset + 1, mask=rhs_mask[None, :], other=0.0)

        diag_offset = L_base + rows_k * stride_L_row + rows_k * stride_L_col
        diag_real = tl.load(L_ptr + diag_offset)
        inv_diag = 1.0 / diag_real
        sv_real = y_real * inv_diag[:, None]
        sv_imag = y_imag * inv_diag[:, None]

        t_off = (
            T_base
            + k * BLOCK_K * 2
            + k_offsets[:, None] * (BLOCK_K * 2)
            + k_offsets[None, :] * 2
        )
        t_real = tl.load(T_ptr + t_off)
        t_imag = tl.load(T_ptr + t_off + 1)
        if USE_SUM:
            w_real = tl.sum(
                t_real[:, :, None] * sv_real[None, :, :]
                - t_imag[:, :, None] * sv_imag[None, :, :],
                axis=1,
            )
            w_imag = tl.sum(
                t_real[:, :, None] * sv_imag[None, :, :]
                + t_imag[:, :, None] * sv_real[None, :, :],
                axis=1,
            )
        else:
            w_real = tl.dot(t_real, sv_real, input_precision="ieee")
            w_real -= tl.dot(t_imag, sv_imag, input_precision="ieee")
            w_imag = tl.dot(t_real, sv_imag, input_precision="ieee")
            w_imag += tl.dot(t_imag, sv_real, input_precision="ieee")

        tl.store(X_ptr + y_offset, w_real, mask=rhs_mask[None, :])
        tl.store(X_ptr + y_offset + 1, w_imag, mask=rhs_mask[None, :])

        for m in range(k + BLOCK_K, N, BLOCK_M):
            rows_m = m + m_offsets
            rows_m_mask = rows_m < N
            if upper:
                tile_offset = (
                    L_base
                    + rows_m[:, None] * stride_L_col
                    + rows_k[None, :] * stride_L_row
                )
                tile_real = tl.load(
                    L_ptr + tile_offset, mask=rows_m_mask[:, None], other=0.0
                )
                tile_imag = tl.load(
                    L_ptr + tile_offset + 1, mask=rows_m_mask[:, None], other=0.0
                )
            else:
                tile_offset = (
                    L_base
                    + rows_k[:, None] * stride_L_col
                    + rows_m[None, :] * stride_L_row
                )
                tile_real = tl.trans(
                    tl.load(L_ptr + tile_offset, mask=rows_m_mask[None, :], other=0.0)
                )
                tile_imag = tl.trans(
                    tl.load(
                        L_ptr + tile_offset + 1, mask=rows_m_mask[None, :], other=0.0
                    )
                )
            if storage_conj != upper:
                tile_imag = -tile_imag

            tail_offset = (
                B_base
                + rows_m[:, None] * stride_B_row
                + rhs_cols[None, :] * stride_B_col
            )
            tail_mask = rows_m_mask[:, None] & rhs_mask[None, :]
            if k == 0:
                tail_real = tl.load(B_ptr + tail_offset, mask=tail_mask, other=0.0)
                tail_imag = tl.load(B_ptr + tail_offset + 1, mask=tail_mask, other=0.0)
            else:
                tail_real = tl.load(X_ptr + tail_offset, mask=tail_mask, other=0.0)
                tail_imag = tl.load(X_ptr + tail_offset + 1, mask=tail_mask, other=0.0)

            if USE_SUM:
                update_real = tl.sum(
                    tile_real[:, :, None] * w_real[None, :, :]
                    - tile_imag[:, :, None] * w_imag[None, :, :],
                    axis=1,
                )
                update_imag = tl.sum(
                    tile_real[:, :, None] * w_imag[None, :, :]
                    + tile_imag[:, :, None] * w_real[None, :, :],
                    axis=1,
                )
            else:
                update_real = tl.dot(tile_real, w_real, input_precision="ieee")
                update_real -= tl.dot(tile_imag, w_imag, input_precision="ieee")
                update_imag = tl.dot(tile_real, w_imag, input_precision="ieee")
                update_imag += tl.dot(tile_imag, w_real, input_precision="ieee")
            tail_real -= update_real
            tail_imag -= update_imag
            tl.store(X_ptr + tail_offset, tail_real, mask=tail_mask)
            tl.store(X_ptr + tail_offset + 1, tail_imag, mask=tail_mask)

    # Backward blocked TRSM: L^H * X = Y, or U * X = Y.
    for k in range(N - BLOCK_K, -1, -BLOCK_K):
        rows_k = k + k_offsets
        tl.debug_barrier()
        x_offset = (
            B_base + rows_k[:, None] * stride_B_row + rhs_cols[None, :] * stride_B_col
        )
        x_real = tl.load(X_ptr + x_offset, mask=rhs_mask[None, :], other=0.0)
        x_imag = tl.load(X_ptr + x_offset + 1, mask=rhs_mask[None, :], other=0.0)

        diag_offset = L_base + rows_k * stride_L_row + rows_k * stride_L_col
        diag_real = tl.load(L_ptr + diag_offset)
        inv_diag = 1.0 / diag_real

        t_off = (
            T_base
            + k * BLOCK_K * 2
            + k_offsets[:, None] * (BLOCK_K * 2)
            + k_offsets[None, :] * 2
        )
        tt_real = tl.load(Tt_ptr + t_off)
        tt_imag = tl.load(Tt_ptr + t_off + 1)
        # T^H = conj(T^T); Tt scratch holds the plain transpose.
        if USE_SUM:
            q_real = tl.sum(
                tt_real[:, :, None] * x_real[None, :, :]
                + tt_imag[:, :, None] * x_imag[None, :, :],
                axis=1,
            )
            q_imag = tl.sum(
                tt_real[:, :, None] * x_imag[None, :, :]
                - tt_imag[:, :, None] * x_real[None, :, :],
                axis=1,
            )
        else:
            q_real = tl.dot(tt_real, x_real, input_precision="ieee")
            q_real += tl.dot(tt_imag, x_imag, input_precision="ieee")
            q_imag = tl.dot(tt_real, x_imag, input_precision="ieee")
            q_imag -= tl.dot(tt_imag, x_real, input_precision="ieee")
        w_real = q_real * inv_diag[:, None]
        w_imag = q_imag * inv_diag[:, None]

        tl.store(X_ptr + x_offset, w_real, mask=rhs_mask[None, :])
        tl.store(X_ptr + x_offset + 1, w_imag, mask=rhs_mask[None, :])

        for m in range(0, k, BLOCK_M):
            rows_m = m + m_offsets
            rows_m_mask = rows_m < k
            if upper:
                tile_offset = (
                    L_base
                    + rows_k[:, None] * stride_L_col
                    + rows_m[None, :] * stride_L_row
                )
                tile_real = tl.trans(
                    tl.load(L_ptr + tile_offset, mask=rows_m_mask[None, :], other=0.0)
                )
                tile_imag = tl.trans(
                    tl.load(
                        L_ptr + tile_offset + 1, mask=rows_m_mask[None, :], other=0.0
                    )
                )
            else:
                tile_offset = (
                    L_base
                    + rows_m[:, None] * stride_L_col
                    + rows_k[None, :] * stride_L_row
                )
                tile_real = tl.load(
                    L_ptr + tile_offset, mask=rows_m_mask[:, None], other=0.0
                )
                tile_imag = tl.load(
                    L_ptr + tile_offset + 1, mask=rows_m_mask[:, None], other=0.0
                )
            if storage_conj == upper:
                tile_imag = -tile_imag

            head_offset = (
                B_base
                + rows_m[:, None] * stride_B_row
                + rhs_cols[None, :] * stride_B_col
            )
            head_mask = rows_m_mask[:, None] & rhs_mask[None, :]
            head_real = tl.load(X_ptr + head_offset, mask=head_mask, other=0.0)
            head_imag = tl.load(X_ptr + head_offset + 1, mask=head_mask, other=0.0)
            if USE_SUM:
                update_real = tl.sum(
                    tile_real[:, :, None] * w_real[None, :, :]
                    - tile_imag[:, :, None] * w_imag[None, :, :],
                    axis=1,
                )
                update_imag = tl.sum(
                    tile_real[:, :, None] * w_imag[None, :, :]
                    + tile_imag[:, :, None] * w_real[None, :, :],
                    axis=1,
                )
            else:
                update_real = tl.dot(tile_real, w_real, input_precision="ieee")
                update_real -= tl.dot(tile_imag, w_imag, input_precision="ieee")
                update_imag = tl.dot(tile_real, w_imag, input_precision="ieee")
                update_imag += tl.dot(tile_imag, w_real, input_precision="ieee")
            head_real -= update_real
            head_imag -= update_imag
            tl.store(X_ptr + head_offset, head_real, mask=head_mask)
            tl.store(X_ptr + head_offset + 1, head_imag, mask=head_mask)


@libentry()
@triton.jit
def cholesky_solve_complex_single_rhs_blocked_portable_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    T_ptr,
    Tt_ptr,
    N: tl.constexpr,
    batch_stride_L,
    batch_stride_B,
    stride_L_row,
    stride_L_col,
    stride_B_row,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    upper: tl.constexpr,
    storage_conj: tl.constexpr,
):
    """Portable blocked complex single-RHS solve (tl.sum matvecs)."""
    batch_pid = program_id(0)
    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    T_base = batch_pid * N * BLOCK_K * 2
    k_offsets = tl.arange(0, BLOCK_K)
    m_offsets = tl.arange(0, BLOCK_M)

    # Forward blocked TRSV: L * Y = B, or U^H * Y = B.
    for k in range(0, N, BLOCK_K):
        rows_k = k + k_offsets
        if k > 0:
            tl.debug_barrier()
        y_offset = B_base + rows_k * stride_B_row
        if k == 0:
            y_real = tl.load(B_ptr + y_offset)
            y_imag = tl.load(B_ptr + y_offset + 1)
        else:
            y_real = tl.load(X_ptr + y_offset)
            y_imag = tl.load(X_ptr + y_offset + 1)

        diag_offset = L_base + rows_k * stride_L_row + rows_k * stride_L_col
        diag_real = tl.load(L_ptr + diag_offset)
        inv_diag = 1.0 / diag_real
        sv_real = y_real * inv_diag
        sv_imag = y_imag * inv_diag

        t_off = (
            T_base
            + k * BLOCK_K * 2
            + k_offsets[:, None] * (BLOCK_K * 2)
            + k_offsets[None, :] * 2
        )
        t_real = tl.load(T_ptr + t_off)
        t_imag = tl.load(T_ptr + t_off + 1)
        w_real = tl.sum(t_real * sv_real[None, :] - t_imag * sv_imag[None, :], axis=1)
        w_imag = tl.sum(t_real * sv_imag[None, :] + t_imag * sv_real[None, :], axis=1)

        tl.store(X_ptr + y_offset, w_real)
        tl.store(X_ptr + y_offset + 1, w_imag)

        for m in range(k + BLOCK_K, N, BLOCK_M):
            rows_m = m + m_offsets
            rows_m_mask = rows_m < N
            if upper:
                tile_offset = (
                    L_base
                    + rows_m[:, None] * stride_L_col
                    + rows_k[None, :] * stride_L_row
                )
            else:
                tile_offset = (
                    L_base
                    + rows_m[:, None] * stride_L_row
                    + rows_k[None, :] * stride_L_col
                )
            tile_real = tl.load(
                L_ptr + tile_offset, mask=rows_m_mask[:, None], other=0.0
            )
            tile_imag = tl.load(
                L_ptr + tile_offset + 1, mask=rows_m_mask[:, None], other=0.0
            )
            if storage_conj != upper:
                tile_imag = -tile_imag
            update_real = tl.sum(
                tile_real * w_real[None, :] - tile_imag * w_imag[None, :], axis=1
            )
            update_imag = tl.sum(
                tile_real * w_imag[None, :] + tile_imag * w_real[None, :], axis=1
            )
            tail_offset = B_base + rows_m * stride_B_row
            if k == 0:
                tail_real = tl.load(B_ptr + tail_offset, mask=rows_m_mask, other=0.0)
                tail_imag = tl.load(
                    B_ptr + tail_offset + 1, mask=rows_m_mask, other=0.0
                )
            else:
                tail_real = tl.load(X_ptr + tail_offset, mask=rows_m_mask, other=0.0)
                tail_imag = tl.load(
                    X_ptr + tail_offset + 1, mask=rows_m_mask, other=0.0
                )
            tl.store(X_ptr + tail_offset, tail_real - update_real, mask=rows_m_mask)
            tl.store(X_ptr + tail_offset + 1, tail_imag - update_imag, mask=rows_m_mask)

    # Backward blocked TRSV: L^H * X = Y, or U * X = Y.
    for k in range(N - BLOCK_K, -1, -BLOCK_K):
        rows_k = k + k_offsets
        tl.debug_barrier()
        x_offset = B_base + rows_k * stride_B_row
        x_real = tl.load(X_ptr + x_offset)
        x_imag = tl.load(X_ptr + x_offset + 1)

        diag_offset = L_base + rows_k * stride_L_row + rows_k * stride_L_col
        diag_real = tl.load(L_ptr + diag_offset)
        inv_diag = 1.0 / diag_real

        t_off = (
            T_base
            + k * BLOCK_K * 2
            + k_offsets[:, None] * (BLOCK_K * 2)
            + k_offsets[None, :] * 2
        )
        tt_real = tl.load(Tt_ptr + t_off)
        tt_imag = tl.load(Tt_ptr + t_off + 1)
        # T^H = conj(T^T); Tt scratch holds the plain transpose.
        q_real = tl.sum(tt_real * x_real[None, :] + tt_imag * x_imag[None, :], axis=1)
        q_imag = tl.sum(tt_real * x_imag[None, :] - tt_imag * x_real[None, :], axis=1)
        w_real = q_real * inv_diag
        w_imag = q_imag * inv_diag

        tl.store(X_ptr + x_offset, w_real)
        tl.store(X_ptr + x_offset + 1, w_imag)

        for m in range(0, k, BLOCK_M):
            rows_m = m + m_offsets
            rows_m_mask = rows_m < k
            if upper:
                tile_offset = (
                    L_base
                    + rows_k[:, None] * stride_L_col
                    + rows_m[None, :] * stride_L_row
                )
            else:
                tile_offset = (
                    L_base
                    + rows_k[:, None] * stride_L_row
                    + rows_m[None, :] * stride_L_col
                )
            tile_real = tl.load(
                L_ptr + tile_offset, mask=rows_m_mask[None, :], other=0.0
            )
            tile_imag = tl.load(
                L_ptr + tile_offset + 1, mask=rows_m_mask[None, :], other=0.0
            )
            if storage_conj == upper:
                tile_imag = -tile_imag
            update_real = tl.sum(
                tile_real * w_real[:, None] - tile_imag * w_imag[:, None], axis=0
            )
            update_imag = tl.sum(
                tile_real * w_imag[:, None] + tile_imag * w_real[:, None], axis=0
            )
            head_offset = B_base + rows_m * stride_B_row
            head_real = tl.load(X_ptr + head_offset, mask=rows_m_mask, other=0.0)
            head_imag = tl.load(X_ptr + head_offset + 1, mask=rows_m_mask, other=0.0)
            tl.store(X_ptr + head_offset, head_real - update_real, mask=rows_m_mask)
            tl.store(X_ptr + head_offset + 1, head_imag - update_imag, mask=rows_m_mask)


@libentry()
@triton.jit
def cholesky_solve_complex_small_portable_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    N: tl.constexpr,
    nrhs: tl.constexpr,
    batch_stride_L,
    batch_stride_B,
    stride_L_row,
    stride_L_col,
    stride_B_row,
    stride_B_col,
    BLOCK_N: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
    upper: tl.constexpr,
    storage_conj: tl.constexpr,
):
    """Portable register-resident complex solve for small N/RHS cases.

    Mirrors cholesky_solve_complex_small_gather_kernel with masked-reduce
    pivot extraction instead of tl.gather.
    """
    batch_pid = program_id(0)
    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B

    rows = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_RHS)
    rows_mask = rows < N
    cols_mask = cols < nrhs
    value_mask = rows_mask[:, None] & cols_mask[None, :]

    b_offset = B_base + rows[:, None] * stride_B_row + cols[None, :] * stride_B_col
    w_real = tl.load(B_ptr + b_offset, mask=value_mask, other=0.0)
    w_imag = tl.load(B_ptr + b_offset + 1, mask=value_mask, other=0.0)

    diag_offset = L_base + rows * stride_L_row + rows * stride_L_col
    diag_real = tl.load(L_ptr + diag_offset, mask=rows_mask, other=1.0)
    # A valid complex Cholesky factor has a positive real diagonal.
    inv_diag = 1.0 / diag_real

    # Phase 1: solve L * Y = B, or U^H * Y = B.
    scaled_real = w_real * inv_diag[:, None]
    scaled_imag = w_imag * inv_diag[:, None]

    for i in range(N):
        if upper:
            factor_offset = L_base + i * stride_L_row + rows * stride_L_col
        else:
            factor_offset = L_base + rows * stride_L_row + i * stride_L_col
        active = (rows > i) & rows_mask
        factor_real = tl.load(L_ptr + factor_offset, mask=active, other=0.0)
        factor_imag = tl.load(L_ptr + factor_offset + 1, mask=active, other=0.0)
        if storage_conj != upper:
            factor_imag = -factor_imag

        normalized_real = factor_real * inv_diag
        normalized_imag = factor_imag * inv_diag
        row_sel = rows[:, None] == i
        pivot_real = tl.sum(tl.where(row_sel, scaled_real, 0.0), axis=0)
        pivot_imag = tl.sum(tl.where(row_sel, scaled_imag, 0.0), axis=0)
        scaled_real -= (
            normalized_real[:, None] * pivot_real[None, :]
            - normalized_imag[:, None] * pivot_imag[None, :]
        )
        scaled_imag -= (
            normalized_real[:, None] * pivot_imag[None, :]
            + normalized_imag[:, None] * pivot_real[None, :]
        )

    # Phase 2: solve L^H * X = Y, or U * X = Y.
    out_real = scaled_real * inv_diag[:, None]
    out_imag = scaled_imag * inv_diag[:, None]

    for i in range(N - 1, -1, -1):
        if upper:
            factor_offset = L_base + rows * stride_L_row + i * stride_L_col
        else:
            factor_offset = L_base + i * stride_L_row + rows * stride_L_col
        active = (rows < i) & rows_mask
        factor_real = tl.load(L_ptr + factor_offset, mask=active, other=0.0)
        factor_imag = tl.load(L_ptr + factor_offset + 1, mask=active, other=0.0)
        if storage_conj == upper:
            factor_imag = -factor_imag

        normalized_real = factor_real * inv_diag
        normalized_imag = factor_imag * inv_diag
        row_sel = rows[:, None] == i
        pivot_real = tl.sum(tl.where(row_sel, out_real, 0.0), axis=0)
        pivot_imag = tl.sum(tl.where(row_sel, out_imag, 0.0), axis=0)
        out_real -= (
            normalized_real[:, None] * pivot_real[None, :]
            - normalized_imag[:, None] * pivot_imag[None, :]
        )
        out_imag -= (
            normalized_real[:, None] * pivot_imag[None, :]
            + normalized_imag[:, None] * pivot_real[None, :]
        )

    tl.store(X_ptr + b_offset, out_real, mask=value_mask)
    tl.store(X_ptr + b_offset + 1, out_imag, mask=value_mask)


def _cholesky_solve_complex(
    B,
    L,
    upper,
    batch_shape,
    N,
    nrhs,
    X=None,
):
    """Launch layout-aware complex64/complex128 specialized kernels."""
    # view_as_real rejects lazy-conjugate tensors. Rebuild the same view from
    # its unconjugated base storage and carry the logical conjugation into the
    # kernels. Calling L.conj() here is not safe under the global FlagGems
    # registration: it dispatches to the physical _conj implementation and
    # adds resolve/conjugate kernels to every upper solve.
    input_storage_conj = L.is_conj()
    if input_storage_conj:
        base = L._base
        if base is not None and not base.is_conj():
            L = base.as_strided(L.shape, L.stride(), L.storage_offset())
        else:
            # Rare fallback for conjugated tensors without an unconjugated
            # base view. The materialized tensor already contains the logical
            # values, so factor loads must no longer conjugate them.
            L = L.resolve_conj()
            input_storage_conj = False
    if B.is_conj():
        B = B.resolve_conj()

    # Complex layout normalization mirrors the real path, with one additional
    # bit of metadata. For a column-major lower factor L, L.mT is a row-major
    # view containing L^T, whereas the corresponding upper factor is L^H.
    # Kernels therefore flip the triangular orientation and conjugate factor
    # loads on the fly. This avoids the F->C copy without sacrificing coalesced
    # row-major panel loads.
    if L.is_contiguous():
        effective_upper = upper
        storage_conj = input_storage_conj
    elif L.mT.is_contiguous():
        L = L.mT
        effective_upper = not upper
        storage_conj = not input_storage_conj
    else:
        L = L.contiguous()
        effective_upper = upper
        storage_conj = input_storage_conj
    if not B.is_contiguous():
        B = B.contiguous()
    if X is None:
        X = torch.empty_like(B)

    batch_size = 1
    for dim in batch_shape:
        batch_size *= dim

    L_real = torch.view_as_real(L).reshape(-1, N, N, 2)
    B_real = torch.view_as_real(B).reshape(-1, N, nrhs, 2)
    X_real = torch.view_as_real(X).reshape(-1, N, nrhs, 2)

    with torch.no_grad():
        if (N <= 32 and nrhs <= 8) or (N < 64 and nrhs == 1):
            # Portable register-resident path: masked-reduce pivots instead
            # of the tl.gather used by the optimized small kernel.
            block_n = triton.next_power_of_2(N)
            block_rhs = triton.next_power_of_2(nrhs)
            cholesky_solve_complex_small_portable_kernel[(batch_size,)](
                L_real,
                B_real,
                X_real,
                N,
                nrhs,
                L_real.stride(0),
                B_real.stride(0),
                L_real.stride(1),
                L_real.stride(2),
                B_real.stride(1),
                B_real.stride(2),
                BLOCK_N=block_n,
                BLOCK_RHS=block_rhs,
                upper=effective_upper,
                storage_conj=storage_conj,
                num_warps=1,
                num_stages=1,
            )
        elif N >= 64 and N % 32 == 0 and nrhs >= 4:
            # Portable blocked path: precompute the diagonal-block inverses
            # (parallel masked-reduce kernel), then apply each block with a
            # single complex matmul.
            is_double = B.dtype == torch.complex128
            use_sum = is_double or nrhs < 16
            config = _get_portable_complex_blocked_launch_config(B.dtype, N, nrhs)
            block_k = config["BLOCK_K"]
            T_scratch = torch.empty(
                (2, batch_size, N, block_k, 2), dtype=B_real.dtype, device=B.device
            )
            cholesky_solve_complex_invert_blocks_portable_kernel[
                (batch_size, N // block_k)
            ](
                L_real,
                T_scratch[0],
                T_scratch[1],
                N,
                L_real.stride(0),
                L_real.stride(1),
                L_real.stride(2),
                BLOCK_K=block_k,
                upper=effective_upper,
                storage_conj=storage_conj,
                num_warps=1,
                num_stages=1,
            )
            grid = (batch_size, triton.cdiv(nrhs, config["BLOCK_RHS"]))
            cholesky_solve_complex_blocked_portable_kernel[grid](
                L_real,
                B_real,
                X_real,
                T_scratch[0],
                T_scratch[1],
                N,
                nrhs,
                L_real.stride(0),
                B_real.stride(0),
                L_real.stride(1),
                L_real.stride(2),
                B_real.stride(1),
                B_real.stride(2),
                upper=effective_upper,
                storage_conj=storage_conj,
                USE_SUM=use_sum,
                num_warps=config["num_warps"],
                num_stages=1,
                BLOCK_K=config["BLOCK_K"],
                BLOCK_M=config["BLOCK_M"],
                BLOCK_RHS=config["BLOCK_RHS"],
            )
        elif nrhs == 1 and N >= 64 and N % 32 == 0:
            # Portable blocked single-RHS path (tl.sum matvecs).
            config = _get_complex_single_rhs_launch_config(B.dtype, N)
            block_k = config["BLOCK_K"]
            T_scratch = torch.empty(
                (2, batch_size, N, block_k, 2), dtype=B_real.dtype, device=B.device
            )
            cholesky_solve_complex_invert_blocks_portable_kernel[
                (batch_size, N // block_k)
            ](
                L_real,
                T_scratch[0],
                T_scratch[1],
                N,
                L_real.stride(0),
                L_real.stride(1),
                L_real.stride(2),
                BLOCK_K=block_k,
                upper=effective_upper,
                storage_conj=storage_conj,
                num_warps=1,
                num_stages=1,
            )
            cholesky_solve_complex_single_rhs_blocked_portable_kernel[(batch_size,)](
                L_real,
                B_real,
                X_real,
                T_scratch[0],
                T_scratch[1],
                N,
                L_real.stride(0),
                B_real.stride(0),
                L_real.stride(1),
                L_real.stride(2),
                B_real.stride(1),
                BLOCK_K=config["BLOCK_K"],
                BLOCK_M=config["BLOCK_M"],
                upper=effective_upper,
                storage_conj=storage_conj,
                num_warps=config["num_warps"],
                num_stages=1,
            )
        else:
            # Keep a scalar-substitution fallback for Triton backends that
            # cannot lower the tl.gather used by the optimized kernels.
            block_rhs = 4 if B.dtype == torch.complex64 else 2
            grid = (batch_size, triton.cdiv(nrhs, block_rhs))
            cholesky_solve_complex_kernel[grid](
                L_real,
                B_real,
                X_real,
                N,
                nrhs,
                L_real.stride(0),
                B_real.stride(0),
                L_real.stride(1),
                L_real.stride(2),
                B_real.stride(1),
                B_real.stride(2),
                BLOCK_RHS=block_rhs,
                upper=effective_upper,
                storage_conj=storage_conj,
                num_warps=1,
                num_stages=1,
            )
    return X


def cholesky_solve(B, L, upper=False, *, _out=None):
    """Solves a system of linear equations with a positive-definite
    matrix using the Cholesky factorization.

    Computes X such that A @ X = B, where A = L @ L^H (or A = U^H @ U if
    upper=True) and L (or U) is the Cholesky factor of A. For real inputs,
    the Hermitian transpose is the ordinary transpose.

    Args:
        B: right-hand side tensor of shape (*, N, nrhs)
        L: Cholesky factor of shape (*, N, N), lower-triangular unless upper=True
        upper: if True, the Cholesky factor is upper-triangular

    Returns:
        X: solution tensor of shape (*, N, nrhs)
    """
    logger.debug("GEMS_METAX CHOLESKY_SOLVE")
    assert L.dtype in (
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ), "cholesky_solve only supports float32, float64, complex64 and complex128"
    assert B.dtype == L.dtype, "B and L must have the same dtype"
    if B.device != L.device:
        raise ValueError("B and L must be on the same device")

    if B.numel() == 0 or L.numel() == 0:
        return B

    L_shape = L.shape
    B_shape = B.shape

    if len(L_shape) < 2:
        raise ValueError("L must be at least 2D")
    if len(B_shape) < 2:
        raise ValueError("B must be at least 2D")

    N = L_shape[-1]
    if L_shape[-2] != N:
        raise ValueError("L must be a square matrix")
    if B_shape[-2] != N:
        raise ValueError(
            f"B's second-to-last dimension must equal L's last dimension, "
            f"got {B_shape[-2]} != {N}"
        )

    nrhs = B_shape[-1]

    # Fast path: when B and L already share their batch dims, skip the
    # torch.broadcast_shapes + expand calls. Each costs several microseconds of
    # host time, which dominates end-to-end latency for the small systems where
    # the GPU kernel itself is tiny. Only broadcast when the batch dims differ.
    B_batch = B_shape[:-2]
    L_batch = L_shape[:-2]
    if B_batch == L_batch:
        batch_shape = B_batch
    else:
        try:
            batch_shape = torch.broadcast_shapes(B_batch, L_batch)
        except RuntimeError as exc:
            raise ValueError(
                f"B and L batch dimensions are not broadcastable: "
                f"{B_batch} vs {L_batch}"
            ) from exc

        L = L.expand(batch_shape + L_shape[-2:])
        B = B.expand(batch_shape + B_shape[-2:])

    if B.is_complex():
        return _cholesky_solve_complex(
            B,
            L,
            upper,
            batch_shape,
            N,
            nrhs,
            X=_out,
        )

    # Zero-copy layout normalization. Every kernel pair exists in both
    # orientations, and solving with a lower factor L is exactly solving with
    # the upper factor U = L^T (L L^T = U^T U), so a transposed *view* flips
    # orientation for free. This must never fall back to a materializing
    # .contiguous() copy for the common layouts: the scored benchmark times
    # this wrapper under the global use_gems() context, where the copy itself
    # dispatches to flag_gems kernels and costs far more than the solve.
    # torch.linalg.cholesky returns a column-major factor, which lands in the
    # transposed-view branch below.
    if L.is_contiguous():
        effective_upper = upper
    elif L.mT.is_contiguous():
        L = L.mT
        effective_upper = not upper
    else:
        L = L.contiguous()
        effective_upper = upper
    if not B.is_contiguous():
        B = B.contiguous()
    X = torch.empty_like(B) if _out is None else _out

    batch_size = 1
    for dim in batch_shape:
        batch_size *= dim

    L_kernel = L.reshape(-1, N, N)
    B_kernel = B.reshape(-1, N, nrhs)
    X_kernel = X.reshape(-1, N, nrhs)

    stride_L = L_kernel.stride(1)
    stride_B = B_kernel.stride(1)
    batch_stride_L = L_kernel.stride(0)
    batch_stride_B = B_kernel.stride(0)

    dtype_flag = 0 if B.dtype == torch.float32 else 1

    with torch.no_grad():
        if _can_use_blocked_path(N, nrhs):
            # num_stages=1: the pipeliner mishandles the cross-warp
            # debug_barrier handoff in these kernels on some backends.
            use_sum = dtype_flag == 1 or nrhs < 16
            tile = _get_portable_blocked_launch_config(B.dtype, N, nrhs)
            block_k = tile["BLOCK_K"]
            T_scratch = torch.empty(
                (2, batch_size, N, block_k), dtype=B.dtype, device=B.device
            )
            cholesky_solve_invert_blocks_portable_kernel[(batch_size, N // block_k)](
                L_kernel,
                T_scratch[0],
                T_scratch[1],
                N,
                batch_stride_L,
                stride_L,
                BLOCK_K=block_k,
                upper=effective_upper,
                dtype_flag=dtype_flag,
                num_warps=1,
                num_stages=1,
            )
            grid = (batch_size, triton.cdiv(nrhs, tile["BLOCK_RHS"]))
            if dtype_flag == 1:
                # fp64 has no correct tl.dot lowering on portable-path
                # backends; use the tl.sum-dedicated kernels instead.
                solve_kernel = (
                    cholesky_solve_blocked_upper_portable_fp64_kernel
                    if effective_upper
                    else cholesky_solve_blocked_lower_portable_fp64_kernel
                )
            else:
                solve_kernel = (
                    cholesky_solve_blocked_upper_portable_kernel
                    if effective_upper
                    else cholesky_solve_blocked_lower_portable_kernel
                )
            solve_args = {
                "BLOCK_K": tile["BLOCK_K"],
                "BLOCK_M": tile["BLOCK_M"],
                "BLOCK_RHS": tile["BLOCK_RHS"],
                "num_warps": tile["num_warps"],
                "num_stages": tile["num_stages"],
            }
            if dtype_flag == 0:
                solve_args["dtype_flag"] = dtype_flag
                solve_args["USE_SUM"] = use_sum
            solve_kernel[grid](
                L_kernel,
                B_kernel,
                X_kernel,
                T_scratch[0],
                T_scratch[1],
                N,
                nrhs,
                batch_stride_L,
                batch_stride_B,
                stride_L,
                stride_B,
                **solve_args,
            )
        elif _can_use_blocked_single_rhs_path(N, nrhs):
            single_rhs_config = _get_portable_single_rhs_blocked_launch_config(
                B.dtype, N
            )
            block_k = single_rhs_config["BLOCK_K"]
            T_scratch = torch.empty(
                (2, batch_size, N, block_k), dtype=B.dtype, device=B.device
            )
            cholesky_solve_invert_blocks_portable_kernel[(batch_size, N // block_k)](
                L_kernel,
                T_scratch[0],
                T_scratch[1],
                N,
                batch_stride_L,
                stride_L,
                BLOCK_K=block_k,
                upper=effective_upper,
                dtype_flag=dtype_flag,
                num_warps=1,
                num_stages=1,
            )
            solve_kernel = (
                cholesky_solve_single_rhs_blocked_upper_portable_kernel
                if effective_upper
                else cholesky_solve_single_rhs_blocked_lower_portable_kernel
            )
            solve_kernel[(batch_size,)](
                L_kernel,
                B_kernel,
                X_kernel,
                T_scratch[0],
                T_scratch[1],
                N,
                batch_stride_L,
                batch_stride_B,
                stride_L,
                stride_B,
                dtype_flag=dtype_flag,
                **single_rhs_config,
            )
        elif _can_use_small_gather_path(N, nrhs):
            block_n = triton.next_power_of_2(N)
            block_rhs = triton.next_power_of_2(nrhs)
            cholesky_solve_small_portable_kernel[(batch_size,)](
                L_kernel,
                B_kernel,
                X_kernel,
                N,
                nrhs,
                batch_stride_L,
                batch_stride_B,
                stride_L,
                stride_B,
                BLOCK_N=block_n,
                BLOCK_RHS=block_rhs,
                dtype_flag=dtype_flag,
                upper=effective_upper,
                num_warps=1,
                num_stages=1,
            )
        elif nrhs == 1:
            cholesky_solve_single_rhs_kernel[(batch_size,)](
                L_kernel,
                B_kernel,
                X_kernel,
                N,
                batch_stride_L,
                batch_stride_B,
                stride_L,
                stride_B,
                dtype_flag=dtype_flag,
                upper=effective_upper,
            )
        else:
            grid = lambda meta: (
                batch_size,
                triton.cdiv(nrhs, meta["BLOCK_RHS"]),
            )
            cholesky_solve_kernel[grid](
                L_kernel,
                B_kernel,
                X_kernel,
                N,
                nrhs,
                batch_stride_L,
                batch_stride_B,
                stride_L,
                stride_B,
                dtype_flag=dtype_flag,
                upper=effective_upper,
            )
    return X


def cholesky_solve_out(B, L, upper=False, *, out):
    """Out variant with direct writes for the common compatible case."""
    logger.debug("GEMS_METAX CHOLESKY_SOLVE_OUT")
    _check_cholesky_solve_out(B, out)
    if _can_write_cholesky_solve_out_direct(B, L, out):
        return cholesky_solve(B, L, upper=upper, _out=out)
    result = cholesky_solve(B, L, upper=upper)
    return _copy_cholesky_solve_out(result, out)


__all__ = ["cholesky_solve", "cholesky_solve_out"]
