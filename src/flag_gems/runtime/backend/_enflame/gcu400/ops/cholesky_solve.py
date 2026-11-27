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
import warnings

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils.triton_lang_extension import program_id

logger = logging.getLogger(__name__)

# gcu400 lowers ``tl.gather`` only for a single warp; with more warps the
# triton-gpu -> triton-gcu pass marks the op illegal and the compiler dies
# (usually with a bare segfault and no diagnostic).  Every kernel in this file
# gathers (per-row pivots or rank-1 updates), so all launches use one warp.
# The H20-tuned launch configs below are therefore pinned to a single warp.


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
    BLOCK_RHS: tl.constexpr,
    dtype_flag: tl.constexpr,
    upper: tl.constexpr,
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


@libentry()
@triton.jit
def cholesky_solve_complex_small_gather_kernel(
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
    """Register-resident complex solve for small N/RHS cases."""
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
    diag_real = tl.load(
        L_ptr + diag_offset,
        mask=rows_mask,
        other=1.0,
    )
    # A valid complex Cholesky factor has a positive real diagonal. Exploit
    # that contract instead of paying for a general complex division.
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
        factor_real = tl.load(
            L_ptr + factor_offset,
            mask=active,
            other=0.0,
        )
        factor_imag = tl.load(
            L_ptr + factor_offset + 1,
            mask=active,
            other=0.0,
        )
        if storage_conj != upper:
            factor_imag = -factor_imag

        normalized_real = factor_real * inv_diag
        normalized_imag = factor_imag * inv_diag
        pivot_real = tl.gather(
            scaled_real,
            tl.full([1, BLOCK_RHS], i, tl.int32),
            0,
        )
        pivot_imag = tl.gather(
            scaled_imag,
            tl.full([1, BLOCK_RHS], i, tl.int32),
            0,
        )
        product_real = (
            normalized_real[:, None] * pivot_real
            - normalized_imag[:, None] * pivot_imag
        )
        product_imag = (
            normalized_real[:, None] * pivot_imag
            + normalized_imag[:, None] * pivot_real
        )
        scaled_real -= product_real
        scaled_imag -= product_imag

    # Phase 2: solve L^H * X = Y, or U * X = Y.
    out_real = scaled_real * inv_diag[:, None]
    out_imag = scaled_imag * inv_diag[:, None]

    for i in range(N - 1, -1, -1):
        if upper:
            factor_offset = L_base + rows * stride_L_row + i * stride_L_col
        else:
            factor_offset = L_base + i * stride_L_row + rows * stride_L_col
        active = (rows < i) & rows_mask
        factor_real = tl.load(
            L_ptr + factor_offset,
            mask=active,
            other=0.0,
        )
        factor_imag = tl.load(
            L_ptr + factor_offset + 1,
            mask=active,
            other=0.0,
        )
        if storage_conj == upper:
            factor_imag = -factor_imag

        normalized_real = factor_real * inv_diag
        normalized_imag = factor_imag * inv_diag
        pivot_real = tl.gather(
            out_real,
            tl.full([1, BLOCK_RHS], i, tl.int32),
            0,
        )
        pivot_imag = tl.gather(
            out_imag,
            tl.full([1, BLOCK_RHS], i, tl.int32),
            0,
        )
        product_real = (
            normalized_real[:, None] * pivot_real
            - normalized_imag[:, None] * pivot_imag
        )
        product_imag = (
            normalized_real[:, None] * pivot_imag
            + normalized_imag[:, None] * pivot_real
        )
        out_real -= product_real
        out_imag -= product_imag

    tl.store(X_ptr + b_offset, out_real, mask=value_mask)
    tl.store(X_ptr + b_offset + 1, out_imag, mask=value_mask)


@libentry()
@triton.jit
def cholesky_solve_complex_blocked_kernel(
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
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
    upper: tl.constexpr,
    storage_conj: tl.constexpr,
    IS_DOUBLE: tl.constexpr,
):
    """Blocked complex TRSM for N >= 64 and multiple right-hand sides.

    The diagonal blocks use the pre-scaled gather formulation. Far-panel
    updates are decomposed into four real matrix products, enabling tensor
    cores for complex64 while retaining native FP64 dot products for
    complex128. N must be divisible by BLOCK_K; panel edges are masked.
    """
    batch_pid = program_id(0)
    rhs_tile_pid = program_id(1)
    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B

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
        w_real = y_real * inv_diag[:, None]
        w_imag = y_imag * inv_diag[:, None]

        for i in range(BLOCK_K):
            if upper:
                factor_offset = L_base + (k + i) * stride_L_row + rows_k * stride_L_col
            else:
                factor_offset = L_base + rows_k * stride_L_row + (k + i) * stride_L_col
            factor_real = tl.load(
                L_ptr + factor_offset,
                mask=k_offsets > i,
                other=0.0,
            )
            factor_imag = tl.load(
                L_ptr + factor_offset + 1,
                mask=k_offsets > i,
                other=0.0,
            )
            if storage_conj != upper:
                factor_imag = -factor_imag
            norm_real = factor_real * inv_diag
            norm_imag = factor_imag * inv_diag
            pivot_real = tl.gather(w_real, tl.full([1, BLOCK_RHS], i, tl.int32), 0)
            pivot_imag = tl.gather(w_imag, tl.full([1, BLOCK_RHS], i, tl.int32), 0)
            w_real -= norm_real[:, None] * pivot_real - norm_imag[:, None] * pivot_imag
            w_imag -= norm_real[:, None] * pivot_imag + norm_imag[:, None] * pivot_real

        tl.store(X_ptr + y_offset, w_real, mask=rhs_mask[None, :])
        tl.store(X_ptr + y_offset + 1, w_imag, mask=rhs_mask[None, :])

        for m in range(k + BLOCK_K, N, BLOCK_M):
            rows_m = m + m_offsets
            rows_m_mask = rows_m < N
            if upper:
                # Direct [M,K] tile: axis 0 (m) rides the contiguous
                # stride_L_col, so the load coalesces and no tl.trans layout
                # conversion is needed before the dot.
                tile_offset = (
                    L_base
                    + rows_m[:, None] * stride_L_col
                    + rows_k[None, :] * stride_L_row
                )
                tile_real = tl.load(
                    L_ptr + tile_offset,
                    mask=rows_m_mask[:, None],
                    other=0.0,
                )
                tile_imag = tl.load(
                    L_ptr + tile_offset + 1,
                    mask=rows_m_mask[:, None],
                    other=0.0,
                )
            else:
                # Coalesced [K,M] load (axis 0 rides stride_L_col); the trans
                # folds into the dot's shared-memory operand staging.
                tile_offset = (
                    L_base
                    + rows_k[:, None] * stride_L_col
                    + rows_m[None, :] * stride_L_row
                )
                tile_real = tl.trans(
                    tl.load(
                        L_ptr + tile_offset,
                        mask=rows_m_mask[None, :],
                        other=0.0,
                    )
                )
                tile_imag = tl.trans(
                    tl.load(
                        L_ptr + tile_offset + 1,
                        mask=rows_m_mask[None, :],
                        other=0.0,
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
                tail_real = tl.load(
                    B_ptr + tail_offset,
                    mask=tail_mask,
                    other=0.0,
                )
                tail_imag = tl.load(
                    B_ptr + tail_offset + 1,
                    mask=tail_mask,
                    other=0.0,
                )
            else:
                tail_real = tl.load(
                    X_ptr + tail_offset,
                    mask=tail_mask,
                    other=0.0,
                )
                tail_imag = tl.load(
                    X_ptr + tail_offset + 1,
                    mask=tail_mask,
                    other=0.0,
                )

            if IS_DOUBLE:
                update_real = tl.dot(tile_real, w_real)
                update_real -= tl.dot(tile_imag, w_imag)
                update_imag = tl.dot(tile_real, w_imag)
                update_imag += tl.dot(tile_imag, w_real)
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
        w_real = x_real * inv_diag[:, None]
        w_imag = x_imag * inv_diag[:, None]

        for ii_idx in range(BLOCK_K - 1, -1, -1):
            if upper:
                factor_offset = (
                    L_base + rows_k * stride_L_row + (k + ii_idx) * stride_L_col
                )
            else:
                factor_offset = (
                    L_base + (k + ii_idx) * stride_L_row + rows_k * stride_L_col
                )
            factor_real = tl.load(
                L_ptr + factor_offset,
                mask=k_offsets < ii_idx,
                other=0.0,
            )
            factor_imag = tl.load(
                L_ptr + factor_offset + 1,
                mask=k_offsets < ii_idx,
                other=0.0,
            )
            if storage_conj == upper:
                factor_imag = -factor_imag
            norm_real = factor_real * inv_diag
            norm_imag = factor_imag * inv_diag
            pivot_real = tl.gather(
                w_real,
                tl.full([1, BLOCK_RHS], ii_idx, tl.int32),
                0,
            )
            pivot_imag = tl.gather(
                w_imag,
                tl.full([1, BLOCK_RHS], ii_idx, tl.int32),
                0,
            )
            w_real -= norm_real[:, None] * pivot_real - norm_imag[:, None] * pivot_imag
            w_imag -= norm_real[:, None] * pivot_imag + norm_imag[:, None] * pivot_real

        tl.store(X_ptr + x_offset, w_real, mask=rhs_mask[None, :])
        tl.store(X_ptr + x_offset + 1, w_imag, mask=rhs_mask[None, :])

        for m in range(0, k, BLOCK_M):
            rows_m = m + m_offsets
            rows_m_mask = rows_m < k
            if upper:
                # Coalesced [K,M] load (axis 0 rides stride_L_col); the trans
                # folds into the dot's shared-memory operand staging.
                tile_offset = (
                    L_base
                    + rows_k[:, None] * stride_L_col
                    + rows_m[None, :] * stride_L_row
                )
                tile_real = tl.trans(
                    tl.load(
                        L_ptr + tile_offset,
                        mask=rows_m_mask[None, :],
                        other=0.0,
                    )
                )
                tile_imag = tl.trans(
                    tl.load(
                        L_ptr + tile_offset + 1,
                        mask=rows_m_mask[None, :],
                        other=0.0,
                    )
                )
            else:
                # Direct [M,K] tile: axis 0 (m) rides the contiguous
                # stride_L_col; no tl.trans layout conversion needed.
                tile_offset = (
                    L_base
                    + rows_m[:, None] * stride_L_col
                    + rows_k[None, :] * stride_L_row
                )
                tile_real = tl.load(
                    L_ptr + tile_offset,
                    mask=rows_m_mask[:, None],
                    other=0.0,
                )
                tile_imag = tl.load(
                    L_ptr + tile_offset + 1,
                    mask=rows_m_mask[:, None],
                    other=0.0,
                )
            if storage_conj == upper:
                tile_imag = -tile_imag

            head_offset = (
                B_base
                + rows_m[:, None] * stride_B_row
                + rhs_cols[None, :] * stride_B_col
            )
            head_mask = rows_m_mask[:, None] & rhs_mask[None, :]
            head_real = tl.load(
                X_ptr + head_offset,
                mask=head_mask,
                other=0.0,
            )
            head_imag = tl.load(
                X_ptr + head_offset + 1,
                mask=head_mask,
                other=0.0,
            )
            if IS_DOUBLE:
                update_real = tl.dot(tile_real, w_real)
                update_real -= tl.dot(tile_imag, w_imag)
                update_imag = tl.dot(tile_real, w_imag)
                update_imag += tl.dot(tile_imag, w_real)
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
def cholesky_solve_complex_invert_blocks_kernel(
    L_ptr,
    T_ptr,
    N: tl.constexpr,
    batch_stride_L,
    stride_L_row,
    stride_L_col,
    BLOCK_K: tl.constexpr,
    upper: tl.constexpr,
    storage_conj: tl.constexpr,
):
    batch_pid = program_id(0)
    block_pid = program_id(1)
    L_base = batch_pid * batch_stride_L
    k = block_pid * BLOCK_K
    k_offsets = tl.arange(0, BLOCK_K)
    rows_k = k + k_offsets

    diag_offset = L_base + rows_k * stride_L_row + rows_k * stride_L_col
    diag_real = tl.load(L_ptr + diag_offset)
    inv_diag = 1.0 / diag_real

    t_real = tl.where(k_offsets[:, None] == k_offsets[None, :], 1.0, 0.0).to(
        L_ptr.dtype.element_ty
    )
    t_imag = tl.zeros([BLOCK_K, BLOCK_K], dtype=L_ptr.dtype.element_ty)

    for i in range(BLOCK_K):
        if upper:
            factor_offset = L_base + (k + i) * stride_L_row + rows_k * stride_L_col
        else:
            factor_offset = L_base + rows_k * stride_L_row + (k + i) * stride_L_col
        factor_real = tl.load(L_ptr + factor_offset, mask=k_offsets > i, other=0.0)
        factor_imag = tl.load(L_ptr + factor_offset + 1, mask=k_offsets > i, other=0.0)
        if storage_conj != upper:
            factor_imag = -factor_imag
        norm_real = factor_real * inv_diag
        norm_imag = factor_imag * inv_diag
        pivot_real = tl.gather(t_real, tl.full([1, BLOCK_K], i, tl.int32), 0)
        pivot_imag = tl.gather(t_imag, tl.full([1, BLOCK_K], i, tl.int32), 0)
        t_real -= norm_real[:, None] * pivot_real - norm_imag[:, None] * pivot_imag
        t_imag -= norm_real[:, None] * pivot_imag + norm_imag[:, None] * pivot_real

    t_offset = (
        batch_pid * N * BLOCK_K * 2
        + k * BLOCK_K * 2
        + k_offsets[:, None] * (BLOCK_K * 2)
        + k_offsets[None, :] * 2
    )
    tl.store(T_ptr + t_offset, t_real)
    tl.store(T_ptr + t_offset + 1, t_imag)


@libentry()
@triton.jit
def cholesky_solve_complex_single_rhs_gather_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
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
    batch_pid = program_id(0)
    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    k_offsets = tl.arange(0, BLOCK_K)
    m_offsets = tl.arange(0, BLOCK_M)

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
        w_real = y_real * inv_diag
        w_imag = y_imag * inv_diag

        for i in range(BLOCK_K):
            if upper:
                factor_offset = L_base + (k + i) * stride_L_row + rows_k * stride_L_col
            else:
                factor_offset = L_base + rows_k * stride_L_row + (k + i) * stride_L_col
            factor_real = tl.load(L_ptr + factor_offset, mask=k_offsets > i, other=0.0)
            factor_imag = tl.load(
                L_ptr + factor_offset + 1, mask=k_offsets > i, other=0.0
            )
            if storage_conj != upper:
                factor_imag = -factor_imag
            norm_real = factor_real * inv_diag
            norm_imag = factor_imag * inv_diag
            pivot_real = tl.gather(w_real, tl.full([1], i, tl.int32), 0)
            pivot_imag = tl.gather(w_imag, tl.full([1], i, tl.int32), 0)
            w_real -= norm_real * pivot_real - norm_imag * pivot_imag
            w_imag -= norm_real * pivot_imag + norm_imag * pivot_real

        tl.store(X_ptr + y_offset, w_real)
        tl.store(X_ptr + y_offset + 1, w_imag)

        for m in range(k + BLOCK_K, N, BLOCK_M):
            rows_m = m + m_offsets
            rows_m_mask = rows_m < N
            if upper:
                # [M,K] tile: axis0 = m (contiguous stride_L_col) -> coalesced;
                # reduction over k = axis 1, intra-thread.
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
                if storage_conj != upper:
                    tile_imag = -tile_imag
                update_real = tl.sum(
                    tile_real * w_real[None, :] - tile_imag * w_imag[None, :], axis=1
                )
                update_imag = tl.sum(
                    tile_real * w_imag[None, :] + tile_imag * w_real[None, :], axis=1
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

    for k in range(N - BLOCK_K, -1, -BLOCK_K):
        rows_k = k + k_offsets
        tl.debug_barrier()
        x_offset = B_base + rows_k * stride_B_row
        x_real = tl.load(X_ptr + x_offset)
        x_imag = tl.load(X_ptr + x_offset + 1)

        diag_offset = L_base + rows_k * stride_L_row + rows_k * stride_L_col
        diag_real = tl.load(L_ptr + diag_offset)
        inv_diag = 1.0 / diag_real
        w_real = x_real * inv_diag
        w_imag = x_imag * inv_diag

        for ii_idx in range(BLOCK_K - 1, -1, -1):
            if upper:
                factor_offset = (
                    L_base + rows_k * stride_L_row + (k + ii_idx) * stride_L_col
                )
            else:
                factor_offset = (
                    L_base + (k + ii_idx) * stride_L_row + rows_k * stride_L_col
                )
            factor_real = tl.load(
                L_ptr + factor_offset, mask=k_offsets < ii_idx, other=0.0
            )
            factor_imag = tl.load(
                L_ptr + factor_offset + 1, mask=k_offsets < ii_idx, other=0.0
            )
            if storage_conj == upper:
                factor_imag = -factor_imag
            norm_real = factor_real * inv_diag
            norm_imag = factor_imag * inv_diag
            pivot_real = tl.gather(w_real, tl.full([1], ii_idx, tl.int32), 0)
            pivot_imag = tl.gather(w_imag, tl.full([1], ii_idx, tl.int32), 0)
            w_real -= norm_real * pivot_real - norm_imag * pivot_imag
            w_imag -= norm_real * pivot_imag + norm_imag * pivot_real

        tl.store(X_ptr + x_offset, w_real)
        tl.store(X_ptr + x_offset + 1, w_imag)

        for m in range(0, k, BLOCK_M):
            rows_m = m + m_offsets
            rows_m_mask = rows_m < k
            if upper:
                # [K,M] tile: axis0 = k (contiguous stride_L_col) -> coalesced;
                # reduction over k = axis 0.
                tile_offset = (
                    L_base
                    + rows_k[:, None] * stride_L_col
                    + rows_m[None, :] * stride_L_row
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
def cholesky_solve_complex_single_rhs_blocked_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    T_ptr,
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
    """Blocked complex TRSV with precomputed half-block inverses.

    Each BLOCK_K diagonal block is solved with two chained SUB_K = BLOCK_K/2
    matvecs against precomputed inverse sub-blocks:
    w1 = T11 @ sv1, w2 = T22 @ (sv2 - M21 @ w1), and the conj-transposed
    counterpart for the backward phase. This replaces the serial per-row
    gather chain, which dominates fp64 single-RHS solves at N >= 256. The
    panel updates split the contraction over the two sub-blocks.
    """
    batch_pid = program_id(0)
    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    SUB_K: tl.constexpr = BLOCK_K // 2
    s = tl.arange(0, SUB_K)
    m_offsets = tl.arange(0, BLOCK_M)

    for k in range(0, N, BLOCK_K):
        if k > 0:
            tl.debug_barrier()
        rows1 = k + s
        rows2 = k + SUB_K + s
        if k == 0:
            y1_real = tl.load(B_ptr + B_base + rows1 * stride_B_row)
            y1_imag = tl.load(B_ptr + B_base + rows1 * stride_B_row + 1)
            y2_real = tl.load(B_ptr + B_base + rows2 * stride_B_row)
            y2_imag = tl.load(B_ptr + B_base + rows2 * stride_B_row + 1)
        else:
            y1_real = tl.load(X_ptr + B_base + rows1 * stride_B_row)
            y1_imag = tl.load(X_ptr + B_base + rows1 * stride_B_row + 1)
            y2_real = tl.load(X_ptr + B_base + rows2 * stride_B_row)
            y2_imag = tl.load(X_ptr + B_base + rows2 * stride_B_row + 1)

        d1 = tl.load(L_ptr + L_base + rows1 * stride_L_row + rows1 * stride_L_col)
        d2 = tl.load(L_ptr + L_base + rows2 * stride_L_row + rows2 * stride_L_col)
        sv1_real = y1_real / d1
        sv1_imag = y1_imag / d1
        sv2_real = y2_real / d2
        sv2_imag = y2_imag / d2

        # load T11, T22 inverse sub-blocks from scratch ([N, SUB_K, 2] layout)
        t1_off = (
            batch_pid * N * SUB_K * 2 + rows1[:, None] * (SUB_K * 2) + s[None, :] * 2
        )
        t1r = tl.load(T_ptr + t1_off)
        t1i = tl.load(T_ptr + t1_off + 1)
        t2_off = (
            batch_pid * N * SUB_K * 2 + rows2[:, None] * (SUB_K * 2) + s[None, :] * 2
        )
        t2r = tl.load(T_ptr + t2_off)
        t2i = tl.load(T_ptr + t2_off + 1)

        # M21 = D2^{-1} R21
        if upper:
            m_off = (
                L_base
                + (k + s[None, :]) * stride_L_row
                + (k + SUB_K + s[:, None]) * stride_L_col
            )
        else:
            m_off = (
                L_base
                + (k + SUB_K + s[:, None]) * stride_L_row
                + (k + s[None, :]) * stride_L_col
            )
        m_r = tl.load(L_ptr + m_off)
        m_i = tl.load(L_ptr + m_off + 1)
        if storage_conj != upper:
            m_i = -m_i
        m_r = m_r / d2[:, None]
        m_i = m_i / d2[:, None]

        # w1 = T11 @ sv1 ; w2 = T22 @ (sv2 - M21 @ w1)
        w1_real = tl.sum(t1r * sv1_real[None, :] - t1i * sv1_imag[None, :], axis=1)
        w1_imag = tl.sum(t1r * sv1_imag[None, :] + t1i * sv1_real[None, :], axis=1)
        q_real = tl.sum(m_r * w1_real[None, :] - m_i * w1_imag[None, :], axis=1)
        q_imag = tl.sum(m_r * w1_imag[None, :] + m_i * w1_real[None, :], axis=1)
        v_real = sv2_real - q_real
        v_imag = sv2_imag - q_imag
        w2_real = tl.sum(t2r * v_real[None, :] - t2i * v_imag[None, :], axis=1)
        w2_imag = tl.sum(t2r * v_imag[None, :] + t2i * v_real[None, :], axis=1)

        tl.store(X_ptr + B_base + rows1 * stride_B_row, w1_real)
        tl.store(X_ptr + B_base + rows1 * stride_B_row + 1, w1_imag)
        tl.store(X_ptr + B_base + rows2 * stride_B_row, w2_real)
        tl.store(X_ptr + B_base + rows2 * stride_B_row + 1, w2_imag)

        for m in range(k + BLOCK_K, N, BLOCK_M):
            rows_m = m + m_offsets
            rows_m_mask = rows_m < N
            off1 = (
                L_base
                + rows_m[:, None] * stride_L_col
                + (k + s)[None, :] * stride_L_row
            )
            off2 = (
                L_base
                + rows_m[:, None] * stride_L_col
                + (k + SUB_K + s)[None, :] * stride_L_row
            )
            tile1r = tl.load(L_ptr + off1, mask=rows_m_mask[:, None], other=0.0)
            tile1i = tl.load(L_ptr + off1 + 1, mask=rows_m_mask[:, None], other=0.0)
            tile2r = tl.load(L_ptr + off2, mask=rows_m_mask[:, None], other=0.0)
            tile2i = tl.load(L_ptr + off2 + 1, mask=rows_m_mask[:, None], other=0.0)
            if storage_conj != upper:
                tile1i = -tile1i
                tile2i = -tile2i
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
            upd_real = tl.sum(
                tile1r * w1_real[None, :] - tile1i * w1_imag[None, :], axis=1
            )
            upd_imag = tl.sum(
                tile1r * w1_imag[None, :] + tile1i * w1_real[None, :], axis=1
            )
            upd_real += tl.sum(
                tile2r * w2_real[None, :] - tile2i * w2_imag[None, :], axis=1
            )
            upd_imag += tl.sum(
                tile2r * w2_imag[None, :] + tile2i * w2_real[None, :], axis=1
            )
            tl.store(X_ptr + tail_offset, tail_real - upd_real, mask=rows_m_mask)
            tl.store(X_ptr + tail_offset + 1, tail_imag - upd_imag, mask=rows_m_mask)

    for k in range(N - BLOCK_K, -1, -BLOCK_K):
        tl.debug_barrier()
        rows1 = k + s
        rows2 = k + SUB_K + s
        x1_real = tl.load(X_ptr + B_base + rows1 * stride_B_row)
        x1_imag = tl.load(X_ptr + B_base + rows1 * stride_B_row + 1)
        x2_real = tl.load(X_ptr + B_base + rows2 * stride_B_row)
        x2_imag = tl.load(X_ptr + B_base + rows2 * stride_B_row + 1)

        d1 = tl.load(L_ptr + L_base + rows1 * stride_L_row + rows1 * stride_L_col)
        d2 = tl.load(L_ptr + L_base + rows2 * stride_L_row + rows2 * stride_L_col)

        t1_off = (
            batch_pid * N * SUB_K * 2 + rows1[:, None] * (SUB_K * 2) + s[None, :] * 2
        )
        t2_off = (
            batch_pid * N * SUB_K * 2 + rows2[:, None] * (SUB_K * 2) + s[None, :] * 2
        )
        # transposed loads for the conj-transposed matvecs
        tt1_off = (
            batch_pid * N * SUB_K * 2 + (k + s[None, :]) * (SUB_K * 2) + s[:, None] * 2
        )
        tt2_off = (
            batch_pid * N * SUB_K * 2
            + (k + SUB_K + s[None, :]) * (SUB_K * 2)
            + s[:, None] * 2
        )
        tt1r = tl.load(T_ptr + tt1_off)
        tt1i = -tl.load(T_ptr + tt1_off + 1)
        tt2r = tl.load(T_ptr + tt2_off)
        tt2i = -tl.load(T_ptr + tt2_off + 1)

        if upper:
            m_off = (
                L_base
                + (k + s[None, :]) * stride_L_row
                + (k + SUB_K + s[:, None]) * stride_L_col
            )
        else:
            m_off = (
                L_base
                + (k + SUB_K + s[:, None]) * stride_L_row
                + (k + s[None, :]) * stride_L_col
            )
        # M21^H: transposed indexing + conj
        mh_r = tl.load(L_ptr + m_off)
        mh_i = tl.load(L_ptr + m_off + 1)
        if storage_conj != upper:
            mh_i = -mh_i
        mh_r = mh_r / d2[:, None]
        mh_i = mh_i / d2[:, None]
        mht_r = tl.trans(mh_r)
        mht_i = -tl.trans(mh_i)

        # q = T22^H y2; x2 = q / d2
        q_real = tl.sum(tt2r * x2_real[None, :] - tt2i * x2_imag[None, :], axis=1)
        q_imag = tl.sum(tt2r * x2_imag[None, :] + tt2i * x2_real[None, :], axis=1)
        w2_real = q_real / d2
        w2_imag = q_imag / d2
        # x1 = (T11^H (y1 - M21^H q)) / d1
        r_real = tl.sum(mht_r * q_real[None, :] - mht_i * q_imag[None, :], axis=1)
        r_imag = tl.sum(mht_r * q_imag[None, :] + mht_i * q_real[None, :], axis=1)
        p1_real = x1_real - r_real
        p1_imag = x1_imag - r_imag
        w1_real = tl.sum(tt1r * p1_real[None, :] - tt1i * p1_imag[None, :], axis=1) / d1
        w1_imag = tl.sum(tt1r * p1_imag[None, :] + tt1i * p1_real[None, :], axis=1) / d1

        tl.store(X_ptr + B_base + rows1 * stride_B_row, w1_real)
        tl.store(X_ptr + B_base + rows1 * stride_B_row + 1, w1_imag)
        tl.store(X_ptr + B_base + rows2 * stride_B_row, w2_real)
        tl.store(X_ptr + B_base + rows2 * stride_B_row + 1, w2_imag)

        for m in range(0, k, BLOCK_M):
            rows_m = m + m_offsets
            rows_m_mask = rows_m < k
            off1 = (
                L_base
                + (k + s)[:, None] * stride_L_col
                + rows_m[None, :] * stride_L_row
            )
            off2 = (
                L_base
                + (k + SUB_K + s)[:, None] * stride_L_col
                + rows_m[None, :] * stride_L_row
            )
            tile1r = tl.load(L_ptr + off1, mask=rows_m_mask[None, :], other=0.0)
            tile1i = tl.load(L_ptr + off1 + 1, mask=rows_m_mask[None, :], other=0.0)
            tile2r = tl.load(L_ptr + off2, mask=rows_m_mask[None, :], other=0.0)
            tile2i = tl.load(L_ptr + off2 + 1, mask=rows_m_mask[None, :], other=0.0)
            if storage_conj == upper:
                tile1i = -tile1i
                tile2i = -tile2i
            head_offset = B_base + rows_m * stride_B_row
            head_real = tl.load(X_ptr + head_offset, mask=rows_m_mask, other=0.0)
            head_imag = tl.load(X_ptr + head_offset + 1, mask=rows_m_mask, other=0.0)
            upd_real = tl.sum(
                tile1r * w1_real[:, None] - tile1i * w1_imag[:, None], axis=0
            )
            upd_imag = tl.sum(
                tile1r * w1_imag[:, None] + tile1i * w1_real[:, None], axis=0
            )
            upd_real += tl.sum(
                tile2r * w2_real[:, None] - tile2i * w2_imag[:, None], axis=0
            )
            upd_imag += tl.sum(
                tile2r * w2_imag[:, None] + tile2i * w2_real[:, None], axis=0
            )
            tl.store(X_ptr + head_offset, head_real - upd_real, mask=rows_m_mask)
            tl.store(X_ptr + head_offset + 1, head_imag - upd_imag, mask=rows_m_mask)


@libentry()
@triton.jit
def cholesky_solve_blocked_lower_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
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
    """Blocked lower-factor Cholesky solve.

    Solves L L^T X = B for one batch and one RHS tile using a row-major
    lower factor.
    """
    batch_pid = program_id(0)
    rhs_tile_pid = program_id(1)

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    rhs_cols = rhs_tile_pid * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    rhs_mask = rhs_cols < nrhs
    k_offsets = tl.arange(0, BLOCK_K)
    m_offsets = tl.arange(0, BLOCK_M)

    # Forward blocked TRSM: L * Y = B. The diagonal-block solve keeps the
    # running tile pre-scaled by the reciprocal diagonal so each serial row
    # step is a constant-index tl.gather (warp shuffle) plus a rank-1
    # update -- no per-row reduce or select (see the single-RHS kernels).
    for k in range(0, N, BLOCK_K):
        rows_k = k + k_offsets
        if k > 0:
            # Cross-warp handoff: rows of X below were written by tail
            # updates of earlier k-blocks, possibly by other warps. bar.sync
            # makes those global stores visible before we read them.
            tl.debug_barrier()
        if k == 0:
            y_block = tl.load(
                B_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
                mask=rhs_mask[None, :],
                other=0.0,
            )
        else:
            y_block = tl.load(
                X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
                mask=rhs_mask[None, :],
                other=0.0,
            )

        diag_block = tl.load(L_ptr + L_base + rows_k * stride_L + rows_k)
        inv_diag = 1.0 / diag_block
        inv_diag = inv_diag * (2.0 - diag_block * inv_diag)

        y_block = y_block * inv_diag[:, None]
        for i in range(BLOCK_K):
            L_col = tl.load(
                L_ptr + L_base + rows_k * stride_L + (k + i),
                mask=k_offsets > i,
                other=0.0,
            )
            w_i = tl.gather(y_block, tl.full([1, BLOCK_RHS], i, tl.int32), 0)
            y_block = y_block - (L_col * inv_diag)[:, None] * w_i

        tl.store(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            y_block,
            mask=rhs_mask[None, :],
        )

        for m in range(k + BLOCK_K, N, BLOCK_M):
            rows_m = m + m_offsets
            L_tile = tl.load(
                L_ptr + L_base + rows_m[:, None] * stride_L + rows_k[None, :]
            )
            if k == 0:
                tail = tl.load(
                    B_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                    mask=rhs_mask[None, :],
                    other=0.0,
                )
            else:
                tail = tl.load(
                    X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                    mask=rhs_mask[None, :],
                    other=0.0,
                )
            tail = tail - tl.dot(L_tile, y_block, input_precision="ieee")
            tl.store(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                tail,
                mask=rhs_mask[None, :],
            )

    # Backward blocked TRSM: L^T * X = Y.
    for k in range(N - BLOCK_K, -1, -BLOCK_K):
        rows_k = k + k_offsets
        tl.debug_barrier()
        x_block = tl.load(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            mask=rhs_mask[None, :],
            other=0.0,
        )

        diag_block = tl.load(L_ptr + L_base + rows_k * stride_L + rows_k)
        inv_diag = 1.0 / diag_block
        inv_diag = inv_diag * (2.0 - diag_block * inv_diag)

        x_block = x_block * inv_diag[:, None]
        for ii in range(BLOCK_K - 1, -1, -1):
            L_row = tl.load(
                L_ptr + L_base + (k + ii) * stride_L + rows_k,
                mask=k_offsets < ii,
                other=0.0,
            )
            w_i = tl.gather(x_block, tl.full([1, BLOCK_RHS], ii, tl.int32), 0)
            x_block = x_block - (L_row * inv_diag)[:, None] * w_i

        tl.store(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            x_block,
            mask=rhs_mask[None, :],
        )

        for m in range(0, k, BLOCK_M):
            rows_m = m + m_offsets
            L_tile = tl.load(
                L_ptr + L_base + rows_k[None, :] * stride_L + rows_m[:, None]
            )
            head = tl.load(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                mask=rhs_mask[None, :],
                other=0.0,
            )
            head = head - tl.dot(L_tile, x_block, input_precision="ieee")
            tl.store(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                head,
                mask=rhs_mask[None, :],
            )


@libentry()
@triton.jit
def cholesky_solve_blocked_upper_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
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
    """Blocked upper-factor Cholesky solve.

    Solves U^T U X = B for one batch and one RHS tile. Mirrors the lower
    blocked path while keeping the upper storage layout intact.

    Includes fast reciprocal (Newton refinement) for diagonal division,
    matching the algorithm used by cuSOLVER's potrs kernel.
    """
    batch_pid = program_id(0)
    rhs_tile_pid = program_id(1)

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    rhs_cols = rhs_tile_pid * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    rhs_mask = rhs_cols < nrhs
    k_offsets = tl.arange(0, BLOCK_K)
    m_offsets = tl.arange(0, BLOCK_M)

    # Forward blocked TRSM: U^T * Y = B.
    for k in range(0, N, BLOCK_K):
        rows_k = k + k_offsets
        if k > 0:
            # Cross-warp handoff: rows of X below were written by tail
            # updates of earlier k-blocks, possibly by other warps. bar.sync
            # makes those global stores visible before we read them.
            tl.debug_barrier()
        if k == 0:
            y_block = tl.load(
                B_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
                mask=rhs_mask[None, :],
                other=0.0,
            )
        else:
            y_block = tl.load(
                X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
                mask=rhs_mask[None, :],
                other=0.0,
            )

        diag_block = tl.load(L_ptr + L_base + rows_k * stride_L + rows_k)
        inv_diag = 1.0 / diag_block
        inv_diag = inv_diag * (2.0 - diag_block * inv_diag)

        y_block = y_block * inv_diag[:, None]
        for i in range(BLOCK_K):
            U_row = tl.load(
                L_ptr + L_base + (k + i) * stride_L + rows_k,
                mask=k_offsets > i,
                other=0.0,
            )
            w_i = tl.gather(y_block, tl.full([1, BLOCK_RHS], i, tl.int32), 0)
            y_block = y_block - (U_row * inv_diag)[:, None] * w_i

        tl.store(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            y_block,
            mask=rhs_mask[None, :],
        )

        for m in range(k + BLOCK_K, N, BLOCK_M):
            rows_m = m + m_offsets
            U_tile_km = tl.load(
                L_ptr + L_base + rows_k[:, None] * stride_L + rows_m[None, :]
            )
            U_tile = tl.trans(U_tile_km)
            if k == 0:
                tail = tl.load(
                    B_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                    mask=rhs_mask[None, :],
                    other=0.0,
                )
            else:
                tail = tl.load(
                    X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                    mask=rhs_mask[None, :],
                    other=0.0,
                )
            tail = tail - tl.dot(U_tile, y_block, input_precision="ieee")
            tl.store(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                tail,
                mask=rhs_mask[None, :],
            )

    # Backward blocked TRSM: U * X = Y.
    for k in range(N - BLOCK_K, -1, -BLOCK_K):
        rows_k = k + k_offsets
        tl.debug_barrier()
        x_block = tl.load(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            mask=rhs_mask[None, :],
            other=0.0,
        )

        diag_block = tl.load(L_ptr + L_base + rows_k * stride_L + rows_k)
        inv_diag = 1.0 / diag_block
        inv_diag = inv_diag * (2.0 - diag_block * inv_diag)

        x_block = x_block * inv_diag[:, None]
        for ii in range(BLOCK_K - 1, -1, -1):
            U_col = tl.load(
                L_ptr + L_base + rows_k * stride_L + (k + ii),
                mask=k_offsets < ii,
                other=0.0,
            )
            w_i = tl.gather(x_block, tl.full([1, BLOCK_RHS], ii, tl.int32), 0)
            x_block = x_block - (U_col * inv_diag)[:, None] * w_i

        tl.store(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            x_block,
            mask=rhs_mask[None, :],
        )

        for m in range(0, k, BLOCK_M):
            rows_m = m + m_offsets
            U_tile = tl.load(
                L_ptr + L_base + rows_m[:, None] * stride_L + rows_k[None, :]
            )
            head = tl.load(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                mask=rhs_mask[None, :],
                other=0.0,
            )
            head = head - tl.dot(U_tile, x_block, input_precision="ieee")
            tl.store(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                head,
                mask=rhs_mask[None, :],
            )


@libentry()
@triton.jit
def cholesky_solve_blocked_lower_fp64_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
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
    """fp64-dedicated blocked lower-factor Cholesky solve.

    Removes dtype_flag branch and fp32 logic. Uses standard row-major
    lower factor layout (diag access via row*stride_L+row).
    """
    batch_pid = program_id(0)
    rhs_tile_pid = program_id(1)

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    rhs_cols = rhs_tile_pid * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    rhs_mask = rhs_cols < nrhs
    k_offsets = tl.arange(0, BLOCK_K)
    m_offsets = tl.arange(0, BLOCK_M)

    # Forward blocked TRSM: L * Y = B.
    for k in range(0, N, BLOCK_K):
        rows_k = k + k_offsets
        if k > 0:
            # Cross-warp handoff: rows of X below were written by tail
            # updates of earlier k-blocks, possibly by other warps. bar.sync
            # makes those global stores visible before we read them.
            tl.debug_barrier()
        if k == 0:
            y_block = tl.load(
                B_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
                mask=(rows_k[:, None] < N) & rhs_mask[None, :],
                other=0.0,
            )
        else:
            y_block = tl.load(
                X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
                mask=(rows_k[:, None] < N) & rhs_mask[None, :],
                other=0.0,
            )

        diag_block = tl.load(
            L_ptr + L_base + rows_k * stride_L + rows_k,
            mask=rows_k < N,
            other=1.0,
        )
        inv_diag_block = 1.0 / diag_block
        inv_diag_block = inv_diag_block * (2.0 - diag_block * inv_diag_block)
        inv_diag_block = inv_diag_block * (2.0 - diag_block * inv_diag_block)

        y_block = y_block * inv_diag_block[:, None]
        for i in range(BLOCK_K):
            L_col = tl.load(
                L_ptr + L_base + rows_k * stride_L + (k + i),
                mask=k_offsets > i,
                other=0.0,
            )
            w_i = tl.gather(y_block, tl.full([1, BLOCK_RHS], i, tl.int32), 0)
            y_block = y_block - (L_col * inv_diag_block)[:, None] * w_i

        tl.store(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            y_block,
            mask=(rows_k[:, None] < N) & rhs_mask[None, :],
        )

        for m in range(k + BLOCK_K, N, BLOCK_M):
            rows_m = m + m_offsets
            L_tile = tl.load(
                L_ptr + L_base + rows_m[:, None] * stride_L + rows_k[None, :],
                mask=(rows_m[:, None] < N) & (rows_k[None, :] < N),
                other=0.0,
            )
            if k == 0:
                tail = tl.load(
                    B_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                    mask=(rows_m[:, None] < N) & rhs_mask[None, :],
                    other=0.0,
                )
            else:
                tail = tl.load(
                    X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                    mask=(rows_m[:, None] < N) & rhs_mask[None, :],
                    other=0.0,
                )
            tail = tail - tl.dot(L_tile, y_block)
            tl.store(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                tail,
                mask=(rows_m[:, None] < N) & rhs_mask[None, :],
            )

    # Backward blocked TRSM: L^T * X = Y.
    for k in range(N - BLOCK_K, -1, -BLOCK_K):
        rows_k = k + k_offsets
        tl.debug_barrier()
        x_block = tl.load(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            mask=(rows_k[:, None] < N) & rhs_mask[None, :],
            other=0.0,
        )

        diag_block = tl.load(
            L_ptr + L_base + rows_k * stride_L + rows_k,
            mask=rows_k < N,
            other=1.0,
        )
        inv_diag_block = 1.0 / diag_block
        inv_diag_block = inv_diag_block * (2.0 - diag_block * inv_diag_block)
        inv_diag_block = inv_diag_block * (2.0 - diag_block * inv_diag_block)

        x_block = x_block * inv_diag_block[:, None]
        for ii in range(BLOCK_K - 1, -1, -1):
            L_row = tl.load(
                L_ptr + L_base + (k + ii) * stride_L + rows_k,
                mask=k_offsets < ii,
                other=0.0,
            )
            w_i = tl.gather(x_block, tl.full([1, BLOCK_RHS], ii, tl.int32), 0)
            x_block = x_block - (L_row * inv_diag_block)[:, None] * w_i

        tl.store(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            x_block,
            mask=(rows_k[:, None] < N) & rhs_mask[None, :],
        )

        for m in range(0, k, BLOCK_M):
            rows_m = m + m_offsets
            rows_m_mask = rows_m < k
            head = tl.load(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                mask=rows_m_mask[:, None] & rhs_mask[None, :],
                other=0.0,
            )
            L_tile = tl.load(
                L_ptr + L_base + rows_k[None, :] * stride_L + rows_m[:, None],
                mask=rows_m_mask[:, None] & (rows_k[None, :] < N),
                other=0.0,
            )
            head = head - tl.dot(L_tile, x_block)
            tl.store(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                head,
                mask=rows_m_mask[:, None] & rhs_mask[None, :],
            )


@libentry()
@triton.jit
def cholesky_solve_blocked_upper_fp64_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
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
    """fp64-dedicated blocked upper-factor Cholesky solve.

    Removes dtype_flag branch and fp32 logic.
    """
    batch_pid = program_id(0)
    rhs_tile_pid = program_id(1)

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    rhs_cols = rhs_tile_pid * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    rhs_mask = rhs_cols < nrhs
    k_offsets = tl.arange(0, BLOCK_K)
    m_offsets = tl.arange(0, BLOCK_M)

    # Forward blocked TRSM: U^T * Y = B.
    for k in range(0, N, BLOCK_K):
        rows_k = k + k_offsets
        if k > 0:
            # Cross-warp handoff: rows of X below were written by tail
            # updates of earlier k-blocks, possibly by other warps. bar.sync
            # makes those global stores visible before we read them.
            tl.debug_barrier()
        if k == 0:
            y_block = tl.load(
                B_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
                mask=(rows_k[:, None] < N) & rhs_mask[None, :],
                other=0.0,
            )
        else:
            y_block = tl.load(
                X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
                mask=(rows_k[:, None] < N) & rhs_mask[None, :],
                other=0.0,
            )

        diag_block = tl.load(
            L_ptr + L_base + rows_k * stride_L + rows_k,
            mask=rows_k < N,
            other=1.0,
        )
        inv_diag_block = 1.0 / diag_block
        inv_diag_block = inv_diag_block * (2.0 - diag_block * inv_diag_block)
        inv_diag_block = inv_diag_block * (2.0 - diag_block * inv_diag_block)

        y_block = y_block * inv_diag_block[:, None]
        for i in range(BLOCK_K):
            U_row = tl.load(
                L_ptr + L_base + (k + i) * stride_L + rows_k,
                mask=k_offsets > i,
                other=0.0,
            )
            w_i = tl.gather(y_block, tl.full([1, BLOCK_RHS], i, tl.int32), 0)
            y_block = y_block - (U_row * inv_diag_block)[:, None] * w_i

        tl.store(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            y_block,
            mask=(rows_k[:, None] < N) & rhs_mask[None, :],
        )

        for m in range(k + BLOCK_K, N, BLOCK_M):
            rows_m = m + m_offsets
            if k == 0:
                tail = tl.load(
                    B_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                    mask=(rows_m[:, None] < N) & rhs_mask[None, :],
                    other=0.0,
                )
            else:
                tail = tl.load(
                    X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                    mask=(rows_m[:, None] < N) & rhs_mask[None, :],
                    other=0.0,
                )
            U_tile_km = tl.load(
                L_ptr + L_base + rows_k[:, None] * stride_L + rows_m[None, :],
                mask=(rows_k[:, None] < N) & (rows_m[None, :] < N),
                other=0.0,
            )
            tail = tail - tl.dot(tl.trans(U_tile_km), y_block)
            tl.store(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                tail,
                mask=(rows_m[:, None] < N) & rhs_mask[None, :],
            )

    # Backward blocked TRSM: U * X = Y.
    for k in range(N - BLOCK_K, -1, -BLOCK_K):
        rows_k = k + k_offsets
        tl.debug_barrier()
        x_block = tl.load(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            mask=(rows_k[:, None] < N) & rhs_mask[None, :],
            other=0.0,
        )

        diag_block = tl.load(
            L_ptr + L_base + rows_k * stride_L + rows_k,
            mask=rows_k < N,
            other=1.0,
        )
        inv_diag_block = 1.0 / diag_block
        inv_diag_block = inv_diag_block * (2.0 - diag_block * inv_diag_block)
        inv_diag_block = inv_diag_block * (2.0 - diag_block * inv_diag_block)

        x_block = x_block * inv_diag_block[:, None]
        for ii in range(BLOCK_K - 1, -1, -1):
            U_col = tl.load(
                L_ptr + L_base + rows_k * stride_L + (k + ii),
                mask=k_offsets < ii,
                other=0.0,
            )
            w_i = tl.gather(x_block, tl.full([1, BLOCK_RHS], ii, tl.int32), 0)
            x_block = x_block - (U_col * inv_diag_block)[:, None] * w_i

        tl.store(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            x_block,
            mask=(rows_k[:, None] < N) & rhs_mask[None, :],
        )

        for m in range(0, k, BLOCK_M):
            rows_m = m + m_offsets
            rows_m_mask = rows_m < k
            U_tile = tl.load(
                L_ptr + L_base + rows_m[:, None] * stride_L + rows_k[None, :],
                mask=rows_m_mask[:, None] & (rows_k[None, :] < N),
                other=0.0,
            )
            head = tl.load(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                mask=rows_m_mask[:, None] & rhs_mask[None, :],
                other=0.0,
            )
            head = head - tl.dot(U_tile, x_block)
            tl.store(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                head,
                mask=rows_m_mask[:, None] & rhs_mask[None, :],
            )


def _can_use_blocked_path(N, nrhs):
    return N >= 64 and N % 32 == 0 and nrhs >= 4


def _can_use_blocked_single_rhs_path(N, nrhs):
    # N <= 32 single-RHS solves use the register-resident small gather
    # kernel instead; the blocked kernels require N % BLOCK_K == 0.
    return nrhs == 1 and N >= 64 and N % 32 == 0


@libentry()
@triton.jit
def cholesky_solve_single_rhs_blocked_lower_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    N: tl.constexpr,
    batch_stride_L,
    batch_stride_B,
    stride_L,
    stride_B,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    dtype_flag: tl.constexpr,
):
    """Blocked lower-factor single-RHS Cholesky solve.

    Requires N % BLOCK_K == 0 (enforced by the dispatcher).

    The diagonal-block solve keeps the running solution pre-scaled by the
    reciprocal diagonal (w = y * inv_diag), so each serial row step needs only
    a constant-index tl.gather (a single warp shuffle) plus one fused
    multiply-add: w -= (L_col * inv_diag) * w[i]. Position i is untouched by
    the masked column, so no select is needed either. This cuts the per-row
    dependency chain from a masked multi-op reduce to shuffle+FMA, which is
    what dominates this inherently serial kernel. Off-diagonal panel updates
    use exact tiles when they divide evenly and masked edge tiles otherwise.
    """
    batch_pid = program_id(0)

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    k_offsets = tl.arange(0, BLOCK_K)
    m_offsets = tl.arange(0, BLOCK_M)

    # Forward blocked TRSV: L * Y = B.
    for k in range(0, N, BLOCK_K):
        rows_k = k + k_offsets
        if k > 0:
            # Cross-warp handoff: rows of X below were written by tail
            # updates of earlier k-blocks, possibly by other warps. bar.sync
            # makes those global stores visible before we read them.
            tl.debug_barrier()
        if k == 0:
            y_block = tl.load(B_ptr + B_base + rows_k * stride_B)
        else:
            y_block = tl.load(X_ptr + B_base + rows_k * stride_B)

        diag_block = tl.load(L_ptr + L_base + rows_k * stride_L + rows_k)
        inv_diag_block = 1.0 / diag_block
        inv_diag_block = inv_diag_block * (2.0 - diag_block * inv_diag_block)
        if dtype_flag == 1:
            inv_diag_block = inv_diag_block * (2.0 - diag_block * inv_diag_block)

        w = y_block * inv_diag_block
        for i in range(BLOCK_K):
            L_col = tl.load(
                L_ptr + L_base + rows_k * stride_L + (k + i),
                mask=k_offsets > i,
                other=0.0,
            )
            w_i = tl.gather(w, tl.full([1], i, tl.int32), 0)
            w = w - (L_col * inv_diag_block) * w_i

        tl.store(X_ptr + B_base + rows_k * stride_B, w)

        for m in range(k + BLOCK_K, N, BLOCK_M):
            rows_m = m + m_offsets
            if m + BLOCK_M <= N:
                L_tile = tl.load(
                    L_ptr + L_base + rows_m[:, None] * stride_L + rows_k[None, :]
                )
                if k == 0:
                    tail = tl.load(B_ptr + B_base + rows_m * stride_B)
                else:
                    tail = tl.load(X_ptr + B_base + rows_m * stride_B)
                tail = tail - tl.sum(L_tile * w[None, :], axis=1)
                tl.store(X_ptr + B_base + rows_m * stride_B, tail)
            else:
                rows_m_mask = rows_m < N
                L_tile = tl.load(
                    L_ptr + L_base + rows_m[:, None] * stride_L + rows_k[None, :],
                    mask=rows_m_mask[:, None],
                    other=0.0,
                )
                if k == 0:
                    tail = tl.load(
                        B_ptr + B_base + rows_m * stride_B,
                        mask=rows_m_mask,
                        other=0.0,
                    )
                else:
                    tail = tl.load(
                        X_ptr + B_base + rows_m * stride_B,
                        mask=rows_m_mask,
                        other=0.0,
                    )
                tail = tail - tl.sum(L_tile * w[None, :], axis=1)
                tl.store(
                    X_ptr + B_base + rows_m * stride_B,
                    tail,
                    mask=rows_m_mask,
                )

    # Backward blocked TRSV: L^T * X = Y.
    for k in range(N - BLOCK_K, -1, -BLOCK_K):
        rows_k = k + k_offsets
        tl.debug_barrier()
        x_block = tl.load(X_ptr + B_base + rows_k * stride_B)

        diag_block = tl.load(L_ptr + L_base + rows_k * stride_L + rows_k)
        inv_diag_block = 1.0 / diag_block
        inv_diag_block = inv_diag_block * (2.0 - diag_block * inv_diag_block)
        if dtype_flag == 1:
            inv_diag_block = inv_diag_block * (2.0 - diag_block * inv_diag_block)

        w = x_block * inv_diag_block
        for ii in range(BLOCK_K - 1, -1, -1):
            L_row = tl.load(
                L_ptr + L_base + (k + ii) * stride_L + rows_k,
                mask=k_offsets < ii,
                other=0.0,
            )
            w_i = tl.gather(w, tl.full([1], ii, tl.int32), 0)
            w = w - (L_row * inv_diag_block) * w_i

        tl.store(X_ptr + B_base + rows_k * stride_B, w)

        for m in range(0, k, BLOCK_M):
            rows_m = m + m_offsets
            if m + BLOCK_M <= k:
                L_tile = tl.load(
                    L_ptr + L_base + rows_k[None, :] * stride_L + rows_m[:, None]
                )
                head = tl.load(X_ptr + B_base + rows_m * stride_B)
                head = head - tl.sum(L_tile * w[None, :], axis=1)
                tl.store(X_ptr + B_base + rows_m * stride_B, head)
            else:
                rows_m_mask = rows_m < k
                L_tile = tl.load(
                    L_ptr + L_base + rows_k[None, :] * stride_L + rows_m[:, None],
                    mask=rows_m_mask[:, None],
                    other=0.0,
                )
                head = tl.load(
                    X_ptr + B_base + rows_m * stride_B,
                    mask=rows_m_mask,
                    other=0.0,
                )
                head = head - tl.sum(L_tile * w[None, :], axis=1)
                tl.store(
                    X_ptr + B_base + rows_m * stride_B,
                    head,
                    mask=rows_m_mask,
                )


@libentry()
@triton.jit
def cholesky_solve_single_rhs_blocked_upper_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    N: tl.constexpr,
    batch_stride_L,
    batch_stride_B,
    stride_L,
    stride_B,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    dtype_flag: tl.constexpr,
):
    """Blocked upper-factor single-RHS Cholesky solve.

    Requires N % BLOCK_K == 0 (enforced by the dispatcher). Mirrors the
    lower kernel's pre-scaled gather formulation (see its docstring) while
    keeping the upper storage layout intact.
    """
    batch_pid = program_id(0)

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    k_offsets = tl.arange(0, BLOCK_K)
    m_offsets = tl.arange(0, BLOCK_M)

    # Forward blocked TRSV: U^T * Y = B.
    for k in range(0, N, BLOCK_K):
        rows_k = k + k_offsets
        if k > 0:
            # Cross-warp handoff: rows of X below were written by tail
            # updates of earlier k-blocks, possibly by other warps. bar.sync
            # makes those global stores visible before we read them.
            tl.debug_barrier()
        if k == 0:
            y_block = tl.load(B_ptr + B_base + rows_k * stride_B)
        else:
            y_block = tl.load(X_ptr + B_base + rows_k * stride_B)

        diag_block = tl.load(L_ptr + L_base + rows_k * stride_L + rows_k)
        inv_diag_block = 1.0 / diag_block
        inv_diag_block = inv_diag_block * (2.0 - diag_block * inv_diag_block)
        if dtype_flag == 1:
            inv_diag_block = inv_diag_block * (2.0 - diag_block * inv_diag_block)

        w = y_block * inv_diag_block
        for i in range(BLOCK_K):
            U_row = tl.load(
                L_ptr + L_base + (k + i) * stride_L + rows_k,
                mask=k_offsets > i,
                other=0.0,
            )
            w_i = tl.gather(w, tl.full([1], i, tl.int32), 0)
            w = w - (U_row * inv_diag_block) * w_i

        tl.store(X_ptr + B_base + rows_k * stride_B, w)

        for m in range(k + BLOCK_K, N, BLOCK_M):
            rows_m = m + m_offsets
            if m + BLOCK_M <= N:
                U_tile = tl.load(
                    L_ptr + L_base + rows_k[:, None] * stride_L + rows_m[None, :]
                )
                if k == 0:
                    tail = tl.load(B_ptr + B_base + rows_m * stride_B)
                else:
                    tail = tl.load(X_ptr + B_base + rows_m * stride_B)
                tail = tail - tl.sum(U_tile * w[:, None], axis=0)
                tl.store(X_ptr + B_base + rows_m * stride_B, tail)
            else:
                rows_m_mask = rows_m < N
                U_tile = tl.load(
                    L_ptr + L_base + rows_k[:, None] * stride_L + rows_m[None, :],
                    mask=rows_m_mask[None, :],
                    other=0.0,
                )
                if k == 0:
                    tail = tl.load(
                        B_ptr + B_base + rows_m * stride_B,
                        mask=rows_m_mask,
                        other=0.0,
                    )
                else:
                    tail = tl.load(
                        X_ptr + B_base + rows_m * stride_B,
                        mask=rows_m_mask,
                        other=0.0,
                    )
                tail = tail - tl.sum(U_tile * w[:, None], axis=0)
                tl.store(
                    X_ptr + B_base + rows_m * stride_B,
                    tail,
                    mask=rows_m_mask,
                )

    # Backward blocked TRSV: U * X = Y.
    for k in range(N - BLOCK_K, -1, -BLOCK_K):
        rows_k = k + k_offsets
        tl.debug_barrier()
        x_block = tl.load(X_ptr + B_base + rows_k * stride_B)

        diag_block = tl.load(L_ptr + L_base + rows_k * stride_L + rows_k)
        inv_diag_block = 1.0 / diag_block
        inv_diag_block = inv_diag_block * (2.0 - diag_block * inv_diag_block)
        if dtype_flag == 1:
            inv_diag_block = inv_diag_block * (2.0 - diag_block * inv_diag_block)

        w = x_block * inv_diag_block
        for ii in range(BLOCK_K - 1, -1, -1):
            U_col = tl.load(
                L_ptr + L_base + rows_k * stride_L + (k + ii),
                mask=k_offsets < ii,
                other=0.0,
            )
            w_i = tl.gather(w, tl.full([1], ii, tl.int32), 0)
            w = w - (U_col * inv_diag_block) * w_i

        tl.store(X_ptr + B_base + rows_k * stride_B, w)

        for m in range(0, k, BLOCK_M):
            rows_m = m + m_offsets
            if m + BLOCK_M <= k:
                U_tile = tl.load(
                    L_ptr + L_base + rows_m[:, None] * stride_L + rows_k[None, :]
                )
                head = tl.load(X_ptr + B_base + rows_m * stride_B)
                head = head - tl.sum(U_tile * w[None, :], axis=1)
                tl.store(X_ptr + B_base + rows_m * stride_B, head)
            else:
                rows_m_mask = rows_m < k
                U_tile = tl.load(
                    L_ptr + L_base + rows_m[:, None] * stride_L + rows_k[None, :],
                    mask=rows_m_mask[:, None],
                    other=0.0,
                )
                head = tl.load(
                    X_ptr + B_base + rows_m * stride_B,
                    mask=rows_m_mask,
                    other=0.0,
                )
                head = head - tl.sum(U_tile * w[None, :], axis=1)
                tl.store(
                    X_ptr + B_base + rows_m * stride_B,
                    head,
                    mask=rows_m_mask,
                )


@libentry()
@triton.jit
def cholesky_solve_small_gather_kernel(
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
    """Small-N register-resident Cholesky solve for few RHS columns.

    Holds the whole [N, nrhs] system in registers, pre-scaled by the
    reciprocal diagonal, so each serial row step is a constant-index
    tl.gather (one warp shuffle) plus a rank-1 update -- no per-row reduce
    or select. See the blocked single-RHS kernel for the derivation. Covers
    both factor orientations and nrhs == 1.
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
        w_i = tl.gather(w, tl.full([1, BLOCK_RHS], i, tl.int32), 0)
        w = w - (col_vals * inv_diag)[:, None] * w_i

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
        w_i = tl.gather(w, tl.full([1, BLOCK_RHS], i, tl.int32), 0)
        w = w - (col_vals * inv_diag)[:, None] * w_i

    tl.store(
        X_ptr + B_base + rows[:, None] * stride_B + cols[None, :],
        w,
        mask=rows_mask[:, None] & cols_mask[None, :],
    )


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


# ══════════════════════════════════════════════════════════════════════════
# GCU400 fp32 fused Neumann fast path (2026-11-18).
#
# cholesky_solve is two triangular solves sharing one factor: solve T*Y = B
# then T^T * X = Y, where T is the lower factor L (or U^T for the upper
# orientation).  Writing T = D(I + M) with M strictly lower and nilpotent,
#
#     X = D^-1 * (I + M)^-T * (I + M)^-1 * D^-1 * B,
#
# and the finite Neumann product (I - M)(I + M^2)(I + M^4)... replaces the
# serial per-row substitution chains (the blocked kernels above spend ~32
# gather steps per diagonal block; a whole-block Neumann solve is a short
# chain of ieee tl.dot's at num_warps=1 -- see the linalg_solve_triangular
# and triangular_solve precedents).  N <= 64 fits one register tile.  Larger
# multiple-of-64 systems precompute per-block inverses in a parallel K1 pass
# and walk the blocks with one diag dot each in K2.
# ══════════════════════════════════════════════════════════════════════════


@libentry()
@triton.jit
def _chol_neumann_1blk_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    N,
    K,
    stride_l,
    stride_b,
    batch_stride_l,
    batch_stride_b,
    BN: tl.constexpr,
    KS: tl.constexpr,
    UPPER: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """Fused two-phase Neumann cholesky solve for N <= BN <= 64.

    Grid: (batch, k-slices).  Both substitution phases collapse into finite
    Neumann products over one register-resident factor tile; X = D^-1 * n^T
    * n * D^-1 * B with n = (I + M)^-1 and M = D^-1 * strictly_lower(T).
    """
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)
    col_start = pid_k * KS

    L = L_ptr + pid_b * batch_stride_l
    Bs = B_ptr + pid_b * batch_stride_b
    X = X_ptr + pid_b * batch_stride_b

    rows = tl.arange(0, BN)
    r2 = rows[:, None]
    c2 = rows[None, :]
    if EVEN_N:
        t_blk = tl.load(L + r2 * stride_l + c2)
        diag = tl.load(L + rows * stride_l + rows)
    else:
        # Clamp masked lanes' addresses in-bounds: this backend's masked
        # loads do not honor `other` for out-of-bounds lanes.
        in_blk = (r2 < N) & (c2 < N)
        t_blk = tl.load(
            L + tl.where(in_blk, r2 * stride_l + c2, 0),
            mask=in_blk,
            other=0.0,
        )
        in_d = rows < N
        diag = tl.load(
            L + tl.where(in_d, rows * stride_l + rows, 0), mask=in_d, other=1.0
        )
    if UPPER:
        # The factor tile in memory is the upper factor U; both phases solve
        # with T = U^T (single transpose, shared by the whole tile).
        t_blk = tl.trans(t_blk)
    dinv = 1.0 / diag
    m_fac = tl.where(c2 < r2, t_blk, 0.0) * dinv[:, None]

    # n_mat = (I + M)^-1 = (I - M)(I + M^2)(I + M^4)..., exact for nilpotent M.
    n_mat = tl.where(r2 == c2, 1.0, 0.0)
    n_mat = n_mat - tl.dot(m_fac, n_mat, input_precision="ieee")
    m2 = tl.dot(m_fac, m_fac, input_precision="ieee")
    n_mat = n_mat + tl.dot(m2, n_mat, input_precision="ieee")
    m4 = tl.dot(m2, m2, input_precision="ieee")
    n_mat = n_mat + tl.dot(m4, n_mat, input_precision="ieee")
    m8 = tl.dot(m4, m4, input_precision="ieee")
    n_mat = n_mat + tl.dot(m8, n_mat, input_precision="ieee")
    if BN >= 32:
        m16 = tl.dot(m8, m8, input_precision="ieee")
        n_mat = n_mat + tl.dot(m16, n_mat, input_precision="ieee")
    if BN >= 64:
        m32 = tl.dot(m16, m16, input_precision="ieee")
        n_mat = n_mat + tl.dot(m32, n_mat, input_precision="ieee")

    kcols = col_start + tl.arange(0, KS)
    if EVEN_K:
        if EVEN_N:
            x_tile = tl.load(Bs + r2 * stride_b + kcols[None, :])
        else:
            rmask = rows[:, None] < N
            x_tile = tl.load(
                Bs + tl.where(rmask, r2 * stride_b + kcols[None, :], 0),
                mask=rmask,
                other=0.0,
            )
    else:
        cmask = kcols < K
        mask2d = (rows[:, None] < N) & cmask[None, :]
        x_tile = tl.load(
            Bs + tl.where(mask2d, r2 * stride_b + kcols[None, :], 0),
            mask=mask2d,
            other=0.0,
        )
    # Forward: Y = n * D^-1 * B; backward: X = D^-1 * n^T * Y.
    y_tile = tl.dot(n_mat, x_tile * dinv[:, None], input_precision="ieee")
    w_tile = tl.dot(tl.trans(n_mat), y_tile, input_precision="ieee")
    out_tile = w_tile * dinv[:, None]
    if EVEN_K:
        if EVEN_N:
            tl.store(X + r2 * stride_b + kcols[None, :], out_tile)
        else:
            rmask = rows[:, None] < N
            tl.store(X + r2 * stride_b + kcols[None, :], out_tile, mask=rmask)
    else:
        tl.store(X + r2 * stride_b + kcols[None, :], out_tile, mask=mask2d)


@libentry()
@triton.jit
def _chol_ainv_build_kernel(
    L_ptr,
    BINV_ptr,
    N,
    stride_l,
    batch_stride_l,
    BLK: tl.constexpr,
    NB: tl.constexpr,
    UPPER: tl.constexpr,
):
    """K1: per-diagonal-block inverses of T, computed once (k-independent).

    Grid: (batch, NB).  N % BLK == 0 is enforced by the dispatcher, so every
    tile access is exact and unmasked.  Stores Binv = (I + M)^-1 D^-1 and its
    transpose at BINV[(batch * NB + blk) * 2 * BLK * BLK + {0, BLK*BLK}], so
    K2 consumes both orientations through natural row-major loads.
    """
    pid_b = tl.program_id(0)
    pid_blk = tl.program_id(1)
    L = L_ptr + pid_b * batch_stride_l

    rows = tl.arange(0, BLK)
    r2 = rows[:, None]
    c2 = rows[None, :]
    row1 = pid_blk * BLK + rows
    t_blk = tl.load(L + row1[:, None] * stride_l + row1[None, :])
    if UPPER:
        t_blk = tl.trans(t_blk)
    diag = tl.load(L + row1 * stride_l + row1)
    dinv = 1.0 / diag
    m_fac = tl.where(c2 < r2, t_blk, 0.0) * dinv[:, None]

    n_mat = tl.where(r2 == c2, 1.0, 0.0)
    n_mat = n_mat - tl.dot(m_fac, n_mat, input_precision="ieee")
    m2 = tl.dot(m_fac, m_fac, input_precision="ieee")
    n_mat = n_mat + tl.dot(m2, n_mat, input_precision="ieee")
    m4 = tl.dot(m2, m2, input_precision="ieee")
    n_mat = n_mat + tl.dot(m4, n_mat, input_precision="ieee")
    m8 = tl.dot(m4, m4, input_precision="ieee")
    n_mat = n_mat + tl.dot(m8, n_mat, input_precision="ieee")
    if BLK >= 32:
        m16 = tl.dot(m8, m8, input_precision="ieee")
        n_mat = n_mat + tl.dot(m16, n_mat, input_precision="ieee")
    if BLK >= 64:
        m32 = tl.dot(m16, m16, input_precision="ieee")
        n_mat = n_mat + tl.dot(m32, n_mat, input_precision="ieee")
    # T^-1 = (I + M)^-1 D^-1: column scale by the reciprocal diagonal.
    binv = n_mat * dinv[None, :]

    base = (pid_b * NB + pid_blk) * 2 * BLK * BLK
    tl.store(BINV_ptr + base + r2 * BLK + c2, binv)
    tl.store(BINV_ptr + base + BLK * BLK + r2 * BLK + c2, tl.trans(binv))


@libentry()
@triton.jit
def _chol_ainv_walk_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    BINV_ptr,
    N,
    K,
    stride_l,
    stride_b,
    batch_stride_l,
    batch_stride_b,
    BLK: tl.constexpr,
    KS: tl.constexpr,
    BM: tl.constexpr,
    NB: tl.constexpr,
    NUM_K: tl.constexpr,
    UPPER: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """K2: two-phase block walk with precomputed diagonal-block inverses.

    Grid: (batch, k-slices).  Forward blocks ascend (solve T * Y = B, one
    Binv diag dot per block) and backward blocks descend (solve T^T * X = Y,
    one transposed-Binv diag dot per block); trailing/leading panel updates
    stay dot-based and accumulate in place in X, reading B only from the
    first block.  BM == BLK keeps every update tile exact (N % BLK == 0).
    The dispatcher guarantees K % KS == 0, so every access is mask-exact.

    Panel addressing mirrors with the orientation: with T = L the panels
    are T[m, k] = L[m, k] (forward) and T[k, m] (backward, transposed load);
    with T = U^T they swap -- T[m, k] = U[k, m] and T[k, m] = U[m, k].
    """
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)
    col_start = pid_k * KS
    L = L_ptr + pid_b * batch_stride_l
    Bs = B_ptr + pid_b * batch_stride_b
    X = X_ptr + pid_b * batch_stride_b
    binv_batch = BINV_ptr + pid_b * NB * 2 * BLK * BLK

    if NUM_K == 1:
        # Constant kcols fold the address vectors of every load/store below;
        # runtime column offsets otherwise keep accesses on the predicated
        # slow path.
        kcols = tl.arange(0, KS)
    else:
        kcols = col_start + tl.arange(0, KS)
    if EVEN_K:
        cmask = tl.full((KS,), 1, tl.int1)
    else:
        cmask = kcols < K
    cm2 = cmask[None, :]
    rows = tl.arange(0, BLK)
    rr = tl.arange(0, BM)

    # Forward phase: T * Y = B.
    for blk in tl.static_range(NB):
        row1 = blk * BLK
        r2 = (row1 + rows)[:, None]
        binv = tl.load(
            binv_batch + blk * 2 * BLK * BLK + rows[:, None] * BLK + rows[None, :]
        )
        # rhs comes from B for the first processed block, from the X
        # accumulation afterwards (a fully-masked load issues no traffic).
        if blk == 0:
            rhs = tl.load(Bs + r2 * stride_b + kcols[None, :], mask=cm2, other=0.0)
        else:
            rhs = tl.load(X + r2 * stride_b + kcols[None, :], mask=cm2, other=0.0)
        y = tl.dot(binv, rhs, input_precision="ieee")
        tl.store(X + r2 * stride_b + kcols[None, :], y, mask=cm2)
        for m_start in range(row1 + BLK, N, BM):
            rm = m_start + rr
            if UPPER:
                # T[m, k] = U[k, m]: natural [BLK, BM] load, one transpose.
                a_nat = tl.load(L + (row1 + rows)[:, None] * stride_l + rm[None, :])
                acc = tl.dot(tl.trans(a_nat), y, input_precision="ieee")
            else:
                a_nat = tl.load(L + rm[:, None] * stride_l + (row1 + rows)[None, :])
                acc = tl.dot(a_nat, y, input_precision="ieee")
            if blk == 0:
                b_curr = tl.load(
                    Bs + rm[:, None] * stride_b + kcols[None, :], mask=cm2, other=0.0
                )
            else:
                b_curr = tl.load(
                    X + rm[:, None] * stride_b + kcols[None, :], mask=cm2, other=0.0
                )
            tl.store(
                X + rm[:, None] * stride_b + kcols[None, :], b_curr - acc, mask=cm2
            )
        # This block's update must be visible to the next block's diag load.
        tl.debug_barrier()

    # Backward phase: T^T * X = Y.
    for kk in tl.static_range(NB):
        blk = NB - 1 - kk
        row1 = blk * BLK
        r2 = (row1 + rows)[:, None]
        rhs = tl.load(X + r2 * stride_b + kcols[None, :], mask=cm2, other=0.0)
        binv_t = tl.load(
            binv_batch + (blk * 2 + 1) * BLK * BLK + rows[:, None] * BLK + rows[None, :]
        )
        xk = tl.dot(binv_t, rhs, input_precision="ieee")
        tl.store(X + r2 * stride_b + kcols[None, :], xk, mask=cm2)
        for m_start in range(0, row1, BM):
            rm = m_start + rr
            # (T^T)[m, blk] = T[blk, m]: lower loads L[k, m] transposed, upper
            # loads U[m, k] directly (the orientation mirror of the forward
            # panels above).
            if UPPER:
                a_nat = tl.load(L + rm[:, None] * stride_l + (row1 + rows)[None, :])
                acc = tl.dot(a_nat, xk, input_precision="ieee")
            else:
                a_nat = tl.load(L + (row1 + rows)[:, None] * stride_l + rm[None, :])
                acc = tl.dot(tl.trans(a_nat), xk, input_precision="ieee")
            x_curr = tl.load(
                X + rm[:, None] * stride_b + kcols[None, :], mask=cm2, other=0.0
            )
            tl.store(
                X + rm[:, None] * stride_b + kcols[None, :], x_curr - acc, mask=cm2
            )
        tl.debug_barrier()


@libentry()
@triton.jit
def _chol_complex_neumann_1blk_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    N,
    K,
    batch_stride_L,
    batch_stride_B,
    stride_L_row,
    stride_L_col,
    stride_B_row,
    stride_B_col,
    BN: tl.constexpr,
    KS: tl.constexpr,
    UPPER: tl.constexpr,
    STORAGE_CONJ: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """Fused two-phase complex Neumann cholesky solve for N <= BN <= 64.

    Complex arithmetic is split into interleaved real/imaginary tiles so
    every complex product lowers to four ieee fp32 tl.dot's.  X = D^-1 *
    n^H * n * D^-1 * B with n = (I + M)^-1, M = D^-1 * strictly_lower(T)
    and T = the lower first-phase factor (L, or U^H for upper orientation).
    """
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)
    col_start = pid_k * KS

    L = L_ptr + pid_b * batch_stride_L
    Bs = B_ptr + pid_b * batch_stride_B
    X = X_ptr + pid_b * batch_stride_B

    rows = tl.arange(0, BN)
    r2 = rows[:, None]
    c2 = rows[None, :]
    if EVEN_N:
        t_off = r2 * stride_L_row + c2 * stride_L_col
        tr = tl.load(L + t_off)
        ti = tl.load(L + t_off + 1)
    else:
        in_blk = (r2 < N) & (c2 < N)
        t_off = tl.where(in_blk, r2 * stride_L_row + c2 * stride_L_col, 0)
        tr = tl.load(L + t_off, mask=in_blk, other=0.0)
        ti = tl.load(L + t_off + 1, mask=in_blk, other=0.0)
    if STORAGE_CONJ:
        ti = -ti
    if UPPER:
        # T = F^H: transpose and conjugate the upper factor tile.
        tr = tl.trans(tr)
        ti = -tl.trans(ti)
    in_d = rows < N
    diag = tl.load(
        L + tl.where(in_d, rows * (stride_L_row + stride_L_col), 0),
        mask=in_d,
        other=1.0,
    )
    dinv = 1.0 / diag
    low = c2 < r2
    mr = tl.where(low, tr, 0.0) * dinv[:, None]
    mi = tl.where(low, ti, 0.0) * dinv[:, None]

    # n = (I + M)^-1 = (I - M)(I + M^2)(I + M^4)..., complex.
    nr = tl.where(r2 == c2, 1.0, 0.0)
    ni = nr * 0.0
    pr = tl.dot(mr, nr, input_precision="ieee") - tl.dot(mi, ni, input_precision="ieee")
    pi = tl.dot(mr, ni, input_precision="ieee") + tl.dot(mi, nr, input_precision="ieee")
    nr -= pr
    ni -= pi
    m2r = tl.dot(mr, mr, input_precision="ieee") - tl.dot(
        mi, mi, input_precision="ieee"
    )
    m2i = tl.dot(mr, mi, input_precision="ieee") + tl.dot(
        mi, mr, input_precision="ieee"
    )
    pr = tl.dot(m2r, nr, input_precision="ieee") - tl.dot(
        m2i, ni, input_precision="ieee"
    )
    pi = tl.dot(m2r, ni, input_precision="ieee") + tl.dot(
        m2i, nr, input_precision="ieee"
    )
    nr += pr
    ni += pi
    m4r = tl.dot(m2r, m2r, input_precision="ieee") - tl.dot(
        m2i, m2i, input_precision="ieee"
    )
    m4i = tl.dot(m2r, m2i, input_precision="ieee") + tl.dot(
        m2i, m2r, input_precision="ieee"
    )
    pr = tl.dot(m4r, nr, input_precision="ieee") - tl.dot(
        m4i, ni, input_precision="ieee"
    )
    pi = tl.dot(m4r, ni, input_precision="ieee") + tl.dot(
        m4i, nr, input_precision="ieee"
    )
    nr += pr
    ni += pi
    m8r = tl.dot(m4r, m4r, input_precision="ieee") - tl.dot(
        m4i, m4i, input_precision="ieee"
    )
    m8i = tl.dot(m4r, m4i, input_precision="ieee") + tl.dot(
        m4i, m4r, input_precision="ieee"
    )
    pr = tl.dot(m8r, nr, input_precision="ieee") - tl.dot(
        m8i, ni, input_precision="ieee"
    )
    pi = tl.dot(m8r, ni, input_precision="ieee") + tl.dot(
        m8i, nr, input_precision="ieee"
    )
    nr += pr
    ni += pi
    if BN >= 32:
        m16r = tl.dot(m8r, m8r, input_precision="ieee") - tl.dot(
            m8i, m8i, input_precision="ieee"
        )
        m16i = tl.dot(m8r, m8i, input_precision="ieee") + tl.dot(
            m8i, m8r, input_precision="ieee"
        )
        pr = tl.dot(m16r, nr, input_precision="ieee") - tl.dot(
            m16i, ni, input_precision="ieee"
        )
        pi = tl.dot(m16r, ni, input_precision="ieee") + tl.dot(
            m16i, nr, input_precision="ieee"
        )
        nr += pr
        ni += pi
    if BN >= 64:
        m32r = tl.dot(m16r, m16r, input_precision="ieee") - tl.dot(
            m16i, m16i, input_precision="ieee"
        )
        m32i = tl.dot(m16r, m16i, input_precision="ieee") + tl.dot(
            m16i, m16r, input_precision="ieee"
        )
        pr = tl.dot(m32r, nr, input_precision="ieee") - tl.dot(
            m32i, ni, input_precision="ieee"
        )
        pi = tl.dot(m32r, ni, input_precision="ieee") + tl.dot(
            m32i, nr, input_precision="ieee"
        )
        nr += pr
        ni += pi

    kcols = col_start + tl.arange(0, KS)
    b_off = r2 * stride_B_row + kcols[None, :] * stride_B_col
    if EVEN_K:
        if EVEN_N:
            xr = tl.load(Bs + b_off)
            xi = tl.load(Bs + b_off + 1)
        else:
            rmask = rows[:, None] < N
            xr = tl.load(Bs + tl.where(rmask, b_off, 0), mask=rmask, other=0.0)
            xi = tl.load(Bs + tl.where(rmask, b_off, 0) + 1, mask=rmask, other=0.0)
    else:
        cmask = kcols < K
        mask2d = (rows[:, None] < N) & cmask[None, :]
        xr = tl.load(Bs + tl.where(mask2d, b_off, 0), mask=mask2d, other=0.0)
        xi = tl.load(Bs + tl.where(mask2d, b_off, 0) + 1, mask=mask2d, other=0.0)
    zr = xr * dinv[:, None]
    zi = xi * dinv[:, None]
    # Forward: Y = n * D^-1 * B; backward: X = D^-1 * n^H * Y.
    yr = tl.dot(nr, zr, input_precision="ieee") - tl.dot(ni, zi, input_precision="ieee")
    yi = tl.dot(nr, zi, input_precision="ieee") + tl.dot(ni, zr, input_precision="ieee")
    nhr = tl.trans(nr)
    nhi = -tl.trans(ni)
    wr = tl.dot(nhr, yr, input_precision="ieee") - tl.dot(
        nhi, yi, input_precision="ieee"
    )
    wi = tl.dot(nhr, yi, input_precision="ieee") + tl.dot(
        nhi, yr, input_precision="ieee"
    )
    outr = wr * dinv[:, None]
    outi = wi * dinv[:, None]
    if EVEN_K:
        if EVEN_N:
            tl.store(X + b_off, outr)
            tl.store(X + b_off + 1, outi)
        else:
            rmask = rows[:, None] < N
            tl.store(X + b_off, outr, mask=rmask)
            tl.store(X + b_off + 1, outi, mask=rmask)
    else:
        tl.store(X + b_off, outr, mask=mask2d)
        tl.store(X + b_off + 1, outi, mask=mask2d)


@libentry()
@triton.jit
def _chol_complex_ainv_build_kernel(
    L_ptr,
    BINV_ptr,
    N,
    batch_stride_L,
    stride_L_row,
    stride_L_col,
    BLK: tl.constexpr,
    NB: tl.constexpr,
    UPPER: tl.constexpr,
    STORAGE_CONJ: tl.constexpr,
):
    """K1: complex per-diagonal-block inverses of T (k-independent).

    Grid: (batch, NB); N % BLK == 0 enforced by the dispatcher.  BINV holds
    two interleaved [BLK, BLK] complex halves per block: Binv = (I+M)^-1
    D^-1 and Binv^H, both in natural row-major so K2 never transposes.
    """
    pid_b = tl.program_id(0)
    pid_blk = tl.program_id(1)
    L = L_ptr + pid_b * batch_stride_L

    rows = tl.arange(0, BLK)
    r2 = rows[:, None]
    c2 = rows[None, :]
    row1 = pid_blk * BLK + rows
    t_off = row1[:, None] * stride_L_row + row1[None, :] * stride_L_col
    tr = tl.load(L + t_off)
    ti = tl.load(L + t_off + 1)
    if STORAGE_CONJ:
        ti = -ti
    if UPPER:
        tr = tl.trans(tr)
        ti = -tl.trans(ti)
    diag = tl.load(L + row1 * (stride_L_row + stride_L_col))
    dinv = 1.0 / diag
    low = c2 < r2
    mr = tl.where(low, tr, 0.0) * dinv[:, None]
    mi = tl.where(low, ti, 0.0) * dinv[:, None]

    nr = tl.where(r2 == c2, 1.0, 0.0)
    ni = nr * 0.0
    pr = tl.dot(mr, nr, input_precision="ieee") - tl.dot(mi, ni, input_precision="ieee")
    pi = tl.dot(mr, ni, input_precision="ieee") + tl.dot(mi, nr, input_precision="ieee")
    nr -= pr
    ni -= pi
    m2r = tl.dot(mr, mr, input_precision="ieee") - tl.dot(
        mi, mi, input_precision="ieee"
    )
    m2i = tl.dot(mr, mi, input_precision="ieee") + tl.dot(
        mi, mr, input_precision="ieee"
    )
    pr = tl.dot(m2r, nr, input_precision="ieee") - tl.dot(
        m2i, ni, input_precision="ieee"
    )
    pi = tl.dot(m2r, ni, input_precision="ieee") + tl.dot(
        m2i, nr, input_precision="ieee"
    )
    nr += pr
    ni += pi
    m4r = tl.dot(m2r, m2r, input_precision="ieee") - tl.dot(
        m2i, m2i, input_precision="ieee"
    )
    m4i = tl.dot(m2r, m2i, input_precision="ieee") + tl.dot(
        m2i, m2r, input_precision="ieee"
    )
    pr = tl.dot(m4r, nr, input_precision="ieee") - tl.dot(
        m4i, ni, input_precision="ieee"
    )
    pi = tl.dot(m4r, ni, input_precision="ieee") + tl.dot(
        m4i, nr, input_precision="ieee"
    )
    nr += pr
    ni += pi
    m8r = tl.dot(m4r, m4r, input_precision="ieee") - tl.dot(
        m4i, m4i, input_precision="ieee"
    )
    m8i = tl.dot(m4r, m4i, input_precision="ieee") + tl.dot(
        m4i, m4r, input_precision="ieee"
    )
    pr = tl.dot(m8r, nr, input_precision="ieee") - tl.dot(
        m8i, ni, input_precision="ieee"
    )
    pi = tl.dot(m8r, ni, input_precision="ieee") + tl.dot(
        m8i, nr, input_precision="ieee"
    )
    nr += pr
    ni += pi
    m16r = tl.dot(m8r, m8r, input_precision="ieee") - tl.dot(
        m8i, m8i, input_precision="ieee"
    )
    m16i = tl.dot(m8r, m8i, input_precision="ieee") + tl.dot(
        m8i, m8r, input_precision="ieee"
    )
    pr = tl.dot(m16r, nr, input_precision="ieee") - tl.dot(
        m16i, ni, input_precision="ieee"
    )
    pi = tl.dot(m16r, ni, input_precision="ieee") + tl.dot(
        m16i, nr, input_precision="ieee"
    )
    nr += pr
    ni += pi
    m32r = tl.dot(m16r, m16r, input_precision="ieee") - tl.dot(
        m16i, m16i, input_precision="ieee"
    )
    m32i = tl.dot(m16r, m16i, input_precision="ieee") + tl.dot(
        m16i, m16r, input_precision="ieee"
    )
    pr = tl.dot(m32r, nr, input_precision="ieee") - tl.dot(
        m32i, ni, input_precision="ieee"
    )
    pi = tl.dot(m32r, ni, input_precision="ieee") + tl.dot(
        m32i, nr, input_precision="ieee"
    )
    nr += pr
    ni += pi
    # T^-1 = (I + M)^-1 D^-1: complex tile, real column scale.
    br = nr * dinv[None, :]
    bi = ni * dinv[None, :]

    tile = 2 * BLK * BLK
    base = (pid_b * NB + pid_blk) * 2 * tile
    o = rows[:, None] * (2 * BLK) + rows[None, :] * 2
    tl.store(BINV_ptr + base + o, br)
    tl.store(BINV_ptr + base + o + 1, bi)
    # Binv^H = conj(T^-1)^T stored naturally.
    tl.store(BINV_ptr + base + tile + o, tl.trans(br))
    tl.store(BINV_ptr + base + tile + o + 1, -tl.trans(bi))


@libentry()
@triton.jit
def _chol_complex_ainv_walk_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    BINV_ptr,
    N,
    K,
    batch_stride_L,
    batch_stride_B,
    stride_L_row,
    stride_L_col,
    stride_B_row,
    stride_B_col,
    BLK: tl.constexpr,
    KS: tl.constexpr,
    BM: tl.constexpr,
    NB: tl.constexpr,
    NUM_K: tl.constexpr,
    UPPER: tl.constexpr,
    STORAGE_CONJ: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """K2: complex two-phase block walk with precomputed block inverses.

    Mirrors _chol_ainv_walk_kernel with complex (interleaved) tiles: each
    diag solve and panel update is four ieee dots.  Backward updates use
    (T[k, m])^H panels: natural load plus one register transpose/conjugate.
    The dispatcher guarantees K % KS == 0.
    """
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)
    col_start = pid_k * KS
    L = L_ptr + pid_b * batch_stride_L
    Bs = B_ptr + pid_b * batch_stride_B
    X = X_ptr + pid_b * batch_stride_B
    binv_batch = BINV_ptr + pid_b * NB * 2 * 2 * BLK * BLK

    if NUM_K == 1:
        kcols = tl.arange(0, KS)
    else:
        kcols = col_start + tl.arange(0, KS)
    if EVEN_K:
        cmask = tl.full((KS,), 1, tl.int1)
    else:
        cmask = kcols < K
    cm2 = cmask[None, :]
    rows = tl.arange(0, BLK)
    rr = tl.arange(0, BM)
    tile = 2 * BLK * BLK

    # Forward phase: T * Y = B.
    for blk in tl.static_range(NB):
        row1 = blk * BLK
        r2 = (row1 + rows)[:, None]
        boff = r2 * stride_B_row + kcols[None, :] * stride_B_col
        bbase = (
            binv_batch + blk * 2 * tile + rows[:, None] * (2 * BLK) + rows[None, :] * 2
        )
        br = tl.load(bbase)
        bi = tl.load(bbase + 1)
        if blk == 0:
            rhs_r = tl.load(Bs + boff, mask=cm2, other=0.0)
            rhs_i = tl.load(Bs + boff + 1, mask=cm2, other=0.0)
        else:
            rhs_r = tl.load(X + boff, mask=cm2, other=0.0)
            rhs_i = tl.load(X + boff + 1, mask=cm2, other=0.0)
        yr = tl.dot(br, rhs_r, input_precision="ieee") - tl.dot(
            bi, rhs_i, input_precision="ieee"
        )
        yi = tl.dot(br, rhs_i, input_precision="ieee") + tl.dot(
            bi, rhs_r, input_precision="ieee"
        )
        tl.store(X + boff, yr, mask=cm2)
        tl.store(X + boff + 1, yi, mask=cm2)
        for m_start in range(row1 + BLK, N, BM):
            rm = m_start + rr
            # Panel T[m, k]: lower = F[m, k] natural; upper = conj(F[k, m])
            # loaded natural then transposed/conjugated.
            if UPPER:
                a_off = (row1 + rows)[:, None] * stride_L_row + rm[
                    None, :
                ] * stride_L_col
                ar = tl.load(L + a_off)
                ai = tl.load(L + a_off + 1)
                if STORAGE_CONJ:
                    ai = -ai
                ar = tl.trans(ar)
                ai = -tl.trans(ai)
            else:
                a_off = (
                    rm[:, None] * stride_L_row + (row1 + rows)[None, :] * stride_L_col
                )
                ar = tl.load(L + a_off)
                ai = tl.load(L + a_off + 1)
                if STORAGE_CONJ:
                    ai = -ai
            ur = tl.dot(ar, yr, input_precision="ieee") - tl.dot(
                ai, yi, input_precision="ieee"
            )
            ui = tl.dot(ar, yi, input_precision="ieee") + tl.dot(
                ai, yr, input_precision="ieee"
            )
            m_off = rm[:, None] * stride_B_row + kcols[None, :] * stride_B_col
            if blk == 0:
                cr = tl.load(Bs + m_off, mask=cm2, other=0.0)
                ci = tl.load(Bs + m_off + 1, mask=cm2, other=0.0)
            else:
                cr = tl.load(X + m_off, mask=cm2, other=0.0)
                ci = tl.load(X + m_off + 1, mask=cm2, other=0.0)
            tl.store(X + m_off, cr - ur, mask=cm2)
            tl.store(X + m_off + 1, ci - ui, mask=cm2)
        tl.debug_barrier()

    # Backward phase: T^H * X = Y.
    for kk in tl.static_range(NB):
        blk = NB - 1 - kk
        row1 = blk * BLK
        r2 = (row1 + rows)[:, None]
        boff = r2 * stride_B_row + kcols[None, :] * stride_B_col
        rhs_r = tl.load(X + boff, mask=cm2, other=0.0)
        rhs_i = tl.load(X + boff + 1, mask=cm2, other=0.0)
        hbase = (
            binv_batch
            + (blk * 2 + 1) * tile
            + rows[:, None] * (2 * BLK)
            + rows[None, :] * 2
        )
        hr = tl.load(hbase)
        hi = tl.load(hbase + 1)
        xkr = tl.dot(hr, rhs_r, input_precision="ieee") - tl.dot(
            hi, rhs_i, input_precision="ieee"
        )
        xki = tl.dot(hr, rhs_i, input_precision="ieee") + tl.dot(
            hi, rhs_r, input_precision="ieee"
        )
        tl.store(X + boff, xkr, mask=cm2)
        tl.store(X + boff + 1, xki, mask=cm2)
        for m_start in range(0, row1, BM):
            rm = m_start + rr
            # Panel (T^H)[m, blk] = conj(T[blk, m]): lower loads F[k, m]
            # transposed/conjugated; upper = F[m, k] natural (T^H = F).
            if UPPER:
                a_off = (
                    rm[:, None] * stride_L_row + (row1 + rows)[None, :] * stride_L_col
                )
                ar = tl.load(L + a_off)
                ai = tl.load(L + a_off + 1)
                if STORAGE_CONJ:
                    ai = -ai
            else:
                a_off = (row1 + rows)[:, None] * stride_L_row + rm[
                    None, :
                ] * stride_L_col
                ar = tl.load(L + a_off)
                ai = tl.load(L + a_off + 1)
                if STORAGE_CONJ:
                    ai = -ai
                ar = tl.trans(ar)
                ai = -tl.trans(ai)
            ur = tl.dot(ar, xkr, input_precision="ieee") - tl.dot(
                ai, xki, input_precision="ieee"
            )
            ui = tl.dot(ar, xki, input_precision="ieee") + tl.dot(
                ai, xkr, input_precision="ieee"
            )
            m_off = rm[:, None] * stride_B_row + kcols[None, :] * stride_B_col
            cr = tl.load(X + m_off, mask=cm2, other=0.0)
            ci = tl.load(X + m_off + 1, mask=cm2, other=0.0)
            tl.store(X + m_off, cr - ur, mask=cm2)
            tl.store(X + m_off + 1, ci - ui, mask=cm2)
        tl.debug_barrier()


def _get_blocked_tile_config(dtype, nrhs, batch_size=1):
    """Return dtype-specific tile sizes for blocked kernels.

    GCU400 measured conclusions (single-variable A/B, same wrapper end-to-end setup):

    - A 32x32 panel is always optimal; enlarging BLOCK_K/BLOCK_M only slows things down (at N=256
      BLOCK_M=128 is 4x slower: the tl.dot width of each row-wise edge update overwhelms registers).
    - BLOCK_RHS is the real master switch and its direction depends on nrhs / batch: the grid is
      (batch, ceil(nrhs/BLOCK_RHS)) and every CTA re-walks the whole factor triangle, so a narrow
      block trades repeated traversal for parallelism. With a single batch and small nrhs the
      parallelism gain far outweighs the repetition cost (N=256/nrhs=4: 0.83ms->0.58ms,
      nrhs=8: 0.83ms→0.60ms, nrhs=16: 0.84ms→0.62ms, N=128/nrhs=16:
      0.38ms->0.27ms); with a single batch and large nrhs the repetition becomes the main cost and
      the block must widen (nrhs=128: 1.73ms->0.98ms, N=128/nrhs=128: 0.79ms->0.45ms).
    - But when the batch already provides parallelism, narrowing RHS multiplies the CTA count by
      batch x ceil(nrhs/RHS): batch=32/nrhs=16 goes 1.11ms -> 2.57ms and batch=8/nrhs=16 goes
      0.79ms -> 1.57ms. Narrow blocks are therefore enabled only for a single batch; multiple
      batches keep BLOCK_RHS=4.

    fp64 keeps 16x32 / BLOCK_RHS=8 unchanged (the fp64 path is not used on GCU400 yet).
    """
    if dtype == torch.float64:
        blk_k, blk_m, blk_rhs = 16, 32, 8
    else:
        blk_k, blk_m = 32, 32
        if batch_size > 1:
            blk_rhs = 4
        elif nrhs >= 128:
            blk_rhs = 8
        elif nrhs <= 16:
            blk_rhs = 1
        else:
            blk_rhs = 4
    return {"BLOCK_K": blk_k, "BLOCK_M": blk_m, "BLOCK_RHS": blk_rhs}


def _get_blocked_warp_config(dtype):
    """Return warp/stage config for blocked kernels based on dtype."""
    if dtype == torch.float64:
        return {"num_warps": 1, "num_stages": 2}
    return {"num_warps": 1, "num_stages": 3}


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
            "num_warps": 1,
            "num_stages": 1,
        }
    if N >= 256:
        return {
            "BLOCK_K": 32,
            "BLOCK_M": 128,
            "num_warps": 1,
            "num_stages": 1,
        }
    if N >= 128:
        # GCU400 re-test: raising BLOCK_M from 32 to 128 changes N=128 single-batch from 0.2157ms→0.2025ms and
        # batch 8 from 0.2262ms→0.2125ms; N=256 already used BLOCK_M=128 (0.4219ms is optimal,
        # while BLOCK_M=32 is 0.4681ms). BLOCK_K=16/64 are both slower, so keep 32.
        return {
            "BLOCK_K": 32,
            "BLOCK_M": 128,
            "num_warps": 1,
            "num_stages": 1,
        }
    return {
        "BLOCK_K": 32,
        "BLOCK_M": 64,
        "num_warps": 1,
        "num_stages": 1,
    }


def _get_small_gather_launch_config(dtype, N):
    """Return pinned H20 winners for the small gather kernel.

    fp64 at N == 32 gains from a second warp sharing the wider fp64 row
    updates; everything else is latency-bound and best with one warp.
    """
    if dtype == torch.float64 and N > 16:
        return {"num_warps": 1, "num_stages": 1}
    return {"num_warps": 1, "num_stages": 1}


def _get_complex_blocked_launch_config(dtype, N, nrhs):
    """Return shape/dtype-specific configs for complex multi-RHS solves.

    complex128 keeps the 16x32x8 configuration for nrhs >= 8, which already
    beats Torch on every measured shape. For nrhs <= 4 it switches to 32-row
    diagonal blocks with BLOCK_RHS=4: an 8-wide RHS tile wastes half of every
    dot on padding when nrhs == 4, and 32-row blocks halve the serial
    block/barrier count. complex64 expands only the N=256 panel from 32 to 64
    rows, halving the number of four-dot panel-update groups without the
    register pressure of a 128-row complex tile.
    """
    if dtype == torch.complex128:
        if nrhs <= 4:
            return {
                "BLOCK_K": 32,
                "BLOCK_M": 32,
                "BLOCK_RHS": 4,
                "num_warps": 1,
                "num_stages": 1,
            }
        return {
            "BLOCK_K": 16,
            "BLOCK_M": 32,
            "BLOCK_RHS": 8,
            "num_warps": 1,
            "num_stages": 1,
        }
    # complex64: BLOCK_RHS=4 is a pathological configuration for N>=128. GCU400 measurements show
    # (single-batch) at N=128, RHS=4 gives nrhs=16: 2.76ms and nrhs=32: 2.53ms,
    # and nrhs=128: 5.94ms; after switching to RHS=2 (nrhs<64) or RHS=8 (nrhs>=64),
    # these drop to 0.68ms / 0.74ms / 0.99ms (3.7x / 3.4x / 6.0x); for batch 8,
    # (128,16) also drops from 6.47ms to 2.34ms. At N=64, RHS=4 is still reasonable
    # (with batch=32/nrhs=16, RHS=2 is instead 1.75x slower because the CTA count is amplified by batch),
    # so only change N>=128 cases and keep the original value for N=64.
    if N >= 128:
        block_rhs = 8 if nrhs >= 64 else 2
    else:
        block_rhs = 4
    return {
        "BLOCK_K": 32,
        "BLOCK_M": 64 if N >= 256 else 32,
        "BLOCK_RHS": block_rhs,
        "num_warps": 1,
        "num_stages": 2,
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
            "num_warps": 1,
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
        "num_warps": 1,
        "num_stages": 1,
    }


def _cholesky_solve_complex(B, L, upper, batch_shape, N, nrhs, X=None):
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
        if B.dtype == torch.complex64 and N <= 64:
            # complex64 single-block fused Neumann solve.
            block_n = 16 if N <= 16 else (32 if N <= 32 else 64)
            ks = 16
            num_k = (nrhs + ks - 1) // ks
            while num_k > 255:
                ks *= 2
                num_k = (nrhs + ks - 1) // ks
            _chol_complex_neumann_1blk_kernel[(batch_size, num_k)](
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
                BN=block_n,
                KS=ks,
                UPPER=effective_upper,
                STORAGE_CONJ=storage_conj,
                EVEN_N=(N == block_n),
                EVEN_K=(nrhs % ks == 0),
                num_warps=1,
            )
        elif (
            B.dtype == torch.complex64
            and N % 64 == 0
            and N <= 512
            and nrhs % 16 == 0
            and nrhs >= 16
        ):
            # complex64 two-launch block-inverse walk (K % KS == 0 exact).
            num_blk = N // 64
            ks = 16
            num_k = (nrhs + ks - 1) // ks
            while num_k > 255:
                ks *= 2
                num_k = (nrhs + ks - 1) // ks
            binv = torch.empty(
                (batch_size * num_blk * 4 * 64 * 64,),
                dtype=B_real.dtype,
                device=B_real.device,
            )
            _chol_complex_ainv_build_kernel[(batch_size, num_blk)](
                L_real,
                binv,
                N,
                L_real.stride(0),
                L_real.stride(1),
                L_real.stride(2),
                BLK=64,
                NB=num_blk,
                UPPER=effective_upper,
                STORAGE_CONJ=storage_conj,
                num_warps=1,
            )
            _chol_complex_ainv_walk_kernel[(batch_size, num_k)](
                L_real,
                B_real,
                X_real,
                binv,
                N,
                nrhs,
                L_real.stride(0),
                B_real.stride(0),
                L_real.stride(1),
                L_real.stride(2),
                B_real.stride(1),
                B_real.stride(2),
                BLK=64,
                KS=ks,
                BM=64,
                NB=num_blk,
                NUM_K=num_k,
                UPPER=effective_upper,
                STORAGE_CONJ=storage_conj,
                EVEN_K=(nrhs % ks == 0),
                num_warps=1,
            )
        elif (N <= 32 and nrhs <= 8) or (N < 64 and nrhs == 1):
            block_n = triton.next_power_of_2(N)
            block_rhs = triton.next_power_of_2(nrhs)
            # gcu400 lowers tl.gather only for a single warp (see the module
            # note), so every launch here uses one warp regardless of dtype.
            num_warps = 1
            cholesky_solve_complex_small_gather_kernel[(batch_size,)](
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
                num_warps=num_warps,
                num_stages=1,
            )
        elif N >= 64 and N % 32 == 0 and nrhs >= 4:
            is_double = B.dtype == torch.complex128
            config = _get_complex_blocked_launch_config(B.dtype, N, nrhs)
            grid = (batch_size, triton.cdiv(nrhs, config["BLOCK_RHS"]))
            cholesky_solve_complex_blocked_kernel[grid](
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
                upper=effective_upper,
                storage_conj=storage_conj,
                IS_DOUBLE=is_double,
                **config,
            )
        elif nrhs == 1 and N >= 64 and N % 32 == 0:
            if B.dtype == torch.complex128 and N >= 256:
                # complex128 single-RHS at N >= 256: precompute the inverse
                # of every 16-row diagonal sub-block (fully parallel across
                # CTAs), then solve with two chained half-block matvecs per
                # 32-row block instead of the serial per-row gather chain,
                # whose fp64 smem round trips dominate at this size.
                sub_k = 16
                block_k = 32
                T_scratch = torch.empty(
                    (batch_size, N, sub_k, 2),
                    dtype=B_real.dtype,
                    device=B.device,
                )
                cholesky_solve_complex_invert_blocks_kernel[(batch_size, N // sub_k)](
                    L_real,
                    T_scratch,
                    N,
                    L_real.stride(0),
                    L_real.stride(1),
                    L_real.stride(2),
                    BLOCK_K=sub_k,
                    upper=effective_upper,
                    storage_conj=storage_conj,
                    num_warps=1,
                    num_stages=1,
                )
                cholesky_solve_complex_single_rhs_blocked_kernel[(batch_size,)](
                    L_real,
                    B_real,
                    X_real,
                    T_scratch,
                    N,
                    L_real.stride(0),
                    B_real.stride(0),
                    L_real.stride(1),
                    L_real.stride(2),
                    B_real.stride(1),
                    BLOCK_K=block_k,
                    BLOCK_M=32,
                    upper=effective_upper,
                    storage_conj=storage_conj,
                    num_warps=1,
                    num_stages=1,
                )
            else:
                config = _get_complex_single_rhs_launch_config(B.dtype, N)
                cholesky_solve_complex_single_rhs_gather_kernel[(batch_size,)](
                    L_real,
                    B_real,
                    X_real,
                    N,
                    L_real.stride(0),
                    B_real.stride(0),
                    L_real.stride(1),
                    L_real.stride(2),
                    B_real.stride(1),
                    upper=effective_upper,
                    storage_conj=storage_conj,
                    **config,
                )
        else:
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
    logger.debug("GEMS_ENFLAME CHOLESKY_SOLVE")
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
        return _cholesky_solve_complex(B, L, upper, batch_shape, N, nrhs, X=_out)

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
        if dtype_flag == 0 and N <= 64:
            # fp32 single-block fused Neumann solve (register-resident tile).
            block_n = 16 if N <= 16 else (32 if N <= 32 else 64)
            ks = 16
            num_k = (nrhs + ks - 1) // ks
            while num_k > 255:
                ks *= 2
                num_k = (nrhs + ks - 1) // ks
            _chol_neumann_1blk_kernel[(batch_size, num_k)](
                L_kernel,
                B_kernel,
                X_kernel,
                N,
                nrhs,
                stride_L,
                stride_B,
                batch_stride_L,
                batch_stride_B,
                BN=block_n,
                KS=ks,
                UPPER=effective_upper,
                EVEN_N=(N == block_n),
                EVEN_K=(nrhs % ks == 0),
                num_warps=1,
            )
        elif (
            dtype_flag == 0
            and N % 64 == 0
            and N <= 512
            and nrhs % 16 == 0
            and nrhs >= 16
        ):
            # fp32 two-launch block-inverse walk: K1 builds every diagonal
            # block inverse in parallel, K2 walks both substitution phases
            # with one diag dot per block.  K % KS == 0 keeps every k-slice
            # exact (no ragged column masks, no stride-1 specialization).
            num_blk = N // 64
            ks = 16
            num_k = (nrhs + ks - 1) // ks
            while num_k > 255:
                ks *= 2
                num_k = (nrhs + ks - 1) // ks
            binv = torch.empty(
                (batch_size * num_blk * 2 * 64 * 64,),
                dtype=B.dtype,
                device=B.device,
            )
            _chol_ainv_build_kernel[(batch_size, num_blk)](
                L_kernel,
                binv,
                N,
                stride_L,
                batch_stride_L,
                BLK=64,
                NB=num_blk,
                UPPER=effective_upper,
                num_warps=1,
            )
            _chol_ainv_walk_kernel[(batch_size, num_k)](
                L_kernel,
                B_kernel,
                X_kernel,
                binv,
                N,
                nrhs,
                stride_L,
                stride_B,
                batch_stride_L,
                batch_stride_B,
                BLK=64,
                KS=ks,
                BM=64,
                NB=num_blk,
                NUM_K=num_k,
                UPPER=effective_upper,
                EVEN_K=(nrhs % ks == 0),
                num_warps=1,
            )
        elif _can_use_blocked_path(N, nrhs):
            tile = _get_blocked_tile_config(B.dtype, nrhs, batch_size)
            warp = _get_blocked_warp_config(B.dtype)
            grid = (batch_size, triton.cdiv(nrhs, tile["BLOCK_RHS"]))
            if effective_upper:
                if dtype_flag == 1:
                    cholesky_solve_blocked_upper_fp64_kernel[grid](
                        L_kernel,
                        B_kernel,
                        X_kernel,
                        N,
                        nrhs,
                        batch_stride_L,
                        batch_stride_B,
                        stride_L,
                        stride_B,
                        BLOCK_K=tile["BLOCK_K"],
                        BLOCK_M=tile["BLOCK_M"],
                        BLOCK_RHS=tile["BLOCK_RHS"],
                        **warp,
                    )
                else:
                    cholesky_solve_blocked_upper_kernel[grid](
                        L_kernel,
                        B_kernel,
                        X_kernel,
                        N,
                        nrhs,
                        batch_stride_L,
                        batch_stride_B,
                        stride_L,
                        stride_B,
                        BLOCK_K=tile["BLOCK_K"],
                        BLOCK_M=tile["BLOCK_M"],
                        BLOCK_RHS=tile["BLOCK_RHS"],
                        **warp,
                    )
            else:
                if dtype_flag == 1:
                    cholesky_solve_blocked_lower_fp64_kernel[grid](
                        L_kernel,
                        B_kernel,
                        X_kernel,
                        N,
                        nrhs,
                        batch_stride_L,
                        batch_stride_B,
                        stride_L,
                        stride_B,
                        BLOCK_K=tile["BLOCK_K"],
                        BLOCK_M=tile["BLOCK_M"],
                        BLOCK_RHS=tile["BLOCK_RHS"],
                        **warp,
                    )
                else:
                    cholesky_solve_blocked_lower_kernel[grid](
                        L_kernel,
                        B_kernel,
                        X_kernel,
                        N,
                        nrhs,
                        batch_stride_L,
                        batch_stride_B,
                        stride_L,
                        stride_B,
                        BLOCK_K=tile["BLOCK_K"],
                        BLOCK_M=tile["BLOCK_M"],
                        BLOCK_RHS=tile["BLOCK_RHS"],
                        **warp,
                    )
        elif _can_use_blocked_single_rhs_path(N, nrhs):
            single_rhs_config = _get_single_rhs_blocked_launch_config(B.dtype, N)
            if effective_upper:
                cholesky_solve_single_rhs_blocked_upper_kernel[(batch_size,)](
                    L_kernel,
                    B_kernel,
                    X_kernel,
                    N,
                    batch_stride_L,
                    batch_stride_B,
                    stride_L,
                    stride_B,
                    dtype_flag=dtype_flag,
                    **single_rhs_config,
                )
            else:
                cholesky_solve_single_rhs_blocked_lower_kernel[(batch_size,)](
                    L_kernel,
                    B_kernel,
                    X_kernel,
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
            small_gather_config = _get_small_gather_launch_config(B.dtype, N)
            cholesky_solve_small_gather_kernel[(batch_size,)](
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
                **small_gather_config,
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
            grid = lambda meta: (batch_size, triton.cdiv(nrhs, meta["BLOCK_RHS"]))
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
    logger.debug("GEMS_ENFLAME CHOLESKY_SOLVE_OUT")
    _check_cholesky_solve_out(B, out)
    if _can_write_cholesky_solve_out_direct(B, L, out):
        return cholesky_solve(B, L, upper=upper, _out=out)
    result = cholesky_solve(B, L, upper=upper)
    return _copy_cholesky_solve_out(result, out)
