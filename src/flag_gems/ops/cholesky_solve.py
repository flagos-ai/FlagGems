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


def _get_blocked_tile_config(dtype):
    """Return dtype-specific tile sizes for blocked kernels.

    fp32 uses 32x32 panels with BLOCK_RHS=4. A narrow RHS tile launches more
    CTAs (grid = batch x ceil(nrhs/BLOCK_RHS)), which matters because the
    triangular solve is dominated by the serial per-row diagonal-block work
    rather than RHS width. On H20 (78 SMs) a single wide-RHS CTA badly
    underutilizes the device, so splitting the RHS dimension parallelizes the
    panel updates across SMs at the cost of cheaply re-deriving the diagonal
    factor per tile -- a uniform 10-20% win for both single and batched fp32
    solves. fp64 keeps BLOCK_RHS=8: its tl.dot lacks tensor-core acceleration,
    so narrow tiles only add serial-loop overhead (measured 2-3x slower at 4).
    fp64 uses 16x32 panels with BLOCK_RHS=8, which won every tested H20
    lower/upper case across N, nrhs, and batch size.
    """
    if dtype == torch.float64:
        blk_k, blk_m, blk_rhs = 16, 32, 8
    else:
        blk_k, blk_m, blk_rhs = 32, 32, 4
    return {"BLOCK_K": blk_k, "BLOCK_M": blk_m, "BLOCK_RHS": blk_rhs}


def _get_blocked_warp_config(dtype):
    """Return warp/stage config for blocked kernels based on dtype."""
    if dtype == torch.float64:
        return {"num_warps": 4, "num_stages": 2}
    return {"num_warps": 4, "num_stages": 3}


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


def _get_small_gather_launch_config(dtype, N):
    """Return pinned H20 winners for the small gather kernel.

    fp64 at N == 32 gains from a second warp sharing the wider fp64 row
    updates; everything else is latency-bound and best with one warp.
    """
    if dtype == torch.float64 and N > 16:
        return {"num_warps": 2, "num_stages": 1}
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
                "num_warps": 4,
                "num_stages": 1,
            }
        return {
            "BLOCK_K": 16,
            "BLOCK_M": 32,
            "BLOCK_RHS": 8,
            "num_warps": 4,
            "num_stages": 1,
        }
    return {
        "BLOCK_K": 32,
        "BLOCK_M": 64 if N >= 256 else 32,
        "BLOCK_RHS": 4,
        "num_warps": 4,
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
        if (N <= 32 and nrhs <= 8) or (N < 64 and nrhs == 1):
            block_n = triton.next_power_of_2(N)
            block_rhs = triton.next_power_of_2(nrhs)
            num_warps = (
                4
                if B.dtype == torch.complex128 and N > 32
                else 2 if B.dtype == torch.complex128 and N > 16 else 1
            )
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
                    num_warps=4,
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
                    num_warps=4,
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
    logger.debug("GEMS CHOLESKY_SOLVE")
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
        if _can_use_blocked_path(N, nrhs):
            tile = _get_blocked_tile_config(B.dtype)
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
    logger.debug("GEMS CHOLESKY_SOLVE_OUT")
    _check_cholesky_solve_out(B, out)
    if _can_write_cholesky_solve_out_direct(B, L, out):
        return cholesky_solve(B, L, upper=upper, _out=out)
    result = cholesky_solve(B, L, upper=upper)
    return _copy_cholesky_solve_out(result, out)
