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
import triton.experimental.tle as tle
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


def check_cholesky_solve_out(B: torch.Tensor, out: torch.Tensor) -> None:
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


def copy_cholesky_solve_out(result: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
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


# ---------------------------------------------------------------------------
# Scalar kernel for general cases (very-large-N fallback)
# ---------------------------------------------------------------------------


@libentry()
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
    """Cholesky solve kernel for Ascend NPU.

    Solves L L^T X = B or U^T U X = B for X, given the lower- or
    upper-triangular Cholesky factor and the right-hand side B.

    Each program computes one RHS tile for one matrix in the batch.
    """
    batch_pid = ext.program_id(0)
    rhs_tile_pid = ext.program_id(1)

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    cols = rhs_tile_pid * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    cols_mask = cols < nrhs

    # Phase 1: Forward substitution
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
        # Fast reciprocal with Newton refinement (1 extra iter for fp32 on Ascend)
        inv_diag = 1.0 / diag
        inv_diag = inv_diag * (2.0 - diag * inv_diag)
        if dtype_flag == 1:
            inv_diag = inv_diag * (2.0 - diag * inv_diag)
        tl.store(
            X_ptr + B_base + i * stride_B + cols, sum_val * inv_diag, mask=cols_mask
        )

    # Phase 2: Backward substitution
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


# ---------------------------------------------------------------------------
# UB-resident blocked kernel (any nrhs, N <= 1024)
# ---------------------------------------------------------------------------


@libentry()
@triton.jit
def cholesky_solve_ub_blocked_kernel(
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
    BLOCK_N: tl.constexpr,
    upper: tl.constexpr,
):
    """UB-resident blocked Cholesky solve for Ascend NPU.

    Solves L L^T X = B or U^T U X = B with blocked TRSM.

    The whole [N, BLOCK_RHS] right-hand-side tile stays in UB for the entire
    solve: B is loaded once and X stored once, so there is no global-memory
    read-back inside the solve.  A store->load round-trip through global
    memory is both a latency hazard and, with many concurrent programs, a
    correctness hazard on Ascend (stale reads were observed at batch >= 32
    with the previous global round-trip formulation).

    Each diagonal block is staged once as a [BLOCK_K, BLOCK_K] tile,
    pre-scaled by the reciprocal diagonal and masked to its strict triangle,
    then transposed.  The serial substitution then needs, per row, one cheap
    DSA row slice of the transposed tile plus one rank-1 FMA -- no global
    loads, no scalar divides, no insert_slice in the dependency chain.  Row
    slices ([1, BLOCK_K]) measure ~20x cheaper than column slices
    ([BLOCK_K, 1]) on Ascend, which is why the tile is transposed once per
    block instead of extracting columns per row.

    k/m loops use tl.static_range so extract/insert offsets are compile-time
    constants (dynamic-offset insert_slice lowers to an unsupported
    cbuf->cbuf copy on this backend).
    """
    batch_pid = ext.program_id(0)
    rhs_tile_pid = ext.program_id(1)
    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    rhs_cols = rhs_tile_pid * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    rhs_mask = rhs_cols < nrhs
    k_offsets = tl.arange(0, BLOCK_K)
    m_offsets = tl.arange(0, BLOCK_M)
    n_offsets = tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N
    strict_lower = k_offsets[:, None] > k_offsets[None, :]
    strict_upper = k_offsets[:, None] < k_offsets[None, :]
    # transposed solve (U^T forward / L^T backward) uses swapped-stride loads
    fwd_T = upper
    bwd_T = not upper

    y = tl.load(
        B_ptr + B_base + n_offsets[:, None] * stride_B + rhs_cols[None, :],
        mask=n_mask[:, None] & rhs_mask[None, :],
        other=0.0,
    )

    # ---------------- Forward phase (ascending k) ----------------
    for k in tl.static_range(0, N, BLOCK_K):
        rows_k = k + k_offsets
        rows_k_mask = rows_k < N
        diag_block = tl.load(
            L_ptr + L_base + rows_k * stride_L + rows_k,
            mask=rows_k_mask,
            other=1.0,
        )
        inv_diag = 1.0 / diag_block
        inv_diag = inv_diag * (2.0 - diag_block * inv_diag)

        if fwd_T:
            Ld = tl.load(
                L_ptr + L_base + rows_k[None, :] * stride_L + rows_k[:, None],
                mask=rows_k_mask[None, :] & rows_k_mask[:, None],
                other=0.0,
            )
        else:
            Ld = tl.load(
                L_ptr + L_base + rows_k[:, None] * stride_L + rows_k[None, :],
                mask=rows_k_mask[:, None] & rows_k_mask[None, :],
                other=0.0,
            )
        scaled = tl.where(strict_lower, Ld * inv_diag[:, None], 0.0)
        scaledT = tl.trans(scaled)

        y_k = tle.dsa.extract_slice(y, (k, 0), (BLOCK_K, BLOCK_RHS), (1, 1))
        y_k = y_k * inv_diag[:, None]
        for i in range(BLOCK_K):
            col_i = tl.reshape(
                tle.dsa.extract_slice(scaledT, (i, 0), (1, BLOCK_K), (1, 1)),
                (BLOCK_K,),
            )
            w_i = tle.dsa.extract_slice(y_k, (i, 0), (1, BLOCK_RHS), (1, 1))
            y_k = y_k - col_i[:, None] * w_i
        y = tle.dsa.insert_slice(y, y_k, (k, 0), (BLOCK_K, BLOCK_RHS), (1, 1))

        for m in tl.static_range(k + BLOCK_K, N, BLOCK_M):
            rows_m = m + m_offsets
            rows_m_mask = rows_m < N
            if fwd_T:
                L_tile = tl.load(
                    L_ptr + L_base + rows_k[None, :] * stride_L + rows_m[:, None],
                    mask=rows_k_mask[None, :] & rows_m_mask[:, None],
                    other=0.0,
                )
            else:
                L_tile = tl.load(
                    L_ptr + L_base + rows_m[:, None] * stride_L + rows_k[None, :],
                    mask=rows_m_mask[:, None] & rows_k_mask[None, :],
                    other=0.0,
                )
            y_m = tle.dsa.extract_slice(y, (m, 0), (BLOCK_M, BLOCK_RHS), (1, 1))
            y_m = y_m - tl.sum(L_tile[:, :, None] * y_k[None, :, :], axis=1)
            y = tle.dsa.insert_slice(y, y_m, (m, 0), (BLOCK_M, BLOCK_RHS), (1, 1))

    # ---------------- Backward phase (descending k) ----------------
    # The top block may be partial; start from the last block boundary so
    # every row is covered even when N % BLOCK_K != 0.
    for k in tl.static_range(
        ((N + BLOCK_K - 1) // BLOCK_K - 1) * BLOCK_K, -1, -BLOCK_K
    ):
        rows_k = k + k_offsets
        rows_k_mask = rows_k < N
        diag_block = tl.load(
            L_ptr + L_base + rows_k * stride_L + rows_k,
            mask=rows_k_mask,
            other=1.0,
        )
        inv_diag = 1.0 / diag_block
        inv_diag = inv_diag * (2.0 - diag_block * inv_diag)

        if bwd_T:
            Ld = tl.load(
                L_ptr + L_base + rows_k[None, :] * stride_L + rows_k[:, None],
                mask=rows_k_mask[None, :] & rows_k_mask[:, None],
                other=0.0,
            )
        else:
            Ld = tl.load(
                L_ptr + L_base + rows_k[:, None] * stride_L + rows_k[None, :],
                mask=rows_k_mask[:, None] & rows_k_mask[None, :],
                other=0.0,
            )
        scaled = tl.where(strict_upper, Ld * inv_diag[:, None], 0.0)
        scaledT = tl.trans(scaled)

        x_k = tle.dsa.extract_slice(y, (k, 0), (BLOCK_K, BLOCK_RHS), (1, 1))
        x_k = x_k * inv_diag[:, None]
        for ii in range(BLOCK_K - 1, -1, -1):
            col_i = tl.reshape(
                tle.dsa.extract_slice(scaledT, (ii, 0), (1, BLOCK_K), (1, 1)),
                (BLOCK_K,),
            )
            w_i = tle.dsa.extract_slice(x_k, (ii, 0), (1, BLOCK_RHS), (1, 1))
            x_k = x_k - col_i[:, None] * w_i
        y = tle.dsa.insert_slice(y, x_k, (k, 0), (BLOCK_K, BLOCK_RHS), (1, 1))

        for m in tl.static_range(0, k, BLOCK_M):
            rows_m = m + m_offsets
            rows_m_mask = rows_m < k
            if bwd_T:
                L_tile = tl.load(
                    L_ptr + L_base + rows_k[None, :] * stride_L + rows_m[:, None],
                    mask=rows_k_mask[None, :] & rows_m_mask[:, None],
                    other=0.0,
                )
            else:
                L_tile = tl.load(
                    L_ptr + L_base + rows_m[:, None] * stride_L + rows_k[None, :],
                    mask=rows_m_mask[:, None] & rows_k_mask[None, :],
                    other=0.0,
                )
            y_m = tle.dsa.extract_slice(y, (m, 0), (BLOCK_M, BLOCK_RHS), (1, 1))
            y_m = y_m - tl.sum(L_tile[:, :, None] * x_k[None, :, :], axis=1)
            y = tle.dsa.insert_slice(y, y_m, (m, 0), (BLOCK_M, BLOCK_RHS), (1, 1))

    tl.store(
        X_ptr + B_base + n_offsets[:, None] * stride_B + rhs_cols[None, :],
        y,
        mask=n_mask[:, None] & rhs_mask[None, :],
    )


# ---------------------------------------------------------------------------
# Single-RHS scalar kernel (very-large-N fallback)
# ---------------------------------------------------------------------------


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
    """Scalar Cholesky solve kernel for nrhs == 1 on Ascend NPU."""
    batch_pid = ext.program_id(0)

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B

    # Phase 1: Forward substitution
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

    # Phase 2: Backward substitution
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
# Full-vector single-RHS kernel (nrhs == 1, N <= 16, small batch)
# ---------------------------------------------------------------------------


@libentry()
@triton.jit
def cholesky_solve_single_rhs_full_vector_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    N: tl.constexpr,
    batch_stride_L,
    batch_stride_B,
    stride_L,
    stride_B,
    BLOCK_N: tl.constexpr,
    dtype_flag: tl.constexpr,
    upper: tl.constexpr,
):
    """Register/UB-resident pre-scaled solve for a single RHS.

    Keeping the complete solution vector live removes the per-pivot global
    loads/stores from the scalar kernel.  A 1-D tensor plus extract_element
    also avoids the padded [N, 1] representation used by the generic gather
    formulation on Ascend.
    """
    batch_pid = ext.program_id(0)
    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    rows = tl.arange(0, BLOCK_N)
    rows_mask = rows < N

    b = tl.load(
        B_ptr + B_base + rows * stride_B,
        mask=rows_mask,
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

    # Forward solve.  Pre-scaling by each destination row's reciprocal
    # diagonal turns every dependent pivot into one vector multiply-add.
    w = b * inv_diag
    for i in range(N):
        if upper:
            factor_col = tl.load(
                L_ptr + L_base + i * stride_L + rows,
                mask=(rows > i) & rows_mask,
                other=0.0,
            )
        else:
            factor_col = tl.load(
                L_ptr + L_base + rows * stride_L + i,
                mask=(rows > i) & rows_mask,
                other=0.0,
            )
        w_i = tle.dsa.extract_element(w, (i,))
        w = w - (factor_col * inv_diag) * w_i

    # Backward solve.  The second pre-scale accounts for the diagonal of the
    # transposed triangular system before pivots are propagated in reverse.
    w = w * inv_diag
    for i in range(N - 1, -1, -1):
        if upper:
            factor_row = tl.load(
                L_ptr + L_base + rows * stride_L + i,
                mask=(rows < i) & rows_mask,
                other=0.0,
            )
        else:
            factor_row = tl.load(
                L_ptr + L_base + i * stride_L + rows,
                mask=(rows < i) & rows_mask,
                other=0.0,
            )
        w_i = tle.dsa.extract_element(w, (i,))
        w = w - (factor_row * inv_diag) * w_i

    tl.store(X_ptr + B_base + rows * stride_B, w, mask=rows_mask)


# ---------------------------------------------------------------------------
# Launch configuration helpers
# ---------------------------------------------------------------------------


def _get_ub_blocked_config(N):
    """Return tile sizes for the UB-resident blocked kernel on Ascend 910B.

    BLOCK_RHS is fixed at 16: narrower tiles miscompile on this backend
    (verified numerically), wider tiles reduce the program count.
    """
    return {
        "BLOCK_K": 32,
        "BLOCK_M": 32,
        "BLOCK_RHS": 16,
        "num_warps": 4,
        "num_stages": 1,
    }


# Largest N whose [N, 16] fp32 RHS tile (plus staged factor tiles) fits
# comfortably in the 192 KB UB of the 910B.
_MAX_UB_BLOCKED_N = 1024


# ---------------------------------------------------------------------------
# Main dispatch function for Ascend
# ---------------------------------------------------------------------------


def cholesky_solve(B, L, upper=False):
    """Solves a system of linear equations with a symmetric positive-definite
    matrix using the Cholesky factorization on Ascend NPU.

    Dispatch rules (based on what works on Ascend 910B):
      - nrhs == 1, N <= 16, batch < 64 → register-resident single-RHS kernel
      - N <= 1024 (any nrhs)           → UB-resident blocked kernel
      - otherwise                      → scalar fallback kernels

    Computes X such that A @ X = B, where A = L @ L^T (or A = U^T @ U if
    upper=True) and L (or U) is the Cholesky factor of A.

    Args:
        B: right-hand side tensor of shape (*, N, nrhs)
        L: Cholesky factor of shape (*, N, N), lower-triangular unless upper=True
        upper: if True, the Cholesky factor is upper-triangular

    Returns:
        X: solution tensor of shape (*, N, nrhs)
    """
    logger.debug("GEMS_ASCEND CHOLESKY_SOLVE")
    if L.dtype not in (torch.float32,):
        raise ValueError("cholesky_solve on Ascend only supports float32")
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
    # torch.broadcast_shapes + expand calls. Each costs several microseconds
    # of host time, which matters for the small systems.
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

    batch_size = 1
    for dim in batch_shape:
        batch_size *= dim

    # Zero-copy layout normalization.  torch.linalg.cholesky commonly
    # returns a transpose-contiguous factor; reinterpret that storage
    # through an mT view and flip the triangular orientation instead of
    # materializing an F-to-C layout conversion.
    if L.is_contiguous():
        effective_upper = upper
    elif L.mT.is_contiguous():
        L = L.mT
        effective_upper = not upper
    else:
        L = L.contiguous()
        effective_upper = upper

    # Broadcasted batch dimensions may introduce zero strides and still need
    # materialization before batch flattening. The common non-broadcast path
    # remains a zero-copy view after the layout normalization above.
    if not L.is_contiguous():
        L = L.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    X = torch.empty_like(B)

    L_kernel = L.reshape(-1, N, N)
    B_kernel = B.reshape(-1, N, nrhs)
    X_kernel = X.reshape(-1, N, nrhs)

    stride_L = L_kernel.stride(1)
    stride_B = B_kernel.stride(1)
    batch_stride_L = L_kernel.stride(0)
    batch_stride_B = B_kernel.stride(0)

    dtype_flag = 0 if B.dtype == torch.float32 else 1
    device = B.device

    with torch_device_fn.device(device):
        # Path 1: register-resident single RHS (small N, small batch)
        if nrhs == 1 and N <= 16 and batch_size < 64:
            block_n = triton.next_power_of_2(N)
            cholesky_solve_single_rhs_full_vector_kernel[(batch_size,)](
                L_kernel,
                B_kernel,
                X_kernel,
                N,
                batch_stride_L,
                batch_stride_B,
                stride_L,
                stride_B,
                BLOCK_N=block_n,
                dtype_flag=dtype_flag,
                upper=effective_upper,
                num_warps=2,
                num_stages=1,
            )
        # Path 2: UB-resident blocked kernel (any nrhs, N <= 1024)
        elif N <= _MAX_UB_BLOCKED_N:
            cfg = _get_ub_blocked_config(N)
            block_n = max(triton.next_power_of_2(N), cfg["BLOCK_K"])
            grid = (batch_size, triton.cdiv(nrhs, cfg["BLOCK_RHS"]))
            cholesky_solve_ub_blocked_kernel[grid](
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
                upper=effective_upper,
                **cfg,
            )
        # Path 3: scalar single-RHS fallback (very large N)
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
        # Path 4: general scalar fallback (very large N)
        else:
            blk_rhs = min(nrhs, 16)
            grid = (batch_size, triton.cdiv(nrhs, blk_rhs))
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
                BLOCK_RHS=blk_rhs,
                dtype_flag=dtype_flag,
                upper=effective_upper,
            )

    return X


def cholesky_solve_out(B, L, upper=False, *, out):
    """Out variant with the same temporary-and-copy semantics as PyTorch."""
    logger.debug("GEMS_ASCEND CHOLESKY_SOLVE_OUT")
    check_cholesky_solve_out(B, out)
    result = cholesky_solve(B, L, upper=upper)
    return copy_cholesky_solve_out(result, out)
