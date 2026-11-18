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
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Cap of the dense output tile the row kernel keeps in registers/flops per step:
# ``BLOCK_E`` nonzeros are accumulated together so that the loads of several
# ``mat2`` rows overlap instead of forming one dependent load-FMA chain.
_ROW_TILE_ELEMENTS = 8192
_MAX_BLOCK_E = 16
_MAX_BLOCK_N = 1024


@libentry()
@triton.jit(do_not_specialize=["alpha"])
def _sspaddmm_row_kernel(
    row_start_ptr,  # [R] start offset of each active row in mat1 nnz
    row_end_ptr,  # [R] end offset of each active row in mat1 nnz
    col_ptr,  # [nnz1] column indices of mat1 nonzeros (i.e. k index)
    val_ptr,  # [nnz1] values of mat1 nonzeros
    mat2_ptr,  # [K, N] dense matrix
    out_ptr,  # [R, N] dense result (alpha * mat1 @ mat2 over active rows)
    alpha,
    N,
    stride_mat2_k,
    stride_mat2_n,
    IS_FP64: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid_r = ext.program_id(0)
    pid_n = ext.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    row_start = tl.load(row_start_ptr + pid_r).to(tl.int64)
    row_end = tl.load(row_end_ptr + pid_r).to(tl.int64)

    acc_dtype = tl.float64 if IS_FP64 else tl.float32
    acc = tl.zeros((BLOCK_N,), dtype=acc_dtype)

    # Walk the row's nonzeros ``BLOCK_E`` at a time: the tile of mat2 rows is
    # loaded as one 2-D block and reduced along the nnz axis, so the loads of
    # consecutive nonzeros are issued together and the accumulator dependency
    # chain is ``nnz / BLOCK_E`` long instead of ``nnz`` long.
    offs_e = tl.arange(0, BLOCK_E)
    for e_base in range(row_start, row_end, BLOCK_E):
        offs = e_base + offs_e
        mask_e = offs < row_end
        k = tl.load(col_ptr + offs, mask=mask_e, other=0).to(tl.int64)
        v = tl.load(val_ptr + offs, mask=mask_e, other=0.0).to(acc_dtype)
        m2 = tl.load(
            mat2_ptr + k[:, None] * stride_mat2_k + offs_n[None, :] * stride_mat2_n,
            mask=mask_e[:, None] & mask_n[None, :],
            other=0.0,
        ).to(acc_dtype)
        acc += tl.sum(v[:, None] * m2, axis=0)

    acc = acc * alpha.to(acc_dtype)
    out_base = out_ptr + pid_r.to(tl.int64) * N + offs_n
    tl.store(out_base, acc, mask=mask_n)


def _sparse_matmul_terms(input, mat1, mat2, beta, alpha, acc_dtype):
    """Build the ``beta * input`` and ``alpha * mat1 @ mat2`` COO contributions.

    Returns the concatenated indices/values plus the sparse-side device, so the
    caller can assemble and coalesce the two contributions.
    """
    n = mat1.shape[0]
    N = mat2.shape[1]

    # GCU400 has no SparseCOO TensorImpl: sparse COO tensors built from GCU data
    # fall back to CPU storage (see _dimI.ensure_sparse_coo_gcu_fallback), so
    # `input`/`mat1` live on CPU while the dense `mat2` stays on GCU. Run the
    # Triton kernel on the dense side's device and copy the sparse metadata there
    # for the launch; assemble the sparse result back on `device` (the sparse
    # side), since GCU cannot host a sparse COO tensor at all.
    device = input.device
    compute_device = mat2.device if mat2.device.type != "cpu" else device

    mat2_c = mat2.contiguous()

    mat1_indices = mat1.indices()
    mat1_rows = mat1_indices[0]
    mat1_cols = mat1_indices[1].contiguous()
    mat1_vals = mat1.values().contiguous()

    # Group mat1 nonzeros by row. Coalesced COO is sorted row-major, so rows are
    # already in non-decreasing order.
    if mat1_rows.numel() == 0:
        active_rows = mat1_rows.new_empty((0,))
        counts = mat1_rows.new_empty((0,))
    else:
        active_rows, counts = flag_gems.unique_consecutive(
            mat1_rows, return_counts=True
        )
    R = active_rows.numel()

    if R > 0:
        row_end = counts.cumsum(0)
        row_start = row_end - counts
        row_start = row_start.to(torch.int64).contiguous()
        row_end = row_end.to(torch.int64).contiguous()

        dense = torch.empty((R, N), dtype=acc_dtype, device=compute_device)

        BLOCK_N = min(triton.next_power_of_2(N), _MAX_BLOCK_N) if N > 0 else 1
        BLOCK_E = max(1, min(_MAX_BLOCK_E, _ROW_TILE_ELEMENTS // BLOCK_N))
        grid = (R, triton.cdiv(N, BLOCK_N) if N > 0 else 1)

        # GCU aborts on int64 index tensors unless TORCH_GCU_ENABLE_INT64_AND_UINT64
        # is enabled, while in-kernel int64 arithmetic is fine. Ship the kernel's
        # index operands as int32 inside the safe domain (row offsets <= nnz1 and
        # column ids < K both fit int32 when nnz1 < 2**31 and K <= 2**31) and let
        # the kernel widen them back to int64 for address arithmetic; outside the
        # safe domain keep int64 operands (env-gated).
        idx_dtype = (
            torch.int32
            if mat1_cols.numel() < 2**31 and mat2.shape[0] <= 2**31
            else torch.int64
        )

        device_context = (
            torch_device_fn.device(compute_device)
            if compute_device.type != "cpu"
            else nullcontext()
        )
        with device_context:
            _sspaddmm_row_kernel[grid](
                row_start.to(idx_dtype).to(compute_device),
                row_end.to(idx_dtype).to(compute_device),
                mat1_cols.to(idx_dtype).to(compute_device),
                mat1_vals.to(compute_device),
                mat2_c,
                dense,
                alpha,
                N,
                mat2_c.stride(0),
                mat2_c.stride(1),
                IS_FP64=acc_dtype == torch.float64,
                BLOCK_N=BLOCK_N,
                BLOCK_E=BLOCK_E,
            )

        # Build indices for the mm contribution: every (active_row, col) pair.
        row_idx = active_rows.to(torch.int64).repeat_interleave(N)
        col_idx = torch.arange(N, device=device, dtype=torch.int64).repeat(R)
        mm_indices = flag_gems.stack([row_idx, col_idx], dim=0)
        mm_values = dense.reshape(-1).to(device)
    else:
        mm_indices = torch.empty((2, 0), dtype=torch.int64, device=device)
        mm_values = torch.empty((0,), dtype=acc_dtype, device=device)

    # beta * input contribution, scaled in the wider accumulation dtype.
    input_indices = input.indices().to(torch.int64)
    input_values = input.values().to(acc_dtype)
    beta_values = input_values * beta if beta != 1.0 else input_values

    out_indices = flag_gems.cat([input_indices, mm_indices], dim=1)
    out_values = flag_gems.cat([beta_values, mm_values], dim=0)

    return n, N, device, out_indices, out_values


def _sspaddmm_impl(input, mat1, mat2, beta=1.0, alpha=1.0, out=None):
    assert input.is_sparse, "sspaddmm: input must be a sparse COO tensor"
    assert mat1.is_sparse, "sspaddmm: mat1 must be a sparse COO tensor"
    assert not mat2.is_sparse, "sspaddmm: mat2 must be a dense tensor"

    input = input.coalesce()
    mat1 = mat1.coalesce()

    dtype = input.dtype

    # Keep the whole reduce/accumulate pipeline in a wider dtype for low
    # precision inputs, casting back to `dtype` only at the very end. This avoids
    # near-cancellation error when beta*input and alpha*mm overlap at an index.
    is_low_prec = dtype in (torch.float16, torch.bfloat16)
    acc_dtype = torch.float32 if is_low_prec else dtype

    n, N, device, out_indices, out_values = _sparse_matmul_terms(
        input, mat1, mat2, beta, alpha, acc_dtype
    )

    # The two contributions are merged by `coalesce`: it sums the duplicates in
    # the wide accumulation dtype, which is exactly why the widening above is
    # needed.
    result = torch.sparse_coo_tensor(
        out_indices, out_values, size=(n, N), dtype=acc_dtype, device=device
    ).coalesce()
    if acc_dtype != dtype:
        # `result` is already coalesced, so its indices are sorted and unique:
        # re-wrapping them in the narrow dtype needs no second coalesce.
        result = torch.sparse_coo_tensor(
            result.indices(),
            result.values().to(dtype),
            size=(n, N),
            dtype=dtype,
            device=device,
            is_coalesced=True,
        )

    if out is not None:
        out.copy_(result)
        return out
    return result


def sspaddmm(input, mat1, mat2, *, beta=1.0, alpha=1.0):
    logger.debug("GEMS_ENFLAME_GCU400 SSPADDMM")
    return _sspaddmm_impl(input, mat1, mat2, beta=beta, alpha=alpha)


def sspaddmm_out(input, mat1, mat2, *, beta=1.0, alpha=1.0, out=None):
    logger.debug("GEMS_ENFLAME_GCU400 SSPADDMM_OUT")
    if out is None:
        raise TypeError("sspaddmm(): out must be provided for the out variant")
    return _sspaddmm_impl(input, mat1, mat2, beta=beta, alpha=alpha, out=out)
