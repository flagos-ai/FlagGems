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
@triton.jit(do_not_specialize=["alpha", "beta", "num_active_rows"])
def _sspaddmm_row_kernel(
    seg_ptr,  # [4 * R] row_start / row_end / inp_start / inp_end, packed
    col_ptr,  # [nnz1] column indices of mat1 nonzeros (i.e. k index)
    val_ptr,  # [nnz1] values of mat1 nonzeros
    inp_col_ptr,  # [nnz_in] column indices of input nonzeros
    inp_val_ptr,  # [nnz_in] values of input nonzeros
    beta,
    mat2_ptr,  # [K, N] dense matrix
    out_ptr,  # [R, N] dense result (alpha * mat1 @ mat2 + beta * input rows)
    alpha,
    N,
    num_active_rows,
    stride_mat2_k,
    stride_mat2_n,
    IS_FP64: tl.constexpr,
    HAS_INPUT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid_r = ext.program_id(0)
    pid_n = ext.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    acc_dtype = tl.float64 if IS_FP64 else tl.float32

    row_start = tl.load(seg_ptr + pid_r).to(tl.int64)
    row_end = tl.load(seg_ptr + num_active_rows + pid_r).to(tl.int64)

    # Walk the row's nonzeros ``BLOCK_E`` at a time: the tile of mat2 rows is
    # loaded as one 2-D block and reduced along the nnz axis, so the loads of
    # consecutive nonzeros are issued together and the accumulator dependency
    # chain is ``nnz / BLOCK_E`` long instead of ``nnz`` long.
    acc_mm = tl.zeros((BLOCK_N,), dtype=acc_dtype)
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
        acc_mm += tl.sum(v[:, None] * m2, axis=0)

    # Fold the ``beta * input`` entries of this row into the same wide
    # accumulator as the mm part: an input value is routed to its output
    # column lane with a one-hot compare, so positions where the two
    # contributions overlap narrow to the output dtype exactly once (the
    # numerical contract of the wide-datum coalesce, without the sort).
    acc_in = tl.zeros((BLOCK_N,), dtype=acc_dtype)
    if HAS_INPUT:
        in_start = tl.load(seg_ptr + 2 * num_active_rows + pid_r).to(tl.int64)
        in_end = tl.load(seg_ptr + 3 * num_active_rows + pid_r).to(tl.int64)
        for e_base in range(in_start, in_end, BLOCK_E):
            offs = e_base + offs_e
            mask_e = offs < in_end
            ic = tl.load(inp_col_ptr + offs, mask=mask_e, other=-1)
            iv = tl.load(inp_val_ptr + offs, mask=mask_e, other=0.0).to(acc_dtype)
            hit = ic[:, None] == offs_n[None, :]
            acc_in += tl.sum(tl.where(hit, iv[:, None], 0.0), axis=0)

    result = acc_mm * alpha.to(acc_dtype) + acc_in * beta.to(acc_dtype)
    out_base = out_ptr + pid_r.to(tl.int64) * N + offs_n
    tl.store(out_base, result.to(out_ptr.dtype.element_ty), mask=mask_n)


def _sparse_matmul_terms(input, mat1, mat2, beta, alpha, out_dtype, acc_dtype):
    """Assemble the uncoalesced COO entries of ``beta * input + alpha * mat1 @ mat2``.

    Returns the concatenated indices/values plus the sparse-side device, so the
    caller can wrap them without any further merge pass.
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

    input_indices = input.indices().to(torch.int64)
    input_rows = input_indices[0]
    input_cols_c = input_indices[1].contiguous()
    input_vals_c = input.values().contiguous()
    nnz_in = input_rows.numel()

    # Group mat1 nonzeros by row. Coalesced COO is sorted row-major, so rows are
    # already in non-decreasing order.
    if mat1_rows.numel() == 0:
        active_rows = mat1_rows.new_empty((0,))
        counts = mat1_rows.new_empty((0,))
    else:
        active_rows, counts = torch.unique_consecutive(mat1_rows, return_counts=True)
    R = active_rows.numel()

    # Zero-valued contributions are dropped from the uncoalesced output: a
    # sparse tensor is free to omit stored zeros, and skipping them avoids
    # launching the kernel / shipping metadata for terms that cannot change
    # the dense value.
    has_mm = R > 0 and N > 0 and alpha != 0.0
    has_input_term = nnz_in > 0 and beta != 0.0

    # GCU aborts on int64 index tensors unless TORCH_GCU_ENABLE_INT64_AND_UINT64
    # is enabled, while in-kernel int64 arithmetic is fine. Ship the kernel's
    # index operands as int32 inside the safe domain (row offsets <= nnz and
    # column ids < K / N both fit int32 when the tensor extents stay below
    # 2**31) and let the kernel widen them back to int64 for address
    # arithmetic; outside the safe domain keep int64 operands (env-gated).
    idx_dtype = (
        torch.int32
        if (
            mat1_cols.numel() < 2**31
            and mat2.shape[0] <= 2**31
            and nnz_in < 2**31
            and N <= 2**31
        )
        else torch.int64
    )

    if has_mm:
        row_end = counts.cumsum(0)
        row_start = row_end - counts
        row_start = row_start.to(torch.int64).contiguous()
        row_end = row_end.to(torch.int64).contiguous()

        # Input entry range of each active row: `input` is coalesced, so its
        # row indices are sorted and the per-row segments come from binary
        # search instead of another grouping pass.
        if has_input_term:
            # GCU's topsatenSearchSorted rejects ``side="right"`` (it aborts with
            # "Side and right not match"), so derive the half-open segment end
            # from the successor key: for integer row indices the first position
            # with ``rows > r`` is the first position with ``rows >= r + 1``.
            successor_rows = active_rows + 1
            inp_start = torch.searchsorted(input_rows, active_rows)
            inp_end = torch.searchsorted(input_rows, successor_rows)
        else:
            inp_start = torch.zeros_like(row_start)
            inp_end = torch.zeros_like(row_start)

        # One packed [4R] segment array keeps the launch at a single index
        # upload instead of four small host-to-device copies.
        segs = torch.cat([row_start, row_end, inp_start, inp_end]).to(idx_dtype)

        dense = torch.empty((R, N), dtype=out_dtype, device=compute_device)

        BLOCK_N = min(triton.next_power_of_2(N), _MAX_BLOCK_N) if N > 0 else 1
        BLOCK_E = max(1, min(_MAX_BLOCK_E, _ROW_TILE_ELEMENTS // BLOCK_N))
        grid = (R, triton.cdiv(N, BLOCK_N) if N > 0 else 1)

        device_context = (
            torch_device_fn.device(compute_device)
            if compute_device.type != "cpu"
            else nullcontext()
        )
        with device_context:
            _sspaddmm_row_kernel[grid](
                segs.to(compute_device),
                mat1_cols.to(idx_dtype).to(compute_device),
                mat1_vals.to(compute_device),
                input_cols_c.to(idx_dtype).to(compute_device),
                input_vals_c.to(compute_device),
                beta,
                mat2_c,
                dense,
                alpha,
                N,
                R,
                mat2_c.stride(0),
                mat2_c.stride(1),
                IS_FP64=acc_dtype == torch.float64,
                HAS_INPUT=has_input_term,
                BLOCK_N=BLOCK_N,
                BLOCK_E=BLOCK_E,
            )

    # beta * input contribution, scaled in the wider accumulation dtype.
    # Entries on rows folded into the mm block are carried as explicit zeros:
    # their value already lives in the mm block, and a stored zero cannot
    # change the dense result of the uncoalesced tensor.
    if has_input_term:
        input_values = input_vals_c.to(acc_dtype)
        beta_values = input_values * beta if beta != 1.0 else input_values
        if has_mm:
            loc = torch.searchsorted(active_rows, input_rows)
            folded = (loc < R) & (
                active_rows[torch.clamp(loc, max=R - 1)] == input_rows
            )
            beta_values = beta_values.masked_fill(folded, 0)
        beta_values = beta_values.to(out_dtype)
    else:
        beta_values = torch.empty((0,), dtype=out_dtype, device=device)

    # Assemble the uncoalesced output in one preallocated index/value pair:
    # materializing the mm block via repeat_interleave + stack + cat costs
    # milliseconds of allocation churn on million-entry blocks, while
    # broadcast copies write the same words without intermediate tensors.
    off = nnz_in if has_input_term else 0
    mm_count = R * N if has_mm else 0
    out_indices = torch.empty((2, off + mm_count), dtype=torch.int64, device=device)
    out_values = torch.empty((off + mm_count,), dtype=out_dtype, device=device)
    if has_input_term:
        out_indices[:, :off] = input_indices
        out_values[:off] = beta_values
    if mm_count > 0:
        out_indices[0, off:].view(R, N).copy_(active_rows.view(-1, 1).expand(R, N))
        out_indices[1, off:].view(R, N).copy_(
            torch.arange(N, device=device, dtype=torch.int64).expand(R, N)
        )
        out_values[off:].copy_(dense.reshape(-1))

    return n, N, device, out_indices, out_values


def _sspaddmm_impl(input, mat1, mat2, beta=1.0, alpha=1.0, out=None):
    assert input.is_sparse, "sspaddmm: input must be a sparse COO tensor"
    assert mat1.is_sparse, "sspaddmm: mat1 must be a sparse COO tensor"
    assert not mat2.is_sparse, "sspaddmm: mat2 must be a dense tensor"

    input = input.coalesce()
    mat1 = mat1.coalesce()

    n = mat1.shape[0]
    N = mat2.shape[1]
    dtype = input.dtype

    # The reduce/accumulate pipeline runs in a wider dtype for low precision
    # inputs (the kernel narrows each output value exactly once), and the
    # non-overlapping input entries are scaled in the wide dtype before the
    # single cast back to `dtype`.
    is_low_prec = dtype in (torch.float16, torch.bfloat16)
    acc_dtype = torch.float32 if is_low_prec else dtype

    n, N, device, out_indices, out_values = _sparse_matmul_terms(
        input, mat1, mat2, beta, alpha, dtype, acc_dtype
    )

    # Like ATen's sspaddmm, the result is returned uncoalesced: the entries of
    # the two contributions sit next to each other, and `to_dense` sums any
    # overlap. This skips the O(nnz log nnz) sort/coalesce of the concatenated
    # index list, which dominates the whole op on this platform (the GCU kernel
    # itself is <2% of the latency); overlap positions are already summed in
    # the wide dtype inside the kernel, so the dense values are unchanged.
    result = torch.sparse_coo_tensor(
        out_indices, out_values, size=(n, N), dtype=dtype, device=device
    )

    if out is not None:
        out.copy_(result)
        return out
    return result


def sspaddmm(input, mat1, mat2, *, beta=1.0, alpha=1.0):
    logger.debug("GEMS_ENFLAME SSPADDMM")
    return _sspaddmm_impl(input, mat1, mat2, beta=beta, alpha=alpha)


def sspaddmm_out(input, mat1, mat2, *, beta=1.0, alpha=1.0, out=None):
    logger.debug("GEMS_ENFLAME SSPADDMM_OUT")
    if out is None:
        raise TypeError("sspaddmm(): out must be provided for the out variant")
    return _sspaddmm_impl(input, mat1, mat2, beta=beta, alpha=alpha, out=out)
