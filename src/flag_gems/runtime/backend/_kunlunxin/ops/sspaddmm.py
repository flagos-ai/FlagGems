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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_SCATTER_BLOCK = 512


@triton.jit
def _csr_scatter_kernel(
    rows_ptr,
    col_ptr,
    val_ptr,
    dense_ptr,
    N,
    nnz,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < nnz
    r = tl.load(rows_ptr + offs, mask=mask, other=0).to(tl.int64)
    c = tl.load(col_ptr + offs, mask=mask, other=0).to(tl.int64)
    v = tl.load(val_ptr + offs, mask=mask, other=0)
    tl.store(dense_ptr + r * N + c, v, mask=mask)


def _densify(rows, cols, vals, shape, dtype):
    """Materialise ``vals`` at ``(rows, cols)`` of a zero dense matrix."""
    dense = torch.zeros(shape, dtype=dtype, device=vals.device)
    nnz = vals.numel()
    if nnz:
        grid = (triton.cdiv(nnz, _SCATTER_BLOCK),)
        with torch_device_fn.device(vals.device):
            _csr_scatter_kernel[grid](
                rows, cols, vals, dense, shape[1], nnz, BLOCK=_SCATTER_BLOCK
            )
    return dense


def _sparse_to_dense(sparse, dtype):
    """Dense image of a COO tensor without the backend sparse->dense path."""
    if sparse._nnz() == 0:
        return torch.zeros(sparse.shape, dtype=dtype, device=sparse.device)
    indices = sparse.indices()
    return _densify(
        indices[0], indices[1], sparse.values().to(dtype), sparse.shape, dtype
    )


def _sspaddmm_impl(input, mat1, mat2, beta=1.0, alpha=1.0, out=None):
    assert input.is_sparse, "sspaddmm: input must be a sparse COO tensor"
    assert mat1.is_sparse, "sspaddmm: mat1 must be a sparse COO tensor"
    assert not mat2.is_sparse, "sspaddmm: mat2 must be a dense tensor"

    if not input.is_coalesced():
        input = input.coalesce()
    if not mat1.is_coalesced():
        mat1 = mat1.coalesce()

    n, N = input.shape
    dtype = input.dtype
    device = input.device

    is_low_prec = dtype in (torch.float16, torch.bfloat16)
    acc_dtype = torch.float32 if is_low_prec else dtype

    # Dense product of the sparse operand with ``mat2``; the stored-entry set
    # is materialised first, so no vendor sparse matmul is needed.
    mat1_indices = mat1.indices()
    mat1_dense = _densify(
        mat1_indices[0],
        mat1_indices[1],
        mat1.values().to(acc_dtype),
        mat1.shape,
        acc_dtype,
    )
    from flag_gems import mm as _gems_mm

    product = _gems_mm(mat1_dense, mat2.to(acc_dtype))

    input_dense = _sparse_to_dense(input, acc_dtype)
    if beta != 0.0:
        result_dense = product * alpha + input_dense * beta
    else:
        result_dense = product * alpha

    # Output pattern: the input's stored entries on the rows that ``mat1``
    # leaves untouched, plus the full row for every active row of ``mat1``.
    active = torch.zeros(n, dtype=torch.bool, device=device)
    if mat1_indices.numel():
        active[mat1_indices[0]] = True
    active_rows = active.nonzero(as_tuple=False).reshape(-1)

    in_indices = input.indices()
    in_rows = in_indices[0].to(torch.int64)
    in_cols = in_indices[1].to(torch.int64)
    nnz_in = in_rows.numel()

    counts_in = torch.bincount(in_rows, minlength=n)
    row_start = counts_in.cumsum(0) - counts_in
    n_out = torch.where(active, torch.full_like(counts_in, N), counts_in)
    offsets = n_out.cumsum(0) - n_out
    total = int(n_out.sum().item())

    out_rows = torch.zeros(total, dtype=torch.int64, device=device)
    out_cols = torch.zeros(total, dtype=torch.int64, device=device)
    out_vals = torch.zeros(total, dtype=acc_dtype, device=device)

    if nnz_in:
        keep = ~active[in_rows]
        kept_e = torch.arange(nnz_in, device=device)[keep]
        kept_rows = in_rows[keep]
        kept_cols = in_cols[keep]
        pos = offsets[kept_rows] + (kept_e - row_start[kept_rows])
        out_rows[pos] = kept_rows
        out_cols[pos] = kept_cols
        out_vals[pos] = result_dense[kept_rows, kept_cols]

    R = active_rows.numel()
    if R:
        r_act = active_rows.to(torch.int64).repeat_interleave(N)
        c_act = torch.arange(N, device=device, dtype=torch.int64).repeat(R)
        pos = offsets[r_act] + c_act
        out_rows[pos] = r_act
        out_cols[pos] = c_act
        out_vals[pos] = result_dense[r_act, c_act]

    indices = torch.stack([out_rows, out_cols], dim=0)
    result = torch.sparse_coo_tensor(
        indices,
        out_vals.to(dtype),
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
    logger.debug("GEMS_KUNLUNXIN SSPADDMM")
    return _sspaddmm_impl(input, mat1, mat2, beta=beta, alpha=alpha)


def sspaddmm_out(input, mat1, mat2, *, beta=1.0, alpha=1.0, out=None):
    logger.debug("GEMS_KUNLUNXIN SSPADDMM_OUT")
    if out is None:
        raise TypeError("sspaddmm(): out must be provided for the out variant")
    return _sspaddmm_impl(input, mat1, mat2, beta=beta, alpha=alpha, out=out)
