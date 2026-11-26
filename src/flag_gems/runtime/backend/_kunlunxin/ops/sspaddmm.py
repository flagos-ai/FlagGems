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
"""``sspaddmm`` / ``sspaddmm.out``: ``beta * input + alpha * (mat1 @ mat2)``.

``input`` and ``mat1`` are sparse COO, ``mat2`` is dense. Both sparse operands
are reduced to a row-ordered entry set by device kernels (row pointers come
from a binary search over the row index, duplicates are merged by a prefix-sum
driven compaction), the product is accumulated row by row by a sparse row
kernel, and the result pattern is emitted with the same device-built offsets.

The result stores one entry for every column of a row that either operand
touches; the values are the exact dense values, so the dense image -- which is
what the operator's contract is defined on -- is identical.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..utils.sparse_index import (
    coalesce_triplets,
    dense_scatter,
    exclusive_scan,
    mark_rows,
    row_pointers_from_sorted,
    scale,
)

logger = logging.getLogger(__name__)

# Column tile of the row kernels. A 128-wide tile does not lower here.
_BLOCK_N = 64


@libentry()
@triton.jit(do_not_specialize=["N", "alpha", "stride_m2_k", "stride_m2_n"])
def _row_product_kernel(
    crow_ptr,
    col_ptr,
    val_ptr,
    mat2_ptr,
    out_ptr,
    N,
    stride_m2_k,
    stride_m2_n,
    alpha,
    IS_FP64: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """``out[r, :] = alpha * sum_e val[e] * mat2[col[e], :]`` per row."""
    r = tl.program_id(0)
    cn = tl.program_id(1)
    offs = cn * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    N64 = N.to(tl.int64)
    start = tl.load(crow_ptr + r).to(tl.int32)
    end = tl.load(crow_ptr + r + 1).to(tl.int32)

    acc_dtype = tl.float64 if IS_FP64 else tl.float32
    acc = tl.zeros((BLOCK_N,), dtype=acc_dtype)
    sk = stride_m2_k.to(tl.int64)
    sn = stride_m2_n.to(tl.int64)
    off64 = offs.to(tl.int64)
    for e in range(start, end):
        k = tl.load(col_ptr + e).to(tl.int64)
        v = tl.load(val_ptr + e).to(acc_dtype)
        m2 = tl.load(mat2_ptr + k * sk + off64 * sn, mask=mask, other=0.0).to(acc_dtype)
        acc += v * m2

    acc = acc * alpha.to(acc_dtype)
    tl.store(out_ptr + r.to(tl.int64) * N64 + off64, acc, mask=mask)


@triton.jit(do_not_specialize=["N", "total", "beta"])
def _emit_kernel(
    starts_ptr,
    prod_ptr,
    input_ptr,
    idx_ptr,
    val_ptr,
    N,
    total,
    beta,
    BLOCK_N: tl.constexpr,
):
    """Write the result pattern of one row: every column of a touched row.

    The written width is taken from the row's own slot range (``starts``
    difference) rather than from a conjunction with a per-row flag: the tile
    keeps a single comparison as its store mask.
    """
    r = tl.program_id(0)
    cn = tl.program_id(1)
    offs = cn * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    N64 = N.to(tl.int64)
    start = tl.load(starts_ptr + r).to(tl.int64)
    span = tl.load(starts_ptr + r + 1).to(tl.int64) - start

    base = r.to(tl.int64) * N64 + offs.to(tl.int64)
    v = (
        tl.load(prod_ptr + base, mask=mask, other=0.0)
        + tl.load(input_ptr + base, mask=mask, other=0.0) * beta
    )

    dst = start + offs.to(tl.int64)
    keep = offs.to(tl.int64) < span
    tl.store(val_ptr + dst, v.to(val_ptr.dtype.element_ty), mask=keep)
    tl.store(idx_ptr + dst, r.to(idx_ptr.dtype.element_ty), mask=keep)
    tl.store(
        idx_ptr + total.to(tl.int64) + dst,
        offs.to(idx_ptr.dtype.element_ty),
        mask=keep,
    )


def _empty_result(shape, dtype, device):
    return torch.sparse_coo_tensor(  # allow-torch-compute: construction only
        torch.empty((2, 0), dtype=torch.int64, device=device),
        torch.empty(0, dtype=dtype, device=device),
        size=shape,
        dtype=dtype,
        device=device,
        is_coalesced=True,
    )


def _sspaddmm_impl(input, mat1, mat2, beta=1.0, alpha=1.0, out=None):
    assert input.is_sparse, "sspaddmm: input must be a sparse COO tensor"
    assert mat1.is_sparse, "sspaddmm: mat1 must be a sparse COO tensor"
    assert not mat2.is_sparse, "sspaddmm: mat2 must be a dense tensor"

    n, N = input.shape
    M, K = mat1.shape
    dtype = input.dtype
    device = input.device
    is_low_prec = dtype in (torch.float16, torch.bfloat16)
    acc_dtype = torch.float32 if is_low_prec else dtype

    in_idx = input.indices()
    if input.is_coalesced():
        rows_in, cols_in, vals_in = in_idx[0], in_idx[1], input.values()
    else:
        rows_in, cols_in, vals_in = coalesce_triplets(
            in_idx[0], in_idx[1], input.values(), N
        )

    m1_idx = mat1.indices()
    if mat1.is_coalesced():
        rows1, cols1, vals1 = m1_idx[0], m1_idx[1], mat1.values()
    else:
        rows1, cols1, vals1 = coalesce_triplets(m1_idx[0], m1_idx[1], mat1.values(), K)

    nnz_in = int(rows_in.numel())
    nnz1 = int(rows1.numel())

    if M == 0 or N == 0 or (nnz_in == 0 and nnz1 == 0):
        result = _empty_result((n, N), dtype, device)
        if out is not None:
            out.copy_(result)
            return out
        return result

    # Row pointers of the sparse product operand (entries are row-ordered).
    crow = row_pointers_from_sorted(rows1.to(torch.int32), M)

    product = torch.zeros((M, N), dtype=acc_dtype, device=device)
    grid = (M, triton.cdiv(N, _BLOCK_N))
    with torch_device_fn.device(device):
        _row_product_kernel[grid](
            crow,
            cols1,
            vals1.to(acc_dtype),
            mat2,
            product,
            N,
            mat2.stride(0),
            mat2.stride(1),
            alpha,
            IS_FP64=acc_dtype == torch.float64,
            BLOCK_N=_BLOCK_N,
        )

    # Dense image of ``beta * input`` over the union pattern.
    input_dense = dense_scatter(rows_in, cols_in, vals_in.to(acc_dtype), M, N)

    flag = mark_rows(rows1, M + 1)
    flag = mark_rows(rows_in, M + 1, flag)
    counts = scale(flag, N)
    starts = exclusive_scan(counts)
    total = int(starts[M].item())

    indices = torch.empty((2, total), dtype=torch.int64, device=device)
    values = torch.empty(total, dtype=acc_dtype, device=device)
    with torch_device_fn.device(device):
        _emit_kernel[grid](
            starts,
            product,
            input_dense,
            indices,
            values,
            N,
            total,
            beta,
            BLOCK_N=_BLOCK_N,
        )

    result = torch.sparse_coo_tensor(  # allow-torch-compute: construction only
        indices,
        values.to(dtype),
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
