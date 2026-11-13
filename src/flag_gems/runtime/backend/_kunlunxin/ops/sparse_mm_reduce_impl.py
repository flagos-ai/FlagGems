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
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


_SPARSE_MM_REDUCE_IMPL_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}

_REDUCE_CODES = {
    "sum": 0,
    "mean": 1,
    "amax": 2,
    "amin": 3,
    "max": 2,
    "min": 3,
}

_REDUCE_SUM = tl.constexpr(0)
_REDUCE_MEAN = tl.constexpr(1)
_REDUCE_AMAX = tl.constexpr(2)
_REDUCE_AMIN = tl.constexpr(3)

# Per-program column tile. A 128-wide tile does not lower on this backend, so
# the tile is capped here and the column range is split over the grid instead.
_MAX_BLOCK_N = 64


@libentry()
@triton.jit
def _sparse_mm_reduce_impl_kernel(
    crow_ptr,
    col_ptr,
    val_ptr,
    mat2_ptr,
    out_ptr,
    M,
    N,
    stride_mat2_k,
    stride_mat2_n,
    stride_out_m,
    stride_out_n,
    IS_FP64: tl.constexpr,
    REDUCE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One program per (row, column block) of the dense output."""
    row = ext.program_id(0)
    rn = ext.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rn < N
    row64 = row.to(tl.int64)
    col_off = (rn * stride_mat2_n).to(tl.int64)

    acc_dtype = tl.float64 if IS_FP64 else tl.float32
    acc = tl.zeros((BLOCK_N,), dtype=acc_dtype)

    start = tl.load(crow_ptr + row64).to(tl.int64)
    end = tl.load(crow_ptr + row64 + 1).to(tl.int64)

    if REDUCE == _REDUCE_AMAX or REDUCE == _REDUCE_AMIN:
        for e in range(start, end):
            c = tl.load(col_ptr + e).to(tl.int64)
            v = tl.load(val_ptr + e).to(acc_dtype)
            b = tl.load(
                mat2_ptr + c * stride_mat2_k + col_off, mask=rmask, other=0.0
            ).to(acc_dtype)
            p = b * v
            if REDUCE == _REDUCE_AMAX:
                acc = tl.where(e == start, p, tl.maximum(acc, p))
            else:
                acc = tl.where(e == start, p, tl.minimum(acc, p))
        acc = tl.where(start < end, acc, 0.0)
    else:
        for e in range(start, end):
            c = tl.load(col_ptr + e).to(tl.int64)
            v = tl.load(val_ptr + e).to(acc_dtype)
            b = tl.load(
                mat2_ptr + c * stride_mat2_k + col_off, mask=rmask, other=0.0
            ).to(acc_dtype)
            acc += b * v
        if REDUCE == _REDUCE_MEAN:
            nnz = (end - start).to(acc_dtype)
            acc = tl.where(nnz > 0, acc / nnz, 0.0)

    cos = (row64 * stride_out_m + (rn * stride_out_n).to(tl.int64)).to(tl.int64)
    tl.store(out_ptr + cos, acc.to(out_ptr.dtype.element_ty), mask=rmask)


@triton.jit
def _csr_scatter_kernel(
    rows_ptr,
    col_ptr,
    val_ptr,
    dense_ptr,
    K,
    nnz,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < nnz
    r = tl.load(rows_ptr + offs, mask=mask, other=0).to(tl.int64)
    c = tl.load(col_ptr + offs, mask=mask, other=0).to(tl.int64)
    v = tl.load(val_ptr + offs, mask=mask, other=0)
    tl.store(dense_ptr + r * K + c, v, mask=mask)


def _dense_row_ids(crow, col):
    """Materialise the row index of every stored CSR entry."""
    counts = crow[1:] - crow[:-1]
    rows = torch.arange(crow.numel() - 1, device=col.device, dtype=torch.int64)
    return torch.repeat_interleave(rows, counts)


def _csr_components(sparse):
    """Return ``(crow, col, values)`` for any supported sparse layout.

    The dense/sparse conversions of the vendor backend only cover a subset of
    the layouts, so the COO and CSC forms are re-expressed here with the
    row-pointer / column-index construction the kernel needs.
    """
    layout = sparse.layout
    if layout == torch.sparse_csr:
        return sparse.crow_indices(), sparse.col_indices(), sparse.values()

    M = sparse.shape[0]
    device = sparse.device
    crow = torch.zeros(M + 1, dtype=torch.int64, device=device)

    if layout == torch.sparse_coo:
        coalesced = sparse if sparse.is_coalesced() else sparse.coalesce()
        indices = coalesced.indices()
        rows = indices[0].to(torch.int64)
        counts = torch.bincount(rows, minlength=M)
        crow[1:] = counts.cumsum(0)
        return crow, indices[1], coalesced.values()

    if layout == torch.sparse_csc:
        K = sparse.shape[1]
        ccol = sparse.ccol_indices().to(torch.int64)
        rows = sparse.row_indices().to(torch.int64)
        values = sparse.values()
        counts = torch.bincount(rows, minlength=M)
        crow[1:] = counts.cumsum(0)
        per_column = ccol[1:] - ccol[:-1]
        col_of_entry = torch.arange(K, device=device, dtype=torch.int64)
        col_of_entry = col_of_entry.repeat_interleave(per_column)
        order = torch.argsort(rows, stable=True)
        return crow, col_of_entry[order], values[order]

    raise RuntimeError(
        f"_sparse_mm_reduce_impl: Expected self to have a sparse layout, but "
        f"got {sparse.layout}"
    )


def _sparse_mm_reduce_impl(self, other, reduce="sum"):
    logger.debug("GEMS_KUNLUNXIN _SPARSE_MM_REDUCE_IMPL")

    if self.layout == torch.strided:
        raise NotImplementedError(
            f"Could not run '_sparse_mm_reduce_impl' with arguments from the "
            f"'{self.layout}' backend. This could be because the operator "
            f"doesn't exist for this backend, or was omitted during the "
            f"selective/custom build process"
        )
    if other.layout != torch.strided:
        raise RuntimeError(
            f"_sparse_mm_reduce_impl: Expected other to have strided layout, "
            f"but got {other.layout}"
        )
    if self.dim() != 2:
        raise RuntimeError(
            f"sparse_mm_reduce: expected self to be a 2-D tensor, got "
            f"{self.dim()}-D tensor."
        )
    if self.dense_dim() != 0:
        raise RuntimeError("sparse_mm_reduce: expected non-hybrid self tensor.")
    if other.dim() != 2 or other.shape[0] != self.shape[1]:
        raise RuntimeError(
            f"Expected a tensor of dimension 2 and tensor.size[0] == "
            f"{self.shape[1]} but got: dimension {other.dim()} and "
            f"tensor.size[0] = {other.shape[0]}"
        )

    if reduce not in _REDUCE_CODES:
        raise RuntimeError(
            f"reduce argument must be either sum, prod, mean, amax or amin, "
            f"got {reduce}"
        )
    if reduce == "prod":
        raise RuntimeError(
            "sparse_mm_reduce: reduce type of prod has not been enabled."
        )

    if self.dtype not in _SPARSE_MM_REDUCE_IMPL_DTYPES:
        raise RuntimeError(f'"spmm_reduce_kernel" not implemented for {self.dtype!r}')
    if self.dtype != other.dtype:
        raise RuntimeError(
            f"Expected tensor for argument #1 'other' to have scalar type "
            f"{self.dtype} but got {other.dtype}"
        )

    M, K = self.shape
    N = other.shape[1]

    out = torch.empty((M, N), dtype=other.dtype, device=other.device)
    arg_out = torch.empty((0,), dtype=torch.int64, device=other.device)
    if M == 0 or N == 0:
        return out, arg_out

    crow, col, val = _csr_components(self)

    if reduce in ("sum", "mean"):
        # ``sum``/``mean`` over the stored entries is exactly the dense product
        # with the unstored positions zeroed, so the reduction is folded into a
        # dense matmul; ``mean`` additionally divides by the stored count.
        from flag_gems import mm as _gems_mm

        dense = torch.zeros((M, K), dtype=other.dtype, device=other.device)
        nnz = val.numel()
        if nnz:
            rows = _dense_row_ids(crow, col)
            block = 512
            _csr_scatter_kernel[(triton.cdiv(nnz, block),)](
                rows, col, val, dense, K, nnz, BLOCK=block
            )
        out = _gems_mm(dense, other)
        if reduce == "mean":
            counts = (crow[1:] - crow[:-1]).to(torch.int64)
            denom = torch.where(counts > 0, counts, torch.ones_like(counts)).to(
                out.dtype
            )
            out = torch.where((counts > 0)[:, None], out / denom[:, None], 0.0)
        return out, arg_out

    block_n = min(triton.next_power_of_2(N), _MAX_BLOCK_N)
    grid = (M, triton.cdiv(N, block_n))
    with torch_device_fn.device(other.device):
        _sparse_mm_reduce_impl_kernel[grid](
            crow,
            col,
            val,
            other,
            out,
            M,
            N,
            other.stride(0),
            other.stride(1),
            out.stride(0),
            out.stride(1),
            IS_FP64=other.dtype == torch.float64,
            REDUCE=_REDUCE_CODES[reduce],
            BLOCK_N=block_n,
            num_warps=4,
        )
    return out, arg_out
