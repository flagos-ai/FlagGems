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
"""``aten::_sparse_mm.reduce``: SpMM plus a per-row reduction.

All sparse layouts are reduced to a CSR view by device kernels: the row
pointer array of a row-ordered entry set comes from a binary search over the
row index, a CSC column pointer array is expanded to a per-entry column index
the same way, and a CSC entry set is brought into row order by the in-tree
radix sort. No host-side index table is built and no duplicate-removal happens
on the host.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._sparse_mm_reduce_impl import _sparse_mm_reduce_impl_kernel
from flag_gems.runtime import torch_device_fn

from ..utils.sparse_index import (
    coalesce_triplets,
    csc_entry_columns,
    csr_to_dense,
    gather,
    row_pointers_from_sorted,
)
from .mm import mm as _gems_mm
from .sort import sort as _sort

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

# Per-program column tile. A 128-wide tile does not lower on this backend, so
# the column range is split over the grid instead.
_MAX_BLOCK_N = 64
_BLOCK_N = 64


@triton.jit(do_not_specialize=["N"])
def _mean_scale_kernel(out_ptr, crow_ptr, N, BLOCK_N: tl.constexpr):
    """Divide row ``r`` by its stored count; empty rows stay zero."""
    r = tl.program_id(0)
    cn = tl.program_id(1)
    offs = cn * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    start = tl.load(crow_ptr + r).to(tl.int32)
    end = tl.load(crow_ptr + r + 1).to(tl.int32)
    count = (end - start).to(tl.float32)
    idx = r.to(tl.int64) * N.to(tl.int64) + offs.to(tl.int64)
    v = tl.load(out_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    v = tl.where(count > 0, v / count, 0.0)
    tl.store(out_ptr + idx, v.to(out_ptr.dtype.element_ty), mask=mask)


def _csr_view(sparse):
    """``(crow, col, values)`` of any supported sparse layout."""
    layout = sparse.layout
    if layout == torch.sparse_csr:
        return sparse.crow_indices(), sparse.col_indices(), sparse.values()

    M = sparse.shape[0]

    if layout == torch.sparse_coo:
        # ``indices()``/``values()`` reject an uncoalesced tensor, so the raw
        # entry arrays are read and merged by the device kernels instead.
        coalesced = sparse.is_coalesced()
        indices = sparse.indices() if coalesced else sparse._indices()
        values = sparse.values() if coalesced else sparse._values()
        rows, cols = indices[0], indices[1]
        if not coalesced:
            rows, cols, values = coalesce_triplets(rows, cols, values, sparse.shape[1])
        crow = row_pointers_from_sorted(rows.to(torch.int32), M)
        return crow, cols, values

    if layout == torch.sparse_csc:
        ccol = sparse.ccol_indices().to(torch.int32)
        nnz = sparse._nnz()
        cols = csc_entry_columns(ccol, nnz)
        rows = sparse.row_indices()
        values = sparse.values()
        if nnz:
            sorted_rows, perm = _sort(rows.to(torch.int32))
            perm = perm.to(torch.int32)
            cols = gather(cols, perm)
            values = gather(values, perm)
        else:
            sorted_rows = rows.to(torch.int32)
        crow = row_pointers_from_sorted(sorted_rows, M)
        return crow, cols, values

    if layout in (torch.sparse_bsr, torch.sparse_bsc):
        return _csr_view(sparse.to_sparse_csr())

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

    crow, col, val = _csr_view(self)

    if reduce in ("sum", "mean"):
        # The reduction runs over the stored entries only, so it is exactly the
        # dense product of the row's stored values with ``other``: zero-fill the
        # unstored positions and let the in-tree matmul do the work. One program
        # fills one dense row, so no per-entry row index has to be derived.
        dense = csr_to_dense(crow, col, val, M, K)
        out = _gems_mm(dense, other)
        if reduce == "mean":
            with torch_device_fn.device(other.device):
                _mean_scale_kernel[(M, triton.cdiv(N, _BLOCK_N))](
                    out, crow, N, BLOCK_N=_BLOCK_N
                )
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
