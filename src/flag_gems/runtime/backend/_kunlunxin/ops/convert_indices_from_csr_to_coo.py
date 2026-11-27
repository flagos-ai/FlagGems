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
"""Triton implementation of ``aten::_convert_indices_from_csr_to_coo``.

The decoded table has shape ``(nd + 2, N * nnz)``: the leading planes carry
the batch coordinate of each stored entry, the next plane carries X and the
last one Y, with ``(X, Y) = (row, col)`` -- or ``(col, row)`` when
``transpose=True``. The row of entry ``p`` inside a batch block is
``searchsorted_right(crow_block, p) - 1``, evaluated by an exact fixed-trip
bisection, one lane per stored entry: the row index of every entry is decoded
on the device, no index table is materialised on the host.

Batch bases come from the declared strides, so non-contiguous inputs are read
correctly with no data movement. The per-dim leading extents, strides and
row-major divisors are the only host-known descriptors; they are packed into
three small int64 device arrays once per geometry.
"""

import functools
import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Index dtypes the native operator accepts; floats, bool and complex have no
# kernel and are rejected by the wrapper.
_ALLOWED_DTYPES = (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)

# Stored entries per program. A 128-wide tile keeps the dependent gather chain
# short enough for the lowering budget of this backend.
_BLOCK = 128
_NUM_WARPS = 2


@triton.jit(
    do_not_specialize=[
        "nnz",
        "n_rows",
        "total",
        "nd",
        "crow_ls",
        "col_ls",
        "steps",
    ]
)
def _csr_to_coo_kernel(
    crow_ptr,
    col_ptr,
    out_ptr,
    ext_ptr,
    crow_stride_ptr,
    col_stride_ptr,
    bdiv_ptr,
    nnz,
    n_rows,
    total,
    nd,
    crow_ls,
    col_ls,
    steps,
    BLOCK: tl.constexpr,
    OUT_INT32: tl.constexpr,
    TRANSPOSE: tl.constexpr,
):
    # Every index-width-sensitive scalar is widened once: mixing a narrow
    # launch argument with the i64 lane index does not lower here, so all
    # arithmetic below is explicitly int64.
    total64 = total.to(tl.int64)
    nnz64 = nnz.to(tl.int64)
    n_rows64 = n_rows.to(tl.int64)
    crow_ls64 = crow_ls.to(tl.int64)
    col_ls64 = col_ls.to(tl.int64)

    e = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = e < total64

    b = e // nnz64
    p = e % nnz64

    # Per-entry batch base offsets and batch-coordinate planes. ``bdiv[k]`` is
    # prod(lead[k+1:]), so the row-major coordinate of dim k is
    # (b // bdiv[k]) % lead[k].
    cb = tl.zeros((BLOCK,), dtype=tl.int64)
    vb = tl.zeros((BLOCK,), dtype=tl.int64)
    for k in range(nd):
        coord = (b // tl.load(bdiv_ptr + k)) % tl.load(ext_ptr + k)
        cb += coord * tl.load(crow_stride_ptr + k)
        vb += coord * tl.load(col_stride_ptr + k)
        tl.store(
            out_ptr + k * total64 + e,
            coord.to(out_ptr.dtype.element_ty),
            mask=m,
        )

    # Exact residual bisection: row = searchsorted_right(crow_block, p) - 1.
    # The last row pointer is the block's sentinel, so the bracket is [0, R).
    lo = tl.zeros((BLOCK,), dtype=tl.int64)
    hi = tl.zeros((BLOCK,), dtype=tl.int64) + n_rows64
    for _ in range(steps):
        mid = (lo + hi) >> 1
        valid = m & (mid < n_rows64)
        mv = tl.load(crow_ptr + cb + mid * crow_ls64, mask=valid, other=0)
        c = valid & (mv.to(tl.int64) <= p)
        lo = tl.where(c, mid + 1, lo)
        hi = tl.where(c, hi, mid)

    row = lo - 1
    colv = tl.load(col_ptr + vb + p * col_ls64, mask=m, other=0).to(tl.int64)

    if TRANSPOSE:
        x = colv
        y = row
    else:
        x = row
        y = colv
    if OUT_INT32:
        x = x.to(tl.int32)
        y = y.to(tl.int32)
    nd64 = tl.full((), nd, tl.int64)
    tl.store(out_ptr + nd64 * total64 + e, x.to(out_ptr.dtype.element_ty), mask=m)
    tl.store(out_ptr + (nd64 + 1) * total64 + e, y.to(out_ptr.dtype.element_ty), mask=m)


@functools.lru_cache(maxsize=128)
def _metadata_arrays(nd, extents, crow_strides, col_strides, device):
    """Pack the host-known geometry into small int64 device tensors.

    These are shape/stride descriptors, not index data: the per-entry row and
    column indices are decoded by the kernel itself.
    """
    extents = list(extents)
    crow_strides = list(crow_strides)
    col_strides = list(col_strides)
    # bdiv[k] = prod(extents[k + 1:]) -- 1 for the innermost dim.
    divisors = [1] * nd
    for k in range(nd - 2, -1, -1):
        divisors[k] = divisors[k + 1] * extents[k + 1]

    def to_dev(values):
        return torch.tensor(values, dtype=torch.int64, device=device)

    return (
        to_dev(extents),
        to_dev(crow_strides),
        to_dev(col_strides),
        to_dev(divisors),
    )


def _batch_metadata(crow, col, nd):
    """Per-dim leading extents / strides and row-major batch divisors."""
    device = crow.device
    extents = tuple(int(crow.shape[k]) for k in range(nd))
    crow_strides = tuple(int(crow.stride(k)) for k in range(nd))
    col_strides = tuple(int(col.stride(k)) for k in range(nd))
    return _metadata_arrays(nd, extents, crow_strides, col_strides, device)


def _launch(crow, col, out_int32, transpose, out):
    nd = crow.dim() - 1
    crow_shape = crow.shape
    for k in range(nd):
        if crow_shape[k] != col.shape[k]:
            raise RuntimeError(
                "The size of tensor a (%d) must match the size of tensor b (%d) "
                "at non-singleton dimension %d" % (crow_shape[k], col.shape[k], k)
            )
    r = int(crow_shape[-1]) - 1
    nnz = int(col.shape[-1])
    n_batch = 1
    for d in crow_shape[:nd]:
        n_batch *= int(d)
    total = n_batch * nnz

    out_dtype = torch.int32 if out_int32 else torch.int64
    if out is None:
        out = torch.empty((nd + 2, total), dtype=out_dtype, device=crow.device)
    else:
        if out.dtype != out_dtype:
            raise RuntimeError(
                f"Expected out tensor to have dtype {out_dtype}, but got {out.dtype}"
            )
        if tuple(out.shape) != (nd + 2, total):
            out.resize_((nd + 2, total))

    if total == 0:
        return out

    ext, crow_strides, col_strides, divisors = _batch_metadata(crow, col, nd)
    grid = (triton.cdiv(total, _BLOCK),)
    _csr_to_coo_kernel[grid](
        crow,
        col,
        out,
        ext,
        crow_strides,
        col_strides,
        divisors,
        nnz,
        r,
        total,
        nd,
        crow.stride(-1),
        col.stride(-1),
        max(int(r).bit_length(), 1),
        BLOCK=_BLOCK,
        OUT_INT32=bool(out_int32),
        TRANSPOSE=bool(transpose),
        num_warps=_NUM_WARPS,
    )
    return out


def _convert_indices_from_csr_to_coo(
    crow_indices, col_indices, *, out_int32=False, transpose=False, out=None
):
    """Decode COO coordinates from CSR row pointers and column indices."""
    logger.debug("GEMS_KUNLUNXIN _CONVERT_INDICES_FROM_CSR_TO_COO")
    if not isinstance(out_int32, bool):
        raise TypeError(f"out_int32 must be a bool, got {type(out_int32).__name__}")
    if not isinstance(transpose, bool):
        raise TypeError(f"transpose must be a bool, got {type(transpose).__name__}")
    crow = crow_indices
    col = col_indices
    for name, t in (("crow_indices", crow), ("col_indices", col)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(
                f"_convert_indices_from_csr_to_coo: {name} must be a Tensor, "
                f"got {type(t).__name__}"
            )
        if t.dtype not in _ALLOWED_DTYPES:
            raise NotImplementedError(
                '"convert_indices_from_csr_to_coo_cuda" not implemented for '
                f"'{str(t.dtype).replace('torch.', '')}'"
            )
    if crow.dim() == 0 or col.dim() == 0:
        raise IndexError("Dimension specified as -1 but tensor has no dimensions")
    if crow.dim() != col.dim():
        raise RuntimeError(
            "crow_indices and col_indices are supposed to have the same "
            "dimensionality, but got %d and %d dimensional tensors, "
            "respectively." % (crow.dim(), col.dim())
        )
    return _launch(crow, col, out_int32, transpose, out)


def _convert_indices_from_csr_to_coo_out(
    crow_indices, col_indices, *, out_int32=False, transpose=False, out
):
    """``aten::_convert_indices_from_csr_to_coo.out``: write into ``out``."""
    logger.debug("GEMS_KUNLUNXIN _CONVERT_INDICES_FROM_CSR_TO_COO_OUT")
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_convert_indices_from_csr_to_coo.out: out must be a Tensor, got "
            f"{type(out).__name__}"
        )
    return _convert_indices_from_csr_to_coo(
        crow_indices,
        col_indices,
        out_int32=out_int32,
        transpose=transpose,
        out=out,
    )
