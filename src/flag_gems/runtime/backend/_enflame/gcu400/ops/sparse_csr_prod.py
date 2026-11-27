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
"""GCU400 implementation of ``aten::_sparse_csr_prod``.

The operator reduces a 2-D sparse CSR tensor along ``dim`` and returns another
sparse CSR tensor. Only stored entries participate: an implicit (unstored) zero
is *not* multiplied in, which is why the result structure follows the layout and
not the values. Storing an explicit zero does drive the product to zero.

* ``dim=[1]`` (row reduction) keeps one entry per row that stores at least one
  value, at column 0. Rows are contiguous CSR segments, so one program per row
  walks the segment and multiplies.
* ``dim=[0]`` (column reduction) keeps one entry per column that stores at least
  one value, in ascending column order. Grouping the payload by column needs
  either a sort or an atomic pass, and neither is usable here (see below), so
  the payload is broadcast into a row-chunked dense buffer whose unstored cells
  hold the 1.0 identity and then folded along the rows. Chunking bounds the
  scratch memory; the fold stays vectorized.
* ``dim=[0, 1]`` (or an empty ``dim``) folds everything into one entry, present
  only when the input stores anything at all.

GCU400 differences from the common source, which aborts the process here:

* ``flag_gems.nonzero`` reduces a boolean flag through ``Tensor.cumsum`` on a
  bool buffer, which topsaten rejects ("The output tensor's data type must
  match the input tensor's data type") and which aborts the interpreter. The
  host side therefore uses ``torch.nonzero`` on an ``int32`` hit counter, the
  same scaffold the gcu400 ``sparse_csr_sum`` sibling uses.
* ``dim=[0]`` has no cheap column grouping on this backend: sorting the 630k
  indices of a benchmark case costs 5.7ms via ``flag_gems.sort`` (42.9ms via
  ``torch.sort``) and one atomic pass costs 3.4ms, since ``tl.atomic_add``
  neither returns the previous value nor runs fast here. The dense chunked fold
  replaces both and is bandwidth bound.
* ``torch_gcu`` stores float64 payloads in float32 storage and its Triton
  backend cannot lower fp64 elementwise regions, so the payload handed to the
  kernels is floored at float32 and the reduced result is widened back to
  ``out_dtype``. The native kernel accumulates in the same width.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_BLOCK = 256
_BLOCK_N = 256
_BLOCK_R = 32
_FILL_BLOCK = 1024
# Dense scratch budget for the dim=[0] fold: 4M cells is 8MB of fp16 payload
# plus 4MB of flags per chunk, and keeps the benchmark shapes at one chunk.
_DENSE_CHUNK_ELEMS = 1 << 22


@triton.jit
def _mul(a, b):
    return a * b


@triton.jit
def _row_prod_kernel(crow_ptr, val_ptr, out_ptr, cnt_ptr, BLOCK: tl.constexpr):
    """One program per row: multiply the row's contiguous payload segment."""
    r = tl.program_id(0)
    lo = tl.load(crow_ptr + r)
    n = tl.load(crow_ptr + r + 1) - lo

    acc = tl.full((), 1.0, dtype=tl.float32)
    off = 0
    while off < n:
        idx = off + tl.arange(0, BLOCK)
        mask = idx < n
        part = tl.reduce(
            tl.load(val_ptr + lo + idx, mask=mask, other=1).to(tl.float32), 0, _mul
        )
        acc *= part
        off += BLOCK

    tl.store(out_ptr + r, acc)
    tl.store(cnt_ptr + r, tl.where(n > 0, 1, 0))


@triton.jit
def _fill_kernel(ptr, n, VALUE: tl.constexpr, BLOCK: tl.constexpr):
    """Fill a flat buffer with ``VALUE`` (dense scratch initialisation)."""
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(ptr + idx, VALUE, mask=idx < n)


@triton.jit
def _scatter_row_kernel(
    crow_ptr,
    col_ptr,
    val_ptr,
    dense_val_ptr,
    dense_mask_ptr,
    row_offset,
    cols,
    BLOCK: tl.constexpr,
):
    """Broadcast one CSR row into its dense row of the chunk buffer.

    One program per chunk-local row: the segment ``crow[r] : crow[r + 1]`` is
    already contiguous, so the scatter is a data movement with no coordination.
    ``dense_val`` keeps the payload and ``dense_mask`` flags stored positions;
    everything not written stays at the 1.0 / 0 identity written by the fill.
    """
    local_row = tl.program_id(0)
    r = row_offset + local_row
    lo = tl.load(crow_ptr + r)
    n = tl.load(crow_ptr + r + 1) - lo
    base = local_row.to(tl.int64) * cols

    off = 0
    while off < n:
        idx = off + tl.arange(0, BLOCK)
        mask = idx < n
        c = tl.load(col_ptr + lo + idx, mask=mask, other=0).to(tl.int64)
        v = tl.load(val_ptr + lo + idx, mask=mask, other=1)
        tl.store(dense_val_ptr + base + c, v, mask=mask)
        tl.store(dense_mask_ptr + base + c, 1, mask=mask)
        off += BLOCK


@triton.jit
def _dense_col_reduce_kernel(
    dense_val_ptr,
    dense_mask_ptr,
    prod_ptr,
    cnt_ptr,
    rows,
    cols,
    BLOCK_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fold a dense chunk along its rows into per-column product and count.

    ``dense_val`` holds ``1.0`` at unstored positions, the multiplicative
    identity, so a plain column product over the chunk is the sparse column
    product restricted to that chunk. Results accumulate into ``prod_ptr`` /
    ``cnt_ptr`` so successive chunks compose by multiplication / addition.
    """
    pid = tl.program_id(0)
    offs_c = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_c = offs_c < cols

    acc_product = tl.load(prod_ptr + offs_c, mask=mask_c, other=1.0).to(tl.float32)
    acc_count = tl.load(cnt_ptr + offs_c, mask=mask_c, other=0).to(tl.int32)

    for r0 in range(0, rows, BLOCK_R):
        offs_r = r0 + tl.arange(0, BLOCK_R)
        tile_mask = (offs_r[:, None] < rows) & mask_c[None, :]
        tile_off = offs_r[:, None].to(tl.int64) * cols + offs_c[None, :]
        tile = tl.load(dense_val_ptr + tile_off, mask=tile_mask, other=1).to(tl.float32)
        flags = tl.load(dense_mask_ptr + tile_off, mask=tile_mask, other=0)
        acc_product *= tl.reduce(tile, 0, _mul)
        acc_count += tl.sum(flags.to(tl.int32), 0)

    tl.store(prod_ptr + offs_c, acc_product, mask=mask_c)
    tl.store(cnt_ptr + offs_c, acc_count, mask=mask_c)


@triton.jit
def _total_prod_kernel(val_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """Fold the whole payload into one value (single program)."""
    acc = tl.full((), 1.0, dtype=tl.float32)
    off = 0
    while off < N:
        idx = off + tl.arange(0, BLOCK)
        mask = idx < N
        part = tl.reduce(
            tl.load(val_ptr + idx, mask=mask, other=1).to(tl.float32), 0, _mul
        )
        acc *= part
        off += BLOCK
    tl.store(out_ptr, acc)


def _normalise_dims(dim, ndim=2):
    """Validate ``dim`` against the two dimensions CSR supports."""
    dims = list(dim)
    if not dims:
        return {0, 1}
    normalised = []
    for d in dims:
        d = int(d)
        if d < 0:
            d += ndim
        if not 0 <= d < ndim:
            raise RuntimeError(
                "dimension out of range for a 2-D sparse CSR tensor: " + str(d)
            )
        normalised.append(d)
    if len(set(normalised)) != len(normalised):
        raise RuntimeError("dim lists duplicate entries: " + str(dims))
    if set(normalised) not in ({0}, {1}, {0, 1}):
        raise RuntimeError(
            "sparse CSR prod supports reducing dim [0], [1], [0, 1] or [], "
            "got " + str(dims)
        )
    return set(normalised)


def _assemble(crow, col, values, size, dtype):
    """Wrap the reduced components into a sparse CSR tensor."""
    return torch.sparse_csr_tensor(crow, col, values, size, dtype=dtype)


def sparse_csr_prod(self, dim, keepdim=False, *, dtype=None):
    """Multiply a sparse CSR tensor along ``dim``, returning sparse CSR."""
    logger.debug("GEMS_ENFLAME SPARSE_CSR_PROD")
    if self.layout != torch.sparse_csr:
        raise RuntimeError("_sparse_csr_prod expects a sparse CSR input")
    if not keepdim:
        raise RuntimeError(
            "reduction operations on CSR tensors with keepdim=False is unsupported"
        )
    if self.dim() != 2 or self.dense_dim() != 0:
        raise RuntimeError(
            "_sparse_csr_prod supports 2-D CSR tensors with scalar payloads only"
        )

    dims = _normalise_dims(dim)
    crow = self.crow_indices()
    col = self.col_indices()
    values = self.values()
    rows, cols = self.shape[-2], self.shape[-1]
    nnz = values.numel()
    out_dtype = values.dtype if dtype is None else dtype
    device = values.device

    # fp64 is held in fp32 on this backend; both the kernel input and the
    # accumulator stay at fp32 and the result is widened once at the end.
    kernel_dtype = torch.float32 if out_dtype == torch.float64 else out_dtype
    if values.dtype != kernel_dtype:
        values = values.to(kernel_dtype)

    if dims == {1}:
        row_prod32 = torch.empty(rows, dtype=torch.float32, device=device)
        touched = torch.empty(rows, dtype=torch.int32, device=device)
        if rows:
            _row_prod_kernel[(rows,)](crow, values, row_prod32, touched, BLOCK=_BLOCK)
        # Empty rows survive: the output crow mirrors the input layout with one
        # entry per storing row at its original row position.
        keep = touched.bool()
        kept = row_prod32[keep].to(out_dtype)
        new_crow = torch.zeros(rows + 1, dtype=crow.dtype, device=device)
        torch.cumsum(keep.to(crow.dtype), 0, out=new_crow[1:])
        new_col = torch.zeros(int(keep.sum()), dtype=col.dtype, device=device)
        return _assemble(new_crow, new_col, kept, (rows, 1), out_dtype)

    if dims == {0}:
        # Grouping the payload by column is what makes this a column reduction
        # over row-major segments. Neither of the two usual primitives exists
        # here: sorting 630k indices costs 5.7ms through flag_gems.sort (42.9ms
        # through torch.sort) and a single atomic pass costs 3.4ms, because
        # tl.atomic_add neither returns the previous value nor runs fast on this
        # backend. The column order is therefore materialised densely, in row
        # chunks, so memory stays bounded while the fold remains vectorized.
        chunk_elems = min(_DENSE_CHUNK_ELEMS, rows * cols) if rows and cols else 0
        chunk_rows = max(1, min(rows, chunk_elems // max(cols, 1))) if cols else 1
        dense_val = torch.empty(chunk_rows * cols, dtype=kernel_dtype, device=device)
        dense_mask = torch.empty(chunk_rows * cols, dtype=torch.int8, device=device)
        prod_acc = torch.ones(cols, dtype=torch.float32, device=device)
        cnt_acc = torch.zeros(cols, dtype=torch.int32, device=device)
        grid_reduce = (triton.cdiv(cols, _BLOCK_N),) if cols else None

        start = 0
        while start < rows and nnz:
            this_rows = min(chunk_rows, rows - start)
            cells = this_rows * cols
            _fill_kernel[(triton.cdiv(cells, _FILL_BLOCK),)](
                dense_val, cells, VALUE=1.0, BLOCK=_FILL_BLOCK
            )
            _fill_kernel[(triton.cdiv(cells, _FILL_BLOCK),)](
                dense_mask, cells, VALUE=0, BLOCK=_FILL_BLOCK
            )
            _scatter_row_kernel[(this_rows,)](
                crow,
                col,
                values,
                dense_val,
                dense_mask,
                start,
                cols,
                BLOCK=_BLOCK,
            )
            _dense_col_reduce_kernel[grid_reduce](
                dense_val,
                dense_mask,
                prod_acc,
                cnt_acc,
                this_rows,
                cols,
                BLOCK_R=_BLOCK_R,
                BLOCK_N=_BLOCK_N,
            )
            start += this_rows

        # Keep only the columns that store something, in ascending column order;
        # the payload is already the per-column product in that same order.
        keep = torch.nonzero(cnt_acc).reshape(-1) if cols else cnt_acc
        kept = prod_acc.to(kernel_dtype)
        kept = kept.index_select(0, keep) if keep.numel() else kept[:0]
        new_col = keep.to(col.dtype)
        new_crow = torch.zeros(2, dtype=crow.dtype, device=device)
        new_crow[1] = keep.numel()
        return _assemble(new_crow, new_col, kept.to(out_dtype), (1, cols), out_dtype)

    # Fully reduced: a single stored value, present only if anything is stored.
    # With nothing stored the native kernel still allocates a 0-d value buffer
    # but leaves the descriptor empty (crow = [0, 0]).
    new_crow = torch.zeros(2, dtype=crow.dtype, device=device)
    if nnz:
        kept32 = torch.empty(1, dtype=torch.float32, device=device)
        _total_prod_kernel[(1,)](values, kept32, nnz, BLOCK=_BLOCK)
        kept = kept32.to(out_dtype)
        new_crow[1] = 1
        new_col = torch.zeros(1, dtype=col.dtype, device=device)
    else:
        kept = torch.empty((), dtype=out_dtype, device=device)
        new_col = torch.empty(0, dtype=col.dtype, device=device)
    return _assemble(new_crow, new_col, kept, (1, 1), out_dtype)


def sparse_csr_prod_out(self, dim, keepdim=False, *, dtype=None, out):
    """Out form: rewrite ``out`` with the reduced descriptor and return it."""
    logger.debug("GEMS_ENFLAME SPARSE_CSR_PROD OUT")
    built = sparse_csr_prod(self, dim, keepdim, dtype=dtype)
    # ``copy_`` between compressed tensors requires an identical stored-element
    # count; anything else raises exactly like the native out kernel.
    out.copy_(built)
    return out
