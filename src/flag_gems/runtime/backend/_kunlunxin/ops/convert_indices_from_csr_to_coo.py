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
"""Kunlunxin XPU3 override for ``aten::_convert_indices_from_csr_to_coo``.

The generic implementation
(``src/flag_gems/ops/convert_indices_from_csr_to_coo.py``) mis-lowers on XPU3:

1. **The generic kernel does not compile.** Its coordinate metadata loop is a
   *dynamic* ``for k in range(nd)`` loop, and the ``tl.load``/``tl.store``
   inside it hit ``TritonXPUUnrollControl``, which emits an invalid
   ``arith.addi``.  Every positive test case fails with
   ``'arith.addi' op requires the same type for all operands and results``.
   Making ``nd`` a ``tl.constexpr`` and iterating ``tl.static_range(ND)``
   removes the dynamic loop and compiles.
2. **Three-plus masked stores combined with a gather load break the same
   pass.** The operator must write ``nd + 2`` output planes.  With
   ``nd >= 1`` the kernel performs at least three masked stores *and* a
   data-dependent gather (the bisection over ``crow``); the XPU pass then
   fails with ``'arith.cmpi' op failed to verify ...``.  Two masked stores
   together with the gather compile fine, and the ``nd`` plane-only stores
   compile fine on their own.  The fix splits the work into two kernels:
   a coordinate-plane kernel (masked stores only) and an X/Y kernel (gather
   plus exactly two masked stores).  Only ``nd >= 1`` needs both kernels.
3. **The generic bisection can issue an all-false masked load, which aborts
   the program on XPU3** and segfaults.  A lane whose search value is the last
   row converges before the fixed trip count, so the remaining iterations have
   ``mid == n_rows`` and an empty ``valid`` mask.  The vendor kernel derives
   ``b``/``p`` from the clamp-terminated entry index ``e_safe`` (so inactive
   lanes use the safe base ``0``), clamps the within-block bisection index to
   ``n_rows - 1`` and issues the probe as an **unmasked** load, turning every
   read into an in-bounds one.  Correctness still uses the true ``mid`` in the
   comparison, so the residual result is identical.

The vendor host reproduces the generic wrapper's validation and output-buffer
semantics exactly; only the launch differs.  The metadata tensors are reused
from the generic module's cached helper.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.convert_indices_from_csr_to_coo import (
    _ALLOWED_DTYPES,
    _batch_metadata,
)

logger = logging.getLogger(__name__)

# Stored entries per program (same width as the generic kernel).
_BLOCK = 256
_NUM_WARPS = 2


@triton.jit
def _csr_to_coo_planes_kernel(
    out_ptr,
    ext_ptr,
    bdiv_ptr,
    nnz,
    total,
    ND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Write the ``ND`` leading batch-coordinate planes (stores only)."""
    e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = e < total
    e_safe = tl.where(m, e, 0)
    b = e_safe // nnz
    for k in tl.static_range(ND):
        bd = tl.load(bdiv_ptr + k).to(tl.int32)
        ex = tl.load(ext_ptr + k).to(tl.int32)
        coord = (b // bd) % ex
        off = e_safe.to(tl.int64) + k * total
        tl.store(out_ptr + off, coord.to(out_ptr.dtype.element_ty), mask=m)


@triton.jit
def _csr_to_coo_xy_kernel(
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
    crow_ls,
    col_ls,
    ND: tl.constexpr,
    STEPS: tl.constexpr,
    HAS_ROWS: tl.constexpr,
    OUT_INT32: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Write the X and Y planes via an exact fixed-trip bisection."""
    e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = e < total
    e_safe = tl.where(m, e, 0)
    # Derive the batch decomposition from the clamped entry index so inactive
    # lanes use the safe base 0 and every address below stays in bounds.
    b = e_safe // nnz
    p = e_safe % nnz

    cb = tl.zeros((BLOCK,), dtype=tl.int64)
    vb = tl.zeros((BLOCK,), dtype=tl.int64)
    for k in tl.static_range(ND):
        bd = tl.load(bdiv_ptr + k).to(tl.int32)
        ex = tl.load(ext_ptr + k).to(tl.int32)
        coord = (b // bd) % ex
        cb += coord.to(tl.int64) * tl.load(crow_stride_ptr + k).to(tl.int64)
        vb += coord.to(tl.int64) * tl.load(col_stride_ptr + k).to(tl.int64)

    lo = tl.zeros((BLOCK,), dtype=tl.int32)
    if HAS_ROWS:
        hi = tl.full((BLOCK,), n_rows, tl.int32)
        for _ in tl.static_range(STEPS):
            mid = (lo + hi) >> 1
            valid = m & (mid < n_rows)
            # Clamp the within-block index and drop the mask: inactive lanes
            # read block 0, and the true `valid` still gates the comparison.
            midc = tl.minimum(mid, n_rows - 1)
            addr = cb + midc.to(tl.int64) * crow_ls
            mv = tl.load(crow_ptr + addr)
            c = valid & (mv.to(tl.int32) <= p)
            lo = tl.where(c, mid + 1, lo)
            hi = tl.where(c, hi, mid)

    row = lo - 1
    colv = tl.load(col_ptr + vb + p.to(tl.int64) * col_ls, mask=m, other=0)

    if TRANSPOSE:
        x = colv
        y = row
    else:
        x = row
        y = colv
    if OUT_INT32:
        x = x.to(tl.int32)
        y = y.to(tl.int32)
    off = e_safe.to(tl.int64) + ND * total
    tl.store(out_ptr + off, x.to(out_ptr.dtype.element_ty), mask=m)
    tl.store(out_ptr + off + total, y.to(out_ptr.dtype.element_ty), mask=m)


def _launch(crow, col, out_int32, transpose, out):
    """Kunlunxin launch path; mirrors the generic wrapper's host semantics."""
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
    if nd > 0:
        _csr_to_coo_planes_kernel[grid](
            out,
            ext,
            divisors,
            nnz,
            total,
            ND=nd,
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
        )
    _csr_to_coo_xy_kernel[grid](
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
        crow.stride(-1),
        col.stride(-1),
        ND=nd,
        STEPS=max(r.bit_length(), 1),
        HAS_ROWS=r > 0,
        OUT_INT32=bool(out_int32),
        TRANSPOSE=bool(transpose),
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )
    return out


def _convert_indices_from_csr_to_coo(
    crow_indices, col_indices, *, out_int32=False, transpose=False, out=None
):
    """Decode COO coordinates from CSR row pointers and column indices."""
    logger.debug("GEMS _CONVERT_INDICES_FROM_CSR_TO_COO")
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
    logger.debug("GEMS _CONVERT_INDICES_FROM_CSR_TO_COO_OUT")
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


__all__ = [
    "_convert_indices_from_csr_to_coo",
    "_convert_indices_from_csr_to_coo_out",
]
