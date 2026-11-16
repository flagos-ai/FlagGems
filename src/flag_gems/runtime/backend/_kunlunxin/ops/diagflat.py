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

"""Kunlunxin XPU3 override for ``diagflat``.

The generic implementation (``src/flag_gems/ops/diagflat.py``) has two fused
paths that are unsafe on this backend:

* ``_diagflat_fill_diag_kernel`` zero-fills a span and then writes the diagonal
  of that same span through a second store, ordered only by
  ``tl.debug_barrier()``.  On XPU3 the barrier does not order global-memory
  writes across threads, so a nondeterministic handful of diagonal cells are
  left at zero (baseline: 18 failed / 229 passed).
* ``_diagflat_tile_kernel`` reads the diagonal via a 2-D masked load
  (``mask=diag, other=zeros``) and stores through a 2-D tile whose row stride is
  the runtime ``m``.  Both are miscompiled here: masked-in lanes lose their load
  (the cell comes back 0) and, for small odd ``m``, the 2-D store writes to
  wrong addresses (measured: m=10, offset +3 puts a spurious value at [8, 0];
  offset -3 loses almost every diagonal cell).

This override keeps only the flat 1-D path, which is bit-exact:

  fill     one vector store of zeros over a same-itemsize integer view of the
           freshly allocated contiguous output (BLOCK=8192; BLOCK>=65536 was
           measured to fault with a misaligned address on this backend).
  scatter  copies the ``n`` flattened input elements onto the offset-k diagonal
           with one strided store at address ``i * (m + 1) + base``.  The source
           is read through the flattened input's element stride, so a zero-copy
           strided view (e.g. ``x[..., ::2]``) is handled correctly.
  gather   the reverse mapping, used by the autograd backward pass.

``n == 0`` short-circuits on the host: no kernel may issue an unmasked load from
an empty source (that escalated to an XPU XID during bring-up).

Both launches are disjoint in what they write (the scatter only writes the
``n`` diagonal cells), so no device-side ordering is required.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_BLOCK_FILL = 8192
_BLOCK_SCATTER = 512
_INT32_MAX = 2147483647

_COMPLEX_REAL = {
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}


def _fill_view(itemsize):
    """A same-or-larger-itemsize integer view type used for the zero fill."""
    if itemsize >= 8:
        return torch.int64
    if itemsize == 4:
        return torch.int32
    if itemsize == 2:
        return torch.int16
    return torch.uint8


@triton.jit
def _diagflat_zero_kernel(out_ptr, count, BLOCK: tl.constexpr, WIDE: tl.constexpr):
    """Vectorised zero fill.

    ``count`` is a runtime scalar, so a plain masked store lets the element
    mask defeat triton's store-vectorisation analysis whenever ``count`` is not
    a multiple of the vector width (which happens for odd ``m * m``).  Only the
    final, possibly partial program needs the mask, so every fully in-range
    program issues an unmasked store and keeps its 128-bit vectorisation.
    """
    pid = tl.program_id(0)
    if WIDE:
        base = pid.to(tl.int64) * BLOCK
    else:
        base = pid * BLOCK
    offs = base + tl.arange(0, BLOCK)
    zero = tl.zeros([BLOCK], dtype=out_ptr.dtype.element_ty)
    if base + BLOCK <= count:
        tl.store(out_ptr + offs, zero)
    else:
        tl.store(out_ptr + offs, zero, mask=offs < count)


@triton.jit
def _diagflat_scatter_kernel(
    src_ptr, dst_ptr, count, sstride, dstride, base, jmul, BLOCK: tl.constexpr
):
    """dst[i * dstride + base - (i & 1) * jmul] = src[i * sstride]"""
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < count
    val = tl.load(
        src_ptr + offs * sstride,
        mask=mask,
        other=tl.zeros([BLOCK], dtype=src_ptr.dtype.element_ty),
    )
    tl.store(dst_ptr + (offs * dstride + base - (offs & 1) * jmul), val, mask=mask)


@triton.jit
def _diagflat_gather_kernel(
    src_ptr, dst_ptr, count, sstride, base, jmul, BLOCK: tl.constexpr
):
    """dst[i] = src[i * sstride + base - (i & 1) * jmul]"""
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < count
    val = tl.load(
        src_ptr + (offs * sstride + base - (offs & 1) * jmul),
        mask=mask,
        other=tl.zeros([BLOCK], dtype=src_ptr.dtype.element_ty),
    )
    tl.store(dst_ptr + offs, val, mask=mask)


def _zero_fill(out):
    buf = out.view(_fill_view(out.element_size()))
    count = buf.numel()
    if count:
        grid = (triton.cdiv(count, _BLOCK_FILL),)
        _diagflat_zero_kernel[grid](
            buf,
            count,
            BLOCK=_BLOCK_FILL,
            WIDE=(count + 2 * _BLOCK_FILL) > _INT32_MAX,
        )


def _diagonal_params(out, k):
    """``(base, jmul, real_dtype)`` for the offset-k diagonal of ``out``.

    For real dtypes the target address is ``i * (m + 1) + base`` with
    ``jmul == 0``.  Complex tensors are viewed as their real component type with
    a doubled inner extent, giving ``i * (m + 1) + base - (i & 1) * m``.
    """
    m = out.shape[0]
    if out.is_complex():
        base = 2 * k if k >= 0 else 2 * (-k) * m
        return base, m, _COMPLEX_REAL[out.dtype]
    base = k if k >= 0 else (-k) * m
    return base, 0, None


def _forward(self, k):
    n = self.numel()
    m = n + (k if k >= 0 else -k)
    out = torch.empty((m, m), dtype=self.dtype, device=self.device)
    if out.numel() == 0:
        return out

    # Every element outside the diagonal is an exact zero.  The fill owns the
    # whole buffer and the scatter owns only the n diagonal addresses below, so
    # the two launches never write the same address and need no ordering.
    _zero_fill(out)

    if n > 0:
        flat = self.reshape(-1)
        base, jmul, real_dtype = _diagonal_params(out, k)
        if real_dtype is not None:
            if not flat.is_contiguous():
                raise NotImplementedError(
                    "diagflat: non-contiguous complex input is not supported"
                )
            src = flat.view(real_dtype)
            dst = out.view(real_dtype)
            count = 2 * n
            sstride = 1
        else:
            src = flat
            dst = out
            count = n
            sstride = flat.stride(0)
        grid = (triton.cdiv(count, _BLOCK_SCATTER),)
        _diagflat_scatter_kernel[grid](
            src,
            dst,
            count,
            sstride,
            m + 1,
            base,
            jmul,
            BLOCK=_BLOCK_SCATTER,
        )
    return out


def _backward(grad_out, shape, k):
    m = grad_out.shape[0]
    n = m - (k if k >= 0 else -k)
    grad_in = torch.empty(shape, dtype=grad_out.dtype, device=grad_out.device)
    if n <= 0 or grad_in.numel() == 0:
        return grad_in

    if grad_out.is_complex():
        real_dtype = _COMPLEX_REAL[grad_out.dtype]
        src = grad_out.view(real_dtype)
        dst = grad_in.view(real_dtype)
        count = 2 * n
        sstride = m + 1
        base = 2 * k if k >= 0 else 2 * (-k) * m
        jmul = m
    else:
        src = grad_out
        dst = grad_in
        count = n
        sstride = m + 1
        base = k if k >= 0 else (-k) * m
        jmul = 0

    grid = (triton.cdiv(count, _BLOCK_SCATTER),)
    _diagflat_gather_kernel[grid](
        src, dst, count, sstride, base, jmul, BLOCK=_BLOCK_SCATTER
    )
    return grad_in


class _DiagFlat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, self, offset):
        ctx.input_shape = self.shape
        ctx.offset = offset
        return _forward(self, offset)

    @staticmethod
    def backward(ctx, grad_out):
        return _backward(grad_out, ctx.input_shape, ctx.offset), None


def diagflat(self, offset=0):
    if not isinstance(self, torch.Tensor):
        raise TypeError("diagflat(): argument 'self' must be a Tensor")
    if not isinstance(offset, int):
        raise TypeError("diagflat(): argument 'offset' must be an integer")
    k = int(offset)
    if self.requires_grad:
        return _DiagFlat.apply(self, k)
    return _forward(self, k)
