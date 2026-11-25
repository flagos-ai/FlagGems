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

"""diagflat(self, offset=0) -> Tensor

Semantics (aten::diagflat): with ``t`` the C-order flatten of ``self``,
``n = t.numel()`` and ``m = n + |offset|``, the result is a contiguous
``(m, m)`` tensor with ``out.dtype == self.dtype`` such that

    out[i, i + offset]   = t[i]   for offset >= 0
    out[i + |offset|, i] = t[i]   for offset <  0

and every other element is exactly zero.

All numerical work happens in triton kernels:

  fill     persistent two-phase zero fill plus diagonal scatter, one launch
           for every real-dtype input (GCU400: a flat one-chunk-per-program
           fill of N programs costs ceil(N / 48) waves at ~2.5us per wave
           almost independently of the stored bytes, so 24 persistent
           programs -- one per SIP -- loop over contiguous BLOCK-sized
           chunk runs instead; 24 also beats 48 and any non-multiple of 24
           such as 32).  Phase one zeroes the program's whole contiguous
           run through a same-itemsize integer view of the freshly
           allocated contiguous output (dtype generic); ONE
           ``tl.debug_barrier`` then orders the run's zero stores before
           the diagonal stores; phase two writes the run's diagonal cells
           with a single SPAN-wide masked scatter at address
           ``i * (m + 1) + base`` (int64 index math, overflow-safe for any
           count below the int32 fallback line).  The source is read
           through the flattened input's *element stride* so a zero-copy
           strided view (e.g. ``x[..., ::2]``) is handled correctly.
  scatter  copies the ``n`` flattened input elements onto the offset-k
           diagonal with a single strided store at address
           ``i * (m + 1) + base``.  Used only by the complex /
           very-large fallback paths.
  gather   the reverse mapping, used by the autograd backward pass.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_BLOCK_FILL = 4096
_BLOCK_SCATTER = 512
_INT32_MAX = 2147483647

# GCU400 persistent fill geometry (measured 2026-11-13, Libra L600):
# a flat one-chunk-per-program fill of N programs costs ceil(N / 48) waves
# at ~2.5us per wave almost independently of the stored bytes (m=4096 fp16
# 16.8M-element fill: 213us at 4096 programs vs 14.2us at 24-48 persistent
# programs).  24 persistent programs (one per SIP) measure equal-or-better
# than 48 across the board and clearly beat non-SIP-multiples (32 is 10-15%
# worse at m>=2048).  Each chunk store keeps a <=128KB tile inside the
# 448KB-per-program DSM line, so the grid stays hardware-resident.
_FILL_PROGS = 24
_FILL_TILE_BYTES = 131072

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
def _diagflat_fill_diag_kernel(
    fill_ptr,
    out_ptr,
    src_ptr,
    count,
    m,
    base,
    sstride,
    ndiag,
    chunks,
    cpb,
    BLOCK: tl.constexpr,
    SPAN: tl.constexpr,
):
    """Persistent two-phase zero fill plus diagonal write in a single launch.

    Program ``pid`` owns the contiguous chunk run ``[pid * cpb, pid * cpb +
    cpb)`` (clamped to ``chunks``).  Phase one zeroes each chunk exactly as
    the pure fill does; phase two writes the diagonal cells that land inside
    the same run.  Diagonal element ``i`` lives at flat address
    ``i * (m + 1) + base_off`` and belongs to exactly one run, so no
    *cross-program* ordering is needed.  Within a program the two store
    sets overlap: the diagonal addresses are a subset of the span the same
    program just zeroed, and the lanes that zero a given address are not
    the lanes that write its value.  The single barrier between the phases
    orders them, without which a zero can land after a value and leave a
    nondeterministic handful of diagonal cells at 0.

    Phase two walks the run's diagonal indices directly (``i0 .. i0 +
    SPAN - 1``, int64 address math so no interior overflow is possible for
    any ``count`` below the host's int32 fallback line), so one masked
    load/store pair per program covers all of its diagonal cells.
    """
    pid = tl.program_id(0)
    start = pid * cpb
    end = tl.minimum(start + cpb, chunks)
    zero = tl.zeros([BLOCK], dtype=fill_ptr.dtype.element_ty)
    for c in range(start, end):
        b = c * BLOCK
        offs = b + tl.arange(0, BLOCK)
        if b + BLOCK <= count:
            tl.store(fill_ptr + offs, zero)
        else:
            tl.store(fill_ptr + offs, zero, mask=offs < count)
    tl.debug_barrier()

    lo = tl.minimum(start, end) * BLOCK
    hi = end * BLOCK
    stride = m + 1
    head = lo.to(tl.int64) - base
    i0 = tl.where(head <= 0, 0, (head + stride - 1) // stride)
    j = tl.arange(0, SPAN).to(tl.int64)
    idx = i0 + j
    addr = idx * stride + base
    valid = (idx < ndiag) & (addr < hi)
    val = tl.load(
        src_ptr + idx * sstride,
        mask=valid,
        other=tl.zeros([SPAN], dtype=src_ptr.dtype.element_ty),
    )
    tl.store(out_ptr + addr, val, mask=valid)


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
            num_warps=4,
        )


def _fill_with_diagonal(out, src, n, m, base, sstride):
    """Single-launch persistent fill + diagonal write for real dtypes.

    The chunk width starts near ``count / 24`` so the 24 persistent programs
    each own roughly one chunk, rounded up to a power of two and capped so
    one chunk tile stays within ``_FILL_TILE_BYTES`` (keeping the grid
    hardware-resident under the 448KB DSM line).  Returns False only when
    the flat int32 address range would overflow, in which case the caller
    falls back to the separate zero-fill and scatter launches.
    """
    itemsize = out.element_size()
    block = min(
        max(16, _FILL_TILE_BYTES // itemsize),
        triton.next_power_of_2(max(16, out.numel() // _FILL_PROGS)),
    )
    if out.numel() > _INT32_MAX - 2 * block:
        return False

    buf = out.view(_fill_view(itemsize))
    count = buf.numel()
    chunks = triton.cdiv(count, block)
    nprog = min(_FILL_PROGS, chunks)
    cpb = triton.cdiv(chunks, nprog)
    span = max(16, triton.next_power_of_2(cpb * block // (m + 1) + 2))
    _diagflat_fill_diag_kernel[(nprog,)](
        buf,
        out,
        src,
        count,
        m,
        base,
        sstride,
        n,
        chunks,
        cpb,
        BLOCK=block,
        SPAN=span,
        num_warps=4,
    )
    return True


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

    base, jmul, real_dtype = _diagonal_params(out, k)

    if real_dtype is None and n > 0:
        flat = self.reshape(-1)
        base, jmul, _ = _diagonal_params(out, k)
        if _fill_with_diagonal(out, flat, n, m, base, flat.stride(0)):
            return out

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
            num_warps=4,
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
        src, dst, count, sstride, base, jmul, BLOCK=_BLOCK_SCATTER, num_warps=4
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


def run(self, offset=0):
    logger.debug("GEMS_ENFLAME_GCU400 RUN")
    if not isinstance(self, torch.Tensor):
        raise TypeError("diagflat(): argument 'self' must be a Tensor")
    if not isinstance(offset, int):
        raise TypeError("diagflat(): argument 'offset' must be an integer")
    k = int(offset)
    if self.requires_grad:
        return _DiagFlat.apply(self, k)
    return _forward(self, k)
