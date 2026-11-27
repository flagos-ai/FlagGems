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

"""gcu400 ``_cast_Half``: narrowing elementwise cast to float16.

Difference from the common implementation: the common kernel loads a
``float64`` / ``complex128`` element directly, which this backend cannot
legalize (``triton_gcu.elementwise_fusion_region`` rejects 64-bit element
types).  On gcu400 a ``float64`` tensor is physically stored as packed
``float32`` words (the dtype keeps its 8-byte claim but element ``i`` lives at
byte offset ``4 * i``), and ``complex128`` is the same storage seen as two
``float32`` words per element.  Reading those sources through a
``triton.reinterpret(..., tl.float32)`` view therefore produces exactly the
native ``double -> float -> half`` rounding while keeping the whole conversion
inside Triton, and it also matches the strided views (the strides stay in
element units, so the packed addressing follows them unchanged).

The float16 identity case, the output layout (``empty_like`` preserve_format
semantics) and the autograd bridge keep the semantics of the common op.  The
backward pass writes the float16 gradient back with the same kernels; for a
``float64`` destination the values are stored through the same packed
``float32`` view.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_INT32_LIMIT = 1 << 31

# Source dtypes that gcu400 stores as float32 words: they are read through a
# float32 pointer instead of a 64-bit load (see the module docstring).
_F32_VIEW_DTYPES = (torch.float64, torch.complex128)
_COMPLEX_DTYPES = (torch.complex64, torch.complex128)

# Launch bands retuned for gcu400 with interleaved do_bench(median) sweeps of
# the flat kernel (see the task report): a persistent grid capped at two CTAs
# per SM plus a large tile is what reaches the copy floor at 1 M elements and
# beyond, while tiny shapes want the smallest tile and one CTA.
_FLAT_BLOCK_TINY = 1024
_FLAT_WARPS_TINY = 4
_FLAT_BLOCK_MID = 2048
_FLAT_WARPS_MID = 4
_FLAT_BLOCK_LARGE = 32768
_FLAT_WARPS_LARGE = 2
_TINY_LIMIT = 2048
_MID_LIMIT = 1 << 16
_MAX_GRID = 48
_STRIDED_BLOCK = 1024
_STRIDED_WARPS = 4
# 2-D strided tile: the innermost row run is loaded in one tile row so the
# stride-2 style gathers stay wide; only small tensors fall back to more rows.
_STRIDED_2D_BLOCK_M = 64
_STRIDED_2D_BLOCK_N = 512
_STRIDED_2D_WARPS = 4
# Shift-split flat gather (power-of-two inner extent); 4096 lanes at two warps
# measured fastest on the stride-2 slice probe.
_SHIFT_BLOCK = 4096
_SHIFT_WARPS = 2


@triton.jit(do_not_specialize=["n_elem"])
def _cast_flat_kernel(
    src,
    dst,
    n_elem,
    WIDE: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Contiguous source: persistent grid, one converted store per element.

    The grid is capped at two CTAs per SM and each CTA strides over its share
    of tiles; a small tile with one CTA per tile leaves the memory system idle
    at 1 M elements and above on gcu400.
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (n_elem + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        offs = block_id * BLOCK + arange
        if WIDE:
            offs = offs.to(tl.int64)
        if EVEN:
            # ``n_elem`` is a multiple of BLOCK, so every tile is full and the
            # per-lane mask is redundant.
            tl.store(dst + offs, tl.load(src + offs).to(tl.float16))
        else:
            mask = offs < n_elem
            val = tl.load(src + offs, mask=mask)
            tl.store(dst + offs, val.to(tl.float16), mask=mask)


@triton.jit(do_not_specialize=["n_elem"])
def _cast_shift_kernel(
    src,
    dst,
    n_elem,
    src_stride_row,
    src_stride_col,
    dst_stride_row,
    dst_stride_col,
    SHIFT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Flat gather of a 2-D view whose inner extent is a power of two.

    Splitting the flat index into (row, column) then costs a shift and a mask
    instead of an integer divide; combined with the large persistent tile this
    is what turns a stride-2 slice from ~7x slower than the native cast into
    ~1.6x faster (see the task report).
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (n_elem + BLOCK - 1) // BLOCK
    inner_mask: tl.constexpr = (1 << SHIFT) - 1
    for block_id in tl.range(pid, num_blocks, num_pids):
        idx = block_id * BLOCK + arange
        mask = idx < n_elem
        row = idx >> SHIFT
        col = idx & inner_mask
        val = tl.load(
            src + row * src_stride_row + col * src_stride_col, mask=mask, other=0
        )
        tl.store(
            dst + row * dst_stride_row + col * dst_stride_col,
            val.to(tl.float16),
            mask=mask,
        )


@triton.jit
def _cast_2d_kernel(
    src,
    dst,
    M,
    N,
    src_stride_m,
    src_stride_n,
    dst_stride_m,
    dst_stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Two-dimensional strided conversion without any integer division.

    A plain per-element unravel costs an int64 divide per dimension, which
    dominates the runtime of every strided case; indexing a 2-D tile with
    multiply-adds instead keeps the gather loop cheap.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    src_off = offs_m[:, None] * src_stride_m + offs_n[None, :] * src_stride_n
    dst_off = offs_m[:, None] * dst_stride_m + offs_n[None, :] * dst_stride_n
    val = tl.load(src + src_off, mask=mask, other=0)
    tl.store(dst + dst_off, val.to(tl.float16), mask=mask)


@triton.jit
def _cast_strided_kernel(
    src,
    dst,
    n_elem,
    SHAPE: tl.constexpr,
    SRC_STRIDES: tl.constexpr,
    DST_STRIDES: tl.constexpr,
    COMPONENTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Unravel the flat index with compile-time shape/stride metadata.

    ``COMPONENTS`` is 2 for complex sources: their strides are expressed in
    complex elements while the pointer is a float32 view, so one complex step
    equals two float32 steps.
    """
    pid = tl.program_id(0)
    offs = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    mask = offs < n_elem
    rem = offs
    src_off = tl.zeros([BLOCK], dtype=tl.int64)
    dst_off = tl.zeros([BLOCK], dtype=tl.int64)
    for dim in tl.static_range(len(SHAPE) - 1, -1, -1):
        extent = SHAPE[dim]
        quotient = rem // extent
        src_off += (rem - quotient * extent) * SRC_STRIDES[dim]
        dst_off += (rem - quotient * extent) * DST_STRIDES[dim]
        rem = quotient
    if COMPONENTS == 2:
        src_off *= 2
    val = tl.load(src + src_off, mask=mask, other=0)
    tl.store(dst + dst_off, val.to(tl.float16), mask=mask)


@triton.jit
def _backward_flat_kernel(
    src,
    dst,
    n_elem,
    OUT_MODE: tl.constexpr,
    ALIGNED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Widen the float16 gradient back to the input dtype."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if ALIGNED:
        val = tl.load(src + offs)
        if OUT_MODE == 0:
            tl.store(dst + offs, val.to(tl.float32))
        else:
            tl.store(dst + offs, val.to(tl.bfloat16))
    else:
        mask = offs < n_elem
        val = tl.load(src + offs, mask=mask)
        if OUT_MODE == 0:
            tl.store(dst + offs, val.to(tl.float32), mask=mask)
        else:
            tl.store(dst + offs, val.to(tl.bfloat16), mask=mask)


@triton.jit
def _complex_backward_kernel(
    grad,
    dst,
    n_elem,
    SHAPE: tl.constexpr,
    GRAD_STRIDES: tl.constexpr,
    OUT_MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Complex gradient: keep the real component and zero the imaginary one."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    rem = offs
    grad_off = tl.full((BLOCK,), 0, tl.int64)
    for dim in tl.static_range(len(SHAPE) - 1, -1, -1):
        quotient = rem // SHAPE[dim]
        grad_off += (rem - quotient * SHAPE[dim]) * GRAD_STRIDES[dim]
        rem = quotient
    value = tl.load(grad + grad_off, mask=mask, other=0)
    if OUT_MODE == 0:
        value = value.to(tl.float32)
    else:
        value = value.to(tl.bfloat16)
    tl.store(dst + 2 * offs, value, mask=mask)
    tl.store(dst + 2 * offs + 1, 0, mask=mask)


def _flat_launch_config(n_elem):
    """Pick the (tile, warps) pair for a fully contiguous conversion."""
    if n_elem <= _TINY_LIMIT:
        return _FLAT_BLOCK_TINY, _FLAT_WARPS_TINY
    if n_elem <= _MID_LIMIT:
        return _FLAT_BLOCK_MID, _FLAT_WARPS_MID
    return _FLAT_BLOCK_LARGE, _FLAT_WARPS_LARGE


def _f32_word_alias(source, components):
    """Alias a packed float32 source as a float32 tensor (metadata only).

    ``torch.Tensor.data_ptr()`` still advances the storage offset by the 8-byte
    (16-byte for complex128) element claim, which does not match the packed
    4-byte word layout, so a strided source with a non-zero storage offset would
    start at the wrong address.  Rebuilding the view over a float32 storage
    fixes both the base offset and the dtype in one metadata-only step.
    """
    words = torch.empty(0, dtype=torch.float32, device=source.device)
    words = words.set_(
        source.untyped_storage(), 0, (source.numel() * components,), (1,)
    )
    return words.as_strided(
        source.shape, source.stride(), source.storage_offset() * components
    )


def _cast_source_pointer(source):
    """Return (load pointer, complex components) for the source tensor."""
    if source.dtype in _F32_VIEW_DTYPES or source.dtype == torch.complex64:
        components = 2 if source.dtype in _COMPLEX_DTYPES else 1
        return _f32_word_alias(source, components), components
    return source, 1


def _inner_extent_is_pow2(shape):
    extent = shape[-1]
    return extent > 0 and extent & (extent - 1) == 0


def _is_dense(shape, strides):
    """True when the layout covers a contiguous span without gaps or overlap."""
    expected = 1
    for dim in sorted(range(len(shape)), key=lambda d: strides[d]):
        if shape[dim] != 1 and strides[dim] != expected:
            return False
        expected *= shape[dim]
    return True


def _launch_flat(src_ptr, out, n_elem):
    block, warps = _flat_launch_config(n_elem)
    grid = min(triton.cdiv(n_elem, block), _MAX_GRID)
    _cast_flat_kernel[(grid,)](
        src_ptr,
        out,
        n_elem,
        WIDE=n_elem >= _INT32_LIMIT,
        EVEN=n_elem % block == 0,
        BLOCK=block,
        num_warps=warps,
    )


def _launch_shift(src_ptr, out, n_elem, src_strides, components):
    """Flat gather with a shift-based row/column split (power-of-two width)."""
    cols = out.shape[-1]
    block = _SHIFT_BLOCK
    grid = min(triton.cdiv(n_elem, block), _MAX_GRID)
    _cast_shift_kernel[(grid,)](
        src_ptr,
        out,
        n_elem,
        src_strides[0] * components,
        src_strides[1] * components,
        out.stride(0),
        out.stride(1),
        SHIFT=cols.bit_length() - 1,
        BLOCK=block,
        num_warps=_SHIFT_WARPS,
    )


def _launch_2d(src_ptr, out, shape, src_strides, components):
    rows, cols = shape
    grid = (
        triton.cdiv(rows, _STRIDED_2D_BLOCK_M),
        triton.cdiv(cols, _STRIDED_2D_BLOCK_N),
    )
    _cast_2d_kernel[grid](
        src_ptr,
        out,
        rows,
        cols,
        src_strides[0] * components,
        src_strides[1] * components,
        out.stride(0),
        out.stride(1),
        BLOCK_M=_STRIDED_2D_BLOCK_M,
        BLOCK_N=_STRIDED_2D_BLOCK_N,
        num_warps=_STRIDED_2D_WARPS,
    )


def _cast_forward(source, out):
    """Convert ``source`` into the float16 buffer ``out``."""
    n_elem = source.numel()
    if n_elem == 0:
        return out
    src_ptr, components = _cast_source_pointer(source)
    source_strides = tuple(source.stride())
    out_strides = tuple(out.stride())
    if source.is_contiguous() and out.is_contiguous():
        _launch_flat(src_ptr, out, n_elem)
    elif (
        components == 1
        and source_strides == out_strides
        and _is_dense(source.shape, source_strides)
    ):
        # The output keeps the preserve_format layout of the source, so equal
        # strides mean both buffers hold the same value at the same storage
        # offset: a permuted (e.g. transposed) input degenerates to a flat copy
        # of its dense span, which is far faster than a per-element gather.
        _launch_flat(src_ptr, out, n_elem)
    elif source.ndim == 2 and _inner_extent_is_pow2(source.shape):
        _launch_shift(src_ptr, out, n_elem, source_strides, components)
    elif source.ndim == 2:
        _launch_2d(src_ptr, out, source.shape, source_strides, components)
    else:
        _cast_strided_kernel[(triton.cdiv(n_elem, _STRIDED_BLOCK),)](
            src_ptr,
            out,
            n_elem,
            SHAPE=tuple(source.shape),
            SRC_STRIDES=source_strides,
            DST_STRIDES=out_strides,
            COMPONENTS=components,
            BLOCK=_STRIDED_BLOCK,
            num_warps=_STRIDED_WARPS,
        )
    return out


def _cast_backward(grad, dst, out_mode):
    """Convert the float16 gradient into the (possibly packed) ``dst``."""
    n_elem = grad.numel()
    if n_elem == 0:
        return dst
    dst_ptr = dst
    if dst.dtype == torch.float64:
        # float64 is stored as packed float32 words on gcu400.
        dst_ptr = triton.reinterpret(dst, tl.float32)
    if grad.is_contiguous() and dst.is_contiguous():
        block, warps = _flat_launch_config(n_elem)
        grid = min(triton.cdiv(n_elem, block), _MAX_GRID)
        _backward_flat_kernel[(grid,)](
            grad,
            dst_ptr,
            n_elem,
            OUT_MODE=out_mode,
            ALIGNED=n_elem % block == 0,
            BLOCK=block,
            num_warps=warps,
        )
    else:
        _cast_strided_kernel[(triton.cdiv(n_elem, _STRIDED_BLOCK),)](
            grad,
            dst_ptr,
            n_elem,
            SHAPE=tuple(grad.shape),
            SRC_STRIDES=tuple(grad.stride()),
            DST_STRIDES=tuple(dst.stride()),
            COMPONENTS=1,
            BLOCK=_STRIDED_BLOCK,
            num_warps=_STRIDED_WARPS,
        )
    return dst


def _complex_backward(grad, dst, out_mode):
    """Write a complex gradient (real component only) into ``dst``."""
    n_elem = grad.numel()
    if n_elem == 0:
        return dst
    dst_ptr = dst
    if dst.dtype == torch.complex128:
        dst_ptr = triton.reinterpret(dst, tl.float32)
    _complex_backward_kernel[(triton.cdiv(n_elem, _STRIDED_BLOCK),)](
        grad,
        dst_ptr,
        n_elem,
        SHAPE=tuple(grad.shape),
        GRAD_STRIDES=tuple(grad.stride()),
        OUT_MODE=out_mode,
        BLOCK=_STRIDED_BLOCK,
        num_warps=_STRIDED_WARPS,
    )
    return dst


_MODE_FP32 = 0
_MODE_BF16 = 1
_BACKWARD_MODE = {
    torch.float32: _MODE_FP32,
    torch.float64: _MODE_FP32,
    torch.bfloat16: _MODE_BF16,
    torch.complex64: _MODE_FP32,
    torch.complex128: _MODE_FP32,
}
_MODE_DTYPE = {
    _MODE_FP32: torch.float32,
    _MODE_BF16: torch.bfloat16,
}


class _CastHalf(torch.autograd.Function):
    """Differentiable wrapper so the float16 gradient can be cast back."""

    @staticmethod
    def forward(ctx, source, backward_mode):
        ctx.backward_mode = backward_mode
        ctx.input_dtype = source.dtype
        ctx.complex_input = source.is_complex()
        # Native ``_cast_Half`` allocates with ``empty_like`` preserve_format
        # semantics, which ``torch.empty_like`` reproduces by default.  The
        # gems allocator wrappers cost tens of microseconds per call here.
        out = torch.empty_like(source, dtype=torch.float16)
        return _cast_forward(source, out)

    @staticmethod
    def backward(ctx, grad):
        if ctx.complex_input:
            out = torch.empty(grad.shape, dtype=ctx.input_dtype, device=grad.device)
            return _complex_backward(grad, out, ctx.backward_mode), None
        out = torch.empty_like(grad, dtype=_MODE_DTYPE[ctx.backward_mode])
        return _cast_backward(grad, out, ctx.backward_mode), None


def cast_Half(self, non_blocking=False):
    """Narrowing cast of ``self`` to float16 (native ``aten::_cast_Half``)."""
    logger.debug("GEMS_ENFLAME CAST_HALF")
    if not isinstance(self, torch.Tensor):
        raise TypeError(
            "_cast_Half(): argument 'self' must be a Tensor, not "
            f"{type(self).__name__}"
        )
    if isinstance(non_blocking, str):
        raise TypeError("_cast_Half(): argument 'non_blocking' must be a bool, not str")

    # The native cast is a no-op for a float16 source: same object, any layout.
    if self.dtype == torch.float16:
        return self

    if self.requires_grad and torch.is_grad_enabled():
        backward_mode = _BACKWARD_MODE.get(self.dtype)
        if backward_mode is not None:
            return _CastHalf.apply(self, backward_mode)

    return _cast_forward(self, torch.empty_like(self, dtype=torch.float16))
