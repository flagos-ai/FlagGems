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
"""In-place Triton implementation of ``aten::_foreach_neg_``.

Semantics: for every tensor in the input sequence (in order, duplicates are
negated as many times as they appear) flip the sign of every element in place,
preserving storage, storage_offset, shape, stride and dtype; return ``None``.

The negation is exact in every element width (IEEE sign flip for floats,
two's-complement wrap for integers, real/imag flip for complex), so no
accumulator or precision decision is involved.  The work is a pure streaming
read-modify-write, so the kernel is bandwidth bound.

Two launch shapes are used:

* a fused kernel that processes up to 16 flat tensors of one dtype in a single
  launch (grid ``(chunks, n_tensors)``), used when the whole list is flat, of
  one dtype and span-disjoint.  On this target a kernel launch costs tens of
  microseconds and each runtime argument a few more, so one narrow launch per
  list instead of one launch per tensor is what makes many-small-tensor lists
  competitive.  Equal-sized groups use a variant with a single shared element
  count (17 runtime arguments instead of 33).
* one launch per tensor otherwise (mixed dtypes, gapped views, empty tensors,
  or spans that could alias).

Because host time is inside the measured latency, the wrapper classifies every
element with one dtype dictionary probe and no per-element helper calls on the
common contiguous path, and reuses the data pointers it already read for the
aliasing check.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_unary import _foreach_neg_ as default__foreach_neg_

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# ---------------------------------------------------------------------------
# dtype plumbing
# ---------------------------------------------------------------------------

_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
    torch.int8: tl.int8,
    torch.int16: tl.int16,
    torch.int32: tl.int32,
    torch.int64: tl.int64,
    torch.uint8: tl.uint8,
}

_ITEMSIZE = {dtype: dtype.itemsize for dtype in _TL_DTYPE}

# A complex element is addressed as two real components; negating both
# components independently is exactly complex negation.
_COMPLEX_TO_REAL = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}

_BOOL = torch.bool

_OVERLAP_MSG = (
    "unsupported operation: more than one element of the written-to tensor "
    "refers to a single memory location. Please clone() the tensor before "
    "performing the operation."
)

_I32_LIMIT = (1 << 31) - 1 - 65536

# Tensors handled by one fused launch.
_MAX_FUSE = 16

# spec kinds
_FLAT64 = 1
_STRIDED = 2


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@triton.jit
def _neg_flat_kernel(x_ptr, n_elements, BLOCK: tl.constexpr, USE_I64: tl.constexpr):
    """Negate ``[0, n_elements)`` of the tensor behind ``x_ptr`` in place.

    ``x_ptr`` is a real tensor argument so the launcher's 16-byte divisibility
    specialization reaches the compiler; that is what lets the 2-byte dtypes
    issue full-width vector accesses instead of one 16-bit access per lane.

    The flat scan is legal for contiguous tensors and for non-overlapping dense
    views, where the storage span covers exactly the tensor's elements.
    """
    pid = tl.program_id(0)
    if USE_I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    tl.store(x_ptr + offs, (-x).to(x.dtype), mask=mask)


@triton.jit
def _neg_fused_uniform_kernel(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    n_elements,
    BLOCK: tl.constexpr,
):
    """Negate up to 16 equal-sized, flat, same-dtype tensors in one launch.

    Grid axis 0 is the chunk within a tensor and axis 1 the tensor, so
    consecutively dispatched CTAs (blockIdx.x-major) sweep one tensor's storage
    contiguously.  With the tensor on the fast axis a resident wave would
    interleave 16 regions whose bases differ only in high address bits, which
    measured 1.14x-1.60x slower as the regions got further apart.  Selecting
    the pointer with a uniform scalar chain (instead of a device-side pointer
    table) keeps the launcher's per-argument divisibility information, so the
    wide vector accesses of the single-tensor kernel are preserved.

    A shared element count keeps the runtime signature at 17 arguments; each
    runtime argument costs a couple of microseconds of launch time here.
    """
    t = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x_ptr = p0
    if t == 1:
        x_ptr = p1
    elif t == 2:
        x_ptr = p2
    elif t == 3:
        x_ptr = p3
    elif t == 4:
        x_ptr = p4
    elif t == 5:
        x_ptr = p5
    elif t == 6:
        x_ptr = p6
    elif t == 7:
        x_ptr = p7
    elif t == 8:
        x_ptr = p8
    elif t == 9:
        x_ptr = p9
    elif t == 10:
        x_ptr = p10
    elif t == 11:
        x_ptr = p11
    elif t == 12:
        x_ptr = p12
    elif t == 13:
        x_ptr = p13
    elif t == 14:
        x_ptr = p14
    elif t == 15:
        x_ptr = p15

    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    tl.store(x_ptr + offs, (-x).to(x.dtype), mask=mask)


@triton.jit
def _neg_fused_kernel(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    c0,
    c1,
    c2,
    c3,
    c4,
    c5,
    c6,
    c7,
    c8,
    c9,
    c10,
    c11,
    c12,
    c13,
    c14,
    c15,
    BLOCK: tl.constexpr,
):
    """Fused variant for lists whose tensors have different element counts.

    Same CTA mapping and pointer selection as
    :func:`_neg_fused_uniform_kernel`, but each tensor carries its own count.
    """
    t = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x_ptr = p0
    n_elements = c0
    if t == 1:
        x_ptr = p1
        n_elements = c1
    elif t == 2:
        x_ptr = p2
        n_elements = c2
    elif t == 3:
        x_ptr = p3
        n_elements = c3
    elif t == 4:
        x_ptr = p4
        n_elements = c4
    elif t == 5:
        x_ptr = p5
        n_elements = c5
    elif t == 6:
        x_ptr = p6
        n_elements = c6
    elif t == 7:
        x_ptr = p7
        n_elements = c7
    elif t == 8:
        x_ptr = p8
        n_elements = c8
    elif t == 9:
        x_ptr = p9
        n_elements = c9
    elif t == 10:
        x_ptr = p10
        n_elements = c10
    elif t == 11:
        x_ptr = p11
        n_elements = c11
    elif t == 12:
        x_ptr = p12
        n_elements = c12
    elif t == 13:
        x_ptr = p13
        n_elements = c13
    elif t == 14:
        x_ptr = p14
        n_elements = c14
    elif t == 15:
        x_ptr = p15
        n_elements = c15

    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    tl.store(x_ptr + offs, (-x).to(x.dtype), mask=mask)


@triton.jit
def _neg_strided_kernel(
    base,
    n_elements,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    RANK: tl.constexpr,
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Negate the elements of a gapped view described by shape/strides.

    Linear index ``i`` is decomposed innermost-dim-first so the generated
    address is ``base + sum_d ((i // prod_{e>d} s_e) % s_d) * t_d``.  Strides
    are in elements of ``DT``, so typed pointer arithmetic is exact.
    """
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_elements

    rem = offs
    off = tl.zeros([BLOCK], tl.int64)
    if RANK > 7:
        off += (rem % s7) * t7
        rem = rem // s7
    if RANK > 6:
        off += (rem % s6) * t6
        rem = rem // s6
    if RANK > 5:
        off += (rem % s5) * t5
        rem = rem // s5
    if RANK > 4:
        off += (rem % s4) * t4
        rem = rem // s4
    if RANK > 3:
        off += (rem % s3) * t3
        rem = rem // s3
    if RANK > 2:
        off += (rem % s2) * t2
        rem = rem // s2
    if RANK > 1:
        off += (rem % s1) * t1
        rem = rem // s1
    off += (rem % s0) * t0

    ptr = base.to(tl.pointer_type(DT))
    x = tl.load(ptr + off, mask=mask, other=0)
    tl.store(ptr + off, (-x).to(DT), mask=mask)


# ---------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------


def _block_for(itemsize: int) -> int:
    """~4 KiB per program: 8 fp32 / 16 fp16 / 32 int8 elements per lane."""
    blk = 4096 // itemsize
    return blk if blk >= 256 else 256


def _is_flat_addressable(t: torch.Tensor) -> bool:
    """True when ``[data_ptr, data_ptr + numel)`` covers exactly the tensor.

    This is ``is_non_overlapping_and_dense`` computed from strides (the MUSA
    torch build does not expose that predicate): sorted by stride, the strides
    of all non-unit dimensions must be the prefix products of their sizes.
    ``is_contiguous`` is checked first because it answers the common case in
    one C-level call.
    """
    if t.is_contiguous():
        return True
    if t.numel() <= 1:
        return True
    expected = 1
    for stride, size in sorted(
        (s, sz) for s, sz in zip(t.stride(), t.shape) if sz != 1
    ):
        if stride != expected:
            return False
        expected *= size
    return True


def _has_internal_overlap(t: torch.Tensor) -> bool:
    probe = getattr(torch, "_debug_has_internal_overlap", None)
    if probe is None:
        return False
    return probe(t) == 1


def _slow_plan(t: torch.Tensor, real, n: int, is_complex: bool):
    """Plan a gapped view / complex tensor (rare paths)."""
    if _has_internal_overlap(t):
        raise RuntimeError(_OVERLAP_MSG)
    shape = list(t.shape)
    stride = list(t.stride())
    if is_complex:
        # Complex element offsets are doubled into real units; the inner size-2
        # dimension selects the real/imaginary component.
        shape.append(2)
        stride = [s * 2 for s in stride] + [1]
        n *= 2
    rank = len(shape)
    if rank > 8:
        raise NotImplementedError("strided negation supports rank <= 8")
    shape += [1] * (8 - rank)
    stride += [0] * (8 - rank)
    return (_STRIDED, t.data_ptr(), n, real, rank, shape, stride)


def _launch_flat(t: torch.Tensor) -> None:
    n = t.numel()
    blk = _block_for(_ITEMSIZE[t.dtype])
    _neg_flat_kernel[((n + blk - 1) // blk,)](t, n, blk, False)


def _launch_plan(plan) -> None:
    kind = plan[0]
    if kind == _FLAT64:
        _, t, n = plan
        blk = _block_for(_ITEMSIZE[t.dtype])
        _neg_flat_kernel[((n + blk - 1) // blk,)](t, n, blk, True)
        return

    _, base, n_real, real, rank, shape, stride = plan
    blk = 256
    _neg_strided_kernel[((n_real + blk - 1) // blk,)](
        base,
        n_real,
        shape[0],
        shape[1],
        shape[2],
        shape[3],
        shape[4],
        shape[5],
        shape[6],
        shape[7],
        stride[0],
        stride[1],
        stride[2],
        stride[3],
        stride[4],
        stride[5],
        stride[6],
        stride[7],
        rank,
        _TL_DTYPE[real],
        blk,
    )


def _launch_fused(buf) -> None:
    nt = len(buf)
    first = buf[0]
    blk = _block_for(_ITEMSIZE[first.dtype])
    counts = [t.numel() for t in buf]
    max_n = counts[0]
    uniform = True
    for c in counts:
        if c > max_n:
            max_n = c
            uniform = False
        elif c < max_n:
            uniform = False
    chunks = (max_n + blk - 1) // blk
    ptrs = list(buf) + [first] * (_MAX_FUSE - nt)
    if uniform:
        _neg_fused_uniform_kernel[(chunks, nt)](*ptrs, max_n, blk)
    else:
        _neg_fused_kernel[(chunks, nt)](*ptrs, *(counts + [0] * (_MAX_FUSE - nt)), blk)


def _foreach_neg_(self):
    logger.debug("GEMS_MTHREADS _FOREACH_NEG_")
    _seq = self if isinstance(self, (list, tuple)) else [self]
    if any(not isinstance(t, torch.Tensor) or t.device.type != "musa" for t in _seq):
        return default__foreach_neg_(self)
    if not isinstance(self, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    if len(self) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    items = []
    dtype0 = None
    spans = []
    fuse_ok = True

    for t in self:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        d = t.dtype
        if d is _BOOL:
            raise RuntimeError("\"foreach\" not implemented for 'bool'")
        real = _COMPLEX_TO_REAL.get(d)
        is_complex = real is not None
        if not is_complex:
            real = d
        if real not in _TL_DTYPE:
            raise RuntimeError(
                "\"foreach\" not implemented for '%s'" % str(d).replace("torch.", "")
            )
        n = t.numel()
        if n == 0:
            continue

        if not is_complex and _is_flat_addressable(t):
            if n <= _I32_LIMIT:
                items.append(t)
                if fuse_ok:
                    if dtype0 is None:
                        dtype0 = d
                    elif d is not dtype0:
                        fuse_ok = False
                    if fuse_ok:
                        start = t.data_ptr()
                        spans.append((start, start + n * _ITEMSIZE[d]))
                continue
            # Very large flat tensor: 64-bit indexing, never fused.
            items.append((_FLAT64, t, n))
            fuse_ok = False
            continue

        fuse_ok = False
        items.append(_slow_plan(t, real, n, is_complex))

    if fuse_ok and len(items) > 1:
        prev_end = -1
        for lo, hi in sorted(spans):
            if lo < prev_end:
                fuse_ok = False
                break
            prev_end = hi

    if fuse_ok and len(items) > 1:
        # One launch per group of at most _MAX_FUSE disjoint equal-dtype
        # tensors.
        for i in range(0, len(items), _MAX_FUSE):
            _launch_fused(items[i : i + _MAX_FUSE])
        return None

    for it in items:
        if type(it) is tuple:
            _launch_plan(it)
        else:
            _launch_flat(it)
    return None
