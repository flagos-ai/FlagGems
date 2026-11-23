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

"""Triton implementation of ``aten::int_repr`` / ``aten::int_repr.out``.

``int_repr`` returns the raw integer codes stored by an affine-quantized
tensor: ``torch.quint8 -> torch.uint8``, ``torch.qint8 -> torch.int8`` and
``torch.qint32 -> torch.int32``.  The scale / zero-point metadata never enters
the result, so the operation is a bit-exact typed memory copy of the quantized
storage (rtol = 0, atol = 0) and the device kernels below perform exactly that:
one masked integer load and one masked integer store per element.

Layout strategy (mirrors the native operator, which preserves memory format):
the freshly allocated output is given the *same* strides as the input alias
whenever that alias is storage-dense.  ``channels_last``, ``transposed``,
``offset_slice`` and per-channel inputs are all storage-dense, so their element
offsets from ``data_ptr()`` are exactly ``[0, numel)`` and the copy degenerates
to one linear span with no index arithmetic and coalesced accesses.  A
rank-decomposing kernel remains for genuinely non-dense layouts (zero-stride
``expand``, stepped slices) and for a caller-supplied ``out`` buffer whose
strides differ from the input's.

Alignment strategy: a 1-byte element offset (for example the ``self[1:]``
slice, whose base pointer sits one byte into the allocation) stops Triton from
proving 16-byte alignment and costs most of the achievable bandwidth.  Such a
span is split into a short unaligned head, a 16-byte-aligned bulk and a short
tail.  Each piece is then copied by the *flat* kernel over a rank-1 alias whose
data pointer already carries the element shift; passing the shift inside the
kernel instead would leave a runtime addend in the address expression and the
profile shows that costs the compiler its wide-access vectorisation (144
registers, 16.6% occupancy, 36.6% DRAM versus 10.8 us for the flat kernel on
the same element count).
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

__all__ = ["int_repr"]


# Quantized storage dtype -> plain integer dtype of the result.
_Q_TO_INT = {
    torch.quint8: torch.uint8,
    torch.qint8: torch.int8,
    torch.qint32: torch.int32,
}

# Number of elements each program copies along a dense/linear span.
_BLOCK = 16384
# GCU400 dense-copy tuning results (from the 1-byte-element viewpoint):
# Time is roughly proportional to the program count (~60ns fixed overhead per program), and per-thread
# memory-access width determines effective bandwidth. So the dense span switched to "int64/int32 wide view + fixed
# grid + an in-program loop moving 64KB": 16M bytes improved from 506 GB/s to about
# 1900 GB/s (ATen clone roofline about 2000 GB/s).
_COPY_CHUNK_BYTES = 65536
_COPY_MAX_PROGRAMS = 48
# Spans smaller than that byte count go straight to the flat kernel; the wide-view benefit does not offset the extra
# metadata overhead.
_COPY_WIDE_MIN_BYTES = 4096
# Small program used for the unaligned head of a split span.
_HEAD_BLOCK = 32
# Longest rank the generic (strided) kernel decomposes in one unrolled pass.
_MAX_RANK = 5
# Byte alignment Triton needs to emit widened (>= 128-bit) loads and stores.
_ALIGN_BYTES = 16
# Below this element count a split costs more (extra launches) than it saves.
_SPLIT_MIN_ELEMS = 1 << 20


@triton.jit
def _int_repr_flat_kernel(in_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    """Linear copy of a dense span: one integer element per lane."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    val = tl.load(in_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, val, mask=mask)


@triton.jit
def _int_repr_wide_copy_kernel(
    in_ptr,
    out_ptr,
    count,
    tail_in_ptr,
    tail_out_ptr,
    tail_count,
    BLOCK: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    """Fixed-grid wide-element copy.

    Each program first copies ``BLOCK`` wide elements (int32/int64) of a chunk, then advances
    ``NUM_CHUNKS`` rounds with a ``grid * BLOCK`` stride. Compared to the "one program moves one
    small segment" unrolling, this squeezes the program count under 48 and amortizes program
    startup, while the per-thread access width is guaranteed by the wide view.

    A tail shorter than one wide element (``numel`` < 8 bytes) is handled by program 0 with byte
    copies, avoiding another kernel launch for a few bytes.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    for _ in range(NUM_CHUNKS):
        mask = offs < count
        val = tl.load(in_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, val, mask=mask)
        offs += step

    if tl.program_id(0) == 0:
        residual = tl.arange(0, 8)
        residual_mask = residual < tail_count
        tail = tl.load(tail_in_ptr + residual, mask=residual_mask)
        tl.store(tail_out_ptr + residual, tail, mask=residual_mask)


@triton.jit
def _int_repr_strided_kernel(
    in_ptr,
    out_ptr,
    numel,
    n0: tl.constexpr,
    n1: tl.constexpr,
    n2: tl.constexpr,
    n3: tl.constexpr,
    n4: tl.constexpr,
    ia0: tl.constexpr,
    ia1: tl.constexpr,
    ia2: tl.constexpr,
    ia3: tl.constexpr,
    ia4: tl.constexpr,
    ob0: tl.constexpr,
    ob1: tl.constexpr,
    ob2: tl.constexpr,
    ob3: tl.constexpr,
    ob4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy through arbitrary (possibly zero / non-dense) strides.

    The logical shape is right-padded with size-1 / stride-0 slots so a single
    unrolled decomposition covers every rank.  Indices are decomposed from the
    innermost dimension outwards, exactly like a row-major multi-index walk.
    Sizes and strides are ``constexpr``: the divisors then fold into
    multiply-shift sequences and the strides become immediates, instead of the
    full 64-bit integer division a runtime divisor would cost per element.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel

    rem = offs
    iidx = offs * 0
    oidx = offs * 0

    j = rem % n4
    iidx += j * ia4
    oidx += j * ob4
    rem = rem // n4

    j = rem % n3
    iidx += j * ia3
    oidx += j * ob3
    rem = rem // n3

    j = rem % n2
    iidx += j * ia2
    oidx += j * ob2
    rem = rem // n2

    j = rem % n1
    iidx += j * ia1
    oidx += j * ob1
    rem = rem // n1

    j = rem % n0
    iidx += j * ia0
    oidx += j * ob0

    val = tl.load(in_ptr + iidx, mask=mask)
    tl.store(out_ptr + oidx, val, mask=mask)


def _int_view(tensor, dtype):
    """Zero-copy integer alias of a quantized tensor's storage.

    ``Tensor.view(dtype)`` is tried first (it preserves size, stride and
    storage offset); when the build refuses the dtype change the storage is
    aliased through ``set_`` with identical metadata, which is a metadata-only
    reinterpretation (no device work, no payload allocation).
    """
    try:
        return tensor.view(dtype)
    except (RuntimeError, TypeError, NotImplementedError):
        alias = torch.empty(0, dtype=dtype, device=tensor.device)
        alias.set_(
            tensor.untyped_storage(),
            tensor.storage_offset(),
            tensor.size(),
            tensor.stride(),
        )
        return alias


def _linear_alias(tensor, dtype, shift):
    """Rank-1 alias of ``tensor``'s storage starting ``shift`` elements later.

    Used to hand the device a base pointer that already carries the element
    shift, so the kernel address expression stays ``base + arange`` and keeps
    Triton's contiguous-access analysis.  The result is a metadata-only view of
    the same storage (no device work, no payload allocation).
    """
    alias = torch.empty(0, dtype=dtype, device=tensor.device)
    alias.set_(
        tensor.untyped_storage(),
        tensor.storage_offset() + shift,
        (tensor.numel() - shift,),
        (1,),
    )
    return alias


def _plain_int_alias(tensor, dtype):
    """Plain (non-quantized) integer alias of ``tensor``'s storage.

    ``_int_view`` keeps the quantized flag when ``Tensor.view(dtype)``
    succeeds, which is harmless for a Triton kernel (it only reads the data
    pointer) but routes a subsequent ATen copy through the quantized dispatch
    keys.  Building the alias with ``set_`` on an empty plain tensor drops the
    flag, so the copy runs on the ordinary integer kernels.
    """
    alias = torch.empty(0, dtype=dtype, device=tensor.device)
    alias.set_(
        tensor.untyped_storage(),
        tensor.storage_offset(),
        tensor.size(),
        tensor.stride(),
    )
    return alias


def _is_storage_dense(numel, shape, stride):
    """True when the strides enumerate one dense span of ``numel`` elements.

    Contiguous layouts short-circuit; otherwise the dimensions are walked from
    the smallest stride upwards and each stride must equal the product of the
    sizes already covered.  Zero strides (``expand``) and stepped slices fail,
    so they keep the generic index-decomposing kernel.
    """
    if numel <= 1:
        return True
    if len(shape) <= 1:
        return stride[-1] == 1
    expect = 1
    for n, s in sorted(zip(shape, stride), key=lambda p: p[1]):
        if n <= 1:
            continue
        if s != expect:
            return False
        expect *= n
    return True


def _padded(n, seq, fill):
    """Right-pad ``seq`` to exactly ``n`` slots.

    A longer sequence must never grow the result: the caller splats it into a
    fixed-arity kernel signature, so one extra entry silently becomes the next
    parameter (it used to land on ``BLOCK`` and abort the launch).
    """
    return (fill,) * max(n - len(seq), 0) + tuple(int(v) for v in seq[-n:])


def _floor_pow2(value):
    """Rounds down to a power of two (tile sizes must be powers of two)."""
    result = 1
    while result * 2 <= value:
        result *= 2
    return result


def _wide_alias(tensor, dtype, count):
    """Aliases the first ``count`` ``dtype`` elements of ``tensor`` storage as rank-1.

    Pure metadata (no data movement). The caller must guarantee the storage start offset is
    already aligned to ``dtype.itemsize``, otherwise the base is not an integer index.
    """
    base = tensor.storage_offset() * tensor.element_size()
    alias = torch.empty(0, dtype=dtype, device=tensor.device)
    alias.set_(tensor.untyped_storage(), base // dtype.itemsize, (count,), (1,))
    return alias


def _launch_dense_copy(in_view, out, numel):
    """Copies ``numel`` elements through the widest aligned view; returns False when unavailable.

    A wide view (int64 / int32) moves more bytes per thread and, with a fixed grid, amortizes
    program overhead so the effective bandwidth of a dense span approaches ATen copy kernels.
    Only enabled for 1-byte elements (quint8 / qint8): qint32 elements are already 4 bytes wide
    and byte aliasing is not general, so those keep the flat kernel.
    """
    if in_view.element_size() != 1 or out.element_size() != 1:
        return False
    if numel < _COPY_WIDE_MIN_BYTES:
        return False
    src_ptr = in_view.data_ptr()
    dst_ptr = out.data_ptr()
    for dtype in (torch.int64, torch.int32):
        width = dtype.itemsize
        if src_ptr % width or dst_ptr % width:
            continue
        count = numel // width
        if count == 0:
            continue
        tail = numel - count * width
        block = min(_COPY_CHUNK_BYTES // width, _floor_pow2(count))
        programs = min(_COPY_MAX_PROGRAMS, triton.cdiv(count, block))
        chunks = triton.cdiv(count, block * programs)
        _int_repr_wide_copy_kernel[(programs,)](
            _wide_alias(in_view, dtype, count),
            _wide_alias(out, dtype, count),
            count,
            _linear_alias(in_view, torch.uint8, count * width),
            _linear_alias(out, torch.uint8, count * width),
            tail,
            BLOCK=block,
            NUM_CHUNKS=chunks,
        )
        return True
    return False


def int_repr(self, out=None):
    """Return the raw integer codes of the affine-quantized tensor ``self``.

    Implements both ``aten::int_repr(Tensor self) -> Tensor`` and
    ``aten::int_repr.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)``.
    """
    logger.debug("GEMS_ENFLAME INT_REPR")
    if not isinstance(self, torch.Tensor):
        raise TypeError(
            "int_repr(): argument 'self' must be a Tensor, not "
            f"{type(self).__name__}"
        )
    if not self.is_quantized:
        raise RuntimeError(
            f"int_repr(): expected a quantized tensor, got dtype {self.dtype}"
        )
    out_dtype = _Q_TO_INT.get(self.dtype)
    if out_dtype is None:
        raise RuntimeError(f"int_repr(): unsupported quantized dtype {self.dtype}")

    shape = self.shape
    numel = self.numel()
    stride = self.stride()
    dense = _is_storage_dense(numel, shape, stride)
    in_view = _int_view(self, out_dtype)

    if out is None:
        if dense:
            # Same strides as the input: element offsets from each data_ptr()
            # are then the same linear span, so the copy is one flat pass.
            out = torch.empty_strided(
                shape, stride, dtype=out_dtype, device=self.device
            )
        else:
            out = torch.empty(shape, dtype=out_dtype, device=self.device)
    else:
        if not isinstance(out, torch.Tensor) or out.dtype != out_dtype:
            raise RuntimeError(
                "int_repr(): expected out tensor to have dtype "
                f"{out_dtype}, but got "
                f"{getattr(out, 'dtype', type(out).__name__)} instead"
            )
        if out.shape != shape:
            out.resize_(shape)

    if numel == 0:
        # Never launch a zero grid: one fully masked program is legal and the
        # empty result is already correct.
        _int_repr_flat_kernel[(1,)](in_view, out, 0, BLOCK=_BLOCK)
        return out

    if stride == out.stride():
        base = self.storage_offset()
        misalign = (base * out.element_size()) % _ALIGN_BYTES
        if misalign == 0 or numel < _SPLIT_MIN_ELEMS:
            if not _launch_dense_copy(in_view, out, numel):
                _int_repr_flat_kernel[(triton.cdiv(numel, _BLOCK),)](
                    in_view, out, numel, BLOCK=_BLOCK
                )
        else:
            itemsize = out.element_size()
            head = min(numel, (_ALIGN_BYTES - misalign) // itemsize)
            rest = numel - head
            bulk = rest - rest % _BLOCK
            if head:
                _int_repr_flat_kernel[(1,)](in_view, out, head, BLOCK=_HEAD_BLOCK)
            if bulk:
                # Shifted aliases keep the bulk base 16-byte aligned while the
                # kernel address expression remains base + arange.
                bulk_in = _linear_alias(in_view, out_dtype, head)
                bulk_out = _linear_alias(out, out_dtype, head)
                if not _launch_dense_copy(bulk_in, bulk_out, bulk):
                    _int_repr_flat_kernel[(bulk // _BLOCK,)](
                        bulk_in,
                        bulk_out,
                        bulk,
                        BLOCK=_BLOCK,
                    )
            if rest - bulk:
                _int_repr_flat_kernel[(1,)](
                    _linear_alias(in_view, out_dtype, head + bulk),
                    _linear_alias(out, out_dtype, head + bulk),
                    rest - bulk,
                    BLOCK=_BLOCK,
                )
        return out

    if len(shape) > _MAX_RANK:
        # The strided kernel decomposes a fixed number of dimensions, so a
        # taller input would spill its extra sizes/strides into the kernel's
        # trailing positional arguments and fail before launching.  ATen's
        # integer copy is rank-agnostic and handles every non-dense layout
        # (stepped slices, zero-stride expand, mismatched out strides), so
        # hand the whole copy over to it on plain integer aliases.
        _plain_int_alias(out, out_dtype).copy_(_plain_int_alias(self, out_dtype))
        return out

    sizes = _padded(_MAX_RANK, shape, 1)
    istrides = _padded(_MAX_RANK, stride, 0)
    ostrides = _padded(_MAX_RANK, out.stride(), 0)
    grid = (triton.cdiv(numel, _BLOCK),)
    _int_repr_strided_kernel[grid](
        in_view, out, numel, *sizes, *istrides, *ostrides, BLOCK=_BLOCK
    )
    return out
