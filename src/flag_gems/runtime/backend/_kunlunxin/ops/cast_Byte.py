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
"""Kunlunxin override for ``aten::_cast_Byte`` (element cast to ``uint8``).

Three XPU3-specific problems in the generic implementation are fixed here:

1. **Float narrowing must go through ``int32``.**  The generic kernel narrows
   floating inputs through ``int64`` ("preserving wraparound"), so ``2**31``
   casts to ``0``.  XPU3/ATen narrows through a *saturating* 32-bit conversion
   instead: ``2**31 -> INT32_MAX -> 0xFF`` and ``NaN -> 0xFF`` while ``256.5``
   still wraps to ``0``.  In-kernel ``x.to(tl.int32).to(tl.uint8)`` reproduces
   the ATen bit-exactly for fp16/bf16/fp32/fp64 (measured).
2. **Output layout must follow ATen's ``empty_like(preserve_format)`` packing.**
   XPU's ``torch.empty_like`` returns a *contiguous* buffer for a non-dense
   input such as ``x.transpose(0, 3)[::2, :, 1::2, :]``, while ATen packs the
   result in the source's stride order (``(1, 12, 4, 60)`` there).  The
   ``actual.stride() == expected.stride()`` contract therefore failed.  This
   file derives the packed strides itself and allocates accordingly.
3. **Throughput.**  The generic kernel uses a fixed ``BLOCK=1024`` with an
   always-on mask: 16.7M fp16 elements take 1.95 ms (torch 0.19 ms).  A bounded
   flat tile with a ``NEED_MASK`` constexpr takes 0.165 ms.  Strided inputs
   whose source offsets are affine in the flat output index (unit stride,
   ``x[:, ::2]``, transposed dense views) skip the per-element ``%``/``//``
   decomposition, and a plain tuple grid avoids the per-call grid-closure
   launch stall.

The residual strided gather (non-affine layouts only) keeps the generic
``%``/``//`` decomposition; those layouts are not in the benchmark matrix.

One deliberate deviation: a ``torch._neg_view`` input is read as its raw
storage.  ATen's ``_cast_Byte`` on this backend ignores the neg bit (measured:
``_cast_Byte(_neg_view(arange(17)))`` returns ``0..16``, and even ``tolist``
shows the positive values), so mirroring the raw-storage read is what keeps
Gems bit-identical to the reference; materialising the sign would differ from
ATen here.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_MIN_BLOCK = 256
_FLAT_MAX_BLOCK = 16384
_AFFINE_MAX_BLOCK = 4096
_GATHER_BLOCK = 2048
_INT32_LIMIT = 1 << 31

_SUPPORTED_DTYPES = (
    torch.bool,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)


@triton.jit
def _cast_byte_value(x):
    """Narrow one element to uint8 with ATen/CPU bit-exact semantics."""
    if x.dtype == tl.bfloat16:
        # bf16 loads are miscompiled in some mixed-width shapes on this backend;
        # widen first (measured identical results, no measurable cost).
        x = x.to(tl.float32)
    if x.dtype.is_floating():
        # Saturating 32-bit conversion, then byte truncation: matches
        # ``static_cast<uint8_t>`` on the reference (2**31 -> 0xFF, NaN -> 0xFF,
        # 256.5 -> 0x00).
        x = x.to(tl.int32)
    return x.to(tl.uint8)


@triton.jit
def _cast_byte_flat_kernel(
    X,
    Y,
    N,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
    WIDE: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if WIDE:
        offs = offs.to(tl.int64)
    if NEED_MASK:
        mask = offs < N
        x = tl.load(X + offs, mask=mask, other=0)
        tl.store(Y + offs, _cast_byte_value(x), mask=mask)
    else:
        x = tl.load(X + offs)
        tl.store(Y + offs, _cast_byte_value(x))


@triton.jit
def _cast_byte_affine_kernel(
    X,
    Y,
    N,
    STEP: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
    WIDE: tl.constexpr,
):
    """Source offset is ``flat_index * STEP`` (no div/mod)."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if WIDE:
        offs = offs.to(tl.int64)
    src = offs * STEP
    if NEED_MASK:
        mask = offs < N
        x = tl.load(X + src, mask=mask, other=0)
        tl.store(Y + offs, _cast_byte_value(x), mask=mask)
    else:
        x = tl.load(X + src)
        tl.store(Y + offs, _cast_byte_value(x))


@triton.jit
def _cast_byte_gather_kernel(
    X,
    Y,
    N,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
    WIDE: tl.constexpr,
):
    """General strided source; the output is packed in the same dim order."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if WIDE:
        offs = offs.to(tl.int64)
    rem = offs
    if WIDE:
        src = tl.zeros((BLOCK,), tl.int64)
    else:
        src = tl.zeros((BLOCK,), tl.int32)
    for dim in tl.static_range(len(SHAPE) - 1, -1, -1):
        src += (rem % SHAPE[dim]) * STRIDES[dim]
        rem //= SHAPE[dim]
    if NEED_MASK:
        mask = offs < N
        x = tl.load(X + src, mask=mask, other=0)
        tl.store(Y + offs, _cast_byte_value(x), mask=mask)
    else:
        x = tl.load(X + src)
        tl.store(Y + offs, _cast_byte_value(x))


def _packed_layout(shape, strides):
    """Packed output layout + dim order (inner first), like ``empty_like``.

    ATen's ``empty_like(preserve_format)`` keeps the source strides for a
    non-overlapping dense tensor and otherwise emits a buffer packed in the
    source's memory order.  Sorting the dims by ascending stride (a 0 stride,
    i.e. an expanded dim, is the least significant one) and packing them in
    that order reproduces it: for the transpose/slice/expand/channels-last
    layouts measured on XPU the result is stride-identical to ATen, including
    the non-dense ``permuted_slice`` case where XPU ``empty_like`` deviates.
    """
    ndim = len(shape)
    # Ties (zero-size / size-1 dims carry arbitrary equal strides) are broken by
    # descending dim index so the result matches ATen's last-dim-first packing.
    order = sorted(
        range(ndim),
        key=lambda d: (strides[d] if strides[d] > 0 else float("inf"), -d),
    )
    out = [0] * ndim
    run = 1
    for d in order:
        out[d] = run
        # ATen clamps zero-size dims to 1 when packing contiguous strides, e.g.
        # ``empty(2, 0, 3)`` has strides ``(3, 3, 1)``, not ``(3, 0, 1)``.
        run *= max(shape[d], 1)
    return tuple(out), order


def _pick_block(n, cap):
    """One fat program for small n, ``cap`` lanes for large n.

    XPU3 charges a large fixed cost per workgroup (measured with
    ``do_bench``): casting 1024 fp16 elements takes 14.1 us as 4 x 256-lane
    programs but 5.7 us as 1 x 1024 lanes, and 65 536 elements take 19.7 us as
    64 x 1024 lanes but 6.4 us as 4 x 16384 lanes.  So the tile grows with n
    instead of splitting n over a fixed program count.
    """
    if n <= 0:
        return _MIN_BLOCK
    return max(_MIN_BLOCK, min(cap, triton.next_power_of_2(n)))


def _affine_step(shape, strides, order):
    """Return the source step if ``src == flat * step``, else None."""
    if not order:
        return 1
    step = strides[order[0]]
    if step <= 0:
        return None
    for inner, outer in zip(order, order[1:]):
        if strides[inner] <= 0:
            return None
        if strides[outer] != strides[inner] * shape[inner]:
            return None
    return step


def _flat_cast(src, n):
    """Cast when the source and destination share one flat stream.

    Allocation only (``empty_like`` copies the strides of a dense source, so the
    output is the same flat layout); the values are produced by the kernel.
    """
    out = torch.empty_like(src, dtype=torch.uint8)
    if n == 0:
        return out
    block = _pick_block(n, _FLAT_MAX_BLOCK)
    _cast_byte_flat_kernel[(triton.cdiv(n, block),)](
        src,
        out,
        n,
        BLOCK=block,
        NEED_MASK=(n % block) != 0,
        WIDE=n >= _INT32_LIMIT,
    )
    return out


def cast_Byte(self, non_blocking=False):
    logger.debug("GEMS CAST_BYTE (kunlunxin)")
    if self.dtype == torch.uint8:
        return self
    if self.device.type == "cpu" or self.layout != torch.strided:
        raise NotImplementedError(
            f"kunlunxin cast_Byte does not support device={self.device} "
            f"layout={self.layout}"
        )
    if self.is_quantized or self.dtype not in _SUPPORTED_DTYPES:
        raise NotImplementedError(
            f"kunlunxin cast_Byte does not support dtype={self.dtype}"
        )

    n = self.numel()
    if self.is_contiguous():
        # Most common case, kept branch-light: the per-call host cost is what
        # dominates the small-shape benchmark cases.
        return _flat_cast(self, n)

    shape = tuple(self.shape)
    strides = tuple(self.stride())
    out_strides, order = _packed_layout(shape, strides)
    if out_strides == strides:
        # Non-overlapping dense (e.g. a transposed view): the packed layout is
        # the source layout, so the cast is still one flat pass.
        return _flat_cast(self, n)
    # Allocation only (no value computation): a buffer with the packed layout
    # ATen returns for a non-dense source.
    out = torch.empty_strided(
        shape, out_strides, dtype=torch.uint8, device=self.device
    )
    if n == 0:
        return out

    # A unit step with a packed layout that differs from the source strides can
    # only happen for degenerate size-1 dims, and a flat pass would then be
    # unsound; require ``step > 1`` and let anything else use the gather.
    step = _affine_step(shape, strides, order)
    if step is not None and step > 1:
        wide = (n - 1) * step >= _INT32_LIMIT
        # A stride-2 gather prefers a narrower tile (measured 4.6x faster at
        # 4096 than at 16384 for the 1024x512 ``[:, ::2]`` case).
        block = _pick_block(n, _AFFINE_MAX_BLOCK)
        _cast_byte_affine_kernel[(triton.cdiv(n, block),)](
            self,
            out,
            n,
            STEP=step,
            BLOCK=block,
            NEED_MASK=(n % block) != 0,
            WIDE=wide,
        )
        return out

    dims = list(reversed(order))
    g_shape = tuple(shape[d] for d in dims)
    g_strides = tuple(strides[d] for d in dims)
    max_src = sum((shape[d] - 1) * max(strides[d], 0) for d in dims)
    wide = max(max_src, n - 1) >= _INT32_LIMIT
    block = _pick_block(n, _GATHER_BLOCK)
    _cast_byte_gather_kernel[(triton.cdiv(n, block),)](
        self,
        out,
        n,
        g_shape,
        g_strides,
        BLOCK=block,
        NEED_MASK=(n % block) != 0,
        WIDE=wide,
    )
    return out
