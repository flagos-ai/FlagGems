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

"""Triton implementation of ``aten::_fw_primal_copy``.

``_fw_primal_copy(self, level)`` is the copy-out counterpart of
``aten::_fw_primal``: it returns a fresh tensor holding the same values as
``self`` (the ``level`` tag only selects the autograd internals of the native
op and carries no data here).  The ``.out`` overload writes into the caller
buffer and returns that object.

GCU400 note (2026-11-17): this backend hard-caps ``grid.x`` at 65535 programs
(``OutOfResources: out of resource: grid.x``), so the core-benchmark shapes
(536870912 flat / 1024*1024*512 elements) overflow the old one-CTA-per-tile
launch (cdiv(2**29, 8192) == 65536).  The flat path is now a grid-stride
streaming copy over int32/uint8 word views, with the grid capped well below
the hardware limit (one program per SIP saturates the DRAM copy wall).
Word views keep the kernel dtype-agnostic (fp8/int64/complex included: the
move is bit-exact on raw bytes) and away from Triton dtype-mapping gaps.
The per-call ``logger.debug`` probe is dropped per the ``_fw_primal`` /
alias no-trace host-op precedent (diagnosability lives in the docstring).

Recalibration note (2026-11-20, standard-bandwidth GCU400): the launch
ladder below was re-swept against the same core shapes, because the earlier
numbers came from a high-bandwidth board and did not transfer.  On the
standard band the flat copy roof is ~2.17 TB/s (native ``copy_`` on 1 GiB:
1006 us), one program per SIP (grid 24, num_warps 2) tracks it, while the
previous 48-program/warp-4 launch lost 3% and the 8 KiB mid band lost 23%
(44.8 us vs 36.5 us on the 16.7M-element tier).  Host side, the two
``reshape(-1)`` calls in front of the dtype views are gone: under the
``use_gems`` dispatch mode every ATen call inside the op pays a dispatcher
round-trip, and dropping the flatten halves that cost.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Streaming tile: 32768 int32 words (128 KiB) over one program per SIP.
# do_bench(median) sweep on the standard-band GCU400 (2026-11-20), BLOCK
# {4096..131072} x grid {24,48,96,192,384} x num_warps {2,4,8}:
#   1 GiB fp16   best 987.6 us  (B32768/G24/W2, 2.17 TB/s = the copy roof)
#   16.7M fp16   best  36.4 us  (B32768-65536/G24/W2-8, all within 0.2%)
#   16.7M fp32   best  66.9 us  (B32768/G24/W2)
# so a single B32768/G24/W2 point is within noise of every per-shape optimum,
# and 24 programs (not 48) is what the DRAM mix wants.
_STREAM_BLOCK = 32768
_STREAM_GRID = 24
_STREAM_WARPS = 2
# Below this many words the copy is launch+ramp bound and a smaller tile wins
# by spreading the work over more programs (measured only to be no worse:
# the core shapes do not sample this band, so it keeps the previous 8 KiB
# tile rather than extrapolating the streaming point downward).
_MID_BLOCK = 8192
_MID_MAX_WORDS = 1 << 16
# Single-CTA exact-tile band: tiny copies sit on the ~2.4us launch floor, so
# one program covering the whole tensor (ones_like precedent) avoids the
# grid-stride loop and idle lanes.
_TINY_MAX = 8192
_TINY_BLOCK = 2048
_TINY_WARPS = 2
# int32 offset domain: the last tile must stay below 2**31 words.
_MAX_I32 = 2**31 - 1
_MAXD = 8


@libentry()
@triton.jit
def _copy_flat_kernel(
    src_ptr,
    dst_ptr,
    n_words,
    BLOCK: tl.constexpr,
    WIDE: tl.constexpr,
):
    """Grid-stride flat copy over int32 (or uint8) words.

    Plain offset loads/stores stay on the synchronous port and are masked
    exactly; block-pointer (DTE) transfers round the tail DMA length up and
    can fault the MMU on page-boundary allocations (copy_ precedent), so
    they are deliberately not used here.
    """
    pid = tl.program_id(axis=0)
    num_ctas = tl.num_programs(axis=0)
    n_tiles = tl.cdiv(n_words, BLOCK)
    for tile in range(pid, n_tiles, num_ctas):
        if WIDE:
            # Beyond 2**31 words the tile offset leaves the int32 domain.
            offs = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        else:
            offs = tile * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_words
        vals = tl.load(src_ptr + offs, mask=mask)
        tl.store(dst_ptr + offs, vals, mask=mask)


@triton.jit
def _copy_strided_kernel(
    src_ptr,
    dst_ptr,
    sizes_ptr,
    strides_ptr,
    dst_strides_ptr,
    ndim,
    numel,
    BLOCK: tl.constexpr,
    MAXD: tl.constexpr,
    DST_STRIDED: tl.constexpr,
):
    # Gather into (or scatter out of) a row-major destination. The flat logical
    # index is row-major: dim 0 is the MOST significant digit, dim ndim-1 the
    # least.  Decompose from the last dim down so dst position i receives the
    # source element at the row-major offset of i.  Grid-stride keeps huge views
    # under the 65535 grid.x cap; offsets stay int64 for the gather math.
    #
    # ``DST_STRIDED`` addresses the destination through its own strides instead
    # of assuming a contiguous buffer: the ``.out`` overload must write the
    # caller's buffer in place, and materialising it with ``.contiguous()``
    # first is both an extra pass and unsupported for GCU views.  The shape is
    # shared, so only the stride vector differs.
    pid = tl.program_id(axis=0)
    num_ctas = tl.num_programs(axis=0)
    n_tiles = tl.cdiv(numel, BLOCK)
    for tile in range(pid, n_tiles, num_ctas):
        offs = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        rem = offs
        src_off = tl.zeros([BLOCK], dtype=tl.int64)
        for d in tl.static_range(MAXD):
            dd = ndim - 1 - d
            active = dd >= 0
            dd_safe = tl.maximum(dd, 0)
            size_d = tl.load(sizes_ptr + dd_safe)
            stride_d = tl.load(strides_ptr + dd_safe)
            size_safe = tl.where(active, size_d, 1)
            stride_safe = tl.where(active, stride_d, 0)
            idx_d = rem % size_safe
            rem = rem // size_safe
            src_off += idx_d.to(tl.int64) * stride_safe
        vals = tl.load(src_ptr + src_off, mask=mask)
        if DST_STRIDED:
            dst_rem = offs
            dst_off = tl.zeros([BLOCK], dtype=tl.int64)
            for d in tl.static_range(MAXD):
                dd = ndim - 1 - d
                active = dd >= 0
                dd_safe = tl.maximum(dd, 0)
                size_d = tl.load(sizes_ptr + dd_safe)
                stride_d = tl.load(dst_strides_ptr + dd_safe)
                size_safe = tl.where(active, size_d, 1)
                stride_safe = tl.where(active, stride_d, 0)
                idx_d = dst_rem % size_safe
                dst_rem = dst_rem // size_safe
                dst_off += idx_d.to(tl.int64) * stride_safe
            tl.store(dst_ptr + dst_off, vals, mask=mask)
        else:
            tl.store(dst_ptr + offs, vals, mask=mask)


def _unravel(linear: int, shape) -> tuple:
    """Row-major multi-index of ``linear`` for ``shape``."""
    index = []
    for extent in reversed(shape):
        index.append(linear % extent)
        linear //= extent
    return tuple(reversed(index))


def _launch_strided_copy(src: torch.Tensor, dst: torch.Tensor, dst_strided: bool):
    """One strided-kernel launch for a rank-``<= _MAXD`` view pair.

    ``src`` is read through its own strides and ``dst`` is either addressed
    flat (``dst_strided=False``, contiguous buffer) or through its strides.
    """
    numel = src.numel()
    sizes = torch.tensor(list(src.shape), dtype=torch.int64, device=src.device)
    strides = torch.tensor(list(src.stride()), dtype=torch.int64, device=src.device)
    dst_strides = (
        torch.tensor(list(dst.stride()), dtype=torch.int64, device=src.device)
        if dst_strided
        else strides
    )
    grid = (min(_STREAM_GRID, triton.cdiv(numel, _STREAM_BLOCK)),)
    _copy_strided_kernel[grid](
        src,
        dst,
        sizes,
        strides,
        dst_strides,
        src.dim(),
        numel,
        BLOCK=_STREAM_BLOCK,
        MAXD=_MAXD,
        DST_STRIDED=dst_strided,
    )


def _copy_strided_into(dst: torch.Tensor, src: torch.Tensor):
    """Copy the (possibly non-contiguous) ``src`` into ``dst``, same shape.

    The kernel decomposes at most ``_MAXD`` trailing dims, so ranks above it
    are split host-side: the leading ``dim - _MAXD`` dims are peeled with basic
    indexing and each remaining ``_MAXD``-rank sub-view is copied by the same
    Triton kernel.  Every element therefore still moves on the device, and the
    destination is written in place -- which is what the ``.out`` overload
    requires, since materialising it with ``.contiguous()`` would both add a
    pass and leave the caller's buffer untouched.
    """
    dst_strided = not dst.is_contiguous()
    dim = src.dim()
    if dim <= _MAXD:
        if src.numel():
            _launch_strided_copy(src, dst, dst_strided)
        return
    lead_shape = src.shape[: dim - _MAXD]
    outer = 1
    for extent in lead_shape:
        outer *= extent
    for linear in range(outer):
        index = _unravel(linear, lead_shape)
        sub_src = src[index]
        sub_dst = dst[index]
        if sub_src.numel():
            _launch_strided_copy(sub_src, sub_dst, not sub_dst.is_contiguous())


_WORD_ITEMSIZE = {torch.int32: 4, torch.uint8: 1}


def _word_view(tensor: torch.Tensor, word_dtype: torch.dtype) -> torch.Tensor:
    """Zero-copy ``int32`` / ``uint8`` view of a contiguous tensor.

    The kernel only needs the base pointer and the word count, so the word
    view is taken straight off the tensor when possible: the ``reshape(-1)``
    that used to precede it is one more ATen call, and under the ``use_gems``
    dispatch mode every ATen call inside the op pays a dispatcher round-trip.

    ``view(dtype)`` additionally requires the trailing dimension's byte size
    to be a multiple of the word size, so a 2-byte dtype whose last dim is odd
    (``(20, 320, 15)`` fp16 -> 30 bytes) has to be flattened first.  On a
    contiguous tensor that ``reshape(-1)`` is still a view, never a copy, and
    the flat path keeps the fast branch for the shapes that matter.  0-dim
    tensors cannot be dtype-viewed either, so rank 0 is promoted first.
    """
    flat = tensor if tensor.dim() else tensor.reshape(1)
    if (flat.element_size() * flat.size(-1)) % _WORD_ITEMSIZE[word_dtype]:
        flat = flat.reshape(-1)
    return flat.view(word_dtype)


def _launch_flat_copy(dst: torch.Tensor, src: torch.Tensor, numel: int):
    """Copy ``numel`` elements of contiguous same-dtype ``src`` into ``dst``.

    Both operands are reinterpreted as int32 words when byte-aligned (128
    bits per lane), else as raw uint8 bytes; the move is bit-exact either
    way.  A WIDE arm guards the int32 offset domain past 2**31 words.
    """
    es = src.element_size()
    nbytes = numel * es
    if (
        (nbytes & 3) == 0
        and (src.storage_offset() * es) & 3 == 0
        and (dst.storage_offset() * es) & 3 == 0
    ):
        n_words = nbytes >> 2
        src_w = _word_view(src, torch.int32)
        dst_w = _word_view(dst, torch.int32)
    else:
        n_words = nbytes
        src_w = _word_view(src, torch.uint8)
        dst_w = _word_view(dst, torch.uint8)
    if n_words <= _TINY_MAX:
        block = max(_TINY_BLOCK, triton.next_power_of_2(n_words))
        warps = _TINY_WARPS
    elif n_words <= _MID_MAX_WORDS:
        block = _MID_BLOCK
        warps = _STREAM_WARPS
    else:
        block = _STREAM_BLOCK
        warps = _STREAM_WARPS
    grid = (min(_STREAM_GRID, triton.cdiv(n_words, block)),)
    _copy_flat_kernel[grid](
        src_w,
        dst_w,
        n_words,
        BLOCK=block,
        WIDE=n_words > _MAX_I32 - block,
        num_warps=warps,
    )


def _fw_primal_copy(self: torch.Tensor, level: int) -> torch.Tensor:
    """``aten::_fw_primal_copy``: fresh copy of ``self`` (``level`` unused)."""
    logger.debug("GEMS_ENFLAME _FW_PRIMAL_COPY")
    numel = self.numel()
    if self.is_contiguous():
        out = torch.empty_like(self)
        if numel:
            _launch_flat_copy(out, self, numel)
        return out
    # Non-contiguous: materialize a fresh CONTIGUOUS copy.
    out = torch.empty_like(self, memory_format=torch.contiguous_format)
    if numel == 0:
        return out
    _copy_strided_into(out, self)
    return out


def _fw_primal_copy_out(
    self: torch.Tensor, level: int, *, out: torch.Tensor
) -> torch.Tensor:
    """``aten::_fw_primal_copy.out``: copy ``self`` into ``out``."""
    logger.debug("GEMS_ENFLAME _FW_PRIMAL_COPY_OUT")
    if out.dtype != self.dtype:
        raise RuntimeError("_fw_primal_copy.out: dtype of input and output must match.")
    if out.device != self.device:
        raise RuntimeError(
            "_fw_primal_copy.out: input and output must be on the same device."
        )
    if out.shape != self.shape:
        out.resize_(self.shape)
    numel = self.numel()
    if numel == 0:
        return out
    if self.is_contiguous() and out.is_contiguous():
        _launch_flat_copy(out, self, numel)
        return out
    # Either side may be strided; the kernel reads ``self`` through its own
    # strides and writes ``out`` in place through its own, so no temporary
    # contiguous buffer is materialised here.
    _copy_strided_into(out, self)
    return out
