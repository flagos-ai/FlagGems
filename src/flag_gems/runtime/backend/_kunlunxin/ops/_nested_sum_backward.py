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

"""Kunlunxin override for ``_nested_sum_backward`` (backward of ``sum(dim)``).

The generic kernel decomposes every flattened output element with two *runtime*
integer divisions (``offset // (dim*inner)`` and ``offset % inner``) and loads a
single element per lane.  On XPU3 that costs 3-10x more than the data movement:

* ``_nested_sum_backward_kernel`` 1368 calls / 43.6 s (89.2 % of the whole
  benchmark probe) with 2-byte lanes and one load per lane;
* the reference expand-copy writes at 1.9-2.2 TB/s on the sizes where the
  reduce dim is not the innermost one, while a 2-byte-lane Triton kernel
  saturates ~0.9 TB/s.

This override keeps the same broadcast semantics but rewrites it as a flat
wide-word copy on the widest integer view:

    out[o, k*inner + i] = grad[o*inner + i]        (k in [0, dim), i in [0, inner))

so the destination is one contiguous stream of 8-byte words whose source index
is ``(w // dim_words) * inner_words + w % inner_words``.  Two more facts drive
the routing (all measured on P800 / KL3, see
``harness/KernelOpt_Knowledge/wiki/kernels/nested-sum-backward-kunlunxin.md``):

* a cyclic source whose period is below ~64 KB collapses the load path
  (32-word period: 205 GB/s; 1024: 650 GB/s; 8192: 1.8 TB/s), so when the
  source row is small the periodic tile is expanded once into a scratch buffer
  (cost <= 25 % of the destination) before the flat copy;
* the reduce dim being innermost makes every output row a constant, so that
  case packs the repeated element into the word itself and needs one scalar
  load per row instead of one per lane.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._nested_sum_backward import (
    _nested_sum_backward as _generic_nested_sum_backward,
)

logger = logging.getLogger(__name__)

# Flat-copy block ladder: >= 64 programs in flight for every size, and the
# 8-byte-word throughput peak at 32768-65536 words per program.
_MAX_BLOCK = 65536
_MIN_BLOCK = 2048
# Cyclic-source period (in 8-byte words) above which the load path is at full
# speed: 8192 words = 64 KB.
_PERIOD_CAP = 8192
# The periodic tile is a scratch buffer; keep it under 1/8 of the destination so
# the extra read traffic cannot dominate.  Measured on (32,256,256,128) d2:
# tile 1024 -> 0.40x, 2048 -> 0.36x, no tile -> 0.13x.
_TILE_RATIO_CAP = 0.125
# Below this the launch of the tile kernel costs more than the copy saves.
_TILE_MIN_WORDS = 1 << 16


@triton.jit
def _copy_flat_kernel(
    src_ptr,
    dst_ptr,
    n_words,
    OUT_WORDS: tl.constexpr,
    SRC_WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """dst[w] = src[(w // OUT_WORDS) * SRC_WORDS + w % SRC_WORDS]."""
    w = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    outer = w // OUT_WORDS
    src = outer * SRC_WORDS + (w % SRC_WORDS)
    if NEED_MASK:
        mask = w < n_words
        tl.store(dst_ptr + w, tl.load(src_ptr + src, mask=mask), mask=mask)
    else:
        tl.store(dst_ptr + w, tl.load(src_ptr + src))


@triton.jit
def _periodic_tile_kernel(
    src_ptr,
    dst_ptr,
    n_words,
    TILE: tl.constexpr,
    ROW: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """dst[w] = src[(w // TILE) * ROW + (w % TILE) % ROW]."""
    w = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    outer = w // TILE
    src = outer * ROW + ((w % TILE) % ROW)
    if NEED_MASK:
        mask = w < n_words
        tl.store(dst_ptr + w, tl.load(src_ptr + src, mask=mask), mask=mask)
    else:
        tl.store(dst_ptr + w, tl.load(src_ptr + src))


@triton.jit
def _bcast_pack4_kernel(
    src_ptr,
    dst_ptr,
    n_words,
    ROW_WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """16-bit rows broadcast into 8-byte words (4 elements per word)."""
    w = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(src_ptr + w // ROW_WORDS)
    lo = value.to(tl.uint32) & 0xFFFF
    word = (lo * 0x00010001).to(tl.uint64) * 0x100000001
    if NEED_MASK:
        mask = w < n_words
        tl.store(dst_ptr + w, word, mask=mask)
    else:
        tl.store(dst_ptr + w, word)


@triton.jit
def _bcast_pack2w64_kernel(
    src_ptr,
    dst_ptr,
    n_words,
    ROW_WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """32-bit rows broadcast into 8-byte words (2 elements per word)."""
    w = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(src_ptr + w // ROW_WORDS)
    word = (value.to(tl.uint64) & 0xFFFFFFFF) * 0x100000001
    if NEED_MASK:
        mask = w < n_words
        tl.store(dst_ptr + w, word, mask=mask)
    else:
        tl.store(dst_ptr + w, word)


@triton.jit
def _bcast_pack2w32_kernel(
    src_ptr,
    dst_ptr,
    n_words,
    ROW_WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """16-bit rows broadcast into 4-byte words (2 elements per word)."""
    w = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(src_ptr + w // ROW_WORDS)
    lo = value.to(tl.uint32) & 0xFFFF
    word = lo * 0x00010001
    if NEED_MASK:
        mask = w < n_words
        tl.store(dst_ptr + w, word, mask=mask)
    else:
        tl.store(dst_ptr + w, word)


def _pick_block(n_words: int) -> int:
    for block in (65536, 32768, 16384, 8192, 4096, 2048):
        if n_words >= block * 64:
            return block
    return _MIN_BLOCK


def _num_warps(block: int) -> int:
    return 32 if block >= 16384 else 8


def _word_plan(grad: torch.Tensor, out: torch.Tensor, need_inner_multiple: int):
    """Pick the widest integer view that keeps every access a whole word.

    Returns ``(view_dtype, words_per_element_group)`` or ``(None, 1)``.  The
    view is metadata only: it reinterprets both tensors, it never moves or
    computes anything.
    """
    elem = grad.element_size()
    if grad.data_ptr() % 8 == 0 and out.data_ptr() % 8 == 0:
        widths = (torch.int64, 8)
    elif grad.data_ptr() % 4 == 0 and out.data_ptr() % 4 == 0:
        widths = (torch.int32, 4)
    else:
        return None, 1
    if widths[1] <= elem:
        return None, 1
    per_word = widths[1] // elem
    if need_inner_multiple % per_word:
        return None, 1
    try:
        grad.view(widths[0])
        out.view(widths[0])
    except RuntimeError:
        return None, 1
    return widths[0], per_word


def _launch_flat_copy(src, dst, n_words, out_words, src_words):
    block = _pick_block(n_words)
    _copy_flat_kernel[(triton.cdiv(n_words, block),)](
        src,
        dst,
        n_words,
        OUT_WORDS=out_words,
        SRC_WORDS=src_words,
        BLOCK=block,
        NEED_MASK=(n_words % block != 0),
        num_warps=_num_warps(block),
    )


def _launch_bcast(grad, out, dim_size, n_words, per_word):
    """Inner size is 1: every output row is one grad element repeated."""
    row_words = dim_size // per_word
    elem = grad.element_size()
    if per_word == 4:
        src = grad.view(torch.int16)
        dst = out.view(torch.int64)
        kernel = _bcast_pack4_kernel
    elif elem == 4:
        src = grad.view(torch.int32)
        dst = out.view(torch.int64)
        kernel = _bcast_pack2w64_kernel
    else:
        src = grad.view(torch.int16)
        dst = out.view(torch.int32)
        kernel = _bcast_pack2w32_kernel
    block = _pick_block(n_words)
    kernel[(triton.cdiv(n_words, block),)](
        src,
        dst,
        n_words,
        ROW_WORDS=row_words,
        BLOCK=block,
        NEED_MASK=(n_words % block != 0),
        num_warps=_num_warps(block),
    )


def _fill_scalar(grad, out, numel):
    """Full reduction: every output element equals the single grad element."""
    grad = grad.contiguous()
    per_word = 1
    view = None
    for width, size in ((torch.int64, 8), (torch.int32, 4)):
        if size <= grad.element_size():
            break
        if numel % (size // grad.element_size()):
            continue
        try:
            grad.view(width)
            out.view(width)
        except RuntimeError:
            continue
        view, per_word = width, size // grad.element_size()
        break
    n_words = numel // per_word
    if view is None:
        block = _pick_block(numel)
        _copy_flat_kernel[(triton.cdiv(numel, block),)](
            grad,
            out,
            numel,
            OUT_WORDS=numel,
            SRC_WORDS=1,
            BLOCK=block,
            NEED_MASK=(numel % block != 0),
            num_warps=_num_warps(block),
        )
        return out
    _launch_bcast(grad, out, numel, n_words, per_word)
    return out


def _nested_sum_backward(grad, self, dim=None, keepdim=False):
    """Backward of ``sum(dim)``: broadcast ``grad`` back to ``self.shape``.

    Mirrors ``flag_gems.ops._nested_sum_backward`` semantics; only the kernel
    that materialises the broadcast is rewritten.
    """
    logger.debug("GEMS_KUNLUNXIN _NESTED_SUM_BACKWARD")

    shape = list(self.shape)
    ndim = self.dim()

    if dim is None:
        out = torch.empty(shape, dtype=grad.dtype, device=grad.device)
        if out.numel() == 0:
            return out
        return _fill_scalar(grad, out, out.numel())

    if isinstance(dim, (list, tuple)):
        d = int(dim[0])
    else:
        d = int(dim)
    if d < 0:
        d += ndim

    outer_size = 1
    for s in shape[:d]:
        outer_size *= s
    dim_size = shape[d]
    inner_size = 1
    for s in shape[d + 1 :]:
        inner_size *= s

    if not grad.is_contiguous():
        grad = grad.contiguous()
    out = torch.empty(shape, dtype=grad.dtype, device=grad.device)
    numel = out.numel()
    if numel == 0:
        return out

    if grad.numel() != outer_size * inner_size:
        # Unexpected grad layout (e.g. an already reduced keepdim shape we do
        # not model): keep the generic broadcast kernel.
        return _generic_nested_sum_backward(grad, self, [d], keepdim)

    if inner_size == 1:
        # Reduced dim is innermost: every output row is one grad element
        # repeated, so pack that element into the destination word and pay a
        # single scalar load per row instead of one load per lane.
        elem = grad.element_size()
        aligned8 = grad.data_ptr() % 8 == 0 and out.data_ptr() % 8 == 0
        aligned4 = grad.data_ptr() % 4 == 0 and out.data_ptr() % 4 == 0
        plan = None
        if aligned8 and elem == 2 and dim_size % 4 == 0:
            try:
                grad.view(torch.int16)
                out.view(torch.int64)
                plan = 4
            except RuntimeError:
                plan = None
        if plan is None and aligned8 and elem == 4 and dim_size % 2 == 0:
            try:
                grad.view(torch.int32)
                out.view(torch.int64)
                plan = 2
            except RuntimeError:
                plan = None
        if plan is None and aligned4 and elem == 2 and dim_size % 2 == 0:
            try:
                grad.view(torch.int16)
                out.view(torch.int32)
                plan = 2
            except RuntimeError:
                plan = None
        if plan is not None:
            _launch_bcast(grad, out, dim_size, numel // plan, plan)
            return out
        _launch_flat_copy(grad, out, numel, dim_size, 1)
        return out

    view_dtype, per_word = _word_plan(grad, out, inner_size)
    if view_dtype is not None:
        iw = inner_size // per_word
        dim_words = dim_size * iw
        n_words = numel // per_word
        src = grad.view(view_dtype)
        dst = out.view(view_dtype)
        tile_budget = min(_PERIOD_CAP, int(_TILE_RATIO_CAP * n_words / outer_size))
        tile = (tile_budget // iw) * iw if iw else 0
        if (
            tile >= 2 * iw
            and dim_words >= 2 * tile
            and n_words >= _TILE_MIN_WORDS
        ):
            # Expand the small cyclic source once into a period >= 2*inner tile
            # so the flat copy reads a wide, prefetch-friendly window.
            tile_words = outer_size * tile
            # scratch buffer only: pure allocation, filled by the kernel below
            scratch = torch.empty(tile_words, dtype=view_dtype, device=grad.device)
            block = _pick_block(tile_words)
            _periodic_tile_kernel[(triton.cdiv(tile_words, block),)](
                src,
                scratch,
                tile_words,
                TILE=tile,
                ROW=iw,
                BLOCK=block,
                NEED_MASK=(tile_words % block != 0),
                num_warps=_num_warps(block),
            )
            _launch_flat_copy(scratch, dst, n_words, dim_words, tile)
            return out
        _launch_flat_copy(src, dst, n_words, dim_words, iw)
        return out

    _launch_flat_copy(grad, out, numel, dim_size * inner_size, inner_size)
    return out
