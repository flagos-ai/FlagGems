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

"""Kunlunxin (XPU) specialization of ``aten::flipud``.

``torch.flipud`` reverses the first dimension; the whole tensor is a
concatenation of ``H`` contiguous rows (``row = prod(shape[1:])``) emitted in
reverse row order.  The generic implementation flattens the task space and
recovers the row of every element with ``offset // row_size`` where
``row_size`` is a *runtime* scalar.  On TritonXPU that per-element soft integer
division plus the always-on mask de-vectorises the access; measured the
generic kernel runs at ~65 GB/s for every large shape (2.06 ms for a 16.7 M
element fp32 tensor, i.e. ~7x slower than ``torch.flipud``).

Two independent problems are fixed here:

1. **Stride contract.**  ``torch.flipud`` returns a *contiguous* result on this
   stack (the CPU reference is materialised by the device-to-host copy), while
   the generic implementation returns ``torch.empty_like(self)``, which keeps
   the input's (transposed / channels-last) strides.  ``tests/test_flipud.py``
   asserts ``result.stride() == expected.stride()``, so the generic code fails
   8 cases (transposed, channels-last, complex).  This file always allocates a
   contiguous output.
2. **Throughput.**  Instead of a flat gather, each program copies one
   power-of-two *word* chunk that lies entirely inside a single row; the source
   row index is ``rows - 1 - row`` and the chunk offset is identical on both
   sides, so load and store are both contiguous block DMA with **no mask**.
   The row base is a per-program scalar (one shift / and on the program id), so
   no per-lane division survives.  Same-dtype data is bit-cast to the widest
   integer word (2-byte elements -> int32, 4-byte -> int64) because 2-byte
   lanes pay a ~3x DMA penalty on this backend.

Fallbacks: non-contiguous inputs, a trailing run shorter than one vector store
(64 lanes), or a ``row_size`` that has no power-of-two divisor >= 64 keep the
plain reversed gather (correct, but the structural negative-stride wall).
``row_size == 1`` (1-D input) is exactly that wall: the flip is along the
innermost axis, for which this backend has no vectorised negative-stride load.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger("flag_gems.ops." + __name__.rsplit(".", 1)[-1])

# Shortest row (in words) that still amortises the per-program cost: 1-D tiles
# narrower than 256 lanes sit on the "narrow tile" cliff on XPU3 (~15 us for
# <=128 lanes vs 3.2-3.5 us at 256), so a short row is handed to the plain
# gather instead of one tiny DMA run per row.
_RUN_MIN_WORDS = 256
# Cap on the per-program chunk (in words).  128 KB per program measured best on
# the flat-copy family; larger chunks start to cost local memory on XPU3.
_RUN_MAX_BLOCK = 16384
# A DMA run shorter than this does not amortise the per-run cost on XPU3:
# measured on (512, 512) fp16, 1 KB runs cost 0.100 ms while 2 KB runs cost
# 0.030 ms and the flat gather 0.019 ms.
_RUN_MIN_CHUNK_BYTES = 2048
# Below this size the whole copy sits on the ~15 us launch floor and one flat
# gather launch beats one DMA descriptor per row even for wide rows: measured
# (512, 512) fp32 1 MB -> run copy 0.030 ms vs gather 0.019 ms, while at 8 MB
# and above the run copy is 2-10x faster.
_GATHER_MAX_BYTES = 2 << 20
# Candidate chunk widths, largest first; the first exact divisor of the run
# length is used, so the no-mask path is always taken.
_RUN_BLOCKS = (16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64)
# Gather tile for the fallback path (same width as the generic implementation)
# and the exact-divisor ladder tried before falling back to a masked tail.
_GATHER_BLOCK = 1024
_GATHER_BLOCKS = (1024, 512, 256, 128, 64)


@triton.jit
def _flipud_run_kernel(
    src_ptr,
    dst_ptr,
    run,
    rows,
    CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
    WIDE: tl.constexpr,
):
    """Copy ``rows`` contiguous runs of ``run`` words, reversing the run order.

    Burst ``(row, chunk)`` is identified by the flat program id; both the source
    and the destination address are ``base + chunk * BLOCK + arange(BLOCK)``, so
    each memory op is a single contiguous run and no mask is needed.
    """
    pid = tl.program_id(0)
    row = pid // CHUNKS
    chunk = pid % CHUNKS
    src_row = rows - 1 - row
    if WIDE:
        src_base = src_row.to(tl.int64) * run
        dst_base = row.to(tl.int64) * run
    else:
        src_base = src_row * run
        dst_base = row * run
    offsets = chunk * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(src_ptr + src_base + offsets)
    tl.store(dst_ptr + dst_base + offsets, values)


@triton.jit
def _flipud_gather_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    SHAPE: tl.constexpr,
    SSTRIDES: tl.constexpr,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Reversed gather into a contiguous destination (any input layout).

    The output is contiguous, so its index is simply the flat offset; only the
    source index has to be decomposed, with dim 0 reversed.

    ``NEED_MASK`` is false when ``BLOCK`` divides ``n_elements`` (the launcher
    picks such a tile whenever one exists): every lane is then in range by
    construction, no mask is emitted and no clamp is needed.

    With a tail, the task index is clamped and the load is *unmasked*: a masked
    gather is mis-lowered on XPU3 whenever its address stream is not a
    contiguous prefix (measured on this operator: ``x[:, 3:37]`` and
    ``x[::3, 1:]`` read garbage in 26/306 and 51/889 elements; the generic
    implementation has the same defect).  Clamping keeps every source address
    legal; only the store keeps a mask.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        linear = tl.minimum(offsets, n_elements - 1).to(tl.int64)
    else:
        linear = offsets.to(tl.int64)
    src_offsets = tl.zeros((BLOCK,), dtype=tl.int64)
    for dim in tl.static_range(NDIM - 1, -1, -1):
        index = linear % SHAPE[dim]
        linear //= SHAPE[dim]
        src_index = SHAPE[dim] - 1 - index if dim == 0 else index
        src_offsets += src_index * SSTRIDES[dim]
    values = tl.load(src_ptr + src_offsets)
    if NEED_MASK:
        tl.store(dst_ptr + offsets, values, mask=offsets < n_elements)
    else:
        tl.store(dst_ptr + offsets, values)


def _word_view(row_size: int, esize: int, src: torch.Tensor, dst: torch.Tensor):
    """Widest integer word that keeps every row a whole number of words.

    Bit-casting a same-dtype copy to a wider integer view is exact and buys the
    4/8-byte DMA width on this backend (2-byte lanes are ~3x slower per lane).
    A word is only taken if the resulting row is still at least
    ``_RUN_MIN_WORDS`` lanes wide; otherwise ``per_word == 0`` tells the caller
    to use the gather (a 128-lane row tile is 3x slower than the gather).
    """
    for wdt, wsize in ((torch.int64, 8), (torch.int32, 4)):
        per_word = wsize // esize
        if (
            esize < wsize
            and (row_size * esize) % wsize == 0
            and src.data_ptr() % wsize == 0
            and dst.data_ptr() % wsize == 0
            and row_size // per_word >= _RUN_MIN_WORDS
        ):
            return wdt, per_word
    if esize >= 8 and row_size >= _RUN_MIN_WORDS:
        return None, 1
    return None, 0


def _pick_run_block(run_words: int):
    for block in _RUN_BLOCKS:
        if block > _RUN_MAX_BLOCK:
            continue
        if run_words % block == 0:
            return block
    return None


def _run_copy(src: torch.Tensor, dst: torch.Tensor, rows: int, row_size: int) -> bool:
    """Reversed-row block copy; returns False when no legal chunk width exists."""
    wdt, per_word = _word_view(row_size, src.element_size(), src, dst)
    if per_word == 0:
        return False
    if wdt is not None:
        src_flat = src.reshape(-1).view(wdt)
        dst_flat = dst.reshape(-1).view(wdt)
        run = row_size // per_word
    else:
        src_flat = src.reshape(-1)
        dst_flat = dst.reshape(-1)
        run = row_size
    block = _pick_run_block(run)
    if block is None:
        return False
    if block * src.element_size() * per_word < _RUN_MIN_CHUNK_BYTES:
        return False
    chunks = run // block
    grid = (rows * chunks,)
    _flipud_run_kernel[grid](
        src_flat,
        dst_flat,
        run,
        rows,
        CHUNKS=chunks,
        BLOCK=block,
        WIDE=(rows * run >= 2**31),
    )
    return True


def _gather_copy(src: torch.Tensor, dst: torch.Tensor) -> None:
    n_elements = src.numel()
    # Prefer a tile that divides the task space exactly (no mask, no clamp:
    # ~2x faster than the tail form), starting at the measured 1024-lane sweet
    # spot.  A few-element tensor keeps the launch floor either way.
    block = None
    for candidate in _GATHER_BLOCKS:
        if n_elements % candidate == 0:
            block = candidate
            break
    need_mask = block is None
    if need_mask:
        block = _GATHER_BLOCK
        if n_elements <= _GATHER_BLOCK:
            block = max(64, triton.next_power_of_2(n_elements))
    grid = ((n_elements + block - 1) // block,)
    _flipud_gather_kernel[grid](
        src,
        dst,
        n_elements,
        SHAPE=tuple(src.shape),
        SSTRIDES=tuple(src.stride()),
        NDIM=src.dim(),
        BLOCK=block,
        NEED_MASK=need_mask,
    )


def _flipud_impl(self: torch.Tensor) -> torch.Tensor:
    if self.ndim < 1:
        raise RuntimeError("Input must be >= 1-d.")

    if self.is_complex():
        return torch.view_as_complex(_flipud_impl(torch.view_as_real(self)))

    # ``torch.flipud`` materialises a contiguous result on this stack.
    output = torch.empty(self.shape, dtype=self.dtype, device=self.device)
    n_elements = self.numel()
    if n_elements == 0:
        return output

    rows = self.shape[0]
    row_size = n_elements // rows
    n_bytes = n_elements * self.element_size()
    if (
        self.is_contiguous()
        and n_bytes > _GATHER_MAX_BYTES
        and row_size >= _RUN_MIN_WORDS
        and rows > 0
    ):
        if _run_copy(self, output, rows, row_size):
            return output
    _gather_copy(self, output)
    return output


class Flipud(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return _flipud_impl(input)

    @staticmethod
    def backward(ctx, grad_output):
        return _flipud_impl(grad_output)


def flipud(self: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS FLIPUD")

    if torch.is_grad_enabled() and self.requires_grad:
        return Flipud.apply(self)
    return _flipud_impl(self)
