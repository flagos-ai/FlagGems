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
"""Kunlunxin (XPU3) override of ``aten::_is_any_true``.

Generic implementation issues on this backend that this file fixes:

* Its large-input path (``numel > 65536``) reduces with a device-wide
  ``tl.atomic_or`` into an int32 accumulator.  ``tl.atomic_or`` does not lower
  here: compilation dies with ``RuntimeError: Failed to tune buffer size``
  (and the strided variant additionally hangs the LLVM toolchain, so a plain
  ``pytest tests/test_is_any_true.py`` never returns).  Device atomics are also
  documented as unreliable on this backend.
* Its strided path walks every lane with a per-lane integer division chain.
  At the 65536-element single-tile boundary that costs ~156 us versus ~9 us
  for the native op (measured, card 3).
* The single tile is always built out of bytes (one lane per bool).  Reading
  the same storage as int64 lanes is 4-8x fewer lanes for the same bytes and
  is the difference between ~0.47x and ~0.83x of native at 1 MiB.

Strategy (no atomics, no host compute):

* ``numel == 0`` -> an allocated/zeroed scalar (never launch an all-false
  masked load, which can abort the program before its store).
* contiguous, 8-byte-aligned, ``numel >= 512`` -> int64-word kernel; a tail of
  up to 7 bytes is folded into the same kernel with a constexpr ``HAS_TAIL``
  (extra 64-lane byte load, no reduce inside ``scf.if``).
* one tile when the word lane count fits in ``_SINGLE_LANES`` / the byte count
  fits in ``_BYTE_SINGLE_MAX``; otherwise a deterministic two-stage
  partial + tree-reduce (never loop-carried reduction, which miscompiles here).
* 1-D views use ``idx * stride`` directly instead of digit decomposition.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# Largest single-program tile (in lanes) that still beats a second launch.
_SINGLE_LANES = 32768
# Largest byte tile kept in a single program; above this the gapped/strided
# access pattern of one CTA starts to serialise.
_BYTE_SINGLE_MAX = 8192
# Streaming tile sizes (lanes per program) for the two-stage path.
_TILE_WORD = 16384
_TILE_BYTE_SMALL = 8192
_TILE_BYTE_LARGE = 16384


def _warps(lanes: int) -> int:
    return max(1, min(32, lanes // 256))


@triton.jit
def _any_word_kernel(
    src,
    dst,
    n8,
    tail_start,
    TAILN: tl.constexpr,
    PARTIAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    idx = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    # Any nonzero byte makes the whole int64 word nonzero, so endianness and
    # noncanonical bool payloads are both irrelevant here.
    v = tl.load(src.to(tl.pointer_type(tl.int64)) + idx, idx < n8, other=0)
    r = tl.max((v != 0).to(tl.int32), 0)
    # The trailing <8 bytes are folded in with scalar loads.  A second
    # `tl.reduce` of a different width in the same kernel trips the XPU
    # legalizer ("reduction rowspercore is not consistent"), so no reduce here.
    # OR is idempotent, hence safe when every program repeats the check.
    if TAILN > 0:
        tail = src.to(tl.pointer_type(tl.int8)) + tail_start
        for k in tl.static_range(TAILN):
            tb = tl.load(tail + k).to(tl.int32)
            r = tl.maximum(r, (tb != 0).to(tl.int32))
    if PARTIAL:
        tl.store(dst + pid, r.to(tl.int8))
    else:
        tl.store(dst, r.to(tl.int8))


@triton.jit
def _any_byte_kernel(
    src,
    dst,
    n,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    CONTIG: tl.constexpr,
    RANK1: tl.constexpr,
    PARTIAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    idx = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    if CONTIG:
        offset = idx
    elif RANK1:
        offset = idx * STRIDES[0]
    else:
        rem = idx
        offset = tl.full((BLOCK,), 0, tl.int64)
        for d in tl.static_range(len(SHAPE) - 1, -1, -1):
            offset += (rem % SHAPE[d]) * STRIDES[d]
            rem = rem // SHAPE[d]
    value = tl.load(src.to(tl.pointer_type(tl.int8)) + offset, idx < n, other=0)
    r = tl.max((value != 0).to(tl.int32), 0)
    if PARTIAL:
        tl.store(dst + pid, r.to(tl.int8))
    else:
        tl.store(dst, r.to(tl.int8))


@triton.jit
def _any_finish_kernel(src, dst, np_, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK).to(tl.int64)
    v = tl.load(src.to(tl.pointer_type(tl.int8)) + idx, idx < np_, other=0)
    r = tl.max((v != 0).to(tl.int32), 0)
    tl.store(dst, r.to(tl.int8))


def _finish(partials, np_, out):
    block = max(64, triton.next_power_of_2(np_))
    _any_finish_kernel[(1,)](
        partials, out, np_, BLOCK=block, num_warps=_warps(block)
    )


def _word_path(self, n):
    n8 = n >> 3
    tail = n - (n8 << 3)
    tail_start = n8 << 3
    lanes = triton.next_power_of_2(n8)
    out = torch.empty((), dtype=torch.bool, device=self.device)
    if lanes <= _SINGLE_LANES:
        _any_word_kernel[(1,)](
            self,
            out,
            n8,
            tail_start,
            TAILN=tail,
            PARTIAL=False,
            BLOCK=lanes,
            num_warps=_warps(lanes),
        )
        return out
    np_ = triton.cdiv(n8, _TILE_WORD)
    partials = torch.empty((np_,), dtype=torch.bool, device=self.device)
    _any_word_kernel[(np_,)](
        self,
        partials,
        n8,
        tail_start,
        TAILN=tail,
        PARTIAL=True,
        BLOCK=_TILE_WORD,
        num_warps=_warps(_TILE_WORD),
    )
    _finish(partials, np_, out)
    return out


def _byte_path(self, n, contig, rank1):
    out = torch.empty((), dtype=torch.bool, device=self.device)
    shape = tuple(self.shape)
    strides = self.stride()
    if n <= _BYTE_SINGLE_MAX:
        block = max(64, triton.next_power_of_2(n))
        _any_byte_kernel[(1,)](
            self,
            out,
            n,
            SHAPE=shape,
            STRIDES=strides,
            CONTIG=contig,
            RANK1=rank1,
            PARTIAL=False,
            BLOCK=block,
            num_warps=_warps(block),
        )
        return out
    tile = _TILE_BYTE_SMALL if n <= 65536 else _TILE_BYTE_LARGE
    np_ = triton.cdiv(n, tile)
    partials = torch.empty((np_,), dtype=torch.bool, device=self.device)
    _any_byte_kernel[(np_,)](
        self,
        partials,
        n,
        SHAPE=shape,
        STRIDES=strides,
        CONTIG=contig,
        RANK1=rank1,
        PARTIAL=True,
        BLOCK=tile,
        num_warps=_warps(tile),
    )
    _finish(partials, np_, out)
    return out


def _is_any_true(self):
    """Reduce every logical boolean element, including arbitrary-rank views."""
    logger.debug("GEMS _IS_ANY_TRUE")
    if not isinstance(self, torch.Tensor):
        raise TypeError("_is_any_true: self must be a Tensor")
    if self.dtype != torch.bool:
        raise RuntimeError("_is_any_true: input must be a bool tensor")
    n = self.numel()
    with torch_device_fn.device(self.device):
        if n == 0:
            # A kernel over an all-false mask can abort before its store here.
            return torch.zeros((), dtype=torch.bool, device=self.device)
        contig = self.is_contiguous()
        if contig and n >= 512 and self.data_ptr() % 8 == 0:
            return _word_path(self, n)
        return _byte_path(self, n, contig, self.dim() == 1)
