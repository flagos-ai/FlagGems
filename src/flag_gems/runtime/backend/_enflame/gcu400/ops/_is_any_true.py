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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)
# One reduction program handles moderate inputs; large inputs stream tiles
# into an int32 atomic OR accumulator, whose low byte is the scalar bool.
# Threshold set to 16384: a larger single-program BLOCK degrades element-wise gather for non-contiguous views,
# and above the threshold, splitting into multi-program chunked reductions reduces latency and increases parallelism.
_SINGLE_MAX = 16384
# The streaming tile aligns with the threshold, balancing single-pass bandwidth and per-program register usage.
_BLOCK = 16384
# Tile for the packed path (4 bools/word): 4096 words = 16 KiB byte plane, comparable in
# working-set size to the byte-wise path's single program; the four tiers 2048/4096/8192/16384 differ by <5% on 64K-1M
# elements,
# so the median 4096 is chosen to balance small single-program and large multi-program cases.
_PACKED_BLOCK = 4096
# Upper limit of the single-program packed path: 16384 words = 65536 bool elements (64 KiB),
# coverable in one tile; measured 5.2us, faster than splitting into multiple programs (which requires zero
# initialization).
_PACKED_SINGLE_WORDS = 16384
_PACKED_WARPS = 4


@triton.jit
def _any_kernel(
    src,
    dst,
    n,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    CONTIG: tl.constexpr,
    ATOMIC: tl.constexpr,
    BLOCK: tl.constexpr,
    ONE_DIM: tl.constexpr,
    ZERO_STRIDE: tl.constexpr,
):
    idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    if CONTIG:
        offset = idx
    elif ZERO_STRIDE:
        # Scalar/expand views have all elements sharing the same address, avoiding redundant gather across the whole
        # tile.
        offset = tl.full((BLOCK,), 0, tl.int64)
    elif ONE_DIM:
        # For 1D strided views, scale linearly by the first-dimension stride to avoid per-element int64 division/modulo.
        offset = idx * STRIDES[0]
    else:
        rem = idx
        offset = tl.full((BLOCK,), 0, tl.int64)
        for d in tl.static_range(len(SHAPE) - 1, -1, -1):
            offset += (rem % SHAPE[d]) * STRIDES[d]
            rem = rem // SHAPE[d]
    # Bool tensors may contain noncanonical nonzero bytes; interpret every
    # nonzero byte as true, as the native reduction does.
    value = tl.load(src.to(tl.pointer_type(tl.int8)) + offset, idx < n, other=0)
    result = tl.max((value != 0).to(tl.int32), 0)
    if ATOMIC:
        tl.atomic_or(dst, result)
    else:
        tl.store(dst, result.to(tl.int8))


@triton.jit
def _any_packed_kernel(
    src,
    dst,
    n_words,
    MULTI: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Reduces a contiguous bool buffer packed as int32 words.

    Every 4 bool bytes form exactly one int32 word and word != 0 is equivalent to any byte being
    non-zero, so the reduction width drops to 1/4 of the per-byte path while int64 offsets and
    per-element gathers disappear: 65536 elements 21.6us -> 4.7us, 1M 38.7us -> 14.9us (2026-11-17).
    """
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    word = tl.load(
        src.to(tl.pointer_type(tl.int32)) + offset, offset < n_words, other=0
    )
    result = tl.max((word != 0).to(tl.int32), 0)
    if MULTI:
        # Multiple programs share the same 1-byte output and write only "hits": zeros are guaranteed by zeros
        # initialization,
        # so no int32 atomic accumulator is needed and empty tiles cannot overwrite it with 0.
        if result != 0:
            tl.store(dst, result.to(tl.int8))
    else:
        tl.store(dst, result.to(tl.int8))


def _packed_eligible(tensor):
    """The packed path needs a contiguous, 4-byte-aligned bool tensor whose numel is a multiple of 4."""
    if not tensor.is_contiguous():
        return False
    if tensor.numel() % 4 != 0 or tensor.numel() == 0:
        return False
    # A view's storage_offset may not be a multiple of 4, so int32 reinterpretation would read misaligned values.
    return tensor.data_ptr() % 4 == 0


def _zeroed_word_buffer(device):
    """Allocates as float32 and reinterprets as int32, bypassing the slow zero-init path for int/bool.

    On GCU ``torch.zeros`` / ``zero_`` cost ~17-19us for bool, int8 and int32 (2026-11-17:
    zeros((),bool)=18.6us, zeros((1,),int32)=16.9us, zeros((4,),int8)=18.7us) but only ~4.7us
    for float32, and an empty allocation only ~2.1us. The reduction accumulator needs just four
    zero bytes, so borrowing float32's zero init yields the same 4-byte zero buffer; the bit
    pattern of 0.0f is int32 0.
    """
    return torch.zeros((1,), dtype=torch.float32, device=device).view(torch.int32)


def _launch_packed(self, n_words):
    if n_words <= _PACKED_SINGLE_WORDS:
        # Single program covering all words in one tile: the output uses empty + unconditional store,
        # so no zero initialization is needed (measured ~5us for 64 KiB input vs 21.6us for the byte-wise path).
        output = torch.empty((), dtype=torch.bool, device=self.device)
        block = max(16, triton.next_power_of_2(n_words))
        _any_packed_kernel[(1,)](
            self,
            output,
            n_words,
            MULTI=False,
            BLOCK=block,
            num_warps=max(1, min(8, block // 1024)),
        )
        return output
    # Multiple programs: write only "hits"; zeros are guaranteed by the zero-initialized buffer, avoiding int32 atomic
    # accumulation.
    output = _zeroed_word_buffer(self.device).view(torch.bool)[0]
    _any_packed_kernel[(triton.cdiv(n_words, _PACKED_BLOCK),)](
        self,
        output,
        n_words,
        MULTI=True,
        BLOCK=_PACKED_BLOCK,
        num_warps=_PACKED_WARPS,
    )
    return output


def _is_any_true(self):
    """Reduce every logical boolean element, including arbitrary-rank views."""
    logger.debug("GEMS_ENFLAME _IS_ANY_TRUE")
    if not isinstance(self, torch.Tensor):
        raise TypeError("_is_any_true: self must be a Tensor")
    if self.dtype != torch.bool:
        raise RuntimeError("_is_any_true: input must be a bool tensor")
    n = self.numel()
    with torch_device_fn.device(self.device):
        # Contiguous, 4-byte-aligned bool buffers prefer the int32 packed reduction (width 1/4).
        if _packed_eligible(self):
            return _launch_packed(self, n // 4)
        atomic = n > _SINGLE_MAX
        if atomic:
            # The accumulator also borrows float32 zero initialization (int32 zeros take ~17us on GCU).
            accumulator = _zeroed_word_buffer(self.device)
            output = accumulator.view(torch.bool)[0]
            target = accumulator
            block, warps = _BLOCK, 4
        else:
            output = torch.empty((), dtype=torch.bool, device=self.device)
            target = output
            block = max(16, triton.next_power_of_2(n))
            warps = max(1, min(8, block // 1024))
        _any_kernel[(max(1, triton.cdiv(n, block)),)](
            self,
            target,
            n,
            SHAPE=tuple(self.shape),
            STRIDES=self.stride(),
            CONTIG=self.is_contiguous(),
            ATOMIC=atomic,
            BLOCK=block,
            ONE_DIM=len(self.shape) == 1,
            ZERO_STRIDE=all(stride == 0 for stride in self.stride()),
            num_warps=warps,
        )
    return output
