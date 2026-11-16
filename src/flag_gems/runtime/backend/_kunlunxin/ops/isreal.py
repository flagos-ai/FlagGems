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
"""Kunlunxin XPU3 override for ``aten::isreal``.

Two device-side rewrites, both measured on P800 (see
``harness/perf_ir/probe_isreal_*.log``):

* Real dtypes: the generic kernel writes the bool result **one byte per lane**
  (``tl.store(i8_ptr + offs, 1, mask=...)``).  On XPU3 a per-lane byte store is
  ~12x slower than a 32-bit store: a 67M-element fill took 577 us (masked byte
  store, BLOCK=4096) while the same fill as packed ``int32`` words takes 73 us
  at BLOCK=16384 (torch is 35 us).  The output buffer is therefore allocated as
  ``int32`` words and every word is filled with ``0x01010101`` (four ``True``
  bytes), then viewed back as a bool tensor.  Padding bytes past ``numel`` live
  inside the word-rounded allocation, so the launch never needs an address mask
  when the word count is a multiple of BLOCK; otherwise a single masked tile
  handles the tail (``EVEN`` constexpr).

* Complex dtypes: unchanged semantics (imaginary word -0.0 inclusive), but the
  flat tile grows from 1024 to 4096 pairs and the tail mask becomes a
  constexpr branch.  The complex path is read bound; bigger tiles were measured
  to spill ``uni_sram`` (BLOCK=8192 falls off a cliff), 4096 is the sweet spot.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_REAL_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.bool,
)
_COMPLEX_DTYPES = (torch.complex64, torch.complex128)

# Four bool bytes per int32 word; ``isreal`` is unconditionally True for every
# real dtype, so the word value is a compile-time constant.
_TRUE_WORD = 0x01010101


@libentry()
@triton.jit
def isreal_fill_word_kernel(
    out_ptr, n_words, BLOCK: tl.constexpr, EVEN: tl.constexpr
):
    """Fill ``n_words`` int32 words with four True bytes each."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        tl.store(out_ptr + offsets, 0x01010101)
    else:
        tl.store(out_ptr + offsets, 0x01010101, mask=offsets < n_words)


@libentry()
@triton.jit
def isreal_complex_kernel(
    in_real_ptr,
    out_ptr,
    n_elements,
    IS_FP64: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    """Test the imaginary component of each complex element for an exact zero."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        imag = tl.load(in_real_ptr + offsets * 2 + 1)
        if IS_FP64:
            bits = imag.to(tl.int64, bitcast=True) & 0x7FFFFFFFFFFFFFFF
        else:
            bits = imag.to(tl.int32, bitcast=True) & 0x7FFFFFFF
        tl.store(out_ptr + offsets, bits == 0)
    else:
        mask = offsets < n_elements
        imag = tl.load(in_real_ptr + offsets * 2 + 1, mask=mask, other=0.0)
        if IS_FP64:
            bits = imag.to(tl.int64, bitcast=True) & 0x7FFFFFFFFFFFFFFF
        else:
            bits = imag.to(tl.int32, bitcast=True) & 0x7FFFFFFF
        tl.store(out_ptr + offsets, bits == 0, mask=mask)


def isreal(input: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS ISREAL")
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"isreal expects a Tensor, got {type(input).__name__}")
    if input.dtype not in _REAL_DTYPES + _COMPLEX_DTYPES:
        raise TypeError(
            "isreal only supports floating point, complex, integral and bool "
            f"inputs, got {input.dtype}"
        )

    n_elements = input.numel()
    if n_elements == 0:
        return torch.empty(input.size(), dtype=torch.bool, device=input.device)

    if input.dtype in _COMPLEX_DTYPES:
        # A lazy conjugate / negation only flips the sign of the imaginary
        # component, which cannot turn a zero into a non-zero, so both bits may
        # be ignored.  They must still be resolved before taking the storage
        # view, since torch.view_as_real rejects lazy conjugate/negation
        # tensors.
        if input.is_conj():
            input = input.resolve_conj()
        if input.is_neg():
            input = input.resolve_neg()
        # The kernel walks the interleaved storage linearly, so non-contiguous
        # inputs need to be materialized first.
        if not input.is_contiguous():
            input = input.contiguous()
        is_fp64 = input.dtype == torch.complex128
        # fp32 pairs tolerate a 4096 tile; fp64 pairs spill uni_sram, so keep
        # the generic 1024 tile there (complex128 is not on the perf matrix).
        # Tiny inputs stay on the 1024 tile: one masked 4096-lane program is
        # measurably slower than one 1024-lane program at the launch floor.
        if is_fp64 or n_elements < 4096:
            BLOCK = 1024
        else:
            BLOCK = 4096
        out = torch.empty(input.size(), dtype=torch.bool, device=input.device)
        isreal_complex_kernel[(triton.cdiv(n_elements, BLOCK),)](
            torch.view_as_real(input),
            out,
            n_elements,
            IS_FP64=is_fp64,
            BLOCK=BLOCK,
            EVEN=(n_elements % BLOCK) == 0,
        )
        return out

    # Real dtype: every element is real, the input is never read.  Write four
    # True bytes per int32 lane instead of one byte per lane.
    n_words = (n_elements + 3) // 4
    # Tile ladder measured on XPU3 (``probe_isreal_ladder.log``): the fill is
    # tile driven, and the best tile grows with the word count.  32768 was
    # verified safe for the wide end (correct values, no fault).
    if n_words >= 1 << 19:
        BLOCK = 32768
    elif n_words >= 1 << 15:
        BLOCK = 8192
    elif n_words >= 1 << 10:
        BLOCK = 1024
    else:
        BLOCK = max(64, triton.next_power_of_2(n_words))
    # Allocate the word-rounded bool buffer so the padded tail bytes stay inside
    # the allocation and the kernel can stay mask free on aligned launches.
    out = torch.empty(n_words * 4, dtype=torch.bool, device=input.device)
    isreal_fill_word_kernel[(triton.cdiv(n_words, BLOCK),)](
        out.view(torch.int32),
        n_words,
        BLOCK=BLOCK,
        EVEN=(n_words % BLOCK) == 0,
    )
    return out[:n_elements].view(input.size())
