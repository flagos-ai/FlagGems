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

"""Kunlunxin XPU3 override for ``aten::_fill_mem_eff_dropout_mask_``.

Semantics are bit-identical to the generic implementation in
``flag_gems/ops/_fill_mem_eff_dropout_mask_.py`` (which mirrors ATen's CUDA
``curand`` stream): element ``i`` of the flat mask is word ``i % 4`` of the
Philox4x32-10 draw for counter ``(offset + 4*(i // 4)) // 4``, mapped by
``x * 2**-32 + 2**-33`` (``curand_uniform``).

Why this file exists
--------------------
The generic kernel assigns one Philox draw (four words) to one lane and then
writes the four words with four *separate* 1-D stores at a stride of four
floats (``out + 4*lane + w``).  On XPU3 a unit-stride 1-D ``tl.store`` lowers to
a block DMA, but a strided store falls back to per-lane discrete DMA.  Measured
on P800 (67.1 M floats, store-only kernels, ``triton.testing.do_bench``
median): stride-4 f32 stores 2.4 GB/s, contiguous f32 148 GB/s, contiguous
u64 576 GB/s.  The generic op is store-bound at 2.3 GB/s (114.7 ms); a
store-only kernel with its exact pattern also takes 110 ms.

The fix gives every store a unit stride and pays redundant RNG work instead:
one lane owns *two* adjacent output elements, so it needs exactly one Philox
draw (``2*lane // 4``) and keeps two of its words, which are packed into a
single contiguous 64-bit store.  That is 2x the Philox ALU of the generic
kernel (four lanes no longer share one draw) and one blocked DMA instead of
four discrete ones: 14.9 ms for 67.1 M floats (18 GB/s), 7.7x the generic
kernel, bit-exact.

Alternatives measured and rejected on XPU3:
* one element per lane, single f32 store (4x redundant Philox): 30.4 ms;
* a 2-D ``[BLOCK, 4]`` store built with ``tl.join``/``tl.interleave``: the
  layout conversion needs local memory and compilation aborts with
  ``OutOfResources: uni_sram`` even at BLOCK=64;
* four elements per lane (1x Philox) with two u64 stores at stride 2:
  49-74 ms -- any stride-2 u64 store collapses to 4-7 GB/s.

The odd-element tail (``numel % 2 == 1``, or a storage offset that breaks
8-byte alignment) uses the one-element-per-lane 32-bit kernel, which still
runs at ~8.8 GB/s.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Program tiles for the packed kernel, chosen by element count from a measured
# sweep (64..4096): small masks are dispatch-bound (BLOCK=128 wins), medium
# ones want ~512, large ones one wide contiguous burst (4096).
_BLOCK_PACKED_TINY = 128
_BLOCK_PACKED_MEDIUM = 512
_BLOCK_PACKED_WIDE = 4096
_TINY_MAX = 1 << 13
_MEDIUM_MAX = 1 << 20
# One-element-per-lane fallback tile.
_BLOCK_SCALAR = 2048


@libentry()
@triton.jit(do_not_specialize=["seed", "offset"])
def _fill_mem_eff_dropout_mask_packed_kernel(
    out_ptr,
    numel,
    seed,
    offset,
    BLOCK: tl.constexpr,
):
    CURAND_2_P32_INV: tl.constexpr = 2.3283064365386963e-10  # 1.0 / (2 ** 32)
    seed = seed.to(tl.int64)
    offset = offset.to(tl.int64)

    pid = tl.program_id(0)
    # One lane owns the *pair* of output elements (2p, 2p+1) and stores it as a
    # single 8-byte value, so consecutive lanes cover consecutive memory.
    pair = pid * BLOCK + tl.arange(0, BLOCK)
    block_idx = pair // 2  # Philox 4x32 draw that owns this pair
    half = pair % 2  # 0 -> words (0, 1); 1 -> words (2, 3)

    # ``offset`` counts outputs skipped by curand's ``curand_init``; one
    # ``curand_uniform4`` call consumes four outputs (one 4x32 counter), so the
    # counter is (offset + 4*block) // 4 -- the generic kernel's formula.
    linear = offset + block_idx * 4
    c0 = (linear // 4).to(tl.uint32)
    zero = tl.zeros_like(c0)
    r0, r1, r2, r3 = tl.philox(seed, c0, zero, zero, zero)

    lo = tl.where(half == 0, r0, r2)
    hi = tl.where(half == 0, r1, r3)
    vlo = lo.to(tl.float32) * CURAND_2_P32_INV + (CURAND_2_P32_INV / 2.0)
    vhi = hi.to(tl.float32) * CURAND_2_P32_INV + (CURAND_2_P32_INV / 2.0)
    packed = vlo.to(tl.uint32, bitcast=True).to(tl.uint64) | (
        vhi.to(tl.uint32, bitcast=True).to(tl.uint64) << 32
    )

    out64 = out_ptr.to(tl.pointer_type(tl.uint64))
    npair = numel // 2
    tl.store(out64 + pair, packed, mask=pair < npair)


@libentry()
@triton.jit(do_not_specialize=["seed", "offset"])
def _fill_mem_eff_dropout_mask_scalar_kernel(
    out_ptr,
    numel,
    seed,
    offset,
    BLOCK: tl.constexpr,
):
    CURAND_2_P32_INV: tl.constexpr = 2.3283064365386963e-10  # 1.0 / (2 ** 32)
    seed = seed.to(tl.int64)
    offset = offset.to(tl.int64)

    pid = tl.program_id(0)
    # One output element per lane, unit stride across the whole grid.  Neighbour
    # lanes repeat the same Philox draw (four outputs share one), which costs 4x
    # the RNG ALU of the generic kernel but keeps the store contiguous.
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    block_idx = idx // 4
    word = idx % 4

    linear = offset + block_idx * 4
    c0 = (linear // 4).to(tl.uint32)
    zero = tl.zeros_like(c0)
    r0, r1, r2, r3 = tl.philox(seed, c0, zero, zero, zero)
    r = tl.where(word == 0, r0, tl.where(word == 1, r1, tl.where(word == 2, r2, r3)))

    val = r.to(tl.float32) * CURAND_2_P32_INV + (CURAND_2_P32_INV / 2.0)
    tl.store(out_ptr + idx, val, mask=idx < numel)


def _fill_mem_eff_dropout_mask_(self, dropout_p, seed, offset):
    logger.debug("GEMS FILL_MEM_EFF_DROPOUT_MASK")
    assert self.is_contiguous(), "self must be contiguous"
    assert self.dim() == 4, "_fill_mem_eff_dropout_mask_ expects a 4D tensor"
    assert self.dtype == torch.float32, "self must be a float32 tensor"

    numel = self.numel()
    if numel == 0:
        return self

    # The packed kernel stores 8 bytes per lane, so it needs an even element
    # count and an 8-byte-aligned base.  Both are host integer checks.
    if numel % 2 == 0 and self.data_ptr() % 8 == 0:
        if numel <= _TINY_MAX:
            BLOCK = _BLOCK_PACKED_TINY
        elif numel <= _MEDIUM_MAX:
            BLOCK = _BLOCK_PACKED_MEDIUM
        else:
            BLOCK = _BLOCK_PACKED_WIDE
        grid = (triton.cdiv(numel // 2, BLOCK),)
        with torch_device_fn.device(self.device):
            _fill_mem_eff_dropout_mask_packed_kernel[grid](
                self,
                numel,
                seed,
                offset,
                BLOCK,
            )
    else:
        BLOCK = _BLOCK_SCALAR
        grid = (triton.cdiv(numel, BLOCK),)
        with torch_device_fn.device(self.device):
            _fill_mem_eff_dropout_mask_scalar_kernel[grid](
                self,
                numel,
                seed,
                offset,
                BLOCK,
            )
    return self
