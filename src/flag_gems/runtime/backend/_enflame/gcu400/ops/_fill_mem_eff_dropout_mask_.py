# Copyright 2026 FlagOS Contributors.
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


@triton.jit(do_not_specialize=["seed", "offset", "counter_base"])
def _fill_mem_eff_dropout_mask_kernel(
    out_ptr,
    numel,
    seed,
    offset,
    counter_base,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    FAST: tl.constexpr,
):
    # ``_fill_mem_eff_dropout_mask_`` fills a contiguous 4D float32 tensor of
    # shape ``(batch, heads, queries, keys)`` with random uniform values in
    # [0, 1) drawn from a Philox4x32-10 stream identified by ``seed`` and
    # ``offset``. The CUDA reference (curand) advances its counter by one per
    # ``curand_uniform4`` call (i.e. per 4 floats), so the counter value for
    # the 4-float block with linear index ``i`` is ``(offset + i) // 4``. The
    # reference writes the four floats of each block consecutively, so the
    # whole tensor is populated in linear index order and a flat 1D grid
    # reproduces the exact same random stream.
    CURAND_2_P32_INV: tl.constexpr = 2.3283064365386963e-10  # 1.0 / (2 ** 32)
    seed = seed.to(tl.int64)

    pid = tl.program_id(0)
    # Each program handles ``BLOCK`` whole Philox blocks; one Philox block
    # yields 4 floats written to 4 consecutive linear positions.
    block_idx = pid * BLOCK + tl.arange(0, BLOCK)  # index of the 4-float block
    if FAST:
        # Host-precomputed trunc(offset / 4): for offset >= 0 (or a multiple
        # of 4) trunc((offset + 4*b) / 4) == trunc(offset / 4) + b, and the
        # uint32 wrap matches the in-kernel formula of the generic kernel.
        # This removes the per-lane int64 division, the dominant cost on GCU.
        c0 = counter_base.to(tl.uint32) + block_idx.to(tl.uint32)
    else:
        # Pathological negative-offset fallback: byte-identical to the generic
        # kernel's in-kernel sdiv formula (truncating division does not
        # distribute across a sign crossing), so bit-exactness holds for any
        # Scalar input at zero cost on the hot path.
        offset64 = offset.to(tl.int64)
        linear = offset64 + block_idx * 4
        c0 = (linear // 4).to(tl.uint32)
    zero = tl.zeros_like(c0)
    r0, r1, r2, r3 = tl.philox(seed, c0, zero, zero, zero)
    r0 = r0.to(tl.float32) * CURAND_2_P32_INV + (CURAND_2_P32_INV / 2.0)
    r1 = r1.to(tl.float32) * CURAND_2_P32_INV + (CURAND_2_P32_INV / 2.0)
    r2 = r2.to(tl.float32) * CURAND_2_P32_INV + (CURAND_2_P32_INV / 2.0)
    r3 = r3.to(tl.float32) * CURAND_2_P32_INV + (CURAND_2_P32_INV / 2.0)

    elem_start = block_idx * 4  # linear (float) index of the block's first element
    if EVEN:
        # Full coverage: numel is a multiple of BLOCK * 4, drop all masks.
        tl.store(out_ptr + elem_start, r0)
        tl.store(out_ptr + elem_start + 1, r1)
        tl.store(out_ptr + elem_start + 2, r2)
        tl.store(out_ptr + elem_start + 3, r3)
    else:
        tl.store(out_ptr + elem_start, r0, mask=elem_start + 0 < numel)
        tl.store(out_ptr + elem_start + 1, r1, mask=elem_start + 1 < numel)
        tl.store(out_ptr + elem_start + 2, r2, mask=elem_start + 2 < numel)
        tl.store(out_ptr + elem_start + 3, r3, mask=elem_start + 3 < numel)


def _fill_mem_eff_dropout_mask_(self, dropout_p, seed, offset):
    logger.debug("GEMS_ENFLAME FILL_MEM_EFF_DROPOUT_MASK")
    assert self.is_contiguous(), "self must be contiguous"
    assert self.dim() == 4, "_fill_mem_eff_dropout_mask_ expects a 4D tensor"
    assert self.dtype == torch.float32, "self must be a float32 tensor"

    numel = self.numel()
    offset = int(offset)
    # curand advances its counter by one per 4-float block; compute the block
    # counter base once on the host instead of dividing per lane on the device.
    counter_base = offset // 4  # floor; == trunc for offset >= 0
    fast = offset >= 0 or offset % 4 == 0
    # A/B sweeps on GCU400 (kernel mode, do_bench median): num_warps=2 and the
    # widest BLOCK win (6.4x vs the generic 256-block kernel); shrink BLOCK
    # only to keep at least ~256 programs so small masks still fill the device.
    BLOCK = 4096
    while BLOCK > 256 and triton.cdiv(numel, BLOCK * 4) < 256:
        BLOCK //= 2
    even = (numel % (BLOCK * 4)) == 0
    grid = (triton.cdiv(numel, BLOCK * 4),)
    with torch_device_fn.device(self.device):
        _fill_mem_eff_dropout_mask_kernel[grid](
            self,
            numel,
            seed,
            offset,
            counter_base,
            BLOCK,
            even,
            fast,
            num_warps=2,
        )
    return self
