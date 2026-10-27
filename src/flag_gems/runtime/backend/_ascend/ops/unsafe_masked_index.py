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

import torch
import triton
import triton.language as tl


@triton.jit
def _unsafe_masked_index_ub(
    self_ptr,
    mask_ptr,
    indices_ptr,
    out_ptr,
    fill,
    SELF_N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Whole self is loaded into UB once (SELF_N <= UB capacity); the random
    # gather then happens in UB via tl.gather, which is far cheaper than
    # per-lane GM/L2 gathers.
    src = tl.load(self_ptr + tl.arange(0, SELF_N))
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    idx = tl.load(indices_ptr + offsets)
    m = tl.load(mask_ptr + offsets)
    mb = m != 0  # Ascend loads bool as i8; convert to i1 for masking
    vals = tl.gather(src, idx, 0)
    tl.store(out_ptr + offsets, tl.where(mb, vals, fill))


@triton.jit
def _unsafe_masked_index_even(
    self_ptr,
    mask_ptr,
    indices_ptr,
    out_ptr,
    fill,
    BLOCK: tl.constexpr,
):
    # Specialization for n % BLOCK == 0 (grid == n // BLOCK): no boundary masks.
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    idx = tl.load(indices_ptr + offsets)
    m = tl.load(mask_ptr + offsets)
    mb = m != 0  # Ascend loads bool as i8; convert to i1 for masking
    # Unsafe masked gather: masked lanes never load self, and receive `fill`.
    vals = tl.load(self_ptr + idx, mask=mb, other=fill)
    tl.store(out_ptr + offsets, vals)


@triton.jit
def _unsafe_masked_index_kernel(
    self_ptr,
    mask_ptr,
    indices_ptr,
    out_ptr,
    n_elements,
    fill,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n_elements

    mask = tl.load(mask_ptr + offsets, mask=valid, other=False)
    mask_b = mask != 0  # Ascend loads bool as i8; convert to i1 for masking
    idx = tl.load(indices_ptr + offsets, mask=valid, other=0)

    # Unsafe masked gather: masked lanes never load self, and receive `fill`.
    vals = tl.load(self_ptr + idx, mask=mask_b & valid, other=fill)

    tl.store(out_ptr + offsets, vals, mask=valid)


def _is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def unsafe_masked_index(self, mask, indices, fill):
    n = self.numel()
    out = torch.empty_like(self)

    # UB-resident gather path when the whole source fits in unified buffer:
    # fp16/bf16 up to 65536 elements (128KB), fp32 up to 32768 (128KB).
    dt = self.dtype
    if dt == torch.float32:
        ub_max = 32768
    elif dt == torch.float16 or dt == torch.bfloat16:
        ub_max = 65536
    else:
        ub_max = 0

    if _is_pow2(n) and n <= ub_max:
        if n <= 2048:
            BLOCK = 128
        elif n <= 8192:
            BLOCK = 256
        else:
            BLOCK = 512
        grid = (n // BLOCK,)
        _unsafe_masked_index_ub[grid](
            self, mask, indices, out, fill, SELF_N=n, BLOCK=BLOCK
        )
    elif n % 512 == 0:
        if n >= 8192:
            BLOCK = 512
        elif n >= 2048:
            BLOCK = 256
        else:
            BLOCK = 128
        grid = (n // BLOCK,)
        _unsafe_masked_index_even[grid](self, mask, indices, out, fill, BLOCK=BLOCK)
    else:
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        _unsafe_masked_index_kernel[grid](
            self, mask, indices, out, n, fill, BLOCK=BLOCK
        )
    return out
