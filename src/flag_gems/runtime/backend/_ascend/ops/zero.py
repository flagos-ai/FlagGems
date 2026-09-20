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
def _zero_masked(out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(
        out_ptr + offsets,
        tl.zeros([BLOCK_SIZE], dtype=out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _zero(out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + offsets, tl.zeros([BLOCK_SIZE], dtype=out_ptr.dtype.element_ty))


# One program per contiguous BLOCK_SIZE chunk; large chunks amortize per-task
# dispatch / UB->GM DMA setup and issue long contiguous bursts. For large
# tensors the widest block that fits the 192KB unified buffer wins (streaming
# store throughput), while small tensors use small blocks so no masked
# overhead is wasted.
MIN_BLOCK = 1024


def _pick_block(n, max_block):
    bs = max_block
    while n % bs != 0 and bs > MIN_BLOCK:
        bs //= 2
    return bs


def zero(x):
    output = torch.empty_like(x, memory_format=torch.contiguous_format)
    n = x.numel()
    # 32768 x 2-byte = 64KB / 4-byte = 128KB: fits the 192KB UB; 32768 was
    # consistently ~0.4% faster on the 32MB workloads than 65536-element blocks.
    bs = _pick_block(n, 32768)
    if n % bs == 0:
        _zero[(n // bs,)](output, BLOCK_SIZE=bs, num_warps=8)
    else:
        _zero_masked[(triton.cdiv(n, bs),)](output, n, BLOCK_SIZE=bs, num_warps=8)
    return output
