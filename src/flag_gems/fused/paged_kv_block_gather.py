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

"""Gather the paged KV blocks of each sequence into one contiguous tensor.

A paged KV cache stores blocks out of order; ``block_table[s]`` lists the
physical blocks that make up sequence ``s``. This op materializes each
sequence's blocks contiguously, which is what an attention kernel that cannot
read the page table itself needs.

``cache`` is ``[num_blocks, block_size, head_dim]`` and ``block_table`` is
``[num_seqs, blocks_per_seq]``; the result is
``[num_seqs, blocks_per_seq, block_size, head_dim]``.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def paged_kv_block_gather_kernel(
    cache,
    block_table,
    out,
    num_blocks,
    table_stride,
    BLOCK_ELEMS: tl.constexpr,
    TILE: tl.constexpr,
):
    """One program per (sequence, block) pair; copies one block in TILE chunks."""
    seq = tl.program_id(0)
    slot = tl.program_id(1)

    physical = tl.load(block_table + seq * table_stride + slot)
    # A block id outside the cache names no block; leave the destination zeroed.
    valid = (physical >= 0) & (physical < num_blocks)

    dst_base = (seq * tl.num_programs(1) + slot) * BLOCK_ELEMS
    src_base = physical * BLOCK_ELEMS

    for off in range(0, BLOCK_ELEMS, TILE):
        idx = off + tl.arange(0, TILE)
        mask = (idx < BLOCK_ELEMS) & valid
        vals = tl.load(cache + src_base + idx, mask=mask, other=0.0)
        tl.store(out + dst_base + idx, vals, mask=mask)


def paged_kv_block_gather(cache, block_table):
    logger.debug("GEMS PAGED_KV_BLOCK_GATHER")
    assert cache.ndim == 3, "cache must be [num_blocks, block_size, head_dim]"
    assert block_table.ndim == 2, "block_table must be [num_seqs, blocks_per_seq]"

    cache = cache.contiguous()
    num_blocks, block_size, head_dim = cache.shape
    num_seqs, blocks_per_seq = block_table.shape

    out = torch.empty(
        (num_seqs, blocks_per_seq, block_size, head_dim),
        dtype=cache.dtype,
        device=cache.device,
    )
    block_elems = block_size * head_dim
    tile = min(triton.next_power_of_2(block_elems), 8192)

    with torch_device_fn.device(cache.device):
        paged_kv_block_gather_kernel[(num_seqs, blocks_per_seq)](
            cache,
            block_table,
            out,
            num_blocks,
            block_table.stride(0),
            BLOCK_ELEMS=block_elems,
            TILE=tile,
        )
    return out
