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

"""Ascend TensorView override for paged_kv_block_gather.

The whole operation is one gather whose indices are a row of the block table, so
a 3-D GatherScatterView expresses it directly: the sparse dimension takes the
physical block ids and the ``[block_size, head_dim]`` payload transfers as a
block DMA per block. One program per sequence handles all of its blocks, where
the generic kernel needs one program per (sequence, block) pair and copies in
chunks.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.fused.paged_kv_block_gather import (
    paged_kv_block_gather as _paged_kv_block_gather_generic,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._ascend.tensorview import (
    HAS_TENSORVIEW,
    tensorview_call,
    UB_BYTES,
)

logger = logging.getLogger(__name__)

if HAS_TENSORVIEW:
    import triton.experimental.tle as tle_exp
    from triton.experimental.tle.language.dsa import tile_load, tile_store

if HAS_TENSORVIEW:

    @triton.jit
    def paged_kv_block_gather_tv_kernel(
        cache,
        block_table,
        out,
        num_blocks,
        num_seqs,
        table_stride,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCKS_PER_SEQ: tl.constexpr,
    ):
        seq = tl.program_id(0)
        slots = tl.arange(0, BLOCKS_PER_SEQ)
        physical = tl.load(block_table + seq * table_stride + slots).to(tl.int32)

        cache_view = tle_exp.make_gather_scatter_view(
            cache,
            shape=[num_blocks, BLOCK_SIZE, HEAD_DIM],
            strides=[BLOCK_SIZE * HEAD_DIM, HEAD_DIM, 1],
            tile=[BLOCKS_PER_SEQ, BLOCK_SIZE, HEAD_DIM],
            sparse_dim=[0],
        )
        gathered = tile_load(
            cache_view,
            index=(physical, 0, 0),
            dst_space=tle_exp.language.dsa.ascend.UB,
        )

        # The destination holds one group of BLOCKS_PER_SEQ blocks per sequence,
        # and a partition view takes tile indices, so the sequence id selects
        # the group directly.
        out_view = tle_exp.make_partition_view(
            out,
            shape=[num_seqs * BLOCKS_PER_SEQ, BLOCK_SIZE, HEAD_DIM],
            strides=[BLOCK_SIZE * HEAD_DIM, HEAD_DIM, 1],
            tile=[BLOCKS_PER_SEQ, BLOCK_SIZE, HEAD_DIM],
        )
        tile_store(
            out_view,
            value=gathered,
            index=(seq, 0, 0),
            src_space=tle_exp.language.dsa.ascend.UB,
        )

    def _paged_kv_block_gather_tensorview(cache, block_table):
        logger.debug("GEMS_ASCEND PAGED_KV_BLOCK_GATHER(TENSORVIEW)")
        cache = cache.contiguous()
        num_blocks, block_size, head_dim = cache.shape
        num_seqs, blocks_per_seq = block_table.shape
        out = torch.empty(
            (num_seqs, blocks_per_seq, block_size, head_dim),
            dtype=cache.dtype,
            device=cache.device,
        )
        with torch_device_fn.device(cache.device):
            paged_kv_block_gather_tv_kernel[(num_seqs,)](
                cache,
                block_table,
                out,
                num_blocks,
                num_seqs,
                block_table.stride(0),
                BLOCK_SIZE=block_size,
                HEAD_DIM=head_dim,
                BLOCKS_PER_SEQ=blocks_per_seq,
            )
        return out

    def _tv_eligible(cache, block_table):
        if cache.ndim != 3 or block_table.ndim != 2:
            return False
        if cache.numel() == 0 or block_table.numel() == 0:
            return False
        blocks_per_seq = block_table.size(1)
        # The sparse tile covers the whole block-table row in one go, so the row
        # length has to be a power of two for tl.arange.
        if triton.next_power_of_2(blocks_per_seq) != blocks_per_seq:
            return False
        # One program stages every block of a sequence at once, and the tile
        # shape is fixed by the block table rather than chosen, so a sequence
        # whose blocks do not fit in UB has to use the generic kernel.
        tile_bytes = blocks_per_seq * cache.size(1) * cache.size(2) * (
            cache.element_size()
        )
        if tile_bytes > UB_BYTES // 2:
            return False
        if block_table.stride(1) != 1:
            return False
        return bool(
            cache.is_contiguous()
            and block_table.dtype in (torch.int32, torch.int64)
        )

    def paged_kv_block_gather(cache, block_table):
        if not _tv_eligible(cache, block_table):
            return _paged_kv_block_gather_generic(cache, block_table)
        return tensorview_call(
            "PAGED_KV_BLOCK_GATHER",
            _paged_kv_block_gather_tensorview,
            _paged_kv_block_gather_generic,
            cache,
            block_table,
        )

else:
    paged_kv_block_gather = _paged_kv_block_gather_generic
