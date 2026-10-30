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

"""Ascend TensorView override for cp_gather_indexer_k_quant_cache.

STATUS: the TensorView path is written but **disabled** (``_TV_ENABLED``), so
this module currently just forwards to the generic implementation. The kernel
faults on device ("vector core exception") even for a single aligned sequence
with no partial tail, and the cause was not isolated. What was ruled out:
the padded-row view itself lowers and reads correctly in isolation; the row
indices are within the declared extent; the scale slice geometry is consistent;
and the gather half alone faults with the scale store removed, so it is not the
scale arithmetic. Of the ten shapes the upstream test exercises, nine are
ineligible anyway -- the inline fp32 scales make the block-row stride a
non-multiple of ``head_dim``, so the row cannot be addressed as a tile index --
which leaves this op a poor fit for a TensorView rewrite regardless.

The value half of this op is a row gather: every token resolves to one
``HEAD_DIM``-wide row of the paged cache, so a GatherScatterView moves it as a
block DMA and a PartitionView writes the consecutive destination rows. The scale
half stays element-wise -- it is a handful of values per token, not a shape a
block DMA helps with.

Token-to-row resolution keeps the generic kernel's two strategies: a direct
scan when the batch is small, a binary search over ``cu_seqlen`` otherwise.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.fused.cp_gather_indexer_k_quant_cache import (
    cp_gather_indexer_k_quant_cache as _cp_gather_generic,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._ascend.tensorview import (
    HAS_TENSORVIEW,
    tensorview_call,
)

logger = logging.getLogger(__name__)

if HAS_TENSORVIEW:
    import triton.experimental.tle as tle_exp
    from triton.experimental.tle.language.dsa import tile_load, tile_store

if HAS_TENSORVIEW:

    @triton.jit
    def cp_gather_indexer_k_quant_cache_tv_kernel(
        kv_cache,
        kv_cache_scale,
        k_fp8,
        k_scale,
        block_table,
        cu_seqlen,
        num_tokens,
        cache_tiles,
        block_size,
        block_table_stride,
        kv_cache_scale_stride,
        num_quant_blocks,
        batch_size,
        HEAD_DIM: tl.constexpr,
        ROW_TILES: tl.constexpr,
        QUANT_SLOTS: tl.constexpr,
        BATCH_SCAN_SIZE: tl.constexpr,
        SEARCH_STEPS: tl.constexpr,
        SMALL_BATCH: tl.constexpr,
        TOKEN_BLOCK: tl.constexpr,
    ):
        token_block = tl.program_id(0)
        tid = token_block * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
        in_range = tid < num_tokens

        if SMALL_BATCH:
            # Few sequences: compare every token against every interval at once.
            batch_offsets = tl.arange(0, BATCH_SCAN_SIZE)
            batch_mask = batch_offsets < batch_size
            seq_starts = tl.load(
                cu_seqlen + batch_offsets, mask=batch_mask, other=0
            )
            seq_ends = tl.load(
                cu_seqlen + batch_offsets + 1, mask=batch_mask, other=0
            )
            in_batch = (
                (tid[:, None] >= seq_starts[None, :])
                & (tid[:, None] < seq_ends[None, :])
                & batch_mask[None, :]
            )
            batch_id = tl.max(tl.where(in_batch, batch_offsets[None, :], -1), axis=1)
        else:
            left = tl.zeros((TOKEN_BLOCK,), dtype=tl.int32)
            right = tl.full((TOKEN_BLOCK,), batch_size + 1, dtype=tl.int32)
            for _ in tl.static_range(0, SEARCH_STEPS):
                mid = (left + right) // 2
                seq_start = tl.load(
                    cu_seqlen + mid, mask=mid <= batch_size, other=2147483647
                )
                before = seq_start <= tid
                left = tl.where(before, mid + 1, left)
                right = tl.where(before, right, mid)
            batch_id = left - 1

        valid_batch = (batch_id >= 0) & (batch_id < batch_size)
        safe_batch_id = tl.minimum(tl.maximum(batch_id, 0), batch_size - 1)
        batch_start = tl.load(cu_seqlen + safe_batch_id, mask=valid_batch, other=0)
        batch_end = tl.load(cu_seqlen + safe_batch_id + 1, mask=valid_batch, other=0)
        valid = in_range & valid_batch & (tid >= batch_start) & (tid < batch_end)

        batch_offset = tid - batch_start
        block_id = tl.load(
            block_table
            + safe_batch_id * block_table_stride
            + batch_offset // block_size,
            mask=valid,
            other=0,
        )
        block_offset = batch_offset % block_size
        # Each block row carries its inline fp32 scales after the values, so the
        # dim-0 stride is the whole padded row rather than block_size*HEAD_DIM.
        # ROW_TILES is that padded stride measured in HEAD_DIM-wide tiles, which
        # the eligibility check has already confirmed is exact, so a token's
        # value row is tile number block*ROW_TILES + block_offset.
        # An invalid token maps to -1, which the view drops instead of clamping.
        rows = tl.where(valid, block_id * ROW_TILES + block_offset, -1).to(tl.int32)

        cache_view = tle_exp.make_gather_scatter_view(
            kv_cache,
            shape=[cache_tiles, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            tile=[TOKEN_BLOCK, HEAD_DIM],
            sparse_dim=[0],
        )
        gathered = tile_load(
            cache_view, index=(rows, 0), dst_space=tle_exp.language.dsa.ascend.UB
        )
        out_view = tle_exp.make_partition_view(
            k_fp8,
            shape=[num_tokens, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            tile=[TOKEN_BLOCK, HEAD_DIM],
        )
        tile_store(
            out_view,
            value=gathered,
            index=(token_block, 0),
            src_space=tle_exp.language.dsa.ascend.UB,
        )

        # Scales: a few float32 values per token, left element-wise.
        quant = tl.arange(0, QUANT_SLOTS)
        quant_mask = quant < num_quant_blocks
        scale_mask = valid[:, None] & quant_mask[None, :]
        scale_val = tl.load(
            kv_cache_scale
            + block_id[:, None] * kv_cache_scale_stride
            + block_offset[:, None] * num_quant_blocks
            + quant[None, :],
            mask=scale_mask,
            other=0.0,
        )
        tl.store(
            k_scale + tid[:, None] * num_quant_blocks + quant[None, :],
            scale_val,
            mask=scale_mask,
        )

    def _cp_gather_tensorview(k_cache, k_fp8, k_fp8_scale, block_table, cu_seqlen):
        logger.debug("GEMS_ASCEND CP_GATHER_INDEXER_K_QUANT_CACHE(TENSORVIEW)")
        num_tokens = k_fp8.size(0)
        block_size = k_cache.size(1)
        head_dim = k_fp8.shape[-1]
        num_blocks = k_cache.shape[0]
        quant_block_size = head_dim * 4 // k_fp8_scale.size(1)
        if head_dim % quant_block_size != 0:
            raise ValueError("head_dim must be divisible by quant_block_size")
        num_quant_blocks = head_dim // quant_block_size

        # Same split as the generic path: the value bytes sit first in each
        # cache row, the float32 scales after them.
        k_cache_flat = k_cache.view(num_blocks, -1)
        k_cache_scale = k_cache_flat[:, block_size * head_dim :].view(torch.float32)
        k_fp8_u8 = k_fp8.view(torch.uint8)
        k_fp8_scale_f32 = k_fp8_scale.view(torch.float32)

        batch_size = block_table.shape[0]
        token_block = 32
        small_batch = batch_size <= 16

        with torch_device_fn.device(k_cache.device):
            cp_gather_indexer_k_quant_cache_tv_kernel[
                (triton.cdiv(num_tokens, token_block),)
            ](
                # The view is built on the whole cache, not the value slice:
                # addressing a row by tile index has to step over the inline
                # scales, and only the full allocation actually owns those
                # bytes. The slice would be overrun by exactly the padding.
                k_cache_flat,
                k_cache_scale,
                k_fp8_u8,
                k_fp8_scale_f32,
                block_table,
                cu_seqlen,
                num_tokens,
                num_blocks * (k_cache_flat.stride(0) // head_dim),
                block_size,
                block_table.stride(0),
                k_cache_scale.stride(0),
                num_quant_blocks,
                batch_size,
                HEAD_DIM=head_dim,
                ROW_TILES=k_cache_flat.stride(0) // head_dim,
                QUANT_SLOTS=triton.next_power_of_2(num_quant_blocks),
                BATCH_SCAN_SIZE=(
                    triton.next_power_of_2(batch_size) if small_batch else 1
                ),
                SEARCH_STEPS=batch_size.bit_length(),
                SMALL_BATCH=small_batch,
                TOKEN_BLOCK=token_block,
            )
        return None

    # Disabled pending a fix -- see the module docstring. The kernel and gate
    # are kept so finishing it is a localized debug, not a rewrite.
    _TV_ENABLED = False

    def _tv_eligible(k_cache, k_fp8, k_fp8_scale, block_table, cu_seqlen):
        if not _TV_ENABLED:
            return False
        if k_cache.ndim < 2 or block_table.ndim != 2:
            return False
        if k_fp8.numel() == 0 or block_table.numel() == 0:
            return False
        if not (k_cache.is_contiguous() and k_fp8.is_contiguous()):
            return False
        head_dim = k_fp8.shape[-1]
        if head_dim == 0 or k_fp8_scale.size(1) == 0:
            return False
        # The destination view assumes packed [num_tokens, head_dim] rows.
        if k_fp8.ndim != 2 or k_fp8.stride(0) != head_dim or k_fp8.stride(1) != 1:
            return False
        quant_block_size = head_dim * 4 // k_fp8_scale.size(1)
        if quant_block_size == 0 or head_dim % quant_block_size != 0:
            return False
        # A token's value row is addressed as a HEAD_DIM-wide tile index, so the
        # padded block-row stride has to be a whole number of those tiles. It
        # usually is not -- the inline scales are only head_dim*4//quant_block
        # bytes per slot -- and those layouts stay on the generic path.
        block_size = k_cache.size(1)
        scale_bytes = head_dim * 4 // quant_block_size
        row_stride = block_size * head_dim + block_size * scale_bytes
        if row_stride % head_dim != 0:
            return False
        if k_cache.numel() % row_stride != 0:
            return False
        if block_table.stride(1) != 1:
            return False
        return bool(
            block_table.dtype in (torch.int32, torch.int64)
            and cu_seqlen.dtype in (torch.int32, torch.int64)
        )

    def cp_gather_indexer_k_quant_cache(
        k_cache: torch.Tensor,
        k_fp8: torch.Tensor,
        k_fp8_scale: torch.Tensor,
        block_table: torch.Tensor,
        cu_seqlen: torch.Tensor,
    ):
        if not _tv_eligible(k_cache, k_fp8, k_fp8_scale, block_table, cu_seqlen):
            return _cp_gather_generic(
                k_cache, k_fp8, k_fp8_scale, block_table, cu_seqlen
            )
        return tensorview_call(
            "CP_GATHER_INDEXER_K_QUANT_CACHE",
            _cp_gather_tensorview,
            _cp_gather_generic,
            k_cache,
            k_fp8,
            k_fp8_scale,
            block_table,
            cu_seqlen,
        )

else:
    cp_gather_indexer_k_quant_cache = _cp_gather_generic
