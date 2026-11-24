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

"""Ascend TensorView override for combine_topk_swa_indices.

Build the per-token index row that joins the compressed top-k entries with the
sliding-window entries. Token ``t`` of request ``b`` sits at position ``pos`` in
its sequence; its row holds, in order, the first
``topk_len = min((pos + 1) // compress_ratio, topk)`` top-k indices shifted by
``M * b``, then ``swa_len = min(pos + 1, window_size)`` consecutive window
indices, then ``-1`` up to the padded row width.

Each program handles one tile of BLOCK_T consecutive tokens of one request. The
top-k rows come in through a StridedView and the combined rows go out through a
StridedView whose shape ends at the last token of the request, so the tokens of
a tile that lie past the request are dropped on store. A tile that starts past
the end of its request loads from the request start instead, which keeps every
load on the whole-tile DMA path; all rows of such a tile are dropped on store.
The whole padded row, ``-1`` filler included, is computed in the tile, so the
output needs no separate fill pass.
"""

import logging
from typing import Tuple

import torch
import triton
import triton.language as tl

from flag_gems.deprecated.deepseek_v4_attention_combine_topk_swa_indices import (
    combine_topk_swa_indices as _combine_topk_swa_indices_generic,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._ascend.tensorview import (
    HAS_TENSORVIEW,
    MIN_TILE_ROWS,
    tensorview_call,
    tile_rows_for_ub,
)

logger = logging.getLogger(__name__)

if HAS_TENSORVIEW:
    import triton.experimental.tle as tle_exp
    from triton.experimental.tle import tile_load, tile_store

# The padded row width is a multiple of this, matching the generic path.
ROW_ALIGNMENT = 128
TV_MAX_BLOCK_T = 32
# Programs per request. The generic kernel launches the same number and strides
# its token loop over them; here each one strides over whole tiles.
TV_WORKERS = 128

if HAS_TENSORVIEW:

    @triton.jit
    def combine_topk_swa_tv_kernel(
        combined,
        lens,
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        num_tokens,
        offset_m,
        offset_n,
        TOPK: tl.constexpr,
        COMBINED_TOPK: tl.constexpr,
        COMPRESS_RATIO: tl.constexpr,
        WINDOW_SIZE: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        batch = tl.program_id(0)
        worker = tl.program_id(1)
        num_workers = tl.num_programs(1)

        base = tl.load(query_start_loc)
        query_start = tl.load(query_start_loc + batch) - base
        query_end = tl.load(query_start_loc + batch + 1) - base
        seq_len = tl.load(seq_lens + batch)
        gather_len = tl.load(gather_lens + batch)
        start_pos = seq_len - (query_end - query_start)
        gather_start = seq_len - gather_len

        # The query lengths live on device, so the host cannot size the grid to
        # the longest request without reading them back. Launch a fixed worker
        # count instead and let each worker stride over the tiles of its
        # request, the way the generic kernel strides over tokens.
        # The views do not depend on the tile index, so they are built once and
        # the loop only moves the index that addresses them.
        topk_view = tle_exp.make_strided_view(
            topk_indices,
            shape=[num_tokens, TOPK],
            strides=[TOPK, 1],
            tile=[BLOCK_T, BLOCK_C],
            traversal_strides=[1, BLOCK_C],
        )
        combined_view = tle_exp.make_strided_view(
            combined,
            shape=[query_end, COMBINED_TOPK],
            strides=[COMBINED_TOPK, 1],
            tile=[BLOCK_T, BLOCK_C],
            traversal_strides=[1, BLOCK_C],
        )
        lens_view = tle_exp.make_strided_view(
            lens,
            shape=[query_end],
            strides=[1],
            tile=[BLOCK_T],
            traversal_strides=[1],
        )

        col = tl.arange(0, BLOCK_C)[None, :]
        tiles = tl.cdiv(query_end - query_start, BLOCK_T)
        for tile in range(worker, tiles, num_workers):
            first_token = query_start + tile * BLOCK_T

            # A tile that starts past the end of the request has no in-bounds
            # row, so read from the request start and let the store drop it.
            load_token = tl.where(first_token < query_end, first_token, query_start)
            topk_vals = tile_load(
                topk_view,
                index=(load_token, 0),
                dst_space=tle_exp.language.dsa.ascend.UB,
            )

            pos = start_pos + tile * BLOCK_T + tl.arange(0, BLOCK_T)
            topk_count = tl.minimum((pos + 1) // COMPRESS_RATIO, TOPK)
            swa_count = tl.minimum(pos + 1, WINDOW_SIZE)
            topk_len = topk_count[:, None]
            swa_len = swa_count[:, None]

            window_vals = (
                batch * offset_m
                + offset_n
                + (col - topk_len)
                + pos[:, None]
                - swa_len
                + 1
                - gather_start
            )
            row = tl.where(
                col < topk_len,
                topk_vals + batch * offset_m,
                tl.where(col < topk_len + swa_len, window_vals, -1),
            )

            tile_store(
                combined_view,
                value=row,
                index=(first_token, 0),
                src_space=tle_exp.language.dsa.ascend.UB,
            )
            tile_store(
                lens_view,
                value=topk_count + swa_count,
                index=(first_token,),
                src_space=tle_exp.language.dsa.ascend.UB,
            )

    def _tv_rows_that_fit(block_c):
        """Tokens per tile UB can hold. The loaded top-k tile and the combined
        row are live as int32, so budget for two."""
        return tile_rows_for_ub(block_c * 4, 2, TV_MAX_BLOCK_T)

    def _tv_block_t(block_c):
        return max(_tv_rows_that_fit(block_c), MIN_TILE_ROWS)

    def _combined_width(topk, window_size):
        return triton.cdiv(topk + window_size, ROW_ALIGNMENT) * ROW_ALIGNMENT

    def _combine_topk_swa_tensorview(
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size,
        compress_ratio,
        topk,
        M,
        N,
    ):
        logger.debug("GEMS_ASCEND COMBINE_TOPK_SWA_INDICES(TENSORVIEW)")
        topk_indices = topk_indices.contiguous()
        num_tokens, topk_width = topk_indices.shape
        num_reqs = seq_lens.numel()
        combined_topk = _combined_width(topk, window_size)
        block_c = triton.next_power_of_2(combined_topk)
        block_t = _tv_block_t(block_c)

        combined = torch.empty(
            (num_tokens, combined_topk),
            dtype=torch.int32,
            device=topk_indices.device,
        )
        lens = torch.empty((num_tokens,), dtype=torch.int32, device=topk_indices.device)
        grid = (num_reqs, TV_WORKERS)
        with torch_device_fn.device(topk_indices.device):
            combine_topk_swa_tv_kernel[grid](
                combined,
                lens,
                topk_indices,
                query_start_loc,
                seq_lens,
                gather_lens,
                num_tokens,
                M,
                N,
                TOPK=topk_width,
                COMBINED_TOPK=combined_topk,
                COMPRESS_RATIO=compress_ratio,
                WINDOW_SIZE=window_size,
                BLOCK_T=block_t,
                BLOCK_C=block_c,
            )
        return combined, lens

    def _tv_eligible(
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size,
        compress_ratio,
        topk,
    ):
        if topk_indices.ndim != 2 or topk_indices.numel() == 0:
            return False
        if topk_indices.shape[1] < topk:
            return False
        if seq_lens.numel() == 0 or gather_lens.numel() != seq_lens.numel():
            return False
        if query_start_loc.numel() != seq_lens.numel() + 1:
            return False
        # The tile is addressed by token index, so the top-k rows have to be a
        # packed [tokens, width] run.
        if topk_indices.stride(1) != 1:
            return False
        # The kernel divides the position by compress_ratio.
        if window_size <= 0 or topk <= 0 or compress_ratio <= 0:
            return False
        block_c = triton.next_power_of_2(_combined_width(topk, window_size))
        return _tv_rows_that_fit(block_c) >= MIN_TILE_ROWS

    def combine_topk_swa_indices(
        topk_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        gather_lens: torch.Tensor,
        window_size: int,
        compress_ratio: int,
        topk: int,
        M: int,
        N: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not _tv_eligible(
            topk_indices,
            query_start_loc,
            seq_lens,
            gather_lens,
            window_size,
            compress_ratio,
            topk,
        ):
            return _combine_topk_swa_indices_generic(
                topk_indices,
                query_start_loc,
                seq_lens,
                gather_lens,
                window_size,
                compress_ratio,
                topk,
                M,
                N,
            )
        return tensorview_call(
            "COMBINE_TOPK_SWA_INDICES",
            _combine_topk_swa_tensorview,
            _combine_topk_swa_indices_generic,
            topk_indices,
            query_start_loc,
            seq_lens,
            gather_lens,
            window_size,
            compress_ratio,
            topk,
            M,
            N,
        )

else:
    combine_topk_swa_indices = _combine_topk_swa_indices_generic
