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

"""Ascend TensorView override for compute_global_topk_indices_and_lens.

Map block-local top-k indices to global cache slots. Token ``t`` belongs to
request ``token_to_req[t]``; a non-negative local index ``i`` selects the slot
``block_table[req, i // block_size] * block_size + i % block_size``, and a
negative index stays ``-1``. The second output counts the non-negative indices
of every token and is 0 for tokens whose validity flag is 0.

Each program handles BLOCK_T tokens. The top-k rows, the request ids and the
validity flags come in through PartitionViews and both outputs go out through
PartitionViews, so tail tokens and the columns past the top-k width fall outside
the views and are dropped on store. The block table lookup is the only
data-dependent access and stays a masked load.
"""

import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.deprecated.deepseek_v4_attention_compute_global_topk_indices_and_lens import (  # noqa: E501
    compute_global_topk_indices_and_lens as _compute_global_topk_generic,
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

TV_MAX_BLOCK_T = 32
# Keep enough programs in flight to fill the cores rather than handing one
# program a tile so tall that most cores sit idle.
MIN_PROGRAMS = 32

if HAS_TENSORVIEW:

    @triton.jit
    def compute_global_topk_tv_kernel(
        global_indices,
        lens,
        local_indices,
        token_to_req,
        block_table,
        is_valid,
        num_tokens,
        block_table_stride,
        block_size,
        TOPK: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        tile = tl.program_id(0)
        rows = tile * BLOCK_T + tl.arange(0, BLOCK_T)
        cols = tl.arange(0, BLOCK_K)

        local_view = tle_exp.make_partition_view(
            local_indices,
            shape=[num_tokens, TOPK],
            strides=[TOPK, 1],
            tile=[BLOCK_T, BLOCK_K],
        )
        req_view = tle_exp.make_partition_view(
            token_to_req, shape=[num_tokens], strides=[1], tile=[BLOCK_T]
        )
        flag_view = tle_exp.make_partition_view(
            is_valid, shape=[num_tokens], strides=[1], tile=[BLOCK_T]
        )
        local = tile_load(
            local_view, index=(tile, 0), dst_space=tle_exp.language.dsa.ascend.UB
        )
        req = tile_load(
            req_view, index=(tile,), dst_space=tle_exp.language.dsa.ascend.UB
        )
        token_valid = tile_load(
            flag_view, index=(tile,), dst_space=tle_exp.language.dsa.ascend.UB
        )

        inside = (rows < num_tokens)[:, None] & (cols < TOPK)[None, :]
        valid = inside & (local >= 0)
        block_idx = local // block_size
        block_off = local - block_idx * block_size
        block_no = tl.load(
            block_table + req[:, None] * block_table_stride + block_idx,
            mask=valid,
            other=0,
        )
        slot = tl.where(valid, block_no * block_size + block_off, -1)

        out_view = tle_exp.make_partition_view(
            global_indices,
            shape=[num_tokens, TOPK],
            strides=[TOPK, 1],
            tile=[BLOCK_T, BLOCK_K],
        )
        tile_store(
            out_view,
            value=slot,
            index=(tile, 0),
            src_space=tle_exp.language.dsa.ascend.UB,
        )

        count = tl.sum(valid.to(tl.int32), axis=1)
        lens_view = tle_exp.make_partition_view(
            lens, shape=[num_tokens], strides=[1], tile=[BLOCK_T]
        )
        tile_store(
            lens_view,
            value=tl.where(token_valid != 0, count, 0),
            index=(tile,),
            src_space=tle_exp.language.dsa.ascend.UB,
        )

    def _tv_rows_that_fit(block_k):
        """Tokens per tile UB can hold. The loaded top-k tile and the computed
        slot row are live as int32, so budget for two."""
        return tile_rows_for_ub(block_k * 4, 2, TV_MAX_BLOCK_T)

    def _tv_block_t(block_k, num_tokens):
        rows = max(_tv_rows_that_fit(block_k), MIN_TILE_ROWS)
        while rows > MIN_TILE_ROWS and triton.cdiv(num_tokens, rows) < MIN_PROGRAMS:
            rows //= 2
        return rows

    def _compute_global_topk_tensorview(
        topk_indices,
        token_to_req_indices,
        block_table,
        block_size,
        is_valid_token=None,
    ):
        logger.debug("GEMS_ASCEND COMPUTE_GLOBAL_TOPK_INDICES_AND_LENS(TENSORVIEW)")
        topk_indices = topk_indices.contiguous()
        num_tokens, topk = topk_indices.shape
        if is_valid_token is None:
            is_valid_token = torch.ones(
                (num_tokens,), dtype=torch.int32, device=topk_indices.device
            )
        block_k = triton.next_power_of_2(topk)
        block_t = _tv_block_t(block_k, num_tokens)

        global_indices = torch.empty_like(topk_indices, dtype=torch.int32)
        lens = torch.empty((num_tokens,), dtype=torch.int32, device=topk_indices.device)
        grid = (triton.cdiv(num_tokens, block_t),)
        with torch_device_fn.device(topk_indices.device):
            compute_global_topk_tv_kernel[grid](
                global_indices,
                lens,
                topk_indices,
                token_to_req_indices,
                block_table,
                is_valid_token,
                num_tokens,
                block_table.stride(0),
                block_size,
                TOPK=topk,
                BLOCK_T=block_t,
                BLOCK_K=block_k,
            )
        return global_indices, lens

    def _tv_eligible(
        topk_indices, token_to_req_indices, block_table, block_size, is_valid_token
    ):
        if topk_indices.ndim != 2 or block_table.ndim != 2:
            return False
        if topk_indices.numel() == 0 or block_table.numel() == 0:
            return False
        if block_size <= 0:
            return False
        num_tokens, topk = topk_indices.shape
        if topk == 0:
            return False
        # The token views are 1-D runs over the tokens of this tile.
        if token_to_req_indices.numel() != num_tokens:
            return False
        if not token_to_req_indices.is_contiguous():
            return False
        if is_valid_token is not None and (
            is_valid_token.numel() != num_tokens or not is_valid_token.is_contiguous()
        ):
            return False
        # The views address rows by tile index, so the top-k rows have to be a
        # packed [tokens, width] run.
        if topk_indices.stride(1) != 1:
            return False
        if block_table.stride(1) != 1:
            return False
        # The output view is int32 and shares the input's shape; a differently
        # typed input would need its own view geometry.
        if topk_indices.dtype != torch.int32:
            return False
        return _tv_rows_that_fit(triton.next_power_of_2(topk)) >= MIN_TILE_ROWS

    def compute_global_topk_indices_and_lens(
        topk_indices: torch.Tensor,
        token_to_req_indices: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        is_valid_token: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not _tv_eligible(
            topk_indices,
            token_to_req_indices,
            block_table,
            block_size,
            is_valid_token,
        ):
            return _compute_global_topk_generic(
                topk_indices,
                token_to_req_indices,
                block_table,
                block_size,
                is_valid_token,
            )
        return tensorview_call(
            "COMPUTE_GLOBAL_TOPK_INDICES_AND_LENS",
            _compute_global_topk_tensorview,
            _compute_global_topk_generic,
            topk_indices,
            token_to_req_indices,
            block_table,
            block_size,
            is_valid_token,
        )

else:
    compute_global_topk_indices_and_lens = _compute_global_topk_generic
