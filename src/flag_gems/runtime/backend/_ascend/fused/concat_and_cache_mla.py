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

"""Ascend TensorView override for concat_and_cache_mla.

Each cache row is ``kv_lora_rank + pe_dim`` wide and holds the two inputs side
by side, so caching a token is two row scatters into the slot named by
``slot_mapping``: one tile of RANK columns at dense index 0, and one tile of
ROPE columns at dense index ``RANK // ROPE``.

Only the ``kv_cache_dtype == "auto"`` path. The fp8 modes quantize by ``scale``
per element and stay on the generic implementation.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.fused.concat_and_cache_mla import (
    concat_and_cache_mla as _concat_and_cache_mla_generic,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._ascend.tensorview import (
    HAS_TENSORVIEW,
    tensorview_call,
    tile_rows_for_ub,
)

logger = logging.getLogger(__name__)

if HAS_TENSORVIEW:
    import triton.experimental.tle as tle_exp
    from triton.experimental.tle.language.dsa import tile_load, tile_store

TV_BLOCK_T = 32

if HAS_TENSORVIEW:

    @triton.jit
    def concat_and_cache_mla_tv_kernel(
        kv_c,
        k_pe,
        kv_cache,
        slot_mapping,
        num_tokens,
        num_slots,
        kv_c_stride,
        k_pe_stride,
        RANK: tl.constexpr,
        ROPE: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        token_block = tl.program_id(0)
        tokens = token_block * BLOCK_T + tl.arange(0, BLOCK_T)
        slots = tl.load(slot_mapping + tokens, mask=tokens < num_tokens, other=-1).to(
            tl.int32
        )

        kv_c_view = tle_exp.make_partition_view(
            kv_c,
            shape=[num_tokens, RANK],
            strides=[kv_c_stride, 1],
            tile=[BLOCK_T, RANK],
        )
        k_pe_view = tle_exp.make_partition_view(
            k_pe,
            shape=[num_tokens, ROPE],
            strides=[k_pe_stride, 1],
            tile=[BLOCK_T, ROPE],
        )
        kv_c_tile = tile_load(
            kv_c_view, index=(token_block, 0), dst_space=tle_exp.language.dsa.ascend.UB
        )
        k_pe_tile = tile_load(
            k_pe_view, index=(token_block, 0), dst_space=tle_exp.language.dsa.ascend.UB
        )

        # One view per half, differing only in tile width. The rope half starts
        # at dense tile index RANK // ROPE, which is why ROPE must divide RANK.
        kv_c_cache = tle_exp.make_gather_scatter_view(
            kv_cache,
            shape=[num_slots, RANK + ROPE],
            strides=[RANK + ROPE, 1],
            tile=[BLOCK_T, RANK],
            sparse_dim=[0],
        )
        k_pe_cache = tle_exp.make_gather_scatter_view(
            kv_cache,
            shape=[num_slots, RANK + ROPE],
            strides=[RANK + ROPE, 1],
            tile=[BLOCK_T, ROPE],
            sparse_dim=[0],
        )
        tile_store(
            kv_c_cache,
            value=kv_c_tile,
            index=(slots, 0),
            src_space=tle_exp.language.dsa.ascend.UB,
        )
        tile_store(
            k_pe_cache,
            value=k_pe_tile,
            index=(slots, RANK // ROPE),
            src_space=tle_exp.language.dsa.ascend.UB,
        )

    def _concat_and_cache_mla_tensorview(
        kv_c,
        k_pe,
        kv_cache,
        slot_mapping,
        kv_cache_dtype,
        scale,
        block_t=None,
    ):
        logger.debug("GEMS_ASCEND CONCAT_AND_CACHE_MLA(TENSORVIEW)")
        num_tokens = slot_mapping.size(0)
        rank = kv_c.size(1)
        rope = k_pe.size(1)
        if block_t is None:
            # The kv_c and k_pe tiles are both live before the two stores.
            block_t = tile_rows_for_ub(
                (rank + rope) * kv_c.element_size(), 1, TV_BLOCK_T
            )
        num_slots = kv_cache.size(0) * kv_cache.size(1)
        grid = (triton.cdiv(num_tokens, block_t),)
        with torch_device_fn.device(kv_c.device):
            concat_and_cache_mla_tv_kernel[grid](
                kv_c,
                k_pe,
                kv_cache,
                slot_mapping,
                num_tokens,
                num_slots,
                kv_c.stride(0),
                k_pe.stride(0),
                RANK=rank,
                ROPE=rope,
                BLOCK_T=block_t,
            )
        return None

    def _tv_eligible(kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype):
        if kv_cache_dtype != "auto":
            return False
        if kv_cache.dtype != kv_c.dtype or k_pe.dtype != kv_c.dtype:
            return False
        if kv_c.ndim != 2 or k_pe.ndim != 2 or kv_cache.ndim != 3:
            return False
        if slot_mapping.numel() == 0 or kv_c.numel() == 0:
            return False
        rank, rope = kv_c.size(1), k_pe.size(1)
        if rope == 0 or rank % rope != 0:
            # The rope half must start on a tile boundary of the narrow tile.
            return False
        if kv_cache.size(2) != rank + rope:
            return False
        # The scatter view treats the cache as [num_slots, rank+rope] packed
        # rows, so the block and slot strides must match that exactly.
        if kv_cache.stride(2) != 1 or kv_cache.stride(1) != rank + rope:
            return False
        if kv_cache.stride(0) != kv_cache.size(1) * (rank + rope):
            return False
        return bool(
            kv_c.stride(1) == 1
            and k_pe.stride(1) == 1
            and slot_mapping.dtype in (torch.int32, torch.int64)
        )

    def concat_and_cache_mla(
        kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale
    ):
        if not _tv_eligible(kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype):
            return _concat_and_cache_mla_generic(
                kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale
            )
        return tensorview_call(
            "CONCAT_AND_CACHE_MLA",
            _concat_and_cache_mla_tensorview,
            _concat_and_cache_mla_generic,
            kv_c,
            k_pe,
            kv_cache,
            slot_mapping,
            kv_cache_dtype,
            scale,
        )

else:
    concat_and_cache_mla = _concat_and_cache_mla_generic
