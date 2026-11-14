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

"""Ascend TensorView override for reshape_and_cache_flash.

Both cache layouts flatten to one contiguous row per slot, so writing a token
into its slot is a row scatter: a PartitionView reads a tile of token rows and a
GatherScatterView writes them to the slots named by ``slot_mapping``. The
generic kernel instead launches one program per token and addresses the cache
element-wise.

The fp8 path and any cache whose block stride is padded fall back.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.fused.reshape_and_cache_flash import (
    reshape_and_cache_flash as _reshape_and_cache_flash_generic,
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


def _tv_block_t(row, elem_size):
    """Tokens per tile: the key and value tiles are both live at once."""
    return tile_rows_for_ub(
        triton.next_power_of_2(row) * elem_size, 2, TV_BLOCK_T
    )


if HAS_TENSORVIEW:

    @triton.jit
    def reshape_and_cache_flash_tv_kernel(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        num_tokens,
        num_slots,
        key_stride,
        value_stride,
        ROW: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_ROW: tl.constexpr,
    ):
        token_block = tl.program_id(0)
        tokens = token_block * BLOCK_T + tl.arange(0, BLOCK_T)
        # A negative slot means "drop this token"; the scatter view leaves any
        # out-of-range row untouched, which is the same thing.
        slots = tl.load(slot_mapping + tokens, mask=tokens < num_tokens, other=-1).to(
            tl.int32
        )

        key_view = tle_exp.make_partition_view(
            key,
            shape=[num_tokens, ROW],
            strides=[key_stride, 1],
            tile=[BLOCK_T, BLOCK_ROW],
        )
        value_view = tle_exp.make_partition_view(
            value,
            shape=[num_tokens, ROW],
            strides=[value_stride, 1],
            tile=[BLOCK_T, BLOCK_ROW],
        )
        key_tile = tile_load(
            key_view, index=(token_block, 0), dst_space=tle_exp.language.dsa.ascend.UB
        )
        value_tile = tile_load(
            value_view,
            index=(token_block, 0),
            dst_space=tle_exp.language.dsa.ascend.UB,
        )

        key_cache_view = tle_exp.make_gather_scatter_view(
            key_cache,
            shape=[num_slots, ROW],
            strides=[ROW, 1],
            tile=[BLOCK_T, BLOCK_ROW],
            sparse_dim=[0],
        )
        value_cache_view = tle_exp.make_gather_scatter_view(
            value_cache,
            shape=[num_slots, ROW],
            strides=[ROW, 1],
            tile=[BLOCK_T, BLOCK_ROW],
            sparse_dim=[0],
        )
        tile_store(
            key_cache_view,
            value=key_tile,
            index=(slots, 0),
            src_space=tle_exp.language.dsa.ascend.UB,
        )
        tile_store(
            value_cache_view,
            value=value_tile,
            index=(slots, 0),
            src_space=tle_exp.language.dsa.ascend.UB,
        )

    def _reshape_and_cache_flash_tensorview(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
        block_t=None,
    ):
        logger.debug("GEMS_ASCEND RESHAPE_AND_CACHE_FLASH(TENSORVIEW)")
        num_tokens = slot_mapping.size(0)
        row = key.size(1) * key.size(2)
        block_t = _tv_block_t(row, key.element_size()) if block_t is None else block_t
        num_slots = key_cache.size(0) * key_cache.size(1)
        grid = (triton.cdiv(num_tokens, block_t),)
        with torch_device_fn.device(key.device):
            reshape_and_cache_flash_tv_kernel[grid](
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                num_tokens,
                num_slots,
                key.stride(0),
                value.stride(0),
                ROW=row,
                BLOCK_T=block_t,
                BLOCK_ROW=triton.next_power_of_2(row),
            )
        return None

    def _tv_eligible(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype):
        # fp8 needs the scales applied per element; that is the generic path.
        if kv_cache_dtype != "auto":
            return False
        if key_cache.dtype != key.dtype or value_cache.dtype != value.dtype:
            return False
        if key.ndim != 3 or value.ndim != 3:
            return False
        if slot_mapping.numel() == 0 or key.numel() == 0:
            return False
        row = key.size(1) * key.size(2)
        # The scatter view assumes slot rows are packed with stride `row`, so a
        # cache whose block stride is padded is not describable this way.
        if key_cache.stride(0) != key_cache.size(1) * row:
            return False
        if value_cache.stride(0) != value_cache.size(1) * row:
            return False
        if key_cache.stride(0) != value_cache.stride(0):
            return False
        # The innermost two axes of each cache row must themselves be packed.
        if key_cache.stride(1) != row or value_cache.stride(1) != row:
            return False
        return bool(
            key.stride(2) == 1
            and value.stride(2) == 1
            and key.stride(1) == key.size(2)
            and value.stride(1) == value.size(2)
            and slot_mapping.dtype in (torch.int32, torch.int64)
        )

    def reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    ):
        if not _tv_eligible(
            key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype
        ):
            return _reshape_and_cache_flash_generic(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                kv_cache_dtype,
                k_scale,
                v_scale,
            )
        return tensorview_call(
            "RESHAPE_AND_CACHE_FLASH",
            _reshape_and_cache_flash_tensorview,
            _reshape_and_cache_flash_generic,
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )

else:
    reshape_and_cache_flash = _reshape_and_cache_flash_generic
