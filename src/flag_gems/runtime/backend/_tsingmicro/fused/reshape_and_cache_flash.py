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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.config import use_c_extension
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def reshape_and_cache_flash_kernel(
    key_ptr,  # [num_tokens, num_heads, head_size]
    value_ptr,  # [num_tokens, num_heads, head_size]
    key_cache_ptr,  # [num_blocks, block_size, num_heads, head_size]
    value_cache_ptr,  # [num_blocks, block_size, num_heads, head_size]
    slot_mapping_ptr,  # [num_tokens]
    k_scale,  # float32
    v_scale,  # float32
    # strides
    key_stride: tl.int64,
    value_stride: tl.int64,
    block_stride: tl.int64,
    page_stride: tl.int64,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    # tune parameters
    TILE_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(axis=0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot_idx < 0:
        # Padding token that should be ignored.
        return

    tile_i = tl.program_id(axis=1)
    tile_offs = tl.arange(0, TILE_SIZE)
    tile_pos = tile_i * TILE_SIZE + tile_offs

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    src_key_idx = token_idx * key_stride
    src_value_idx = token_idx * value_stride

    tgt_idx = block_idx * block_stride + block_offset * page_stride

    # [TILE_SIZE]
    key_tile = tl.load(
        key_ptr + src_key_idx + tile_pos, mask=tile_pos < (num_heads * head_size)
    )
    # [TILE_SIZE]
    value_tile = tl.load(
        value_ptr + src_value_idx + tile_pos, mask=tile_pos < (num_heads * head_size)
    )

    tl.store(
        key_cache_ptr + tgt_idx + tile_pos,
        key_tile,
        mask=tile_pos < (num_heads * head_size),
    )
    tl.store(
        value_cache_ptr + tgt_idx + tile_pos,
        value_tile,
        mask=tile_pos < (num_heads * head_size),
    )
    return


def reshape_and_cache_flash(
    key,  # [num_tokens, num_heads, head_size]
    value,  # [num_tokens, num_heads, head_size]
    key_cache,  # [num_blocks, block_size, num_heads, head_size]
    value_cache,  # [num_blocks, block_size, num_heads, head_size]
    slot_mapping,  # [num_tokens]
    kv_cache_dtype,
    k_scale,
    v_scale,
):
    if use_c_extension:
        logger.debug("GEMS_TSINGMICRO RESHAPE_AND_CACHE_FLASH(C EXTENSION)")
        torch.ops.flag_gems.reshape_and_cache_flash(
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
        logger.debug("GEMS_TSINGMICRO RESHAPE_AND_CACHE_FLASH")
        num_heads = key.shape[1]
        head_size = key.shape[2]
        block_size = key_cache.shape[1]
        n = num_heads * head_size

        key_stride = key.stride()[0]
        value_stride = value.stride()[0]
        block_stride = key_cache.stride()[0]
        page_stride = key_cache.stride()[1]

        head_stride = key_cache.stride()[2]
        assert head_stride == head_size, "only continous heads are supported"

        assert kv_cache_dtype == "auto", (
            f"unsupported kv_cache_dtype (str), got {kv_cache_dtype}. "
            "fp8 kv cache is not supported by flag_gems reshape_and_cache_flash."
        )

        # heuristics instead of autotuning
        TILE_SIZE = min(2048, triton.next_power_of_2(n))

        # TODO(ngl): maybe replace with static launch grid to avoid overhead if
        #   using cudagraphs
        grid = lambda meta: (
            slot_mapping.shape[0],
            triton.cdiv(n, meta["TILE_SIZE"]),
        )
        with torch_device_fn.device(key.device):
            reshape_and_cache_flash_kernel[grid](
                key_ptr=key,
                value_ptr=value,
                key_cache_ptr=key_cache,
                value_cache_ptr=value_cache,
                slot_mapping_ptr=slot_mapping,
                k_scale=k_scale,
                v_scale=v_scale,
                # strides
                key_stride=key_stride,
                value_stride=value_stride,
                block_stride=block_stride,
                page_stride=page_stride,
                num_heads=num_heads,
                head_size=head_size,
                block_size=block_size,
                # autotune parameters
                TILE_SIZE=TILE_SIZE,
            )
