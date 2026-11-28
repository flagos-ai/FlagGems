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
import os

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def _slot_word_stride() -> int:
    # Consistent with reshape_and_cache_flash: when the int64 gate is off, torch_gcu compresses aten int64
    # into compact int32 (element j lands in word j); with the gate on, each element takes 8 bytes with its low word in
    # word 2j.
    # Reading uniformly through an int32 view yields the correct slot value in both environment states.
    return 2 if os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64") == "1" else 1


def _slot_words(slot_mapping):
    if slot_mapping.dtype == torch.int32:
        return slot_mapping, 1
    if slot_mapping.dtype != torch.int64:
        raise NotImplementedError(
            "reshape_and_cache supports int32/int64 slot_mapping, got "
            f"{slot_mapping.dtype}"
        )
    return slot_mapping.contiguous().view(torch.int32), _slot_word_stride()


@libentry()
@triton.jit
def reshape_and_cache_kernel(
    key,
    value,
    key_cache,
    value_cache,
    slot_words,
    slot_word_stride,
    key_stride,
    value_stride,
    num_heads,
    head_size,
    block_size,
    x,
    k_scale,
    v_scale,
    n: tl.constexpr,
):
    token_idx = tl.program_id(0)
    slot_idx = tl.load(slot_words + token_idx * slot_word_stride)
    if slot_idx < 0:
        return

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    i = tl.arange(0, triton.next_power_of_2(n))
    mask = i < n

    src_key_idx = token_idx * key_stride + i
    src_value_idx = token_idx * value_stride + i
    head_idx = i // head_size
    head_offset = i % head_size
    x_idx = head_offset // x
    x_offset = head_offset % x

    # The number of paged KV cache elements can exceed 2**31; fold scalar offset terms into an i64 expression first to
    # avoid i32
    # truncation in the addptr chain.
    key_cache_base = (
        block_idx.to(tl.int64) * num_heads * (head_size // x) * block_size * x
    )
    value_cache_base = block_idx.to(tl.int64) * num_heads * head_size * block_size

    tgt_key_idx = key_cache_base + (
        head_idx * (head_size // x) * block_size * x
        + x_idx * block_size * x
        + block_offset * x
        + x_offset
    )
    tgt_value_idx = value_cache_base + (
        head_idx * head_size * block_size + head_offset * block_size + block_offset
    )

    tgt_key = tl.load(key + src_key_idx, mask=mask)
    tgt_value = tl.load(value + src_value_idx, mask=mask)

    # TODO: support fp8 dtype
    tl.store(key_cache + tgt_key_idx, tgt_key, mask=mask)
    tl.store(value_cache + tgt_value_idx, tgt_value, mask=mask)


def reshape_and_cache(
    key,  # [num_tokens, num_heads, head_size]
    value,  # [num_tokens, num_heads, head_size]
    key_cache,  # [num_blocks, num_heads, head_size/x, block_size, x]
    value_cache,  # [num_blocks, num_heads, head_size, block_size]
    slot_mapping,  # [num_tokens]
    kv_cache_dtype,
    k_scale,
    v_scale,
):
    logger.debug("GEMS_ENFLAME RESHAPE_AND_CACHE")
    num_tokens = slot_mapping.size(0)
    num_heads = key.size(1)
    head_size = key.size(2)
    block_size = key_cache.size(3)
    x = key_cache.size(4)

    key_stride = key.stride(0)
    value_stride = value.stride(0)

    slot_words, slot_word_stride = _slot_words(slot_mapping)

    grid = (num_tokens,)
    with torch_device_fn.device(key.device):
        reshape_and_cache_kernel[grid](
            key,
            value,
            key_cache,
            value_cache,
            slot_words,
            slot_word_stride,
            key_stride,
            value_stride,
            num_heads,
            head_size,
            block_size,
            x,
            k_scale,
            v_scale,
            num_heads * head_size,
        )
