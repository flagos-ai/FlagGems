import logging

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def reshape_and_cache_kernel(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
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
    slot_idx = tl.load(slot_mapping + token_idx)
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

    tgt_key_idx = (
        block_idx * num_heads * (head_size // x) * block_size * x
        + head_idx * (head_size // x) * block_size * x
        + x_idx * block_size * x
        + block_offset * x
        + x_offset
    )
    tgt_value_idx = (
        block_idx * num_heads * head_size * block_size
        + head_idx * head_size * block_size
        + head_offset * block_size
        + block_offset
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
    logger.debug("GEMS RESHAPE_AND_CACHE")
    num_tokens = slot_mapping.size(0)
    num_heads = key.size(1)
    head_size = key.size(2)
    block_size = key_cache.size(3)
    x = key_cache.size(4)

    key_stride = key.stride(0)
    value_stride = value.stride(0)

    grid = (num_tokens,)
    with torch_device_fn.device(key.device):
        reshape_and_cache_kernel[grid](
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
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
