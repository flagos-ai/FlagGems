import random

import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES

from .performance_utils import GenericBenchmark


def torch_reshape_and_cache_ref(
    key,  # [num_tokens, num_heads, head_size]
    value,  # [num_tokens, num_heads, head_size]
    key_cache,  # [num_blocks, num_heads, head_size/x, block_size, x]
    value_cache,  # [num_blocks, num_heads, head_size, block_size]
    slot_mapping,  # [num_tokens]
    kv_cache_dtype,
    k_scale,
    v_scale,
):
    num_tokens = slot_mapping.numel()
    block_size = key_cache.size(3)
    reshaped_key = key.reshape(num_tokens, *key_cache[0, :, :, 0, :].shape)
    for i in range(num_tokens):
        slot = slot_mapping[i].item()
        block_idx = slot // block_size
        block_offset = slot % block_size
        key_cache[block_idx, :, :, block_offset, :] = reshaped_key[i]
        value_cache[block_idx, :, :, block_offset] = value[i]


class ReshapeAndCacheBenchmark(GenericBenchmark):
    """
    benchmark for reshape_and_cache
    """

    def set_more_shapes(self):
        return None


@pytest.mark.reshape_and_cache
def test_reshape_and_cache():
    def input_kwargs(shape, dtype, device):
        (
            num_tokens,
            num_heads,
            head_size,
            block_size,
            num_blocks,
        ) = shape
        num_slots = block_size * num_blocks
        slot_mapping_lst = random.sample(range(num_slots), num_tokens)
        slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long, device=device)

        qkv = torch.randn(
            num_tokens, 3, num_heads, head_size, dtype=dtype, device=device
        )
        _, key, value = qkv.unbind(dim=1)

        scale = head_size**-0.5
        x = 16 // torch.tensor([], dtype=dtype).element_size()
        key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
        key_caches: list[torch.Tensor] = []
        key_cache = torch.empty(size=key_cache_shape, dtype=dtype, device=device)
        key_cache.uniform_(-scale, scale)
        key_caches.append(key_cache)
        value_cache_shape = (num_blocks, num_heads, head_size, block_size)
        value_caches: list[torch.Tensor] = []
        value_cache = torch.empty(size=value_cache_shape, dtype=dtype, device=device)
        value_cache.uniform_(-scale, scale)
        value_caches.append(value_cache)

        key_cache, value_cache = key_caches[0], value_caches[0]

        k_scale = (key.amax() / 64.0).to(torch.float32)
        v_scale = (value.amax() / 64.0).to(torch.float32)

        yield (
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            {
                "kv_cache_dtype": "auto",
                "k_scale": k_scale,
                "v_scale": v_scale,
            },
        )

    bench = ReshapeAndCacheBenchmark(
        op_name="reshape_and_cache",
        input_fn=input_kwargs,
        torch_op=torch_reshape_and_cache_ref,
        gems_op=flag_gems.reshape_and_cache,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
