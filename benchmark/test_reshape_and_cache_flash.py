import random
from typing import Any

import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES

from .performance_utils import GenericBenchmark


def torch_reshape_and_cache_flash_ref(
    key: Any,
    value: Any,
    key_cache: Any,
    value_cache: Any,
    slot_mapping: Any,
    kv_cache_dtype: Any = "auto",
    k_scale: Any = None,
    v_scale: Any = None,
):
    block_size = key_cache.size(1)
    num_tokens = slot_mapping.numel()
    for i in range(num_tokens):
        slot = slot_mapping[i].item()
        block_idx = slot // block_size
        block_offset = slot % block_size
        key_cache[block_idx, block_offset] = key[i]
        value_cache[block_idx, block_offset] = value[i]


class ReshapeAndCacheFlashBenchmark(GenericBenchmark):
    """
    benchmark for reshape_and_cache_flash
    """

    def set_more_shapes(self):
        return None


@pytest.mark.reshape_and_cache_flash
def test_reshape_and_cache_flash():
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

        key_value_cache_shape = (num_blocks, 2, block_size, num_heads, head_size)
        scale = head_size**-0.5
        key_caches: list[torch.Tensor] = []
        value_caches: list[torch.Tensor] = []
        key_value_cache = torch.empty(
            size=key_value_cache_shape, dtype=dtype, device=device
        )
        key_value_cache.uniform_(-scale, scale)
        key_caches.append(key_value_cache[:, 0])
        value_caches.append(key_value_cache[:, 1])
        key_cache, value_cache = (
            key_caches[0].contiguous(),
            value_caches[0].contiguous(),
        )
        del key_caches
        del value_caches

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

    bench = ReshapeAndCacheFlashBenchmark(
        op_name="reshape_and_cache_flash",
        input_fn=input_kwargs,
        torch_op=torch_reshape_and_cache_flash_ref,
        gems_op=flag_gems.reshape_and_cache_flash,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
