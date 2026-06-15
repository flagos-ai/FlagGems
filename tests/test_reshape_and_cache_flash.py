import random

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

device = flag_gems.device


def create_kv_caches_with_random_flash(
    num_blocks,
    block_size,
    num_layers,
    num_heads,
    head_size,
    cache_dtype,
    model_dtype,
    seed,
    device,
):
    utils.init_seed(seed)
    torch_dtype = model_dtype
    key_value_cache_shape = (num_blocks, 2, block_size, num_heads, head_size)
    scale = head_size**-0.5

    key_caches: list[torch.Tensor] = []
    value_caches: list[torch.Tensor] = []

    for _ in range(num_layers):
        key_value_cache = torch.empty(
            size=key_value_cache_shape, dtype=torch_dtype, device=device
        )

        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            key_value_cache.uniform_(-scale, scale)
        else:
            raise ValueError(f"Key cache type {cache_dtype} is not supported")

        key_caches.append(key_value_cache[:, 0])
        value_caches.append(key_value_cache[:, 1])

    return key_caches, value_caches


@pytest.mark.reshape_and_cache_flash
@pytest.mark.parametrize("num_tokens", [42])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("head_size", [64, 80, 120, 256])
@pytest.mark.parametrize("block_size", [8, 16, 32])
@pytest.mark.parametrize("num_blocks", [1024, 10000])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16, torch.float])
@pytest.mark.parametrize("kv_cache_dtype", ["auto"])
@pytest.mark.parametrize("seed", [2025])
@torch.inference_mode()
def test_reshape_and_cache_flash(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    seed: int,
) -> None:
    utils.init_seed(seed)

    with torch.device(device):
        # Create a random slot mapping.
        num_slots = block_size * num_blocks
        slot_mapping_lst = random.sample(range(num_slots), num_tokens)
        slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long, device=device)
        qkv = torch.randn(
            num_tokens, 3, num_heads, head_size, dtype=dtype, device=device
        )
        _, key, value = qkv.unbind(dim=1)

        # Create the KV caches.
        key_caches, value_caches = create_kv_caches_with_random_flash(
            num_blocks,
            block_size,
            1,
            num_heads,
            head_size,
            kv_cache_dtype,
            dtype,
            seed=seed,
            device=device,
        )
        key_cache, value_cache = (
            key_caches[0].contiguous(),
            value_caches[0].contiguous(),
        )
        del key_caches
        del value_caches

        k_scale = (key.amax() / 64.0).to(torch.float32)
        v_scale = (value.amax() / 64.0).to(torch.float32)

        # Clone the KV caches.
        cloned_key_cache = key_cache.clone()
        cloned_value_cache = value_cache.clone()

        # Call the reshape_and_cache kernel.
        flag_gems.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )

        # Run the reference implementation.
        block_indicies = torch.div(slot_mapping, block_size, rounding_mode="floor")
        block_indicies_lst = block_indicies.cpu().tolist()
        block_offsets = slot_mapping % block_size
        block_offsets_lst = block_offsets.cpu().tolist()

        for i in range(num_tokens):
            block_idx = block_indicies_lst[i]
            block_offset = block_offsets_lst[i]
            cloned_key_cache[block_idx, block_offset, :, :] = key[i]
            cloned_value_cache[block_idx, block_offset, :, :] = value[i]

        torch.testing.assert_close(key_cache.cpu(), cloned_key_cache.cpu())
        torch.testing.assert_close(value_cache.cpu(), cloned_value_cache.cpu())
