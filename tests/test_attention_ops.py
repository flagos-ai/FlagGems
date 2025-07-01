import random

import numpy as np
import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_close, init_seed, to_reference

device = flag_gems.device


@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.skipif(flag_gems.device == "musa", reason="RuntimeError")
@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.scaled_dot_product_attention
@pytest.mark.parametrize("batch", [8, 16])
@pytest.mark.parametrize("num_head", [1, 8])
@pytest.mark.parametrize("q_seq_len", [17, 64, 128])
@pytest.mark.parametrize("kv_seq_len", [7, 87, 128, 577, 2048])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("add_bias", [True, False])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scaled_dot_product_attention(
    batch, num_head, q_seq_len, kv_seq_len, head_size, add_bias, is_causal, dtype
):
    np.random.seed(0)
    np_query = np.random.uniform(
        -0.05, 0.05, (batch, num_head, q_seq_len, head_size)
    ).astype(np.float32)
    np_key = np.random.uniform(
        -0.05, 0.05, (batch, num_head, kv_seq_len, head_size)
    ).astype(np.float32)
    np_value = np.random.uniform(
        -0.05, 0.05, (batch, num_head, kv_seq_len, head_size)
    ).astype(np.float32)
    np_attn_bias = np.random.uniform(
        -0.05, 0.05, (batch, num_head, q_seq_len, kv_seq_len)
    ).astype(np.float32)

    query = torch.tensor(np_query, device=device, dtype=dtype)
    key = torch.tensor(np_key, device=device, dtype=dtype)
    value = torch.tensor(np_value, device=device, dtype=dtype)
    if add_bias:
        attn_bias = torch.tensor(np_attn_bias, device=device, dtype=dtype)
    else:
        attn_bias = None

    ref_query = to_reference(query, False)
    ref_key = to_reference(key, False)
    ref_value = to_reference(value, False)
    ref_attn_bias = to_reference(attn_bias, False) if add_bias else None

    scale = float(1.0 / np.sqrt(head_size))

    if is_causal:
        torch_result = torch.nn.functional.scaled_dot_product_attention(
            ref_query,
            ref_key,
            ref_value,
            scale=scale,
            is_causal=is_causal,
        )
    else:
        torch_result = torch.nn.functional.scaled_dot_product_attention(
            ref_query,
            ref_key,
            ref_value,
            attn_mask=ref_attn_bias,
            scale=scale,
            is_causal=is_causal,
        )

    with flag_gems.use_gems():
        if is_causal:
            flaggem_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, scale=scale, is_causal=is_causal
            )
        else:
            flaggem_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_bias, scale=scale, is_causal=is_causal
            )
    gems_assert_close(flaggem_result, torch_result, dtype, reduce_dim=head_size)


DTYPES = [torch.half, torch.bfloat16, torch.float]
# Parameters for MLA tests.
KV_LORA_RANKS = [512]
QK_ROPE_HEAD_DIMS = [64]
NUM_TOKENS_MLA = [42]
BLOCK_SIZES_MLA = [16]
NUM_BLOCKS_MLA = [8]
SEEDS = [0]
CUDA_DEVICES = [f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)]
# We assume fp8 is always enabled for testing.
KV_CACHE_DTYPE = [
    "auto",
]


def _create_mla_cache(
    num_blocks: int,
    block_size: int,
    entry_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
) -> torch.Tensor:
    cache_dtype = torch.uint8 if kv_cache_dtype == "fp8" else dtype
    return torch.zeros(
        num_blocks, block_size, entry_size, dtype=cache_dtype, device=device
    )


# Custom implementation for FP8 conversion (only for testing)
def convert_fp8(
    dst: torch.Tensor, src: torch.Tensor, scale: float, kv_dtype: str
) -> None:
    if kv_dtype == "fp8":
        dst_ = (src / scale).to(torch.float8_e4m3fn).view(dst.dtype)
        dst.copy_(dst_)
    else:
        dst.copy_(src)


@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.skipif(flag_gems.device == "musa", reason="RuntimeError")
@pytest.mark.concat_and_cache_mla
@pytest.mark.parametrize("kv_lora_rank", KV_LORA_RANKS)
@pytest.mark.parametrize("qk_rope_head_dim", QK_ROPE_HEAD_DIMS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS_MLA)
@pytest.mark.parametrize("block_size", BLOCK_SIZES_MLA)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS_MLA)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
@torch.inference_mode()
def test_concat_and_cache_mla(
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    num_tokens: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
    kv_cache_dtype: str,
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_default_device(device)

    total_slots = num_blocks * block_size
    slot_mapping_lst = random.sample(range(total_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long, device=device)

    kv_c = torch.randn(num_tokens, kv_lora_rank, dtype=dtype, device=device)
    k_pe = torch.randn(num_tokens, qk_rope_head_dim, dtype=dtype, device=device)
    entry_size = kv_lora_rank + qk_rope_head_dim

    scale = torch.tensor(0.1, dtype=torch.float32, device=device)
    kv_cache = _create_mla_cache(
        num_blocks, block_size, entry_size, dtype, kv_cache_dtype, device
    )
    ref_temp = to_reference(torch.zeros(*kv_cache.shape, dtype=dtype, device=device))

    for i in range(num_tokens):
        slot = slot_mapping[i].item()
        block_idx = slot // block_size
        block_offset = slot % block_size
        ref_temp[block_idx, block_offset, :kv_lora_rank] = kv_c[i]
        ref_temp[block_idx, block_offset, kv_lora_rank:] = k_pe[i]

    if kv_cache_dtype == "fp8":
        ref_kv_cache = to_reference(torch.empty_like(ref_temp, dtype=kv_cache.dtype))
        convert_fp8(ref_kv_cache, ref_temp, scale.item(), kv_dtype=kv_cache_dtype)
    else:
        ref_kv_cache = to_reference(ref_temp)
    with flag_gems.use_gems():
        flag_gems.concat_and_cache_mla(
            kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale
        )

    if kv_cache_dtype == "fp8":
        result_temp = torch.empty_like(kv_cache, dtype=torch.uint8)
        convert_fp8(
            result_temp, kv_cache.contiguous(), scale.item(), kv_dtype=kv_cache_dtype
        )
        expected_temp = to_reference(torch.empty_like(ref_kv_cache, dtype=torch.uint8))
        convert_fp8(expected_temp, ref_kv_cache, scale.item(), kv_dtype=kv_cache_dtype)
        dtype = torch.float8_e4m3fn
        # TODO: RuntimeError: Comparing
        # maybe a bug in torch.testing.assert_close
        # gems_assert_close(kv_cache.view(dtype), ref_kv_cache.view(dtype), dtype)
        torch.testing.assert_close(result_temp, expected_temp, atol=0.001, rtol=0.1)
    else:
        gems_assert_close(kv_cache, ref_kv_cache, kv_cache.dtype)


def create_kv_caches_with_random(
    num_blocks,
    block_size,
    num_layers,
    num_heads,
    head_size,
    cache_dtype,
    model_dtype=None,
    seed=None,
):
    init_seed(seed)
    torch_dtype = model_dtype
    scale = head_size**-0.5
    x = 16 // torch.tensor([], dtype=torch_dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_caches: list[torch.Tensor] = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape, dtype=torch_dtype, device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            key_cache.uniform_(-scale, scale)
        else:
            raise ValueError(f"Does not support key cache of type {cache_dtype}")
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_caches: list[torch.Tensor] = []
    for _ in range(num_layers):
        value_cache = torch.empty(
            size=value_cache_shape, dtype=torch_dtype, device=device
        )
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            value_cache.uniform_(-scale, scale)
        else:
            raise ValueError(f"Does not support value cache of type {cache_dtype}")
        value_caches.append(value_cache)
    return key_caches, value_caches


@pytest.mark.test_reshape_and_cache
@pytest.mark.parametrize("num_tokens", [42])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("head_size", [64, 80, 120, 256])
@pytest.mark.parametrize("block_size", [8, 16, 32])
@pytest.mark.parametrize("num_blocks", [1024, 10000])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16, torch.float])
@pytest.mark.parametrize("kv_cache_dtype", ["auto"])
@pytest.mark.parametrize("seed", [2025])
def test_reshape_and_cache(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    seed: int,
) -> None:
    init_seed(seed)
    torch.set_default_device(device)
    # Create a random slot mapping.
    num_slots = block_size * num_blocks
    slot_mapping_lst = random.sample(range(num_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long)

    qkv = torch.randn(num_tokens, 3, num_heads, head_size, dtype=dtype)
    _, key, value = qkv.unbind(dim=1)

    # Create the KV caches.
    key_caches, value_caches = create_kv_caches_with_random(
        num_blocks, block_size, 1, num_heads, head_size, kv_cache_dtype, dtype, seed
    )
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Using default kv_scale
    k_scale = (key.amax() / 64.0).to(torch.float32)
    v_scale = (value.amax() / 64.0).to(torch.float32)

    # Clone the KV caches.
    cloned_key_cache = key_cache.clone()
    cloned_value_cache = value_cache.clone()

    # Call the reshape_and_cache kernel.
    flag_gems.reshape_and_cache(
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
    reshaped_key = key.reshape(num_tokens, *key_cache[0, :, :, 0, :].shape)
    block_indicies = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_indicies_lst = block_indicies.cpu().tolist()
    block_offsets = slot_mapping % block_size
    block_offsets_lst = block_offsets.cpu().tolist()
    for i in range(num_tokens):
        block_idx = block_indicies_lst[i]
        block_offset = block_offsets_lst[i]
        cloned_key_cache[block_idx, :, :, block_offset, :] = reshaped_key[i]
        cloned_value_cache[block_idx, :, :, block_offset] = value[i]

    torch.testing.assert_close(key_cache, cloned_key_cache)
    torch.testing.assert_close(value_cache, cloned_value_cache)
