# SPDX-License-Identifier: Apache-2.0
import random

import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_close, to_reference

device = flag_gems.device

COPYING_DIRECTION = [("cuda", "cpu"), ("cuda", "cuda"), ("cpu", "cuda")]
DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [42]  # Arbitrary values for testing
NUM_LAYERS = [1]  # Arbitrary values for testing
NUM_HEADS = [8]  # Arbitrary values for testing
HEAD_SIZES = [64, 80, 120, 256]
BLOCK_SIZES = [8, 16, 32]
CACHE_LAYOUTS = ["NHD", "HND"]

# Parameters for MLA tests.
KV_LORA_RANKS = [512]
QK_ROPE_HEAD_DIMS = [64]
NUM_TOKENS_MLA = [42]
BLOCK_SIZES_MLA = [16]
NUM_BLOCKS_MLA = [8]

# Arbitrary values for testing
# don't make it too large. e.g. [1024, 36000] will OOM
NUM_BLOCKS = [1024, 10000]

NUM_MAPPINGS = [256]  # Arbitrary values for testing
SEEDS = [0]
CUDA_DEVICES = [f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)]

# We assume fp8 is always enabled for testing.
if flag_gems.vendor_name in ["kunlunxin", "cambricon"]:
    KV_CACHE_DTYPE = ["auto"]
else:
    KV_CACHE_DTYPE = ["auto", "fp8"]


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


@pytest.mark.skipif(flag_gems.vendor_name == "metax", reason="RuntimeError")
@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.concat_and_cache_mla
@pytest.mark.parametrize("kv_lora_rank", KV_LORA_RANKS)
@pytest.mark.parametrize("qk_rope_head_dim", QK_ROPE_HEAD_DIMS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS_MLA)
@pytest.mark.parametrize("block_size", BLOCK_SIZES_MLA)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS_MLA)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    "device",
    [flag_gems.device] if flag_gems.vendor_name == "mthreads" else CUDA_DEVICES,
)
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

    with torch.device(device):
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
        ref_temp = to_reference(
            torch.zeros(*kv_cache.shape, dtype=dtype, device=device)
        )

        for i in range(num_tokens):
            slot = slot_mapping[i].item()
            block_idx = slot // block_size
            block_offset = slot % block_size
            ref_temp[block_idx, block_offset, :kv_lora_rank] = kv_c[i]
            ref_temp[block_idx, block_offset, kv_lora_rank:] = k_pe[i]

        if kv_cache_dtype == "fp8":
            ref_kv_cache = to_reference(
                torch.empty_like(ref_temp, dtype=kv_cache.dtype)
            )
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
                result_temp,
                kv_cache.contiguous(),
                scale.item(),
                kv_dtype=kv_cache_dtype,
            )
            expected_temp = to_reference(
                torch.empty_like(ref_kv_cache, dtype=torch.uint8)
            )
            convert_fp8(
                expected_temp, ref_kv_cache, scale.item(), kv_dtype=kv_cache_dtype
            )
            dtype = torch.float8_e4m3fn
            if flag_gems.vendor_name == "mthreads":
                result_temp = to_reference(result_temp)
            # TODO: RuntimeError: Comparing
            # maybe a bug in torch.testing.assert_close
            # gems_assert_close(kv_cache.view(dtype), ref_kv_cache.view(dtype), dtype)
            torch.testing.assert_close(result_temp, expected_temp, atol=0.001, rtol=0.1)
        else:
            if flag_gems.vendor_name == "mthreads":
                kv_cache = to_reference(kv_cache)
            gems_assert_close(kv_cache, ref_kv_cache, kv_cache.dtype)
