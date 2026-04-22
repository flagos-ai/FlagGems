import random

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

device = flag_gems.device
vendor_name = flag_gems.vendor_name

CUDA_DEVICES = [f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)]


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


@pytest.mark.skipif(vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.concat_and_cache_mla
@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("qk_rope_head_dim", [64])
@pytest.mark.parametrize("num_tokens", [42])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("num_blocks", [8])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16, torch.float])
@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize(
    "device",
    [flag_gems.device] if vendor_name == "mthreads" else CUDA_DEVICES,
)
@pytest.mark.parametrize("kv_cache_dtype", ["auto"])
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
        ref_temp = utils.to_reference(
            torch.zeros(*kv_cache.shape, dtype=dtype, device=device)
        )

        for i in range(num_tokens):
            slot = slot_mapping[i].item()
            block_idx = slot // block_size
            block_offset = slot % block_size
            ref_temp[block_idx, block_offset, :kv_lora_rank] = kv_c[i]
            ref_temp[block_idx, block_offset, kv_lora_rank:] = k_pe[i]

        if kv_cache_dtype == "fp8":
            ref_kv_cache = utils.to_reference(
                torch.empty_like(ref_temp, dtype=kv_cache.dtype)
            )
            convert_fp8(ref_kv_cache, ref_temp, scale.item(), kv_dtype=kv_cache_dtype)
        else:
            ref_kv_cache = utils.to_reference(ref_temp)
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
            expected_temp = utils.to_reference(
                torch.empty_like(ref_kv_cache, dtype=torch.uint8)
            )
            convert_fp8(
                expected_temp, ref_kv_cache, scale.item(), kv_dtype=kv_cache_dtype
            )
            dtype = torch.float8_e4m3fn
            # TODO: RuntimeError: Comparing
            # maybe a bug in torch.testing.assert_close
            # utils.gems_assert_close(kv_cache.view(dtype), ref_kv_cache.view(dtype), dtype)
            torch.testing.assert_close(result_temp, expected_temp, atol=0.001, rtol=0.1)
        else:
            utils.gems_assert_close(kv_cache, ref_kv_cache, kv_cache.dtype)
