import random

import pytest
import torch

import flag_gems

from . import base, consts


class ConcatAndCacheMLABenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return None


def torch_concat_and_cache_mla_ref(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str = "auto",
    scale: torch.Tensor | None = None,
) -> None:
    kv_lora_rank = kv_c.size(1)
    block_size = kv_cache.size(1)
    temp_cache = torch.zeros(kv_cache.shape, dtype=kv_c.dtype, device=kv_cache.device)

    for token_idx in range(slot_mapping.numel()):
        slot = slot_mapping[token_idx].item()
        block_id = slot // block_size
        block_offset = slot % block_size
        temp_cache[block_id, block_offset, :kv_lora_rank] = kv_c[token_idx]
        temp_cache[block_id, block_offset, kv_lora_rank:] = k_pe[token_idx]

    if kv_cache_dtype != "auto":
        scale_val = scale.item() if scale is not None else 1.0
        kv_cache.copy_(
            (temp_cache / scale_val).to(torch.float8_e4m3fn).view(torch.uint8)
        )
    else:
        kv_cache.copy_(temp_cache)


@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="#2885: RuntimeError")
@pytest.mark.concat_and_cache_mla
def test_concat_and_cache_mla():
    def input_kwargs(shape, dtype, device):
        (
            kv_lora_rank,
            qk_rope_head_dim,
            num_tokens,
            block_size,
            num_blocks,
        ) = shape
        total_slots = num_blocks * block_size
        slot_mapping_lst = random.sample(range(total_slots), num_tokens)
        slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long, device=device)

        kv_c = torch.randn(num_tokens, kv_lora_rank, dtype=dtype, device=device)
        k_pe = torch.randn(num_tokens, qk_rope_head_dim, dtype=dtype, device=device)
        entry_size = kv_lora_rank + qk_rope_head_dim

        scale = torch.tensor(0.1, dtype=torch.float32, device=device)

        kv_cache = torch.zeros(
            num_blocks,
            block_size,
            entry_size,
            dtype=dtype,
            device=device,
        )

        yield (
            kv_c,
            k_pe,
            kv_cache,
            slot_mapping,
            {"kv_cache_dtype": "auto", "scale": scale},
        )

    bench = ConcatAndCacheMLABenchmark(
        op_name="concat_and_cache_mla",
        input_fn=input_kwargs,
        torch_op=torch_concat_and_cache_mla_ref,
        gems_ops=flag_gems.concat_and_cache_mla,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
