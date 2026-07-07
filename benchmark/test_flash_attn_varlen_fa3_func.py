from typing import Any, List, Optional, Tuple

import pytest
import torch

import flag_gems

from . import base, utils
from .test_flash_attn_varlen_func import make_paged_kv_cache

vendor_name = flag_gems.vendor_name


def _is_hopper() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


def _fa3_call_args(
    q,
    k,
    v,
    max_q,
    cu_q,
    max_k,
    cu_k,
    seqused_k,
    scale,
    causal,
    block_table,
    out,
):
    return (
        q,
        k,
        v,
        max_q,
        cu_q,
        max_k,
        cu_k,
        seqused_k,
        None,
        0.0,
        scale,
        causal,
        [-1, -1],
        0.0,
        None,
        False,
        False,
        block_table,
        False,
        out,
        None,
        None,
        None,
        None,
        {"fa_version": 3},
    )


# name, paged, seq_lens, nh_q, nh_k, head_dim, causal
_HOPPER_CONFIGS = [
    ("prefill_b4_s2k_d128_mha", False, [(2048, 2048)] * 4, 32, 32, 128, True),
    ("prefill_b4_s4k_d128_mha", False, [(4096, 4096)] * 4, 32, 32, 128, True),
    ("prefill_b4_s8k_d128_mha", False, [(8192, 8192)] * 4, 32, 32, 128, True),
    ("prefill_b2_s16k_d128_mha", False, [(16384, 16384)] * 2, 32, 32, 128, True),
    ("prefill_b4_s4k_d128_gqa4", False, [(4096, 4096)] * 4, 32, 8, 128, True),
    ("prefill_b4_s8k_d128_gqa4", False, [(8192, 8192)] * 4, 32, 8, 128, True),
    ("prefill_b8_s2k_d64_mha", False, [(2048, 2048)] * 8, 16, 16, 64, False),
    ("decode_b16_kv1k_d128_gqa4", False, [(1, 1024)] * 16, 32, 8, 128, True),
    ("decode_b8_kv1k_d192_gqa4", False, [(1, 1024)] * 8, 32, 8, 192, True),
    ("decode_b8_kv1k_d256_gqa4", False, [(1, 1024)] * 8, 32, 8, 256, True),
    (
        "decode_b16_mixed_d128_gqa4",
        False,
        [(1, 512), (1, 1024), (1, 2048), (1, 4096)] * 4,
        32,
        8,
        128,
        True,
    ),
    ("decode_b32_kv2k_d128_gqa4", False, [(1, 2048)] * 32, 32, 8, 128, True),
    (
        "varlen_mixed_d128_gqa4",
        False,
        [(2048, 2048), (1, 4096), (1, 4096), (1024, 1024), (1, 8192), (1, 1024)],
        32,
        8,
        128,
        True,
    ),
    (
        "varlen_serve_b32_1pf_31dec_d128_gqa4",
        False,
        [(2048, 2048)] + [(1, 1024 + 64 * i) for i in range(31)],
        32,
        8,
        128,
        True,
    ),
    (
        "varlen_longtail_d128_gqa4",
        False,
        [(16384, 16384)] + [(256, 256)] * 16,
        32,
        8,
        128,
        True,
    ),
    (
        "paged_decode_b16_kvmix_bs16_d128_gqa4",
        True,
        [(1, 1024 + 256 * i) for i in range(16)],
        32,
        8,
        128,
        True,
    ),
    (
        "paged_decode_b8_bs16_d192_gqa4",
        True,
        [(1, 1024 + 128 * i) for i in range(8)],
        32,
        8,
        192,
        True,
    ),
    (
        "paged_decode_b8_bs16_d256_gqa4",
        True,
        [(1, 1024 + 128 * i) for i in range(8)],
        32,
        8,
        256,
        True,
    ),
    (
        "paged_decode_b64_bs16_d128_gqa4",
        True,
        [(1, 512 + 128 * i) for i in range(64)],
        32,
        8,
        128,
        True,
    ),
    (
        "paged_serve_b32_1pf_31dec_bs16_d128_gqa4",
        True,
        [(2048, 2048)] + [(1, 1024 + 96 * i) for i in range(31)],
        32,
        8,
        128,
        True,
    ),
    ("paged_uniform_b4_s4k_bs16_d128_mha", True, [(4096, 4096)] * 4, 32, 32, 128, True),
]


class FlashAttnVarlenFa3Benchmark(base.Benchmark):
    """benchmark for flash_attn_varlen_func (FA3, Hopper only)"""

    def set_shapes(self, shape_file_path: Optional[List[Any]] = None):
        all_cu_seqlens_q = [
            (0, 512),
            (0, 1, 2, 72),
            tuple(range(0, 45))
            + (105, 121, 137, 153, 169, 185, 201, 217, 233, 249, 265),
            tuple(range(0, 196)) + (211, 226, 240, 253, 265),
        ]
        all_seqused_k = [
            (512,),
            (1, 1, 70),
            (515,) + (514,) * 20 + (513,) * 20 + (512,) * 14,
            (2333,)
            + (2331,) * 20
            + (2330,) * 20
            + (2329,) * 14
            + (2328,) * 18
            + (2327,) * 15
            + (2326,) * 17
            + (2325,) * 18
            + (2324,) * 21
            + (2323,) * 22
            + (2322,) * 24
            + (2321,) * 5
            + (2320, 2319, 2318, 2317, 2316),
        ]
        qwen_configs = [
            ("qwen", cu_seqlens_q, seqused_k)
            for cu_seqlens_q, seqused_k in zip(all_cu_seqlens_q, all_seqused_k)
        ]
        self.shapes = qwen_configs + _HOPPER_CONFIGS

    def get_input_iter(self, dtype):
        for config in self.shapes:
            yield self.flash_attn_varlen_fa3_input_fn(config, dtype, self.device)

    def flash_attn_varlen_fa3_input_fn(self, config, dtype, device):
        if config[0] == "qwen":
            _, cu_query_lens, seqused_k = config
            num_query_heads, num_kv_heads = 16, 8
            head_size, block_size, num_blocks = 128, 16, 2000
            causal = True

            num_seqs = len(cu_query_lens) - 1
            max_query_len = max(
                x - y for x, y in zip(cu_query_lens[1:], cu_query_lens[:-1])
            )
            max_kv_len = max(seqused_k)
            scale = head_size**-0.5

            with torch.device(device):
                query = torch.randn(
                    cu_query_lens[-1],
                    num_query_heads,
                    head_size,
                    dtype=dtype,
                    device=device,
                )
                out = torch.empty_like(query)
                key_cache, value_cache = make_paged_kv_cache(
                    num_blocks,
                    block_size,
                    num_kv_heads,
                    head_size,
                    dtype=dtype,
                    device=device,
                    non_contiguous=False,
                )
                cu_query_lens = torch.tensor(
                    cu_query_lens, dtype=torch.int32, device=device
                )
                seqused_k = torch.tensor(seqused_k, dtype=torch.int32, device=device)
                max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
                block_tables = torch.randint(
                    0,
                    num_blocks,
                    (num_seqs, max_num_blocks_per_seq),
                    dtype=torch.int32,
                    device=device,
                )

            return _fa3_call_args(
                query,
                key_cache,
                value_cache,
                max_query_len,
                cu_query_lens,
                max_kv_len,
                None,
                seqused_k,
                scale,
                causal,
                block_tables,
                out,
            )

        _, paged, seq_lens, num_query_heads, num_kv_heads, head_size, causal = config
        cu_q, cu_k = [0], [0]
        for q_len, k_len in seq_lens:
            cu_q.append(cu_q[-1] + q_len)
            cu_k.append(cu_k[-1] + k_len)
        max_query_len = max(q for q, _ in seq_lens)
        max_kv_len = max(k for _, k in seq_lens)
        scale = head_size**-0.5

        with torch.device(device):
            query = torch.randn(
                cu_q[-1], num_query_heads, head_size, dtype=dtype, device=device
            ) * 0.5
            out = torch.empty_like(query)
            cu_query_lens = torch.tensor(cu_q, dtype=torch.int32, device=device)

            if paged:
                block_size = 16
                blocks_per_req = [
                    (k_len + block_size - 1) // block_size for _, k_len in seq_lens
                ]
                max_blocks_per_req = max(blocks_per_req)
                total_virtual_blocks = sum(blocks_per_req)
                num_physical_blocks = max(1, int(total_virtual_blocks * 1.5))
                perm = torch.randperm(num_physical_blocks, device=device)[
                    :total_virtual_blocks
                ]
                block_tables = torch.zeros(
                    (len(seq_lens), max_blocks_per_req),
                    dtype=torch.int32,
                    device=device,
                )
                offset = 0
                for req_idx, n_blocks in enumerate(blocks_per_req):
                    block_tables[req_idx, :n_blocks] = perm[offset : offset + n_blocks]
                    offset += n_blocks
                key_cache = torch.randn(
                    num_physical_blocks,
                    block_size,
                    num_kv_heads,
                    head_size,
                    dtype=dtype,
                    device=device,
                ) * 0.5
                value_cache = torch.randn_like(key_cache)
                seqused_k = torch.tensor(
                    [k for _, k in seq_lens], dtype=torch.int32, device=device
                )
                cu_seqlens_k = None
            else:
                key_cache = torch.randn(
                    cu_k[-1], num_kv_heads, head_size, dtype=dtype, device=device
                ) * 0.5
                value_cache = torch.randn_like(key_cache)
                cu_seqlens_k = torch.tensor(cu_k, dtype=torch.int32, device=device)
                seqused_k = None
                block_tables = None

        return _fa3_call_args(
            query,
            key_cache,
            value_cache,
            max_query_len,
            cu_query_lens,
            max_kv_len,
            cu_seqlens_k,
            seqused_k,
            scale,
            causal,
            block_tables,
            out,
        )


@pytest.mark.skipif(not _is_hopper(), reason="FA3 requires Hopper GPU (sm_90+)")
@pytest.mark.skipif(
    utils.SkipVersion("vllm", "<0.9"),
    reason="vLLM version prior to 0.9 does not include the flash_attn_varlen_func API.",
)
@pytest.mark.skipif(
    utils.SkipVersion("torch", "<2.7"),
    reason="Torch version prior to 2.7 is not compatible with VLLM.",
)
@pytest.mark.skipif(vendor_name == "hygon", reason="Not working")
@pytest.mark.skipif(vendor_name == "cambricon", reason="Not supported")
@pytest.mark.flash_attn_varlen_func
def test_flash_attn_varlen_fa3_func(monkeypatch):
    monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "0")

    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func as _vllm_fa

    def vllm_fa3(*args, **kwargs):
        kwargs.pop("fa_version", None)
        return _vllm_fa(*args, fa_version=3, **kwargs)

    bench = FlashAttnVarlenFa3Benchmark(
        op_name="flash_attn_varlen_fa3_func",
        torch_op=vllm_fa3,
        gems_op=flag_gems.flash_attn_varlen_func,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.run()