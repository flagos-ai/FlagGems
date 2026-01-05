import math
import os
from typing import Any, List, Optional

import pytest
import torch
import triton

import flag_gems

from .performance_utils import Benchmark, GenericBenchmark, SkipVersion, vendor_name


class AttentionBenchmark(GenericBenchmark):
    """
    benchmark for attention
    """

    def set_more_shapes(self):
        # self.shapes is a list of tuples, each containing three elements:
        # (batch, num_heads, seq_len, head_size).
        return None


@pytest.mark.skipif(vendor_name == "kunlunxin", reason="RESULT TODOFIX")
# @pytest.mark.skipif(vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.scaled_dot_product_attention
@pytest.mark.parametrize("dropout_p", [0.0])
@pytest.mark.parametrize("is_causal", [True, False])
def test_perf_scaled_dot_product_attention(dropout_p, is_causal):
    if flag_gems.vendor_name == "hygon":
        os.environ["TRITON_HIP_USE_NEW_STREAM_PIPELINE"] = "0"

    def scaled_dot_product_attention_kwargs(shape, dtype, device):
        query = torch.randn(shape, device=device, dtype=dtype)
        key = torch.randn(shape, device=device, dtype=dtype)
        value = torch.randn(shape, device=device, dtype=dtype)
        yield query, key, value, None, dropout_p, is_causal

    def sdpa_flash(
        query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=is_causal
    ):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )

    bench = AttentionBenchmark(
        op_name="scaled_dot_product_attention",
        input_fn=scaled_dot_product_attention_kwargs,
        # torch_op=torch.nn.functional.scaled_dot_product_attention,
        torch_op=sdpa_flash,
        dtypes=[
            torch.float16,
            torch.bfloat16,
        ],
    )
    bench.set_gems(flag_gems.scaled_dot_product_attention)
    bench.run()
    if flag_gems.vendor_name == "hygon":
        del os.environ["TRITON_HIP_USE_NEW_STREAM_PIPELINE"]


class FlashMLABenchmark(GenericBenchmark):
    """
    benchmark for flash_mla
    """

    def set_more_shapes(self):
        # self.shapes is a list of tuples, each containing three elements:
        # (batch, num_heads, seq_len, head_size).
        return None


@pytest.mark.skipif(vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.skipif(vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.flash_mla
def test_perf_flash_mla():
    def flash_mla_kwargs(shape, dtype, device):
        seqlen = shape[0]
        b = 128
        s_q = 1
        h_q = 128
        h_kv = 1
        d = 576
        dv = 512
        causal = True
        block_size = 64
        cache_seqlens = torch.tensor(
            [seqlen + 2 * i for i in range(b)], dtype=torch.int32, device=device
        )
        max_seqlen = cache_seqlens.max().item()
        max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256

        q = torch.randn([b, s_q, h_q, d], dtype=dtype, device=device)
        block_table = torch.arange(
            b * max_seqlen_pad // block_size, dtype=torch.int32, device=device
        ).view(b, max_seqlen_pad // block_size)
        blocked_k = torch.randn(
            [block_table.numel(), block_size, h_kv, d], dtype=dtype, device=device
        )
        yield q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal

    def scaled_dot_product_attention(query, key, value, h_q, h_kv, is_causal=False):
        query = query.float()
        key = key.float()
        value = value.float()
        key = key.repeat_interleave(h_q // h_kv, dim=0)
        value = value.repeat_interleave(h_q // h_kv, dim=0)
        attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        if is_causal:
            s_q = query.shape[-2]
            s_k = key.shape[-2]
            attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype, device=query.device)
            temp_mask = torch.ones(
                s_q, s_k, dtype=torch.bool, device=query.device
            ).tril(diagonal=s_k - s_q)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)
            attn_weight += attn_bias
        lse = attn_weight.logsumexp(dim=-1)
        attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
        return attn_weight @ value, lse

    def ref_mla(
        q,
        block_table,
        blocked_k,
        max_seqlen_pad,
        block_size,
        b,
        s_q,
        cache_seqlens,
        h_q,
        h_kv,
        d,
        dv,
        causal,
    ):
        device = q.device
        blocked_v = blocked_k[..., :dv]
        out = torch.empty(b, s_q, h_q, dv, dtype=torch.float32, device=device)
        lse = torch.empty(b, h_q, s_q, dtype=torch.float32, device=device)
        for i in range(b):
            begin = i * max_seqlen_pad
            end = begin + cache_seqlens[i]
            O, LSE = scaled_dot_product_attention(
                q[i].transpose(0, 1),
                blocked_k.view(-1, h_kv, d)[begin:end].transpose(0, 1),
                blocked_v.view(-1, h_kv, dv)[begin:end].transpose(0, 1),
                h_q=h_q,
                h_kv=h_kv,
                is_causal=causal,
            )
            out[i] = O.transpose(0, 1)
            lse[i] = LSE
        return out, lse

    if flag_gems.vendor_name == "mthreads":
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    bench = FlashMLABenchmark(
        op_name="flash_mla",
        input_fn=flash_mla_kwargs,
        torch_op=ref_mla,
        dtypes=[
            torch.bfloat16,
        ],
    )
    bench.set_gems(flag_gems.flash_mla)
    bench.run()

    if flag_gems.vendor_name == "mthreads":
        del os.environ["MUSA_ENABLE_SQMMA"]


class FlashAttnVarlenBenchmark(Benchmark):
    """
    benchmark for flash_attn_varlen_func
    """

    def set_shapes(self, shape_file_path: Optional[List[Any]] = None):
        # Collecting from qwen/Qwen3-1.7B --random-input 512 --random-output 2048 --num-prompts 200 --request-rate inf
        # Format: (cu_seq_lens_q, seqused_k, num_heads, head_size, block_size, num_blocks, alibi, soft_cap)

        all_cu_seq_lens_q = [
            (
                0,
                512,
            ),
            (
                0,
                1,
                2,
                72,
            ),
            tuple(range(0, 45))
            + (
                105,
                121,
                137,
                153,
                169,
                185,
                201,
                217,
                233,
                249,
                265,
            ),
            tuple(range(0, 196))
            + (
                211,
                226,
                240,
                253,
                265,
            ),
        ]
        all_seqused_k = [
            (512,),
            (
                1,
                1,
                70,
            ),
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
            + (
                2320,
                2319,
                2318,
                2317,
                2316,
            ),
        ]

        num_heads = 16
        num_heads_k = 8
        head_dim = 128
        block_size = 16
        num_blocks = 2000
        alibi = False
        soft_cap = None

        all_configs = [
            (
                cu_seq_lens_q,
                seqused_k,
                num_heads,
                num_heads_k,
                head_dim,
                block_size,
                num_blocks,
                alibi,
                soft_cap,
            )
            for cu_seq_lens_q, seqused_k in zip(all_cu_seq_lens_q, all_seqused_k)
        ]

        self.shapes = all_configs

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield self.flash_attn_varlen_input_fn(config, cur_dtype, self.device)

    def flash_attn_varlen_input_fn(self, config, dtype, device):
        """Input function for flash attention varlen benchmark"""
        (
            cu_query_lens,
            seqused_k,
            num_query_heads,
            num_kv_heads,
            head_size,
            block_size,
            num_blocks,
            alibi,
            soft_cap,
        ) = config

        if alibi is True and soft_cap is not None:
            return

        num_seqs = len(cu_query_lens) - 1
        max_query_len = max(
            map(lambda x, y: x - y, cu_query_lens[1:], cu_query_lens[:-1])
        )
        max_kv_len = max(seqused_k)
        window_size = (-1, -1)
        scale = head_size**-0.5

        assert num_seqs == len(seqused_k)

        with torch.device(device):
            query = torch.randn(
                cu_query_lens[-1],
                num_query_heads,
                head_size,
                dtype=dtype,
                device=device,
            )
            out = torch.empty_like(query)
            key_cache = torch.randn(
                num_blocks,
                block_size,
                num_kv_heads,
                head_size,
                dtype=dtype,
                device=device,
            )
            value_cache = torch.randn_like(key_cache)
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

            causal = True

            if alibi:
                alibi_slopes = (
                    torch.ones(
                        num_seqs, num_query_heads, device=device, dtype=torch.float32
                    )
                    * 0.3
                )
            else:
                alibi_slopes = None

        return (
            query,
            key_cache,
            value_cache,
            max_query_len,
            cu_query_lens,
            max_kv_len,
            None,
            seqused_k,
            None,
            0.0,
            scale,
            causal,
            window_size,
            soft_cap if soft_cap is not None else 0,
            alibi_slopes,
            False,
            False,
            block_tables,
            False,
            out,
            None,
            None,
            None,
            None,
            {
                "s_aux": None,
                "num_splits": 0,
                "cp_world_size": 1,
                "cp_rank": 0,
                "cp_tot_seqused_k": None,
                "fa_version": 2,
            },
        )


def flash_attn_varlen_legacy(*args):
    """
    Compatibility wrapper for running old flash_attn_varlen_func.
    """
    (
        query,
        key_cache,
        value_cache,
        max_query_len,
        cu_query_lens,
        max_kv_len,
        _,
        seqused_k,
        _,
        dropout_p,
        scale,
        causal,
        window_size,
        soft_cap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        block_tables,
        _,
        out,
        *_,
    ) = args

    k_flat = key_cache.reshape(-1, key_cache.shape[2], key_cache.shape[3])
    v_flat = value_cache.reshape(-1, value_cache.shape[2], value_cache.shape[3])
    cu_seqlens_k = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=seqused_k.device),
            torch.cumsum(seqused_k, dim=0),
        ]
    ).to(torch.int32)

    from flash_attn import flash_attn_varlen_func

    result = flash_attn_varlen_func(
        query,  # q
        k_flat,  # k (flattened from key_cache)
        v_flat,  # v (flattened from value_cache)
        cu_query_lens,  # cu_seqlens_q
        cu_seqlens_k,  # cu_seqlens_k (constructed from seqused_k)
        max_query_len,  # max_seqlen_q
        max_kv_len,  # max_seqlen_k
        dropout_p,  # dropout_p
        scale,  # softmax_scale
        causal,  # causal
        tuple(window_size),  # window_size
        float(soft_cap),  # softcap
        alibi_slopes,  # alibi_slopes
        deterministic,  # deterministic
        return_attn_probs,  # return_attn_probs
        block_tables,  # block_table
        alibi_slopes is not None,  # use_alibi (derived from alibi_slopes)
        0,  # alibi_mode
        1,  # imp_mode
        out=out,  # out
        bias=None,  # bias
    )
    return result


@pytest.mark.skipif(
    SkipVersion("vllm", "<0.9"),
    reason="The version prior to 0.9 does not include the flash_attn_varlen_func API in vllm.",
)
@pytest.mark.skipif(
    SkipVersion("torch", "<2.7"),
    reason="The version prior to 2.7 is not compatible with VLLM.",
)
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.skipif(vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.skipif(vendor_name == "mthreads", reason="Torch < 2.7")
@pytest.mark.skipif(flag_gems.vendor_name == "cambricon", reason="TypeError")
@pytest.mark.flash_attn_varlen_func
def test_perf_flash_attn_varlen_func():
    os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
    if vendor_name == "iluvatar":
        # iluvatar does not have updated vllm_flash_attn, use conversion wrapper
        flash_attn_varlen_func = flash_attn_varlen_legacy
    else:
        from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

    bench = FlashAttnVarlenBenchmark(
        op_name="flash_attn_varlen_func",
        torch_op=flash_attn_varlen_func,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.set_gems(flag_gems.ops.flash_attn_varlen_func)
    bench.run()


class GetSchedulerMetadataBenchmark(GenericBenchmark):
    """
    benchmark for get_scheduler_metadata
    """

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (8, 8, 1024, 16, 4, 128, 128),
            (32, 32, 512, 8, 8, 64, 64),
            (256, 256, 2048, 32, 32, 128, 128),
            (512, 512, 4096, 32, 8, 128, 128),
            (1024, 1024, 8192, 64, 16, 128, 128),
        ]

    def set_more_shapes(self):
        return None


@pytest.mark.get_scheduler_metadata
def test_perf_get_scheduler_metadata():
    try:
        os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
        from vllm.vllm_flash_attn.flash_attn_interface import (
            get_scheduler_metadata as vllm_get_scheduler_metadata,
        )
    except ImportError:
        pytest.skip("vllm is not available, skipping performance test")

    def input_kwargs(shape, dtype, device):
        (
            batch_size,
            max_seqlen_q,
            max_seqlen_k,
            num_heads_q,
            num_heads_kv,
            headdim,
            headdim_v,
        ) = shape
        cache_seqlens = torch.randint(
            1, max_seqlen_k + 1, (batch_size,), dtype=torch.int32, device=device
        )

        yield (
            batch_size,
            max_seqlen_q,
            max_seqlen_k,
            num_heads_q,
            num_heads_kv,
            headdim,
            cache_seqlens,
            dtype,  # qkv_dtype
            headdim_v,  # headdim_v
            None,  # cu_seqlens_q
            None,  # cu_seqlens_k_new
            None,  # cache_leftpad
            None,  # page_size
            0,  # max_seqlen_k_new
            False,  # causal
            (-1, -1),  # window_size
            False,  # has_softcap
            0,  # num_splits
            None,  # pack_gqa
            0,  # sm_margin
        )

    def flaggems_wrapper(
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads_q,
        num_heads_kv,
        headdim,
        cache_seqlens,
        qkv_dtype=torch.bfloat16,
        headdim_v=None,
        cu_seqlens_q=None,
        cu_seqlens_k_new=None,
        cache_leftpad=None,
        page_size=None,
        max_seqlen_k_new=0,
        causal=False,
        window_size=(-1, -1),
        has_softcap=False,
        num_splits=0,
        pack_gqa=None,
        sm_margin=0,
    ):
        return flag_gems.ops.get_scheduler_metadata(
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            num_heads=num_heads_q,
            num_heads_k=num_heads_kv,
            headdim=headdim,
            headdim_v=headdim_v or headdim,
            qkv_dtype=qkv_dtype,
            seqused_k=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=None,
            cu_seqlens_k_new=cu_seqlens_k_new,
            seqused_q=None,
            leftpad_k=cache_leftpad,
            page_size=page_size,
            max_seqlen_k_new=max_seqlen_k_new,
            is_causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            has_softcap=has_softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            sm_margin=sm_margin,
        )

    bench = GetSchedulerMetadataBenchmark(
        op_name="get_scheduler_metadata",
        input_fn=input_kwargs,
        torch_op=vllm_get_scheduler_metadata,
        dtypes=[
            torch.float16,
            torch.bfloat16,
        ],
    )
    bench.set_gems(flaggems_wrapper)
    bench.run()
