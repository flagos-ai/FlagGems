from typing import Any, List, Optional

import pytest
import torch

import flag_gems

from . import base, utils

vendor_name = flag_gems.vendor_name


class FlashAttnVarlenOptInitBenchmark(base.Benchmark):
    """
    benchmark for flash_attn_varlen_lse_func
    """

    def set_shapes(self, shape_file_path: Optional[List[Any]] = None):
        # Collecting from qwen/Qwen3-1.7B --random-input 512 --random-output 2048 --num-prompts 200 --request-rate inf
        # Format: (cu_seq_lens_q, seqused_k, num_heads, head_size, block_size, num_blocks, alibi, soft_cap)

        all_cu_seq_lens_q = [
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
            (
                0,
                1,
                2,
                72,
            ),
            (
                0,
                512,
            ),
        ]
        all_seqused_k = [
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
            (
                1,
                1,
                70,
            ),
            (512,),
        ]

        num_heads = 16
        num_heads_k = 8
        head_dim = 128
        block_size = 16
        num_blocks = 2000
        alibi = False
        soft_cap = None

        # cu_seq_lens_q = all_cu_seq_lens_q[1]
        # seqused_k = all_seqused_k[1]
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

    def get_input_iter(self, dtype):
        for config in self.shapes:
            yield self.flash_attn_varlen_input_fn(config, dtype, self.device)

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
            lse = torch.empty(
                (num_query_heads, cu_query_lens[-1]), dtype=torch.float, device=device
            )
            # lse = None
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
            lse,
            None,
            None,
            None,
            None,
            None,
            0,
            1,
            0,
            None,
            2,
        )


def flash_attn_varlen_func_ref(*args, **kwargs):
    (
        q,
        k,
        v,
        max_seqlen_q,
        cu_seqlens_q,
        max_seqlen_k,
        cu_seqlens_k,  # only used for non-paged prefill
        seqused_k,
        q_v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,  # 0.0 means deactivated
        alibi_slopes,
        deterministic,
        return_attn_probs,
        block_table,
        return_softmax_lse,
        out,
        lse,
        # Dummy FA3 arguments
        scheduler_metadata,
        q_descale,
        k_descale,
        v_descale,
        s_aux,
        num_splits,
        cp_world_size,
        cp_rank,
        cp_tot_seqused_k,
        fa_version,
    ) = args

    # TODO(Qiming): don't import things in the middle
    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

    result = flash_attn_varlen_func(
        q,
        k,
        v,
        max_seqlen_q,
        cu_seqlens_q,
        max_seqlen_k,
        cu_seqlens_k,  # only used for non-paged prefill
        seqused_k,
        q_v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,  # 0.0 means deactivated
        alibi_slopes,
        deterministic,
        return_attn_probs,
        block_table,
        return_softmax_lse,
        out,
        # Dummy FA3 arguments
        scheduler_metadata,
        q_descale,
        k_descale,
        v_descale,
        fa_version,
    )
    return result


@pytest.mark.flash_attn_varlen_opt_func
@pytest.mark.skipif(
    utils.SkipVersion("vllm", "<0.9"),
    reason="vLLM version prior to 0.9 does not include the flash_attn_varlen_func API.",
)
@pytest.mark.skipif(
    utils.SkipVersion("torch", "<2.7"),
    reason="Torch version prior to 2.7 is not compatible with VLLM.",
)
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.skipif(vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.skipif(vendor_name == "mthreads", reason="Torch < 2.7")
@pytest.mark.skipif(flag_gems.vendor_name == "cambricon", reason="TypeError")
def test_flash_attn_varlen_opt_func(monkeypatch):
    monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "0")

    bench = FlashAttnVarlenOptInitBenchmark(
        op_name="flash_attn_varlen_func",
        torch_op=flash_attn_varlen_func_ref,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.set_gems(flag_gems.ops.flash_attn_varlen_opt_func)
    bench.run()
